#!/usr/bin/env python3
"""OpenCV oracle harness — per-stage reference dumps for the Rust port.

Replicates cv2.Stitcher PANORAMA mode stage by stage (see docs/pipeline.md)
and dumps every intermediate, so panoloom-core can assert parity per stage:

  dumps/<set>/
    meta.json                 scales, parameters, versions, image list
    features/img_XXX.json     ORB keypoints (work scale)
    features/img_XXX.desc.npy 500x32 uint8 descriptors
    matches/pair_I_J.json     matches, inlier mask, H (centered coords), confidence
    cameras_initial.json      after HomographyBasedEstimator
    cameras_ba.json           after BundleAdjusterRay
    cameras_final.json        after waveCorrect (horizontal)
    gains/gain_XXX.npy        BlocksGainCompensator per-image gain maps (seam scale)
    seams/mask_XXX.png        graph-cut seam masks (seam scale)
    compose.json              corners/sizes at compose scale, blend params
    result.jpg                final blended panorama

Determinism: ORB, findHomography (fixed-seed RANSAC), BundleAdjusterRay and
the blender are deterministic. OpenCV's own BestOf2NearestMatcher is NOT for
binary descriptors (FLANN-LSH), so raw matching is re-implemented here with
exact brute-force Hamming, byte-for-byte following matchers.cpp semantics —
this is also exactly what the Rust port implements.

Usage:
  oracle.py --images ../testdata/generated/ring_kloppenheim_06 [--out dumps/ring]
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Stitcher PANORAMA-mode constants (docs/pipeline.md §0-§2).
REGISTR_RESOL_MP = 0.6
SEAM_EST_RESOL_MP = 0.1
CONF_THRESH = 1.0
MATCH_CONF = 0.3
NUM_MATCHES_THRESH1 = 6
NUM_MATCHES_THRESH2 = 6
MATCHES_CONFIDENCE_THRESH = 3.0
BLEND_STRENGTH = 5  # stitcher.cpp: blend width = sqrt(area) * strength / 100


def keypoints_json(kps: list) -> list[dict]:
    return [
        {
            "x": kp.pt[0], "y": kp.pt[1], "size": kp.size, "angle": kp.angle,
            "response": kp.response, "octave": kp.octave,
        }
        for kp in kps
    ]


def camera_json(cam) -> dict:
    return {
        "focal": cam.focal, "aspect": cam.aspect,
        "ppx": cam.ppx, "ppy": cam.ppy,
        "R": np.asarray(cam.R, dtype=np.float64).tolist(),
    }


def best_of_2_nearest(desc_a, desc_b, kps_a, kps_b, size_a, size_b):
    """BestOf2NearestMatcher::match with exact BF-Hamming (matchers.cpp:149-475)."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    pair_set = set()
    matches = []
    # 1->2 with ratio test: d0 < (1 - match_conf) * d1
    for knn in bf.knnMatch(desc_a, desc_b, k=2):
        if len(knn) == 2 and knn[0].distance < (1 - MATCH_CONF) * knn[1].distance:
            matches.append(cv2.DMatch(knn[0].queryIdx, knn[0].trainIdx, knn[0].distance))
            pair_set.add((knn[0].queryIdx, knn[0].trainIdx))
    # 2->1, swapped, skipping already-found pairs
    for knn in bf.knnMatch(desc_b, desc_a, k=2):
        if len(knn) == 2 and knn[0].distance < (1 - MATCH_CONF) * knn[1].distance:
            if (knn[0].trainIdx, knn[0].queryIdx) not in pair_set:
                matches.append(cv2.DMatch(knn[0].trainIdx, knn[0].queryIdx, knn[0].distance))

    info = {"matches": matches, "H": None, "inliers_mask": np.zeros(len(matches), np.uint8),
            "num_inliers": 0, "confidence": 0.0}
    if len(matches) < NUM_MATCHES_THRESH1:
        return info

    # Centered coordinates (matchers.cpp:415-423).
    src = np.array(
        [(kps_a[m.queryIdx].pt[0] - size_a[0] * 0.5, kps_a[m.queryIdx].pt[1] - size_a[1] * 0.5)
         for m in matches], np.float32)
    dst = np.array(
        [(kps_b[m.trainIdx].pt[0] - size_b[0] * 0.5, kps_b[m.trainIdx].pt[1] - size_b[1] * 0.5)
         for m in matches], np.float32)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC)
    if H is None or abs(np.linalg.det(H)) < np.finfo(np.float64).eps:
        return info
    mask = mask.ravel().astype(np.uint8)
    num_inliers = int(mask.sum())
    confidence = num_inliers / (8 + 0.3 * len(matches))
    # Near-duplicate rejection: too-good confidence is zeroed (matchers.cpp:441-443).
    if confidence > MATCHES_CONFIDENCE_THRESH:
        confidence = 0.0
    info.update(H=H, inliers_mask=mask, num_inliers=num_inliers, confidence=confidence)

    if num_inliers >= NUM_MATCHES_THRESH2:
        H2, _ = cv2.findHomography(src[mask.astype(bool)], dst[mask.astype(bool)], cv2.RANSAC)
        if H2 is not None:
            info["H"] = H2
    return info


def match_all(features) -> list:
    """Dense N x N MatchesInfo grid like FeaturesMatcher::operator() (matchers.cpp:338-363).

    Python-constructed cv2.detail.MatchesInfo objects segfault OpenCV when
    passed back into C++ (observed with opencv-python 4.14), so we obtain a
    properly-initialized grid from the real matcher, then overwrite EVERY
    pair with the deterministic BF-Hamming results — no LSH randomness
    survives in the dumps.
    """
    cv2.setRNGSeed(12345)
    grid = list(cv2.detail_BestOf2NearestMatcher(False).apply2(features))
    n = len(features)
    for i in range(n):
        for j in range(i + 1, n):
            fa, fb = features[i], features[j]
            info = best_of_2_nearest(
                fa.descriptors.get() if isinstance(fa.descriptors, cv2.UMat) else fa.descriptors,
                fb.descriptors.get() if isinstance(fb.descriptors, cv2.UMat) else fb.descriptors,
                fa.keypoints, fb.keypoints, fa.img_size, fb.img_size,
            )
            empty_h = np.zeros((0, 0), np.float64)
            mi = grid[i * n + j]
            mi.src_img_idx, mi.dst_img_idx = i, j
            mi.matches = info["matches"]
            mi.inliers_mask = info["inliers_mask"]
            mi.num_inliers = info["num_inliers"]
            mi.confidence = info["confidence"]
            mi.H = info["H"] if info["H"] is not None else empty_h
            # Dual (j, i): inverse H, swapped match indices (matchers.cpp:88-99).
            dual = grid[j * n + i]
            dual.src_img_idx, dual.dst_img_idx = j, i
            dual.matches = [cv2.DMatch(m.trainIdx, m.queryIdx, m.distance) for m in info["matches"]]
            dual.inliers_mask = info["inliers_mask"]
            dual.num_inliers = info["num_inliers"]
            dual.confidence = info["confidence"]
            dual.H = np.linalg.inv(info["H"]) if info["H"] is not None else empty_h
    return grid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, required=True, help="directory of source images")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--orb-features", type=int, default=500,
                    help="ORB nfeatures (500 = cv2.Stitcher default)")
    args = ap.parse_args()

    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if len(paths) < 2:
        raise SystemExit(f"need >= 2 images in {args.images}")
    out = args.out or Path(__file__).parent / "dumps" / args.images.name
    for sub in ("features", "matches", "gains", "seams"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    full_imgs = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in paths]
    for p, im in zip(paths, full_imgs):
        if im is None:
            raise SystemExit(f"cannot read {p}")

    # --- scales, latched from the FIRST image only (stitcher.cpp:424-453) ---
    area0 = full_imgs[0].shape[0] * full_imgs[0].shape[1]
    work_scale = min(1.0, math.sqrt(REGISTR_RESOL_MP * 1e6 / area0))
    seam_scale = min(1.0, math.sqrt(SEAM_EST_RESOL_MP * 1e6 / area0))
    seam_work_aspect = seam_scale / work_scale

    def scaled(img, s):
        if s == 1.0:
            return img.copy()
        return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR_EXACT)

    work_imgs = [scaled(im, work_scale) for im in full_imgs]
    seam_imgs = [scaled(im, seam_scale) for im in full_imgs]

    # --- stage 1: ORB features on work-scale images ---
    orb = cv2.ORB_create(nfeatures=args.orb_features)
    features = cv2.detail.computeImageFeatures(orb, work_imgs)
    for i, f in enumerate(features):
        f.img_idx = i
        desc = f.descriptors.get() if isinstance(f.descriptors, cv2.UMat) else f.descriptors
        (out / "features" / f"img_{i:03d}.json").write_text(
            json.dumps(keypoints_json(f.keypoints)))
        np.save(out / "features" / f"img_{i:03d}.desc.npy", desc)
    print(f"features: {[len(f.keypoints) for f in features]}")

    # --- stage 2: pairwise matching (deterministic BF-Hamming) ---
    pairwise = match_all(features)
    n = len(features)
    for i in range(n):
        for j in range(i + 1, n):
            mi = pairwise[i * n + j]
            (out / "matches" / f"pair_{i:03d}_{j:03d}.json").write_text(json.dumps({
                "numMatches": len(mi.matches),
                "numInliers": mi.num_inliers,
                "confidence": mi.confidence,
                "H": None if mi.H is None or np.asarray(mi.H).size == 0
                     else np.asarray(mi.H, dtype=np.float64).tolist(),
                "matches": [[m.queryIdx, m.trainIdx, m.distance] for m in mi.matches],
                "inliersMask": np.asarray(mi.inliers_mask).ravel().astype(int).tolist(),
            }))
    strong = sum(1 for i in range(n) for j in range(i + 1, n) if pairwise[i * n + j].confidence > CONF_THRESH)
    print(f"matches: {strong} pairs above confidence {CONF_THRESH}")

    # --- stage 3: keep the biggest connected component ---
    indices = cv2.detail.leaveBiggestComponent(features, pairwise, CONF_THRESH)
    indices = [int(x) for x in np.asarray(indices).ravel()]
    if len(indices) < n:
        # Like cv2.Stitcher, continue with the biggest connected component.
        # leaveBiggestComponent's Python binding returns kept indices; re-run
        # matching on the subset so downstream grids stay dense & consistent,
        # and subset every per-image list to keep indexing aligned.
        print(f"WARNING: only {len(indices)}/{n} images connected: {indices}")
        features = [features[i] for i in indices]
        full_imgs = [full_imgs[i] for i in indices]
        seam_imgs = [seam_imgs[i] for i in indices]
        paths = [paths[i] for i in indices]
        for k, f in enumerate(features):
            f.img_idx = k
        pairwise = match_all(features)
        n = len(features)

    # --- stage 4: rotation estimation + bundle adjustment + wave correction ---
    ok, cameras = cv2.detail_HomographyBasedEstimator().apply(features, pairwise, None)
    if not ok:
        raise SystemExit("homography-based estimation failed")
    for cam in cameras:
        cam.R = cam.R.astype(np.float32)
    (out / "cameras_initial.json").write_text(json.dumps([camera_json(c) for c in cameras]))

    ba = cv2.detail_BundleAdjusterRay()
    ba.setConfThresh(CONF_THRESH)
    ok, cameras = ba.apply(features, pairwise, cameras)
    if not ok:
        raise SystemExit("bundle adjustment failed")
    (out / "cameras_ba.json").write_text(json.dumps([camera_json(c) for c in cameras]))

    rmats = cv2.detail.waveCorrect([np.asarray(c.R) for c in cameras], cv2.detail.WAVE_CORRECT_HORIZ)
    for cam, r in zip(cameras, rmats):
        cam.R = r
    (out / "cameras_final.json").write_text(json.dumps([camera_json(c) for c in cameras]))

    focals = sorted(c.focal for c in cameras)
    if len(focals) % 2 == 1:
        warped_image_scale = focals[len(focals) // 2]
    else:
        # f32 cast of the sum BEFORE halving (stitcher.cpp:517-528).
        warped_image_scale = float(np.float32(focals[len(focals) // 2 - 1] + focals[len(focals) // 2]) * 0.5)
    print(f"cameras: focals median {warped_image_scale:.2f} (work-scale px)")

    # --- stage 5: seam-scale warp + gain compensation + graph-cut seams ---
    warper = cv2.PyRotationWarper("spherical", warped_image_scale * seam_work_aspect)
    corners, sizes, warped, warped_masks = [], [], [], []
    for i, img in enumerate(seam_imgs):
        K = np.array(cameras[i].K(), dtype=np.float32)
        K[0, 0] *= seam_work_aspect; K[0, 2] *= seam_work_aspect
        K[1, 1] *= seam_work_aspect; K[1, 2] *= seam_work_aspect
        corner, img_w = warper.warp(img, K, cameras[i].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        mask = np.full(img.shape[:2], 255, np.uint8)
        _, mask_w = warper.warp(mask, K, cameras[i].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        corners.append(corner); sizes.append((img_w.shape[1], img_w.shape[0]))
        warped.append(img_w); warped_masks.append(cv2.UMat(mask_w))

    compensator = cv2.detail_BlocksGainCompensator()
    compensator.feed(corners=corners, images=warped, masks=warped_masks)
    for i, g in enumerate(compensator.getMatGains()):
        np.save(out / "gains" / f"gain_{i:03d}.npy", np.asarray(g))

    seam_finder = cv2.detail_GraphCutSeamFinder("COST_COLOR")
    warped_f = [w.astype(np.float32) for w in warped]
    seam_finder.find(warped_f, corners, warped_masks)
    seam_masks = [np.asarray(m.get()) for m in warped_masks]
    for i, m in enumerate(seam_masks):
        cv2.imwrite(str(out / "seams" / f"mask_{i:03d}.png"), m)
    print(f"seams: {len(seam_masks)} masks at seam scale")

    # --- stage 6: full-res compose with multiband blending ---
    compose_work_aspect = 1.0 / work_scale
    warper = cv2.PyRotationWarper("spherical", warped_image_scale * compose_work_aspect)
    compose_corners, compose_sizes = [], []
    for i in range(n):
        K = np.array(cameras[i].K(), dtype=np.float32)
        K[0, 0] *= compose_work_aspect; K[0, 2] *= compose_work_aspect
        K[1, 1] *= compose_work_aspect; K[1, 2] *= compose_work_aspect
        h, w = full_imgs[i].shape[:2]
        roi = warper.warpRoi((w, h), K, cameras[i].R)
        compose_corners.append((roi[0], roi[1])); compose_sizes.append((roi[2], roi[3]))

    dst_roi = cv2.detail.resultRoi(corners=compose_corners, sizes=compose_sizes)
    blend_width = math.sqrt(dst_roi[2] * dst_roi[3]) * BLEND_STRENGTH / 100
    num_bands = max(1, int(math.ceil(math.log(blend_width) / math.log(2.0)) - 1))
    blender = cv2.detail_MultiBandBlender(try_gpu=0, num_bands=num_bands)
    blender.prepare(dst_roi)

    for i in range(n):
        K = np.array(cameras[i].K(), dtype=np.float32)
        K[0, 0] *= compose_work_aspect; K[0, 2] *= compose_work_aspect
        K[1, 1] *= compose_work_aspect; K[1, 2] *= compose_work_aspect
        corner, img_w = warper.warp(full_imgs[i], K, cameras[i].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        mask = np.full(full_imgs[i].shape[:2], 255, np.uint8)
        _, mask_w = warper.warp(mask, K, cameras[i].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        img_w = compensator.apply(i, corner, img_w, mask_w)
        # Upscale seam mask, constrain by warped coverage (stitcher.cpp compose loop).
        dilated = cv2.dilate(seam_masks[i], None)
        seam_up = cv2.resize(dilated, (mask_w.shape[1], mask_w.shape[0]), interpolation=cv2.INTER_LINEAR_EXACT)
        mask_final = cv2.bitwise_and(seam_up, mask_w)
        blender.feed(cv2.UMat(img_w.astype(np.int16)), mask_final, corner)

    result, result_mask = blender.blend(None, None)
    result = np.clip(np.asarray(result), 0, 255).astype(np.uint8)
    cv2.imwrite(str(out / "result.jpg"), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"result: {result.shape[1]}x{result.shape[0]} -> {out / 'result.jpg'}")

    (out / "compose.json").write_text(json.dumps({
        "corners": compose_corners, "sizes": compose_sizes,
        "dstRoi": list(dst_roi), "numBands": num_bands, "blendWidth": blend_width,
    }))
    (out / "meta.json").write_text(json.dumps({
        "opencv": cv2.__version__,
        "images": [p.name for p in paths],
        "keptIndices": indices,
        "workScale": work_scale, "seamScale": seam_scale,
        "seamWorkAspect": seam_work_aspect,
        "warpedImageScale": warped_image_scale,
        "constants": {
            "orbFeatures": args.orb_features,
            "matchConf": MATCH_CONF, "confThresh": CONF_THRESH,
            "numMatchesThresh": [NUM_MATCHES_THRESH1, NUM_MATCHES_THRESH2],
            "matchesConfidenceThresh": MATCHES_CONFIDENCE_THRESH,
            "blendStrength": BLEND_STRENGTH,
        },
        "elapsedSec": round(time.time() - t0, 2),
    }, indent=2))
    print(f"done in {time.time() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    sys.exit(main())
