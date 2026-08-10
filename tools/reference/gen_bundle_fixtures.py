#!/usr/bin/env python
"""Generate OpenCV reference fixtures for the bundle-adjustment port (bundle.rs).

Run with the project venv:

    tools/.venv/bin/python tools/reference/gen_bundle_fixtures.py

Emits fixtures under tools/reference/fixtures/:

  rodrigues/m2v_f32.json   cv2.Rodrigues(3x3 CV_32F) -> rvec CV_32F
  rodrigues/v2m_f64.json   cv2.Rodrigues(3x1 CV_64F) -> R CV_64F
  bundle/svd3x3_f32.json   cv2.SVDecomp(FULL_UV) on CV_32F 3x3 + the whole
                           setUpInitialCameraParams orthonormalize+Rodrigues
                           chain (u*vt via cv2.gemm, cv2.determinant, sign
                           flip, Rodrigues)
  bundle/norms_f64.json    cv2.norm NORM_L2 / NORM_L2|NORM_RELATIVE on CV_64F
  bundle/solve_svd.json    cv2.solve(DECOMP_SVD) on square CV_64F systems.
                           n < 25 exercises OpenCV's own JacobiSVD; n >= 25
                           goes to LAPACK (Accelerate dgesdd on macOS wheels)
                           and is marked "lapack": true (bit-parity is not
                           expected there, only closeness).
  bundle/gemm_jt.json      cv2.gemm(J, J, GEMM_1_T) and cv2.gemm(J, e,
                           GEMM_1_T) - JtJ/JtErr shapes. rows < 100 uses
                           OpenCV's own gemm; rows >= 100 is delegated to
                           BLAS (Accelerate) and marked "lapack": true.
  bundle/gemm3x3.json      3x3 cv2.gemm in f64 and f32 + cv2.invert
                           (DECOMP_LU) 3x3 in f64 and f32 + determinant f32
  bundle/ba_trajectory_{dataset}.json
                           BundleAdjusterRay outputs after k = 1,2,3,...
                           iterations (setTermCriteria), for step-by-step
                           debugging of the LM loop against the dumps.

All floats are stored as IEEE-754 bit patterns (u64/u32 ints) so the fixtures
are exact.  Layout of every matrix is row-major nested lists.
"""
import json
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).parent
FIX = ROOT / "fixtures"
DUMPS = ROOT / "dumps"

rng = np.random.default_rng(20260810)


def b64(x) -> int:
    return int(np.float64(x).view(np.uint64))


def b32(x) -> int:
    return int(np.float32(x).view(np.uint32))


def mat64(M):
    return [[b64(v) for v in row] for row in np.asarray(M, np.float64)]


def mat32(M):
    return [[b32(v) for v in row] for row in np.asarray(M, np.float32)]


def vec64(v):
    return [b64(x) for x in np.asarray(v, np.float64).ravel()]


def vec32(v):
    return [b32(x) for x in np.asarray(v, np.float32).ravel()]


def rand_rot():
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def axis_angle(axis, angle):
    axis = np.asarray(axis, np.float64)
    axis = axis / np.linalg.norm(axis)
    return cv2.Rodrigues(axis * angle)[0]


# --------------------------------------------------------------------------
# rodrigues fixtures
# --------------------------------------------------------------------------
def gen_rodrigues():
    out = FIX / "rodrigues"
    out.mkdir(parents=True, exist_ok=True)

    m2v = []
    mats = []
    for i in range(40):
        r = rand_rot()
        if i % 3 == 1:
            # BA feeds slightly non-orthonormal CV_32F matrices.
            r = r + rng.standard_normal((3, 3)) * 1e-3
        mats.append(r)
    mats.append(np.eye(3))                        # theta = 0 branch
    mats.append(axis_angle([1, 2, 3], np.pi))     # s < 1e-5, c < 0 branch
    mats.append(axis_angle([0, 0, 1], np.pi))
    mats.append(axis_angle([1, 0, 0], np.pi))
    mats.append(axis_angle([1, 2, 3], 1e-7))      # tiny angle
    for r in mats:
        r32 = np.float32(r)
        rvec = cv2.Rodrigues(r32)[0]
        assert rvec.dtype == np.float32
        m2v.append({"r": mat32(r32), "rvec": vec32(rvec)})
    (out / "m2v_f32.json").write_text(json.dumps(m2v))

    v2m = []
    vecs = [rng.standard_normal(3) * s for s in (1e-20, 1e-8, 0.1, 1.0, 2.0, 3.0) for _ in range(6)]
    vecs.append(np.zeros(3))
    vecs.append(np.array([np.pi, 0.0, 0.0]))
    for v in vecs:
        v = np.asarray(v, np.float64).reshape(3, 1)
        r = cv2.Rodrigues(v)[0]
        assert r.dtype == np.float64
        v2m.append({"rvec": vec64(v), "r": mat64(r)})
    (out / "v2m_f64.json").write_text(json.dumps(v2m))
    print(f"rodrigues: {len(m2v)} m2v + {len(v2m)} v2m cases")


# --------------------------------------------------------------------------
# SVD 3x3 f32 + full setUpInitialCameraParams chain
# --------------------------------------------------------------------------
def gen_svd3x3():
    out = FIX / "bundle"
    out.mkdir(parents=True, exist_ok=True)
    cases = []
    mats = []
    for i in range(40):
        r = rand_rot()
        if i % 2 == 1:
            r = r + rng.standard_normal((3, 3)) * (1e-4 if i % 4 == 1 else 1e-2)
        if i % 7 == 3:
            r = -r  # negative determinant input
        mats.append(np.float32(r))
    mats.append(np.float32(np.eye(3)))
    for r32 in mats:
        w, u, vt = cv2.SVDecomp(r32, flags=cv2.SVD_FULL_UV)
        assert u.dtype == np.float32
        ortho = cv2.gemm(u, vt, 1.0, None, 0.0)
        det = cv2.determinant(ortho)
        flipped = det < 0
        if flipped:
            ortho = np.float32(ortho * -1)  # Mat *= -1 (exact sign flip)
        rvec = cv2.Rodrigues(ortho)[0]
        cases.append({
            "r": mat32(r32),
            "w": vec32(w),
            "u": mat32(u),
            "vt": mat32(vt),
            "ortho": mat32(ortho),
            "det": b64(det),
            "flipped": bool(flipped),
            "rvec": vec32(rvec),
        })
    (out / "svd3x3_f32.json").write_text(json.dumps(cases))
    print(f"svd3x3: {len(cases)} cases")


# --------------------------------------------------------------------------
# norms
# --------------------------------------------------------------------------
def gen_norms():
    out = FIX / "bundle"
    out.mkdir(parents=True, exist_ok=True)
    l2 = []
    for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 23, 32, 33,
              104, 105, 1191, 8655]:
        x = rng.standard_normal(n) * 3
        l2.append({"x": vec64(x), "norm": b64(cv2.norm(x, cv2.NORM_L2))})
    rel = []
    for n in [3, 7, 32, 33, 104, 105]:
        a = rng.standard_normal(n).reshape(-1, 1)
        b = a + rng.standard_normal(n).reshape(-1, 1) * 1e-3
        r = cv2.norm(a, b, cv2.NORM_L2 | cv2.NORM_RELATIVE)
        rel.append({"a": vec64(a), "b": vec64(b), "norm": b64(r)})
    (out / "norms_f64.json").write_text(json.dumps({"l2": l2, "rel_l2": rel}))
    print(f"norms: {len(l2)} l2 + {len(rel)} relative cases")


# --------------------------------------------------------------------------
# solve DECOMP_SVD
# --------------------------------------------------------------------------
def gen_solve():
    out = FIX / "bundle"
    out.mkdir(parents=True, exist_ok=True)
    cases = []
    for n in [4, 8, 12, 16, 24, 32, 104]:
        for damp in (1e-3, 1.0):
            j = rng.standard_normal((n + 13, n))
            a = j.T @ j
            a[np.diag_indices(n)] *= 1.0 + damp
            b = rng.standard_normal((n, 1))
            ok, x = cv2.solve(a, b, flags=cv2.DECOMP_SVD)
            assert ok
            cases.append({
                "n": n,
                "lapack": n >= 25,  # HAL_SVD_SMALL_MATRIX_THRESH
                "a": mat64(a),
                "b": vec64(b),
                "x": vec64(x),
            })
    (out / "solve_svd.json").write_text(json.dumps(cases))
    print(f"solve: {len(cases)} cases")


# --------------------------------------------------------------------------
# gemm JtJ / JtErr shapes
# --------------------------------------------------------------------------
def gen_gemm():
    out = FIX / "bundle"
    out.mkdir(parents=True, exist_ok=True)
    cases = []
    # rows < 100: OpenCV's own gemm (bit-parity expected).
    #   (57, 8):  n < 64 -> scalar 4-column j-outer branch
    #   (96, 32): kj SIMD branch, even columns, no dot tail
    #   (96, 33): kj SIMD branch + odd trailing column
    #   (99, 12): odd rows -> dot-product scalar tail of 1
    # rows >= 100: Accelerate BLAS on the oracle machine ("lapack": true);
    # (1191, 32) is the real ring JtJ shape, (150, 104) exercises the
    # cols > 64 dispatch (blocked path on a LAPACK-free OpenCV build).
    for rows, cols in [(57, 8), (96, 32), (96, 33), (99, 12),
                       (1191, 32), (150, 104)]:
        j = rng.standard_normal((rows, cols))
        e = rng.standard_normal((rows, 1))
        jtj = cv2.gemm(j, j, 1.0, None, 0.0, flags=cv2.GEMM_1_T)
        jte = cv2.gemm(j, e, 1.0, None, 0.0, flags=cv2.GEMM_1_T)
        cases.append({
            "rows": rows,
            "cols": cols,
            "lapack": rows >= 100,  # HAL_GEMM_SMALL_MATRIX_THRESH
            "j": mat64(j),
            "e": vec64(e),
            "jtj": mat64(jtj),
            "jte": vec64(jte),
        })
    (out / "gemm_jt.json").write_text(json.dumps(cases))

    g3 = {"f64": [], "f32": [], "inv_f64": [], "inv_f32": [], "det_f32": []}
    for _ in range(30):
        a = rng.standard_normal((3, 3)) * 2
        b = rng.standard_normal((3, 3)) * 2
        g3["f64"].append({"a": mat64(a), "b": mat64(b),
                          "d": mat64(cv2.gemm(a, b, 1.0, None, 0.0))})
        a32, b32 = np.float32(a), np.float32(b)
        g3["f32"].append({"a": mat32(a32), "b": mat32(b32),
                          "d": mat32(cv2.gemm(a32, b32, 1.0, None, 0.0))})
        k = np.array([[500.0 + rng.random() * 300, 0.0, rng.random() * 500],
                      [0.0, 500.0 + rng.random() * 300, rng.random() * 500],
                      [0.0, 0.0, 1.0]])
        _, ki = cv2.invert(k, flags=cv2.DECOMP_LU)
        g3["inv_f64"].append({"a": mat64(k), "inv": mat64(ki)})
        r32 = np.float32(rand_rot())
        _, ri = cv2.invert(r32, flags=cv2.DECOMP_LU)
        g3["inv_f32"].append({"a": mat32(r32), "inv": mat32(ri)})
        g3["det_f32"].append({"a": mat32(r32), "det": b64(cv2.determinant(r32))})
    (out / "gemm3x3.json").write_text(json.dumps(g3))
    print(f"gemm: {len(cases)} JtJ cases + 30x 3x3 cases")


# --------------------------------------------------------------------------
# BundleAdjusterRay trajectory (per-iteration oracle) from the dumps
# --------------------------------------------------------------------------
def load_ba_inputs(root: Path, n: int):
    orb = cv2.ORB_create(nfeatures=500)
    work = [cv2.imread(str(root / "work" / f"img_{i:03d}.png"), cv2.IMREAD_COLOR)
            for i in range(n)]
    features = cv2.detail.computeImageFeatures(orb, work)
    for i, f in enumerate(features):
        f.img_idx = i
        kps = json.loads((root / "features" / f"img_{i:03d}.json").read_text())
        assert len(f.keypoints) == len(kps), "work PNGs no longer match dumps"

    # Python-built MatchesInfo segfault workaround (see oracle.py): mutate a
    # matcher-initialized grid.
    cv2.setRNGSeed(12345)
    grid = list(cv2.detail_BestOf2NearestMatcher(False).apply2(features))
    empty_h = np.zeros((0, 0), np.float64)
    for i in range(n):
        for j in range(n):
            mi = grid[i * n + j]
            if i == j:
                mi.src_img_idx, mi.dst_img_idx = -1, -1
                mi.matches = []
                mi.inliers_mask = np.zeros((0,), np.uint8)
                mi.num_inliers = 0
                mi.confidence = 0.0
                mi.H = empty_h
                continue
            a, b = (i, j) if i < j else (j, i)
            d = json.loads((root / "matches" / f"pair_{a:03d}_{b:03d}.json").read_text())
            mi.src_img_idx, mi.dst_img_idx = i, j
            if i < j:
                mi.matches = [cv2.DMatch(int(q), int(t), float(dist))
                              for q, t, dist in d["matches"]]
            else:
                mi.matches = [cv2.DMatch(int(t), int(q), float(dist))
                              for q, t, dist in d["matches"]]
            mi.inliers_mask = np.array(d["inliersMask"], np.uint8)
            mi.num_inliers = int(d["numInliers"])
            mi.confidence = float(d["confidence"])
            if d["H"] is None:
                mi.H = empty_h
            else:
                h = np.array(d["H"], np.float64)
                mi.H = h if i < j else np.linalg.inv(h)

    cams_init = json.loads((root / "cameras_initial.json").read_text())
    ok, cameras = cv2.detail_HomographyBasedEstimator().apply(features, grid, None)
    assert ok
    for cam, c in zip(cameras, cams_init):
        cam.focal = c["focal"]
        cam.aspect = c["aspect"]
        cam.ppx = c["ppx"]
        cam.ppy = c["ppy"]
        cam.R = np.array(c["R"], np.float32)
    return features, grid, cameras


def clone_cameras(cameras):
    out = []
    for c in cameras:
        d = cv2.detail.CameraParams()
        d.focal = c.focal
        d.aspect = c.aspect
        d.ppx = c.ppx
        d.ppy = c.ppy
        d.R = c.R.copy()
        d.t = c.t.copy()
        out.append(d)
    return out


def gen_trajectory():
    out = FIX / "bundle"
    out.mkdir(parents=True, exist_ok=True)
    eps = float(np.finfo(np.float64).eps)
    for ds in ["ring_kloppenheim_06", "sphere_kloppenheim_06"]:
        root = DUMPS / ds
        if not root.exists():
            print(f"trajectory: {ds} dumps missing, skipped")
            continue
        meta = json.loads((root / "meta.json").read_text())
        n = min(len(meta["keptIndices"]), len(meta["images"]))
        features, grid, cameras = load_ba_inputs(root, n)

        # Sanity: default criteria must reproduce cameras_ba.json bit-exactly.
        ba = cv2.detail_BundleAdjusterRay()
        ba.setConfThresh(1.0)
        ok, full = ba.apply(features, grid, clone_cameras(cameras))
        assert ok
        cams_ba = json.loads((root / "cameras_ba.json").read_text())
        for c, o in zip(full, cams_ba):
            assert c.focal == o["focal"]
            assert (np.asarray(c.R, np.float64) == np.array(o["R"])).all()

        steps = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 1000]
        traj = []
        full_state = [(c.focal, np.asarray(c.R, np.float64)) for c in full]
        iters_used = None
        for k in steps:
            ba_k = cv2.detail_BundleAdjusterRay()
            ba_k.setConfThresh(1.0)
            ba_k.setTermCriteria(
                (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, k, eps))
            ok, cams_k = ba_k.apply(features, grid, clone_cameras(cameras))
            assert ok
            traj.append({
                "iters": k,
                "cameras": [{"focal": b64(c.focal), "r": mat64(np.asarray(c.R, np.float64))}
                            for c in cams_k],
            })
            if iters_used is None and all(
                    c.focal == f and (np.asarray(c.R, np.float64) == r).all()
                    for c, (f, r) in zip(cams_k, full_state)):
                iters_used = k
        (out / f"ba_trajectory_{ds}.json").write_text(json.dumps({
            "opencv": cv2.__version__,
            "convergedByIters": iters_used,
            "trajectory": traj,
        }))
        print(f"trajectory {ds}: converged within {iters_used} iterations")


if __name__ == "__main__":
    gen_rodrigues()
    gen_svd3x3()
    gen_norms()
    gen_solve()
    gen_gemm()
    gen_trajectory()
    print("fixtures written to", FIX)
