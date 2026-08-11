#!/usr/bin/env python
"""Generate OpenCV reference fixtures for the MultiBandBlender port (blend.rs).

Run with the project venv:

    tools/.venv/bin/python tools/reference/gen_blend_fixtures.py

Emits fixtures under tools/reference/fixtures/blend/:

  pyrdown_i16c3.json   cv2.pyrDown on CV_16SC3 (FixPtCast<short,8>: int
                       accumulation, (sum + 128) >> 8, BORDER_REFLECT_101)
  pyrup_i16c3.json     cv2.pyrUp on CV_16SC3 (FixPtCast<short,6>), including
                       odd dst sizes (2s-1 / 2s+1)
  pyrdown_f32.json     cv2.pyrDown on CV_32FC1 (FltCast<float,8>). float
                       accumulation is NOT order-independent, and the CPU
                       path mixes two structures (established empirically by
                       testing every candidate association per element region
                       against this wheel — unique survivors):
                       * universal-intrinsic body (4 f32 NEON lanes):
                         horizontal fma(a,6,fma(b+c,4,d+e)) for columns
                         [1, 1+4k), vertical fma((r1+r3)+r2,4,(r0+r4)+2r2)/256
                         for columns [0, 4m);
                       * "scalar" tails, which clang -ffp-contract=on also
                         fuses: fma(a,6,(b+c)*4)+d+e, resp.
                         (fma(r2,6,(r1+r3)*4)+r0+r4)/256.
                       The Rust port replicates exactly this; an AVX2 build
                       of OpenCV (8 lanes) would differ in the last ulp once
                       values need >24 mantissa bits (weight pyramid levels
                       >= 4). f32 stored as u32 bit patterns.
  multiband.json       cv2.detail_MultiBandBlender(try_gpu=0) end-to-end:
                       prepare(dst_roi), feed(CV_16SC3, CV_8U mask, corner)*,
                       blend() -> raw CV_16SC3 result + mask + clipped u8.
  num_bands.json       stitcher blend_strength=5 num_bands formula samples.

Determinism: OpenCL is force-disabled (cv2.ocl.setUseOpenCL(False)) — the
detail_MultiBandBlender works on UMats internally and would otherwise take
the OpenCL kernels on this machine, which have different float semantics
from the CPU path the Rust port mirrors. cv2.pyrDown/pyrUp on numpy arrays
always take the CPU path. All inputs are closed-form modular patterns (no
RNG), so regeneration is reproducible byte-for-byte.

Integer arrays are stored as plain ints, f32 as IEEE-754 u32 bit patterns.
All matrices are row-major, channels interleaved.
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np

cv2.ocl.setUseOpenCL(False)
assert not cv2.ocl.useOpenCL()
# pyrDown values are independent of the parallel_for_ split (each range
# recomputes its ring buffer); single-threading is belt-and-braces for
# reproducibility and avoids the GCD backend spinning under sandboxes.
cv2.setNumThreads(1)

OUT = Path(__file__).parent / "fixtures" / "blend"
OUT.mkdir(parents=True, exist_ok=True)

BLEND_STRENGTH = 5


def i16_list(a) -> list:
    return [int(v) for v in np.asarray(a, np.int16).ravel()]


def u8_list(a) -> list:
    return [int(v) for v in np.asarray(a, np.uint8).ravel()]


def f32_bits(a) -> list:
    return [int(v) for v in np.asarray(a, np.float32).ravel().view(np.uint32)]


def pat_i16c3(w: int, h: int, lo: int, hi: int, seed: int) -> np.ndarray:
    """Deterministic full-range CV_16SC3 pattern."""
    y, x = np.mgrid[0:h, 0:w]
    span = hi - lo + 1
    out = np.empty((h, w, 3), np.int64)
    for c in range(3):
        out[..., c] = (x * (13 + seed) + y * 7 + c * 31 + x * y * (3 + c) + seed * 17) % span + lo
    return out.astype(np.int16)


def pat_img_u8c3(w: int, h: int, seed: int) -> np.ndarray:
    """Deterministic image-like (0..255) pattern, returned as int16 (the
    pipeline feeds warped u8 images converted to CV_16S)."""
    return pat_i16c3(w, h, 0, 255, seed)


def pat_f32(w: int, h: int, seed: int) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    v = (x * (31 + seed) + y * 17 + x * y * 3 + seed * 5) % 511 - 255
    return (v.astype(np.float32) * np.float32(1.0 / 255.0)).astype(np.float32)


def mask_full(w: int, h: int) -> np.ndarray:
    return np.full((h, w), 255, np.uint8)


def mask_vsplit(w: int, h: int, frac_num: int, frac_den: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    m[:, w * frac_num // frac_den:] = 255
    return m


def mask_diag(w: int, h: int) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    return np.where(x * h >= y * w, 255, 0).astype(np.uint8)


def mask_ramp(w: int, h: int) -> np.ndarray:
    """Non-binary mask (INTER_LINEAR_EXACT seam upscales produce grays)."""
    x = np.arange(w, dtype=np.int64)
    row = np.clip(x * 512 // max(w - 1, 1), 0, 255).astype(np.uint8)
    return np.tile(row, (h, 1))


def mask_hole(w: int, h: int) -> np.ndarray:
    m = np.full((h, w), 255, np.uint8)
    m[h // 4: h // 2 + 2, w // 3: 2 * w // 3] = 0
    return m


def gen_pyrdown_i16c3() -> None:
    cases = []
    sizes = [(1, 1), (2, 2), (3, 3), (4, 3), (5, 5), (6, 4), (8, 6), (9, 7),
             (12, 8), (16, 10), (31, 17), (40, 30), (64, 2), (7, 1)]
    for i, (w, h) in enumerate(sizes):
        src = pat_i16c3(w, h, -1020, 1020, i)
        dst = cv2.pyrDown(src)
        dh, dw = dst.shape[:2]
        cases.append({"name": f"{w}x{h}", "w": w, "h": h, "src": i16_list(src),
                      "dw": dw, "dh": dh, "dst": i16_list(dst)})
    (OUT / "pyrdown_i16c3.json").write_text(json.dumps({"cases": cases}))
    print(f"pyrdown_i16c3.json: {len(cases)} cases")


def gen_pyrup_i16c3() -> None:
    cases = []
    specs = [(1, 1, None), (2, 2, None), (3, 2, None), (4, 3, None),
             (4, 3, (7, 5)), (4, 3, (9, 7)), (5, 4, None), (5, 4, (9, 7)),
             (5, 4, (11, 9)), (8, 6, None), (13, 9, None), (16, 10, None)]
    for i, (w, h, dsz) in enumerate(specs):
        src = pat_i16c3(w, h, -1020, 1020, i)
        dst = cv2.pyrUp(src, dstsize=dsz) if dsz else cv2.pyrUp(src)
        dh, dw = dst.shape[:2]
        cases.append({"name": f"{w}x{h}->{dw}x{dh}", "w": w, "h": h,
                      "src": i16_list(src), "dw": dw, "dh": dh,
                      "dst": i16_list(dst)})
    (OUT / "pyrup_i16c3.json").write_text(json.dumps({"cases": cases}))
    print(f"pyrup_i16c3.json: {len(specs)} cases")


def gen_pyrdown_f32() -> None:
    cases = []
    # Widths straddle the width0 / SIMD-lane boundaries: width0 =
    # min((w-3)/2+1, dw), FMA lanes cover x in [1, width0-4] in steps of 4.
    sizes = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 4), (7, 3), (8, 8),
             (9, 5), (11, 7), (16, 16), (23, 9), (24, 12), (37, 11), (64, 6),
             (101, 7), (40, 1), (5, 1), (12, 2)]
    for i, (w, h) in enumerate(sizes):
        src = pat_f32(w, h, i)
        dst = cv2.pyrDown(src)
        dh, dw = dst.shape[:2]
        cases.append({"name": f"{w}x{h}", "w": w, "h": h,
                      "src_bits": f32_bits(src), "dw": dw, "dh": dh,
                      "dst_bits": f32_bits(dst)})
    # Weight-map-like input: binary mask * 1/255 (the exact feed conversion).
    m = mask_diag(30, 22)
    src = m.astype(np.float32) * np.float32(1.0 / 255.0)
    dst = cv2.pyrDown(src)
    cases.append({"name": "weightmap_30x22", "w": 30, "h": 22,
                  "src_bits": f32_bits(src), "dw": dst.shape[1],
                  "dh": dst.shape[0], "dst_bits": f32_bits(dst)})
    (OUT / "pyrdown_f32.json").write_text(json.dumps({"cases": cases}))
    print(f"pyrdown_f32.json: {len(cases)} cases")


def gen_multiband() -> None:
    cases = []

    def run(name: str, num_bands: int, dst_roi, feeds) -> None:
        blender = cv2.detail_MultiBandBlender(try_gpu=0, num_bands=num_bands)
        blender.prepare(dst_roi)
        for img, mask, corner in feeds:
            assert img.dtype == np.int16 and mask.dtype == np.uint8
            blender.feed(img, mask, corner)
        result, result_mask = blender.blend(None, None)
        result = np.asarray(result)
        result_mask = np.asarray(result_mask)
        assert result.dtype == np.int16
        assert result.shape[:2] == (dst_roi[3], dst_roi[2])
        clipped = np.clip(result, 0, 255).astype(np.uint8)
        cases.append({
            "name": name, "num_bands": num_bands, "dst_roi": list(dst_roi),
            "feeds": [{"corner": [int(corner[0]), int(corner[1])],
                       "w": img.shape[1], "h": img.shape[0],
                       "img": i16_list(img), "mask": u8_list(mask)}
                      for img, mask, corner in feeds],
            "result_i16": i16_list(result),
            "result_mask": u8_list(result_mask),
            "result_u8": u8_list(clipped),
        })
        print(f"  {name}: roi={dst_roi} bands={num_bands} feeds={len(feeds)}")

    # A: two overlapping tiles, negative origin, partial tiles (gap=12 <<
    # roi) so the tl/br snapping arithmetic is exercised for real: corner 2's
    # snapped offset (18+... - 12 - (-7) = 13) is not a multiple of 2^nb, and
    # the roi (73x49) needs padding to 76x52.
    corners = [(-7, -3), (18, 10)]
    sizes = [(48, 36), (48, 36)]
    x0 = min(c[0] for c in corners)
    y0 = min(c[1] for c in corners)
    x1 = max(c[0] + s[0] for c, s in zip(corners, sizes))
    y1 = max(c[1] + s[1] for c, s in zip(corners, sizes))
    roi = (x0, y0, x1 - x0, y1 - y0)
    run("two_overlap_nb2", 2, roi, [
        (pat_img_u8c3(48, 36, 1), mask_full(48, 36), corners[0]),
        (pat_img_u8c3(48, 36, 2), mask_diag(48, 36), corners[1]),
    ])

    # B: num_bands=5 -> roi padded 70x50 -> 96x64, gap covers the whole roi,
    # BORDER_REFLECT tile expansion dominates. Ramp (non-binary) mask.
    run("pad_nb5", 5, (0, 0, 70, 50), [
        (pat_img_u8c3(40, 40, 3), mask_vsplit(40, 40, 1, 4), (0, 0)),
        (pat_img_u8c3(40, 30, 4), mask_ramp(40, 30), (25, 15)),
    ])

    # C: three images, nb=1, mask with a hole; includes a seam-ish vsplit.
    run("three_nb1", 1, (0, 0, 60, 24), [
        (pat_img_u8c3(30, 24, 5), mask_full(30, 24), (0, 0)),
        (pat_img_u8c3(30, 24, 6), mask_hole(30, 24), (15, 0)),
        (pat_img_u8c3(30, 24, 7), mask_vsplit(30, 24, 1, 3), (30, 0)),
    ])

    # D: tiny roi crops num_bands: min(5, ceil(log2(6))) = 3, roi 6x6 -> 8x8,
    # feeds smaller than the padded roi (reflect border + weight zero-pad).
    run("tiny_nb_crop", 5, (0, 0, 6, 6), [
        (pat_img_u8c3(6, 6, 8), mask_full(6, 6), (0, 0)),
        (pat_img_u8c3(4, 4, 9), mask_full(4, 4), (2, 2)),
    ])

    (OUT / "multiband.json").write_text(json.dumps({"cases": cases}))
    print(f"multiband.json: {len(cases)} cases")


def gen_num_bands() -> None:
    cases = []
    for w, h in [(100, 80), (640, 480), (1920, 1080), (4000, 1500),
                 (70, 50), (16, 16), (2, 2), (7000, 3000)]:
        blend_width = math.sqrt(w * h) * BLEND_STRENGTH / 100
        nb = max(1, int(math.ceil(math.log(blend_width) / math.log(2.0)) - 1))
        cases.append({"w": w, "h": h, "num_bands": nb})
    (OUT / "num_bands.json").write_text(json.dumps({"cases": cases}))
    print(f"num_bands.json: {len(cases)} cases")


if __name__ == "__main__":
    print(f"OpenCV {cv2.__version__}, OpenCL active: {cv2.ocl.useOpenCL()}")
    gen_pyrdown_i16c3()
    gen_pyrup_i16c3()
    gen_pyrdown_f32()
    gen_multiband()
    gen_num_bands()
    print(f"fixtures -> {OUT}")
