#!/usr/bin/env python
"""Generate cv2.findHomography(RANSAC) reference fixtures for the Rust port.

Run with the project venv:

    tools/.venv/bin/python tools/reference/gen_homography_fixtures.py

For each case we dump the float32 src/dst points that were fed to OpenCV,
plus the H matrix and inlier mask that cv2.findHomography(src, dst,
cv2.RANSAC) returned (defaults: ransacReprojThreshold=3.0, maxIters=2000,
confidence=0.995 — the exact configuration find_homography() in
crates/panoloom-core/src/homography.rs hard-codes).

IMPORTANT: points are float32 end to end. The Rust port takes [[f32; 2]]
(the matcher feeds centered Point2f coordinates), so the arrays handed to
cv2 must be float32 too or the RANSAC subsets diverge. float32 values
round-trip exactly through JSON (json emits the shortest repr of the f64
promotion, which parses back to the identical f32).

Fixtures land in tools/reference/fixtures/homography/case_N.json.
"""

import json
import os

import cv2
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "homography")


def apply_h(h, pts):
    """Project Nx2 points through a 3x3 homography (float64 math)."""
    pts = np.asarray(pts, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1))
    p = np.hstack([pts, ones]) @ h.T
    return p[:, :2] / p[:, 2:3]


def make_case(name, noise_free, src, dst):
    src32 = np.ascontiguousarray(np.asarray(src, dtype=np.float32).reshape(-1, 1, 2))
    dst32 = np.ascontiguousarray(np.asarray(dst, dtype=np.float32).reshape(-1, 1, 2))
    h, mask = cv2.findHomography(src32, dst32, cv2.RANSAC)
    assert h is not None, f"{name}: cv2.findHomography failed"
    assert mask is not None and mask.size == src32.shape[0]
    n_inl = int(mask.sum())
    assert n_inl >= 4, f"{name}: too few inliers ({n_inl})"
    print(f"  {name}: {src32.shape[0]} pts, {n_inl} inliers, h22={h[2, 2]:.17g}")
    return {
        "name": name,
        # Rust test tolerance selector: 1e-6 (noise-free) vs 1e-4 (noisy).
        "noise_free": noise_free,
        "method": "RANSAC",
        "ransac_reproj_threshold": 3.0,
        "max_iters": 2000,
        "confidence": 0.995,
        "src": [[float(p[0, 0]), float(p[0, 1])] for p in src32],
        "dst": [[float(p[0, 0]), float(p[0, 1])] for p in dst32],
        "H": [[float(x) for x in row] for row in h],
        "mask": [int(v) for v in mask.ravel()],
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"OpenCV {cv2.__version__} -> {OUT_DIR}")
    cases = []

    # --- case 1: clean 4-point (direct DLT branch, no RANSAC/LM) ---------
    h1 = np.array([[1.05, 0.03, 12.0], [-0.02, 0.97, -7.0], [1e-5, -2e-5, 1.0]])
    src = np.array([[0.0, 0.0], [640.0, 0.0], [640.0, 480.0], [0.0, 480.0]])
    dst = apply_h(h1, src)
    cases.append(make_case("clean_4pt", True, src, dst))

    # --- case 2: 30 points, exact mapping (f32 rounding only) ------------
    rng = np.random.default_rng(2)
    h2 = np.array([[0.95, -0.04, 25.0], [0.05, 1.08, -14.0], [3e-5, 1e-5, 1.0]])
    src = rng.uniform([0.0, 0.0], [640.0, 480.0], size=(30, 2))
    dst = apply_h(h2, src)
    cases.append(make_case("exact_30pt", True, src, dst))

    # --- case 3: 50 points, gaussian noise sigma=1 ------------------------
    rng = np.random.default_rng(3)
    h3 = np.array([[1.02, 0.06, -18.0], [-0.03, 0.99, 22.0], [-2e-5, 4e-5, 1.0]])
    src = rng.uniform([0.0, 0.0], [640.0, 480.0], size=(50, 2))
    dst = apply_h(h3, src) + rng.normal(0.0, 1.0, size=(50, 2))
    cases.append(make_case("noise_sigma1_50pt", False, src, dst))

    # --- case 4: 50 points, 30% gross outliers ----------------------------
    rng = np.random.default_rng(4)
    h4 = np.array([[0.9, 0.02, 40.0], [-0.05, 1.1, -30.0], [1e-5, -3e-5, 1.0]])
    src = rng.uniform([0.0, 0.0], [640.0, 480.0], size=(50, 2))
    dst = apply_h(h4, src) + rng.normal(0.0, 0.5, size=(50, 2))
    outliers = rng.choice(50, size=15, replace=False)
    dst[outliers] = rng.uniform([0.0, 0.0], [640.0, 480.0], size=(15, 2)) + [700.0, 500.0]
    cases.append(make_case("outliers_30pct_50pt", False, src, dst))

    # --- case 5: near-degenerate — collinear triple in the sample pool ----
    # Three points exactly collinear in f32 (cross products are exactly 0
    # in the f64 collinearity test on both sides, FMA or not), so RANSAC
    # subsets that draw one of them last with the other two present are
    # rejected by checkSubset/haveCollinearPoints.
    rng = np.random.default_rng(5)
    h5 = np.array([[1.0, 0.05, 8.0], [0.04, 0.96, -12.0], [2e-5, 2e-5, 1.0]])
    src = np.vstack(
        [
            np.array([[50.0, 50.0], [150.0, 150.0], [250.0, 250.0]]),  # collinear
            rng.uniform([0.0, 0.0], [640.0, 480.0], size=(12, 2)),
        ]
    )
    dst = apply_h(h5, src)
    dst[7] += [120.0, -90.0]  # two gross outliers keep RANSAC iterating
    dst[11] += [-80.0, 140.0]
    cases.append(make_case("near_degenerate_collinear", True, src, dst))

    # --- case 6: centered coordinates (negative values, matcher-style) ----
    rng = np.random.default_rng(6)
    ang = np.deg2rad(4.0)
    h6 = np.array(
        [
            [np.cos(ang) * 1.01, -np.sin(ang), 6.5],
            [np.sin(ang), np.cos(ang) * 0.99, -4.25],
            [4e-5, -1e-5, 1.0],
        ]
    )
    src = rng.uniform([-320.0, -240.0], [320.0, 240.0], size=(40, 2))
    dst = apply_h(h6, src) + rng.normal(0.0, 0.5, size=(40, 2))
    outliers = rng.choice(40, size=4, replace=False)
    dst[outliers] += rng.uniform(50.0, 200.0, size=(4, 2)) * np.sign(
        rng.normal(size=(4, 2))
    )
    cases.append(make_case("centered_negative_coords", False, src, dst))

    for i, case in enumerate(cases, start=1):
        path = os.path.join(OUT_DIR, f"case_{i}.json")
        with open(path, "w") as f:
            json.dump(case, f, indent=1)
            f.write("\n")
    print(f"wrote {len(cases)} fixtures")


if __name__ == "__main__":
    main()
