#!/usr/bin/env python3
"""Score oracle-recovered camera poses against a dataset's ground truth.

Compares RELATIVE rotations (pairwise), which cancels the global rotation
ambiguity (including OpenCV waveCorrect's occasional 180° flip). The same
scoring applies to panoloom-core's optimizer output at milestone M2.

Usage: compare_poses.py --dump dumps/<set> --truth ../testdata/generated/<set>
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "testdata"))
from generate import camera_rotation  # noqa: E402


def rotation_angle_deg(m: np.ndarray) -> float:
    return math.degrees(math.acos(np.clip((np.trace(m) - 1) / 2, -1.0, 1.0)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", type=Path, required=True, help="oracle dump directory")
    ap.add_argument("--truth", type=Path, required=True, help="generated dataset directory")
    ap.add_argument("--max-mean", type=float, default=None, help="fail if mean error exceeds this (degrees)")
    args = ap.parse_args()

    cams = json.loads((args.dump / "cameras_final.json").read_text())
    meta = json.loads((args.dump / "meta.json").read_text())
    truth = json.loads((args.truth / "ground_truth.json").read_text())
    truth_by_name = {im["fileName"]: im for im in truth["images"]}

    # meta.json's image list is already subset to the stitched component.
    names = meta["images"]
    kept = meta["keptIndices"]
    if len(kept) < len(names):
        names = [names[i] for i in kept]

    r_est = [np.array(c["R"]) for c in cams]
    r_true = [
        camera_rotation(t["yaw"], t["pitch"], t["roll"])
        for t in (truth_by_name[n] for n in names)
    ]
    assert len(r_est) == len(r_true), f"{len(r_est)} cameras vs {len(r_true)} truths"

    errs = []
    for i in range(len(r_est)):
        for j in range(i + 1, len(r_est)):
            rel_est = r_est[i].T @ r_est[j]
            rel_true = r_true[i].T @ r_true[j]
            errs.append(rotation_angle_deg(rel_est @ rel_true.T))

    mean, mx = float(np.mean(errs)), float(np.max(errs))
    print(f"{len(r_est)} cameras, {len(errs)} pairs: mean {mean:.3f}°, p95 "
          f"{float(np.percentile(errs, 95)):.3f}°, max {mx:.3f}°")
    if args.max_mean is not None and mean > args.max_mean:
        raise SystemExit(f"FAIL: mean error {mean:.3f}° > {args.max_mean}°")


if __name__ == "__main__":
    main()
