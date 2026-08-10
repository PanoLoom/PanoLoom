# OpenCV oracle harness

The quality reference for the Rust engine. `oracle.py` replicates
`cv2.Stitcher` PANORAMA mode stage by stage (constants and behavior documented
in `docs/pipeline.md`) and dumps every intermediate; panoloom-core's tests
assert parity against these dumps stage by stage.

## Setup

```sh
cd tools
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # opencv-python pinned < 5
```

## Usage

```sh
# generate a dataset first (see ../testdata), then:
.venv/bin/python reference/oracle.py --images testdata/generated/ring_kloppenheim_06

# score recovered poses against the dataset's ground truth:
.venv/bin/python reference/compare_poses.py \
    --dump reference/dumps/ring_kloppenheim_06 \
    --truth testdata/generated/ring_kloppenheim_06
```

Dump layout is documented in `oracle.py`'s docstring.

## Determinism

Two runs produce byte-identical dumps (only `meta.json`'s `elapsedSec`
differs). This requires the harness's own brute-force Hamming matcher —
OpenCV's `BestOf2NearestMatcher` uses randomized FLANN-LSH for binary
descriptors and is NOT deterministic. The BF matcher follows
`matchers.cpp` semantics exactly and is what the Rust port implements.

## Measured baselines (2026-08-10, OpenCV 4.14, defaults)

Relative-pose error vs synthetic ground truth — this is OpenCV's own
accuracy, i.e. the bar for the Rust port (match within ~10-20%):

| Dataset | Cameras | Mean | p95 | Max |
|---|---|---|---|---|
| ring_kloppenheim_06 (1 row, landscape) | 8 | 0.611° | 1.249° | 1.466° |
| sphere_kloppenheim_06 (3 rows + z/n, portrait) | 26 | 0.204° | 0.374° | 0.512° |

Single-row geometry is weakly constrained; more rows → tighter poses.

## Known quirks

- **Upside-down spherical results**: `waveCorrect(HORIZ)` has a global
  180° sign ambiguity and can flip the pano. Relative geometry is
  unaffected (compare_poses.py cancels it). The engine will resolve
  orientation properly in the auto-level milestone; the oracle keeps
  OpenCV's behavior for parity.
- **Python-constructed `cv2.detail.MatchesInfo` segfaults** (opencv-python
  4.14) when passed back into C++ (`leaveBiggestComponent`, estimators).
  Workaround in `match_all`: obtain the grid from the real matcher, then
  overwrite every field with the deterministic BF results.
- Landscape shots with rows 45° apart have ~0° vertical overlap (hfov 65 →
  vfov ~46°) and won't connect — the sphere test preset shoots portrait,
  like real pano-head workflows.
