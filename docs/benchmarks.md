# Benchmarks

PanoLoom's engine is a stage-by-stage port of OpenCV's `stitching::detail`
pipeline, validated for parity against it (see [pipeline.md](pipeline.md)
and the oracle harness in `tools/reference`). Because both implementations
run the same algorithms on the same inputs, the comparison below is as
close to apples-to-apples as stitching benchmarks get.

## Setup

- Hardware: Apple M2 MacBook (10-core), macOS.
- **OpenCV**: `opencv-python` 4.14 wheel via the oracle harness
  (`tools/reference/oracle.py`), OpenCL disabled, OpenCV's own default
  multithreading left on. Timed with the harness's built-in wall clock
  (`elapsedSec` in `meta.json`); it runs the full pipeline — features,
  matching, bundle adjustment, seams, exposure, multiband blend — and
  writes the panorama plus all intermediates (dump I/O included in its
  time; noted below).
- **PanoLoom native**: `cargo run --release --example profile_align`
  (align + full preview compose) on the identical registration-scale
  inputs, rayon across all cores.
- **PanoLoom browser**: Chromium, wasm SIMD + a 10-worker thread pool
  over SharedArrayBuffer, timings from the Playwright e2e suite (these
  include browser-side JPEG decode of the sources).

Runs were repeated and medians taken. Numbers refresh with
`tools/reference/oracle.py --images <set>` and
`PANOLOOM_TIMING=1 cargo run --release --example profile_align -- <work-dir>`.

## Full-stitch wall time (align + compose)

| Test set | OpenCV 4.14 (native)¹ | PanoLoom engine (native)² | PanoLoom in the browser³ |
|---|---|---|---|
| Ring — 8 × 1.7 MP | 0.9 s | **0.4 s** | 1.4 s preview · ~8 s with full-res export |
| Sphere — 26 × 1.7 MP | 39.7 s | **26.1 s** | — |
| DJI drone sphere — 33 × 12.6 MP | 50.4 s *(places 25/33)* | **32.0 s** *(places 33/33)* | 55.6 s preview · 4.2 min with 147 MP export *(33/33)* |

¹ Full pipeline to a full-resolution panorama (ring 7849 px, sphere 5257 px,
DJI 17,141 px wide), `elapsedSec` from the harness, medians of 2 runs.
² Align + interactive preview (4096 px cap) via `profile_align`. The
dominant cost on multi-row sets — graph-cut seam finding — runs at the
same 0.1 MP seam scale in both implementations, so the rows are closely
comparable; only the final compose resolution differs.
³ Chromium, 10-worker wasm thread pool; includes browser-side JPEG decode.
The export figure is the complete flow to a finished GPano-tagged JPEG.

## Robustness: the 33-shot drone sphere

The DJI set contains 8 near-featureless sky shots. OpenCV's stitcher can
only place what it can match:

```
WARNING: only 25/33 images connected: [0, 1, 3, 4, 5, ...]
```

PanoLoom reads the DJI gimbal pose from each file's XMP, fits the
metadata frame against the feature-solved cameras (0.8° median residual
on this set), and places all **33/33** shots — the sky closes instead of
gaping. This is a PanoLoom extension, not an OpenCV defect; it applies
to any rig that records per-shot orientation.

## Caveats, honestly

- The oracle's time includes writing per-stage dump files (its purpose is
  fixtures, not racing); on the small sets that overhead is a large
  fraction, so treat the ring row as indicative only.
- PanoLoom's native numbers use the same work-scale inputs the oracle
  derives internally from the originals, so both do equivalent work per
  stage; PanoLoom's browser numbers additionally include image decode in
  JS.
- The shot-placement comparison is a feature difference, not an OpenCV
  defect: gimbal-metadata rescue is a PanoLoom extension for shots that
  have no usable features at all.
- Quality parity is the design constraint, not a benchmark outcome: many
  stages are bit-exact against OpenCV, and the compose path reaches
  SSIM 1.0000 on the reference sets.
