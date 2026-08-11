# PanoLoom

**A modern, free, open-source panorama stitcher that runs entirely in your browser.**

**Live at [panoloom.pages.dev](https://panoloom.pages.dev)** — drop in overlapping photos
(or click *try a sample set*), hit **Align & Preview**, spin the result in a 360° viewer,
and export a full-resolution equirectangular JPEG that Google Photos and any panorama
viewer recognizes as a 360° photo. No upload, no install, no server: everything runs on
your machine via Rust compiled to WebAssembly, hosted as a static site.

## What it does today (v1)

- **360°×180° spherical panoramas** from JPEG/PNG shots (handheld, pano head, drone).
- **Auto alignment**: ORB features → pairwise matching → bundle adjustment → wave
  correction, the same pipeline design as OpenCV's stitcher — ported, not wrapped.
- **Pose-prior rescue**: featureless shots (blank sky) that carry rig metadata (DJI
  gimbal XMP) are placed from their recorded pose instead of being dropped.
- **Production compositing**: photometric gain compensation, graph-cut seams, multiband
  blending — with the panorama treated as a true cylinder, so seams cross the ±180°
  wrap invisibly.
- **Orientation editing**: recenter and level the finished panorama (numeric
  yaw/pitch/roll or "center on current view") with an instant live preview; Apply bakes
  the rotation into the cameras so the export matches exactly.
- **Full-resolution export**: composed in memory-bounded horizontal bands (a 17,000 px
  DJI panorama fits comfortably in wasm's address space), JPEG-encoded in wasm, stamped
  with GPano XMP (Photo Sphere) metadata. Partial panoramas export cropped to their
  actual coverage — the GPano croppedArea fields place the crop on the full sphere, so
  a single-row panorama isn't padded with baked-in black.
- **Projects**: save the alignment as a `.panoproj` file; reopening skips registration
  entirely (the alignment restores bit-for-bit).
- **Threads when available**: on cross-origin-isolated browsers the engine runs on a
  rayon pool over Web Workers (the status bar shows the pool size); otherwise it falls
  back to a single-threaded build automatically.

- **Installable & offline**: the app is a PWA — after the first visit it loads and
  stitches with no connection at all (nothing ever leaves your machine anyway).

Typical numbers (M2 MacBook, Chrome, 10-thread pool): an 8-shot ring stitches in ~1.4 s;
a real 33-shot 12.6 MP DJI sphere aligns + previews in ~56 s and exports its 147 MP
JPEG in ~3.3 min.

## Why

The industry standard, PTGui, is excellent but expensive and desktop-bound, and every
serious competitor (Microsoft ICE, Autopano Giga) has been discontinued. No browser tool
offers the "pro layer": control points, lens models, projection choice, seam control,
photometric optimization, 360° metadata. PanoLoom aims to fill that gap.

## How it's built: the oracle method

The engine is a stage-by-stage **port of OpenCV's `stitching::detail`** (Apache-2.0, see
`NOTICE`) to pure Rust — and every stage is gated on parity with the original:

- `tools/reference` runs the genuine OpenCV pipeline (pinned, OpenCL off) over test sets
  and dumps every intermediate: keypoints, descriptors, matches, cameras before/after
  bundle adjustment, gains, seam masks, warped images, blended output.
- Rust tests replay the same inputs and compare per stage. Many stages are bit-exact
  (gains, graph-cut seams, multiband pyramids); the rest are within documented tolerances
  (libm trig differs by ≤2 ulp across platforms; LAPACK-backed solves differ in the last
  float digits).
- Deliberate deviations from OpenCV (there are a few, e.g. re-validating RANSAC inlier
  masks against the final homography, wrap-aware seam finding, pose-prior rescue) are
  documented at the call site and covered by their own tests.

This is why a from-scratch Rust engine can make production-quality panoramas: quality is
measured against the reference at every step, not eyeballed.

## Repository layout

```
crates/panoloom-core    pure Rust stitching library (native + wasm)
crates/panoloom-wasm    wasm-bindgen bindings for the browser (st + mt builds)
packages/shared         .panoproj types & pose math (TS)
packages/metadata       GPano XMP injection (TS)
packages/app            the web app (React + Vite)
tools/reference         OpenCV oracle harness (Python)
tools/testdata          synthetic ground-truth dataset generator (Python)
docs/                   engineering docs (pipeline study)
```

## Development

Prerequisites: Rust via rustup — stable **and nightly with `rust-src`** (the threaded
wasm build uses `-Z build-std`) — plus the `wasm32-unknown-unknown` target, Node 20+,
pnpm, wasm-pack, and Python 3.11+ if you want to run the oracle/testdata tools.

```sh
pnpm install
pnpm build:wasm       # single-thread engine -> packages/app/src/engine/pkg
pnpm build:wasm-mt    # threaded engine      -> packages/app/src/engine/pkg-mt
pnpm dev              # app dev server (COOP/COEP headers included)

cargo test            # native engine tests (parity suite needs oracle dumps)
pnpm -r test          # TS unit tests (metadata, shared)

# browser end-to-end (needs `pnpm --filter @panoloom/app build` + `vite preview`):
node packages/app/e2e/m5-stitch.mjs     # import -> align -> 360 viewer
node packages/app/e2e/m6-adjust.mjs     # live orientation preview == baked result
node packages/app/e2e/m7-export.mjs     # cropped full-res export + GPano validation
node packages/app/e2e/m8-project.mjs    # project save/load round-trip
node packages/app/e2e/m8-sample.mjs     # bundled sample set
```

Stage-level profiling of the engine on a directory of registration-scale PNGs:

```sh
PANOLOOM_TIMING=1 cargo run --release --example profile_align -- <dir> [priors.json]
```

## Roadmap

v1.x: control-point editor and optimizer variable control → masking & seam steering →
AKAZE/SIFT for low-texture scenes. Later: RAW input, HDR/exposure fusion, WebGPU
compositing, 16-bit TIFF, more projections.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) (OpenCV-derived portions).
The bundled sample set is rendered from a CC0 HDRI by [Poly Haven](https://polyhaven.com).
