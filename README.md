# PanoLoom

**A modern, free, open-source panorama stitcher that runs entirely in your browser.**

PanoLoom weaves overlapping photos into panoramas — including full 360°×180° spherical
panoramas — with no upload, no install, and no server: all processing happens on your
machine via Rust compiled to WebAssembly, hosted as a static site.

> Status: early development (M0 — toolchain & reference oracle). Not usable yet.

## Why

The industry standard, PTGui, is excellent but expensive and desktop-bound, and every
serious competitor (Microsoft ICE, Autopano Giga) has been discontinued. No browser tool
offers the "pro layer": control points, lens models, projection choice, seam control,
photometric optimization, 360° metadata. PanoLoom aims to fill that gap.

## How it works

- **Engine:** `crates/panoloom-core` — a pure-Rust stitching pipeline (features → matching →
  bundle adjustment → warping → photometric compensation → seam finding → multiband blending),
  ported stage-by-stage from OpenCV's battle-tested `stitching::detail` module (Apache-2.0,
  see `NOTICE`) and validated against it for parity.
- **Oracle:** `tools/reference` — a Python + OpenCV harness that dumps per-stage intermediates;
  every Rust stage must match it on the test suite before we rely on it.
- **App:** `packages/app` — React + TypeScript + Vite, dark pro-studio UI, engine running in a
  Web Worker (SIMD always; threads via wasm-bindgen-rayon when cross-origin isolated).
- **Hosting:** static, on Cloudflare Pages (`public/_headers` provides COOP/COEP).

## Repository layout

```
crates/panoloom-core    pure Rust stitching library (native + wasm)
crates/panoloom-wasm    wasm-bindgen bindings for the browser
packages/shared         project-file types & projection math (TS)
packages/metadata       EXIF reading, GPano XMP injection, encoders (TS)
packages/app            the web app (React + Vite)
tools/reference         OpenCV oracle harness (Python)
tools/testdata          synthetic ground-truth dataset generator (Python)
docs/                   engineering docs (pipeline study, ADRs)
```

## Development

Prerequisites: Rust (via rustup, with `wasm32-unknown-unknown`), Node 20+, pnpm, wasm-pack,
Python 3.11+ (for the oracle/testdata tools).

```sh
pnpm install
pnpm build:wasm      # build the engine to packages/app/src/engine/pkg
pnpm dev             # start the app dev server
cargo test           # native engine tests (fast)
```

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) (OpenCV-derived portions).
