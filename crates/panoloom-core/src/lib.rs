//! PanoLoom stitching engine.
//!
//! Pipeline stages (each ported from OpenCV `stitching::detail` and validated
//! against the Python oracle in `tools/reference` — see `docs/pipeline.md`):
//!
//! features → matching → rotation estimation → bundle adjustment →
//! wave correction → warping → photometric compensation → seam finding →
//! multiband blending
//!
//! This crate is wasm-agnostic: it also compiles natively for fast tests and
//! benchmarks. Browser bindings live in `panoloom-wasm`.

pub mod image;
pub mod project;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
