//! Browser bindings for the PanoLoom engine.
//!
//! Built with `wasm-pack build --target web` (see the root `build:wasm` script).
//! Structured data crosses this boundary as JSON strings; pixel data as
//! `Uint8Array` views into linear memory (copied on entry).

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn engine_version() -> String {
    panoloom_core::VERSION.to_string()
}

/// M0 smoke test: prove pixel buffers survive the JS↔wasm round trip by
/// converting an RGBA image to grayscale with the engine's luma coefficients.
#[wasm_bindgen]
pub fn smoke_grayscale(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>, JsError> {
    let img = panoloom_core::image::RgbaImage::new(width, height, rgba.to_vec())
        .map_err(|e| JsError::new(&e))?;
    Ok(img.to_gray())
}
