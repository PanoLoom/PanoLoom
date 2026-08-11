//! Browser bindings for the PanoLoom engine.
//!
//! Built with `wasm-pack build --target web` (root `build:wasm` script) and
//! driven from a Web Worker. Pixels cross as `Uint8Array`s (RGBA in, RGBA
//! out); structured results cross as JSON strings.

use panoloom_core::pipeline::{align, render_preview, Alignment, SourceImage};
use panoloom_core::warp::PixelImage;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn engine_version() -> String {
    panoloom_core::VERSION.to_string()
}

/// One stitching project. Feed registration-scale images, align, render.
#[wasm_bindgen]
pub struct Engine {
    sources: Vec<SourceImage>,
    alignment: Option<Alignment>,
}

#[wasm_bindgen]
pub struct PreviewImage {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

#[wasm_bindgen]
impl PreviewImage {
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Moves the pixels out (call once).
    pub fn take_rgba(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.rgba)
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl Engine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Engine {
        Engine {
            sources: Vec::new(),
            alignment: None,
        }
    }

    /// Add a registration-scale image (RGBA8, e.g. canvas readback).
    /// `pose_prior`: optional [yaw, pitch, roll] in degrees from shooting-rig
    /// metadata (DJI gimbal XMP) — rescues feature-poor images.
    pub fn add_image(
        &mut self,
        id: u32,
        rgba: &[u8],
        width: u32,
        height: u32,
        pose_prior: Option<Vec<f64>>,
    ) -> Result<(), JsError> {
        let (w, h) = (width as usize, height as usize);
        if rgba.len() != w * h * 4 {
            return Err(JsError::new("rgba buffer does not match dimensions"));
        }
        if self.sources.iter().any(|s| s.id == id) {
            return Err(JsError::new("duplicate image id"));
        }
        let prior = match pose_prior {
            Some(v) if v.len() == 3 => Some([v[0], v[1], v[2]]),
            Some(_) => return Err(JsError::new("pose_prior must have 3 elements")),
            None => None,
        };
        let mut rgb = Vec::with_capacity(w * h * 3);
        for px in rgba.chunks_exact(4) {
            rgb.extend_from_slice(&px[..3]);
        }
        self.sources.push(SourceImage {
            id,
            rgb: PixelImage::new(w, h, 3, rgb),
            pose_prior: prior,
        });
        self.alignment = None;
        Ok(())
    }

    pub fn remove_image(&mut self, id: u32) {
        self.sources.retain(|s| s.id != id);
        self.alignment = None;
    }

    pub fn clear(&mut self) {
        self.sources.clear();
        self.alignment = None;
    }

    pub fn image_count(&self) -> u32 {
        self.sources.len() as u32
    }

    /// Runs full registration. Returns JSON:
    /// `{"aligned":[...],"rescued":[...],"dropped":[...],"warpedImageScale":f}`.
    pub fn align(&mut self) -> Result<String, JsError> {
        let alignment = align(&self.sources).map_err(|e| JsError::new(&e))?;
        let aligned: Vec<String> = alignment
            .images
            .iter()
            .filter(|a| !a.rescued)
            .map(|a| a.id.to_string())
            .collect();
        let rescued: Vec<String> = alignment
            .images
            .iter()
            .filter(|a| a.rescued)
            .map(|a| a.id.to_string())
            .collect();
        let dropped: Vec<String> = alignment.dropped.iter().map(|d| d.to_string()).collect();
        let json = format!(
            "{{\"aligned\":[{}],\"rescued\":[{}],\"dropped\":[{}],\"warpedImageScale\":{}}}",
            aligned.join(","),
            rescued.join(","),
            dropped.join(","),
            alignment.warped_image_scale
        );
        self.alignment = Some(alignment);
        Ok(json)
    }

    /// Renders the blended preview as a full equirectangular RGBA canvas.
    /// Requires a prior successful `align()`.
    pub fn render_preview(&self, max_width: u32) -> Result<PreviewImage, JsError> {
        let alignment = self
            .alignment
            .as_ref()
            .ok_or_else(|| JsError::new("align() has not succeeded yet"))?;
        let srcs: Vec<&PixelImage> = alignment
            .images
            .iter()
            .map(|ai| {
                &self
                    .sources
                    .iter()
                    .find(|s| s.id == ai.id)
                    .expect("aligned id present")
                    .rgb
            })
            .collect();
        let preview =
            render_preview(&srcs, alignment, max_width as usize).map_err(|e| JsError::new(&e))?;
        Ok(PreviewImage {
            width: preview.width as u32,
            height: preview.height as u32,
            rgba: preview.rgba,
        })
    }
}
