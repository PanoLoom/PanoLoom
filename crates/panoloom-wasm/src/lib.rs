//! Browser bindings for the PanoLoom engine.
//!
//! Built with `wasm-pack build --target web` (root `build:wasm` script) and
//! driven from a Web Worker. Pixels cross as `Uint8Array`s (RGBA in, RGBA
//! out); structured results cross as JSON strings.

use panoloom_core::export::Exporter;
use panoloom_core::pipeline::{align, render_preview, Alignment, SourceImage};
use panoloom_core::warp::PixelImage;
use wasm_bindgen::prelude::*;

// mt build: re-export `initThreadPool` (JS calls it once after init with the
// desired worker count; rayon then fans out across Web Workers over SAB).
#[cfg(feature = "mt")]
pub use wasm_bindgen_rayon::init_thread_pool;

#[wasm_bindgen]
pub fn engine_version() -> String {
    panoloom_core::VERSION.to_string()
}

/// Widest panorama this build can compose. Larger sets must export below
/// it; the UI offers it as an option so the ceiling is reachable in one
/// click instead of guessed at.
#[wasm_bindgen]
pub fn max_export_width() -> u32 {
    panoloom_core::export::max_export_width() as u32
}

/// True when this module was built with the rayon thread pool.
#[wasm_bindgen]
pub fn engine_threaded() -> bool {
    cfg!(feature = "mt")
}

/// One stitching project. Feed registration-scale images, align, render.
#[wasm_bindgen]
pub struct Engine {
    sources: Vec<SourceImage>,
    alignment: Option<Alignment>,
    exporter: Option<Exporter>,
    /// Painted seam masks (registration dims, 0 none / 1 exclude / 2
    /// prefer), keyed by image id.
    user_masks: std::collections::HashMap<u32, panoloom_core::imgproc::GrayImage>,
    /// Called with a stage label whenever a long call enters a new stage.
    progress: Option<js_sys::Function>,
}

impl Engine {
    /// Forwards engine stage labels to `self.progress` for as long as the
    /// guard lives. `align` and `preview` run for minutes on large sets, so
    /// without this the UI cannot distinguish slow from hung.
    fn report_progress(&self) -> Option<panoloom_core::progress::Guard> {
        let cb = self.progress.clone()?;
        Some(panoloom_core::progress::scoped(Box::new(
            move |stage: &str| {
                // A throwing or detached callback must not abort the stitch.
                let _ = cb.call1(&JsValue::NULL, &JsValue::from_str(stage));
            },
        )))
    }

    /// User masks ordered like `alignment.images` (None where unset).
    fn ordered_masks(
        &self,
        alignment: &Alignment,
    ) -> Vec<Option<&panoloom_core::imgproc::GrayImage>> {
        alignment
            .images
            .iter()
            .map(|ai| self.user_masks.get(&ai.id))
            .collect()
    }
}

/// Finished export: JPEG bytes (the coverage crop) + where that crop sits
/// on the full sphere (GPano croppedArea semantics).
#[wasm_bindgen]
pub struct ExportResult {
    width: u32,
    height: u32,
    full_width: u32,
    full_height: u32,
    left: u32,
    top: u32,
    jpeg: Vec<u8>,
}

#[wasm_bindgen]
impl ExportResult {
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[wasm_bindgen(getter)]
    pub fn full_width(&self) -> u32 {
        self.full_width
    }

    #[wasm_bindgen(getter)]
    pub fn full_height(&self) -> u32 {
        self.full_height
    }

    #[wasm_bindgen(getter)]
    pub fn left(&self) -> u32 {
        self.left
    }

    #[wasm_bindgen(getter)]
    pub fn top(&self) -> u32 {
        self.top
    }

    /// Moves the bytes out (call once).
    pub fn take_jpeg(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.jpeg)
    }
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
            exporter: None,
            user_masks: std::collections::HashMap::new(),
            progress: None,
        }
    }

    /// Starts a full-resolution export. `ids/widths/heights` describe the
    /// ORIGINAL dimensions of every aligned image. Returns the band plan:
    /// `{"width":..,"height":..,"bands":[{"y0":..,"y1":..,"needed":[ids]}]}`.
    pub fn begin_export(
        &mut self,
        target_width: u32,
        ids: Vec<u32>,
        widths: Vec<u32>,
        heights: Vec<u32>,
    ) -> Result<String, JsError> {
        let alignment = self
            .alignment
            .as_ref()
            .ok_or_else(|| JsError::new("align() has not succeeded yet"))?;
        if ids.len() != widths.len() || ids.len() != heights.len() {
            return Err(JsError::new("ids/widths/heights length mismatch"));
        }
        let full_sizes: Vec<(u32, u32, u32)> = ids
            .iter()
            .zip(&widths)
            .zip(&heights)
            .map(|((&i, &w), &h)| (i, w, h))
            .collect();
        let masks = self.ordered_masks(alignment);
        let exporter = Exporter::new(
            &self.sources,
            alignment,
            &masks,
            &full_sizes,
            target_width as usize,
        )
        .map_err(|e| JsError::new(&e))?;

        let (fw, fh) = exporter.canvas_size();
        let (cx, cy, cw, ch) = exporter.crop();
        let bands: Vec<String> = exporter
            .bands()
            .iter()
            .map(|b| {
                let needed: Vec<String> = b.needed.iter().map(|i| i.to_string()).collect();
                format!(
                    "{{\"y0\":{},\"y1\":{},\"needed\":[{}]}}",
                    b.y0,
                    b.y1,
                    needed.join(",")
                )
            })
            .collect();
        let plan = format!(
            "{{\"width\":{cw},\"height\":{ch},\"left\":{cx},\"top\":{cy},\"fullWidth\":{fw},\"fullHeight\":{fh},\"bands\":[{}]}}",
            bands.join(",")
        );
        self.exporter = Some(exporter);
        Ok(plan)
    }

    /// Provides a FULL-RESOLUTION source image (RGBA8) for the export.
    pub fn export_set_image(
        &mut self,
        id: u32,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(), JsError> {
        let exporter = self
            .exporter
            .as_mut()
            .ok_or_else(|| JsError::new("no export in progress"))?;
        let (w, h) = (width as usize, height as usize);
        if rgba.len() != w * h * 4 {
            return Err(JsError::new("rgba buffer does not match dimensions"));
        }
        let mut rgb = Vec::with_capacity(w * h * 3);
        for px in rgba.as_chunks::<4>().0 {
            rgb.extend_from_slice(&px[..3]);
        }
        exporter
            .set_full_image(id, PixelImage::new(w, h, 3, rgb))
            .map_err(|e| JsError::new(&e))
    }

    pub fn export_drop_image(&mut self, id: u32) {
        if let Some(e) = self.exporter.as_mut() {
            e.drop_full_image(id);
        }
    }

    /// Composites one band (see the plan from `begin_export`).
    pub fn export_band(&mut self, band: u32) -> Result<(), JsError> {
        let exporter = self
            .exporter
            .as_mut()
            .ok_or_else(|| JsError::new("no export in progress"))?;
        exporter
            .composite_band(band as usize)
            .map_err(|e| JsError::new(&e))
    }

    /// Encodes and returns the finished panorama; ends the export session.
    pub fn finish_export(&mut self, quality: u8) -> Result<ExportResult, JsError> {
        let exporter = self
            .exporter
            .take()
            .ok_or_else(|| JsError::new("no export in progress"))?;
        let (fw, fh) = exporter.canvas_size();
        let (cx, cy, _, _) = exporter.crop();
        let (jpeg, w, h) = exporter.finish(quality).map_err(|e| JsError::new(&e))?;
        Ok(ExportResult {
            width: w as u32,
            height: h as u32,
            full_width: fw as u32,
            full_height: fh as u32,
            left: cx as u32,
            top: cy as u32,
            jpeg,
        })
    }

    pub fn cancel_export(&mut self) {
        self.exporter = None;
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
        for px in rgba.as_chunks::<4>().0 {
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
        self.user_masks.remove(&id);
        self.alignment = None;
        self.exporter = None;
    }

    pub fn clear(&mut self) {
        self.sources.clear();
        self.alignment = None;
        self.exporter = None;
    }

    pub fn image_count(&self) -> u32 {
        self.sources.len() as u32
    }

    /// Runs full registration. Returns JSON:
    /// `{"aligned":[...],"rescued":[...],"dropped":[...],"warpedImageScale":f}`.
    /// Installs a `(stage: string) => void` callback invoked as `align` and
    /// `preview` move between stages. Pass `None` to clear it.
    pub fn set_progress_callback(&mut self, cb: Option<js_sys::Function>) {
        self.progress = cb;
    }

    pub fn align(&mut self) -> Result<String, JsError> {
        let _progress = self.report_progress();
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

    /// Sets a painted seam mask for an image (bytes at REGISTRATION dims:
    /// 0 = none, 1 = exclude, 2 = prefer). Invalidates a running export.
    pub fn set_mask(
        &mut self,
        id: u32,
        mask: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(), JsError> {
        let src = self
            .sources
            .iter()
            .find(|s| s.id == id)
            .ok_or_else(|| JsError::new("unknown image id"))?;
        if (width as usize, height as usize) != (src.rgb.width, src.rgb.height) {
            return Err(JsError::new("mask dimensions must match the image"));
        }
        if mask.len() != (width * height) as usize {
            return Err(JsError::new("mask buffer does not match dimensions"));
        }
        if mask.iter().all(|&v| v == 0) {
            self.user_masks.remove(&id);
        } else {
            self.user_masks.insert(
                id,
                panoloom_core::imgproc::GrayImage::new(
                    width as usize,
                    height as usize,
                    mask.to_vec(),
                ),
            );
        }
        self.exporter = None;
        Ok(())
    }

    pub fn clear_mask(&mut self, id: u32) {
        self.user_masks.remove(&id);
        self.exporter = None;
    }

    /// Number of images with painted masks (diagnostics).
    pub fn mask_count(&self) -> u32 {
        self.user_masks.len() as u32
    }

    /// Rotates the whole panorama by a pano-frame rotation (row-major 3x3).
    /// Content at direction d moves to r·d. Invalidates any running export.
    pub fn orient(&mut self, r: Vec<f64>) -> Result<(), JsError> {
        if r.len() != 9 {
            return Err(JsError::new("rotation must have 9 elements"));
        }
        let alignment = self
            .alignment
            .as_mut()
            .ok_or_else(|| JsError::new("align() has not succeeded yet"))?;
        let r_g = [[r[0], r[1], r[2]], [r[3], r[4], r[5]], [r[6], r[7], r[8]]];
        panoloom_core::pipeline::orient_alignment(alignment, &r_g);
        self.exporter = None;
        Ok(())
    }

    /// Feature-match derived control points (registration-scale coords),
    /// as a JSON array. Recomputes features/matches on the current images.
    pub fn auto_control_points(&self, max_per_pair: u32) -> String {
        let cps = panoloom_core::cp::auto_control_points(&self.sources, max_per_pair as usize);
        serde_json::to_string(&cps).unwrap_or_else(|_| "[]".into())
    }

    /// Optimizes the alignment against control points (JSON array, coords
    /// at registration scale) with PTGui-style variable flags. Returns the
    /// report JSON (rms before/after, per-CP errors, fitted lens).
    pub fn optimize_cps(&mut self, cps_json: &str, flags_json: &str) -> Result<String, JsError> {
        let alignment = self
            .alignment
            .as_mut()
            .ok_or_else(|| JsError::new("align() has not succeeded yet"))?;
        let cps: Vec<panoloom_core::cp::ControlPoint> = serde_json::from_str(cps_json)
            .map_err(|e| JsError::new(&format!("bad control points: {e}")))?;
        let flags: panoloom_core::optimizer::OptimizeFlags = serde_json::from_str(flags_json)
            .map_err(|e| JsError::new(&format!("bad flags: {e}")))?;
        let dims: std::collections::HashMap<u32, (u32, u32)> = self
            .sources
            .iter()
            .map(|s| (s.id, (s.rgb.width as u32, s.rgb.height as u32)))
            .collect();
        let report =
            panoloom_core::optimizer::optimize_control_points(alignment, &cps, &dims, &flags)
                .map_err(|e| JsError::new(&e))?;
        self.exporter = None;
        serde_json::to_string(&report).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Serializes the current alignment (exact float round-trip) for
    /// project save. Requires a prior successful `align()`.
    pub fn export_alignment(&self) -> Result<String, JsError> {
        let alignment = self
            .alignment
            .as_ref()
            .ok_or_else(|| JsError::new("align() has not succeeded yet"))?;
        serde_json::to_string(alignment).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Restores an alignment saved by `export_alignment`. Every aligned id
    /// must already be loaded via `add_image` (at the same work scale the
    /// project was saved with). Returns the same JSON shape as `align()`.
    pub fn import_alignment(&mut self, json: &str) -> Result<String, JsError> {
        let alignment: Alignment =
            serde_json::from_str(json).map_err(|e| JsError::new(&format!("bad project: {e}")))?;
        for ai in &alignment.images {
            if !self.sources.iter().any(|s| s.id == ai.id) {
                return Err(JsError::new(&format!(
                    "project image id {} has not been loaded",
                    ai.id
                )));
            }
        }
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
        let _progress = self.report_progress();
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
        let preview = render_preview(
            &srcs,
            alignment,
            &self.ordered_masks(alignment),
            max_width as usize,
        )
        .map_err(|e| JsError::new(&e))?;
        Ok(PreviewImage {
            width: preview.width as u32,
            height: preview.height as u32,
            rgba: preview.rgba,
        })
    }
}
