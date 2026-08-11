//! High-level stitching pipeline: the staged engine the app drives.
//!
//! Mirrors cv::Stitcher PANORAMA-mode orchestration (docs/pipeline.md §0)
//! over the ported stages. Inputs arrive as registration-scale RGB images
//! (the browser decodes + downscales); seam-scale copies are derived here.

#![allow(clippy::needless_range_loop)]

use crate::blend::{num_bands_for, result_roi, MultiBandBlender};
use crate::bundle::bundle_adjust_ray;
use crate::camera::CameraParams;
use crate::estimation::{
    homography_based_estimate, leave_biggest_component, warped_image_scale, wave_correct_horiz,
    FeatureSet, MatchGraph,
};
use crate::exposure::{BlocksGainCompensator, RgbImage};
use crate::imgproc::{rgb_to_gray_cv, GrayImage};
use crate::matcher::match_pair;
use crate::orb::{orb_detect_and_compute, OrbParams};
use crate::seam::find_seams_graph_cut_color;
use crate::warp::{Border, Interp, PixelImage, SphericalWarper};

/// Seam-estimation area relative to the registration area. The stitcher
/// uses absolute megapixels (0.6 / 0.1); inputs here are already at
/// registration scale, so the seam scale is the ratio sqrt(0.1/0.6).
const SEAM_FROM_WORK_SCALE: f64 = 0.408_248_290_463_863; // sqrt(1/6)

pub struct SourceImage {
    pub id: u32,
    /// Registration-scale RGB pixels.
    pub rgb: PixelImage,
}

pub struct AlignedImage {
    pub id: u32,
    pub camera: CameraParams,
}

pub struct Alignment {
    pub images: Vec<AlignedImage>,
    /// ids that could not be matched into the panorama.
    pub dropped: Vec<u32>,
    pub warped_image_scale: f64,
}

/// Full registration: features → matching → biggest component →
/// estimation → bundle adjustment → wave correction.
pub fn align(sources: &[SourceImage]) -> Result<Alignment, String> {
    if sources.len() < 2 {
        return Err("need at least two images".into());
    }
    let n = sources.len();

    let mut pts: Vec<Vec<[f32; 2]>> = Vec::with_capacity(n);
    let mut descs = Vec::with_capacity(n);
    let mut sizes = Vec::with_capacity(n);
    for s in sources {
        let gray = rgb_to_gray_cv(&s.rgb.data, s.rgb.width, s.rgb.height);
        let (kps, d) = orb_detect_and_compute(&gray, &OrbParams::default());
        pts.push(kps.iter().map(|k| [k.x, k.y]).collect());
        descs.push(d);
        sizes.push((s.rgb.width as u32, s.rgb.height as u32));
    }

    let mut upper = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            upper.push((
                (i, j),
                match_pair(&pts[i], &descs[i], sizes[i], &pts[j], &descs[j], sizes[j]),
            ));
        }
    }
    let graph = MatchGraph::from_upper_triangle(n, upper);

    let kept = leave_biggest_component(&graph, 1.0);
    if kept.len() < 2 {
        return Err("images do not overlap enough to align".into());
    }
    let dropped: Vec<u32> = (0..n)
        .filter(|i| !kept.contains(i))
        .map(|i| sources[i].id)
        .collect();

    // Subset when needed (rebuild a dense graph over the kept indices).
    let (features, graph, kept): (Vec<FeatureSet>, MatchGraph, Vec<usize>) = if kept.len() == n {
        let features = (0..n)
            .map(|i| FeatureSet {
                width: sizes[i].0,
                height: sizes[i].1,
                keypoints: pts[i].clone(),
            })
            .collect();
        (features, graph, kept)
    } else {
        let m = kept.len();
        let features = kept
            .iter()
            .map(|&i| FeatureSet {
                width: sizes[i].0,
                height: sizes[i].1,
                keypoints: pts[i].clone(),
            })
            .collect();
        let mut sub_upper = Vec::new();
        for a in 0..m {
            for b in (a + 1)..m {
                sub_upper.push(((a, b), graph.at(kept[a], kept[b]).clone()));
            }
        }
        (
            features,
            MatchGraph::from_upper_triangle(m, sub_upper),
            kept,
        )
    };

    let mut cameras = homography_based_estimate(&features, &graph);
    if !bundle_adjust_ray(&features, &graph, &mut cameras) {
        return Err("bundle adjustment failed".into());
    }
    let mut rmats: Vec<[[f32; 3]; 3]> = cameras.iter().map(|c| c.r).collect();
    wave_correct_horiz(&mut rmats);
    for (c, r) in cameras.iter_mut().zip(&rmats) {
        c.r = *r;
    }

    let scale = warped_image_scale(&cameras);
    Ok(Alignment {
        images: kept
            .iter()
            .zip(cameras)
            .map(|(&i, camera)| AlignedImage {
                id: sources[i].id,
                camera,
            })
            .collect(),
        dropped,
        warped_image_scale: scale,
    })
}

pub struct Preview {
    /// Full 360x180 equirectangular RGBA canvas (uncovered pixels alpha 0).
    pub rgba: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

/// Renders a blended preview onto a FULL equirectangular canvas so a 360°
/// viewer can consume it directly. `sources` must correspond 1:1 with
/// `alignment.images` (already subset by the caller via ids).
pub fn render_preview(
    sources: &[&PixelImage],
    alignment: &Alignment,
    max_width: usize,
) -> Result<Preview, String> {
    let n = alignment.images.len();
    if sources.len() != n {
        return Err("sources/alignment mismatch".into());
    }

    // Choose the warp scale so the full 360° canvas is <= max_width.
    let swa = SEAM_FROM_WORK_SCALE;
    let natural_scale = alignment.warped_image_scale * swa;
    let full_width_at = |s: f64| (2.0 * std::f64::consts::PI * s).ceil() as usize;
    let scale = if full_width_at(natural_scale) > max_width {
        max_width as f64 / (2.0 * std::f64::consts::PI)
    } else {
        natural_scale
    };
    let img_scale = scale / alignment.warped_image_scale; // K multiplier

    let mut warper = SphericalWarper::new(scale as f32);
    let mut corners = Vec::new();
    let mut sizes = Vec::new();
    let mut warped_imgs: Vec<PixelImage> = Vec::new();
    let mut warped_masks: Vec<GrayImage> = Vec::new();

    for (src, ai) in sources.iter().zip(&alignment.images) {
        // Downscale the registration-scale source to the warp working size.
        let (sw, sh) = (
            ((src.width as f64) * img_scale).round().max(2.0) as usize,
            ((src.height as f64) * img_scale).round().max(2.0) as usize,
        );
        let gray_scaled_rgb = resize_rgb(src, sw, sh);

        let c = &ai.camera;
        let k = [
            [
                (c.focal * img_scale) as f32,
                0.0,
                (c.ppx * img_scale) as f32,
            ],
            [
                0.0,
                (c.focal * img_scale) as f32,
                (c.ppy * img_scale) as f32,
            ],
            [0.0, 0.0, 1.0],
        ];
        let (tl, w_img) = warper.warp(&gray_scaled_rgb, &k, &c.r, Interp::Linear, Border::Reflect);
        let mask_src = PixelImage::new(sw, sh, 1, vec![255u8; sw * sh]);
        let (_, w_mask) = warper.warp(&mask_src, &k, &c.r, Interp::Nearest, Border::Constant0);
        corners.push(tl);
        sizes.push((w_img.width as i32, w_img.height as i32));
        warped_masks.push(GrayImage::new(w_mask.width, w_mask.height, w_mask.data));
        warped_imgs.push(w_img);
    }

    // Gain compensation + seams at this scale.
    let rgb_for_feed: Vec<RgbImage> = warped_imgs
        .iter()
        .map(|w| RgbImage::new(w.width, w.height, w.data.clone()))
        .collect();
    let compensator = BlocksGainCompensator::feed(&corners, &rgb_for_feed, &warped_masks);
    let mut seam_masks: Vec<GrayImage> = warped_masks.clone();
    find_seams_graph_cut_color(&warped_imgs, &corners, &mut seam_masks);

    let roi = result_roi(&corners, &sizes);
    let bands = num_bands_for(roi.2, roi.3);
    let mut blender = MultiBandBlender::new(bands);
    blender.prepare(roi.0, roi.1, roi.2, roi.3);
    for i in 0..n {
        let mut rgb = rgb_for_feed[i].clone();
        compensator.apply(i, &mut rgb);
        // Dilate seam mask (3x3), AND with coverage (compose-loop semantics).
        let dilated = dilate3(&seam_masks[i]);
        let mut final_mask = vec![0u8; dilated.data.len()];
        for p in 0..final_mask.len() {
            final_mask[p] = dilated.data[p] & warped_masks[i].data[p];
        }
        blender.feed(
            &rgb.data,
            rgb.width,
            rgb.height,
            &GrayImage::new(dilated.width, dilated.height, final_mask),
            corners[i],
        );
    }
    let (blended, coverage) = blender.blend();

    // Paste onto the full equirect canvas. In warp coordinates the full
    // sphere spans u in [-pi*scale, pi*scale), v in [0, pi*scale].
    let canvas_w = full_width_at(scale);
    let canvas_h = (std::f64::consts::PI * scale).ceil() as usize;
    let mut rgba = vec![0u8; canvas_w * canvas_h * 4];
    let off_x = (-std::f64::consts::PI * scale) as i32;
    for y in 0..roi.3 {
        let cy = roi.1 + y as i32; // canvas v origin coincides with warp v=0
        if cy < 0 || cy >= canvas_h as i32 {
            continue;
        }
        for x in 0..roi.2 {
            let mut cx = roi.0 - off_x + x as i32;
            // Wrap horizontally.
            let w = canvas_w as i32;
            cx = ((cx % w) + w) % w;
            if coverage.data[y * roi.2 + x] == 0 {
                continue;
            }
            let src = (y * roi.2 + x) * 3;
            let dst = (cy as usize * canvas_w + cx as usize) * 4;
            rgba[dst] = blended[src];
            rgba[dst + 1] = blended[src + 1];
            rgba[dst + 2] = blended[src + 2];
            rgba[dst + 3] = 255;
        }
    }

    Ok(Preview {
        rgba,
        width: canvas_w,
        height: canvas_h,
    })
}

/// Plain bilinear RGB resize (browser-quality; not a parity surface).
fn resize_rgb(src: &PixelImage, dst_w: usize, dst_h: usize) -> PixelImage {
    assert_eq!(src.channels, 3);
    let sx = src.width as f64 / dst_w as f64;
    let sy = src.height as f64 / dst_h as f64;
    let mut data = vec![0u8; dst_w * dst_h * 3];
    for dy in 0..dst_h {
        let fy = ((dy as f64 + 0.5) * sy - 0.5).clamp(0.0, src.height as f64 - 1.0);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(src.height - 1);
        let wy = (fy - y0 as f64) as f32;
        for dx in 0..dst_w {
            let fx = ((dx as f64 + 0.5) * sx - 0.5).clamp(0.0, src.width as f64 - 1.0);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(src.width - 1);
            let wx = (fx - x0 as f64) as f32;
            for c in 0..3 {
                let p00 = src.data[(y0 * src.width + x0) * 3 + c] as f32;
                let p01 = src.data[(y0 * src.width + x1) * 3 + c] as f32;
                let p10 = src.data[(y1 * src.width + x0) * 3 + c] as f32;
                let p11 = src.data[(y1 * src.width + x1) * 3 + c] as f32;
                let top = p00 + (p01 - p00) * wx;
                let bot = p10 + (p11 - p10) * wx;
                data[(dy * dst_w + dx) * 3 + c] =
                    (top + (bot - top) * wy).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    PixelImage::new(dst_w, dst_h, 3, data)
}

/// 3x3 rect dilation (cv2.dilate(mask, None), one iteration).
fn dilate3(mask: &GrayImage) -> GrayImage {
    let (w, h) = (mask.width, mask.height);
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut v = 0u8;
            for dy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                for dx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                    v = v.max(mask.data[dy * w + dx]);
                }
            }
            out[y * w + x] = v;
        }
    }
    GrayImage::new(w, h, out)
}
