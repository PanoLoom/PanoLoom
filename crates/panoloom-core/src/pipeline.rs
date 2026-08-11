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
    /// Optional shooting-rig pose prior (yaw, pitch, roll in degrees, e.g.
    /// DJI `drone-dji:Gimbal*Degree` XMP). Used to RESCUE images that have
    /// too few features to match (blank sky): after the feature-based
    /// solve, unmatched images with a prior are placed via the best-fit
    /// rotation between the prior frame and the solved frame.
    pub pose_prior: Option<[f64; 3]>,
}

pub struct AlignedImage {
    pub id: u32,
    pub camera: CameraParams,
    /// True when placed from a pose prior rather than feature matches.
    pub rescued: bool,
}

pub struct Alignment {
    pub images: Vec<AlignedImage>,
    /// ids that could not be matched into the panorama (and had no prior).
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
    let mut images: Vec<AlignedImage> = kept
        .iter()
        .zip(cameras)
        .map(|(&i, camera)| AlignedImage {
            id: sources[i].id,
            camera,
            rescued: false,
        })
        .collect();

    // Pose-prior rescue: place unmatched images that carry a shooting-rig
    // pose (blank-sky shots in DJI sphere sets, etc.).
    let offset = fit_prior_offset(sources, &images);
    let mut still_dropped = Vec::new();
    let rescuable: Vec<&SourceImage> = sources.iter().filter(|s| dropped.contains(&s.id)).collect();
    if !rescuable.is_empty() {
        for s in rescuable {
            match (s.pose_prior, offset) {
                (Some(prior), Some(off)) => {
                    let r = mat3_mul_f64(&off, &prior_rotation(prior));
                    let mut cam = CameraParams {
                        focal: scale,
                        ..Default::default()
                    };
                    cam.ppx = 0.5 * s.rgb.width as f64;
                    cam.ppy = 0.5 * s.rgb.height as f64;
                    for a in 0..3 {
                        for b in 0..3 {
                            cam.r[a][b] = r[a][b] as f32;
                        }
                    }
                    images.push(AlignedImage {
                        id: s.id,
                        camera: cam,
                        rescued: true,
                    });
                }
                _ => still_dropped.push(s.id),
            }
        }
    }

    // Orientation fix: waveCorrect has a global 180° ambiguity (panos can
    // come out upside down). When pose priors exist, they define earth-up
    // authoritatively: if the fitted offset maps earth-up to pano-down,
    // roll the whole panorama 180° (Rz(pi), a pure rotation).
    if let Some(off) = offset {
        // Earth up in the prior frame is -y (y points down); mapped up's y
        // component > 0 means it points DOWN in the pano frame.
        let mapped_up_y = -off[1][1];
        if mapped_up_y > 0.0 {
            for ai in images.iter_mut() {
                let mut r = [[0.0f32; 3]; 3];
                for b in 0..3 {
                    // Rz(pi) · R: negate the first two rows.
                    r[0][b] = -ai.camera.r[0][b];
                    r[1][b] = -ai.camera.r[1][b];
                    r[2][b] = ai.camera.r[2][b];
                }
                ai.camera.r = r;
            }
        }
    }

    Ok(Alignment {
        images,
        dropped: still_dropped,
        warped_image_scale: scale,
    })
}

type Mat3d = [[f64; 3]; 3];

fn mat3_mul_f64(a: &Mat3d, b: &Mat3d) -> Mat3d {
    let mut o = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            for k in 0..3 {
                o[r][c] += a[r][k] * b[k][c];
            }
        }
    }
    o
}

/// Rig pose (yaw, pitch, roll degrees, earth-referenced like DJI gimbal
/// values) to a pano<-camera rotation in our convention (y down, +pitch up):
/// R = Ry(yaw) · Rx(-pitch) · Rz(roll).
fn prior_rotation(p: [f64; 3]) -> Mat3d {
    let (y, pi, r) = (p[0].to_radians(), (-p[1]).to_radians(), p[2].to_radians());
    let ry = [
        [y.cos(), 0.0, y.sin()],
        [0.0, 1.0, 0.0],
        [-y.sin(), 0.0, y.cos()],
    ];
    let rx = [
        [1.0, 0.0, 0.0],
        [0.0, pi.cos(), -pi.sin()],
        [0.0, pi.sin(), pi.cos()],
    ];
    let rz = [
        [r.cos(), -r.sin(), 0.0],
        [r.sin(), r.cos(), 0.0],
        [0.0, 0.0, 1.0],
    ];
    mat3_mul_f64(&mat3_mul_f64(&ry, &rx), &rz)
}

/// Wahba/Kabsch fit of the single rotation `off` minimizing
/// Σ angle(off · R_prior_i, R_solved_i) over the feature-solved cameras
/// that carry priors. Returns None when fewer than 2 anchors exist or the
/// fit is poor (median residual > 8°, i.e. the priors are untrustworthy).
fn fit_prior_offset(sources: &[SourceImage], solved: &[AlignedImage]) -> Option<Mat3d> {
    use nalgebra::Matrix3;

    let mut m = Matrix3::<f64>::zeros();
    let mut anchors = Vec::new();
    for ai in solved {
        let src = sources.iter().find(|s| s.id == ai.id)?;
        let Some(prior) = src.pose_prior else {
            continue;
        };
        let rp = prior_rotation(prior);
        let mut rs = [[0.0f64; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                rs[a][b] = ai.camera.r[a][b] as f64;
            }
        }
        // M += R_solved · R_priorᵀ
        for a in 0..3 {
            for b in 0..3 {
                let mut acc = 0.0;
                for k in 0..3 {
                    acc += rs[a][k] * rp[b][k];
                }
                m[(a, b)] += acc;
            }
        }
        anchors.push((rp, rs));
    }
    if anchors.len() < 2 {
        return None;
    }

    let svd = m.svd(true, true);
    let (u, v_t) = (svd.u?, svd.v_t?);
    let d = (u * v_t).determinant();
    let correction = Matrix3::from_diagonal(&nalgebra::Vector3::new(1.0, 1.0, d.signum()));
    let off_m = u * correction * v_t;
    let mut off = [[0.0f64; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            off[a][b] = off_m[(a, b)];
        }
    }

    // Residual check: reject unusable priors.
    let mut residuals: Vec<f64> = anchors
        .iter()
        .map(|(rp, rs)| {
            let pred = mat3_mul_f64(&off, rp);
            let mut tr = 0.0;
            for a in 0..3 {
                for k in 0..3 {
                    tr += pred[a][k] * rs[a][k];
                }
            }
            (((tr - 1.0) / 2.0).clamp(-1.0, 1.0)).acos().to_degrees()
        })
        .collect();
    residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = residuals[residuals.len() / 2];
    if median > 8.0 {
        return None;
    }
    Some(off)
}

/// Diagnostic helper: median Wahba residual of a prior set against solved
/// cameras, ignoring the acceptance threshold. For convention probing.
pub fn debug_prior_fit_residual(sources: &[SourceImage], solved: &[AlignedImage]) -> Option<f64> {
    use nalgebra::Matrix3;
    let mut m = Matrix3::<f64>::zeros();
    let mut anchors = Vec::new();
    for ai in solved.iter().filter(|a| !a.rescued) {
        let src = sources.iter().find(|s| s.id == ai.id)?;
        let Some(prior) = src.pose_prior else {
            continue;
        };
        let rp = prior_rotation(prior);
        let mut rs = [[0.0f64; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                rs[a][b] = ai.camera.r[a][b] as f64;
            }
        }
        for a in 0..3 {
            for b in 0..3 {
                let mut acc = 0.0;
                for k in 0..3 {
                    acc += rs[a][k] * rp[b][k];
                }
                m[(a, b)] += acc;
            }
        }
        anchors.push((rp, rs));
    }
    if anchors.len() < 2 {
        return None;
    }
    let svd = m.svd(true, true);
    let (u, v_t) = (svd.u?, svd.v_t?);
    let d = (u * v_t).determinant();
    let corr = Matrix3::from_diagonal(&nalgebra::Vector3::new(1.0, 1.0, d.signum()));
    let off_m = u * corr * v_t;
    let mut off = [[0.0f64; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            off[a][b] = off_m[(a, b)];
        }
    }
    let mut residuals: Vec<f64> = anchors
        .iter()
        .map(|(rp, rs)| {
            let pred = mat3_mul_f64(&off, rp);
            let mut tr = 0.0;
            for a in 0..3 {
                for k in 0..3 {
                    tr += pred[a][k] * rs[a][k];
                }
            }
            (((tr - 1.0) / 2.0).clamp(-1.0, 1.0)).acos().to_degrees()
        })
        .collect();
    residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(residuals[residuals.len() / 2])
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

    // Two-stage compose, mirroring the stitcher: gains + graph-cut seams at
    // SEAM scale (cheap), composite at REGISTRATION scale (sharp), capped
    // so the full 360° canvas stays <= max_width.
    let full_width_at = |s: f64| (2.0 * std::f64::consts::PI * s).ceil() as usize;
    let seam_scale = alignment.warped_image_scale * SEAM_FROM_WORK_SCALE;
    let compose_scale = if full_width_at(alignment.warped_image_scale) > max_width {
        max_width as f64 / (2.0 * std::f64::consts::PI)
    } else {
        alignment.warped_image_scale
    };

    let k_for = |c: &CameraParams, m: f64| -> [[f32; 3]; 3] {
        [
            [(c.focal * m) as f32, 0.0, (c.ppx * m) as f32],
            [0.0, (c.focal * m) as f32, (c.ppy * m) as f32],
            [0.0, 0.0, 1.0],
        ]
    };

    // --- stage 1: seam scale — warp, feed gains, find seams ---
    let seam_mul = seam_scale / alignment.warped_image_scale;
    let mut seam_warper = SphericalWarper::new(seam_scale as f32);
    let mut s_corners = Vec::new();
    let mut s_imgs: Vec<PixelImage> = Vec::new();
    let mut s_masks: Vec<GrayImage> = Vec::new();
    for (src, ai) in sources.iter().zip(&alignment.images) {
        let (sw, sh) = (
            ((src.width as f64) * seam_mul).round().max(2.0) as usize,
            ((src.height as f64) * seam_mul).round().max(2.0) as usize,
        );
        let small = resize_rgb(src, sw, sh);
        let k = k_for(&ai.camera, seam_mul);
        let (tl, w_img) =
            seam_warper.warp(&small, &k, &ai.camera.r, Interp::Linear, Border::Reflect);
        let mask_src = PixelImage::new(sw, sh, 1, vec![255u8; sw * sh]);
        let (_, w_mask) = seam_warper.warp(
            &mask_src,
            &k,
            &ai.camera.r,
            Interp::Nearest,
            Border::Constant0,
        );
        s_corners.push(tl);
        s_masks.push(GrayImage::new(w_mask.width, w_mask.height, w_mask.data));
        s_imgs.push(w_img);
    }
    // Rescued shots are placed from pose metadata (~1° accuracy) — good
    // enough to fill featureless holes, not good enough to overlap matched
    // shots (a 1° offset ghosts hard edges like ridge lines). Suppress
    // their masks wherever matched coverage exists, keeping a small eroded
    // overlap band for blending.
    suppress_rescued_masks(&mut s_masks, &s_corners, alignment, 2);

    let s_rgb: Vec<RgbImage> = s_imgs
        .iter()
        .map(|w| RgbImage::new(w.width, w.height, w.data.clone()))
        .collect();
    let compensator = BlocksGainCompensator::feed(&s_corners, &s_rgb, &s_masks);
    let mut seam_masks: Vec<GrayImage> = s_masks.clone();
    find_seams_graph_cut_color(&s_imgs, &s_corners, &mut seam_masks);

    // --- stage 2: compose scale — warp sharp, apply gains, blend ---
    let compose_mul = compose_scale / alignment.warped_image_scale;
    let mut warper = SphericalWarper::new(compose_scale as f32);
    let mut corners = Vec::new();
    let mut sizes = Vec::new();
    let mut fed: Vec<(RgbImage, GrayImage)> = Vec::new();
    for (i, (src, ai)) in sources.iter().zip(&alignment.images).enumerate() {
        let (sw, sh) = (
            ((src.width as f64) * compose_mul).round().max(2.0) as usize,
            ((src.height as f64) * compose_mul).round().max(2.0) as usize,
        );
        let scaled = if (compose_mul - 1.0).abs() < 1e-9 {
            (*src).clone()
        } else {
            resize_rgb(src, sw, sh)
        };
        let k = k_for(&ai.camera, compose_mul);
        let (tl, w_img) = warper.warp(&scaled, &k, &ai.camera.r, Interp::Linear, Border::Reflect);
        let mask_src = PixelImage::new(
            scaled.width,
            scaled.height,
            1,
            vec![255u8; scaled.width * scaled.height],
        );
        let (_, w_mask) = warper.warp(
            &mask_src,
            &k,
            &ai.camera.r,
            Interp::Nearest,
            Border::Constant0,
        );

        let mut rgb = RgbImage::new(w_img.width, w_img.height, w_img.data);
        compensator.apply(i, &mut rgb);

        // Upscale dilated seam masks to compose size, AND with coverage.
        let dilated = dilate3(&seam_masks[i]);
        let up = crate::imgproc::resize_bilinear(&dilated, w_mask.width, w_mask.height);
        let mut final_mask = vec![0u8; w_mask.width * w_mask.height];
        for p in 0..final_mask.len() {
            final_mask[p] = up.data[p] & w_mask.data[p];
        }
        corners.push(tl);
        sizes.push((rgb.width as i32, rgb.height as i32));
        fed.push((rgb, GrayImage::new(w_mask.width, w_mask.height, final_mask)));
    }

    let roi = result_roi(&corners, &sizes);
    let bands = num_bands_for(roi.2, roi.3);
    let mut blender = MultiBandBlender::new(bands);
    blender.prepare(roi.0, roi.1, roi.2, roi.3);
    for (i, (rgb, mask)) in fed.iter().enumerate() {
        blender.feed(&rgb.data, rgb.width, rgb.height, mask, corners[i]);
    }
    let (blended, coverage) = blender.blend();
    let scale = compose_scale;

    // Paste onto the full equirect canvas. In warp coordinates the full
    // sphere spans u in [-pi*scale, pi*scale), v in [0, pi*scale].
    // Canvas width uses FLOOR: a full-360 ROI spans 2*trunc(pi*s)+1 >=
    // floor(2*pi*s) columns, so every canvas column is covered (extras
    // wrap-fold); ceil left a one-pixel black hairline at the wrap seam.
    let canvas_w = (2.0 * std::f64::consts::PI * scale).floor() as usize;
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

/// Zeroes rescued images' mask pixels wherever ERODED matched coverage
/// exists: metadata-placed shots only fill holes, with a small overlap band
/// (the erosion ring) left for blending. Planar coordinates — pairs
/// spanning the 360° wrap are ignored, consistent with OpenCV's own
/// overlap handling.
fn suppress_rescued_masks(
    masks: &mut [GrayImage],
    corners: &[(i32, i32)],
    alignment: &Alignment,
    erode_iters: usize,
) {
    let rescued: Vec<usize> = (0..masks.len())
        .filter(|&i| alignment.images[i].rescued)
        .collect();
    if rescued.is_empty() || rescued.len() == masks.len() {
        return;
    }
    let matched: Vec<usize> = (0..masks.len())
        .filter(|&i| !alignment.images[i].rescued)
        .collect();

    let m_corners: Vec<(i32, i32)> = matched.iter().map(|&i| corners[i]).collect();
    let m_sizes: Vec<(i32, i32)> = matched
        .iter()
        .map(|&i| (masks[i].width as i32, masks[i].height as i32))
        .collect();
    let roi = crate::blend::result_roi(&m_corners, &m_sizes);
    let mut union = vec![0u8; roi.2 * roi.3];
    for &i in &matched {
        let (cx0, cy0) = (corners[i].0 - roi.0, corners[i].1 - roi.1);
        for y in 0..masks[i].height {
            for x in 0..masks[i].width {
                if masks[i].data[y * masks[i].width + x] != 0 {
                    union[(cy0 as usize + y) * roi.2 + cx0 as usize + x] = 255;
                }
            }
        }
    }
    for _ in 0..erode_iters {
        union = erode3(&union, roi.2, roi.3);
    }

    for &i in &rescued {
        let (w, h) = (masks[i].width, masks[i].height);
        for y in 0..h {
            for x in 0..w {
                let gx = corners[i].0 + x as i32 - roi.0;
                let gy = corners[i].1 + y as i32 - roi.1;
                if gx >= 0
                    && gy >= 0
                    && (gx as usize) < roi.2
                    && (gy as usize) < roi.3
                    && union[gy as usize * roi.2 + gx as usize] != 0
                {
                    masks[i].data[y * w + x] = 0;
                }
            }
        }
    }
}

/// 3x3 rect erosion (min filter), border treated as covered so the erosion
/// only recedes at real coverage boundaries.
fn erode3(mask: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut v = 255u8;
            for dy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                for dx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                    v = v.min(mask[dy * w + dx]);
                }
            }
            out[y * w + x] = v;
        }
    }
    out
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
