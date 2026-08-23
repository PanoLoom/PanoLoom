//! M4 compose parity: full-resolution warp + gain apply + seam-mask
//! upscale + multiband blend, from ORACLE-fixed inputs (cameras, seam
//! masks, full-res pixels), gated by SSIM against the oracle's final
//! panorama plus exact corner/size/num_bands agreement.

#![allow(clippy::needless_range_loop)]

use std::path::{Path, PathBuf};

use panoloom_core::blend::{num_bands_for, result_roi, MultiBandBlender};
use panoloom_core::estimation::warped_image_scale;
use panoloom_core::exposure::{BlocksGainCompensator, RgbImage};
use panoloom_core::imgproc::{resize_bilinear, GrayImage};
use panoloom_core::warp::{Border, Interp, PixelImage, SphericalWarper};

fn dumps_dir(set: &str) -> Option<PathBuf> {
    let p =
        Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../tools/reference/dumps/{set}"));
    p.exists().then_some(p)
}

fn load_png(path: &Path) -> PixelImage {
    let decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap()));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    buf.truncate(info.buffer_size());
    let ch = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Grayscale => 1,
        other => panic!("{other:?}"),
    };
    PixelImage::new(info.width as usize, info.height as usize, ch, buf)
}

#[derive(serde::Deserialize)]
struct OracleCamera {
    focal: f64,
    ppx: f64,
    ppy: f64,
    #[serde(rename = "R")]
    r: [[f64; 3]; 3],
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct Meta {
    images: Vec<String>,
    work_scale: f64,
}

#[derive(serde::Deserialize)]
struct WarpMeta {
    corners: Vec<(i32, i32)>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct ComposeMeta {
    corners: Vec<(i32, i32)>,
    sizes: Vec<(i32, i32)>,
    num_bands: usize,
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

/// Uniform-window (8x8) grayscale SSIM.
fn ssim_gray(a: &[u8], b: &[u8], w: usize, h: usize) -> f64 {
    const C1: f64 = 6.5025; // (0.01*255)^2
    const C2: f64 = 58.5225; // (0.03*255)^2
    let win = 8usize;
    let mut sum = 0.0;
    let mut count = 0usize;
    for by in (0..h.saturating_sub(win)).step_by(win) {
        for bx in (0..w.saturating_sub(win)).step_by(win) {
            let (mut ma, mut mb) = (0.0f64, 0.0f64);
            for y in by..by + win {
                for x in bx..bx + win {
                    ma += a[y * w + x] as f64;
                    mb += b[y * w + x] as f64;
                }
            }
            let n = (win * win) as f64;
            ma /= n;
            mb /= n;
            let (mut va, mut vb, mut cov) = (0.0f64, 0.0, 0.0);
            for y in by..by + win {
                for x in bx..bx + win {
                    let da = a[y * w + x] as f64 - ma;
                    let db = b[y * w + x] as f64 - mb;
                    va += da * da;
                    vb += db * db;
                    cov += da * db;
                }
            }
            va /= n - 1.0;
            vb /= n - 1.0;
            cov /= n - 1.0;
            sum += ((2.0 * ma * mb + C1) * (2.0 * cov + C2))
                / ((ma * ma + mb * mb + C1) * (va + vb + C2));
            count += 1;
        }
    }
    sum / count as f64
}

#[test]
fn full_res_compose_matches_oracle() {
    let Some(dir) = dumps_dir("ring_kloppenheim_06") else {
        eprintln!("SKIP: dumps not present");
        return;
    };
    if !dir.join("full").exists() {
        eprintln!("SKIP: full-res fixtures not dumped (oracle --dump-full)");
        return;
    }
    let meta: Meta =
        serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
    let cams: Vec<OracleCamera> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("cameras_final.json")).unwrap())
            .unwrap();
    let wm: WarpMeta =
        serde_json::from_str(&std::fs::read_to_string(dir.join("warped/corners.json")).unwrap())
            .unwrap();
    let cm: ComposeMeta =
        serde_json::from_str(&std::fs::read_to_string(dir.join("compose.json")).unwrap()).unwrap();
    let n = meta.images.len();

    let cameras: Vec<panoloom_core::camera::CameraParams> = cams
        .iter()
        .map(|o| {
            let mut r = [[0.0f32; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    r[i][j] = o.r[i][j] as f32;
                }
            }
            panoloom_core::camera::CameraParams {
                focal: o.focal,
                aspect: 1.0,
                ppx: o.ppx,
                ppy: o.ppy,
                r,
            }
        })
        .collect();

    // Gain compensator fed at SEAM scale from oracle warp dumps.
    let mut seam_imgs = Vec::new();
    let mut seam_masks_cov = Vec::new();
    for i in 0..n {
        let im = load_png(&dir.join(format!("warped/img_{i:03}.png")));
        seam_imgs.push(RgbImage::new(im.width, im.height, im.data));
        let m = load_png(&dir.join(format!("warped/mask_{i:03}.png")));
        seam_masks_cov.push(GrayImage::new(m.width, m.height, m.data));
    }
    let compensator = BlocksGainCompensator::feed(&wm.corners, &seam_imgs, &seam_masks_cov);

    // Full-res warp with compose-scale K.
    let cwa = 1.0 / meta.work_scale;
    let scale = (warped_image_scale(&cameras) * cwa) as f32;
    let mut warper = SphericalWarper::new(scale);
    let mut corners = Vec::new();
    let mut sizes = Vec::new();
    let mut fed: Vec<(RgbImage, GrayImage)> = Vec::new();
    for i in 0..n {
        let src = load_png(&dir.join(format!("full/img_{i:03}.png")));
        let c = &cameras[i];
        let k = [
            [(c.focal * cwa) as f32, 0.0, (c.ppx * cwa) as f32],
            [0.0, (c.focal * cwa) as f32, (c.ppy * cwa) as f32],
            [0.0, 0.0, 1.0],
        ];
        let (tl, w_img) = warper.warp(&src, &k, &c.r, Interp::Linear, Border::Reflect);
        let mask_src = PixelImage::new(
            src.width,
            src.height,
            1,
            vec![255u8; src.width * src.height],
        );
        let (_, w_mask) = warper.warp(&mask_src, &k, &c.r, Interp::Nearest, Border::Constant0);
        assert_eq!((tl.0, tl.1), cm.corners[i], "compose corner {i}");
        assert_eq!(
            (w_img.width as i32, w_img.height as i32),
            cm.sizes[i],
            "compose size {i}"
        );

        // Gain apply at full res.
        let mut rgb = RgbImage::new(w_img.width, w_img.height, w_img.data);
        compensator.apply(i, &mut rgb);

        // Seam mask: dilate at seam scale, upscale, AND with coverage.
        let sm = load_png(&dir.join(format!("seams/mask_{i:03}.png")));
        let seam_mask = GrayImage::new(sm.width, sm.height, sm.data);
        let dilated = dilate3(&seam_mask);
        let up = resize_bilinear(&dilated, w_mask.width, w_mask.height);
        let mut final_mask = vec![0u8; w_mask.width * w_mask.height];
        for p in 0..final_mask.len() {
            final_mask[p] = up.data[p] & w_mask.data[p];
        }
        fed.push((rgb, GrayImage::new(w_mask.width, w_mask.height, final_mask)));
        corners.push(tl);
        sizes.push((w_img.width as i32, w_img.height as i32));
    }

    let roi = result_roi(&corners, &sizes);
    let bands = num_bands_for(roi.2, roi.3);
    assert_eq!(bands, cm.num_bands, "num_bands");
    let mut blender = MultiBandBlender::new(bands);
    blender.prepare(roi.0, roi.1, roi.2, roi.3);
    for (i, (img, mask)) in fed.iter().enumerate() {
        blender.feed(&img.data, img.width, img.height, mask, corners[i]);
    }
    let (pano, _cov) = blender.blend();

    // SSIM against the oracle's lossless final result.
    let oracle = load_png(&dir.join("result.png"));
    assert_eq!((oracle.width, oracle.height), (roi.2, roi.3), "canvas size");
    let to_gray = |rgb: &[u8]| -> Vec<u8> {
        rgb.as_chunks::<3>()
            .0
            .iter()
            .map(|p| {
                ((p[0] as u32 * 4899 + p[1] as u32 * 9617 + p[2] as u32 * 1868 + 8192) >> 14) as u8
            })
            .collect()
    };
    let ga = to_gray(&pano);
    let gb = to_gray(&oracle.data);
    let ssim = ssim_gray(&ga, &gb, roi.2, roi.3);
    let mean_abs: f64 = pano
        .iter()
        .zip(&oracle.data)
        .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs() as f64)
        .sum::<f64>()
        / pano.len() as f64;
    eprintln!("compose: SSIM {ssim:.4}, mean abs diff {mean_abs:.3}");
    assert!(ssim >= 0.98, "SSIM below gate: {ssim}");
}
