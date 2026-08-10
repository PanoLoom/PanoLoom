//! M3 preview stitch: the full Rust pipeline (features → matching →
//! alignment → warp → gain compensation → feather blend) rendering actual
//! panoramas. Writes result PNGs under the target tmp dir for eyeballing;
//! asserts coverage.

#![allow(clippy::needless_range_loop)]

use std::path::{Path, PathBuf};

use panoloom_core::blend::{result_roi, FeatherBlender};
use panoloom_core::bundle::bundle_adjust_ray;
use panoloom_core::estimation::{
    homography_based_estimate, leave_biggest_component, warped_image_scale, wave_correct_horiz,
    FeatureSet, MatchGraph,
};
use panoloom_core::exposure::{BlocksGainCompensator, RgbImage};
use panoloom_core::imgproc::{rgb_to_gray_cv, GrayImage};
use panoloom_core::matcher::match_pair;
use panoloom_core::orb::{orb_detect_and_compute, OrbParams};
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

fn save_png_rgb(path: &Path, data: &[u8], w: usize, h: usize) {
    let file = std::fs::File::create(path).unwrap();
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(data).unwrap();
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct Meta {
    images: Vec<String>,
    seam_work_aspect: f64,
}

fn run_preview(set: &str, min_coverage: f64) {
    let Some(dir) = dumps_dir(set) else {
        eprintln!("SKIP {set}: dumps not present");
        return;
    };
    let meta: Meta =
        serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
    let n = meta.images.len();
    let swa = meta.seam_work_aspect as f32;

    // --- registration on work-scale pixels ---
    let mut feats = Vec::new();
    for i in 0..n {
        let img = load_png(&dir.join(format!("work/img_{i:03}.png")));
        let gray = match img.channels {
            3 => rgb_to_gray_cv(&img.data, img.width, img.height),
            _ => GrayImage::new(img.width, img.height, img.data.clone()),
        };
        let (kps, d) = orb_detect_and_compute(&gray, &OrbParams::default());
        let pts: Vec<[f32; 2]> = kps.iter().map(|k| [k.x, k.y]).collect();
        feats.push((pts, d, (img.width as u32, img.height as u32)));
    }
    let mut upper = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            upper.push((
                (i, j),
                match_pair(
                    &feats[i].0,
                    &feats[i].1,
                    feats[i].2,
                    &feats[j].0,
                    &feats[j].1,
                    feats[j].2,
                ),
            ));
        }
    }
    let graph = MatchGraph::from_upper_triangle(n, upper);
    assert_eq!(leave_biggest_component(&graph, 1.0).len(), n);
    let features: Vec<FeatureSet> = feats
        .iter()
        .map(|(pts, _, (w, h))| FeatureSet {
            width: *w,
            height: *h,
            keypoints: pts.clone(),
        })
        .collect();
    let mut cameras = homography_based_estimate(&features, &graph);
    assert!(bundle_adjust_ray(&features, &graph, &mut cameras));
    let mut rmats: Vec<[[f32; 3]; 3]> = cameras.iter().map(|c| c.r).collect();
    wave_correct_horiz(&mut rmats);
    for (c, r) in cameras.iter_mut().zip(&rmats) {
        c.r = *r;
    }

    // --- warp seam-scale images with OUR cameras ---
    let scale = warped_image_scale(&cameras) as f32 * swa;
    let mut warper = SphericalWarper::new(scale);
    let mut corners = Vec::new();
    let mut sizes = Vec::new();
    let mut warped_imgs = Vec::new();
    let mut warped_masks = Vec::new();
    for i in 0..n {
        let src = load_png(&dir.join(format!("warped/src_{i:03}.png")));
        let c = &cameras[i];
        let k = [
            [c.focal as f32 * swa, 0.0, c.ppx as f32 * swa],
            [0.0, c.focal as f32 * swa, c.ppy as f32 * swa],
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
        corners.push(tl);
        sizes.push((w_img.width as i32, w_img.height as i32));
        warped_masks.push(GrayImage::new(w_mask.width, w_mask.height, w_mask.data));
        warped_imgs.push(w_img);
    }

    // --- gain compensation (BlocksGainCompensator, bit-exact port) ---
    let mut rgb_imgs: Vec<RgbImage> = warped_imgs
        .iter()
        .map(|w| RgbImage::new(w.width, w.height, w.data.clone()))
        .collect();
    let compensator = BlocksGainCompensator::feed(&corners, &rgb_imgs, &warped_masks);
    for (i, img) in rgb_imgs.iter_mut().enumerate() {
        compensator.apply(i, img);
    }

    // --- feather blend ---
    let roi = result_roi(&corners, &sizes);
    let mut blender = FeatherBlender::new(0.02);
    blender.prepare(roi.0, roi.1, roi.2, roi.3);
    for i in 0..n {
        blender.feed(
            &rgb_imgs[i].data,
            rgb_imgs[i].width,
            rgb_imgs[i].height,
            &warped_masks[i],
            corners[i],
        );
    }
    let (pano, coverage) = blender.blend();

    let covered =
        coverage.data.iter().filter(|&&m| m != 0).count() as f64 / coverage.data.len() as f64;
    eprintln!(
        "{set}: {}x{} canvas, coverage {:.1}%",
        roi.2,
        roi.3,
        covered * 100.0
    );
    assert!(covered > min_coverage, "{set}: coverage too low: {covered}");

    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("preview_{set}.png"));
    save_png_rgb(&out, &pano, roi.2, roi.3);
    eprintln!("wrote {}", out.display());
}

#[test]
fn preview_stitch_ring() {
    run_preview("ring_kloppenheim_06", 0.5);
}

#[test]
fn preview_stitch_sphere() {
    run_preview("sphere_kloppenheim_06", 0.8);
}
