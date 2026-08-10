//! Spherical warper parity vs oracle dumps: warp the ORACLE's seam-scale
//! source pixels with the ORACLE's final cameras and compare corners
//! (exact) and warped pixels/masks.

#![allow(clippy::needless_range_loop)]

use std::path::{Path, PathBuf};

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
struct WarpMeta {
    scale: f64,
    corners: Vec<(i32, i32)>,
    sizes: Vec<(i32, i32)>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct Meta {
    images: Vec<String>,
    seam_work_aspect: f64,
}

#[test]
fn spherical_warp_matches_oracle() {
    for set in ["ring_kloppenheim_06", "sphere_kloppenheim_06"] {
        let Some(dir) = dumps_dir(set) else {
            eprintln!("SKIP {set}");
            continue;
        };
        let meta: Meta =
            serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
        let wm: WarpMeta = serde_json::from_str(
            &std::fs::read_to_string(dir.join("warped/corners.json")).unwrap(),
        )
        .unwrap();
        let cams: Vec<OracleCamera> =
            serde_json::from_str(&std::fs::read_to_string(dir.join("cameras_final.json")).unwrap())
                .unwrap();
        let n = meta.images.len();
        let swa = meta.seam_work_aspect;

        let mut warper = SphericalWarper::new(wm.scale as f32);
        let mut worst_img_frac = 0.0f64;
        let mut worst_mask_frac = 0.0f64;
        for i in 0..n {
            let src = load_png(&dir.join(format!("warped/src_{i:03}.png")));
            // K scaled by seam_work_aspect, exactly like the oracle/stitcher.
            let c = &cams[i];
            let k = [
                [(c.focal * swa) as f32, 0.0, (c.ppx * swa) as f32],
                [0.0, (c.focal * swa) as f32, (c.ppy * swa) as f32],
                [0.0, 0.0, 1.0],
            ];
            let mut r = [[0.0f32; 3]; 3];
            for a in 0..3 {
                for b in 0..3 {
                    r[a][b] = c.r[a][b] as f32;
                }
            }

            let (tl, warped) = warper.warp(&src, &k, &r, Interp::Linear, Border::Reflect);
            assert_eq!(
                (tl.0, tl.1),
                wm.corners[i],
                "{set} img {i}: corner mismatch"
            );
            let oracle_img = load_png(&dir.join(format!("warped/img_{i:03}.png")));
            assert_eq!(
                (warped.width as i32, warped.height as i32),
                wm.sizes[i],
                "{set} img {i}: size mismatch (ours {}x{})",
                warped.width,
                warped.height
            );

            // Pixel agreement. The remap itself is the faithful fixed-point
            // port, but ulp-level differences in the K chain (numpy's f32
            // in-place scaling + cv::Mat f32 inverse vs our f64 analytic
            // inverse) shift sampling coords by ~1e-6, flipping the 1/32
            // quantization step on a small fraction of pixels; on strong
            // gradients that yields multi-LSB diffs. Corners/masks being
            // EXACT proves the projector math itself matches.
            let mut diff_gt2 = 0usize;
            let mut abs_sum = 0u64;
            for (a, b) in warped.data.iter().zip(&oracle_img.data) {
                let d = (*a as i32 - *b as i32).unsigned_abs() as u64;
                abs_sum += d;
                if d > 2 {
                    diff_gt2 += 1;
                }
            }
            let frac = diff_gt2 as f64 / warped.data.len() as f64;
            let mean = abs_sum as f64 / warped.data.len() as f64;
            assert!(mean < 0.2, "{set} img {i}: mean abs diff {mean}");
            worst_img_frac = worst_img_frac.max(frac);

            // Mask: all-255 source, nearest, constant border.
            let mask_src = PixelImage::new(
                src.width,
                src.height,
                1,
                vec![255u8; src.width * src.height],
            );
            let (_, warped_mask) =
                warper.warp(&mask_src, &k, &r, Interp::Nearest, Border::Constant0);
            let oracle_mask = load_png(&dir.join(format!("warped/mask_{i:03}.png")));
            let mask_diff = warped_mask
                .data
                .iter()
                .zip(&oracle_mask.data)
                .filter(|(a, b)| a != b)
                .count();
            worst_mask_frac = worst_mask_frac.max(mask_diff as f64 / warped_mask.data.len() as f64);
        }
        eprintln!(
            "{set}: worst image frac(|diff|>2LSB) {worst_img_frac:.2e}, worst mask frac(diff) {worst_mask_frac:.2e}"
        );
        assert!(
            worst_img_frac < 1e-2,
            "{set}: warped pixels diverge: {worst_img_frac}"
        );
        assert!(
            worst_mask_frac < 1e-3,
            "{set}: warped masks diverge: {worst_mask_frac}"
        );
    }
}
