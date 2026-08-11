//! Graph-cut seam parity: feed the ORACLE's warped images + pre-seam masks
//! into our GraphCutSeamFinder(COST_COLOR) port and compare the resulting
//! seam masks with the oracle's. Deterministic max-flow on identical f32
//! inputs — expect (near-)bit-equality.

#![allow(clippy::needless_range_loop)]

use std::path::{Path, PathBuf};

use panoloom_core::imgproc::GrayImage;
use panoloom_core::seam::find_seams_graph_cut_color;
use panoloom_core::warp::PixelImage;

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
struct WarpMeta {
    corners: Vec<(i32, i32)>,
}

#[test]
fn graph_cut_seams_match_oracle() {
    for set in ["ring_kloppenheim_06", "sphere_kloppenheim_06"] {
        let Some(dir) = dumps_dir(set) else {
            eprintln!("SKIP {set}");
            continue;
        };
        let wm: WarpMeta = serde_json::from_str(
            &std::fs::read_to_string(dir.join("warped/corners.json")).unwrap(),
        )
        .unwrap();
        let n = wm.corners.len();

        let mut images = Vec::new();
        let mut masks = Vec::new();
        for i in 0..n {
            images.push(load_png(&dir.join(format!("warped/img_{i:03}.png"))));
            let m = load_png(&dir.join(format!("warped/mask_{i:03}.png")));
            masks.push(GrayImage::new(m.width, m.height, m.data));
        }

        find_seams_graph_cut_color(&images, &wm.corners, &mut masks);

        let mut worst_frac = 0.0f64;
        for i in 0..n {
            let oracle = load_png(&dir.join(format!("seams/mask_{i:03}.png")));
            let diff = masks[i]
                .data
                .iter()
                .zip(&oracle.data)
                .filter(|(a, b)| (**a != 0) != (**b != 0))
                .count();
            let frac = diff as f64 / oracle.data.len() as f64;
            worst_frac = worst_frac.max(frac);
        }
        eprintln!("{set}: worst seam-mask disagreement {worst_frac:.2e}");
        assert!(worst_frac < 1e-3, "{set}: seam masks diverge: {worst_frac}");
    }
}
