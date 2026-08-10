//! ORB parity vs the OpenCV oracle dumps (tools/reference/dumps).
//!
//! Skips silently when dumps are absent (they are generated locally, see
//! tools/reference/README.md). Run with `-- --nocapture` for the metrics.

use std::path::{Path, PathBuf};

use panoloom_core::imgproc::{rgb_to_gray_cv, GrayImage};
use panoloom_core::orb::{orb_detect_and_compute, OrbParams};

fn dumps_dir() -> Option<PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tools/reference/dumps/ring_kloppenheim_06");
    p.exists().then_some(p)
}

fn load_png_gray(path: &Path) -> GrayImage {
    let decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap()));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    buf.truncate(info.buffer_size());
    let (w, h) = (info.width as usize, info.height as usize);
    match info.color_type {
        png::ColorType::Rgb => rgb_to_gray_cv(&buf, w, h),
        png::ColorType::Grayscale => GrayImage::new(w, h, buf),
        other => panic!("unexpected png color type {other:?}"),
    }
}

#[derive(serde::Deserialize)]
struct OracleKp {
    x: f32,
    y: f32,
    angle: f32,
    octave: i32,
    #[allow(dead_code)]
    response: f32,
}

/// Minimal .npy (v1.0) reader for 2-D u8 arrays.
fn load_npy_u8_2d(path: &Path) -> (usize, usize, Vec<u8>) {
    let raw = std::fs::read(path).unwrap();
    assert_eq!(&raw[..6], b"\x93NUMPY");
    let header_len = u16::from_le_bytes([raw[8], raw[9]]) as usize;
    let header = std::str::from_utf8(&raw[10..10 + header_len]).unwrap();
    assert!(header.contains("'|u1'"), "not u8: {header}");
    assert!(header.contains("'fortran_order': False"));
    let shape_part = header.split("'shape':").nth(1).unwrap();
    let shape_str = shape_part
        .split('(')
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap();
    let dims: Vec<usize> = shape_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    assert_eq!(dims.len(), 2, "shape: {shape_str}");
    let data = raw[10 + header_len..].to_vec();
    assert_eq!(data.len(), dims[0] * dims[1]);
    (dims[0], dims[1], data)
}

fn hamming(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}

#[test]
fn orb_keypoints_and_descriptors_vs_oracle() {
    let Some(dir) = dumps_dir() else {
        eprintln!("SKIP: oracle dumps not present");
        return;
    };

    let mut total_recall_num = 0usize;
    let mut total_recall_den = 0usize;
    let mut exact_desc = 0usize;
    let mut close_desc = 0usize;
    let mut matched_desc = 0usize;
    let mut ham_sum = 0u64;

    for img_idx in 0..8 {
        let img = load_png_gray(&dir.join(format!("work/img_{img_idx:03}.png")));
        let (kps, descs) = orb_detect_and_compute(&img, &OrbParams::default());

        let oracle_kps: Vec<OracleKp> = serde_json::from_str(
            &std::fs::read_to_string(dir.join(format!("features/img_{img_idx:03}.json"))).unwrap(),
        )
        .unwrap();
        let (n_desc, desc_w, oracle_desc) =
            load_npy_u8_2d(&dir.join(format!("features/img_{img_idx:03}.desc.npy")));
        assert_eq!(desc_w, 32);
        assert_eq!(n_desc, oracle_kps.len());

        // Greedy coordinate matching: same octave, position within 1px
        // (level-0 detections should coincide exactly; upper levels may
        // drift via the f32-vs-fixed-point pyramid resize).
        let mut found = 0usize;
        let mut per_level = [[0usize; 2]; 8];
        for (oi, okp) in oracle_kps.iter().enumerate() {
            per_level[okp.octave as usize][1] += 1;
            let best = kps
                .iter()
                .enumerate()
                .filter(|(_, k)| k.octave == okp.octave)
                .map(|(i, k)| {
                    let d = ((k.x - okp.x).powi(2) + (k.y - okp.y).powi(2)).sqrt();
                    (i, d)
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            if let Some((ki, dist)) = best {
                if dist <= 1.0 {
                    found += 1;
                    per_level[okp.octave as usize][0] += 1;
                    // Angle agreement gate for descriptor comparison.
                    let da = (kps[ki].angle - okp.angle)
                        .abs()
                        .min(360.0 - (kps[ki].angle - okp.angle).abs());
                    if dist < 1e-3 && da < 0.5 {
                        let h = hamming(&descs[ki], &oracle_desc[oi * 32..(oi + 1) * 32]);
                        matched_desc += 1;
                        ham_sum += h as u64;
                        if h == 0 {
                            exact_desc += 1;
                        }
                        if h <= 16 {
                            close_desc += 1;
                        }
                    }
                }
            }
        }
        total_recall_num += found;
        total_recall_den += oracle_kps.len();
        eprintln!(
            "img_{img_idx:03}: ours={} oracle={} recall={:.1}% per-level(found/oracle)={:?}",
            kps.len(),
            oracle_kps.len(),
            100.0 * found as f64 / oracle_kps.len() as f64,
            per_level
                .iter()
                .take(4)
                .map(|c| format!("{}/{}", c[0], c[1]))
                .collect::<Vec<_>>()
        );
    }

    let recall = total_recall_num as f64 / total_recall_den as f64;
    let mean_ham = ham_sum as f64 / matched_desc.max(1) as f64;
    eprintln!(
        "TOTAL recall {:.1}%  |  descriptors on exact-position matches: n={} exact={:.1}% close(≤16 bits)={:.1}% meanHamming={:.1}",
        100.0 * recall,
        matched_desc,
        100.0 * exact_desc as f64 / matched_desc.max(1) as f64,
        100.0 * close_desc as f64 / matched_desc.max(1) as f64,
        mean_ham
    );

    // Initial gates — tightened as the port converges (see docs/pipeline.md).
    assert!(recall >= 0.70, "keypoint recall too low: {recall}");
    assert!(
        close_desc as f64 / matched_desc.max(1) as f64 >= 0.80,
        "descriptors diverge from oracle"
    );
}
