//! End-to-end alignment (M2 acceptance): full Rust pipeline from work-scale
//! pixels to wave-corrected camera poses, scored against synthetic ground
//! truth. Gates: mean relative-pose error within 20% of the OpenCV oracle's
//! own accuracy on the same datasets (ring 0.611°, sphere 0.204°).

#![allow(clippy::needless_range_loop)]

use std::path::{Path, PathBuf};

use panoloom_core::bundle::bundle_adjust_ray;
use panoloom_core::estimation::{
    homography_based_estimate, leave_biggest_component, wave_correct_horiz, FeatureSet, MatchGraph,
};
use panoloom_core::imgproc::{rgb_to_gray_cv, GrayImage};
use panoloom_core::matcher::match_pair;
use panoloom_core::orb::{orb_detect_and_compute, OrbParams};

fn dumps_dir(set: &str) -> Option<PathBuf> {
    let p =
        Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../tools/reference/dumps/{set}"));
    p.exists().then_some(p)
}

fn truth_path(set: &str) -> Option<PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
        "../../tools/testdata/generated/{set}/ground_truth.json"
    ));
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
#[serde(rename_all = "camelCase")]
struct TruthImage {
    file_name: String,
    yaw: f64,
    pitch: f64,
    roll: f64,
}

#[derive(serde::Deserialize)]
struct Truth {
    images: Vec<TruthImage>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct Meta {
    images: Vec<String>,
}

type Mat3 = [[f64; 3]; 3];

fn mat3_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            for k in 0..3 {
                out[r][c] += a[r][k] * b[k][c];
            }
        }
    }
    out
}

/// Ground-truth rotation, matching tools/testdata/generate.py conventions:
/// R = Ry(yaw) · Rx(-pitch) · Rz(roll), degrees, y down, +pitch looks up.
fn truth_rotation(yaw: f64, pitch: f64, roll: f64) -> Mat3 {
    let (y, p, r) = (yaw.to_radians(), (-pitch).to_radians(), roll.to_radians());
    let ry = [
        [y.cos(), 0.0, y.sin()],
        [0.0, 1.0, 0.0],
        [-y.sin(), 0.0, y.cos()],
    ];
    let rx = [
        [1.0, 0.0, 0.0],
        [0.0, p.cos(), -p.sin()],
        [0.0, p.sin(), p.cos()],
    ];
    let rz = [
        [r.cos(), -r.sin(), 0.0],
        [r.sin(), r.cos(), 0.0],
        [0.0, 0.0, 1.0],
    ];
    mat3_mul(&mat3_mul(&ry, &rx), &rz)
}

fn rotation_angle_deg(a: &Mat3, b: &Mat3) -> f64 {
    let mut tr = 0.0;
    for i in 0..3 {
        for k in 0..3 {
            tr += a[i][k] * b[i][k];
        }
    }
    (((tr - 1.0) / 2.0).clamp(-1.0, 1.0)).acos().to_degrees()
}

fn run_set(set: &str, oracle_baseline_deg: f64) {
    let (Some(dir), Some(truth_file)) = (dumps_dir(set), truth_path(set)) else {
        eprintln!("SKIP {set}: dumps or ground truth not present");
        return;
    };
    let meta: Meta =
        serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
    let truth: Truth = serde_json::from_str(&std::fs::read_to_string(truth_file).unwrap()).unwrap();
    let n = meta.images.len();

    // 1. Features (our ORB on the oracle's work-scale pixels).
    let mut features = Vec::new();
    let mut descs = Vec::new();
    let mut pts: Vec<Vec<[f32; 2]>> = Vec::new();
    let mut sizes = Vec::new();
    for i in 0..n {
        let img = load_png_gray(&dir.join(format!("work/img_{i:03}.png")));
        let (kps, d) = orb_detect_and_compute(&img, &OrbParams::default());
        sizes.push((img.width as u32, img.height as u32));
        pts.push(kps.iter().map(|k| [k.x, k.y]).collect());
        features.push(FeatureSet {
            width: img.width as u32,
            height: img.height as u32,
            keypoints: kps.iter().map(|k| [k.x, k.y]).collect(),
        });
        descs.push(d);
    }

    // 2. Pairwise matching over all pairs.
    let mut upper = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let pm = match_pair(&pts[i], &descs[i], sizes[i], &pts[j], &descs[j], sizes[j]);
            upper.push(((i, j), pm));
        }
    }
    let graph = MatchGraph::from_upper_triangle(n, upper);

    // 3. Biggest component (subset everything if needed).
    let kept = leave_biggest_component(&graph, 1.0);
    assert!(
        kept.len() >= n * 4 / 5,
        "{set}: too many images dropped: kept {}/{n}",
        kept.len()
    );
    let (features, graph, truth_rs): (Vec<FeatureSet>, MatchGraph, Vec<Mat3>) = if kept.len() == n {
        let t = truth
            .images
            .iter()
            .map(|t| truth_rotation(t.yaw, t.pitch, t.roll))
            .collect();
        (features, graph, t)
    } else {
        let sub_features: Vec<FeatureSet> = kept.iter().map(|&i| features[i].clone()).collect();
        let m = kept.len();
        let mut sub_upper = Vec::new();
        for a in 0..m {
            for b in (a + 1)..m {
                sub_upper.push(((a, b), graph.at(kept[a], kept[b]).clone()));
            }
        }
        let by_name: std::collections::HashMap<&str, &TruthImage> = truth
            .images
            .iter()
            .map(|t| (t.file_name.as_str(), t))
            .collect();
        let t = kept
            .iter()
            .map(|&i| {
                let ti = by_name[meta.images[i].as_str()];
                truth_rotation(ti.yaw, ti.pitch, ti.roll)
            })
            .collect();
        (
            sub_features,
            MatchGraph::from_upper_triangle(m, sub_upper),
            t,
        )
    };

    // 4. Rotation estimation -> bundle adjustment -> wave correction.
    let mut cameras = homography_based_estimate(&features, &graph);
    assert!(
        bundle_adjust_ray(&features, &graph, &mut cameras),
        "{set}: bundle adjustment failed"
    );
    let mut rmats: Vec<[[f32; 3]; 3]> = cameras.iter().map(|c| c.r).collect();
    wave_correct_horiz(&mut rmats);

    // 5. Score relative poses against ground truth.
    let ours: Vec<Mat3> = rmats
        .iter()
        .map(|m| {
            let mut r = [[0.0f64; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    r[i][j] = m[i][j] as f64;
                }
            }
            r
        })
        .collect();
    let m = ours.len();
    let mut errs = Vec::new();
    for i in 0..m {
        for j in (i + 1)..m {
            let rel_ours = mat3_mul(&transpose(&ours[i]), &ours[j]);
            let rel_true = mat3_mul(&transpose(&truth_rs[i]), &truth_rs[j]);
            // rotation_angle_deg computes angle(a·bᵀ) — exactly the geodesic
            // distance between the two relative rotations.
            errs.push(rotation_angle_deg(&rel_ours, &rel_true));
        }
    }
    let mean = errs.iter().sum::<f64>() / errs.len() as f64;
    let max = errs.iter().cloned().fold(0.0f64, f64::max);
    eprintln!(
        "{set}: {m} cameras, mean relative-pose error {mean:.3}° (max {max:.3}°); oracle baseline {oracle_baseline_deg}°"
    );
    assert!(
        mean <= oracle_baseline_deg * 1.2,
        "{set}: mean error {mean:.3}° exceeds 1.2x oracle baseline {oracle_baseline_deg}°"
    );
}

fn transpose(m: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = m[j][i];
        }
    }
    out
}

#[test]
fn e2e_alignment_ring() {
    run_set("ring_kloppenheim_06", 0.611);
}

#[test]
fn e2e_alignment_sphere() {
    run_set("sphere_kloppenheim_06", 0.204);
}
