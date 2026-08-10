//! Estimator + waveCorrect parity vs the oracle dumps: feed the ORACLE's
//! features/matches into our ports and compare camera outputs stage by
//! stage (isolates each stage from upstream feature differences).

#![allow(clippy::needless_range_loop)] // matrix loops mirror the C++

use std::path::{Path, PathBuf};

use panoloom_core::estimation::{
    homography_based_estimate, leave_biggest_component, wave_correct_horiz, FeatureSet, MatchGraph,
};
use panoloom_core::matcher::{PairMatches, RawMatch};

fn dumps_dir(set: &str) -> Option<PathBuf> {
    let p =
        Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../tools/reference/dumps/{set}"));
    p.exists().then_some(p)
}

#[derive(serde::Deserialize)]
struct OracleKp {
    x: f32,
    y: f32,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct OraclePair {
    num_inliers: usize,
    confidence: f64,
    #[serde(rename = "H")]
    h: Option<[[f64; 3]; 3]>,
    matches: Vec<(u32, u32, f32)>,
    inliers_mask: Vec<u8>,
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
    kept_indices: Vec<usize>,
}

/// Loads oracle features + dense match graph for the images the oracle
/// actually stitched (its meta.json image list is pre-subset).
fn load_oracle_graph(dir: &Path) -> (Vec<FeatureSet>, MatchGraph) {
    let meta: Meta =
        serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
    let n = meta.kept_indices.len().min(meta.images.len());

    let mut features = Vec::new();
    for i in 0..n {
        let kps: Vec<OracleKp> = serde_json::from_str(
            &std::fs::read_to_string(dir.join(format!("features/img_{i:03}.json"))).unwrap(),
        )
        .unwrap();
        // Work-scale image dimensions from the dumped PNG header.
        let png = std::fs::File::open(dir.join(format!("work/img_{i:03}.png"))).unwrap();
        let reader = png::Decoder::new(std::io::BufReader::new(png))
            .read_info()
            .unwrap();
        let info = reader.info();
        features.push(FeatureSet {
            width: info.width,
            height: info.height,
            keypoints: kps.iter().map(|k| [k.x, k.y]).collect(),
        });
    }

    let mut upper = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let op: OraclePair = serde_json::from_str(
                &std::fs::read_to_string(dir.join(format!("matches/pair_{i:03}_{j:03}.json")))
                    .unwrap(),
            )
            .unwrap();
            upper.push((
                (i, j),
                PairMatches {
                    matches: op
                        .matches
                        .iter()
                        .map(|&(q, t, d)| RawMatch {
                            query: q as usize,
                            train: t as usize,
                            distance: d,
                        })
                        .collect(),
                    inliers: op.inliers_mask.iter().map(|&b| b != 0).collect(),
                    num_inliers: op.num_inliers,
                    h: op.h,
                    confidence: op.confidence,
                },
            ));
        }
    }
    (features, MatchGraph::from_upper_triangle(n, upper))
}

fn rotation_angle_deg(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> f64 {
    // angle(a * b^T)
    let mut tr = 0.0;
    for i in 0..3 {
        for k in 0..3 {
            tr += a[i][k] * b[i][k];
        }
    }
    (((tr - 1.0) / 2.0).clamp(-1.0, 1.0)).acos().to_degrees()
}

/// Ring: our estimator output must equal the oracle's bit-near-exactly
/// (the ring has no spanning-tree weight ties, so the tree is unique).
#[test]
fn estimator_matches_cameras_initial_ring() {
    let Some(dir) = dumps_dir("ring_kloppenheim_06") else {
        eprintln!("SKIP: dumps not present");
        return;
    };
    let (features, graph) = load_oracle_graph(&dir);
    let kept = leave_biggest_component(&graph, 1.0);
    assert_eq!(kept.len(), features.len(), "oracle dumps are pre-subset");

    let cameras = homography_based_estimate(&features, &graph);
    let oracle: Vec<OracleCamera> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("cameras_initial.json")).unwrap())
            .unwrap();
    assert_eq!(cameras.len(), oracle.len());

    let mut worst_focal = 0.0f64;
    let mut worst_rot = 0.0f64;
    for (c, o) in cameras.iter().zip(&oracle) {
        worst_focal = worst_focal.max(((c.focal - o.focal) / o.focal).abs());
        assert!((c.ppx - o.ppx).abs() < 1e-9 && (c.ppy - o.ppy).abs() < 1e-9);
        // NOTE: cameras_initial R's are NOT orthonormal — CalcRotation
        // chains K^-1·H^-1·K without removing the homography's scale.
        // Angle metrics are invalid here; compare elementwise.
        let mut max_abs = 0.0f64;
        let mut max_diff = 0.0f64;
        for i in 0..3 {
            for j in 0..3 {
                max_abs = max_abs.max(o.r[i][j].abs());
                max_diff = max_diff.max((c.r[i][j] as f64 - o.r[i][j]).abs());
            }
        }
        worst_rot = worst_rot.max(max_diff / max_abs);
    }
    eprintln!(
        "ring: estimator worst focal rel diff {worst_focal:.2e}, worst R elementwise rel diff {worst_rot:.2e}"
    );
    assert!(worst_focal < 1e-6, "focal diverges: {worst_focal}");
    assert!(worst_rot < 1e-5, "rotations diverge: {worst_rot}");
}

/// Sphere: the 26-image graph has many tied edge weights, and OpenCV's
/// unstable std::sort tie order is not reproducible (nor contractual), so
/// the chosen spanning tree — and hence the raw initial rotations — can
/// legitimately differ. Instead we validate the MATH: the oracle's own
/// camera outputs must satisfy our CalcRotation chain relation
/// (R_to = R_from · K_from⁻¹·H⁻¹·K_to) on the edges OpenCV actually used,
/// and those edges must form a spanning tree.
#[test]
fn estimator_math_validates_on_sphere_oracle_edges() {
    let Some(dir) = dumps_dir("sphere_kloppenheim_06") else {
        eprintln!("SKIP: dumps not present");
        return;
    };
    let (features, graph) = load_oracle_graph(&dir);
    let oracle: Vec<OracleCamera> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("cameras_initial.json")).unwrap())
            .unwrap();
    let n = features.len();

    // Focals must still be exact (median formula is tie-free).
    let focals = panoloom_core::estimation::estimate_focal(&features, &graph);
    for (f, o) in focals.iter().zip(&oracle) {
        assert!(((f - o.focal) / o.focal).abs() < 1e-12);
    }

    let k_of = |o: &OracleCamera| -> [[f64; 3]; 3] {
        // ppx/ppy in cameras_initial are already image-center based; the
        // chain ran with pp = 0, so use pp = 0 K's (like CalcRotation did).
        [[o.focal, 0.0, 0.0], [0.0, o.focal, 0.0], [0.0, 0.0, 1.0]]
    };

    let mat3_mul = |a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]| -> [[f64; 3]; 3] {
        let mut out = [[0.0; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                for k in 0..3 {
                    out[r][c] += a[r][k] * b[k][c];
                }
            }
        }
        out
    };

    let mut used_edges: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let Some(h) = &graph.at(i, j).h else { continue };
            let m = mat3_mul(
                &mat3_mul(
                    &panoloom_core::estimation::invert_3x3(&k_of(&oracle[i])),
                    &panoloom_core::estimation::invert_3x3(h),
                ),
                &k_of(&oracle[j]),
            );
            let predicted = mat3_mul(&oracle[i].r, &m);
            // f32-storage tolerance: oracle R's went through CV_32F.
            let mut max_abs = 0.0f64;
            let mut max_diff = 0.0f64;
            for r in 0..3 {
                for c in 0..3 {
                    max_abs = max_abs.max(predicted[r][c].abs());
                    max_diff = max_diff.max((predicted[r][c] - oracle[j].r[r][c]).abs());
                }
            }
            if max_diff / max_abs < 1e-5 {
                used_edges.push((i, j));
            }
        }
    }

    // The verified edges must connect all cameras (a spanning tree walked
    // from the center verifies in one direction; duals may also verify).
    let mut ds = panoloom_core::estimation::DisjointSets::new(n);
    for &(i, j) in &used_edges {
        let a = ds.find_set_by_elem(i);
        let b = ds.find_set_by_elem(j);
        if a != b {
            ds.merge_sets(a, b);
        }
    }
    let root = ds.find_set_by_elem(0);
    let connected = (0..n).all(|i| ds.find_set_by_elem(i) == root);
    eprintln!(
        "sphere: {} oracle edges satisfy our CalcRotation relation; graph connected: {connected}",
        used_edges.len()
    );
    assert!(used_edges.len() >= n - 1, "too few verified edges");
    assert!(connected, "verified edges do not span all cameras");
}

#[test]
fn wave_correct_matches_cameras_final() {
    for set in ["ring_kloppenheim_06", "sphere_kloppenheim_06"] {
        let Some(dir) = dumps_dir(set) else {
            eprintln!("SKIP {set}: dumps not present");
            continue;
        };
        let ba: Vec<OracleCamera> =
            serde_json::from_str(&std::fs::read_to_string(dir.join("cameras_ba.json")).unwrap())
                .unwrap();
        let fin: Vec<OracleCamera> =
            serde_json::from_str(&std::fs::read_to_string(dir.join("cameras_final.json")).unwrap())
                .unwrap();

        let mut rmats: Vec<[[f32; 3]; 3]> = ba
            .iter()
            .map(|c| {
                let mut m = [[0.0f32; 3]; 3];
                for i in 0..3 {
                    for j in 0..3 {
                        m[i][j] = c.r[i][j] as f32;
                    }
                }
                m
            })
            .collect();
        wave_correct_horiz(&mut rmats);

        let mut worst = 0.0f64;
        for (m, o) in rmats.iter().zip(&fin) {
            let mut r64 = [[0.0f64; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    r64[i][j] = m[i][j] as f64;
                }
            }
            worst = worst.max(rotation_angle_deg(&r64, &o.r));
        }
        eprintln!("{set}: waveCorrect worst rotation diff {worst:.5}°");
        assert!(worst < 0.01, "{set}: wave correction diverges: {worst}°");
    }
}
