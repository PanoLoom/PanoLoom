//! Parity test: `find_homography` vs `cv2.findHomography(src, dst, RANSAC)`.
//!
//! Fixtures are produced by `tools/reference/gen_homography_fixtures.py`
//! (OpenCV 4.14.0). Because the port replicates cv::RNG, getSubset,
//! checkSubset, the Jacobi eigensolver, computeError (f32) and the LM
//! refinement call-for-call, the inlier masks are required to be BIT-EQUAL
//! on every case — a mask mismatch means a bug in the port, not a
//! tolerance problem.
//!
//! H tolerances (after normalizing both sides to h22 == 1, elementwise
//! max-abs diff relative to max |H_cv|):
//! - noise-free cases: 1e-6 (observed agreement is ~1e-12; the slack only
//!   covers libm/FMA-contraction ulp noise, see homography.rs header)
//! - noisy cases: 1e-4 (LM's branchy lambda schedule can amplify last-ulp
//!   reduction-order differences; observed agreement is still far tighter)

use panoloom_core::homography::find_homography;
use serde_json::Value;
use std::path::PathBuf;

struct Case {
    name: String,
    noise_free: bool,
    src: Vec<[f32; 2]>,
    dst: Vec<[f32; 2]>,
    h_cv: [[f64; 3]; 3],
    mask_cv: Vec<bool>,
}

fn load_case(n: usize) -> Case {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tools/reference/fixtures/homography")
        .join(format!("case_{n}.json"));
    let data = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "cannot read {} ({e}); run tools/reference/gen_homography_fixtures.py",
            path.display()
        )
    });
    let v: Value = serde_json::from_str(&data).expect("invalid fixture JSON");

    let pts = |key: &str| -> Vec<[f32; 2]> {
        v[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| [p[0].as_f64().unwrap() as f32, p[1].as_f64().unwrap() as f32])
            .collect()
    };

    let h_arr = v["H"].as_array().unwrap();
    let mut h_cv = [[0.0f64; 3]; 3];
    for (r, row) in h_arr.iter().enumerate() {
        for c in 0..3 {
            h_cv[r][c] = row[c].as_f64().unwrap();
        }
    }

    Case {
        name: v["name"].as_str().unwrap().to_string(),
        noise_free: v["noise_free"].as_bool().unwrap(),
        src: pts("src"),
        dst: pts("dst"),
        h_cv,
        mask_cv: v["mask"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m.as_i64().unwrap() != 0)
            .collect(),
    }
}

fn run_case(n: usize) {
    let case = load_case(n);
    let res = find_homography(&case.src, &case.dst)
        .unwrap_or_else(|| panic!("case {n} ({}): find_homography returned None", case.name));

    // 1. Inlier masks must be bit-equal. The RANSAC RNG is fixed-seed, so
    // this is deterministic; do NOT loosen this on failure — debug the port.
    assert_eq!(
        res.inliers, case.mask_cv,
        "case {n} ({}): inlier mask differs from cv2",
        case.name
    );

    // 2. H agreement after normalizing both so h[2][2] == 1.
    let hr = res.h;
    let hc = case.h_cv;
    assert!(hr[2][2].abs() > 1e-8 && hc[2][2].abs() > 1e-8);
    let mut max_diff = 0.0f64;
    let mut max_ref = 0.0f64;
    for r in 0..3 {
        for c in 0..3 {
            let a = hr[r][c] / hr[2][2];
            let b = hc[r][c] / hc[2][2];
            max_diff = max_diff.max((a - b).abs());
            max_ref = max_ref.max(b.abs());
        }
    }
    let rel = max_diff / max_ref;
    let tol = if case.noise_free { 1e-6 } else { 1e-4 };
    // Visible with `cargo test -- --nocapture`; handy when tightening tols.
    eprintln!("case {n} ({}): H relative diff {rel:.3e}", case.name);
    assert!(
        rel < tol,
        "case {n} ({}): H relative diff {rel:.3e} exceeds {tol:.0e}\n rust: {hr:?}\n cv2 : {hc:?}",
        case.name
    );
}

#[test]
fn case_1_clean_4pt() {
    run_case(1);
}

#[test]
fn case_2_exact_30pt() {
    run_case(2);
}

#[test]
fn case_3_noise_sigma1_50pt() {
    run_case(3);
}

#[test]
fn case_4_outliers_30pct_50pt() {
    run_case(4);
}

#[test]
fn case_5_near_degenerate_collinear() {
    run_case(5);
}

#[test]
fn case_6_centered_negative_coords() {
    run_case(6);
}
