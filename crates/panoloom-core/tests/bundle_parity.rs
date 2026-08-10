//! Bundle-adjustment parity vs the OpenCV oracle.
//!
//! Two layers:
//! 1. Primitive fixtures (tools/reference/gen_bundle_fixtures.py): every
//!    numeric kernel the BA runs on is checked bit-for-bit against cv2 —
//!    Rodrigues, the 3x3 float SVD/orthonormalization chain, norms, gemm
//!    and solve. Fixtures marked `lapack: true` were produced through Apple
//!    Accelerate (`dgesdd` for SVD systems >= 25 rows, `cblas_dgemm` for
//!    gemm left operands >= 100 rows — core/src/hal_internal.cpp) which is
//!    closed-source; those are checked to tight tolerances instead.
//! 2. Full `bundle_adjust_ray` runs on the dumped datasets, compared to
//!    cameras_ba.json (focal relative diff + rotation angular diff).
//!
//! All tests skip gracefully when the dumps/fixtures are absent.

#![allow(clippy::needless_range_loop)] // matrix loops mirror the C++

use std::path::{Path, PathBuf};

use panoloom_core::bundle::{bundle_adjust_ray, cvnum};
use panoloom_core::camera::CameraParams;
use panoloom_core::estimation::{FeatureSet, MatchGraph};
use panoloom_core::matcher::{PairMatches, RawMatch};

fn reference_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/reference")
}

fn dumps_dir(set: &str) -> Option<PathBuf> {
    let p = reference_dir().join("dumps").join(set);
    p.exists().then_some(p)
}

fn fixture(rel: &str) -> Option<serde_json::Value> {
    let p = reference_dir().join("fixtures").join(rel);
    if !p.exists() {
        eprintln!("SKIP: fixture {rel} not present (run tools/reference/gen_bundle_fixtures.py)");
        return None;
    }
    Some(serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap())
}

// --- bit-pattern decoding (fixtures store IEEE-754 bits as integers) ---

fn f64_of(v: &serde_json::Value) -> f64 {
    f64::from_bits(v.as_u64().unwrap())
}

fn f32_of(v: &serde_json::Value) -> f32 {
    f32::from_bits(u32::try_from(v.as_u64().unwrap()).unwrap())
}

fn vec_f64(v: &serde_json::Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(f64_of).collect()
}

fn vec_f32(v: &serde_json::Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(f32_of).collect()
}

fn mat3_f64(v: &serde_json::Value) -> [[f64; 3]; 3] {
    let rows = v.as_array().unwrap();
    std::array::from_fn(|i| {
        let r = rows[i].as_array().unwrap();
        std::array::from_fn(|j| f64_of(&r[j]))
    })
}

fn mat3_f32(v: &serde_json::Value) -> [[f32; 3]; 3] {
    let rows = v.as_array().unwrap();
    std::array::from_fn(|i| {
        let r = rows[i].as_array().unwrap();
        std::array::from_fn(|j| f32_of(&r[j]))
    })
}

fn matn_f64(v: &serde_json::Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .flat_map(|row| row.as_array().unwrap().iter().map(f64_of))
        .collect()
}

// ---------------------------------------------------------------------
// primitive parity
// ---------------------------------------------------------------------

#[test]
fn rodrigues_m2v_f32_bit_exact() {
    let Some(cases) = fixture("rodrigues/m2v_f32.json") else {
        return;
    };
    for (n, case) in cases.as_array().unwrap().iter().enumerate() {
        let r = mat3_f32(&case["r"]);
        let want = vec_f32(&case["rvec"]);
        let got = cvnum::rodrigues_m2v_f32(&r);
        assert_eq!(
            got.map(f32::to_bits),
            [want[0], want[1], want[2]].map(f32::to_bits),
            "m2v case {n}: got {got:?} want {want:?}"
        );
    }
}

#[test]
fn rodrigues_v2m_f64_bit_exact() {
    let Some(cases) = fixture("rodrigues/v2m_f64.json") else {
        return;
    };
    for (n, case) in cases.as_array().unwrap().iter().enumerate() {
        let rvec = vec_f64(&case["rvec"]);
        let want = mat3_f64(&case["r"]);
        let got = cvnum::rodrigues_v2m(&[rvec[0], rvec[1], rvec[2]]);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(
                    got[i][j].to_bits(),
                    want[i][j].to_bits(),
                    "v2m case {n} ({i},{j}): got {} want {}",
                    got[i][j],
                    want[i][j]
                );
            }
        }
    }
}

#[test]
fn svd3x3_and_setup_chain_bit_exact() {
    let Some(cases) = fixture("bundle/svd3x3_f32.json") else {
        return;
    };
    for (n, case) in cases.as_array().unwrap().iter().enumerate() {
        let r = mat3_f32(&case["r"]);
        let (w, u, vt) = cvnum::svd3x3_f32_full(&r);
        let want_w = vec_f32(&case["w"]);
        let want_u = mat3_f32(&case["u"]);
        let want_vt = mat3_f32(&case["vt"]);
        for i in 0..3 {
            assert_eq!(w[i].to_bits(), want_w[i].to_bits(), "svd case {n} w[{i}]");
            for j in 0..3 {
                assert_eq!(
                    u[i][j].to_bits(),
                    want_u[i][j].to_bits(),
                    "svd case {n} u({i},{j}): got {} want {}",
                    u[i][j],
                    want_u[i][j]
                );
                assert_eq!(
                    vt[i][j].to_bits(),
                    want_vt[i][j].to_bits(),
                    "svd case {n} vt({i},{j})"
                );
            }
        }

        // Orthonormalization chain: u*vt (float small-path gemm), det sign.
        let mut ortho = cvnum::gemm3x3_f32(&u, &vt);
        let det = cvnum::det3x3_f32(&ortho);
        assert_eq!(
            det.to_bits(),
            f64_of(&case["det"]).to_bits(),
            "svd case {n} det"
        );
        let flipped = case["flipped"].as_bool().unwrap();
        assert_eq!(det < 0.0, flipped, "svd case {n} sign");
        if flipped {
            for row in ortho.iter_mut() {
                for v in row.iter_mut() {
                    *v *= -1.0;
                }
            }
        }
        let want_ortho = mat3_f32(&case["ortho"]);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(
                    ortho[i][j].to_bits(),
                    want_ortho[i][j].to_bits(),
                    "svd case {n} ortho({i},{j})"
                );
            }
        }
        let rvec = cvnum::rodrigues_m2v_f32(&ortho);
        let want_rvec = vec_f32(&case["rvec"]);
        for k in 0..3 {
            assert_eq!(
                rvec[k].to_bits(),
                want_rvec[k].to_bits(),
                "svd case {n} rvec[{k}]: got {} want {}",
                rvec[k],
                want_rvec[k]
            );
        }
    }
}

#[test]
fn norms_f64_bit_exact() {
    let Some(fx) = fixture("bundle/norms_f64.json") else {
        return;
    };
    for (n, case) in fx["l2"].as_array().unwrap().iter().enumerate() {
        let x = vec_f64(&case["x"]);
        let want = f64_of(&case["norm"]);
        let got = cvnum::norm_l2(&x);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "norm case {n} (len {}): got {got:e} want {want:e}",
            x.len()
        );
    }
    for (n, case) in fx["rel_l2"].as_array().unwrap().iter().enumerate() {
        let a = vec_f64(&case["a"]);
        let b = vec_f64(&case["b"]);
        let want = f64_of(&case["norm"]);
        let got = cvnum::norm_rel_l2(&a, &b);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "relative norm case {n} (len {}): got {got:e} want {want:e}",
            a.len()
        );
    }
}

#[test]
fn gemm3x3_and_invert_bit_exact() {
    let Some(fx) = fixture("bundle/gemm3x3.json") else {
        return;
    };
    for (n, case) in fx["f64"].as_array().unwrap().iter().enumerate() {
        let d = cvnum::gemm3x3_f64(&mat3_f64(&case["a"]), &mat3_f64(&case["b"]));
        let want = mat3_f64(&case["d"]);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(d[i][j].to_bits(), want[i][j].to_bits(), "gemm f64 case {n}");
            }
        }
    }
    for (n, case) in fx["f32"].as_array().unwrap().iter().enumerate() {
        let d = cvnum::gemm3x3_f32(&mat3_f32(&case["a"]), &mat3_f32(&case["b"]));
        let want = mat3_f32(&case["d"]);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(d[i][j].to_bits(), want[i][j].to_bits(), "gemm f32 case {n}");
            }
        }
    }
    for (n, case) in fx["inv_f64"].as_array().unwrap().iter().enumerate() {
        let d = cvnum::invert3x3_lu_f64(&mat3_f64(&case["a"]));
        let want = mat3_f64(&case["inv"]);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(d[i][j].to_bits(), want[i][j].to_bits(), "inv f64 case {n}");
            }
        }
    }
    for (n, case) in fx["inv_f32"].as_array().unwrap().iter().enumerate() {
        let d = cvnum::invert3x3_lu_f32(&mat3_f32(&case["a"]));
        let want = mat3_f32(&case["inv"]);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(d[i][j].to_bits(), want[i][j].to_bits(), "inv f32 case {n}");
            }
        }
    }
    for (n, case) in fx["det_f32"].as_array().unwrap().iter().enumerate() {
        let d = cvnum::det3x3_f32(&mat3_f32(&case["a"]));
        assert_eq!(
            d.to_bits(),
            f64_of(&case["det"]).to_bits(),
            "det f32 case {n}"
        );
    }
}

#[test]
fn gemm_jt_parity() {
    let Some(cases) = fixture("bundle/gemm_jt.json") else {
        return;
    };
    for case in cases.as_array().unwrap() {
        let rows = case["rows"].as_u64().unwrap() as usize;
        let cols = case["cols"].as_u64().unwrap() as usize;
        let lapack = case["lapack"].as_bool().unwrap();
        let j = matn_f64(&case["j"]);
        let e = vec_f64(&case["e"]);
        let want_jtj = matn_f64(&case["jtj"]);
        let want_jte = vec_f64(&case["jte"]);

        let mut jtj = vec![0.0; cols * cols];
        let mut jte = vec![0.0; cols];
        cvnum::gemm_jtj(&j, rows, cols, &mut jtj);
        cvnum::gemm_jterr(&j, rows, cols, &e, &mut jte);

        if !lapack {
            for (idx, (g, w)) in jtj.iter().zip(&want_jtj).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "JtJ ({rows}x{cols}) elem {idx}: got {g:e} want {w:e}"
                );
            }
            for (idx, (g, w)) in jte.iter().zip(&want_jte).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "JtErr ({rows}x{cols}) elem {idx}: got {g:e} want {w:e}"
                );
            }
        } else {
            // Accelerate cblas_dgemm on the oracle side: expect agreement to
            // a few ulps, not bits.
            let mut worst = 0.0f64;
            for (g, w) in jtj.iter().chain(&jte).zip(want_jtj.iter().chain(&want_jte)) {
                let scale = w.abs().max(1e-30);
                worst = worst.max((g - w).abs() / scale);
            }
            eprintln!("gemm {rows}x{cols} (Accelerate oracle): worst rel diff {worst:.2e}");
            assert!(worst < 1e-12, "gemm {rows}x{cols} diverges: {worst:e}");
        }
    }
}

#[test]
fn solve_svd_parity() {
    let Some(cases) = fixture("bundle/solve_svd.json") else {
        return;
    };
    for case in cases.as_array().unwrap() {
        let n = case["n"].as_u64().unwrap() as usize;
        let lapack = case["lapack"].as_bool().unwrap();
        let a = matn_f64(&case["a"]);
        let b = vec_f64(&case["b"]);
        let want = vec_f64(&case["x"]);
        let mut x = vec![0.0; n];
        cvnum::solve_svd(&a, &b, &mut x, n);
        if !lapack {
            for (idx, (g, w)) in x.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "solve n={n} x[{idx}]: got {g:e} want {w:e}"
                );
            }
        } else {
            let mut worst = 0.0f64;
            for (g, w) in x.iter().zip(&want) {
                worst = worst.max((g - w).abs() / w.abs().max(1e-30));
            }
            eprintln!("solve n={n} (Accelerate dgesdd oracle): worst rel diff {worst:.2e}");
            assert!(worst < 1e-9, "solve n={n} diverges: {worst:e}");
        }
    }
}

// ---------------------------------------------------------------------
// full bundle adjustment vs the dumps
// ---------------------------------------------------------------------

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
    aspect: f64,
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

fn load_cameras(path: &Path) -> Vec<OracleCamera> {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn to_camera_params(o: &OracleCamera) -> CameraParams {
    let mut r = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = o.r[i][j] as f32;
        }
    }
    CameraParams {
        focal: o.focal,
        aspect: o.aspect,
        ppx: o.ppx,
        ppy: o.ppy,
        r,
    }
}

fn mat3_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
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

fn transpose(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = m[c][r];
        }
    }
    out
}

/// Angle of q = a·bᵀ in degrees. For near-identity q the acos-of-trace
/// formula bottoms out at ~sqrt(eps) resolution (0.018° on f32-stored
/// rotations), so small angles use the skew part instead: |axis| = 2·sin(θ).
fn rotation_angle_deg(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> f64 {
    let q = mat3_mul(a, &transpose(b));
    let tr = q[0][0] + q[1][1] + q[2][2];
    let axis = [q[2][1] - q[1][2], q[0][2] - q[2][0], q[1][0] - q[0][1]];
    let sin2 = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt() * 0.5;
    if tr > 3.0 - 1e-4 {
        sin2.clamp(-1.0, 1.0).asin().to_degrees()
    } else {
        (((tr - 1.0) / 2.0).clamp(-1.0, 1.0)).acos().to_degrees()
    }
}

fn r64(c: &CameraParams) -> [[f64; 3]; 3] {
    let mut m = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = c.r[i][j] as f64;
        }
    }
    m
}

fn run_bundle_parity(set: &str) {
    let Some(dir) = dumps_dir(set) else {
        eprintln!("SKIP {set}: dumps not present");
        return;
    };
    let (features, graph) = load_oracle_graph(&dir);
    let initial = load_cameras(&dir.join("cameras_initial.json"));
    let oracle = load_cameras(&dir.join("cameras_ba.json"));
    assert_eq!(initial.len(), features.len());

    let mut cameras: Vec<CameraParams> = initial.iter().map(to_camera_params).collect();
    let ok = bundle_adjust_ray(&features, &graph, &mut cameras);
    assert!(ok, "{set}: bundle adjustment failed");
    assert_eq!(cameras.len(), oracle.len());

    // ppx/ppy/aspect are untouched by Ray.
    for (c, o) in cameras.iter().zip(&oracle) {
        assert_eq!(c.ppx, o.ppx);
        assert_eq!(c.ppy, o.ppy);
        assert_eq!(c.aspect, o.aspect);
    }

    let mut worst_focal = 0.0f64;
    for (c, o) in cameras.iter().zip(&oracle) {
        worst_focal = worst_focal.max(((c.focal - o.focal) / o.focal).abs());
    }

    // Absolute rotation diff. If the spanning-tree center differed from
    // OpenCV's (unstable std::sort ties), this shows up as a CONSTANT offset
    // across all cameras — fall back to relative rotations in that case.
    let mut worst_abs = 0.0f64;
    for (c, o) in cameras.iter().zip(&oracle) {
        worst_abs = worst_abs.max(rotation_angle_deg(&r64(c), &o.r));
    }
    let mut worst_rel = 0.0f64;
    for i in 1..cameras.len() {
        let ours = mat3_mul(&r64(&cameras[i]), &transpose(&r64(&cameras[0])));
        let theirs = mat3_mul(&oracle[i].r, &transpose(&oracle[0].r));
        worst_rel = worst_rel.max(rotation_angle_deg(&ours, &theirs));
    }

    eprintln!(
        "{set}: BA worst focal rel diff {worst_focal:.3e}, worst rotation diff {worst_abs:.3e}° \
         (relative-to-cam0: {worst_rel:.3e}°)"
    );

    assert!(worst_focal < 1e-4, "{set}: focal diverges: {worst_focal:e}");
    if worst_abs >= 0.01 {
        // Constant-offset case: all cameras must still agree relative to
        // each other far below the absolute threshold.
        eprintln!(
            "{set}: absolute rotations differ by a global offset (different \
             spanning-tree center); comparing relative rotations"
        );
        assert!(
            worst_rel < 0.01,
            "{set}: relative rotations diverge: {worst_rel}°"
        );
    }

    // Tightened bounds: measured parity is ~6e-11 relative focal and
    // ~5e-6 deg rotation (the f32 rotation-storage noise floor). The
    // residual comes from the oracle wheel delegating its big gemm/SVD to
    // Apple Accelerate (see bundle.rs header) — irreducible portably.
    assert!(
        worst_focal < 1e-9,
        "{set}: focal parity regressed: {worst_focal:e}"
    );
    assert!(
        worst_rel.min(worst_abs) < 1e-4,
        "{set}: rotation parity regressed: {worst_abs:e}° / {worst_rel:e}°"
    );
}

#[test]
fn bundle_adjust_matches_cameras_ba_ring() {
    run_bundle_parity("ring_kloppenheim_06");
}

#[test]
fn bundle_adjust_matches_cameras_ba_sphere() {
    run_bundle_parity("sphere_kloppenheim_06");
}
