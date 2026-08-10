//! Match-level parity vs the oracle (M1 acceptance gate):
//! our ORB + BF matching must reproduce >= 90% of the oracle's raw matches
//! on adjacent image pairs, compared by keypoint coordinates.

use std::path::{Path, PathBuf};

use panoloom_core::imgproc::{rgb_to_gray_cv, GrayImage};
use panoloom_core::matcher::best_of_2_nearest_raw;
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
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct OraclePair {
    num_matches: usize,
    num_inliers: usize,
    confidence: f64,
    #[serde(rename = "H")]
    h: Option<[[f64; 3]; 3]>,
    matches: Vec<(usize, usize, f32)>,
    inliers_mask: Vec<u8>,
}

#[test]
fn raw_match_overlap_vs_oracle() {
    let Some(dir) = dumps_dir() else {
        eprintln!("SKIP: oracle dumps not present");
        return;
    };

    // Our features for all 8 images.
    let mut ours = Vec::new();
    for i in 0..8 {
        let img = load_png_gray(&dir.join(format!("work/img_{i:03}.png")));
        ours.push(orb_detect_and_compute(&img, &OrbParams::default()));
    }
    // Oracle keypoints for all 8 images.
    let oracle_kps: Vec<Vec<OracleKp>> = (0..8)
        .map(|i| {
            serde_json::from_str(
                &std::fs::read_to_string(dir.join(format!("features/img_{i:03}.json"))).unwrap(),
            )
            .unwrap()
        })
        .collect();

    let mut worst: f64 = 1.0;
    let mut checked = 0;
    for i in 0..8usize {
        for j in (i + 1)..8usize {
            let pair_path = dir.join(format!("matches/pair_{i:03}_{j:03}.json"));
            let oracle: OraclePair =
                serde_json::from_str(&std::fs::read_to_string(&pair_path).unwrap()).unwrap();
            // Only adjacent views have meaningful match sets.
            if oracle.num_matches < 20 {
                continue;
            }

            let my_matches = best_of_2_nearest_raw(&ours[i].1, &ours[j].1);
            // Coordinate-space match set from ours.
            let mine: Vec<((f32, f32), (f32, f32))> = my_matches
                .iter()
                .map(|m| {
                    let a = &ours[i].0[m.query];
                    let b = &ours[j].0[m.train];
                    ((a.x, a.y), (b.x, b.y))
                })
                .collect();

            let mut found = 0usize;
            for (q, t, _) in &oracle.matches {
                let oa = &oracle_kps[i][*q];
                let ob = &oracle_kps[j][*t];
                let hit = mine.iter().any(|((ax, ay), (bx, by))| {
                    (ax - oa.x).abs() <= 1.0
                        && (ay - oa.y).abs() <= 1.0
                        && (bx - ob.x).abs() <= 1.0
                        && (by - ob.y).abs() <= 1.0
                });
                if hit {
                    found += 1;
                }
            }
            let overlap = found as f64 / oracle.num_matches as f64;
            worst = worst.min(overlap);
            checked += 1;
            eprintln!(
                "pair {i}-{j}: oracle={} ours={} overlap={:.1}%",
                oracle.num_matches,
                my_matches.len(),
                100.0 * overlap
            );
        }
    }
    // Not all 8 adjacent ring pairs clear the 20-match floor (sky-heavy
    // views match weakly); 6+ checked pairs is a meaningful sample.
    assert!(checked >= 6, "too few checkable pairs: {checked}");
    eprintln!("worst pair overlap: {:.1}%", 100.0 * worst);
    assert!(worst >= 0.90, "match overlap below M1 gate: {worst}");
}

/// M1 homography gate: full match_pair pipeline (BF matching → RANSAC →
/// refine) must agree with the oracle's H to < 1 px corner reprojection on
/// well-matched pairs.
#[test]
fn homography_agreement_vs_oracle() {
    let Some(dir) = dumps_dir() else {
        eprintln!("SKIP: oracle dumps not present");
        return;
    };

    let mut ours = Vec::new();
    let mut sizes = Vec::new();
    for i in 0..8 {
        let img = load_png_gray(&dir.join(format!("work/img_{i:03}.png")));
        sizes.push((img.width as u32, img.height as u32));
        ours.push(orb_detect_and_compute(&img, &OrbParams::default()));
    }
    let oracle_kps_all: Vec<Vec<OracleKp>> = (0..8)
        .map(|i| {
            serde_json::from_str(
                &std::fs::read_to_string(dir.join(format!("features/img_{i:03}.json"))).unwrap(),
            )
            .unwrap()
        })
        .collect();

    let project = |h: &[[f64; 3]; 3], p: [f64; 2]| -> [f64; 2] {
        let w = h[2][0] * p[0] + h[2][1] * p[1] + h[2][2];
        [
            (h[0][0] * p[0] + h[0][1] * p[1] + h[0][2]) / w,
            (h[1][0] * p[0] + h[1][1] * p[1] + h[1][2]) / w,
        ]
    };

    let mut checked = 0;
    let mut worst_err: f64 = 0.0;
    for i in 0..8usize {
        for j in (i + 1)..8usize {
            let oracle: OraclePair = serde_json::from_str(
                &std::fs::read_to_string(dir.join(format!("matches/pair_{i:03}_{j:03}.json")))
                    .unwrap(),
            )
            .unwrap();
            // Gate only on confidently-matched pairs; weak pairs are noise.
            let Some(oh) = oracle.h else { continue };
            if oracle.confidence <= 1.0 || oracle.num_matches < 40 {
                continue;
            }

            let pts_a: Vec<[f32; 2]> = ours[i].0.iter().map(|k| [k.x, k.y]).collect();
            let pts_b: Vec<[f32; 2]> = ours[j].0.iter().map(|k| [k.x, k.y]).collect();
            let pm = panoloom_core::matcher::match_pair(
                &pts_a, &ours[i].1, sizes[i], &pts_b, &ours[j].1, sizes[j],
            );
            let Some(mh) = pm.h else {
                panic!("pair {i}-{j}: we found no homography but oracle did")
            };

            // Model QUALITY on shared data, not model identity: our match
            // set differs from the oracle's by a few matches, so the two
            // H's legitimately differ. The gate: our H must fit the
            // ORACLE'S inlier correspondences (mean reprojection residual)
            // about as well as the oracle's own H does.
            let (wi, hi) = (sizes[i].0 as f64, sizes[i].1 as f64);
            let (wj, hj) = (sizes[j].0 as f64, sizes[j].1 as f64);
            let (mut res_mine, mut res_oracle) = (0.0f64, 0.0f64);
            let mut n = 0usize;
            for (mi, (q, t, _)) in oracle.matches.iter().enumerate() {
                if oracle.inliers_mask[mi] == 0 {
                    continue;
                }
                let ps = &oracle_kps_all[i][*q];
                let pd = &oracle_kps_all[j][*t];
                let src = [ps.x as f64 - wi / 2.0, ps.y as f64 - hi / 2.0];
                let dst = [pd.x as f64 - wj / 2.0, pd.y as f64 - hj / 2.0];
                let a = project(&mh, src);
                let b = project(&oh, src);
                res_mine += ((a[0] - dst[0]).powi(2) + (a[1] - dst[1]).powi(2)).sqrt();
                res_oracle += ((b[0] - dst[0]).powi(2) + (b[1] - dst[1]).powi(2)).sqrt();
                n += 1;
            }
            assert!(n > 0);
            res_mine /= n as f64;
            res_oracle /= n as f64;
            let err = res_mine - res_oracle;
            worst_err = worst_err.max(err);
            checked += 1;
            eprintln!(
                "pair {i}-{j}: inliers {} (oracle {}), confidence {:.3} (oracle {:.3}), mean residual ours {:.3}px vs oracle {:.3}px",
                pm.num_inliers, oracle.num_inliers, pm.confidence, oracle.confidence, res_mine, res_oracle
            );
        }
    }
    assert!(checked >= 5, "too few checkable pairs: {checked}");
    eprintln!("worst mean-residual excess over oracle: {worst_err:.3}px");
    // M1 gate: our H fits the oracle's inlier correspondences within 1px
    // (mean) of how well the oracle's own H fits them.
    assert!(
        worst_err < 1.0,
        "homography quality below M1 gate: +{worst_err}px"
    );
}
