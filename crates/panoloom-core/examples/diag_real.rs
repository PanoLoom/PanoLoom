//! Diagnostic for real-world sets: run align at several ORB feature budgets,
//! report drops and weak links. Usage: diag_real <dir-of-work-scale-pngs>

#![allow(clippy::needless_range_loop)]

use std::path::Path;

use panoloom_core::estimation::{leave_biggest_component, MatchGraph};
use panoloom_core::imgproc::rgb_to_gray_cv;
use panoloom_core::matcher::match_pair;
use panoloom_core::orb::{orb_detect_and_compute, OrbParams};
use panoloom_core::pipeline::{align, SourceImage};
use panoloom_core::warp::PixelImage;

fn load_png(path: &Path) -> PixelImage {
    let decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap()));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    buf.truncate(info.buffer_size());
    PixelImage::new(info.width as usize, info.height as usize, 3, buf)
}

fn main() {
    let dir = std::env::args().nth(1).expect("dir argument");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "png"))
        .collect();
    paths.sort();
    let images: Vec<PixelImage> = paths.iter().map(|p| load_png(p)).collect();
    let n = images.len();
    println!("{n} images {}x{}", images[0].width, images[0].height);

    for nfeatures in [500usize, 1500, 3000] {
        let params = OrbParams {
            nfeatures,
            ..Default::default()
        };
        let mut pts = Vec::new();
        let mut descs = Vec::new();
        let mut sizes = Vec::new();
        let t0 = std::time::Instant::now();
        for im in &images {
            let gray = rgb_to_gray_cv(&im.data, im.width, im.height);
            let (kps, d) = orb_detect_and_compute(&gray, &params);
            pts.push(kps.iter().map(|k| [k.x, k.y]).collect::<Vec<_>>());
            descs.push(d);
            sizes.push((im.width as u32, im.height as u32));
        }
        let t_feat = t0.elapsed();
        let t0 = std::time::Instant::now();
        let mut upper = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                upper.push((
                    (i, j),
                    match_pair(&pts[i], &descs[i], sizes[i], &pts[j], &descs[j], sizes[j]),
                ));
            }
        }
        let t_match = t0.elapsed();
        let graph = MatchGraph::from_upper_triangle(n, upper);
        let kept = leave_biggest_component(&graph, 1.0);
        let dropped: Vec<usize> = (0..n).filter(|i| !kept.contains(i)).collect();
        let strong = (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .filter(|&(i, j)| graph.at(i, j).confidence > 1.0)
            .count();
        println!(
            "nfeatures={nfeatures:<5} kept {}/{n}  strong pairs {strong}  dropped {dropped:?}  (features {:.1}s match {:.1}s)",
            kept.len(),
            t_feat.as_secs_f64(),
            t_match.as_secs_f64(),
        );
        // Weak links: for dropped images show their best confidence.
        for &d in &dropped {
            let best = (0..n)
                .filter(|&j| j != d)
                .map(|j| (graph.at(d.min(j), d.max(j)).confidence, j))
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                .unwrap();
            println!("   img {d}: best conf {:.2} with img {}", best.0, best.1);
        }
    }

    // Full align, optionally with pose priors from a JSON file
    // ({"idx": [yaw, pitch, roll], ...}).
    let priors: std::collections::HashMap<usize, [f64; 3]> = std::env::args()
        .nth(2)
        .map(|p| {
            let v: std::collections::HashMap<String, [f64; 3]> =
                serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
            v.into_iter()
                .map(|(k, v)| (k.parse().unwrap(), v))
                .collect()
        })
        .unwrap_or_default();
    let sources: Vec<SourceImage> = images
        .iter()
        .enumerate()
        .map(|(i, im)| SourceImage {
            id: i as u32,
            rgb: im.clone(),
            pose_prior: priors.get(&i).copied(),
        })
        .collect();
    // Convention probe: which (yaw, pitch, roll) sign combination makes the
    // priors consistent with the feature-solved rotations?
    if !priors.is_empty() {
        if let Ok(a) = align(
            &sources
                .iter()
                .map(|s| SourceImage {
                    id: s.id,
                    rgb: s.rgb.clone(),
                    pose_prior: None,
                })
                .collect::<Vec<_>>(),
        ) {
            for (sy, sp, sr) in [
                (1.0, 1.0, 1.0),
                (-1.0, 1.0, 1.0),
                (1.0, -1.0, 1.0),
                (-1.0, -1.0, 1.0),
                (1.0, 1.0, -1.0),
                (-1.0, 1.0, -1.0),
                (1.0, -1.0, -1.0),
                (-1.0, -1.0, -1.0),
            ] {
                let probe_sources: Vec<SourceImage> = sources
                    .iter()
                    .map(|s| SourceImage {
                        id: s.id,
                        rgb: PixelImage::new(2, 2, 3, vec![0; 12]),
                        pose_prior: s.pose_prior.map(|p| [p[0] * sy, p[1] * sp, p[2] * sr]),
                    })
                    .collect();
                if let Some(med) =
                    panoloom_core::pipeline::debug_prior_fit_residual(&probe_sources, &a.images)
                {
                    println!(
                        "convention (y{sy:+.0} p{sp:+.0} r{sr:+.0}): median residual {med:.2}°"
                    );
                }
            }
        }
    }

    match align(&sources) {
        Ok(a) => {
            let rescued: Vec<u32> = a
                .images
                .iter()
                .filter(|x| x.rescued)
                .map(|x| x.id)
                .collect();
            println!(
                "align: {} placed ({} rescued: {:?}), {} dropped {:?}, warped scale {:.1}",
                a.images.len(),
                rescued.len(),
                rescued,
                a.dropped.len(),
                a.dropped,
                a.warped_image_scale
            );
            // Render a preview PNG for eyeballing.
            let srcs: Vec<&PixelImage> = a
                .images
                .iter()
                .map(|ai| &sources.iter().find(|s| s.id == ai.id).unwrap().rgb)
                .collect();
            let p = panoloom_core::pipeline::render_preview(
                &srcs,
                &a,
                &vec![None; srcs.len()],
                4096,
                None,
            )
            .unwrap();
            let out = std::path::Path::new(&dir).join("../diag_preview.png");
            let file = std::fs::File::create(&out).unwrap();
            let mut enc = png::Encoder::new(
                std::io::BufWriter::new(file),
                p.width as u32,
                p.height as u32,
            );
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            enc.write_header()
                .unwrap()
                .write_image_data(&p.rgba)
                .unwrap();
            println!("preview {}x{} -> {}", p.width, p.height, out.display());
        }
        Err(e) => println!("align failed: {e}"),
    }
}
