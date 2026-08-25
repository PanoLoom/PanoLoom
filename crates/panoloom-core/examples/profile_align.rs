//! Stage-level profiling on a directory of registration-scale PNGs.
//! Usage: PANOLOOM_TIMING=1 cargo run --release --example profile_align -- \
//!            <dir-with-img_NNN.png> [priors.json]

use panoloom_core::pipeline::{align, render_preview, SourceImage};
use panoloom_core::warp::PixelImage;
use std::time::Instant;

fn load_png(path: &std::path::Path) -> PixelImage {
    let dec = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap()));
    let mut reader = dec.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    buf.truncate(info.buffer_size());
    let (w, h) = (info.width as usize, info.height as usize);
    let rgb = match info.color_type {
        png::ColorType::Rgb => buf,
        png::ColorType::Rgba => buf
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect(),
        other => panic!("unsupported color type {other:?}"),
    };
    PixelImage::new(w, h, 3, rgb)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().expect("dir argument"));
    let priors: Option<serde_json::Value> = args
        .next()
        .map(|p| serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap());

    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| {
            let p = e.unwrap().path();
            (p.extension().is_some_and(|x| x == "png")).then_some(p)
        })
        .collect();
    files.sort();

    let sources: Vec<SourceImage> = files
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let pose_prior = priors.as_ref().and_then(|j| {
                j.get(i.to_string()).map(|v| {
                    let a = v.as_array().unwrap();
                    [
                        a[0].as_f64().unwrap(),
                        a[1].as_f64().unwrap(),
                        a[2].as_f64().unwrap(),
                    ]
                })
            });
            SourceImage {
                id: i as u32,
                rgb: load_png(p),
                pose_prior,
            }
        })
        .collect();
    eprintln!("loaded {} images", sources.len());
    let _progress = panoloom_core::progress::scoped(Box::new(|s: &str| {
        // Bundle adjustment reports every LM iteration; show every 100th so
        // a non-converging run is visible without drowning the log.
        if let Some(d) = s.strip_prefix("bundle-adjust:") {
            let n: usize = d.split('/').next().unwrap_or("0").parse().unwrap_or(0);
            if n % 100 != 0 {
                return;
            }
        }
        eprintln!("[stage] {s}");
    }));

    // Alignment cache: registration is deterministic for a given input set,
    // so seam/compose work can be iterated on without paying for it again.
    // Delete the file (or set PANOLOOM_REALIGN) to force a fresh solve.
    let cache = dir.join(".alignment.json");
    let alignment = if cache.exists() && std::env::var_os("PANOLOOM_REALIGN").is_none() {
        eprintln!("align: reusing {}", cache.display());
        serde_json::from_str(&std::fs::read_to_string(&cache).unwrap()).unwrap()
    } else {
        let t = Instant::now();
        let a = align(&sources).expect("align");
        eprintln!(
            "align total: {:.0}ms ({} placed, {} dropped)",
            t.elapsed().as_secs_f64() * 1e3,
            a.images.len(),
            a.dropped.len()
        );
        std::fs::write(&cache, serde_json::to_string(&a).unwrap()).unwrap();
        a
    };

    // render_preview wants sources ordered like alignment.images.
    let by_id: std::collections::HashMap<u32, &PixelImage> =
        sources.iter().map(|s| (s.id, &s.rgb)).collect();
    let ordered: Vec<&PixelImage> = alignment.images.iter().map(|ai| by_id[&ai.id]).collect();

    let t = Instant::now();
    let p =
        render_preview(&ordered, &alignment, &vec![None; ordered.len()], 4096).expect("preview");
    eprintln!(
        "preview total: {:.0}ms ({}x{})",
        t.elapsed().as_secs_f64() * 1e3,
        p.width,
        p.height,
    );

    // Write the preview out — the point of a stitch is the picture, and
    // timings alone cannot tell you whether it is any good.
    let out = dir.join("preview.png");
    let file = std::fs::File::create(&out).unwrap();
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), p.width as u32, p.height as u32);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .unwrap()
        .write_image_data(&p.rgba)
        .unwrap();
    eprintln!("wrote {}", out.display());
}
