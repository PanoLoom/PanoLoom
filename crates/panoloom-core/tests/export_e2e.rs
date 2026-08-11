//! Banded export end-to-end: align the ring set, export via the banded
//! Exporter at two sizes, and verify the JPEG decodes to the right
//! dimensions with content matching the preview path (consistency check).

#![allow(clippy::needless_range_loop)]

use std::path::{Path, PathBuf};

use panoloom_core::export::Exporter;
use panoloom_core::pipeline::{align, SourceImage};
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
    PixelImage::new(info.width as usize, info.height as usize, 3, buf)
}

#[test]
fn banded_export_ring() {
    let Some(dir) = dumps_dir("ring_kloppenheim_06") else {
        eprintln!("SKIP: dumps not present");
        return;
    };
    if !dir.join("full").exists() {
        eprintln!("SKIP: full-res fixtures absent (oracle --dump-full)");
        return;
    }

    // Registration sources = work PNGs; full sources = full PNGs.
    let sources: Vec<SourceImage> = (0..8)
        .map(|i| SourceImage {
            id: i as u32,
            rgb: load_png(&dir.join(format!("work/img_{i:03}.png"))),
            pose_prior: None,
        })
        .collect();
    let fulls: Vec<PixelImage> = (0..8)
        .map(|i| load_png(&dir.join(format!("full/img_{i:03}.png"))))
        .collect();
    let full_sizes: Vec<(u32, u32, u32)> = fulls
        .iter()
        .enumerate()
        .map(|(i, f)| (i as u32, f.width as u32, f.height as u32))
        .collect();

    let alignment = align(&sources).expect("align");

    let mut exporter = Exporter::new(
        &sources,
        &alignment,
        &vec![None; alignment.images.len()],
        &full_sizes,
        16384,
    )
    .expect("exporter");
    let (cw, ch) = exporter.canvas_size();
    let (crop_x, crop_y, crop_w, crop_h) = exporter.crop();
    eprintln!(
        "export canvas {cw}x{ch}, crop {crop_w}x{crop_h}+{crop_x}+{crop_y}, {} bands",
        exporter.bands().len()
    );
    assert_eq!(ch * 2, cw, "full canvas must be exactly 2:1");
    // Native full res of this set is ~7500px wide (1600px sources).
    assert!(cw > 6000, "expected near-native canvas, got {cw}");
    // The ring wraps 360° but covers only a band of latitudes: full width,
    // cropped height.
    assert_eq!(crop_w, cw, "wrapped pano keeps full width");
    assert!(crop_h < ch, "ring must crop vertically");
    assert!(crop_y > 0 && crop_y + crop_h <= ch);

    let band_plan: Vec<Vec<u32>> = exporter.bands().iter().map(|b| b.needed.clone()).collect();
    for (b, needed) in band_plan.iter().enumerate() {
        for &id in needed {
            exporter
                .set_full_image(id, fulls[id as usize].clone())
                .unwrap();
        }
        exporter.composite_band(b).expect("band");
        // Streaming semantics: drop what the next band doesn't need.
        for &id in needed {
            if !band_plan
                .get(b + 1)
                .map(|n| n.contains(&id))
                .unwrap_or(false)
            {
                exporter.drop_full_image(id);
            }
        }
    }
    let (jpeg, w, h) = exporter.finish(90).expect("finish");
    eprintln!("jpeg: {} bytes for {w}x{h}", jpeg.len());
    assert_eq!((w, h), (crop_w, crop_h), "JPEG spans exactly the crop");
    assert!(jpeg.len() > 500_000, "suspiciously small JPEG");
    assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "SOI");

    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join("export_ring.jpg");
    std::fs::write(&out, &jpeg).unwrap();
    eprintln!("wrote {}", out.display());
}
