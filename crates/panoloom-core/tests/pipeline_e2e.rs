//! End-to-end test of the high-level pipeline module (the engine the wasm
//! API wraps): align + render_preview on the ring dumps.

use std::path::{Path, PathBuf};

use panoloom_core::pipeline::{align, render_preview, SourceImage};
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
    PixelImage::new(
        info.width as usize,
        info.height as usize,
        match info.color_type {
            png::ColorType::Rgb => 3,
            other => panic!("{other:?}"),
        },
        buf,
    )
}

#[test]
fn pipeline_align_and_preview_ring() {
    let Some(dir) = dumps_dir("ring_kloppenheim_06") else {
        eprintln!("SKIP: dumps not present");
        return;
    };
    let sources: Vec<SourceImage> = (0..8)
        .map(|i| SourceImage {
            id: 100 + i as u32,
            rgb: load_png(&dir.join(format!("work/img_{i:03}.png"))),
            pose_prior: None,
        })
        .collect();

    let alignment = align(&sources).expect("align");
    assert_eq!(alignment.images.len(), 8);
    assert!(alignment.dropped.is_empty());
    assert!(alignment.warped_image_scale > 500.0 && alignment.warped_image_scale < 1000.0);

    let srcs: Vec<&PixelImage> = alignment
        .images
        .iter()
        .map(|ai| &sources.iter().find(|s| s.id == ai.id).unwrap().rgb)
        .collect();
    let preview =
        render_preview(&srcs, &alignment, &vec![None; srcs.len()], 1024).expect("preview");
    assert!(preview.width <= 1024);
    assert_eq!(preview.rgba.len(), preview.width * preview.height * 4);

    // A 360° ring must produce coverage across the full horizontal extent.
    let mut covered_cols = 0usize;
    for x in 0..preview.width {
        let mut any = false;
        for y in 0..preview.height {
            if preview.rgba[(y * preview.width + x) * 4 + 3] != 0 {
                any = true;
                break;
            }
        }
        if any {
            covered_cols += 1;
        }
    }
    let frac = covered_cols as f64 / preview.width as f64;
    eprintln!(
        "preview {}x{}, {:.1}% columns covered",
        preview.width,
        preview.height,
        frac * 100.0
    );
    assert!(frac > 0.95, "ring should cover nearly all columns: {frac}");
}
