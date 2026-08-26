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

fn ring_sources(dir: &Path) -> Vec<SourceImage> {
    (0..8)
        .map(|i| SourceImage {
            id: 100 + i as u32,
            rgb: load_png(&dir.join(format!("work/img_{i:03}.png"))),
            pose_prior: None,
        })
        .collect()
}

/// The wasm worker drives its progress UI off these labels, so a rename or a
/// dropped `stage_timed!` call must fail here rather than silently leaving
/// the browser on an opaque spinner.
#[test]
fn align_and_preview_report_their_stages() {
    let Some(dir) = dumps_dir("ring_kloppenheim_06") else {
        eprintln!("SKIP: dumps not present");
        return;
    };
    let sources = ring_sources(&dir);

    let seen: std::rc::Rc<std::cell::RefCell<Vec<String>>> = Default::default();
    let sink = std::rc::Rc::clone(&seen);
    let guard =
        panoloom_core::progress::scoped(Box::new(move |s| sink.borrow_mut().push(s.to_string())));

    let alignment = align(&sources).expect("align");
    let srcs: Vec<&PixelImage> = alignment
        .images
        .iter()
        .map(|ai| &sources.iter().find(|s| s.id == ai.id).unwrap().rgb)
        .collect();
    render_preview(&srcs, &alignment, &vec![None; srcs.len()], 512, None).expect("preview");
    drop(guard);

    let got = seen.borrow();
    for expected in [
        "orb-detect",
        "match-pairs",
        "estimate",
        "bundle-adjust",
        "seam-stage",
        "graph-cut-seams",
        "blend",
    ] {
        assert!(
            got.iter().any(|s| s == expected),
            "missing {expected}: {got:?}"
        );
    }
    // Ordering matters: the UI shows the latest label.
    let pos = |l: &str| got.iter().position(|s| s == l).unwrap();
    assert!(pos("orb-detect") < pos("bundle-adjust"));
    assert!(pos("bundle-adjust") < pos("blend"));
}

#[test]
fn pipeline_align_and_preview_ring() {
    let Some(dir) = dumps_dir("ring_kloppenheim_06") else {
        eprintln!("SKIP: dumps not present");
        return;
    };
    let sources = ring_sources(&dir);

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
        render_preview(&srcs, &alignment, &vec![None; srcs.len()], 1024, None).expect("preview");
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

/// Reusing a `SeamStage` must change nothing about the result.
///
/// The preview and every export both need it, and it costs minutes on a
/// large set — ~19 of the 21 minutes a 137-shot export was taking. Sharing
/// it is only safe if it is genuinely the same computation, so this pins
/// byte-equality rather than trusting that it is.
#[test]
fn a_reused_seam_stage_gives_an_identical_preview() {
    let Some(dir) = dumps_dir("ring_kloppenheim_06") else {
        eprintln!("SKIP: dumps not present");
        return;
    };
    let sources = ring_sources(&dir);
    let alignment = align(&sources).expect("align");
    let srcs: Vec<&PixelImage> = alignment
        .images
        .iter()
        .map(|ai| &sources.iter().find(|s| s.id == ai.id).unwrap().rgb)
        .collect();
    let masks = vec![None; srcs.len()];

    let fresh = render_preview(&srcs, &alignment, &masks, 768, None).expect("preview");
    let stage = panoloom_core::pipeline::seam_stage(&srcs, &alignment, &masks);
    let reused = render_preview(&srcs, &alignment, &masks, 768, Some(&stage)).expect("preview");

    assert_eq!((fresh.width, fresh.height), (reused.width, reused.height));
    assert_eq!(
        fresh.rgba, reused.rgba,
        "a reused seam stage changed the panorama"
    );

    // Reusing it twice must also be stable — the stage is borrowed, not
    // consumed, and must not be mutated in passing.
    let again = render_preview(&srcs, &alignment, &masks, 768, Some(&stage)).expect("preview");
    assert_eq!(fresh.rgba, again.rgba, "reuse is not idempotent");
}
