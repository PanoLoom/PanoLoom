//! Synthetic ground-truth validation of the control-point optimizer:
//! generate control points from KNOWN cameras + KNOWN lens distortion,
//! perturb the starting state, and require the optimizer to recover the
//! geometry (residuals to ~zero) and the lens parameters.

#![allow(clippy::needless_range_loop, clippy::manual_range_contains)]

use std::collections::HashMap;

use panoloom_core::camera::CameraParams;
use panoloom_core::cp::ControlPoint;
use panoloom_core::lens::LensParams;
use panoloom_core::optimizer::{optimize_control_points, OptimizeFlags};
use panoloom_core::pipeline::{AlignedImage, Alignment};
use panoloom_core::rng::CvRng;

const W: f64 = 800.0;
const H: f64 = 600.0;
const F_TRUE: f64 = 700.0;

fn ry(deg: f64) -> [[f64; 3]; 3] {
    let (s, c) = deg.to_radians().sin_cos();
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}

fn rx(deg: f64) -> [[f64; 3]; 3] {
    let (s, c) = deg.to_radians().sin_cos();
    [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
}

fn mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut o = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            o[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
        }
    }
    o
}

/// Projects pano direction d into camera (R, f) IDEAL pixels; None if
/// behind or outside the frame.
fn project(d: [f64; 3], r: &[[f64; 3]; 3], f: f64) -> Option<(f64, f64)> {
    // camera ray = R^T d
    let v = [
        r[0][0] * d[0] + r[1][0] * d[1] + r[2][0] * d[2],
        r[0][1] * d[0] + r[1][1] * d[1] + r[2][1] * d[2],
        r[0][2] * d[0] + r[1][2] * d[1] + r[2][2] * d[2],
    ];
    if v[2] <= 0.1 {
        return None;
    }
    let x = W / 2.0 + f * v[0] / v[2];
    let y = H / 2.0 + f * v[1] / v[2];
    (x >= 5.0 && x < W - 5.0 && y >= 5.0 && y < H - 5.0).then_some((x, y))
}

#[test]
fn recovers_rotations_focal_and_distortion() {
    let lens_true = LensParams {
        a: 0.004,
        b: -0.025,
        c: 0.012,
        d: 0.0,
        e: 0.0,
    };
    // Five cameras: a 4-shot yaw ring plus one pitched up (overlap-rich).
    let rots_true: Vec<[[f64; 3]; 3]> = vec![
        ry(0.0),
        ry(35.0),
        ry(70.0),
        ry(105.0),
        mul(&ry(35.0), &rx(-30.0)),
    ];

    // Control points: random pano directions seen by image pairs, with the
    // TRUE distortion applied (CPs live on the distorted images).
    let mut rng = CvRng::new(u64::MAX);
    let mut cps = Vec::new();
    let mut id = 0;
    for i in 0..rots_true.len() {
        for j in (i + 1)..rots_true.len() {
            let mut count = 0;
            for _ in 0..4000 {
                if count >= 24 {
                    break;
                }
                let yaw = (rng.uniform_int(0, 3600) as f64 / 10.0).to_radians();
                let pitch = ((rng.uniform_int(0, 1200) as f64 / 10.0) - 60.0).to_radians();
                let d = [
                    pitch.cos() * yaw.sin(),
                    -pitch.sin(),
                    pitch.cos() * yaw.cos(),
                ];
                let (Some(pa), Some(pb)) = (
                    project(d, &rots_true[i], F_TRUE),
                    project(d, &rots_true[j], F_TRUE),
                ) else {
                    continue;
                };
                let (xa, ya) = lens_true.distort(pa.0, pa.1, W / 2.0, H / 2.0, W, H);
                let (xb, yb) = lens_true.distort(pb.0, pb.1, W / 2.0, H / 2.0, W, H);
                cps.push(ControlPoint {
                    id,
                    img_a: i as u32,
                    img_b: j as u32,
                    x_a: xa,
                    y_a: ya,
                    x_b: xb,
                    y_b: yb,
                    error_px: None,
                });
                id += 1;
                count += 1;
            }
        }
    }
    assert!(cps.len() > 100, "need overlap, got {} cps", cps.len());

    // Perturbed starting state: rotations off by up to ~1.5°, focal off by
    // 2%, lens zeroed.
    let mut images = Vec::new();
    for (i, rt) in rots_true.iter().enumerate() {
        let wobble = mul(
            &ry(0.9 * ((i as f64 * 37.0).sin())),
            &rx(0.8 * ((i as f64 * 23.0).cos())),
        );
        let rp = mul(&wobble, rt);
        let mut cam = CameraParams {
            focal: F_TRUE * 1.02,
            aspect: 1.0,
            ppx: W / 2.0,
            ppy: H / 2.0,
            r: [[0.0; 3]; 3],
        };
        for a in 0..3 {
            for b in 0..3 {
                cam.r[a][b] = rp[a][b] as f32;
            }
        }
        images.push(AlignedImage {
            id: i as u32,
            camera: cam,
            rescued: false,
        });
    }
    let mut alignment = Alignment {
        images,
        dropped: vec![],
        warped_image_scale: F_TRUE * 1.02,
        lens: LensParams::default(),
    };

    let dims: HashMap<u32, (u32, u32)> = (0..5).map(|i| (i as u32, (W as u32, H as u32))).collect();
    let report = optimize_control_points(
        &mut alignment,
        &cps,
        &dims,
        &OptimizeFlags {
            focal: true,
            distortion: true,
            shift: false,
        },
    )
    .expect("optimize");

    eprintln!(
        "rms {:.3}px -> {:.4}px in {} iters; lens a={:.5} b={:.5} c={:.5}; focal {:.2}",
        report.rms_px_before,
        report.rms_px,
        report.iterations,
        report.lens.a,
        report.lens.b,
        report.lens.c,
        alignment.images[0].camera.focal,
    );
    assert!(
        report.rms_px_before > 1.0,
        "perturbation should start clearly misaligned"
    );
    assert!(report.rms_px < 0.05, "rms {} px too high", report.rms_px);
    assert!(
        (report.lens.a - lens_true.a).abs() < 2e-3,
        "a={}",
        report.lens.a
    );
    assert!(
        (report.lens.b - lens_true.b).abs() < 2e-3,
        "b={}",
        report.lens.b
    );
    assert!(
        (report.lens.c - lens_true.c).abs() < 2e-3,
        "c={}",
        report.lens.c
    );
    let f = alignment.images[0].camera.focal;
    assert!((f - F_TRUE).abs() / F_TRUE < 0.005, "focal {}", f);
}
