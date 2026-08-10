//! Blending — port of `cv::detail::Blender` / `FeatherBlender`
//! (blenders.cpp). MultiBandBlender follows in M4.

#![allow(clippy::needless_range_loop)]

use crate::imgproc::GrayImage;

const WEIGHT_EPS: f32 = 1e-5;

/// `distanceTransform(mask, DIST_L1, 3)`: two-pass 3x3 chamfer with
/// a = 1, b = 2 (exact for L1), f32 output.
pub fn distance_transform_l1(mask: &GrayImage) -> Vec<f32> {
    let (w, h) = (mask.width, mask.height);
    const INF: f32 = 1e15;
    let mut dist = vec![0.0f32; w * h];
    for i in 0..w * h {
        dist[i] = if mask.data[i] == 0 { 0.0 } else { INF };
    }
    let (a, b) = (1.0f32, 2.0f32);
    // Forward pass.
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if dist[idx] == 0.0 {
                continue;
            }
            let mut d = dist[idx];
            if x > 0 {
                d = d.min(dist[idx - 1] + a);
            }
            if y > 0 {
                d = d.min(dist[idx - w] + a);
                if x > 0 {
                    d = d.min(dist[idx - w - 1] + b);
                }
                if x + 1 < w {
                    d = d.min(dist[idx - w + 1] + b);
                }
            }
            dist[idx] = d;
        }
    }
    // Backward pass.
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let idx = y * w + x;
            if dist[idx] == 0.0 {
                continue;
            }
            let mut d = dist[idx];
            if x + 1 < w {
                d = d.min(dist[idx + 1] + a);
            }
            if y + 1 < h {
                d = d.min(dist[idx + w] + a);
                if x + 1 < w {
                    d = d.min(dist[idx + w + 1] + b);
                }
                if x > 0 {
                    d = d.min(dist[idx + w - 1] + b);
                }
            }
            dist[idx] = d;
        }
    }
    dist
}

/// `createWeightMap`: distance transform scaled by sharpness, truncated at 1.
pub fn create_weight_map(mask: &GrayImage, sharpness: f32) -> Vec<f32> {
    let mut w = distance_transform_l1(mask);
    for v in w.iter_mut() {
        *v = (*v * sharpness).min(1.0);
    }
    w
}

/// `FeatherBlender` (sharpness default 0.02): weighted accumulation into a
/// CV_16SC3-equivalent buffer, normalized by the summed weight map.
pub struct FeatherBlender {
    sharpness: f32,
    roi: (i32, i32, usize, usize), // (x, y, w, h)
    dst: Vec<[i16; 3]>,
    dst_weight: Vec<f32>,
}

impl FeatherBlender {
    pub fn new(sharpness: f32) -> Self {
        Self {
            sharpness,
            roi: (0, 0, 0, 0),
            dst: Vec::new(),
            dst_weight: Vec::new(),
        }
    }

    pub fn prepare(&mut self, x: i32, y: i32, w: usize, h: usize) {
        self.roi = (x, y, w, h);
        self.dst = vec![[0i16; 3]; w * h];
        self.dst_weight = vec![0.0f32; w * h];
    }

    /// `FeatherBlender::feed`: img is RGB u8 (converted to i16 exactly like
    /// the compose loop's `convertTo(CV_16S)`); per-addend products are
    /// truncated toward zero via the `static_cast<short>` port.
    pub fn feed(
        &mut self,
        img_rgb: &[u8],
        img_w: usize,
        img_h: usize,
        mask: &GrayImage,
        tl: (i32, i32),
    ) {
        assert_eq!(img_rgb.len(), img_w * img_h * 3);
        assert_eq!(mask.width, img_w);
        assert_eq!(mask.height, img_h);
        let weight = create_weight_map(mask, self.sharpness);
        let (rx, ry, rw, _rh) = self.roi;
        let dx = (tl.0 - rx) as usize;
        let dy = (tl.1 - ry) as usize;

        for y in 0..img_h {
            for x in 0..img_w {
                let wv = weight[y * img_w + x];
                let s = &img_rgb[(y * img_w + x) * 3..(y * img_w + x + 1) * 3];
                let d = &mut self.dst[(dy + y) * rw + dx + x];
                d[0] += (s[0] as i16 as f32 * wv) as i16;
                d[1] += (s[1] as i16 as f32 * wv) as i16;
                d[2] += (s[2] as i16 as f32 * wv) as i16;
                self.dst_weight[(dy + y) * rw + dx + x] += wv;
            }
        }
    }

    /// `FeatherBlender::blend` + `Blender::blend`: normalize by weights,
    /// zero uncovered pixels, return (RGB u8 clamped, coverage mask).
    pub fn blend(self) -> (Vec<u8>, GrayImage) {
        let (_x, _y, w, h) = self.roi;
        let mut out = vec![0u8; w * h * 3];
        let mut mask = vec![0u8; w * h];
        for i in 0..w * h {
            let cov = self.dst_weight[i] > WEIGHT_EPS;
            mask[i] = if cov { 255 } else { 0 };
            if cov {
                for c in 0..3 {
                    let v = (self.dst[i][c] as f32 / (self.dst_weight[i] + WEIGHT_EPS)) as i16;
                    out[i * 3 + c] = v.clamp(0, 255) as u8;
                }
            }
        }
        (out, GrayImage::new(w, h, mask))
    }
}

/// `resultRoi(corners, sizes)`: union rectangle. Returns (x, y, w, h).
pub fn result_roi(corners: &[(i32, i32)], sizes: &[(i32, i32)]) -> (i32, i32, usize, usize) {
    assert!(!corners.is_empty());
    let mut tl = (i32::MAX, i32::MAX);
    let mut br = (i32::MIN, i32::MIN);
    for (c, s) in corners.iter().zip(sizes) {
        tl.0 = tl.0.min(c.0);
        tl.1 = tl.1.min(c.1);
        br.0 = br.0.max(c.0 + s.0);
        br.1 = br.1.max(c.1 + s.1);
    }
    (tl.0, tl.1, (br.0 - tl.0) as usize, (br.1 - tl.1) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_transform_simple() {
        // 5x1 mask: 0 255 255 255 0 -> distances 0 1 2 1 0.
        let mask = GrayImage::new(5, 1, vec![0, 255, 255, 255, 0]);
        let d = distance_transform_l1(&mask);
        assert_eq!(d, vec![0.0, 1.0, 2.0, 1.0, 0.0]);
    }

    #[test]
    fn feather_blend_two_overlapping() {
        // Two 4x1 images overlapping by 2px; constant colors 100 and 200.
        let mask = GrayImage::new(4, 1, vec![255; 4]);
        let mut b = FeatherBlender::new(1.0);
        b.prepare(0, 0, 6, 1);
        b.feed(&[100u8; 12], 4, 1, &mask, (0, 0));
        b.feed(&[200u8; 12], 4, 1, &mask, (2, 0));
        let (out, cov) = b.blend();
        assert!(cov.data.iter().all(|&m| m == 255));
        // Left edge ~100, right edge ~200 — MINUS one: OpenCV divides by
        // (weight + 1e-5) and truncates via static_cast<short>, so solo
        // pixels lose one LSB. Faithfully ported.
        assert_eq!(out[0], 99);
        assert_eq!(out[5 * 3], 199);
        let mid = out[2 * 3 + 1];
        assert!(mid > 100 && mid < 200, "overlap should mix: {mid}");
    }

    #[test]
    fn result_roi_union() {
        let roi = result_roi(&[(-5, 2), (10, -3)], &[(20, 10), (5, 8)]);
        assert_eq!(roi, (-5, -3, 20, 15));
    }
}
