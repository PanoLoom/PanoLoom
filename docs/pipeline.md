# OpenCV 4.x Panorama Stitching Pipeline — Engineering Study for the Rust Port

Source of truth: the `4.x` branch of `opencv/opencv`, fetched 2026-08-10 from
`https://raw.githubusercontent.com/opencv/opencv/4.x/...`. Line numbers below are from that
snapshot and may drift by a few lines; symbol names will not. Files studied:

| Area | File(s) |
|---|---|
| Orchestration | `modules/stitching/include/opencv2/stitching.hpp`, `modules/stitching/src/stitcher.cpp` |
| Features | `modules/features2d/src/orb.cpp`, `modules/features2d/include/opencv2/features2d.hpp` |
| Matching | `modules/stitching/src/matchers.cpp`, `.../detail/matchers.hpp` |
| Homography/RANSAC | `modules/calib3d/src/fundam.cpp`, `modules/calib3d/src/ptsetreg.cpp` |
| Rotation estimation / BA | `modules/stitching/src/motion_estimators.cpp`, `.../detail/motion_estimators.hpp`, `modules/stitching/src/autocalib.cpp`, `modules/calib3d/include/opencv2/calib3d/private.hpp` (LevMarq) |
| Warping | `modules/stitching/src/warpers.cpp`, `.../detail/warpers.hpp`, `.../detail/warpers_inl.hpp` |
| Exposure | `modules/stitching/src/exposure_compensate.cpp`, `.../detail/exposure_compensate.hpp` |
| Seams | `modules/stitching/src/seam_finders.cpp`, `.../detail/seam_finders.hpp` |
| Blending | `modules/stitching/src/blenders.cpp`, `.../detail/blenders.hpp` |
| Support | `modules/stitching/src/util.cpp`, `src/camera.cpp`, `.../detail/camera.hpp` |

Throughout: "PANORAMA mode" means the configuration built by `Stitcher::create(Stitcher::PANORAMA)`
(`stitcher.cpp:53-99`), which is what the port replicates.

---

## 0. Pipeline overview and the three resolution tiers

```
inputs (full res)
  ├─ resize(work_scale)   → ORB features → pairwise match+RANSAC → leaveBiggestComponent
  │                         → HomographyBasedEstimator → BundleAdjusterRay → waveCorrect
  ├─ resize(seam_scale)   → spherical warp (scale·seam_work_aspect) → BlocksGainCompensator.feed
  │                         → GraphCutSeamFinder(COST_COLOR) → seam masks (low res)
  └─ resize(compose_scale)→ spherical warp (scale·compose_work_aspect) → gain apply
                            → dilate+upscale seam masks → MultiBandBlender feed/blend → CV_8U pano
```

### PANORAMA-mode component wiring (`stitcher.cpp:53-99`)

| Component | Instance |
|---|---|
| features | `ORB::create()` (all defaults) |
| matcher | `BestOf2NearestMatcher(false)` (all defaults) |
| estimator | `HomographyBasedEstimator` |
| bundle adjuster | `BundleAdjusterRay` |
| wave correction | on, `WAVE_CORRECT_HORIZ` |
| warper | `SphericalWarper` |
| exposure | `BlocksGainCompensator` |
| seams | `GraphCutSeamFinder(COST_COLOR)` — note: **not** the class default `COST_COLOR_GRAD` |
| blender | `MultiBandBlender(false)` |
| interp | `INTER_LINEAR` |
| pano conf threshold | `1` (`setPanoConfidenceThresh(1)`, `stitcher.cpp:60`) |

### Resolution tiers (`stitcher.cpp:57-59, 424-453, 247-255`)

* `registr_resol_ = 0.6` MP, `seam_est_resol_ = 0.1` MP, `compose_resol_ = ORIG_RESOL = -1.0`
  (keep original resolution).
* `work_scale = min(1.0, sqrt(0.6e6 / full_area))` — `stitcher.cpp:434`.
* `seam_scale = min(1.0, sqrt(0.1e6 / full_area))`; `seam_work_aspect = seam_scale / work_scale`
  — `stitcher.cpp:441-442`.
* `compose_scale = min(1.0, sqrt(compose_resol*1e6 / full_area))` if `compose_resol_ > 0`, else `1`;
  `compose_work_aspect = compose_scale / work_scale` — `stitcher.cpp:250-255`.
* **Gotcha:** all three scales are computed from the **first image only** (`is_work_scale_set`
  latches). With mixed-size inputs OpenCV silently applies the first image's scale to all.
  Replicate exactly.
* Image resizes use `INTER_LINEAR_EXACT` (bit-exact fixed-point bilinear — good for parity);
  feature masks use `INTER_NEAREST` (`stitcher.cpp:437, 448, 453`).
* Compose-time resize only happens when `abs(compose_scale-1) > 1e-1` (`stitcher.cpp:284`) —
  a 10% dead zone, not an equality test.

### Data crossing stage boundaries

* `ImageFeatures { img_idx: i32, img_size: Size, keypoints: Vec<KeyPoint>, descriptors: Mat<u8> }`
  (`detail/matchers.hpp:58-64`). Keypoint coordinates are in **work_scale** pixels.
* `MatchesInfo { src_img_idx, dst_img_idx, matches: Vec<DMatch>, inliers_mask: Vec<u8>,
  num_inliers: i32, H: Mat<f64> (3x3), confidence: f64 }` (`detail/matchers.hpp:99-113`).
  Stored in a dense `num_images × num_images` row-major vector; `(i,j)` and `(j,i)` both filled.
* `CameraParams { focal, aspect, ppx, ppy: f64, R: Mat (3x3, CV_32F after estimation), t }`;
  `K() = [[focal,0,ppx],[0,focal*aspect,ppy],[0,0,1]]` as CV_64F (`camera.hpp:57-70`,
  `camera.cpp:64-70`). `t` is always zero in this pipeline. Focal is in **work_scale px**.
* Between registration and compositing: `warped_image_scale_` = median focal
  (`stitcher.cpp:517-528`); note the even-count median is `float(f[n/2-1]+f[n/2]) * 0.5f` —
  the sum is **cast to f32 before halving**.

---

## 1. ORB feature detection (`orb.cpp`)

**Purpose.** Scale/rotation-tolerant keypoints + 256-bit binary descriptors on the work-scale
grayscale image.

**Algorithm.**
1. Build an `nlevels` image pyramid; level scale `getScale(level) = 1.2^(level - firstLevel)`
   (`orb.cpp:653-656`). Each level is resized **from the previous level** (`prevImg = currImg`,
   `orb.cpp:1111-1158`) with `INTER_LINEAR_EXACT` — cascaded, not from the base image.
   Levels live in one big buffer image with `border = max(edgeThreshold, max(ceil(15·√2)=22, 4))+1
   = 32` px of `BORDER_REFLECT_101` padding (level 0) / `+BORDER_ISOLATED` (others)
   (`orb.cpp:1027-1031, 1140-1152`).
2. Per level: FAST (threshold 20, nonmax suppression on) → drop keypoints within
   `edgeThreshold=31` of the border (`runByImageBorder`) → `retainBest(2 * per_level_quota)`
   (`orb.cpp:889-899`).
3. Per-level quota is a geometric series: `n_0 = nfeatures·(1-1/1.2)/(1-(1/1.2)^nlevels)`,
   `n_{l+1} = n_l/1.2`, last level gets the remainder (`orb.cpp:845-855`).
4. Harris re-scoring (blockSize 7, `HARRIS_K = 0.04f`, `orb.cpp:50, 944`): Sobel-ish gradients
   `Ix = 2(p[1]-p[-1]) + (p[-s+1]-p[-s-1]) + (p[s+1]-p[s-1])` over a 7×7 block around
   `cvRound(pt)`, response `= (a·b - c² - k(a+b)²) · scale⁴` with `scale = 1/(4·blockSize·255)`.
   Then `retainBest(quota)` per level.
5. Orientation by intensity centroid (`ICAngles`, radius `halfPatchSize = 15`, precomputed `umax`
   circle table): `angle = fastAtan2(m01, m10)` in **degrees** (`fastAtan2` is a polynomial
   approximation, ~0.3° accuracy).
6. Keypoints scaled back to level-0 coordinates: `pt *= layerScale[octave]`;
   `size = patchSize·layerScale` (`orb.cpp:904-909, 996-1000`).
7. Descriptors: each pyramid level is blurred **in place** with `GaussianBlur(7×7, σ=2,
   BORDER_REFLECT_101)` (`orb.cpp:1234`), then rBRIEF: 256 point pairs from the hardcoded
   `bit_pattern_31_` table, each pair rotated by the keypoint angle with full
   `cos/sin(angle·π/180)` (OpenCV does **not** quantize the angle to 12° steps as the paper
   does), sampled with `cvRound` nearest-neighbor (`orb.cpp:GET_VALUE`, ~line 1310 region),
   bit `= (t0 < t1)`.

**Key constants** (`features2d.hpp:460-461`, `orb.cpp:50`):

| Constant | Value |
|---|---|
| nfeatures | 500 |
| scaleFactor | 1.2f |
| nlevels | 8 |
| edgeThreshold | 31 |
| firstLevel | 0 |
| WTA_K | 2 (⇒ 32-byte descriptor, NORM_HAMMING) |
| scoreType | HARRIS_SCORE |
| patchSize | 31 |
| fastThreshold | 20 |
| HARRIS_K | 0.04f |
| Harris blockSize | 7 |
| descriptor blur | Gaussian 7×7, σ=2, REFLECT_101 |
| pyramid border | 32 px |

**Rust-port gotchas.**
* `KeyPointsFilter::retainBest` uses `std::nth_element` and then **keeps ties** at the cutoff
  response — output can exceed the quota, and ordering within equal responses is
  implementation-defined. For parity, sort by (response desc) with a stable tiebreak and accept
  count differences at ties, or compare as sets.
* `cvRound` is round-half-to-even (banker's rounding via SSE `cvtsd2si`). Rust `f32::round`
  rounds half away from zero — **must implement cvRound**.
* FAST: OpenCV's nonmax suppression computes a score = max threshold at which the pixel is still
  a corner; replicate the exact score function or keypoint sets will differ.
* Cascaded pyramid resize means small f32 differences compound per level; use
  `INTER_LINEAR_EXACT`-equivalent fixed-point bilinear (it is integer-exact, so this is
  reproducible).
* `fastAtan2` is approximate; using `atan2` changes angles by up to ~0.3°, which flips rBRIEF
  bits. Port OpenCV's polynomial.
* Grayscale conversion is `COLOR_BGR2GRAY` (`orb.cpp:1044-1045`): `0.299R + 0.587G + 0.114B`
  in fixed point — replicate the fixed-point coefficients.
* Ignore the OpenCL (`useOCL`) branches entirely.

**Parity strategy.** Python oracle: `cv2.ORB_create().detectAndCompute(gray, None)`. Dump
keypoints (x, y, angle, response, octave, size) and the 500×32 descriptor matrix to `.npz`.
Compare: keypoint sets with tolerance 0 (they should be bit-identical if cvRound/fastAtan2/
INTER_LINEAR_EXACT are faithful); descriptors byte-exact. Test with synthetic images (checkers,
noise) and real photos at several sizes; include a size that triggers `work_scale < 1`.

---

## 2. Pairwise matching — BestOf2NearestMatcher (`matchers.cpp`)

**Purpose.** For every image pair: descriptor matches → RANSAC homography → inlier count →
Brown–Lowe confidence score.

**Algorithm** (`BestOf2NearestMatcher::match`, `matchers.cpp:397-475`):
1. Raw matching (`CpuMatcher::match`, `matchers.cpp:149-211`): 2-NN both directions.
   Ratio test: accept `m0` iff `m0.distance < (1 - match_conf) · m1.distance`, i.e. **0.7·d1**
   with the default `match_conf = 0.3` (`matchers.hpp:196`). 2→1 matches are added with
   query/train swapped, skipping pairs already found in 1→2 (a `std::set` of index pairs).
2. If `matches.size() < num_matches_thresh1 (6)` → stop (no H, confidence 0).
3. Build point arrays with coordinates **centered**: `p -= img_size·0.5f` (`matchers.cpp:415-423`).
   All downstream homographies live in this centered frame — this is what lets
   `focalsFromHomography` assume principal point 0.
4. `H = findHomography(src, dst, inliers_mask, RANSAC)` (defaults: reproj threshold 3.0,
   confidence 0.995, maxIters 2000 — see §3). Reject if `H` empty or `|det(H)| < DBL_EPSILON`.
5. `num_inliers` = popcount of mask; confidence
   `= num_inliers / (8 + 0.3 · matches.size())` (`matchers.cpp:437-439`, from Brown & Lowe,
   "Automatic Panoramic Image Stitching using Invariant Features": inliers > α + β·n_f with
   α=8.0, β=0.3).
6. **Near-duplicate rejection:** `confidence = (confidence > matches_confindece_thresh) ? 0 : confidence`
   with threshold **3.0** (`matchers.cpp:441-443`). Yes: a *too good* match is zeroed, so
   duplicate/near-identical frames don't join the graph. (The misspelling `confindece` is the
   actual API name.)
7. If `num_inliers >= num_matches_thresh2 (6)`: re-estimate `H` by a second
   `findHomography(inliers_only, RANSAC)` (`matchers.cpp:449-474`).
8. Driver (`FeaturesMatcher::match(features, pairwise_matches, mask)`, `matchers.cpp:338-363`):
   enumerate pairs `(i, j), i<j` where both have keypoints and mask allows; results stored at
   `pair_idx = i·N+j`, and the dual `(j,i)` gets copied with `H⁻¹` and swapped
   query/trainIdx (`matchers.cpp:88-99`).

**Key constants** (`matchers.hpp:196-201`):

| Constant | Value |
|---|---|
| match_conf | 0.3f (ratio = 1−0.3 = 0.7) |
| num_matches_thresh1 | 6 |
| num_matches_thresh2 | 6 |
| matches_confindece_thresh | 3.0 |
| confidence formula | `ni / (8 + 0.3·nm)` |

**Rust-port gotchas.**
* For CV_8U descriptors the "FLANN" matcher is switched to **LSH**
  (`matchers.cpp:170-176`) — a randomized, *approximate* index. This is the single biggest
  determinism hole in the OpenCV pipeline: results depend on LSH table seeding (that is why
  `MatchPairsBody` reseeds `theRNG() = RNG(state + pair_index)` per pair,
  `matchers.cpp:74-78`). **Do not replicate LSH.** Use exact brute-force Hamming 2-NN.
  Exact search returns a superset of LSH's matches; downstream RANSAC absorbs the difference,
  but per-pair dumps against a Python oracle will differ unless the oracle also uses
  `cv2.BFMatcher(cv2.NORM_HAMMING)` — build the oracle accordingly.
* Tie handling in 2-NN: with Hamming distances ties are common; order of equal-distance
  neighbors decides `m0` vs `m1`. Define a deterministic tiebreak (lowest trainIdx) and use the
  same in the oracle comparison.
* The 2→1 pass appends `DMatch(m0.trainIdx, m0.queryIdx, m0.distance)` — note `imgIdx` is left
  default and distances stay float.
* Confidence is computed from `matches.size()` (post-ratio-test count, both directions merged),
  not from the raw keypoint count.
* Parallelism: pairs may be matched in any order; results are index-addressed so order doesn't
  matter *if* the matcher itself is deterministic (BF is). Single-thread first, parallelize later.

**Parity strategy.** Oracle: `cv2.detail.BestOf2NearestMatcher_create(False, 0.3, 6, 6)` +
`matcher.apply2(features)` (the full `cv::detail` API is exposed in Python). But swap the inner
matcher story by writing a small Python reimplementation with `cv2.BFMatcher` + ratio test +
`cv2.findHomography` on centered points — 30 lines — and validate that against
`BestOf2NearestMatcher` statistically (inlier counts within a few %), then hold the Rust port
bit-comparable to the Python reimplementation. Compare per pair: match set, `num_inliers`, `H`
(after normalizing `H /= H[2,2]`, tolerance ~1e-9 if RNG is replicated, else compare inlier sets
and reprojection residuals).

---

## 3. findHomography / RANSAC essentials (`fundam.cpp`, `ptsetreg.cpp`)

**Purpose.** Robust 3×3 homography from noisy correspondences; used only with `method=RANSAC`
here (LMEDS/RHO/USAC not needed).

**Algorithm** (`fundam.cpp:357-463`):
1. Points converted to CV_32FC2. If `npoints == 4` → direct DLT, no RANSAC.
2. `RANSACPointSetRegistrator(cb, modelPoints=4, threshold=3.0, confidence=0.995, maxIters=2000)`
   (`ptsetreg.cpp:78-263`):
   * RNG is `RNG((uint64)-1)` — **fixed seed**, so RANSAC is deterministic for a given point
     ordering (`ptsetreg.cpp:171`).
   * `getSubset` (`ptsetreg.cpp:104-158`): draw 4 distinct indices via `rng.uniform(0, count)`
     with duplicate rejection; up to 10000 attempts; each subset is validated by
     `checkSubset`: no 3 collinear points in either set, plus the chirality test — for the 4
     triangles of the quad, `sign(det(A_src)·det(A_dst))` must agree (all 0 or all 4 negative)
     (`fundam.cpp` `HomographyEstimatorCallback::checkSubset`).
   * Kernel (`runKernel`): normalized DLT. Normalization is **L1-style**: shift by centroid,
     scale `s = count / Σ|x−cx|` per axis (not the Hartley RMS-√2 flavor). Build 9×9 `LᵀL`,
     smallest-eigenvector via `cv::eigen`, denormalize, then scale so `H[2][2] = 1`
     (`scaleFor`).
   * `findInliers`: reprojection error per point (squared L2, f32), inlier iff
     `err ≤ threshold²` (`ptsetreg.cpp:85-102`).
   * Adaptive iteration count `RANSACUpdateNumIters(0.995, outlier_ratio, 4, niters)` =
     `log(1-p)/log(1-(1-ep)⁴)` with clamping (`ptsetreg.cpp:55-75`).
   * Keep the model with max inlier count (`goodCount > MAX(maxGoodCount, modelPoints-1)`).
3. Post-polish (`fundam.cpp:415-446`): re-run DLT on all inliers, then Levenberg–Marquardt
   refinement of the 8 free parameters, `LMSolver::create(cb, 10)` — **10 iterations** on the
   9-vector with `H[2][2]` fixed by renormalization; then the inlier mask is **recomputed**
   against the refined H over the original points.

**Key constants:** threshold 3.0 px (`calib3d.hpp:842-845` defaults `method=0,
ransacReprojThreshold=3, maxIters=2000, confidence=0.995`); RANSAC subset attempts 10000; LM
refine iterations 10; RNG seed `0xFFFFFFFFFFFFFFFF`.

**Rust-port gotchas.**
* **cv::RNG must be ported bit-exactly** for identical inlier sets: multiply-with-carry,
  `state = state·4164903690u64 + (state >> 32)`, `next()` returns the low 32 bits,
  `uniform(a,b) = a + (int)(next() % (b−a))`. Fixed seed `u64::MAX`.
* Error thresholding happens in **f32** on squared errors; keep f32 there.
* DLT uses `cv::eigen` on the symmetric 9×9 (Jacobi eigensolver in doubles). A generic SVD will
  produce an equivalent H up to sign/scale; after `H /= H[2,2]` differences are ~1e-12 —
  acceptable, but inlier flips near the 3.0 px boundary can cascade. If exact parity is wanted,
  port the Jacobi routine.
* Note the reproj threshold is in **work-scale, centered** pixel units (matcher passes centered
  points).
* Skip: USAC, RHO, LMEDS branches; `usac::findHomography`.

**Parity strategy.** `cv2.findHomography(src, dst, cv2.RANSAC)` with recorded `src/dst` from the
matcher stage. With ported RNG + eigen: expect identical masks and `H` to ~1e-9. Without: assert
same inlier set on well-conditioned fixtures and `‖H_rust·x − H_cv·x‖ < 0.1 px` over a grid.

---

## 4. Match-graph pruning: leaveBiggestComponent & findMaxSpanningTree (`motion_estimators.cpp`)

**Purpose.** Keep only images that belong to the same panorama; pick a chaining order and a
reference image.

**Algorithm.**
* `leaveBiggestComponent(features, pairwise_matches, conf_thresh=1.0)`
  (`motion_estimators.cpp:1079-1135`): union-find over all `(i,j)` with
  `confidence ≥ conf_thresh`; keep the largest set; rebuild `features` and the dense
  `pairwise_matches` matrix for the subset (re-indexing `src/dst_img_idx`); returns the kept
  original indices (Stitcher stores them as `indices_`, `stitcher.cpp:474`).
* `findMaxSpanningTree(num_images, pairwise_matches, span_tree, centers)`
  (`motion_estimators.cpp:1138-1206`): edges for every pair with non-empty `H`, weight =
  `num_inliers` (both `(i,j)` and `(j,i)` are inserted). Kruskal on edges sorted **descending**
  by weight (`std::sort` + `greater<GraphEdge>` — not stable). Then: leaves → BFS max distance
  per node → centers = nodes minimizing the max distance (1 or 2 of them); `centers[0]` becomes
  the reference camera.

**Gotchas.** Equal-weight edges: `std::sort` is unstable, so the chosen tree can differ between
STL implementations; a Rust port should use a stable sort with the same comparator — this
reproduces libstdc++/libc++ in practice on distinct weights, and on ties any difference shows up
only as a different (equally valid) reference chain. Comparison tests should therefore compare
final camera rotations up to a global rotation, not the tree itself.

---

## 5. HomographyBasedEstimator (`motion_estimators.cpp:126-192`, `autocalib.cpp`)

**Purpose.** Initial focal length + rotation for every camera from pairwise homographies.

**Algorithm.**
1. Focals (`estimateFocal`, `autocalib.cpp:102-147`): for every pairwise `H` (both directions),
   `focalsFromHomography` (`autocalib.cpp:63-99`) extracts two candidate focals from the
   rotation-only decomposition (Szeliski/Shum): two candidate values per side chosen by the
   larger-|denominator| rule; if both `f0, f1` are valid push `sqrt(f0·f1)`. If at least
   `num_images − 1` estimates exist, every camera gets the **median** (even count: average of
   the two middle values); otherwise the fallback focal for *all* cameras is
   `Σ(width_i + height_i) / num_images` (`autocalib.cpp:138-146`).
2. Rotations: BFS over the max spanning tree from `centers[0]` with
   `R_to = R_from · (K_from⁻¹ · H_{from,to}⁻¹ · K_to)` (`CalcRotation`,
   `motion_estimators.cpp:59-87`) in **CV_64F**; K built from focal with `aspect`, `ppx=ppy=0`
   during propagation.
3. Principal points restored to image center afterwards: `ppx += 0.5·w, ppy += 0.5·h`
   (`motion_estimators.cpp:184-188`) — consistent with the matcher's centered coordinates.
4. Stitcher then converts every `R` to CV_32F (`stitcher.cpp:504-510`).

**Gotchas.** `focalsFromHomography` divides by tiny denominators when H is near-affine; the
`v1>0 && v2>0` guards handle most of it but NaN can propagate into `all_focals` if `H` has
`h[6]=h[7]=0` exactly (then `v1 = -x/0 = ∓inf`) — the `> 0` comparisons reject inf/NaN as false,
matching IEEE semantics; keep IEEE behavior (no `fast-math`). The median-vs-fallback branch is a
behavioral cliff: test both. R chaining order comes from the spanning-tree BFS — same caveat as
§4 about comparing up to a global rotation.

**Parity.** `cv2.detail_HomographyBasedEstimator().apply(features, pairwise_matches, None)`
returns cameras; compare focals exactly (doubles) and `R_i · R_ref⁻¹` pairwise.

---

## 6. Bundle adjustment — BundleAdjusterBase / Ray / Reproj (`motion_estimators.cpp:222-643`)

**Purpose.** Jointly refine camera parameters by minimizing a robust-free least-squares cost
over all inlier correspondences of confident pairs.

**Framework** (`BundleAdjusterBase::estimate`, `motion_estimators.cpp:222-321`):
* Edges = pairs with `confidence > conf_thresh_` (Stitcher sets `conf_thresh_ = 1.0`,
  `stitcher.cpp:512`; the member default is also 1.0, `motion_estimators.hpp:161`).
* `total_num_matches` = Σ `num_inliers` over edges.
* Solver: `LevMarq(num_images·num_params_per_cam, total_num_matches·num_errs_per_measurement,
  term_criteria_)` — the C++ re-implementation of the legacy **CvLevMarq** in
  `calib3d/private.hpp:69-104`. Term criteria default:
  `TermCriteria(EPS + COUNT, 1000, DBL_EPSILON)` (`motion_estimators.hpp:163`).
* LevMarq semantics to replicate (documented in `calib3d/private.hpp:58-68`):
  damping `λ` starts at `10^-3` (`lambdaLg10 = -3`); error decreased → `λ /= 10`; increased →
  `λ *= 10` (up to 16 retries within a step); augmented normal equations scale the diagonal of
  `JᵀJ` by `(1 + λ)` (multiplicative, **not** `+λ·diag`); stop on max iterations or relative
  step norm `‖Δx‖₂/‖x‖₂ < eps`; inner solve `cv::solve(JtJN, JtErr, DECOMP_SVD)`.
  Do **not** use the newer `LMSolver` gain-ratio strategy — different iterate path.
* Jacobians are **numeric central differences**, full re-evaluation of the error vector per
  parameter: step `1e-3` for Ray (`motion_estimators.cpp:628`), `1e-4` for Reproj
  (`motion_estimators.cpp:447`).
* After convergence: NaN check on parameters → `ERR_CAMERA_PARAMS_ADJUST_FAIL`; then all
  rotations are re-referenced: `R_i ← R_center⁻¹ · R_i` using the spanning-tree center
  (`motion_estimators.cpp:311-317`).

**BundleAdjusterRay** (the PANORAMA default; 4 params/cam, 3 errors/match):
* Params: `[focal, rvec0, rvec1, rvec2]` per camera. Initialization projects R onto SO(3) via
  SVD (`R = U·Vᵀ`, negate if det < 0) then `Rodrigues` (`motion_estimators.cpp:507-527`).
* Error (`calcError`, `motion_estimators.cpp:549-620`): for each inlier match, unproject both
  keypoints to unit ray directions using `H = R · K⁻¹` with K from the *current* focal and
  principal point at **feature-image center** (`width·0.5, height·0.5` — not the camera's
  ppx/ppy), normalize to unit length, and emit
  `err = sqrt(f1·f2) · (ray1 − ray2)` (3 components).
* Refinement mask is ignored by Ray (only Reproj honors it).

**BundleAdjusterReproj** (7 params/cam: focal, ppx, ppy, aspect, rvec; 2 errors/match):
* Error = classic transfer error `p2 − H·p1` with
  `H = K2 · R2⁻¹ · R1 · K1⁻¹` (`motion_estimators.cpp:374-438`).
* Refinement mask (3×3 CV_8U, default all ones, `motion_estimators.hpp:150-158`): position
  (0,0)→focal, (0,2)→ppx, (1,2)→ppy, (1,1)→aspect; rvec always refined
  (`motion_estimators.cpp:449-500`).

**Gotchas.**
* Everything is **CV_64F** except the incoming `R` (CV_32F) — `Rodrigues` on a CV_32F matrix
  returns a CV_32F rvec (the code asserts this, `motion_estimators.cpp:522`), so initial rvecs
  are f32-truncated. Replicate the truncation or accept ~1e-7 divergence at iteration 0 which
  LM will amplify.
* `Rodrigues` must match OpenCV's formulation, including the θ→0 series branch.
* Finite-difference Jacobians make the whole optimization exquisitely sensitive to error-vector
  bit-parity; port `calcError` first and diff it standalone before wiring LM.
* Match order inside the error vector = edge order (i<j lexicographic) then match order within
  `MatchesInfo.matches` filtered by `inliers_mask` — preserve it.
* SVD-projection of R at setup: sign convention of `cv::SVD` differs from LAPACK's — only the
  product `U·Vᵀ` matters, which is stable.
* The LM inner solve uses `DECOMP_SVD` on a small dense system (`4N×4N`); N is small (≤ dozens),
  so port cost is negligible.

**Parity.** Oracle: `ba = cv2.detail_BundleAdjusterRay(); ba.setConfThresh(1.0);
ba.apply(features, pairwise_matches, cameras)`. Dump per-iteration `cam_params_` is not exposed;
instead compare (a) the standalone error vector for fixed cameras (write it via the Rust side
and a NumPy reimplementation), (b) final focals (rel. tol 1e-6) and rotations up to global
rotation (angular distance < 0.01°).

---

## 7. waveCorrect (`motion_estimators.cpp:885-1008`)

**Purpose.** Remove the global "wavy" tilt: choose a global rotation so camera x-axes are
coplanar with the horizon (horizontal panoramas).

**Algorithm** (`waveCorrect(rmats, kind)`):
1. No-op if ≤ 1 camera. `WAVE_CORRECT_AUTO` picks HORIZ/VERT by comparing spans of
   `rmat[0][2]/rmat[2][2]` vs `rmat[1][2]/rmat[2][2]` (`autoDetectWaveCorrectKind`,
   `motion_estimators.cpp:885-922`); Stitcher uses HORIZ.
2. `moment = Σ col0(R_i)·col0(R_i)ᵀ` (3×3, CV_32F) → `cv::eigen` (descending eigenvalues).
   `rg1` = eigenvector row 2 (smallest) for HORIZ, row 0 (largest) for VERT — this is the new
   global "up" (HORIZ).
3. `img_k = Σ col2(R_i)`; `rg0 = rg1 × img_k`, normalized — the new "east".
   **Guard:** if `‖rg0‖ ≤ DBL_MIN` return with rmats unchanged (degenerate: all views parallel
   to up).
4. Sign disambiguation: HORIZ — if `Σ rg0·col0(R_i) < 0` flip `rg0, rg1`; VERT — same with
   `−Σ rg1·col0(R_i)`.
5. `rg2 = rg0 × rg1`; stack rows `[rg0; rg1; rg2]` into `R` and apply `R_i ← R·R_i` to all.

**Gotchas.** All in **CV_32F** (asserted). `cv::eigen` for symmetric 3×3 — eigenvector sign is
arbitrary; the conf-flip only disambiguates `rg0/rg1` jointly, so a port whose eigensolver
returns the opposite sign of `rg1` produces a pano flipped upside-down. Match OpenCV's Jacobi
eigen (or post-normalize: force `rg1·(Σcol1(R_i)) > 0`... — simplest is to port the Jacobi
routine for 3×3). This function is tiny; port it verbatim.

**Parity.** `cv2.detail.waveCorrect(rmats, cv2.detail.WAVE_CORRECT_HORIZ)` mutates the list —
compare each matrix elementwise (f32 exact if eigen is ported, else 1e-6 with sign fix-ups).

---

## 8. Warping — SphericalWarper / CylindricalWarper (`warpers_inl.hpp`, `warpers.cpp`, `warpers.hpp`)

**Purpose.** Project each image onto the composition surface (unit sphere for PANORAMA) using
only rotation + intrinsics; produce warped image, warped mask, and the destination ROI corner.

**Machinery.**
* `ProjectorBase::setCameraParams(K, R, T=0)` (`warpers.cpp:~100-130`): requires K and R
  **CV_32F 3×3**; precomputes float arrays `k = K`, `rinv = Rᵀ`, `r_kinv = R·K⁻¹`,
  `k_rinv = K·Rᵀ`, `t`. **All warp math is f32.**
* `scale` (pixels per radian on the unit sphere) = focal in pixels at the current tier:
  seam tier `warped_image_scale_·seam_work_aspect` (`stitcher.cpp:184`), compose tier
  `warped_image_scale_·compose_work_aspect` (`stitcher.cpp:258`). K is likewise rescaled per
  tier (`stitcher.cpp:186-192, 265-278`).
* Forward map (`SphericalProjector::mapForward`, `warpers_inl.hpp:252-262`):

  ```
  [x_, y_, z_]ᵀ = r_kinv · [x, y, 1]ᵀ
  u = scale · atan2f(x_, z_)
  w = y_ / sqrtf(x_² + y_² + z_²)
  v = scale · (π − acosf(w == w ? w : 0))        // NaN guard: w!=w ⇒ 0
  ```
  so u ∈ [−π·scale, π·scale], v ∈ [0, π·scale].
* Backward map (`mapBackward`, `warpers_inl.hpp:265-283`):

  ```
  u /= scale; v /= scale
  sinv = sinf(π − v)
  d = [sinv·sinf(u), cosf(π − v), sinv·cosf(u)]ᵀ
  [x, y, z]ᵀ = k_rinv · d
  if z > 0 { x /= z; y /= z } else { x = y = -1 }   // behind camera ⇒ maps outside
  ```
* Cylindrical (`warpers_inl.hpp:286-315`): forward `u = scale·atan2f(x_, z_)`,
  `v = scale·y_/sqrtf(x_²+z_²)`; backward direction `[sinf(u), v, cosf(u)]`, same z>0 guard.
* `buildMaps` (`warpers_inl.hpp:74-98`): `detectResultRoi` → allocate
  `(br−tl+1)` CV_32F xmap/ymap → fill via `mapBackward(u, v)` per integer dest pixel → return
  `Rect(tl, br)`. `warp` then does `dst.create(roi.height+1, roi.width+1)` and
  `remap(src, dst, xmap, ymap, interp, border_mode)` (`warpers_inl.hpp:101-112`).
  Images: `INTER_LINEAR + BORDER_REFLECT`; masks: `INTER_NEAREST + BORDER_CONSTANT(0)`
  (`stitcher.cpp:194-197, 306-315`).
* `warpRoi` = `Rect(tl, br + (1,1))` (`warpers_inl.hpp:146-155`), used at compose time to place
  corners without warping (`stitcher.cpp:279-281`).
* ROI detection: generic base scans **every source pixel** through mapForward
  (`warpers_inl.hpp:158-181`); `SphericalWarper::detectResultRoi` (`warpers.cpp:375-416`)
  instead scans only the border (`detectResultRoiByBorder`) and then patches the poles: if the
  north/south pole (±column 1 of `rinv`) projects inside the source image, extend v-range to
  `π·scale` / `0`. `CylindricalWarper` uses border-scan only (`warpers.hpp:355`).

**Gotchas.**
* ROI corners are produced by `static_cast<int>(f)` — **truncation toward zero**, not floor
  (`warpers_inl.hpp:177-180`). For negative coordinates (common: tl is usually negative in u),
  this is effectively `ceil`. Using `floor` shifts the pano by 1 px and breaks corner parity.
* All trig in f32 (`atan2f/acosf/sinf/cosf`). Rust `f32::atan2` etc. are correctly-rounded via
  libm and match glibc within 1 ulp; residual 1-ulp differences can move a map value by ~1e-5 px
  — harmless for image diffing, fatal for bit-exactness. Parity tests on maps should use
  tolerance ~1e-4 px.
* `remap(INTER_LINEAR)` uses fixed-point bilinear with 5-bit fractional tables
  (`INTER_TAB_SIZE=32`); to match pixels exactly, implement the same fixed-point scheme
  (coefficients quantized to 1/2048... in `BLOCK` tables) — or accept ±1 intensity level
  differences.
* `BORDER_REFLECT` (`gfedcb|abcdefgh`) for image warps at seam *and* compose time; masks use
  constant 0. Mask/image asymmetry is intentional (reflection avoids dark fringes inside the
  mask).
* The NaN guard `w == w ? w : 0` must be preserved (occurs when x_=y_=z_=0, i.e. principal ray
  at K⁻¹ singularity).
* Ignore: `warpPointBackward`, PlaneWarper T-vector support (t = 0 always here), UMat/OpenCL
  `buildMaps` overloads in `warpers.cpp` for Plane/Spherical/Cylindrical (`ocl_warp...`), CUDA
  warpers, and all Fisheye/Stereographic/CompressedRectilinear/Pani/Mercator/Transverse
  projectors.

**Parity.** Oracle: `w = cv2.PyRotationWarper('spherical', scale)`; `w.warpRoi(size, K, R)`,
`w.buildMaps(size, K, R)`, `w.warp(img, K, R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)` and
`warpPoint`. Compare ROI exactly; maps to 1e-4; warped images with ≤1 intensity level per pixel
(or exactly, if fixed-point remap is ported). Test R matrices including identity, ±90° yaw, and
a rotation putting a pole in view (exercises the pole patch).

---

## 9. Exposure compensation — GainCompensator / BlocksGainCompensator (`exposure_compensate.cpp`)

**Purpose.** Multiplicative gain per image (or per block) minimizing brightness mismatch in
overlaps, per Brown & Lowe §6.

**GainCompensator::singleFeed** (`exposure_compensate.cpp:116-278`):
1. For every pair `(i, j), j ≥ i` with overlapping ROI (`overlapRoi` on corners+sizes):
   intersection mask = `mask_i == 255 & mask_j == 255`; `N(i,j) = N(j,i) = max(1, |intersect|)`;
   mean intensities `I(i,j) = ΣnormL2(BGR_i)/N`, `I(j,i) = ΣnormL2(BGR_j)/N` — note intensity
   of a pixel is the **L2 norm of the BGR triple** (≈ brightness·√3), not the mean channel.
2. Least squares on gains g (error `Σ_ij N_ij[(g_i·I_ij − g_j·I_ji)²·α + (1−g_i)²·β]`):
   with `alpha = 0.01`, `beta = 100` (`exposure_compensate.cpp:215-216`), assembled as
   `A(ki,ki) += β·N + 2α·I_ij²·N`, `A(ki,kj) −= 2α·I_ij·I_ji·N`, `b(ki) += β·N`, skipping
   images with no overlap (their gain stays 1).
3. Solve `A·g = b`. **Build-dependent:** with Eigen available OpenCV solves in **f32 LLT**
   (`Eigen::MatrixXf`, `exposure_compensate.cpp:251-268`); otherwise `cv::solve` (f64 LU).
   Gains differ in the 5th decimal between builds. For the Rust port: solve in f64 (Cholesky)
   and give parity tests a 1e-3 relative tolerance on gains.
4. `nr_feeds_ = 1` by default (`exposure_compensate.hpp:115-117`) so the outer feed loop runs
   once; `similarity_threshold_ = 1` disables the similarity mask
   (`prepareSimilarityMask` early-outs, `exposure_compensate.cpp:318-321`).

**BlocksGainCompensator** (`exposure_compensate.cpp:462-609`, defaults
`bl_width = bl_height = 32`, `exposure_compensate.hpp:211-214`):
1. Each (seam-scale, warped) image is tiled into `ceil(w/32) × ceil(h/32)` blocks (block size is
   recomputed as `ceil(w / bl_per_img.width)` so blocks are even); every block of every image
   becomes a pseudo-image (with its global corner) fed to one big `GainCompensator`.
2. Per-image gain map: `bl_per_img` CV_32F matrix of the block gains, then smoothed by a
   separable [0.25, 0.5, 0.25] kernel (`sepFilter2D`, default BORDER_REFLECT_101) applied
   `nr_gain_filtering_iterations_ = 2` times (`exposure_compensate.cpp:509-527`).
3. `apply(idx, corner, image, mask)` (`exposure_compensate.cpp:560-582`): resize gain map to the
   image size with `INTER_LINEAR` (**not** the EXACT variant), broadcast to 3 channels,
   `multiply(image, gain_map, image, 1, image.type())` — saturating u8 output. Asserts
   `CV_8UC3` input.

**Stitcher call sites.** `feed(corners, images_warped, masks_warped)` at seam scale
(`stitcher.cpp:204`), then `apply` on the *seam-scale* images before seam finding
(`stitcher.cpp:205-206`) and again on each *compose-scale* warped image (`stitcher.cpp:322`) —
the same gain maps get bilinearly stretched to compose size.

**Gotchas.** Block sub-images share the parent's mask value 255; block corners are global
(`corners[img] + bl_tl`) so the overlap test is geometric. Gain-map resize + multiply is where
f32 arithmetic differences show; saturate-cast rounding is `cvRound`-based. The `apply` at seam
time mutates the images the seam finder sees — order matters. Ignore `ChannelsCompensator`,
`BlocksChannelsCompensator`, `nr_feeds > 1` and similarity-mask code paths.

**Parity.** Oracle: `c = cv2.detail.ExposureCompensator_createDefault(
cv2.detail.ExposureCompensator_GAIN_BLOCKS)`; `c.feed(corners, images, masks)`;
`c.getMatGains(...)` returns the per-image gain maps — compare to 1e-3 rel; compare applied
images with ±1 level tolerance. Also unit-test plain `GainCompensator` (GAIN) whose gains are
scalars — exposed via `gains()`.

---

## 10. Seam finding — GraphCutSeamFinder (`seam_finders.cpp`)

**Purpose.** Partition each pairwise overlap so each output pixel is claimed by exactly one
image, cutting along low-difference paths; results are recorded by zeroing mask pixels.

**Configuration.** Stitcher builds `GraphCutSeamFinder(COST_COLOR)` (`stitcher.cpp:61`);
constructor defaults are `cost_type = COST_COLOR_GRAD, terminal_cost = 10000.f,
bad_region_penalty = 1000.f` (`seam_finders.hpp:243-246`). Inputs: seam-scale warped images
converted to **CV_32FC3** (values still 0..255, `stitcher.cpp:209-212`), corners, and the warped
masks (modified in place).

**Algorithm** (`GraphCutSeamFinder::Impl`, `seam_finders.cpp:1108-1388`):
1. `find`: precompute per-image Sobel gradient magnitude images `dx_, dy_` (3-channel Sobel →
   `normL2` per pixel) — only used by COST_COLOR_GRAD (`seam_finders.cpp:1133-1161`).
2. `PairwiseSeamFinder::run` (`seam_finders.cpp:83-94`): for each pair `i<j` with overlapping
   ROI, `findInPair(i, j, roi)` — masks are updated **sequentially**, so later pairs see
   earlier cuts. Pair order is part of the algorithm.
3. `findInPair` (`seam_finders.cpp:1266-1361`): copy the overlap ROI **plus a gap of 10 px** on
   all sides into padded subimages/submasks (out-of-image texels = 0 / masked out). Build a
   `GCGraph` with one vertex per padded pixel:
   * terminal weights: `(mask1 ? terminal_cost : 0, mask2 ? terminal_cost : 0)` — 10000 binds
     a pixel to whichever sources actually cover it (`seam_finders.cpp:1170-1178`).
   * COST_COLOR n-edges (4-neighborhood, `setGraphWeightsColor`, `seam_finders.cpp:1164-1208`):
     `w = ‖I1(p)−I2(p)‖₂ + ‖I1(q)−I2(q)‖₂ + weight_eps` with `weight_eps = 1.f`; add
     `bad_region_penalty = 1000` if any endpoint is outside either mask.
   * COST_COLOR_GRAD divides the color term by
     `(dx1(p)+dx1(q)+dx2(p)+dx2(q)+1)` (resp. dy for vertical edges) before adding eps
     (`seam_finders.cpp:1212-1263`).
4. `graph.maxFlow()` — Boykov–Kolmogorov (`modules/imgproc/src/gcgraph.hpp`). Pixels in the
   source segment stay with image 1 (`mask2 = 0` there if mask1 covers), else with image 2
   (`seam_finders.cpp:1345-1360`).

**DpSeamFinder — the simpler alternative** (`seam_finders.cpp:178-1105`). Dynamic-programming
seam within each connected component of the overlap: per-component cost matrices `costV/costH`
where cost = average of the two "cross" color diffs (squared L2, `diffL2Square3`), optionally
divided by `(|grad| sums + 1)` for COLOR_GRAD; unreachable/masked cells get
`badRegionCost = ‖(255,255,255)‖₂ ≈ 441.67` (`seam_finders.cpp:746-826`). It then DPs a
monotone seam between component corner points. Cheaper (O(n) per overlap, no maxflow), less
general (single seam per component, source-ordering quirks). A Rust port can ship DpSeamFinder
(or `VoronoiSeamFinder` — pure distance transform, `seam_finders.cpp:96-176`) first and add
graph-cut later; final quality target should be graph-cut.

**Gotchas.**
* GCGraph float arithmetic and the BK traversal order are deterministic; a different max-flow
  implementation gives a different (equally minimal) cut when costs tie — pixel-exact parity
  requires porting GCGraph's exact edge insertion order and BK algorithm.
* The `gap = 10` padding is load-bearing: it lets the cut escape the ROI and default the
  outside; keep it.
* Costs use `normL2` = `sqrt` of the sum of squared channel diffs (f32).
* Masks are CV_8U 0/255; the graph only reads them as booleans but writes hard zeros.
* Seam masks come out at **seam scale**; compositing later dilates them (3×3, once) and resizes
  up with `INTER_LINEAR_EXACT`, then ANDs with the compose-scale warped mask
  (`stitcher.cpp:333-337`) — that dilation+AND is what hides 1-px seam misalignment.
* Ignore `GraphCutSeamFinderGpu`.

**Parity.** Oracle: `sf = cv2.detail_GraphCutSeamFinder('COST_COLOR')`;
`masks = sf.find(images_f32, corners, masks)`. Compare masks pixel-exact on fixtures without
cost ties (natural photos are fine); on synthetic constant-color overlaps expect divergence —
assert instead the seam property: masks partition the overlap, and total cut cost within 0.1%.

---

## 11. Blending — MultiBandBlender / FeatherBlender (`blenders.cpp`)

**Purpose.** Merge exposure-compensated, seam-masked warped images without visible transitions
by blending each Laplacian band over a distance proportional to its wavelength.

**Base class Blender** (`blenders.cpp:80-134`): `prepare(corners, sizes)` computes
`dst_roi = resultRoi` (union of `Rect(corner, size)`, `util.cpp:125-139`), allocates
`dst_ = CV_16SC3` zeros and `dst_mask_ = CV_8U` zeros. `feed` = masked copy; `blend` zeroes
pixels where the mask is 0.

**MultiBandBlender** (`blenders.cpp:216-693`; Stitcher uses `MultiBandBlender(false)` — defaults
`num_bands = 5`, `weight_type = CV_32F`, `blenders.hpp:130`):

* `prepare(dst_roi)` (`blenders.cpp:233-300`):
  `num_bands_ = min(5, ceil(log2(max(dst_roi.width, dst_roi.height))))`; then pad
  `dst_roi.width/height` up to multiples of `2^num_bands_`. Allocate
  `dst_pyr_laplace_[0..num_bands]` (CV_16SC3, level k+1 size = `(prev+1)/2`) and matching
  `dst_band_weights_` (CV_32F), all zeros.
  Note: in `Stitcher` the band count is 5 for any pano bigger than 32 px — the
  "num_bands from blend width" formula (`blend_strength = 5`,
  `blend_width = sqrt(area(dst))·5/100`, `num_bands = ceil(log2(blend_width)) − 1`) lives in the
  **sample** `samples/cpp/stitching_detailed.cpp`, not in the Stitcher class. Decide which
  behavior the port exposes; default to the class behavior (5).
* `feed(img CV_16SC3, mask CV_8U, tl)` (`blenders.cpp:328-601`):
  1. Compute a bordered rectangle around the image: `gap = 3·2^num_bands` on each side, clamped
     to `dst_roi_`, then **snapped so `tl_new − dst_roi.tl` and the size are multiples of
     `2^num_bands`** (bit-shift trickery at `blenders.cpp:368-391`); if `br` overflows, shift
     the window back. This guarantees inter-level scale is exactly 2 for the sub-pyramids.
  2. `copyMakeBorder(img, BORDER_REFLECT)` to that rectangle; build its Laplacian pyramid
     (`createLaplacePyr`, `blenders.cpp:788-837`): CV_8U input path does
     `pyrDown` chain + `pyrUp(next, size=current)` + `subtract(..., CV_16S)`; top level is the
     plain downsample converted to CV_16S. (For CV_16S input as fed here, the `else` branch
     runs entirely in 16S: pyrDown/pyrUp/subtract — `pyrDown/pyrUp` use the [1 4 6 4 1]/16
     Gaussian with `BORDER_REFLECT_101`.)
  3. Weight pyramid: `mask → CV_32F/255` bordered with `BORDER_CONSTANT(0)`, then `pyrDown`
     chain (`blenders.cpp:507-526`) — a plain Gaussian pyramid.
  4. Accumulate per band over the snapped rectangle:
     `dst += short(src · w)` per channel and `dst_w += w` (`blenders.cpp:552-568`) — note
     **`static_cast<short>` truncates toward zero**, it does not round.
* `blend(dst, dst_mask)` (`blenders.cpp:604-693`):
  `normalizeUsingWeightMap` per band: `v = short(v / (w + WEIGHT_EPS))` with
  `WEIGHT_EPS = 1e-5f` (`blenders.cpp:66, 720-775`) — again truncation;
  `restoreImageFromLaplacePyr`: `pyrUp` (size-matched) + `add` from the top down
  (`blenders.cpp:868-878`); crop to `dst_roi_final_`; final mask =
  `dst_band_weights_[0] > WEIGHT_EPS`; `Blender::blend` zeroes outside; Stitcher converts to
  CV_8U with saturation (`stitcher.cpp:365-373`).

**FeatherBlender** (`blenders.cpp:136-213`; default `sharpness = 0.02f`, `blenders.hpp:103`):
weight map = `min(1, distanceTransform(mask, DIST_L1, 3) · sharpness)`
(`createWeightMap`, `blenders.cpp:778-785`) — i.e. full weight 50 px inside the mask; feed
accumulates `short(src·w)` into 16S and `w` into f32; blend divides by the weight sum. Good
first target before multiband: 10× simpler, oracle-comparable, same data types.

**Gotchas.**
* The 16S truncation (`static_cast<short>`) in feed **and** normalize is the main source of
  off-by-one pixel diffs; replicate `as i16`-style truncation, not rounding.
* `pyrDown/pyrUp` border handling is `BORDER_REFLECT_101` and their kernels are exact integer
  shifts for 16S (`>>8` after 2D [1 4 6 4 1] convolution) — port OpenCV's fixed-point kernel,
  not a float Gaussian.
* The rectangle-snapping code (step 1 of feed) is subtle and must be copied exactly; errors show
  up as 1-px band misregistration (ghosting) only for images near the pano edge.
* Weight type CV_16S path (`>>8` accumulate, `(w+1)` normalize) is dead by default — skip it.
* `MultiBandBlender::feed` in the f32 path also has an OpenCL kernel — ignore; the scalar loop
  is the reference.
* Blend must handle `num_bands` shrinking in `prepare` (tiny panos).

**Parity.** Oracle: `b = cv2.detail_MultiBandBlender(0, 5, cv2.CV_32F); b.prepare(rect);
b.feed(img16s, mask, tl); ...; result, rmask = b.blend(None, None)`. Feed identical CV_16S
inputs recorded from the compose stage. Compare result CV_16S exactly (achievable — the whole
path is fixed-point except the f32 weights; weight math in f32 is deterministic). Also test
`createLaplacePyr`/`restoreImageFromLaplacePyr` standalone against
`cv2.pyrDown/pyrUp` chains.

---

## 12. Compositing flow details worth copying verbatim (`stitcher.cpp:129-376`)

1. Seam-tier: masks all-255 at seam-scale size; warp images (`INTER_LINEAR, BORDER_REFLECT`)
   and masks (`INTER_NEAREST, BORDER_CONSTANT`); K scaled by `seam_work_aspect` on the f32 K
   copy (`stitcher.cpp:176-198`).
2. Exposure `feed` + `apply` on seam-scale images, **then** seam find on the CV_32F conversions
   (`stitcher.cpp:203-212`).
3. Compose-tier per image: optional resize (`INTER_LINEAR_EXACT`, only if
   `|compose_scale−1| > 0.1`); K from cameras scaled by `compose_work_aspect`; corners/sizes
   from `warpRoi` on rounded scaled full sizes (`stitcher.cpp:262-282`);
   warp image + mask; exposure `apply`; `img.convertTo(CV_16S)`;
   `dilate(seam_mask, 3×3 default kernel)`; `resize` to the compose mask size
   (`INTER_LINEAR_EXACT`); `mask_warped = seam_mask & mask_warped`; `blender.feed`
   (`stitcher.cpp:284-358`). Blender `prepare` happens lazily on the first image with the
   precomputed corners/sizes.
4. `blender.blend` → convert CV_16SC3 result (values already 0..255) to CV_8U
   (`stitcher.cpp:365-373`). The result mask is exposed as `resultMask()`.
5. Failure statuses: `ERR_NEED_MORE_IMGS` (<2 images before or after component pruning),
   `ERR_HOMOGRAPHY_EST_FAIL`, `ERR_CAMERA_PARAMS_ADJUST_FAIL` (NaN in BA).

Note `cameras_scaled` only scales `focal/ppx/ppy` — `R` is shared; and `w->warp` at compose time
uses `cameras_[i].R` (f32).

---

## 13. Recommended porting order

Follow the data flow; every stage is testable against the Python oracle using recorded inputs
from the previous stage, so bugs never alias across stages:

1. **Core kit first** (prerequisites, not a stage): `cvRound`, `cv::RNG`, `INTER_LINEAR_EXACT`
   resize, fixed-point `remap`, `pyrDown/pyrUp` (16S + f32), `Rodrigues`, symmetric Jacobi
   `eigen`, `distanceTransform(DIST_L1)`. Everything below leans on these.
2. **ORB** — pure function of one image; largest constant surface; bit-exact target proves the
   core kit.
3. **Matcher + RANSAC/findHomography** — consumes ImageFeatures; deterministic once BF-Hamming
   replaces LSH and cv::RNG is ported; establishes `MatchesInfo` exactly.
4. **Homography-based rotation estimation** (leaveBiggestComponent, spanning tree,
   estimateFocal, CalcRotation) — small, double-precision, easy to verify; gives valid
   `CameraParams` so later stages can run even before BA works.
5. **BundleAdjusterRay** — hardest numerics (LevMarq); by now its inputs are bit-controlled.
   Port `calcError` first, diff standalone, then the solver.
6. **waveCorrect** — 40 lines, unblocks visually correct output orientation.
7. **Spherical warper** (+ buildMaps/warpRoi/remap integration) — first stage producing images;
   corners/ROI parity is the gate for everything after.
8. **Gain compensation** — needs warped images+corners; scalar GainCompensator first, then
   blocks.
9. **Graph-cut seams** — port Voronoi or DpSeamFinder as a stopgap (pipeline runs end-to-end),
   then GCGraph+BK for parity.
10. **Multiband blend** — FeatherBlender first as a scaffold, then MultiBandBlender; final
    visual+numeric sign-off on full-pipeline fixtures.

Rationale: 2–5 are the "registration" half where errors are invisible until the very end —
they must be locked numerically before any pixels are produced; 7–10 are the "compositing" half
where each stage is visually debuggable and tolerances can be looser.

---

## 14. Master table of tuning constants the Rust port must replicate

| Stage | Constant | Value | Source |
|---|---|---|---|
| Stitcher | registration_resol | 0.6 MP | stitcher.cpp:57 |
| Stitcher | seam_est_resol | 0.1 MP | stitcher.cpp:58 |
| Stitcher | compositing_resol | ORIG_RESOL (−1 ⇒ full) | stitcher.cpp:59, stitching.hpp:146 |
| Stitcher | pano conf_thresh | 1.0 | stitcher.cpp:60 |
| Stitcher | work/seam/compose scale | `min(1, sqrt(res·1e6/area₀))` (first image) | stitcher.cpp:434,441,250 |
| Stitcher | compose resize dead zone | `abs(scale−1) > 1e-1` | stitcher.cpp:284 |
| Stitcher | interp | INTER_LINEAR (+ BORDER_REFLECT warps) | stitcher.cpp:64,194,306 |
| ORB | nfeatures / scaleFactor / nlevels | 500 / 1.2 / 8 | features2d.hpp:460 |
| ORB | edgeThreshold / patchSize / fastThreshold | 31 / 31 / 20 | features2d.hpp:460 |
| ORB | WTA_K / scoreType / firstLevel | 2 / HARRIS / 0 | features2d.hpp:460 |
| ORB | HARRIS_K, Harris block | 0.04f, 7 | orb.cpp:50,944 |
| ORB | descriptor blur | Gauss 7×7 σ=2 REFLECT_101 | orb.cpp:1234 |
| ORB | pyramid border | 32, REFLECT_101(+ISOLATED) | orb.cpp:1031 |
| Matcher | match_conf (ratio 1−c) | 0.3f | matchers.hpp:196 |
| Matcher | num_matches_thresh1/2 | 6 / 6 | matchers.hpp:196 |
| Matcher | matches_confindece_thresh | 3.0 (conf>3 ⇒ 0) | matchers.hpp:197, matchers.cpp:443 |
| Matcher | confidence | `ni/(8+0.3·nm)` | matchers.cpp:439 |
| Matcher | point centering | `p − 0.5·(w,h)` | matchers.cpp:416 |
| RANSAC | reproj threshold | 3.0 px | calib3d.hpp:843, fundam.cpp:367 |
| RANSAC | confidence / maxIters | 0.995 / 2000 | calib3d.hpp:844 |
| RANSAC | subset attempts / LM refine iters | 10000 / 10 | ptsetreg.cpp:208, fundam.cpp:433 |
| RANSAC | RNG seed | `(uint64)-1`, MWC coeff 4164903690 | ptsetreg.cpp:171 |
| Graph | leaveBiggestComponent threshold | conf ≥ 1.0 | stitcher.cpp:474 |
| Graph | spanning-tree edge weight | num_inliers | motion_estimators.cpp:1151 |
| BA | conf_thresh (edge filter) | 1.0 | stitcher.cpp:512 |
| BA | TermCriteria | COUNT+EPS, 1000, DBL_EPSILON | motion_estimators.hpp:163 |
| BA | Ray: params/errs, FD step | 4 / 3, 1e-3 | motion_estimators.cpp:507,551,628 |
| BA | Reproj: params/errs, FD step | 7 / 2, 1e-4 | motion_estimators.cpp:328,376,447 |
| BA | Ray error scaling | `sqrt(f1·f2)·Δray` | motion_estimators.cpp:612 |
| BA | refinement mask default | all-ones 3×3 | motion_estimators.hpp:162 |
| LevMarq | λ init / schedule | 1e-3; ÷10 / ×10 (≤16), diag×(1+λ), SVD solve | calib3d/private.hpp:58-68 |
| waveCorrect | kind | HORIZ; rg1 = eigvec row 2 | stitcher.cpp:77, motion_estimators.cpp:952 |
| Warper | scale | median focal × tier aspect | stitcher.cpp:184,258,517-528 |
| Warper | ROI int conversion | trunc toward 0 | warpers_inl.hpp:177 |
| Warper | mask warp | NEAREST + CONSTANT(0) | stitcher.cpp:197,315 |
| Gain | alpha / beta | 0.01 / 100 | exposure_compensate.cpp:215-216 |
| Gain | intensity | L2 norm of BGR | exposure_compensate.cpp:189 |
| Gain | N(i,j) floor | max(1, count) | exposure_compensate.cpp:165 |
| Blocks | block size / filter | 32×32; [.25 .5 .25]² ×2 iters | exposure_compensate.hpp:172-173, cpp:509-527 |
| Gain | nr_feeds / similarity_threshold | 1 / 1 (off) | exposure_compensate.hpp:115-117 |
| Seam | cost type (Stitcher) | COST_COLOR | stitcher.cpp:61 |
| Seam | terminal_cost / bad_region_penalty | 10000 / 1000 | seam_finders.hpp:243-244 |
| Seam | weight_eps / gap | 1.0f / 10 px | seam_finders.cpp:1181,1274 |
| DpSeam | badRegionCost | √3·255 ≈ 441.673 | seam_finders.cpp:772 |
| Blend | num_bands | min(5, ceil(log2(max_dim))) | blenders.hpp:130, blenders.cpp:239 |
| Blend | feed gap / alignment | 3·2ⁿ / snap to 2ⁿ | blenders.cpp:369-385 |
| Blend | WEIGHT_EPS | 1e-5f | blenders.cpp:66 |
| Blend | weight map | mask/255 f32, CONSTANT border | blenders.cpp:513,523 |
| Blend | img border in feed | BORDER_REFLECT | blenders.cpp:492 |
| Feather | sharpness | 0.02f | blenders.hpp:103 |
| Compose | seam mask upscale | dilate 3×3 ×1 → resize LINEAR_EXACT → AND | stitcher.cpp:334-337 |
| Compose | accumulation type | CV_16SC3, final saturate to CV_8U | stitcher.cpp:328,373 |

---

## 15. OpenCV behaviors NOT worth replicating

1. **FLANN-LSH for binary descriptors** (`matchers.cpp:170-176`). Approximate and randomized;
   the per-pair `theRNG()` reseeding (`matchers.cpp:74-78`) exists only to paper over it. Use
   exact brute-force Hamming 2-NN. Strictly a quality *improvement* (more true matches);
   documented divergence from the C++ oracle, none from a `BFMatcher`-based Python oracle.
2. **Eigen-float32 gain solve** (`exposure_compensate.cpp:251-268`). Build-dependent precision.
   Solve in f64 Cholesky and give tests a tolerance instead of emulating `Eigen::LLT<float>`.
3. **UMat/OpenCL and CUDA paths everywhere** (`ocl_*` kernels in blenders/warpers/matchers,
   `GpuMatcher`, `GraphCutSeamFinderGpu`, `MultiBandBlender` CUDA branch, `createLaplacePyrGpu`).
   The scalar CPU loops are the reference; the GPU paths differ numerically from their own CPU
   fallbacks (e.g. `GpuMatcher` uses NORM_L1 for ORB, admitted as wrong in the comment at
   `matchers.cpp:226-229`).
4. **CV_16S weight_type in MultiBandBlender** (`>>8` accumulate, `w+1` normalize,
   `blenders.cpp:570-586,750-767`) — dead by default, strictly worse precision.
5. **`BestOf2NearestRangeMatcher`** and the affine family (`AffineBestOf2NearestMatcher`,
   `AffineBasedEstimator`, `BundleAdjusterAffine*`, `AffineWarper`) — SCANS mode, out of scope.
6. **The double insertion of spanning-tree edges** (both `(i,j)` and `(j,i)` with identical
   weight, `motion_estimators.cpp:1145-1155`) — harmless redundancy; a port can insert each
   undirected edge once, provided the sort order (weight desc) is preserved.
7. **`Stitcher::composePanorama(images, ...)` re-feed path** (`stitcher.cpp:137-162`) — the
   variant that swaps in new full-res images after `estimateTransform`; keep the single-shot
   `stitch()` contract initially.
8. **`nr_feeds > 1` / similarity-mask machinery** in exposure compensation — off by default,
   adds an erode/dilate + per-pixel similarity path few people use.
9. **Legacy quirks to be aware of but not emulate as API:** the misspelled
   `matches_confindece_thresh` (keep the *behavior*, fix the name); `LOGLN` timing scaffolding;
   `#if 0` focal-from-rotating-camera block in `HomographyBasedEstimator`
   (`motion_estimators.cpp:138-157`).
10. **DpSeamFinder's full state machine** (component classification, corner finding,
    `hasOnlyOneNeighbor` etc., ~900 lines) — if graph-cut is ported, Dp is redundant; if used
    as a stopgap, a clean-room monotone-DP seam is fine since it is not the parity target.

What must NOT be "fixed", despite looking like bugs: truncation-toward-zero ROI corners
(§8), `short` truncation in blending (§11), first-image-only scale computation (§0),
confidence-zeroing of too-good matches (§2), and the `max(1, N)` overlap floor (§9). These all
shift pixels or camera parameters and are part of observable behavior.

---

## 16. Oracle harness sketch

All `cv::detail` classes are exposed in `cv2.detail`. One Python script can dump every stage
boundary for a fixture set:

```
gray → orb.detectAndCompute            → features.npz (kps, desc)
features → BestOf2NearestMatcher.apply2 → matches.npz (per-pair idx, mask, H, conf)
        (parallel: BFMatcher+findHomography reimpl for the Rust-facing oracle)
matches → HomographyBasedEstimator      → cameras0.npz (focal, ppx, ppy, R)
cameras0 → BundleAdjusterRay            → cameras1.npz
cameras1 → waveCorrect                  → cameras2.npz
cameras2 → PyRotationWarper('spherical')→ corners, sizes, warped imgs/masks (.npz per tier)
warped → ExposureCompensator(GAIN_BLOCKS)→ gain maps (getMatGains)
gains  → GraphCutSeamFinder('COST_COLOR')→ seam masks
all    → MultiBandBlender               → pano16s + mask, and Stitcher.stitch() end-to-end
```

Fixture recommendations: 2-, 3-, and 6-image sets; one set with a rotation putting a pole in
view; one with mixed exposures; one duplicate pair (exercises the conf>3 rejection); one
non-overlapping outlier image (exercises leaveBiggestComponent). Store fixtures at a size where
`work_scale < 1` **and** one where `work_scale == 1`.
