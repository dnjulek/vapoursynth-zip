# Changelog

## 24.0.0

Changes since 23.1.0.

### Added

- **Butteraugli** (VapourSynth and AviSynth+): libjxl master's Butteraugli, the comparison `butteraugli_main` runs. avszip's `avszip_Butteraugli` is bit-exact with vszip and keeps an RGBAP reference's alpha.
  - RGB input, 8-16-bit integer or 32-bit float, read as sRGB (or as linear light when frame 0 has `_Transfer=8`).
  - Per-frame properties `Butteraugli` (the distance) and `Butteraugli3Norm`. `distmap=1` returns libjxl's heatmap, `distmap=2` the raw difference map.
  - Parameters `intensity_target`, `hf_asymmetry` and `xmul`, as in libjxl. The default `hf_asymmetry=1.0` is `butteraugli_main`'s.
  - Matches `butteraugli_main` within float rounding: on real frames up to 1080p the distance agrees to about one float32 ulp and each difference-map pixel to about 1e-6.
  - 1080p RGB24: 10.4 fps on one thread and 22 fps on 16. The blurs stream through row rings, as SSIMULACRA2's blur does, and the steps around them run on the blurs' output rows instead of in separate full-plane passes.

- **NLMeans** (VapourSynth and AviSynth+): a CPU port of vszipcu's CUDA NLMeans, with KNLMeansCL's arguments and defaults (`d`, `a`, `s`, `h`, `channels`, `wmode`, `wref`, `rclip`; no OpenCL device arguments). It also accepts 9-15 bit input. The output is close to the GPU versions but not guaranteed bit-identical.
- **BM3D** (VapourSynth and AviSynth+): a CPU port of vszipcu's BM3DCUDA `BM3D`, with the same arguments minus those that only serve the GPU (`device_id`, `num_streams`, `fast_fused`, `zero_init`, `extractor_exp`).
  - 32-bit float input. Results are close to the GPU's but not bit-identical (mean difference about 4e-8): the GPU's fast-math reciprocal and divide are not emulated.
  - `radius > 0` is aggregated inside the filter: it returns the final clip, with no stacked clip, so there is no `BM3Dv2` or `VAggregate` (they exist in the GPU plugins only to finish the stacked clip).
  - With every `sigma` 0 the clip is returned unchanged, like the other filters.
- **Tonemap** (VapourSynth and AviSynth+): vs-placebo's Tonemap (libplacebo v7.387) on the CPU.
  - HDR10, HLG and Dolby Vision sources. Dolby Vision reads the `DolbyVisionRPU` frame property, including RPUs that continue an earlier one, and composes the FEL enhancement layer given as `dovi_el`.
  - Input: 9-16 bit integer (10-16 on AviSynth+) or 32-bit float RGB or YUV. Output: RGB48 / YUV444P16 (float: RGBS / YUV444PS).
  - Every output-affecting libplacebo color-map, peak-detection and tone/gamut constant parameter is an argument, under libplacebo's name.
  - Peak detection is deterministic: frame n depends only on the frames before it, whatever order frames are requested in.
  - `cache_frame` (default on) keeps a frame's linear RGB from the detection pass instead of decoding it twice; turning it off saves memory.
  - The output is close to vs-placebo's but not bit-identical (no GPU half-float buffers, approximate transfer functions, rounded instead of dithered 16-bit output).

### Fixed

- **Clips of up to 2^31 - 1 frames:** WNNM with `radius > 0` and Checkmate with `tthr2 > 0` overflowed a 32-bit sum when forming the neighbouring frame numbers of the last frames of such a clip. That panicked in debug builds and produced invalid frame numbers in release builds. Neighbour indices are now computed in 64 bits; the new BM3D, NLMeans and Tonemap filters use the same code.

- **SSIMULACRA2 precision (VapourSynth and AviSynth+).** The SSIM denominator was formed from blur((a+b)^2), whose float32 rounding in smooth areas came out as a positive bias after the `max(d, 0)` clip. Scores were up to 1 point low on real 1080p frames and 6-7 points low on near-identical flat ones. It's now blur((a-b)^2) - (mu1-mu2)^2 plus the covariance term: same speed, and within 0.0006 (p95) of a float64 evaluation over 528 real-frame pairs, closer than libjxl's own float32 math. Most scores move by less than 0.03. Builds without FMA now agree with FMA builds too.
- **SSIMULACRA2 (VapourSynth) no longer crashes when frame 0 of an input fails.** Creation reads frame 0 for its frame properties and used a failed (null) frame as is. The upstream error is now reported as the filter's error.

### Changed

- **SSIMULACRA2 follows libjxl's reference (VapourSynth and AviSynth+).** Scores move by up to several points; the test image's blur cases rose by 3-4.5.
  - The blur is libjxl's: its recursive Gaussian is exactly a 9-tap kernel, now used with zero padding at the plane edges like libjxl. The previous kernel was a sampled Gaussian with mirrored edges.
  - Like libjxl, it stops at the first scale below 8 px and weights the scales as libjxl does.
  - It matches libjxl's algorithm evaluated in float64 within 3e-4. Speed is unchanged.
  - Against libjxl's `ssimulacra2` tool on 1056 real-frame pairs: mean difference 0.39, down from 2.8 (max 4.7, down from 72). What remains is libjxl's own float32 error, which lowers its scores of high-quality pairs.

- **SSIMULACRA2 converts YUV inside the filter (VapourSynth and AviSynth+).**
  - YUV with 2x or no chroma subsampling, 8-16 bit or float, is converted to RGB row by row inside the filter. It no longer goes through an internal `resize.Bicubic(format=RGBS)` node.
  - The conversion follows zimg's steps and reads each frame's own `_Matrix`, `_Range` (falling back to `_ColorRange`) and `_ChromaLocation`.
  - Those queued 1080p RGBS frames cost about 2.3 GB of RAM at 16 threads and a third of the throughput.
  - Scores move slightly, because the conversion rounds differently from zimg (about 1e-7 on the RGB values). The test goldens changed by about 0.002, and by 0.03 for the tiny clip.
  - Gray and other subsamplings still use the internal conversion. On AviSynth+, the reference is then returned as RGBPS (RGBAPS with alpha).
- **WNNM with `radius > 0` (VapourSynth) aggregates the temporal window inside the filter.**
  - It uses BM3D's temporal cache, streaming rows through a per-thread ring, instead of building a stacked raw frame for an internal VAggregate.
  - Output is identical: the summation order is the same.
  - Peak RAM at r=1, 1080p, 16 threads went from 3.6 to 1.7 GB.
- **WNNM with `radius > 0` is 24-35% faster (VapourSynth and AviSynth+).** The temporal search now builds the union of its predictive windows as row runs instead of merging position lists one window at a time. Output is bit-identical. At 1080p chroma-only (`sigma=[0, 3, 3]`), 16 threads, radius 1 went from 169 to 210 fps and radius 2 from 110 to 148 fps.
- **BoxBlur:** the compile-time float kernels are now used only for radius 1-3 (previously 1-8). Re-measured, the running-sum path is faster from radius 4 on: 1080p 1T fps at radius 8 is 598 against 248. Float output for radius 4-8 changes by about 1e-5.
- **AdaptiveGrainMask (float):**
  - The mask is computed as 2^(temp * log2(base)) with a shorter log2 and vszip's polynomial exp2, in two passes over L1-sized blocks. That's within about 1e-6 relative of libm `powf`.
  - The mask saturates at 2^127.5 (about 2.4e38), as in 23.1.0, so extreme settings give a large finite value instead of inf.
  - Integer formats are unchanged.
- **PlaneAverage and PlaneMinMax (float):** the float sums (PlaneAverage's average and `Diff`, PlaneMinMax's `Diff` with a reference clip) are vectorized with per-lane f64 accumulators instead of one serial chain. Results change by about 1e-16 relative.

### Performance

- **Bilateral, algorithm 1:**
  - The PBFIC levels are streamed one at a time instead of being kept as 16-17 full planes, which took about 140 MB per 1080p luma call at the default `sigmaR`. That made the filter bandwidth- and page-fault-bound at 16 threads.
  - The horizontal recursive Gaussian now runs 8 rows at a time with 8x8 transposes, and each recursion step depends on the previous output through a single FMA.
  - Vectors with no pixel at a level skip that level's divide and memory traffic, which is most of them on real content.
  - The test goldens are unchanged.
- **CLAHE:**
  - Each tile's LUT is built only over the histogram bins its 3x3 neighbourhood actually uses.
  - Clipping, redistribution and the cumulative sum are fused into one pass.
  - The four sub-histograms are only used when a tile has many pixels per bin.
  - The test goldens are unchanged.
- **Builds without hardware FMA** (the baseline `x86_64` wheel): every kernel now goes through `simd.madd` instead of calling `@mulAdd` directly. Without FMA, `@mulAdd` compiled to a libm `fmaf` call per lane; one AdaptiveGrainMask loop made 180 of them. FMA builds (the `.avx2`/`.zn4` wheels, CI) produce identical output; no-FMA builds may differ in the last bit of float output. Integer AdaptiveGrainMask stays bit-exact with the Rust plugin on every build: its LUT bases are computed at compile time with fused operations.