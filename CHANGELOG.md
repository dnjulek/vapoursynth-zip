# Changelog

## Unreleased

Changes since 23.1.0 (Tonemap and BM3D work is not listed here).

### Changed

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
- **BoxBlur:** the compile-time float kernels are now used only for radius 1-3 (previously 1-8). Re-measured, the running-sum path is faster from radius 4 on: 1080p 1T fps at radius 8 is 598 against 248. Float output for radius 4-8 changes by about 1e-5.
- **AdaptiveGrainMask (float):**
  - The mask is computed as 2^(temp * log2(base)) with a shorter log2 and vszip's polynomial exp2, in two passes over L1-sized blocks. That's within about 1e-6 relative of libm `powf`.
  - The exponent is capped at 127.99, so extreme settings give a large finite value instead of inf.
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
- **Builds without hardware FMA** (the baseline `x86_64` wheel): every kernel now goes through `simd.madd` instead of calling `@mulAdd` directly. Without FMA, `@mulAdd` compiled to a libm `fmaf` call per lane; one AdaptiveGrainMask loop made 180 of them. FMA builds (the `.avx2`/`.zn4` wheels, CI) produce identical output; no-FMA builds may differ in the last bit.