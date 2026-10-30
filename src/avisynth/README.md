# vszip for AviSynth+ (avszip)

The same repository builds an AviSynth+ plugin, `avszip`, from the same kernels as the VapourSynth plugin (`src/filters/`), on top of [avisynth-zig](https://github.com/dnjulek/avisynth-zig). On identical input its outputs are **bit-exact with vszip**: a differential test suite (`tests/avisynth/`) runs both plugins in one process and compares planes and frame properties.

Requirements: AviSynth+ with interface ≥ 11 (3.7.4+) and Zig 0.17.0. Windows x86_64 is the tested path; the plugin also cross-compiles for Linux.

To build it yourself:

```sh
zig build -Doptimize=fast -Davisynth=true                     # vszip.dll + avszip.dll
zig build -Doptimize=fast -Davisynth=true -Dvapoursynth=false # avszip.dll only
```

The AviSynth build is opt-in: a plain `zig build` (what pip/hatch run) builds only the VapourSynth plugin and never fetches the `avisynth` dependency (it is a lazy dependency in `build.zig.zon`).

```avs
LoadPlugin("avszip.dll")
src = LWLibavVideoSource("in.mkv").ConvertBits(16)
src.avszip_Deband(thr=[2.0], grain=[1.0]).avszip_Limiter(tv_range=true)
```

Every function is registered as `avszip_<Filter>`. AviSynth+ also exposes it as `avszip_avszip_<Filter>`, the DLL-basename alias. Parameter docs are vszip's [wiki](https://github.com/dnjulek/vapoursynth-zip/wiki): names, defaults, and ranges are the same. Arrays use AviSynth+ syntax (`planes=[0, 1]`, or a scalar `planes=0`). Every parameter after the first clip is named, so `avszip_EEDI3(field=1)`, `avszip_Dither(bitdepth=8)` and positional calls both work.

## Functions

| AviSynth+ signature | Notes |
| --- | --- |
| `avszip_AdaptiveBinarize(clip, clip clip2, int "c")` | 8-bit; output `_ColorRange`=0 |
| `avszip_AdaptiveGrainMask(clip, float "luma_scaling")` | returns Gray (Y8..Y32) |
| `avszip_Bilateral(clip, clip "ref", float[] "sigmaS", float[] "sigmaR", int[] "planes", int[] "algorithm", int[] "PBFICnum")` | |
| `avszip_BilateralDither(clip, clip "ref", int[] "radius", float[] "thr", float[] "flat", float[] "wmin", float[] "subspl", int[] "planes")` | |
| `avszip_BM3D(clip, clip "ref", float[] "sigma", int[] "block_step", int[] "bm_range", int "radius", int[] "ps_num", int[] "ps_range", bool "chroma", string[] "bm_error_s", string[] "transform_2d_s", string[] "transform_1d_s")` | 32-bit float; vszipcu's BM3D; `radius` > 0 is aggregated inside the filter, as in vszip |
| `avszip_BoxBlur(clip, int[] "planes", int "hradius", int "hpasses", int "vradius", int "vpasses")` | |
| `avszip_Checkmate(clip, int "thr", int "tmax", int "tthr2")` | 8-bit, temporal |
| `avszip_CLAHE(clip, int "limit", int[] "tiles")` | output `_ColorRange`=0 |
| `avszip_ColorMap(clip, int "color")` | Y8 in, RGBP out |
| `avszip_CombMask(clip, int "cthresh", int "mthresh", bool "expand", int "metric")` | 8-bit |
| `avszip_CombMaskMT(clip, int "thY1", int "thY2")` | 8-bit |
| `avszip_Compress(clip, int "codec", int "qscale", int "quality", int "dc_prec", bool "chroma")` | 8-bit Gray/YUV |
| `avszip_Deband(clip, int "range", float[] "thr", float[] "grain", int "sample_mode", int "seed", bool "blur_first", bool "dynamic_grain", bool "keep_tv_range", int "random_algo_ref", int "random_algo_grain", float "random_param_ref", float "random_param_grain", float[] "thr1", float[] "thr2", float "angle_boost", float "max_angle")` | <16-bit int is processed at 16 bit, as in vszip |
| `avszip_Dither(clip, int bitdepth, int "dither_type", int "sample_type", bool "fulls", bool "fulld")` | 12 modes (fmtconv + zimg) |
| `avszip_EEDI3(clip, int field, bool "dh", float "alpha", float "beta", float "gamma", int "nrad", int "mdis", bool "hp", int "vcheck", float "vthresh0", float "vthresh1", float "vthresh2", clip "sclip", clip "mclip")` | field 2/3 bob: 2× frames and fps |
| `avszip_EEDI3H(...)` | same parameters, interpolates along the width |
| `avszip_ImageRead(string[] path, bool "validate")` | source filter (zigimg; files or http(s) URLs) |
| `avszip_LimitFilter(clip flt, clip src, clip "ref", float[] "dark_thr", float[] "bright_thr", float[] "elast", int[] "planes")` | |
| `avszip_Limiter(clip, float[] "min", float[] "max", bool "tv_range", bool "mask", int[] "planes")` | |
| `avszip_MosquitoNR(clip, int[] "strength", int[] "restore", int[] "radius", int[] "planes")` | |
| `avszip_NLMeans(clip, int "d", int "a", int "s", float "h", string "channels", int "wmode", float "wref", clip "rclip")` | KNLMeansCL port; channels `Y`/`UV`/`YUV`/`RGB`/`auto`, temporal |
| `avszip_PlaneAverage(clip, int[] exclude, clip "clipb", int[] "planes", string "prop")` | property-only (`psmAvg`/`psmDiff`) |
| `avszip_PlaneMinMax(clip, float "minthr", float "maxthr", clip "clipb", int[] "planes", string "prop")` | property-only (`psmMin`/`psmMax`/`psmDiff`) |
| `avszip_RFS(clip, clip clipb, int[] frames, bool "mismatch", int[] "planes")` | |
| `avszip_SSIMULACRA2(clip reference, clip distorted)` | property `SSIMULACRA2` |
| `avszip_Tonemap(clip, int "src_csp", int "dst_csp", int "dst_prim", float "src_max", float "src_min", float "dst_max", float "dst_min", bool "dynamic_peak_detection", float "smoothing_period", float "scene_threshold_low", float "scene_threshold_high", float "percentile", float "black_cutoff", int "gamut_mapping", int "tone_mapping_function", string "tone_mapping_function_s", float "tone_mapping_param", int "metadata", bool "use_dovi", clip "dovi_el", bool "show_clipping", float "contrast_recovery", float "contrast_smoothness", int "lut_size", float "mastering_clip", float "film_strength", bool "inverse_tone_mapping", float "knee_adaptation", float "knee_minimum", float "knee_maximum", float "knee_default", float "knee_offset", float "slope_tuning", float "slope_offset", float "spline_contrast", float "reinhard_contrast", float "linear_knee", float "exposure", bool "gamut_expansion", int[] "lut3d_size", bool "lut3d_tricubic", float "perceptual_deadzone", float "perceptual_strength", float "colorimetric_gamma", float "softclip_knee", float "softclip_desat", bool "cache_frame")` | 10-16 bit or float RGB / YUV 4:2:0, 4:2:2, 4:4:4; out RGBP16 / YUV444P16 (float: RGBPS / YUV444PS); HDR10, HLG, Dolby Vision (`DolbyVisionRPU` prop, FEL via `dovi_el`) |
| `avszip_WNNM(clip, float[] "sigma", int "block_size", int "block_step", int "group_size", int "bm_range", int "radius", int "ps_num", int "ps_range", bool "residual", bool "adaptive_aggregation", clip "rclip")` | 32-bit float |
| `avszip_XPSNR(clip reference, clip distorted, bool "temporal", bool "verbose")` | props `XPSNR_Y/U/V` |

`avszip_WNNMRaw` is registered as well. It is WNNM's internal first stage, a function only so that the host caches it; do not call it directly.

vszip's int-typed booleans (`tv_range=1`) are AviSynth `bool`s here (`tv_range=true`). Enum-like ints such as `metric`, `codec`, and `dither_type` stay ints.

## Differences from vszip

AviSynth+ is a different host, so a few behaviors are adapted rather than copied:

- **Formats.** AviSynth+ has no half-float, 9-bit, or 32-bit-integer planar formats, so vszip's f16/u9/u32 paths do not exist. Alpha planes of YUVA/RGBAP clips pass through untouched (mask filters such as CombMask copy the input's alpha too); vszip never sees them because VapourSynth keeps alpha separately. `Dither` converts alpha as full-range data with plain rounding (never dithered), so internal depth round trips (Deband below 16 bit, XPSNR's depth match) return it unchanged. `EEDI3(dh=true)` rejects alpha. SSIMULACRA2 keeps a YUVA reference's alpha (see below). Secondary clips may differ from the primary in alpha presence, except in PlaneAverage/PlaneMinMax when the alpha plane is selected (`planes=[3]`): then `clipb` must carry alpha too.
- **Range metadata.** AviSynth+ uses `_ColorRange` (0 = full, 1 = limited), while VapourSynth R4 uses `_Range` (1 = full). Range-dependent filters (Deband, LimitFilter) read `_ColorRange`, and filters that tag full range write it.
- **RFS `mismatch`.** AviSynth clips cannot vary in format or size, so `mismatch=true` only tolerates a frame-rate difference; the output keeps clipa's rate.
- **XPSNR averages.** vszip adds to running totals per evaluation, so a frame recomputed by the cache counts twice. Here each frame index stores its result once, and the printed average reduces over the evaluated frames in index order (it reports "N of M frames"). Per-frame props are identical.
- **SSIMULACRA2 on YUV.** YUV with 2x or no subsampling (8-16 bit int, 32-bit float) is converted to RGB inside the filter by the same code as vszip (each frame's `_ColorRange` (limited when unset), `_Matrix` (BT.709, BT.601, BT.2020 NCL, FCC, 240M; 2 or unset uses BT.709 above 650 lines, else BT.601; other matrices raise a frame error) and `_ChromaLocation`, zimg's Bicubic chroma upsampling), so those scores match vszip; the reference (with any alpha) is returned unchanged. Metadata may change from frame to frame, independently on both inputs. Gray and other YUV layouts are normalized per frame with their own `_ColorRange`, upsampled in float by AviSynth's `ConvertToYUV444` and converted with the frame's `_Matrix`: their scores differ slightly from vszip's (zimg), and such a reference is returned as RGBPS, or RGBAPS carrying its alpha (integer alpha as full-range float). The distorted clip's alpha is ignored.
- **Tonemap.** The input range is read from `_ColorRange` and written there on output (`_ChromaLocation` is removed, as vszip removes it). The output has no alpha plane: an input alpha is dropped. AviSynth+ has no 4:4:0 or 9-bit formats, so those vszip inputs (BL or `dovi_el`) don't exist here. Peak detection and Dolby Vision RPU chains fetch the earlier frames they need synchronously; the results are the same as vszip's in any request order.
- **BM3D temporal cache.** `radius` > 0 aggregates through the same in-filter cache as vszip. Its eviction threshold counts requests in flight; AviSynth+ doesn't report the `Prefetch` thread count, so it assumes one per CPU. That only affects memory and speed, not the output.
- **ImageRead alpha.** RGBA images use the native A plane (RGBAP). Gray+alpha has no AviSynth planar format, so it returns Y plus a separate Y frame in the `_Alpha` frame property (vszip's representation).
- **ImageRead later frames.** Every frame is checked when decoded, not only with `validate=true`: a file whose size or decoded layout (gray/RGB, depth, alpha) differs from the first image raises a frame error. The VapourSynth build does the same.
- **Scratch pools.** vszip keys EEDI3/WNNM/NLMeans/BM3D scratch by thread ID. Here every request checks a block out of a mutex-guarded pool and returns it, so blocks are exclusive per request, including nested use on one thread.

Every filter registers `AVS_MT_NICE_FILTER`, and frame n depends only on n (temporal filters clamp neighbors like vszip). On AviSynth+ ≤ 3.7.5, concurrent C-filter calls share an error slot, so under `Prefetch` an error message may name the wrong frame. Successful frames are unaffected.

## Layout

- `src/filters/`: the kernels, shared by both plugins and host-independent (`src/common.zig` holds the globals and pure helpers both hosts use). Argument derivations that are not host-specific live there too (e.g. `bilateral.Data.init`, `eedi3.Data.check`/`init`, `wnnm.Config`, `dither.Config`, `boxblur_runtime.Params`, `deband_setup.zig`, `image_read.zig`), so the adapters only parse arguments and move frames. `zig build test` compiles every kernel without either host binding.
- `src/vapoursynth/` and `src/avisynth/`: one adapter per filter and host.
- `src/avisynth/helper.zig`: AviSynth-side helpers: format checks, range conventions, unprocessed-plane copies, the props sink, and **canonical sources**. A plane that is unaligned or has a foreign pitch (horizontal crops, subframes) is repacked to the output pitch, so the single-stride kernels and `fullSlice()` stay valid. Freshly allocated upstream frames are never copied.
- `src/avszip.zig`: AviSynth plugin entry and registration.

## Tests

```sh
zig build -Davisynth=true && python -m pytest tests/avisynth -q
zig build -Doptimize=fast -Davisynth=true && python -m pytest -q   # both suites
```

The AviSynth suite needs Python with `vapoursynth`, `numpy`, `pytest`, and `Pillow`, plus an installed AviSynth+. It loads `zig-out/bin/avszip.dll` and `zig-out/bin/vszip.dll` (override with `AVSZIP_LIB` / `VSZIP_LIB`). Without the DLL or AviSynth+ the directory is skipped at collection, so the default `pytest` run (and CI) is unaffected. `tests/avisynth/avsrt.py` drives AviSynth+ through its C API with ctypes; its numpy-backed source feeds both hosts bit-identical frames, and `tests/avisynth/harness.py` compares planes and props.