# vszip for AviSynth+ (avszip)

The same repository builds an AviSynth+ plugin, `avszip`, from the same kernels as the VapourSynth plugin (`src/filters/`), on top of [avisynth-zig](https://github.com/dnjulek/avisynth-zig). On identical input its outputs are **bit-exact with vszip**: a differential test suite (`tests/avisynth/`) runs both plugins in one process and compares planes and frame properties.

Requirements: AviSynth+ with interface ≥ 11 (3.7.4+) and Zig 0.17.0. Windows x86_64 is the tested path; the plugin also cross-compiles for Linux.

```sh
zig build -Doptimize=fast -Davisynth=true                     # vszip.dll + avszip.dll
zig build -Doptimize=fast -Davisynth=true -Dvapoursynth=false # avszip.dll only
```

The AviSynth build is opt-in: a plain `zig build` (what pip/hatch run) builds only the VapourSynth plugin and never fetches the `avisynth` dependency (it is a lazy dependency in `build.zig.zon`).

```avs
LoadPlugin("avszip.dll")
src = LWLibavVideoSource("in.mkv").ConvertBits(16)
src.vszip_Deband(thr=[2.0], grain=[1.0]).vszip_Limiter(tv_range=true)
```

Every function is registered as `vszip_<Filter>`. AviSynth+ also exposes it as `avszip_vszip_<Filter>`, the DLL-basename alias. Parameter docs are vszip's [wiki](https://github.com/dnjulek/vapoursynth-zip/wiki): names, defaults, and ranges are the same. Arrays use AviSynth+ syntax (`planes=[0, 1]`, or a scalar `planes=0`). Every parameter after the first clip is named, so `vszip_EEDI3(field=1)`, `vszip_Dither(bitdepth=8)` and positional calls both work.

## Functions

| AviSynth+ signature | Notes |
| --- | --- |
| `vszip_AdaptiveBinarize(clip, clip clip2, int "c")` | 8-bit; output `_ColorRange`=0 |
| `vszip_AdaptiveGrainMask(clip, float "luma_scaling")` | returns Gray (Y8..Y32) |
| `vszip_Bilateral(clip, clip "ref", float[] "sigmaS", float[] "sigmaR", int[] "planes", int[] "algorithm", int[] "PBFICnum")` | |
| `vszip_BilateralDither(clip, clip "ref", int[] "radius", float[] "thr", float[] "flat", float[] "wmin", float[] "subspl", int[] "planes")` | |
| `vszip_BoxBlur(clip, int[] "planes", int "hradius", int "hpasses", int "vradius", int "vpasses")` | |
| `vszip_Checkmate(clip, int "thr", int "tmax", int "tthr2")` | 8-bit, temporal |
| `vszip_CLAHE(clip, int "limit", int[] "tiles")` | output `_ColorRange`=0 |
| `vszip_ColorMap(clip, int "color")` | Y8 in, RGBP out |
| `vszip_CombMask(clip, int "cthresh", int "mthresh", bool "expand", int "metric")` | 8-bit |
| `vszip_CombMaskMT(clip, int "thY1", int "thY2")` | 8-bit |
| `vszip_Compress(clip, int "codec", int "qscale", int "quality", int "dc_prec", bool "chroma")` | 8-bit Gray/YUV |
| `vszip_Deband(clip, int "range", float[] "thr", float[] "grain", int "sample_mode", int "seed", bool "blur_first", bool "dynamic_grain", bool "keep_tv_range", int "random_algo_ref", int "random_algo_grain", float "random_param_ref", float "random_param_grain", float[] "thr1", float[] "thr2", float "angle_boost", float "max_angle")` | <16-bit int is processed at 16 bit, as in vszip |
| `vszip_Dither(clip, int bitdepth, int "dither_type", int "sample_type", bool "fulls", bool "fulld")` | 12 modes (fmtconv + zimg) |
| `vszip_EEDI3(clip, int field, bool "dh", float "alpha", float "beta", float "gamma", int "nrad", int "mdis", bool "hp", int "vcheck", float "vthresh0", float "vthresh1", float "vthresh2", clip "sclip", clip "mclip")` | field 2/3 bob: 2× frames and fps |
| `vszip_EEDI3H(...)` | same parameters, interpolates along the width |
| `vszip_ImageRead(string[] path, bool "validate")` | source filter (zigimg; files or http(s) URLs) |
| `vszip_LimitFilter(clip flt, clip src, clip "ref", float[] "dark_thr", float[] "bright_thr", float[] "elast", int[] "planes")` | |
| `vszip_Limiter(clip, float[] "min", float[] "max", bool "tv_range", bool "mask", int[] "planes")` | |
| `vszip_MosquitoNR(clip, int[] "strength", int[] "restore", int[] "radius", int[] "planes")` | |
| `vszip_PlaneAverage(clip, int[] exclude, clip "clipb", int[] "planes", string "prop")` | property-only (`psmAvg`/`psmDiff`) |
| `vszip_PlaneMinMax(clip, float "minthr", float "maxthr", clip "clipb", int[] "planes", string "prop")` | property-only (`psmMin`/`psmMax`/`psmDiff`) |
| `vszip_RFS(clip, clip clipb, int[] frames, bool "mismatch", int[] "planes")` | |
| `vszip_SSIMULACRA2(clip reference, clip distorted)` | property `SSIMULACRA2` |
| `vszip_WNNM(clip, float[] "sigma", int "block_size", int "block_step", int "group_size", int "bm_range", int "radius", int "ps_num", int "ps_range", bool "residual", bool "adaptive_aggregation", clip "rclip")` | 32-bit float |
| `vszip_XPSNR(clip reference, clip distorted, bool "temporal", bool "verbose")` | props `XPSNR_Y/U/V` |

`vszip_WNNMRaw` is registered as well. It is WNNM's internal first stage, a function only so that the host caches it; do not call it directly.

vszip's int-typed booleans (`tv_range=1`) are AviSynth `bool`s here (`tv_range=true`). Enum-like ints such as `metric`, `codec`, and `dither_type` stay ints.

## Differences from vszip

AviSynth+ is a different host, so a few behaviors are adapted rather than copied:

- **Formats.** AviSynth+ has no half-float, 9-bit, or 32-bit-integer planar formats, so vszip's f16/u9/u32 paths do not exist. Alpha planes of YUVA/RGBAP clips pass through untouched (mask filters such as CombMask copy the input's alpha too); vszip never sees them because VapourSynth keeps alpha separately. `Dither` converts alpha as full-range data with plain rounding (never dithered), so internal depth round trips (Deband below 16 bit, XPSNR's depth match) return it unchanged. `EEDI3(dh=true)` rejects alpha. SSIMULACRA2 keeps a YUVA reference's alpha (see below). Secondary clips may differ from the primary in alpha presence, except in PlaneAverage/PlaneMinMax when the alpha plane is selected (`planes=[3]`): then `clipb` must carry alpha too.
- **Range metadata.** AviSynth+ uses `_ColorRange` (0 = full, 1 = limited), while VapourSynth R4 uses `_Range` (1 = full). Range-dependent filters (Deband, LimitFilter) read `_ColorRange`, and filters that tag full range write it.
- **RFS `mismatch`.** AviSynth clips cannot vary in format or size, so `mismatch=true` only tolerates a frame-rate difference; the output keeps clipa's rate.
- **XPSNR averages.** vszip adds to running totals per evaluation, so a frame recomputed by the cache counts twice. Here each frame index stores its result once, and the printed average reduces over the evaluated frames in index order (it reports "N of M frames"). Per-frame props are identical.
- **SSIMULACRA2 on YUV.** vszip converts YUV/Gray input with zimg; here each frame is normalized to float with its own `_ColorRange` (limited when unset), subsampled chroma is upsampled in float by AviSynth's `ConvertToYUV444`, and the frame's own `_Matrix` is applied (BT.709, BT.601, BT.2020 NCL, FCC, 240M; 2 or unset falls back to vszip's BT.709-above-650-lines rule; other matrices raise a frame error). Metadata may change from frame to frame, independently on both inputs. 4:4:4 input converts within float rounding of zimg; 4:2:x chroma goes through AviSynth's resampler, so those scores differ slightly. A YUV reference is returned as RGBPS, or RGBAPS carrying its alpha (integer alpha as full-range float); the distorted clip's alpha is ignored.
- **ImageRead alpha.** RGBA images use the native A plane (RGBAP). Gray+alpha has no AviSynth planar format, so it returns Y plus a separate Y frame in the `_Alpha` frame property (vszip's representation).
- **ImageRead later frames.** Every frame is checked when decoded, not only with `validate=true`: a file whose size or decoded layout (gray/RGB, depth, alpha) differs from the first image raises a frame error. The VapourSynth build does the same.
- **Scratch pools.** vszip keys EEDI3/WNNM scratch by thread ID. Here every request checks a block out of a mutex-guarded pool and returns it, so blocks are exclusive per request, including nested use on one thread.

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

Coverage per filter:
- Every supported format family and depth, with the parameter variants that select different kernels.
- Cropped (unaligned / foreign-pitch) inputs.
- Error messages.
- `test_avs_mt.py`: `Prefetch(4)` with shuffled and repeated requests must equal serial output.
- `test_avs_registration.py`: every name is reachable and dispatches to its own create.
- `test_avs_regressions.py`: combinations the matrices miss (temporal windows on verified foreign-pitch inputs, mixed alpha layouts, later ImageRead format changes, RGB plane routing, extreme WNNM ranges, alpha pass-through); cases that used to abort run in a child process. `avsrt.Clip.geometry()` gives the plane addresses and pitches the host delivers, so a test can assert the layout it relies on (a small horizontal crop is often realigned by the host).
- A sensitivity test that proves a 1-LSB difference is caught.

Run the tests against both debug and fast builds.

## Benchmark

`python bench/avisynth/bench.py [--threads 1,16] [--frames 240] [--filter Deband]` times avszip against vszip on the same prepared 1080p frames. See [`bench/avisynth/README.md`](../../bench/avisynth/README.md) for method and results.

## License

MIT, as vszip. The AviSynth build adds avisynth-zig (MIT; its loader is MPL-2.0, the AviSynth+ headers GPL with the C-interface exception).
