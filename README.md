## VapourSynth Zig Image Process

[READ THE DOCS](https://github.com/dnjulek/vapoursynth-zip/wiki)

# FILTERS
- [AdaptiveBinarize](https://github.com/dnjulek/vapoursynth-zip/wiki/AdaptiveBinarize): based on [OpenCV's Adaptive Thresholding](https://docs.opencv.org/5.x/d7/d4d/tutorial_py_thresholding.html).
- [Bilateral](https://github.com/dnjulek/vapoursynth-zip/wiki/Bilateral): A faster version of [VapourSynth-Bilateral](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Bilateral).
- [BoxBlur](https://github.com/dnjulek/vapoursynth-zip/wiki/BoxBlur): A faster version of [std.BoxBlur](https://www.vapoursynth.com/doc/functions/video/boxblur.html).
- [Checkmate](https://github.com/dnjulek/vapoursynth-zip/wiki/Checkmate): Spatial and temporal dot crawl reducer [from AviSynth](https://github.com/tp7/checkmate).
- [CLAHE](https://github.com/dnjulek/vapoursynth-zip/wiki/CLAHE): Contrast Limited Adaptive Histogram Equalization [from OpenCV](https://docs.opencv.org/5.x/d5/daf/tutorial_py_histogram_equalization.html).
- [ColorMap](https://github.com/dnjulek/vapoursynth-zip/wiki/ColorMap): A port of the [OpenCV ColorMap](https://docs.opencv.org/5.x/d3/d50/group__imgproc__colormap.html).
- [CombMask](https://github.com/dnjulek/vapoursynth-zip/wiki/CombMask): Port of CombMask [from AviSynth](http://avisynth.nl/index.php/CombMask).
- [CombMaskMT](https://github.com/dnjulek/vapoursynth-zip/wiki/CombMaskMT): Port of MTCombMask [from AviSynth](http://avisynth.nl/index.php/MTCombMask).
- [Deband](https://github.com/dnjulek/vapoursynth-zip/wiki/Deband): A faster version of [neo_f3kdb](https://github.com/HomeOfAviSynthPlusEvolution/neo_f3kdb) with float support.
- [ImageRead](https://github.com/dnjulek/vapoursynth-zip/wiki/ImageRead): Load image using [Zig Image library](https://github.com/zigimg/zigimg).
- [Limiter](https://github.com/dnjulek/vapoursynth-zip/wiki/Limiter): A faster version of [core.std.Limiter](https://www.vapoursynth.com/doc/functions/video/limiter.html).
- [LimitFilter](https://github.com/dnjulek/vapoursynth-zip/wiki/LimitFilter): Performs a soft-limiting between two clips to limit the difference of filtering while avoiding artifacts.
- [PackRGB](https://github.com/dnjulek/vapoursynth-zip/wiki/PackRGB): Planar to interleaved RGB filter.
- [PlaneAverage](https://github.com/dnjulek/vapoursynth-zip/wiki/PlaneAverage): Vapoursynth [PlaneStats](https://www.vapoursynth.com/doc/functions/video/planestats.html) with threshold.
- [PlaneMinMax](https://github.com/dnjulek/vapoursynth-zip/wiki/PlaneMinMax): Vapoursynth [PlaneStats](https://www.vapoursynth.com/doc/functions/video/planestats.html) with threshold.
- [RFS](https://github.com/dnjulek/vapoursynth-zip/wiki/RFS): Replace frames plugin.
- [SSIMULACRA2](https://github.com/dnjulek/vapoursynth-zip/wiki/SSIMULACRA2): Image metric [SSIMULACRA2](https://github.com/cloudinary/ssimulacra2).
- [XPSNR](https://github.com/dnjulek/vapoursynth-zip/wiki/XPSNR): Image metric [XPSNR](https://github.com/fraunhoferhhi/xpsnr).

# BENCHMARK

```py
src = core.std.BlankClip(None, 1920, 1080, vs.YUV420P16, 5000)

src.bilateral.Bilateral(ref=None, sigmaS=2, sigmaR=2, planes=[0,1,2]).set_output(1) 
# Output 5000 frames in 43.35 seconds (115.35 fps)
src.vszip.Bilateral(ref=None, sigmaS=2, sigmaR=2, planes=[0,1,2]).set_output(2) 
# Output 5000 frames in 35.37 seconds (141.36 fps)

src.std.BoxBlur(hradius=13, hpasses=1, vradius=13, vpasses=1).set_output(3) 
# Output 5000 frames in 16.74 seconds (298.60 fps)
src.vszip.BoxBlur(hradius=13, hpasses=1, vradius=13, vpasses=1).set_output(4) 
# Output 5000 frames in 4.78 seconds (1046.11 fps)

src.std.BoxBlur(hradius=13, hpasses=5, vradius=13, vpasses=5).set_output(5) 
# Output 5000 frames in 76.72 seconds (65.17 fps)
src.vszip.BoxBlur(hradius=13, hpasses=5, vradius=13, vpasses=5).set_output(6) 
# Output 5000 frames in 13.62 seconds (367.01 fps)
```

## Building

- Via manual download:\
Put [zig-0.15.2](https://ziglang.org/download/) in your PATH and run: ``zig build -Doptimize=ReleaseFast``.
- Via automated scripts:
```ps
git clone https://github.com/dnjulek/vapoursynth-zip
cd vapoursynth-zip/build-help && ./build.ps1
```
- Via AUR (for Arch Linux):\
Run ``paru -S vapoursynth-plugin-vszip-git``
- Via vsrepo (for Windows):\
Run ``vsrepo install vszip``
