## VapourSynth Zig Image Process

[READ THE DOCS](https://github.com/dnjulek/vapoursynth-zip/wiki)

# FILTERS
- [AdaptiveBinarize](https://github.com/dnjulek/vapoursynth-zip/wiki/AdaptiveBinarize): based on [OpenCV's Adaptive Thresholding](https://docs.opencv.org/5.x/d7/d4d/tutorial_py_thresholding.html).
- [Bilateral](https://github.com/dnjulek/vapoursynth-zip/wiki/Bilateral): A faster version of [VapourSynth-Bilateral](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Bilateral).
- [BoxBlur](https://github.com/dnjulek/vapoursynth-zip/wiki/BoxBlur): A faster version of [std.BoxBlur](https://www.vapoursynth.com/doc/functions/video/boxblur.html).
- [Checkmate](https://github.com/dnjulek/vapoursynth-zip/wiki/Checkmate): Spatial and temporal dot crawl reducer [from AviSynth](https://github.com/tp7/checkmate).
- [CLAHE](https://github.com/dnjulek/vapoursynth-zip/wiki/CLAHE): Contrast Limited Adaptive Histogram Equalization [from OpenCV](https://docs.opencv.org/5.x/d5/daf/tutorial_py_histogram_equalization.html).
- [Metrics](https://github.com/dnjulek/vapoursynth-zip/wiki/Metrics): Image metrics like [SSIMULACRA2](https://github.com/cloudinary/ssimulacra2) and [XPSNR](https://github.com/fraunhoferhhi/xpsnr).
- [PlaneAverage](https://github.com/dnjulek/vapoursynth-zip/wiki/PlaneAverage): Vapoursynth [PlaneStats](https://www.vapoursynth.com/doc/functions/video/planestats.html) with threshold.
- [PlaneMinMax](https://github.com/dnjulek/vapoursynth-zip/wiki/PlaneMinMax): Vapoursynth [PlaneStats](https://www.vapoursynth.com/doc/functions/video/planestats.html) with threshold.
- [RFS](https://github.com/dnjulek/vapoursynth-zip/wiki/RFS): Replace frames plugin.

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

Put [zig-master](https://ziglang.org/download/) in your PATH and run: ``zig build -Doptimize=ReleaseFast``

Or run the script in [build-help](/build-help).
