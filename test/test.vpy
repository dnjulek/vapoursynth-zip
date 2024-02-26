import vapoursynth as vs
core = vs.core

core.std.LoadPlugin("../zig-out/lib/vszip.dll")
src = core.std.BlankClip(None, 1920, 1080, vs.YUV420P16, 1000)
src = core.zip.RFS(clipa=src, clipb=src, frames=[0, 88, 22, 34, 234, 523], mismatch=False)
src = src.std.Trim(0, 50)
src = core.zip.PlaneAverage(clipa=src, exclude=[300, 5000], clipb=src, planes=[0, 1, 2])
src = core.zip.PlaneMinMax(clipa=src, minthr=0.2, maxthr=0.3, clipb=src, planes=[0, 1, 2])
src = core.zip.BoxBlur(src, planes=[0,1,2], hradius=30, vradius=60, hpasses=6, vpasses=8)
src = core.zip.BoxBlur(src, planes=[1,2], hradius=10, vradius=10, hpasses=6, vpasses=8)

src.set_output(0)