import vapoursynth as vs
core = vs.core

core.std.LoadPlugin("../zig-out/lib/vszip.dll")
src = core.std.BlankClip(None, 1920, 1080, vs.YUV420P16, 600)
src2 = core.std.BlankClip(None, 1920, 1080, vs.YUV420P16, 600)
src = core.vszip.RFS(clipa=src, clipb=src, frames=[0, 88, 22, 34, 234, 523], mismatch=False)
src = core.vszip.PlaneAverage(clipa=src, exclude=[300, 5000], clipb=src, planes=[0, 1, 2])
src = core.vszip.PlaneMinMax(clipa=src, minthr=0.2, maxthr=0.3, clipb=src, planes=[0, 1, 2])
src = core.vszip.BoxBlur(src, planes=[0,1,2], hradius=30, vradius=60, hpasses=6, vpasses=8)
src = core.vszip.BoxBlur(src, planes=[1,2], hradius=10, vradius=10, hpasses=6, vpasses=8)
src = core.vszip.Bilateral(src, ref=None, sigmaS=2, sigmaR=2, planes=[0,1,2], algorithm=0)
src = core.vszip.Bilateral(src, ref=src2, sigmaS=2, sigmaR=2, planes=[0], algorithm=0)

src.set_output(0)