import vapoursynth as vs
core = vs.core

core.std.LoadPlugin("../zig-out/bin/vszip.dll")

src8 = core.std.BlankClip(None, 1920, 1080, vs.YUV420P8, 10, color=[255,128,0])
src8b = core.std.BlankClip(None, 1920, 1080, vs.YUV420P8, 10, color=[255,128,0])
src8 = core.vszip.RFS(clipa=src8, clipb=src8b, frames=[0, 3, 5, 9], mismatch=False)
src8 = core.vszip.PlaneAverage(clipa=src8, exclude=[300, 5000], clipb=src8b, planes=[0, 1, 2])
src8 = core.vszip.PlaneMinMax(clipa=src8, minthr=0.2, maxthr=0.3, clipb=src8b, planes=[0, 1, 2])
src8 = core.vszip.BoxBlur(src8, planes=[0,1,2], hradius=30, vradius=60, hpasses=6, vpasses=8)
src8 = core.vszip.BoxBlur(src8, planes=[1,2], hradius=10, vradius=10, hpasses=6, vpasses=8)
src8 = core.vszip.Bilateral(src8, ref=None, sigmaS=2, sigmaR=2, planes=[0,1,2], algorithm=0)
src8 = core.vszip.Bilateral(src8, ref=src8b, sigmaS=2, sigmaR=2, planes=[0], algorithm=0)

src16 = core.std.BlankClip(None, 1920, 1080, vs.YUV420P16, 10, color=[65535,32768,0])
src16b = core.std.BlankClip(None, 1920, 1080, vs.YUV420P16, 10, color=[65535,32768,0])
src16 = core.vszip.RFS(clipa=src16, clipb=src16b, frames=[0, 3, 5, 9], mismatch=False)
src16 = core.vszip.PlaneAverage(clipa=src16, exclude=[300, 5000], clipb=src16b, planes=[0, 1, 2])
src16 = core.vszip.PlaneMinMax(clipa=src16, minthr=0.2, maxthr=0.3, clipb=src16b, planes=[0, 1, 2])
src16 = core.vszip.BoxBlur(src16, planes=[0,1,2], hradius=30, vradius=60, hpasses=6, vpasses=8)
src16 = core.vszip.BoxBlur(src16, planes=[1,2], hradius=10, vradius=10, hpasses=6, vpasses=8)
src16 = core.vszip.Bilateral(src16, ref=None, sigmaS=2, sigmaR=2, planes=[0,1,2], algorithm=0)
src16 = core.vszip.Bilateral(src16, ref=src16b, sigmaS=2, sigmaR=2, planes=[0], algorithm=0)

src32 = core.std.BlankClip(None, 1920, 1080, vs.YUV444PS, 10)
src32b = core.std.BlankClip(None, 1920, 1080, vs.YUV444PS, 10)
src32 = core.vszip.RFS(clipa=src32, clipb=src32b, frames=[0, 3, 5, 9], mismatch=False)
src32 = core.vszip.PlaneAverage(clipa=src32, exclude=[300, 5000], clipb=src32b, planes=[0, 1, 2])
src32 = core.vszip.PlaneMinMax(clipa=src32, minthr=0.2, maxthr=0.3, clipb=src32b, planes=[0, 1, 2])
src32 = core.vszip.BoxBlur(src32, planes=[0,1,2], hradius=30, vradius=60, hpasses=6, vpasses=8)
src32 = core.vszip.BoxBlur(src32, planes=[1,2], hradius=10, vradius=10, hpasses=1, vpasses=1)
src32 = core.vszip.Bilateral(src32, ref=None, sigmaS=2, sigmaR=2, planes=[0,1,2], algorithm=0)
src32 = core.vszip.Bilateral(src32, ref=src32b, sigmaS=2, sigmaR=2, planes=[0], algorithm=0)

src8.set_output(1)
src16.set_output(2)
src32.set_output(3)