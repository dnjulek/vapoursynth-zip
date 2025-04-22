import vapoursynth as vs
core = vs.core
import os

path = "./../zig-out/lib/libvszip.so" if (os.name == "posix")  else "./../zig-out/bin/vszip.dll"
core.std.LoadPlugin(path)

def testFilters(src1, src2) -> vs.VideoNode:
    src1 = core.vszip.RFS(clipa=src1, clipb=src2, frames=[0, 3, 5, 9], mismatch=False, planes=[0, 1])
    src1 = core.vszip.RFS(clipa=src1, clipb=src2, frames=[0, 3, 5, 9], mismatch=False)
    src1 = core.vszip.PlaneAverage(clipa=src1, exclude=[300, 5000], clipb=src2, planes=[0, 1])
    src1 = core.vszip.PlaneAverage(clipa=src1, exclude=[300, 5000], clipb=None)
    src1 = core.vszip.PlaneMinMax(clipa=src1, minthr=0.2, maxthr=0.3, clipb=src2, planes=[0, 1, 2])
    src1 = core.vszip.PlaneMinMax(clipa=src2, minthr=0.2, maxthr=0.3, clipb=None)
    src1 = core.vszip.BoxBlur(src1, hradius=30, vradius=60, hpasses=6, vpasses=8)
    src1 = core.vszip.BoxBlur(src1, planes=[1,2], hradius=10, vradius=10, hpasses=6, vpasses=8)
    src1 = core.vszip.Bilateral(src1, ref=None, sigmaS=2, sigmaR=2, algorithm=0)
    src1 = core.vszip.Bilateral(src1, ref=src2, sigmaS=2, sigmaR=2, planes=[0], algorithm=0)
    src1 = core.vszip.Limiter(src1, [22,33,44], [200,210,240])
    src1 = core.vszip.Limiter(src1, tv_range=True)
    src1 = core.vszip.Limiter(src1)
    return src1

src8 = core.std.BlankClip(None, 1920, 1080, vs.YUV420P8, 10, color=[255,128,0])
src8b = core.std.BlankClip(None, 1920, 1080, vs.YUV420P8, 10, color=[255,128,0])

src16 = core.std.BlankClip(None, 1920, 1080, vs.YUV420P16, 10, color=[65535,32768,0])
src16b = core.std.BlankClip(None, 1920, 1080, vs.YUV420P16, 10, color=[65535,32768,0])

src32 = core.std.BlankClip(None, 1920, 1080, vs.YUV444PS, 10)
src32b = core.std.BlankClip(None, 1920, 1080, vs.YUV444PS, 10)

testFilters(src8, src8b).set_output(1)
testFilters(src16, src16b).set_output(2)
testFilters(src32, src32b).set_output(3)

# vspipe -p ./test/test.vpy . -o 1