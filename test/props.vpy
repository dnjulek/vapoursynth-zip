import vapoursynth as vs
core = vs.core
import os

path = "./../zig-out/lib/libvszip.so" if (os.name == "posix")  else "./../zig-out/bin/vszip.dll"
core.std.LoadPlugin(path)

src = core.std.BlankClip(None, 1920, 1080, vs.YUV420P16, 10, color=[6777,32768,0])
props = core.vszip.PlaneAverage(clipa=src, exclude=[300, 5000])
props = core.vszip.PlaneAverage(clipa=props, exclude=[300, 5000], prop="avg_test")
props = core.vszip.PlaneMinMax(clipa=props, minthr=0.2, maxthr=0.3)
props = core.vszip.PlaneMinMax(clipa=props, minthr=0.2, maxthr=0.3, prop="mm_test")

p = props.get_frame(0).props
print(p)

assert(p.psmMin == 6777)
assert(p.psmMax == 6777)
assert(p.mm_testMin == 6777)
assert(p.mm_testMax == 6777)
assert(p.psmAvg > 0.1034 and p.psmAvg < 0.1035)
assert(p.avg_testAvg > 0.1034 and p.avg_testAvg < 0.1035)

props.set_output()

# vspipe -p ./test/props.vpy .
# vspreview ./test/props.vpy