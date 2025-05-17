import vapoursynth as vs

core = vs.core

core.std.LoadPlugin("./../zig-out/lib/libvszip.so")

src = core.std.BlankClip(None, 1920, 1080, vs.YUV420P16, 10, color=[6777, 32768, 0])
props = core.vszip.PlaneAverage(clipa=src, exclude=[300, 5000])
props = core.vszip.PlaneAverage(clipa=props, exclude=[300, 5000], prop="avg_test")
props = core.vszip.PlaneMinMax(clipa=props, minthr=0.2, maxthr=0.3)
props = core.vszip.PlaneMinMax(clipa=props, minthr=0.2, maxthr=0.3, prop="mm_test")

p = props.get_frame(0).props
assert p.psmMin == 6777, f"prop is: {p.psmMin}"
assert p.psmMax == 6777, f"prop is: {p.psmMax}"
assert p.mm_testMin == 6777, f"prop is: {p.mm_testMin}"
assert p.mm_testMax == 6777, f"prop is: {p.mm_testMax}"
assert p.psmAvg == 0.10341039139391164, f"prop is: {p.psmAvg}"
assert p.avg_testAvg == 0.10341039139391164, f"prop is: {p.avg_testAvg}"

props.set_output()

# vspipe -p ./plane_props.vpy .
