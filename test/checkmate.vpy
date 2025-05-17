import vapoursynth as vs

core = vs.core

core.std.LoadPlugin("./../zig-out/lib/libvszip.so")

src = core.vszip.ImageRead("./image.png")
src = src.std.Crop(left=src.width - 640)
src0 = src.std.Crop(bottom=src.height - 320)
src1 = src.std.Crop(bottom=src.height - 321, top=1)
src2 = src.std.Crop(bottom=src.height - 322, top=2)
src = src0 + src1 + src2

src8 = src.resize.Bilinear(format=vs.GRAY8, matrix=1).std.RemoveFrameProps("_Matrix")

out0 = src8.vszip.Checkmate(thr=12, tmax=12, tthr2=0)
p0 = out0.std.PlaneStats().get_frame(1).props
assert p0.PlaneStatsAverage == 0.4871367378982843, f"prop is: {p0.PlaneStatsAverage}"

out1 = src8.vszip.Checkmate(thr=14, tmax=11, tthr2=4)
p1 = out1.std.PlaneStats().get_frame(1).props
assert p1.PlaneStatsAverage == 0.48719424019607843, f"prop is: {p1.PlaneStatsAverage}"

# stride test
st1 = out0.std.Crop(left=27)
st2 = src8.std.Crop(left=27).vszip.Checkmate(thr=12, tmax=12, tthr2=0)
pst = st2.std.PlaneStats(st1).get_frame(1).props
assert pst.PlaneStatsDiff == 3.142692639861818e-05, f"prop is: {pst.PlaneStatsDiff}"

st2.set_output()

# vspipe -p ./checkmate.vpy .
