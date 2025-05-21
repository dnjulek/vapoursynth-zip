import vapoursynth as vs

core = vs.core

core.std.LoadPlugin("./../zig-out/lib/libvszip.so")

src = core.vszip.ImageRead("./image.png")
src = src.std.Crop(left=src.width - 640, bottom=src.height - 320)
src16 = src.resize.Bilinear(format=vs.GRAY16, matrix=1).std.RemoveFrameProps("_Matrix")

out0 = src16.vszip.CLAHE(limit=15, tiles=3)
p0 = out0.std.PlaneStats().get_frame(0).props
assert (p0.PlaneStatsAverage == 0.6039824106244517) or (p0.PlaneStatsAverage == 0.6039824242592222), f"prop is: {p0.PlaneStatsAverage}"

out1 = src16.vszip.CLAHE(limit=15)
p1 = out1.std.PlaneStats(out0).get_frame(0).props
assert p1.PlaneStatsDiff == 0.0, f"prop is: {p1.PlaneStatsDiff}"

out2 = src16.vszip.CLAHE(limit=15, tiles=[3, 3])
p2 = out2.std.PlaneStats(out0).get_frame(0).props
assert p2.PlaneStatsDiff == 0.0, f"prop is: {p2.PlaneStatsDiff}"


out2.set_output()

# vspipe -p ./clahe.vpy .
