import vapoursynth as vs

core = vs.core

core.std.LoadPlugin("./../zig-out/lib/libvszip.so")

src = core.vszip.ImageRead("./image.png")
src = src.std.Crop(left=src.width - 640, bottom=src.height - 320)
src8 = src.resize.Bilinear(format=vs.GRAY8, matrix=1).std.RemoveFrameProps("_Matrix")


out0 = src8.vszip.CombMaskMT()
p0 = out0.std.PlaneStats().get_frame(0).props
assert p0.PlaneStatsAverage == 0.1150439453125, f"prop is: {p0.PlaneStatsAverage}"

out1 = src8.vszip.CombMaskMT(0, 255)
p1 = out1.std.PlaneStats().get_frame(0).props
assert p1.PlaneStatsAverage == 0.10427868412990196, f"prop is: {p1.PlaneStatsAverage}"


out1.set_output()

# vspipe -p ./comb_mask_mt.vpy .
