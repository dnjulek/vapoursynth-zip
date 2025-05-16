import vapoursynth as vs

core = vs.core


core.std.LoadPlugin("./../zig-out/lib/libvszip.so")

src = core.vszip.ImageRead("./image.png")
src = src.std.Crop(left=src.width - 640, bottom=src.height - 320)
src = src.resize.Bilinear(format=vs.GRAY8, matrix=1).std.RemoveFrameProps("_Matrix")
srcb = src.std.BoxBlur(hradius=5, vradius=5)

out0 = src.vszip.AdaptiveBinarize(srcb)
p0 = out0.std.PlaneStats().get_frame(0).props
print(p0)
assert p0.PlaneStatsAverage == 0.45087890625

# stride test
st1 = out0.std.Crop(left=27)
st2 = src.std.Crop(left=27).vszip.AdaptiveBinarize(srcb.std.Crop(left=27))
pst = st2.std.PlaneStats(st1).get_frame(0).props

print(pst)
assert pst.PlaneStatsDiff == 0.0


st2.set_output()

# vspipe -p ./adaptive_binarize.vpy .
