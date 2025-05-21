import vapoursynth as vs

core = vs.core


core.std.LoadPlugin("./../zig-out/lib/libvszip.so")

src = core.vszip.ImageRead("./image.png")
src = src.std.Crop(left=src.width - 640, bottom=src.height - 320)
src32 = src.resize.Bilinear(format=vs.GRAYS, matrix=1).std.RemoveFrameProps("_Matrix")
src16 = src.resize.Bilinear(format=vs.GRAY16, matrix=1).std.RemoveFrameProps("_Matrix")

out32 = core.vszip.Bilateral(src32, ref=None, sigmaS=2, sigmaR=2)
p32 = out32.std.PlaneStats().get_frame(0).props
assert (p32.PlaneStatsAverage == 0.4959264570310188) or (p32.PlaneStatsAverage == 0.4959264618139059), f"prop is: {p32.PlaneStatsAverage}"

out32r = core.vszip.Bilateral(src32, ref=src32.std.BoxBlur(hradius=5, vradius=5))
p32r = out32r.std.PlaneStats().get_frame(0).props
assert (p32r.PlaneStatsAverage == 0.4959947573716272) or (p32r.PlaneStatsAverage == 0.4959947604804256), f"prop is: {p32r.PlaneStatsAverage}"

out16 = core.vszip.Bilateral(src16, ref=None, sigmaS=2, sigmaR=2)
p16 = out16.std.PlaneStats().get_frame(0).props
assert (p16.PlaneStatsAverage == 0.4867585293312972) or (p16.PlaneStatsAverage == 0.4867585349938249), f"prop is: {p16.PlaneStatsAverage}"

out16r = core.vszip.Bilateral(src16, ref=src16.std.BoxBlur(hradius=5, vradius=5))
p16r = out16r.std.PlaneStats().get_frame(0).props
assert (p16r.PlaneStatsAverage == 0.4867979883572471) or (p16r.PlaneStatsAverage == 0.4867979948393511), f"prop is: {p16r.PlaneStatsAverage}"

# stride test
st1 = out16.std.Crop(left=27)
st2 = core.vszip.Bilateral(src16.std.Crop(left=27), ref=None, sigmaS=2, sigmaR=2)
pst = st2.std.PlaneStats(st1).get_frame(0).props
assert pst.PlaneStatsDiff == 8.92147154721379e-05, f"prop is: {pst.PlaneStatsDiff}"

st2.set_output()

# vspipe -p ./bilateral.vpy .
