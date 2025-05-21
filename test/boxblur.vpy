import vapoursynth as vs

core = vs.core

core.std.LoadPlugin("./../zig-out/lib/libvszip.so")

src = core.vszip.ImageRead("./image.png")
src = src.std.Crop(left=src.width - 640, bottom=src.height - 320)
src32 = src.resize.Bilinear(format=vs.GRAYS, matrix=1).std.RemoveFrameProps("_Matrix")
src16 = src.resize.Bilinear(format=vs.GRAY16, matrix=1).std.RemoveFrameProps("_Matrix")

out32_big_r = src32.vszip.BoxBlur(hradius=30, vradius=60, hpasses=6, vpasses=8)
p32br = out32_big_r.std.PlaneStats().get_frame(0).props
assert (p32br.PlaneStatsAverage == 0.49595518544825606) or (p32br.PlaneStatsAverage == 0.4959551664479659), f"prop is: {p32br.PlaneStatsAverage}"

out32_low_r = src32.vszip.BoxBlur(hradius=3, vradius=3)
p32lr = out32_low_r.std.PlaneStats().get_frame(0).props
assert (p32lr.PlaneStatsAverage == 0.49599070191539796) or (p32lr.PlaneStatsAverage == 0.49599070499105435), f"prop is: {p32lr.PlaneStatsAverage}"

out16_big_r = src16.vszip.BoxBlur(hradius=30, vradius=33, hpasses=1, vpasses=3)
p16br = out16_big_r.std.PlaneStats().get_frame(0).props
assert (p16br.PlaneStatsAverage == 0.4867611337214847) or (p16br.PlaneStatsAverage == 0.48676114109767205), f"prop is: {p16br.PlaneStatsAverage}"

out16_low_r = src16.vszip.BoxBlur(hradius=10, vradius=10)
p16lr = out16_low_r.std.PlaneStats().get_frame(0).props
assert (p16lr.PlaneStatsAverage == 0.4869014934022612) or (p16lr.PlaneStatsAverage == 0.48690149869225413), f"prop is: {p16lr.PlaneStatsAverage}"

# stride test
st1 = out16_low_r.std.Crop(left=27)
st2 = src16.std.Crop(left=27).vszip.BoxBlur(hradius=10, vradius=10)
pst = st2.std.PlaneStats(st1).get_frame(0).props
assert (pst.PlaneStatsDiff == 0.00033399498804606234) or (pst.PlaneStatsDiff == 0.0003339957659325783), f"prop is: {pst.PlaneStatsDiff}"


st2.set_output()

# vspipe -p ./boxblur.vpy .
