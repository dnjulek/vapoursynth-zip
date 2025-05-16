import vapoursynth as vs

core = vs.core

core.std.LoadPlugin("./../zig-out/lib/libvszip.so")

src = core.vszip.ImageRead("./image.png")
src = src.std.Crop(left=src.width - 640, bottom=src.height - 320)
src32 = src.resize.Bilinear(format=vs.GRAYS, matrix=1).std.RemoveFrameProps("_Matrix")
src16 = src.resize.Bilinear(format=vs.GRAY16, matrix=1).std.RemoveFrameProps("_Matrix")

out32_big_r = src32.vszip.BoxBlur(hradius=30, vradius=60, hpasses=6, vpasses=8)
p32br = out32_big_r.std.PlaneStats().get_frame(0).props
print(p32br)
assert p32br.PlaneStatsAverage == 0.49595518544825606

out32_low_r = src32.vszip.BoxBlur(hradius=3, vradius=3)
p32lr = out32_low_r.std.PlaneStats().get_frame(0).props
print(p32lr)
assert p32lr.PlaneStatsAverage == 0.49599070191539796

out16_big_r = src16.vszip.BoxBlur(hradius=30, vradius=33, hpasses=1, vpasses=3)
p16br = out16_big_r.std.PlaneStats().get_frame(0).props
print(p16br)
assert p16br.PlaneStatsAverage == 0.4867611337214847

out16_low_r = src16.vszip.BoxBlur(hradius=10, vradius=10)
p16lr = out16_low_r.std.PlaneStats().get_frame(0).props
print(p16lr)
assert p16lr.PlaneStatsAverage == 0.4869014934022612

# stride test
st1 = out16_low_r.std.Crop(left=27)
st2 = src16.std.Crop(left=27).vszip.BoxBlur(hradius=10, vradius=10)
pst = st2.std.PlaneStats(st1).get_frame(0).props

print(pst)
assert pst.PlaneStatsDiff == 0.00033399498804606234


st2.set_output()

# vspipe -p ./boxblur.vpy .
