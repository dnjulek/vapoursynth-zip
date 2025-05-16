import vapoursynth as vs

core = vs.core


core.std.LoadPlugin("./../zig-out/lib/libvszip.so")

src = core.vszip.ImageRead("./image.png")
src = src.std.Crop(left=src.width - 640, bottom=src.height - 320)
src32 = src.resize.Bilinear(format=vs.GRAYS, matrix=1).std.RemoveFrameProps("_Matrix")
src16 = src.resize.Bilinear(format=vs.GRAY16, matrix=1).std.RemoveFrameProps("_Matrix")

out32 = core.vszip.Bilateral(src32, ref=None, sigmaS=2, sigmaR=2)
p32 = out32.std.PlaneStats().get_frame(0).props
print(p32)
assert p32.PlaneStatsAverage == 0.4959264570310188

out32r = core.vszip.Bilateral(src32, ref=src32.std.BoxBlur(hradius=5, vradius=5))
p32r = out32r.std.PlaneStats().get_frame(0).props
print(p32r)
assert p32r.PlaneStatsAverage == 0.4959947573716272

out16 = core.vszip.Bilateral(src16, ref=None, sigmaS=2, sigmaR=2)
p16 = out16.std.PlaneStats().get_frame(0).props
print(p16)
assert p16.PlaneStatsAverage == 0.4867585293312972

out16r = core.vszip.Bilateral(src16, ref=src16.std.BoxBlur(hradius=5, vradius=5))
p16r = out16r.std.PlaneStats().get_frame(0).props
print(p16r)
assert p16r.PlaneStatsAverage == 0.4867979883572471

# stride test
st1 = out16.std.Crop(left=27)
st2 = core.vszip.Bilateral(src16.std.Crop(left=27), ref=None, sigmaS=2, sigmaR=2)
pst = st2.std.PlaneStats(st1).get_frame(0).props

print(pst)
assert pst.PlaneStatsDiff == 8.92147154721379e-05

st2.set_output()

# vspipe -p ./bilateral.vpy .
