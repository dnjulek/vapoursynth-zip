import vapoursynth as vs

core = vs.core

core.std.LoadPlugin("./../zig-out/lib/libvszip.so")

src = core.vszip.ImageRead("./image.png")
src = src.std.Crop(left=src.width - 640, bottom=src.height - 320)
src = src.resize.Bicubic(format=vs.YUV420P16, matrix=1)
srcb = src.resize.Bicubic(src.width * 2, src.height * 2).resize.Bicubic(src.width, src.height)

out0 = core.vszip.SSIMULACRA2(src, srcb)
p0 = out0.get_frame(0).props
assert (p0.SSIMULACRA2 == 68.62413413403678) or (p0.SSIMULACRA2 == 68.62366753474497), f"prop is: {p0.SSIMULACRA2}"

rgbs = src.resize.Bicubic(format=vs.RGBS).std.SetFrameProps(_Transfer=13)
rgbs2 = srcb.resize.Bicubic(format=vs.RGBS).std.SetFrameProps(_Transfer=13)
rgbs_linear = rgbs.resize.Bicubic(transfer=8)
rgbs2_linear = rgbs2.resize.Bicubic(transfer=8)
out1 = core.vszip.SSIMULACRA2(rgbs_linear, rgbs2_linear)
p1 = out1.get_frame(0).props
assert p0.SSIMULACRA2 == p1.SSIMULACRA2, f"prop is: {p0.SSIMULACRA2} and {p1.SSIMULACRA2}"

out1.set_output()


# vspipe -p ./ssimulacra2.vpy .
