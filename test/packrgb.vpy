import vapoursynth as vs

core = vs.core

core.std.LoadPlugin("./../zig-out/lib/libvszip.so")

src = core.vszip.ImageRead("./image.png")
src = src.std.Crop(left=src.width - 647, bottom=src.height - 320)
src10 = src.resize.Point(format=vs.RGB30)

out0 = src.vszip.PackRGB().vszip.PlaneAverage(exclude=[-1])
p0 = out0.get_frame(0).props
assert p0.psmAvg == 0.9976244760539695, f"prop is: {p0.psmAvg}"


out1 = src10.vszip.PackRGB().vszip.PlaneAverage(exclude=[-1])
p1 = out1.get_frame(0).props
assert p1.psmAvg == 0.8479205358684719, f"prop is: {p1.psmAvg}"


out1.set_output()

# vspipe -p ./packrgb.vpy .
