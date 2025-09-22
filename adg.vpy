import vapoursynth as vs
from vsmasktools import adg_mask
core = vs.core


core.std.LoadPlugin("./zig-out/lib/libvszip.so")

src = core.vszip.ImageRead("./test/image.png") * 1500
src = src.resize.Point(format=vs.GRAYS, matrix=1).std.RemoveFrameProps("_Matrix") 

a2 = src.vszip.ADGMask()
a3 = adg_mask(src)
a4 = core.std.Expr([a2, a3], "x y - abs")

src.set_output(1)
a2.set_output(2)
a3.set_output(3)
a4.set_output(4)

# vspipe -p ./adg.vpy . -o 2
