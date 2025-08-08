import vapoursynth as vs

core = vs.core

core.std.LoadPlugin("./../zig-out/lib/libvszip.so")


img = core.vszip.ImageRead("./image.png")
img = img.std.Crop(left=img.width - 640, bottom=img.height - 320)
src16 = img.resize.Point(format=vs.YUV420P16, matrix=1)
src32 = img.resize.Point(format=vs.YUV420PS, matrix=1)
src16b = src16.vszip.BoxBlur(hradius=1, vradius=1)
src32b = src32.vszip.BoxBlur(hradius=1, vradius=1)

out = src16.vszip.PlaneMinMax(minthr=0.2, maxthr=0.3, planes=[0, 1, 2])
p = out.get_frame(0).props
assert p.psmMin == [21177, 42248, 19079] and p.psmMax == [38762, 52119, 34778], (
    f"{p.psmMin} == [21177, 42248, 19079] and {p.psmMax} == [38762, 52119, 34778]"
)

out = src32.vszip.PlaneMinMax(minthr=0.2, maxthr=0.3)
p = out.get_frame(0).props
assert p.psmMin == 0.30467689037323 and p.psmMax == 0.618341326713562, (
    f"{p.psmMin} == 0.30467689037323 and {p.psmMax} == 0.618341326713562"
)

out = src32.std.Expr(["", "x 0.4 > 2 x ?", ""]).std.Expr(["", "x 0.3 < -2 x ?", ""])
out = out.vszip.PlaneMinMax(planes=[0, 1, 2])
p = out.get_frame(0).props
assert p.psmMin == [
    0.07218431681394577,
    -2.0,
    -0.47576838731765747,
] and p.psmMax == [
    0.9401216506958008,
    2.0,
    0.34346699714660645,
], (
    f"{p.psmMin} == [0.07218431681394577, -2.0, -0.47576838731765747] and {p.psmMax} == [0.9401216506958008, 2.0, 0.34346699714660645]"
)

out = src16.vszip.PlaneMinMax(minthr=0, maxthr=0, clipb=src16b, planes=[0, 1, 2])
p = out.get_frame(0).props
assert p.psmDiff == [
    0.04060555502903982,
    0.03211699557011521,
    0.07426088248407339,
] or p.psmDiff == [
    0.040605553091859314,
    0.03211700242475395,
    0.07426087771562905,
], f"{p.psmDiff} == [0.04060555502903982, 0.03211699557011521, 0.07426088248407339]"

out = src32.vszip.PlaneMinMax(minthr=0, maxthr=0, clipb=src32b, planes=[0, 1, 2])
p = out.get_frame(0).props
assert p.psmDiff == [
    0.04750444493987743,
    0.03674831314230687,
    0.08495122875109701,
] or p.psmDiff == [
    0.04750444565572252,
    0.036748313202770076,
    0.0849512294415399,
], f"{p.psmDiff} == [0.04750444493987743, 0.03674831314230687, 0.08495122875109701]"

out = src16.vszip.PlaneMinMax(minthr=0.2, maxthr=0.3, clipb=src16b, planes=[0, 1, 2])
p = out.get_frame(0).props
assert p.psmDiff == [
    0.04060555502903982,
    0.03211699557011521,
    0.07426088248407339,
] or p.psmDiff == [
    0.040605553091859314,
    0.03211700242475395,
    0.07426087771562905,
], f"{p.psmDiff} == [0.04060555502903982, 0.03211699557011521, 0.07426088248407339]"

out = src32.vszip.PlaneMinMax(minthr=0.2, maxthr=0.3, clipb=src32b)
p = out.get_frame(0).props
assert p.psmDiff == 0.04750444493987743 or p.psmDiff == 0.04750444565572252, f"{p.psmDiff} == 0.04750444493987743"


out.set_output()

# vspipe -p ./planeminmax.vpy .
