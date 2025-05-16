import vapoursynth as vs

core = vs.core

core.std.LoadPlugin("./../zig-out/lib/libvszip.so")

src = core.vszip.ImageRead("./image.png")
src = src.std.Crop(left=src.width - 640, bottom=src.height - 320)
src8 = src.resize.Bilinear(format=vs.GRAY8, matrix=1).std.RemoveFrameProps("_Matrix")

avg = [
    0.5451398207720588,
    0.4651330231311275,
    0.6759822495404412,
    0.4043458754595588,
    0.6391776960784313,
    0.2597525275735294,
    0.6336821384803921,
    0.576895814185049,
    0.5283999693627451,
    0.551301087622549,
    0.6377368642769607,
    0.41900467218137255,
    0.5718827550551471,
    0.36308752680759804,
    0.3589300130208333,
    0.3995364200367647,
    0.46880407475490193,
    0.47296392463235293,
    0.2965622319240196,
    0.6093295419730392,
    0.6932635952818628,
    0.3864266429227941,
]

for i in range(22):
    color = src8.vszip.ColorMap(i).resize.Bilinear(format=vs.GRAY8, matrix=1)
    prop = color.std.PlaneStats().get_frame(0).props
    assert prop.PlaneStatsAverage == avg[i], f"ColorMap {i} failed"


src8.vszip.ColorMap(0).set_output()

# vspipe -p ./color_map.vpy .
