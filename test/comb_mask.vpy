import vapoursynth as vs

core = vs.core


def test(src: vs.VideoNode, cthresh: int = 6, mthresh: int = 9, expand: bool = True, metric: int = 0, avg: float = 0) -> vs.VideoNode:
    out0 = src8.vszip.CombMask(cthresh=cthresh, mthresh=mthresh, expand=expand, metric=metric)
    p0 = out0.std.PlaneStats().get_frame(1).props
    assert p0.PlaneStatsAverage == avg, f"prop is: {p0.PlaneStatsAverage}"
    return out0


core.std.LoadPlugin("./../zig-out/lib/libvszip.so")

src = core.vszip.ImageRead("./image.png")
src1 = src.std.Crop(left=src.width - 640, bottom=src.height - 320)
src2 = src.std.Crop(left=src.width - 640, bottom=src.height - 322, top=2)
src = src1 + src2
src8 = src.resize.Point(format=vs.GRAY8, matrix=1).std.RemoveFrameProps("_Matrix")

t1 = test(src8, cthresh=8, mthresh=2, metric=0, avg=0.1969091796875)
t2 = test(src8, cthresh=8, mthresh=0, metric=0, avg=0.1985546875)
t3 = test(src8, cthresh=8, mthresh=2, metric=1, avg=0.2353759765625)
t3 = test(src8, cthresh=8, mthresh=0, metric=1, avg=0.2369677734375)


src8.set_output()

# vspipe -p ./comb_mask.vpy .
