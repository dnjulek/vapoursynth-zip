import vapoursynth as vs

core = vs.core

core.std.LoadPlugin("./../zig-out/lib/libvszip.so")


def blank(src: vs.VideoNode) -> vs.VideoNode:
    peak = (1 << src.format.bits_per_sample) - 1
    floor = -2 if (src.format.sample_type == vs.FLOAT) else 0
    src1 = src.std.BlankClip(color=[peak, peak, peak])
    src2 = src.std.BlankClip(color=[floor, floor, floor])
    return core.std.StackVertical([src1, src2])


def limitTest(src: vs.VideoNode, mi, ma, mask=False) -> vs.VideoNode:
    src = src.vszip.Limiter(tv_range=True, mask=mask).vszip.PlaneMinMax(planes=[0, 1, 2])
    p = src.get_frame(0).props
    assert p.psmMin == mi and p.psmMax == ma, f"{p.psmMin} == {mi} and {p.psmMax} == {ma}"
    return src


src = core.std.BlankClip(None, 64, 32, vs.YUV420P8, 1)
src = blank(src)
src = limitTest(src, [16, 16, 16], [235, 240, 240])
src = blank(src.std.BlankClip(format=vs.YUV420P10))
src = limitTest(src, [64, 64, 64], [940, 960, 960])
src = blank(src.std.BlankClip(format=vs.YUV420P16))
src = limitTest(src, [4096, 4096, 4096], [60160, 61440, 61440])
src = blank(src.std.BlankClip(format=vs.YUV420PS))
src = limitTest(src, [0.0, -0.5, -0.5], [1, 0.5, 0.5])
src = blank(src.std.BlankClip(format=vs.YUV420PS))
src = limitTest(src, [0.0, 0.0, 0.0], [1, 1, 1], True)


src.set_output()

# vspipe -p ./limiter.vpy .
