import vapoursynth as vs
core = vs.core

core.std.LoadPlugin("./../zig-out/lib/libvszip.so")


def filterTest(src: vs.VideoNode, args, avg, ref=None) -> vs.VideoNode:
    flt = src.vszip.BoxBlur(hradius=2, vradius=2)
    src = core.vszip.LimitFilter(flt, src, ref, args[0], args[1], args[2])
    src = src.vszip.PlaneAverage(exclude=1 << 16, planes=[0, 1, 2])
    p = src.get_frame(0).props
    assert p.psmAvg == avg, f"{p.psmAvg} == {avg}"
    return src


img = core.vszip.ImageRead("./image.png")
img = img.std.Crop(left=img.width - 640, bottom=img.height - 320)

args = [[10.86, 15.93, 19.29], [19.86, 17.67, 10.34], [23.88, 27.67, 20.47]]
src = img.resize.Bilinear(format=vs.YUV420P8, matrix=1)
out = filterTest(src, args, [0.4903492838541667, 0.7358145680147059, 0.44216314338235296])

src = img.resize.Bilinear(format=vs.YUV420P16, matrix=1)
out = filterTest(src, args, [0.48858497874465934, 0.73300166359102, 0.44039273205634394])

src = img.resize.Bilinear(format=vs.YUV420PS, matrix=1)
out = filterTest(src, args, [0.49807171091622876, 0.2662752162836841, -0.06806828150764488])

src = img.resize.Bilinear(format=vs.YUV420P10, matrix=1)
out = filterTest(src, args, [0.48910823997281283, 0.7337130185422777, 0.4407805244990225], src)

avg = [0.48910823997281283, 0.7339695213220919, 0.44344561415566963]
args = [[10.86, 10.86, 10.86], [19.86, 19.86, 19.86], [23.88, 23.88, 23.88]]
out = filterTest(src, args, avg)

args = [[10.86, 10.86], [19.86, 19.86], [23.88, 23.88]]
out = filterTest(src, args, avg)

args = [[10.86], [19.86], [23.88]]
out = filterTest(src, args, avg)

out.set_output()

# vspipe -p ./limit_filter.vpy .
