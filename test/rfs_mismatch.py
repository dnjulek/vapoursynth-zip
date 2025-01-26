import vapoursynth as vs
core = vs.core

clip1 = core.std.BlankClip(format=vs.YUV420P8, width=1280, height=720, fpsnum=24000, fpsden=1001, length=10)
clip2 = core.std.BlankClip(format=vs.YUV420P16, width=1920, height=1080, fpsnum=30000, fpsden=1, length=10)

rfs = core.vszip.RFS(clip1, clip2, frames=[0, 3, 5, 1], mismatch=True)
print(rfs)