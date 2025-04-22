import vapoursynth as vs
core = vs.core
import os

path = "./../zig-out/lib/libvszip.so" if (os.name == "posix")  else "./../zig-out/bin/vszip.dll"
core.std.LoadPlugin(path)

clip1 = core.std.BlankClip(format=vs.YUV420P8, width=1280, height=720, fpsnum=24000, fpsden=1001, length=10)
clip2 = core.std.BlankClip(format=vs.YUV420P16, width=1920, height=1080, fpsnum=30000, fpsden=1, length=10)

rfs = core.vszip.RFS(clip1, clip2, frames=[0, 3, 5, 1], mismatch=True)

assert(rfs.width == 0)
assert(rfs.height == 0)
assert(rfs.fps == 0)
assert(rfs.format == None)

print(rfs)