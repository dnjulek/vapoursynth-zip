import pytest

import harness as H
from avsrt import AvsError


@pytest.mark.parametrize("fmt", ["Y8", "YV12", "YUV444P16", "RGBPS"])
@pytest.mark.parametrize("planes", [None, [0], [1, 2], [0, 1, 2]])
def test_rfs_matches_vszip(env, fmt, planes):
    a = H.Pair(env, fmt, H.make_frames(fmt, n=6, seed=1))
    b = H.Pair(env, fmt, H.make_frames(fmt, n=6, seed=9))
    env.set_clip("b", b.avs)
    args = {"frames": [1, 4, 5]}
    if planes is not None:
        args["planes"] = planes
    avs_args = dict(args, **{"clipb_": None})
    del avs_args["clipb_"]
    env.set_clip("src", a.avs)
    out_a = env.eval("src." + H.avs_call("RFS", args).replace("vszip_RFS(", "vszip_RFS(b, "))
    out_v = H.vszip().RFS(a.vs, b.vs, **args)
    for i, (x, y) in enumerate(zip(out_a.frames(), H.vs_frames(out_v))):
        H.compare_planes(x, y, what=f"frame {i}")


def test_rfs_props_follow_plane0_source(env):
    a = H.Pair(env, "YV12", H.make_frames("YV12", n=3, seed=1))
    b = H.Pair(env, "YV12", H.make_frames("YV12", n=3, seed=2))
    env.set_clip("a", a.avs)
    env.set_clip("b", b.avs)
    env.eval('a = a.propSet("who", 1)\nb = b.propSet("who", 2)', want_clip=False)
    full = env.eval("vszip_RFS(a, b, frames=[1])")
    p0 = env.eval("vszip_RFS(a, b, frames=[1], planes=[0])")
    chroma = env.eval("vszip_RFS(a, b, frames=[1], planes=[1, 2])")
    assert full.frame(0).props["who"] == 1 and full.frame(1).props["who"] == 2
    assert p0.frame(1).props["who"] == 2
    assert chroma.frame(1).props["who"] == 1


def test_rfs_errors(env):
    env.eval('a = BlankClip(pixel_type="YV12", length=5)\nb = BlankClip(pixel_type="YV24", length=5)\n'
             'c = BlankClip(pixel_type="YV12", length=5, fps=30)', want_clip=False)
    with pytest.raises(AvsError, match="formats don't match"):
        env.eval("vszip_RFS(a, b, frames=[1])")
    with pytest.raises(AvsError, match="variable format"):
        env.eval("vszip_RFS(a, b, frames=[1], mismatch=true)")
    with pytest.raises(AvsError, match="frame rates don't match"):
        env.eval("vszip_RFS(a, c, frames=[1])")
    assert env.eval("vszip_RFS(a, c, frames=[1], mismatch=true)").fps == env.eval("a").fps
    with pytest.raises(AvsError, match="> last frame index"):
        env.eval("vszip_RFS(a, a, frames=[5])")
    with pytest.raises(AvsError, match="non-negative"):
        env.eval("vszip_RFS(a, a, frames=[-1])")
    with pytest.raises(AvsError, match="frames is required"):
        env.eval("vszip_RFS(a, a)")


BB_FMTS = ["Y8", "YV12", "YUV444P10", "YUV420P16", "RGBP", "Y32", "YUV420PS", "RGBPS"]


@pytest.mark.parametrize("fmt", BB_FMTS)
@pytest.mark.parametrize("args", [
    {},
    {"hradius": 3, "vradius": 3},
    {"hradius": 5, "vradius": 2, "hpasses": 2, "vpasses": 3},
    {"hradius": 12, "vradius": 12},
    {"hradius": 0, "vradius": 4},
    {"hradius": 4, "vpasses": 0},
    {"hradius": 2, "vradius": 2, "planes": [0]},
])
def test_boxblur_matches_vszip(env, fmt, args):
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=96, h=64))
    H.assert_same(env, pair, "BoxBlur", args)


@pytest.mark.parametrize("w", [33, 47, 64, 65])
def test_boxblur_simd_boundary_widths_and_crop(env, w):
    fmt = "YUV444P16"
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=w + 15, h=40))
    env.set_clip("src", pair.avs)
    # Crop(left=6) gives an unaligned, foreign-pitch source plane.
    a = env.eval(f"src.Crop(6, 0, {w}, 0).vszip_BoxBlur(hradius=3, vradius=2, vpasses=2)")
    b = H.vszip().BoxBlur(H.core.std.Crop(pair.vs, 6, (w + 15) - 6 - w), hradius=3, vradius=2, vpasses=2)
    for x, y in zip(a.frames(), H.vs_frames(b)):
        H.compare_planes(x, y)


def test_boxblur_errors(env):
    env.eval('src = BlankClip(pixel_type="YV12", length=1, width=32, height=32)', want_clip=False)
    with pytest.raises(AvsError, match="nothing to be performed"):
        env.eval("src.vszip_BoxBlur(hradius=0, vradius=0)")
    with pytest.raises(AvsError, match="hradius too large"):
        env.eval("src.vszip_BoxBlur(hradius=8)")
