"""NLMeans vs vszip: bit-exact across formats, channels, parameters, temporal
window edges, rclip, cropped sources, alpha passthrough and request order."""

import random

import numpy as np
import pytest

import harness as H
from avsrt import AvsError


def same(avs_clip, vs_clip, order=None, planes=None):
    assert avs_clip.num_frames == vs_clip.num_frames
    ns = list(range(avs_clip.num_frames)) if order is None else order
    for n, x, y in zip(ns, [avs_clip.frame(n) for n in ns], H.vs_frames(vs_clip, ns)):
        H.compare_planes(x, y, 0.0, what=f"frame {n}", planes=planes)


@pytest.mark.parametrize(("fmt", "args"), [
    ("Y8", {}),
    ("Y10", {}),
    ("Y16", {}),
    ("Y32", {}),
    ("YV12", {}),
    ("YUV420P16", {"channels": "UV"}),
    ("YUV422P10", {"channels": "UV", "a": 1}),
    ("YV24", {"channels": "YUV"}),
    ("YUV444PS", {"channels": "YUV"}),
    ("RGBP", {}),
    ("RGBP16", {}),
    ("RGBPS", {}),
])
def test_formats(env, fmt, args):
    # n=4 with d=1: frame 0 has radius 0, the last frame clamps its future side
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=72, h=40, n=4))
    a, v = H.run_both(env, pair, "NLMeans", args)
    same(a, v)


@pytest.mark.parametrize("args", [
    {"d": 0},
    {"d": 2},
    {"d": 3, "a": 1, "s": 1},
    {"a": 4, "s": 2},
    {"s": 0},
    {"h": 3.0},
    {"wmode": 1},
    {"wmode": 2, "h": 2.0},
    {"wmode": 3, "h": 4.0},
    {"wref": 0.25},
    {"wref": 0.0},
])
def test_parameters(env, args):
    fmt = "Y16"
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=72, h=40, n=5))
    a, v = H.run_both(env, pair, "NLMeans", args)
    same(a, v)


def test_tile_and_vector_boundaries(env):
    # wider than a 512-wide tile, taller than a 64-row tile, odd sizes
    fmt = "Y32"
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=531, h=71, n=2))
    a, v = H.run_both(env, pair, "NLMeans", {"a": 1, "s": 2})
    same(a, v)


def test_rclip(env):
    fmt = "YUV420P16"
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=64, h=48, n=3, seed=1))
    ref = H.Pair(env, fmt, H.make_frames(fmt, w=64, h=48, n=3, seed=2, kind="smooth"))
    env.set_clip("r", ref.avs)
    a, v = H.run_both(env, pair, "NLMeans", {"rclip": H.Var("r", ref.vs)})
    same(a, v)


def test_cropped_source(env):
    """A horizontally cropped plane has an unaligned start and a foreign
    pitch; the kernel reads it in place."""
    fmt = "YV12"
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=128, h=48, n=3))
    env.set_clip("src", pair.avs)
    a = env.eval("src.Crop(6, 2, -42, -4).avszip_NLMeans(channels=\"UV\", a=1)")
    v = H.vszip().NLMeans(H.core.std.Crop(pair.vs, 6, 42, 2, 4), channels="UV", a=1)
    same(a, v)


def test_alpha_passes_through(env):
    fmt = "YUV420P16"
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=64, h=32, n=2))
    env.set_clip("src", pair.avs)
    a = env.eval("src.AddAlphaPlane(1234).avszip_NLMeans()")
    v = H.vszip().NLMeans(pair.vs)
    same(a, v, planes=[0, 1, 2])
    for n in range(a.num_frames):
        alpha = a.frame(n).planes[3]
        assert (alpha == 1234).all()


def test_request_order(env):
    fmt = "Y8"
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=48, h=32, n=6))
    a, v = H.run_both(env, pair, "NLMeans", {"d": 2})
    order = list(range(6))
    random.Random(3).shuffle(order)
    same(a, v, order=order + [5, 0, 3, 3])


@pytest.mark.parametrize(("call", "msg"), [
    ("g.avszip_NLMeans(d=17)", "d must be 0..16"),
    ("g.avszip_NLMeans(a=0)", "a must be 1..64"),
    ("g.avszip_NLMeans(s=9)", "s must be 0..8"),
    ("g.avszip_NLMeans(h=0.0)", "h must be > 0"),
    ("g.avszip_NLMeans(h=-1.0)", "h value -1 is out of range"),
    ("g.avszip_NLMeans(wmode=4)", "wmode must be 0..3"),
    ("g.avszip_NLMeans(wref=-1.0)", "wref value -1 is out of range"),
    ("g.avszip_NLMeans(channels=\"UV\")", "'channels' must be 'Y' with Gray"),
    ("y12.avszip_NLMeans(channels=\"YUV\")", "requires 4:4:4"),
    ("y12.avszip_NLMeans(channels=\"RGB\")", "'channels' must be 'YUV', 'Y' or 'UV' with YUV"),
    ("rgb.avszip_NLMeans(channels=\"Y\")", "'channels' must be 'RGB' with RGB"),
    ("g.avszip_NLMeans(a=20)", r"research window \(2\*a\+1\) larger"),
    ("g.avszip_NLMeans(rclip=g16)", "same bit depth"),
    ("g.avszip_NLMeans(rclip=g.Trim(0, 1))", "same length"),
])
def test_errors(env, call, msg):
    env.eval('g = BlankClip(pixel_type="Y8", length=3, width=40, height=32)\n'
             'g16 = BlankClip(pixel_type="Y16", length=3, width=40, height=32)\n'
             'y12 = BlankClip(pixel_type="YV12", length=3, width=40, height=32)\n'
             'rgb = BlankClip(pixel_type="RGBP", length=3, width=40, height=32)', want_clip=False)
    with pytest.raises(AvsError, match=msg) as e:
        env.eval(call)
    assert str(e.value).startswith("avszip_NLMeans:")
