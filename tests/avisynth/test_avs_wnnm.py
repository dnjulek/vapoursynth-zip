"""WNNM vs vszip (spatial and two-stage temporal graph)."""

import random

import pytest

import harness as H
from avsrt import AvsError


def same(avs_clip, vs_clip, order=None, tol=0.0):
    assert avs_clip.num_frames == vs_clip.num_frames
    ns = list(range(avs_clip.num_frames)) if order is None else order
    for n, x, y in zip(ns, [avs_clip.frame(n) for n in ns], H.vs_frames(vs_clip, ns)):
        H.compare_planes(x, y, tol, what=f"frame {n}")


@pytest.mark.parametrize("fmt", ["Y32", "YUV420PS", "YUV444PS", "RGBPS"])
@pytest.mark.parametrize("args", [
    {}, {"sigma": [5.0, 2.0]}, {"block_size": 4, "block_step": 2, "group_size": 4}, {"block_size": 6, "bm_range": 3},
    {"residual": True}, {"adaptive_aggregation": False}, {"sigma": [3.0, 0.0]}, {"block_size": 5, "group_size": 12},
])
def test_wnnm_spatial(env, fmt, args):
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=48, h=32, n=2))
    a, v = H.run_both(env, pair, "WNNM", args)
    same(a, v)


@pytest.mark.parametrize("fmt", ["Y32", "YUV420PS"])
@pytest.mark.parametrize("args", [{"radius": 1}, {"radius": 2, "ps_num": 3, "ps_range": 2}, {"radius": 1, "residual": True, "block_size": 4}])
def test_wnnm_temporal(env, fmt, args):
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=40, h=32, n=5))
    a, v = H.run_both(env, pair, "WNNM", args)
    same(a, v)
    order = list(range(5))
    random.Random(1).shuffle(order)
    same(a, v, order=order + [4, 0, 2])


def test_wnnm_rclip(env):
    fmt = "Y32"
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=40, h=32, n=3, seed=1))
    ref = H.Pair(env, fmt, H.make_frames(fmt, w=40, h=32, n=3, seed=2, kind="smooth"))
    env.set_clip("r", ref.avs)
    for radius in (0, 1):
        a, v = H.run_both(env, pair, "WNNM", {"radius": radius, "rclip": H.Var("r", ref.vs)})
        same(a, v)


def test_wnnm_errors(env):
    env.eval('i8 = BlankClip(pixel_type="YV12", length=1, width=32, height=32)\n'
             'f = BlankClip(pixel_type="Y32", length=1, width=32, height=32)', want_clip=False)
    with pytest.raises(AvsError, match="32 bit float"):
        env.eval("i8.vszip_WNNM()")
    with pytest.raises(AvsError, match="block_size"):
        env.eval("f.vszip_WNNM(block_size=65)")
    with pytest.raises(AvsError, match="radius"):
        env.eval("f.vszip_WNNM(radius=16)")
    with pytest.raises(AvsError, match="non-negative"):
        env.eval("f.vszip_WNNM(sigma=[-1.0])")
