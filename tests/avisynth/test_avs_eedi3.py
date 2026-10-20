"""EEDI3 / EEDI3H vs vszip."""

import random

import pytest

import harness as H
from avsrt import AvsError

FMTS = ["Y8", "YV12", "YUV444P10", "YUV420P16", "Y32", "YUV420PS"]


def same(avs_clip, vs_clip, props=(), order=None):
    assert avs_clip.num_frames == vs_clip.num_frames
    assert (avs_clip.width, avs_clip.height) == (vs_clip.width, vs_clip.height)
    ns = list(range(avs_clip.num_frames)) if order is None else order
    for n, x, y in zip(ns, [avs_clip.frame(n) for n in ns], H.vs_frames(vs_clip, ns)):
        H.compare_planes(x, y, what=f"frame {n}")
        H.compare_props(x, y, props, what=f"frame {n}")


@pytest.mark.parametrize("name", ["EEDI3", "EEDI3H"])
@pytest.mark.parametrize("fmt", FMTS)
@pytest.mark.parametrize("args", [
    {"field": 0}, {"field": 1}, {"field": 1, "dh": True}, {"field": 0, "dh": True, "hp": True},
    {"field": 1, "vcheck": 0}, {"field": 0, "vcheck": 3, "mdis": 8, "nrad": 1}, {"field": 1, "alpha": 0.5, "beta": 0.3, "gamma": 40.0},
])
def test_eedi3(env, name, fmt, args):
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=64, h=48, n=2))
    a, v = H.run_both(env, pair, name, args)
    same(a, v, props=["_FieldBased"])


@pytest.mark.parametrize("name", ["EEDI3", "EEDI3H"])
@pytest.mark.parametrize("field", [2, 3])
def test_eedi3_bob(env, name, field):
    fmt = "YV12"
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=64, h=48, n=3))
    a, v = H.run_both(env, pair, name, {"field": field})
    assert a.fps == (48, 1)
    assert v.fps.numerator == 48 and v.fps.denominator == 1
    same(a, v, props=["_FieldBased"])
    # Bob output depends only on n: shuffled and repeated requests agree.
    order = list(range(a.num_frames))
    random.Random(5).shuffle(order)
    same(a, v, order=order + [0, 5, 0])


def test_eedi3_field_prop_and_clips(env):
    fmt = "YUV420P16"
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=64, h=48, n=2), props={"_FieldBased": 2})
    s = H.Pair(env, fmt, H.make_frames(fmt, w=64, h=48, n=2, seed=6))
    m = H.Pair(env, "Y16", H.make_frames("Y16", w=64, h=48, n=2, seed=8))
    env.set_clip("s", s.avs)
    env.set_clip("m", m.avs)
    a, v = H.run_both(env, pair, "EEDI3", {"field": 0, "sclip": H.Var("s", s.vs), "mclip": H.Var("m", m.vs)})
    same(a, v, props=["_FieldBased"])


def test_eedi3_errors(env):
    env.eval('src = BlankClip(pixel_type="YV12", length=2, width=64, height=48)\n'
             'odd = BlankClip(pixel_type="YV24", length=2, width=64, height=47)', want_clip=False)
    with pytest.raises(AvsError, match="field must be 0, 1, 2, or 3"):
        env.eval("src.vszip_EEDI3(4)")
    with pytest.raises(AvsError, match="field must be 0 or 1 when dh"):
        env.eval("src.vszip_EEDI3(2, dh=true)")
    with pytest.raises(AvsError, match="height must be mod 2"):
        env.eval("odd.vszip_EEDI3(0)")
    with pytest.raises(AvsError, match="alpha \\+ beta"):
        env.eval("src.vszip_EEDI3(0, alpha=0.8, beta=0.5)")
    with pytest.raises(AvsError, match="mclip must be Gray"):
        env.eval("src.vszip_EEDI3(0, mclip=src)")
