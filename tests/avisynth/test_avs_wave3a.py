"""Bilateral, BilateralDither, MosquitoNR vs vszip."""

import pytest

import harness as H
from avsrt import AvsError


def same(avs_clip, vs_clip, tol=0.0):
    for i, (x, y) in enumerate(zip(avs_clip.frames(), H.vs_frames(vs_clip))):
        H.compare_planes(x, y, tol, what=f"frame {i}")


BIL_FMTS = ["Y8", "YV12", "YUV444P10", "YUV420P16", "RGBP", "Y32", "YUV444PS"]


@pytest.mark.parametrize("fmt", BIL_FMTS)
@pytest.mark.parametrize("args", [
    {},
    {"sigmaS": [1.5], "sigmaR": [0.05]},
    {"sigmaS": [2.0, 1.0], "sigmaR": [0.1, 0.02], "algorithm": [1]},
    {"sigmaS": [4.0], "sigmaR": [0.01], "algorithm": [2]},
    {"sigmaS": [6.0], "algorithm": [1], "PBFICnum": [8]},
    {"planes": [0]},
])
def test_bilateral(env, fmt, args):
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=96, h=64, n=2))
    a, v = H.run_both(env, pair, "Bilateral", args)
    same(a, v)


@pytest.mark.parametrize("fmt", ["YV12", "YUV444P16", "Y32"])
def test_bilateral_joint(env, fmt):
    pair = H.Pair(env, fmt, H.make_frames(fmt, n=2, seed=1))
    ref = H.Pair(env, fmt, H.make_frames(fmt, n=2, seed=4, kind="smooth"))
    env.set_clip("r", ref.avs)
    a, v = H.run_both(env, pair, "Bilateral", {"ref": H.Var("r", ref.vs), "sigmaS": [2.0]})
    same(a, v)


@pytest.mark.parametrize("fmt", ["Y8", "YV12", "YUV444P10", "YUV420P16", "Y32", "YUV420PS"])
@pytest.mark.parametrize("args", [
    {"radius": [4]},
    {"radius": [6], "thr": [4.0], "flat": [0.2], "wmin": [0.5]},
    {"radius": [5, 3], "subspl": [4.0]},
    {"radius": [3], "planes": [0]},
])
def test_bilateral_dither(env, fmt, args):
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=80, h=48, n=2))
    a, v = H.run_both(env, pair, "BilateralDither", args)
    same(a, v)


def test_bilateral_dither_ref(env):
    fmt = "YUV420P16"
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=80, h=48, n=2, seed=1))
    ref = H.Pair(env, fmt, H.make_frames(fmt, w=80, h=48, n=2, seed=8))
    env.set_clip("r", ref.avs)
    a, v = H.run_both(env, pair, "BilateralDither", {"ref": H.Var("r", ref.vs), "radius": [4]})
    same(a, v)


@pytest.mark.parametrize("fmt", ["Y8", "YV12", "YUV444P10", "YUV420P16", "Y32", "YUV444PS"])
@pytest.mark.parametrize("args", [
    {}, {"strength": [8], "restore": [64]}, {"radius": [1]}, {"strength": [32, 16], "planes": [0, 1, 2]}, {"strength": [0]},
])
def test_mosquito_nr(env, fmt, args):
    if "planes" in args and H.fmt_info(fmt).num_planes == 1:
        pytest.skip("gray")
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=90, h=50, n=3))
    a, v = H.run_both(env, pair, "MosquitoNR", args)
    same(a, v)


def test_wave3a_errors(env):
    env.eval('y8 = BlankClip(pixel_type="Y8", length=1, width=64, height=64)\n'
             'small = BlankClip(pixel_type="Y8", length=1, width=8, height=8)\n'
             'rgb = BlankClip(pixel_type="RGBP", length=1)', want_clip=False)
    with pytest.raises(AvsError, match="non-negative"):
        env.eval("y8.avszip_Bilateral(sigmaS=[-1.0])")
    with pytest.raises(AvsError, match="except 1"):
        env.eval("y8.avszip_Bilateral(PBFICnum=[1])")
    with pytest.raises(AvsError, match="16x16 min"):
        env.eval("small.avszip_BilateralDither()")
    with pytest.raises(AvsError, match="YUV or Gray"):
        env.eval("rgb.avszip_MosquitoNR()")
    with pytest.raises(AvsError, match=r"strength\[0\] value 40 is out of range"):
        env.eval("y8.avszip_MosquitoNR(strength=[40])")
