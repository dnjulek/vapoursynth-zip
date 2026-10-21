"""Checkmate, CLAHE, XPSNR, SSIMULACRA2 vs vszip."""

import random

import pytest

import harness as H
from avsrt import AvsError


def same(avs_clip, vs_clip, tol=0.0, props=(), prop_tol=0.0, order=None):
    assert avs_clip.num_frames == vs_clip.num_frames
    ns = list(range(avs_clip.num_frames)) if order is None else order
    fa = [avs_clip.frame(n) for n in ns]
    fb = H.vs_frames(vs_clip, ns)
    for n, x, y in zip(ns, fa, fb):
        H.compare_planes(x, y, tol, what=f"frame {n}")
        H.compare_props(x, y, props, prop_tol, what=f"frame {n}")


@pytest.mark.parametrize("fmt", ["Y8", "YV12", "YV24"])
@pytest.mark.parametrize("args", [{}, {"thr": 20, "tmax": 5}, {"tthr2": 10}, {"thr": 0, "tmax": 255, "tthr2": 3}])
def test_checkmate(env, fmt, args):
    pair = H.Pair(env, fmt, H.make_frames(fmt, n=7, kind="noise"))
    a, v = H.run_both(env, pair, "Checkmate", args)
    same(a, v)


def test_checkmate_request_order_independent(env):
    fmt = "YV12"
    pair = H.Pair(env, fmt, H.make_frames(fmt, n=8, kind="noise"))
    a, v = H.run_both(env, pair, "Checkmate", {"tthr2": 8})
    order = list(range(8))
    random.Random(3).shuffle(order)
    order = order + [7, 0, 7, 3]
    same(a, v, order=order)


@pytest.mark.parametrize("fmt", ["Y8", "YV12", "YUV444P10", "YUV420P16", "RGBP", "Y32", "YUV444PS"])
@pytest.mark.parametrize("args", [{}, {"limit": 3, "tiles": [4]}, {"limit": 15, "tiles": [2, 5]}])
def test_clahe(env, fmt, args):
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=96, h=64, n=2))
    a, v = H.run_both(env, pair, "CLAHE", args)
    same(a, v, props=["_ColorRange"])


@pytest.mark.parametrize("fmt", ["YV12", "YUV420P10", "YV24"])
@pytest.mark.parametrize("args", [{}, {"temporal": False}])
@pytest.mark.parametrize("fps", [(24, 1), (60, 1)])
def test_xpsnr(env, fmt, args, fps):
    ref = H.make_frames(fmt, w=128, h=96, n=5, seed=1)
    dist = [[p.copy() for p in f] for f in ref]
    import numpy as np
    rng = np.random.default_rng(9)
    for f in dist:
        for p in f:
            noise = rng.integers(-3, 4, size=p.shape)
            p[:] = np.clip(p.astype(np.int64) + noise, 0, (1 << H.fmt_info(fmt).bits_per_sample) - 1)
    pr = H.Pair(env, fmt, ref)
    pd = H.Pair(env, fmt, dist)
    env.set_clip("r", env.eval(f"r0 = r0") if False else pr.avs)
    env.set_clip("dd", pd.avs)
    vs_r = H.core.std.AssumeFPS(pr.vs, fpsnum=fps[0], fpsden=fps[1])
    vs_d = H.core.std.AssumeFPS(pd.vs, fpsnum=fps[0], fpsden=fps[1])
    extra = "".join(f", {k}={'true' if v else 'false'}" for k, v in args.items())
    a = env.eval(f"avszip_XPSNR(r.AssumeFPS({fps[0]}, {fps[1]}), dd.AssumeFPS({fps[0]}, {fps[1]}), verbose=false{extra})")
    v = H.vszip().XPSNR(vs_r, vs_d, verbose=0, **{k: int(x) for k, x in args.items()})
    same(a, v, props=["XPSNR_Y", "XPSNR_U", "XPSNR_V"])


@pytest.mark.parametrize("fmt,tol", [("RGBP", 0.0), ("RGBP16", 0.0), ("RGBPS", 0.0), ("YV12", 5.0), ("YUV444P16", 0.25)])
def test_ssimulacra2(env, fmt, tol):
    import numpy as np
    ref = H.make_frames(fmt, w=96, h=64, n=2, seed=1, kind="smooth")
    dist = [[p.copy() for p in f] for f in ref]
    rng = np.random.default_rng(4)
    f = H.fmt_info(fmt)
    for fr in dist:
        for p in fr:
            if f.sample_type == 1:
                p += rng.normal(0, 0.02, size=p.shape).astype(np.float32)
            else:
                peak = (1 << f.bits_per_sample) - 1
                p[:] = np.clip(p.astype(np.int64) + rng.integers(-peak // 40, peak // 40 + 1, size=p.shape), 0, peak)
    pr = H.Pair(env, fmt, ref)
    pd = H.Pair(env, fmt, dist)
    env.set_clip("r", pr.avs)
    env.set_clip("dd", pd.avs)
    a = env.eval("avszip_SSIMULACRA2(r, dd)")
    v = H.vszip().SSIMULACRA2(pr.vs, pd.vs)
    fa, fb = a.frames(), H.vs_frames(v)
    for x, y in zip(fa, fb):
        if tol == 0.0:
            H.compare_planes(x, y)
        H.compare_props(x, y, ["SSIMULACRA2"], tol)


def test_wave3b_errors(env):
    env.eval('y8 = BlankClip(pixel_type="Y8", length=3)\nyv12 = BlankClip(pixel_type="YV12", length=3)\n'
             'y16 = BlankClip(pixel_type="YUV420P16", length=3)', want_clip=False)
    with pytest.raises(AvsError, match="tmax value"):
        env.eval("y8.avszip_Checkmate(tmax=0)")
    with pytest.raises(AvsError, match="tiles values must be"):
        env.eval("y8.avszip_CLAHE(tiles=[0])")
    with pytest.raises(AvsError, match="only supports YUV"):
        env.eval("avszip_XPSNR(y8, y8)")
    with pytest.raises(AvsError, match="only supports 8 or 10 bit"):
        env.eval("avszip_XPSNR(y16, y16)")
    with pytest.raises(AvsError, match="same length"):
        env.eval("avszip_SSIMULACRA2(yv12, yv12.Trim(0, 1))")
