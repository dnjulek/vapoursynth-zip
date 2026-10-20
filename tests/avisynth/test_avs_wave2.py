"""AdaptiveBinarize, AdaptiveGrainMask, ColorMap, CombMask, CombMaskMT,
LimitFilter, Compress vs vszip."""

import pytest

import harness as H
from avsrt import AvsError


def two(env, fmt, w=96, h=64, n=4):
    a = H.Pair(env, fmt, H.make_frames(fmt, w=w, h=h, n=n, seed=1))
    b = H.Pair(env, fmt, H.make_frames(fmt, w=w, h=h, n=n, seed=5, kind="smooth"))
    env.set_clip("a", a.avs)
    env.set_clip("b", b.avs)
    return a, b


def same(avs_clip, vs_clip, tol=0.0, props=()):
    for i, (x, y) in enumerate(zip(avs_clip.frames(), H.vs_frames(vs_clip))):
        H.compare_planes(x, y, tol, what=f"frame {i}")
        H.compare_props(x, y, props, what=f"frame {i}")


@pytest.mark.parametrize("fmt", ["Y8", "YV12", "YV24", "RGBP"])
@pytest.mark.parametrize("cval", [None, 0, 3, -20, 300])
def test_adaptive_binarize(env, fmt, cval):
    a, b = two(env, fmt)
    extra = {} if cval is None else {"c": cval}
    out = env.eval("vszip_AdaptiveBinarize(a, b" + ("" if cval is None else f", c={cval}") + ")")
    same(out, H.vszip().AdaptiveBinarize(a.vs, b.vs, **extra), props=["_ColorRange"])


@pytest.mark.parametrize("fmt", ["Y8", "YV12", "YUV420P10", "YUV444P16", "Y32", "YUV420PS"])
@pytest.mark.parametrize("ls", [None, 1.0, 10.0, 25.5])
def test_adaptive_grain_mask(env, fmt, ls):
    pair = H.Pair(env, fmt, H.make_frames(fmt))
    args = {} if ls is None else {"luma_scaling": ls}
    a, v = H.run_both(env, pair, "AdaptiveGrainMask", args)
    assert a.family == "gray" and a.bits == v.format.bits_per_sample
    same(a, v)


@pytest.mark.parametrize("color", [None, 0, 2, 12, 20, 21])
def test_color_map(env, color):
    pair = H.Pair(env, "Y8", H.make_frames("Y8", w=100, h=48))
    args = {} if color is None else {"color": color}
    a, v = H.run_both(env, pair, "ColorMap", args)
    assert a.family == "rgb"
    same(a, v, props=["_Matrix", "_Transfer", "_Primaries", "_ColorRange"])


@pytest.mark.parametrize("fmt", ["Y8", "YV12", "YV24"])
@pytest.mark.parametrize("args", [
    {}, {"cthresh": 10, "mthresh": 0}, {"expand": False}, {"metric": 1, "cthresh": 200},
    {"metric": 1, "mthresh": 0, "expand": False},
])
def test_comb_mask(env, fmt, args):
    pair = H.Pair(env, fmt, H.make_frames(fmt, n=5, kind="noise"))
    a, v = H.run_both(env, pair, "CombMask", args)
    same(a, v)


@pytest.mark.parametrize("fmt", ["Y8", "YV12", "YV24"])
@pytest.mark.parametrize("args", [{}, {"thY1": 10, "thY2": 40}, {"thY1": 0, "thY2": 255}])
def test_comb_mask_mt(env, fmt, args):
    pair = H.Pair(env, fmt, H.make_frames(fmt, kind="noise"))
    a, v = H.run_both(env, pair, "CombMaskMT", args)
    same(a, v)


@pytest.mark.parametrize("fmt", ["Y8", "YV12", "YUV444P10", "YUV420P16", "RGBP16", "Y32", "YUV444PS"])
@pytest.mark.parametrize("args", [
    {}, {"dark_thr": [3.0], "bright_thr": [1.5]}, {"elast": [4.0, 1.5]}, {"planes": [0]},
    {"dark_thr": [10.0, 2.0, 0.5], "bright_thr": [0.0], "elast": [1.0]},
])
@pytest.mark.parametrize("use_ref", [False, True])
def test_limit_filter(env, fmt, args, use_ref):
    if "planes" in args and H.fmt_info(fmt).num_planes == 1:
        pytest.skip("gray")
    flt, src = two(env, fmt)
    ref = H.Pair(env, fmt, H.make_frames(fmt, seed=3))
    env.set_clip("r", ref.avs)
    call = H.avs_call("LimitFilter", args)
    call = call.replace("vszip_LimitFilter(", "vszip_LimitFilter(a, b" + (", ref=r" if use_ref else "") + (", " if args else ""))
    out = env.eval(call)
    kw = dict(args)
    if use_ref:
        kw["ref"] = ref.vs
    same(out, H.vszip().LimitFilter(flt.vs, src.vs, **kw))


@pytest.mark.parametrize("fmt", ["Y8", "YV12", "YV16", "YV24"])
@pytest.mark.parametrize("args", [
    {}, {"qscale": 1}, {"qscale": 31, "dc_prec": 3}, {"codec": 1}, {"codec": 1, "quality": 95}, {"chroma": False},
])
@pytest.mark.parametrize("wh", [(96, 64), (75, 42)])
def test_compress(env, fmt, args, wh):
    w, h = wh
    if fmt != "YV24" and fmt != "Y8":
        w, h = w - w % 2, h - h % 2
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=w, h=h))
    a, v = H.run_both(env, pair, "Compress", args)
    same(a, v)


def test_wave2_errors(env):
    env.eval('y16 = BlankClip(pixel_type="Y16", length=1)\ny8 = BlankClip(pixel_type="Y8", length=1)\n'
             'rgb = BlankClip(pixel_type="RGBP", length=1)\nyv12 = BlankClip(pixel_type="YV12", length=1)', want_clip=False)
    with pytest.raises(AvsError, match="only 8 bit int"):
        env.eval("vszip_AdaptiveBinarize(y16, y16)")
    with pytest.raises(AvsError, match="only Gray8"):
        env.eval("yv12.vszip_ColorMap()")
    with pytest.raises(AvsError, match="between 0 and 21"):
        env.eval("y8.vszip_ColorMap(color=22)")
    with pytest.raises(AvsError, match="thY1 can't be greater"):
        env.eval("y8.vszip_CombMaskMT(thY1=50, thY2=20)")
    with pytest.raises(AvsError, match="cthresh must be between 0 and 255"):
        env.eval("y8.vszip_CombMask(cthresh=300)")
    with pytest.raises(AvsError, match="Gray or YUV"):
        env.eval("rgb.vszip_Compress()")
    with pytest.raises(AvsError, match="qscale must be between"):
        env.eval("y8.vszip_Compress(qscale=0)")
    with pytest.raises(AvsError, match=r"dark_thr\[0\] value 300 is out of range"):
        env.eval("vszip_LimitFilter(y8, y8, dark_thr=[300])")
