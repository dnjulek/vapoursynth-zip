"""Tonemap vs vszip: bit-exact across formats, parameters, HDR metadata
props, peak detection and Dolby Vision RPU chains in any request order, the
FEL, output props, cropped sources and errors."""

import random
import types

import pytest

import harness as H
from avsrt import AvsError
from test_tonemap import _DM_A, _DM_B, RPU_FEL, RPU_MEL, RPU_P8, _rpu_build

OUT_PROPS = ["_Matrix", "_Transfer", "_Primaries", "_ColorRange"]


def pair(env, fmt, frames, props=None):
    """`frames` in both hosts with per-frame `props` (VapourSynth names;
    `_Range` becomes AviSynth's inverted `_ColorRange`)."""
    props = props or [{}] * len(frames)
    avs_props = []
    for fp in props:
        q = dict(fp)
        if "_Range" in q:
            q["_ColorRange"] = 1 - q.pop("_Range")
        avs_props.append(q)
    clip = H.vs_source(frames, fmt)

    def sel(n, f):
        fout = f.copy()
        for k, v in props[n].items():
            fout.props[k] = v
        return fout

    return types.SimpleNamespace(
        name=fmt,
        avs=env.source(frames, fmt, props=avs_props),
        vs=H.core.std.ModifyFrame(clip, clip, sel),
    )


def same(avs_clip, vs_clip, order=None, props=OUT_PROPS):
    assert avs_clip.num_frames == vs_clip.num_frames
    ns = list(range(avs_clip.num_frames)) if order is None else order
    for n, x, y in zip(ns, [avs_clip.frame(n) for n in ns], H.vs_frames(vs_clip, ns)):
        H.compare_planes(x, y, 0.0, what=f"frame {n}")
        H.compare_props(x, y, props, what=f"frame {n}")
        assert "_ChromaLocation" not in x.props


@pytest.mark.parametrize("fmt", ["YUV420P10", "YUV420P16", "YUV422P10", "YUV444P16", "YUV420PS", "YUV444PS", "RGBP10", "RGBP16", "RGBPS"])
@pytest.mark.parametrize("args", [{}, {"dynamic_peak_detection": False}, {"src_csp": 2}, {"dst_csp": 1, "dst_prim": 3}])
def test_formats(env, fmt, args):
    p = pair(env, fmt, H.make_frames(fmt, w=96, h=64, n=3))
    a, v = H.run_both(env, p, "Tonemap", {"smoothing_period": 1.0, **args})
    same(a, v)


@pytest.mark.parametrize("args", [
    {"gamut_mapping": 0, "tone_mapping_function_s": "bt2390"},
    {"tone_mapping_function": 3, "tone_mapping_param": 0.7},
    {"contrast_recovery": 0.5, "contrast_smoothness": 2.0},
    {"contrast_recovery": 0.5, "cache_frame": False},
    {"show_clipping": True},
    {"cache_frame": False},
    {"percentile": 99.5, "black_cutoff": 0.0, "scene_threshold_low": 0.5},
    {"lut3d_size": [24, 0, 64], "lut3d_tricubic": True},
    {"inverse_tone_mapping": True, "dst_csp": 1},
    {"gamut_expansion": True, "knee_adaptation": 0.6, "spline_contrast": 0.8},
    {"metadata": 2, "src_max": 4000.0, "dst_max": 150.0, "dst_min": 0.1},
])
def test_parameters(env, args):
    fmt = "YUV420P16"
    p = pair(env, fmt, H.make_frames(fmt, w=96, h=64, n=4))
    a, v = H.run_both(env, p, "Tonemap", {"smoothing_period": 2.0, **args})
    same(a, v)


def test_metadata_props(env):
    fmt = "YUV420P10"
    meta = {"MasteringDisplayMaxLuminance": 1000.0, "MasteringDisplayMinLuminance": 0.005,
            "ContentLightLevelMax": 1200, "ContentLightLevelAverage": 300.0,
            "MasteringDisplayPrimariesX": [0.68, 0.265, 0.15], "MasteringDisplayPrimariesY": [0.32, 0.69, 0.06],
            "MasteringDisplayWhitePointX": 0.3127, "MasteringDisplayWhitePointY": 0.329,
            "_ChromaLocation": 2, "_Range": 1}
    frames = H.make_frames(fmt, w=96, h=64, n=3)
    p = pair(env, fmt, frames, [meta, {**meta, "_Range": 0}, {"PLSceneMax": [600.0, 500.0, 400.0], "PLSceneAvg": 80.0}])
    for args in ({}, {"dynamic_peak_detection": False}, {"metadata": 3}):
        a, v = H.run_both(env, p, "Tonemap", {"smoothing_period": 1.0, **args})
        same(a, v)


def test_peak_detection_any_order(env):
    fmt = "YUV420P16"
    p = pair(env, fmt, H.make_frames(fmt, w=96, h=64, n=10, kind="smooth"))
    a, v = H.run_both(env, p, "Tonemap", {"smoothing_period": 2.0})
    order = list(range(10))
    random.Random(4).shuffle(order)
    same(a, v, order=order + [9, 0, 5])


def _dovi_rpus():
    """test_tonemap's mixed chain: complete RPUs, then partial ones (use_prev,
    no DM data, compressed DM data) that continue the last complete one."""
    seq = [dict(dm=_DM_A), dict(use_prev=True, dm=_DM_B), dict(), dict(compressed=True),
           dict(use_prev=True, compressed=True), dict(dm=_DM_B), dict(), dict(use_prev=True, dm=_DM_A),
           dict(compressed=True), dict(dm=_DM_A), dict(use_prev=True, compressed=True), dict()]
    return [_rpu_build(**f) for f in seq] + [RPU_P8]


@pytest.mark.parametrize("detect", [False, True])
def test_dovi_rpu_chain_any_order(env, detect):
    fmt = "YUV420P10"
    rpus = _dovi_rpus()
    p = pair(env, fmt, H.make_frames(fmt, w=96, h=64, n=len(rpus)), [{"DolbyVisionRPU": r} for r in rpus])
    a, v = H.run_both(env, p, "Tonemap", {"src_csp": 3, "dynamic_peak_detection": detect, "smoothing_period": 2.0})
    order = list(range(len(rpus)))
    random.Random(5).shuffle(order)
    same(a, v, order=order)
    same(a, v)


def test_rpu_metadata_applies_to_hdr10(env):
    fmt = "YUV420P16"
    p = pair(env, fmt, H.make_frames(fmt, w=96, h=64, n=2), [{"DolbyVisionRPU": _rpu_build(dm=_DM_A)}] * 2)
    a, v = H.run_both(env, p, "Tonemap", {})
    same(a, v)


@pytest.mark.parametrize("detect", [False, True])
@pytest.mark.parametrize(("el_fmt", "el_size"), [("YUV420P10", (48, 32)), ("YUV444P16", (96, 64))])
def test_dovi_fel(env, detect, el_fmt, el_size):
    fmt = "YUV420P16"
    rpus = [RPU_FEL, _rpu_build(use_prev=True, dm=_DM_A), RPU_MEL, RPU_FEL]
    n = len(rpus)
    bl = pair(env, fmt, H.make_frames(fmt, w=96, h=64, n=n))
    el = pair(env, el_fmt, H.make_frames(el_fmt, w=el_size[0], h=el_size[1], n=n, seed=7), [{"DolbyVisionRPU": r} for r in rpus])
    env.set_clip("el", el.avs)
    a, v = H.run_both(env, bl, "Tonemap", {"src_csp": 3, "dovi_el": H.Var("el", el.vs), "dynamic_peak_detection": detect,
                                           "smoothing_period": 2.0})
    same(a, v, order=[3, 1, 0, 2])
    same(a, v)


def test_cropped_source(env):
    fmt = "YUV420P16"
    p = pair(env, fmt, H.make_frames(fmt, w=160, h=96, n=2))
    env.set_clip("src", p.avs)
    a = env.eval("src.Crop(10, 4, -46, -12).avszip_Tonemap(smoothing_period=1.0)")
    v = H.vszip().Tonemap(H.core.std.Crop(p.vs, 10, 46, 4, 12), smoothing_period=1.0)
    same(a, v)


def test_alpha_is_dropped(env):
    fmt = "YUV420P16"
    p = pair(env, fmt, H.make_frames(fmt, w=96, h=64, n=2))
    env.set_clip("src", p.avs)
    a = env.eval("src.AddAlphaPlane(1000).avszip_Tonemap()")
    v = H.vszip().Tonemap(p.vs)
    same(a, v)


def test_dovi_frame_errors(env):
    """Frame-time errors, as in vszip: an unparsable RPU, a missing one, and
    a partial RPU whose walk back reaches the missing one."""
    fmt = "YUV420P10"
    props = [{"DolbyVisionRPU": _rpu_build(dm=_DM_A)}, {"DolbyVisionRPU": b"\x00\x01garbage"}, {},
             {"DolbyVisionRPU": _rpu_build(compressed=True)}]
    p = pair(env, fmt, H.make_frames(fmt, w=96, h=64, n=4), props)
    env.set_clip("src", p.avs)
    a = env.eval("src.avszip_Tonemap(src_csp=3, dynamic_peak_detection=false)")
    a.frame(0)
    for n, msg in ((1, "invalid `DolbyVisionRPU`"), (2, "missing `DolbyVisionRPU`"), (3, "missing `DolbyVisionRPU`")):
        with pytest.raises(AvsError, match=msg):
            a.frame(n)
    v = H.vszip().Tonemap(p.vs, src_csp=3, dynamic_peak_detection=0)
    for n, msg in ((1, "invalid `DolbyVisionRPU`"), (2, "missing `DolbyVisionRPU`"), (3, "missing `DolbyVisionRPU`")):
        with pytest.raises(H.vs.Error, match=msg):
            v.get_frame(n)


@pytest.mark.parametrize(("call", "msg"), [
    ("g16.avszip_Tonemap()", "input must be 10- to 16-bit"),
    ("y8.avszip_Tonemap()", "input must be 10- to 16-bit"),
    ("small.avszip_Tonemap()", "at least 64x64"),
    ("yuv.avszip_Tonemap(src_csp=0)", "src_csp must be 1"),
    ("yuv.avszip_Tonemap(dst_csp=3)", "not supported"),
    ("rgb.avszip_Tonemap(src_csp=3)", "integer YUV"),
    ("yuv.avszip_Tonemap(tone_mapping_function_s=\"nope\")", "unknown tone_mapping_function_s"),
    ("yuv.avszip_Tonemap(lut3d_size=[2, 2])", "3 values"),
    ("yuv.avszip_Tonemap(lut3d_size=[1, 2, 2])", "0 \\(default\\) or 2..1024"),
    ("yuv.avszip_Tonemap(smoothing_period=61.0)", "smoothing_period"),
    ("yuv.avszip_Tonemap(src_csp=3, dovi_el=rgb)", "dovi_el must be"),
    ("yuv.avszip_Tonemap(src_csp=3, dovi_el=yuv.BilinearResize(80, 64))", "dovi_el must be"),
])
def test_errors(env, call, msg):
    env.eval('g16 = BlankClip(pixel_type="Y16", length=2, width=96, height=64)\n'
             'y8 = BlankClip(pixel_type="YV12", length=2, width=96, height=64)\n'
             'small = BlankClip(pixel_type="YUV420P16", length=2, width=96, height=48)\n'
             'yuv = BlankClip(pixel_type="YUV420P16", length=2, width=96, height=64)\n'
             'rgb = BlankClip(pixel_type="RGBP16", length=2, width=96, height=64)', want_clip=False)
    with pytest.raises(AvsError, match=msg) as e:
        env.eval(call)
    assert str(e.value).startswith("avszip_Tonemap:")
