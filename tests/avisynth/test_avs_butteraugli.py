"""Butteraugli vs vszip: planes and props bit-exact on planar RGB."""

import numpy as np
import pytest

import harness as H
from avsrt import AvsError

PROPS = ["Butteraugli", "Butteraugli3Norm"]


def _pairs(env, fmt, props=None, w=96, h=64, n=2):
    ref = H.make_frames(fmt, w=w, h=h, n=n, seed=1, kind="smooth")
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
    pr, pd = H.Pair(env, fmt, ref, props), H.Pair(env, fmt, dist, props)
    env.set_clip("r", pr.avs)
    env.set_clip("dd", pd.avs)
    return pr, pd


def _same(a, v, props=PROPS):
    fa, fv = a.frames(), H.vs_frames(v)
    assert len(fa) == len(fv)
    for i, (x, y) in enumerate(zip(fa, fv)):
        H.compare_planes(x, y, what=f"frame {i}")
        H.compare_props(x, y, props, what=f"frame {i}")
    return fa


@pytest.mark.parametrize("distmap", [0, 1, 2])
@pytest.mark.parametrize("fmt", ["RGBP", "RGBP10", "RGBP16", "RGBPS"])
def test_matches_vszip(env, fmt, distmap):
    pr, pd = _pairs(env, fmt)
    a = env.eval(f"avszip_Butteraugli(r, dd, distmap={distmap})")
    v = H.vszip().Butteraugli(pr.vs, pd.vs, distmap=distmap)
    fa = _same(a, v)
    assert all(f.props["Butteraugli"] > 0 for f in fa)
    if distmap == 2:
        assert len(fa[0].planes) == 1 and fa[0].planes[0].dtype == np.float32


@pytest.mark.parametrize("args", [
    {"intensity_target": 203.0},
    {"intensity_target": 0.001, "hf_asymmetry": 0.001, "xmul": 0.0},
    {"hf_asymmetry": 2.5, "xmul": 1000.0},
])
def test_arguments_match_vszip(env, args):
    pr, pd = _pairs(env, "RGBP16")
    extra = "".join(f", {k}={v}" for k, v in args.items())
    a = env.eval(f"avszip_Butteraugli(r, dd, distmap=2{extra})")
    v = H.vszip().Butteraugli(pr.vs, pd.vs, distmap=2, **args)
    _same(a, v)


@pytest.mark.parametrize("fmt", ["RGBP16", "RGBPS"])
def test_linear_transfer_matches_vszip(env, fmt):
    """Frame 0's _Transfer = 8 skips the sRGB decode on both hosts."""
    pr, pd = _pairs(env, fmt, props={"_Transfer": 8})
    a = env.eval("avszip_Butteraugli(r, dd)")
    v = H.vszip().Butteraugli(pr.vs, pd.vs)
    fa = _same(a, v)
    _pairs(env, fmt)
    srgb = env.eval("avszip_Butteraugli(r, dd)").frame(0).props["Butteraugli"]
    assert srgb != fa[0].props["Butteraugli"]


def test_output_tags(env):
    _pairs(env, "RGBP16", props={"_Transfer": 8, "_Primaries": 9, "_Matrix": 0})
    heat = env.eval("avszip_Butteraugli(r, dd, distmap=1)").frame(0).props
    assert (heat["_Transfer"], heat["_Primaries"], heat["_Matrix"], heat["_ColorRange"]) == (13, 1, 0, 0)
    raw = env.eval("avszip_Butteraugli(r, dd, distmap=2)").frame(0).props
    assert not {"_Transfer", "_Primaries", "_Matrix"} & raw.keys()
    plain = env.eval("avszip_Butteraugli(r, dd)").frame(0).props
    assert plain["_Transfer"] == 8 and plain["_Primaries"] == 9


def test_reference_alpha_passes_through(env):
    """RGBAP reference: same scores as without alpha; distmap 0 and 1 keep the
    reference's alpha, the distorted clip's alpha is ignored."""
    _pairs(env, "RGBP")
    base = env.eval("avszip_Butteraugli(r, dd, distmap=1)").frame(0)
    for dist in ("dd", "dd.AddAlphaPlane(7)"):
        for distmap in (0, 1):
            f = env.eval(f"avszip_Butteraugli(r.AddAlphaPlane(200), {dist}, distmap={distmap})").frame(0)
            assert len(f.planes) == 4 and (f.planes[3] == 200).all()
            assert f.props["Butteraugli"] == base.props["Butteraugli"]
            assert f.props["Butteraugli3Norm"] == base.props["Butteraugli3Norm"]
            if distmap == 1:
                for p in range(3):
                    np.testing.assert_array_equal(f.planes[p], base.planes[p])


def test_identical_inputs_score_zero(env):
    _pairs(env, "RGBPS")
    f = env.eval("avszip_Butteraugli(r, r)").frame(0)
    assert f.props["Butteraugli"] == 0.0 and f.props["Butteraugli3Norm"] == 0.0


def test_errors(env):
    env.eval('rgb = BlankClip(pixel_type="RGBP", width=64, height=64, length=3)\n'
             'yv24 = BlankClip(pixel_type="YV24", width=64, height=64, length=3)\n'
             'small = BlankClip(pixel_type="RGBP", width=63, height=64, length=3)', want_clip=False)
    cases = [
        ("avszip_Butteraugli(yv24, yv24)", "only RGB"),
        ("avszip_Butteraugli(small, small)", "at least 64x64"),
        ("avszip_Butteraugli(rgb, rgb.Trim(0, 1))", "same length"),
        ("avszip_Butteraugli(rgb, rgb.ConvertBits(16))", "same bit depth"),
        ("avszip_Butteraugli(rgb, rgb, distmap=3)", "distmap"),
        ("avszip_Butteraugli(rgb, rgb, intensity_target=0.0)", "intensity_target"),
        ("avszip_Butteraugli(rgb, rgb, hf_asymmetry=1000.5)", "hf_asymmetry"),
        ("avszip_Butteraugli(rgb, rgb, xmul=-0.5)", "xmul"),
    ]
    for script, msg in cases:
        with pytest.raises(AvsError, match=msg):
            env.eval(script)
