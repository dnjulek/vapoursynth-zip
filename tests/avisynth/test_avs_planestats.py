"""PlaneAverage / PlaneMinMax: property-only metrics vs vszip."""

import pytest
import numpy as np

import harness as H
from avsrt import AvsError

FMTS = ["Y8", "YV12", "YUV444P10", "YUV420P16", "RGBP", "Y32", "YUV444PS", "RGBPS"]


def pair_with_ref(env, fmt):
    pair = H.Pair(env, fmt, H.make_frames(fmt, seed=1))
    ref = H.Pair(env, fmt, H.make_frames(fmt, seed=7))
    env.set_clip("ref", ref.avs)
    return pair, H.Var("ref", ref.vs)


def names(prop, *sfx):
    return [prop + s for s in sfx]


@pytest.mark.parametrize("fmt", FMTS)
@pytest.mark.parametrize("variant", ["plain", "exclude", "ref", "planes", "prop"])
def test_planeaverage(env, fmt, variant):
    pair, ref = pair_with_ref(env, fmt)
    f = H.fmt_info(fmt)
    args = {"exclude": [-1]}
    keys = ["psmAvg"]
    if variant == "exclude":
        args["exclude"] = [0, 128, 1023] if f.sample_type == 0 else [0, 1]
    elif variant == "ref":
        args["clipb"] = ref
        keys = ["psmAvg", "psmDiff"]
    elif variant == "planes":
        args["planes"] = list(range(f.num_planes))
    elif variant == "prop":
        args["prop"] = "zz"
        keys = ["zzAvg"]
    # Exact float equality: identical kernels on identical input.
    H.assert_same(env, pair, "PlaneAverage", args, props=keys)


@pytest.mark.parametrize("fmt", FMTS)
@pytest.mark.parametrize("variant", ["plain", "thr", "ref", "ref_thr", "planes", "prop"])
def test_planeminmax(env, fmt, variant):
    pair, ref = pair_with_ref(env, fmt)
    f = H.fmt_info(fmt)
    args = {}
    keys = ["psmMin", "psmMax"]
    if variant in ("thr", "ref_thr"):
        args.update(minthr=0.05, maxthr=0.1)
    if variant in ("ref", "ref_thr"):
        args["clipb"] = ref
        keys = ["psmMin", "psmMax", "psmDiff"]
    if variant == "planes":
        if f.color_family == 3 and f.sample_type == 1:  # YUV float: thresholds disallowed only
            pass
        args["planes"] = list(range(f.num_planes))
    if variant == "prop":
        args["prop"] = "q"
        keys = ["qMin", "qMax"]
    H.assert_same(env, pair, "PlaneMinMax", args, props=keys)


def test_pixels_untouched_and_source_props_isolated(env):
    pair = H.Pair(env, "YV12", H.make_frames("YV12"))
    env.set_clip("src", pair.avs)
    a = env.eval("src.avszip_PlaneAverage(exclude=[-1], planes=[0, 1, 2])")
    src = env.eval("src")
    for i in range(pair.avs.num_frames):
        x, s = a.frame(i), src.frame(i)
        H.compare_planes(x, s)
        assert len(x.props["psmAvg"]) == 3
        assert "psmAvg" not in s.props


def test_errors(env):
    env.eval('src = BlankClip(pixel_type="YV12", length=1)', want_clip=False)
    env.eval('src16 = BlankClip(pixel_type="YUV420P16", length=1)', want_clip=False)
    env.eval('srcf = BlankClip(pixel_type="YUV420PS", length=1)', want_clip=False)
    with pytest.raises(AvsError, match="exclude is required"):
        env.eval("src.avszip_PlaneAverage()")
    with pytest.raises(AvsError, match="same bit depth"):
        env.eval("src.avszip_PlaneAverage(exclude=[-1], clipb=src16)")
    with pytest.raises(AvsError, match="between 0.0 and 1.0"):
        env.eval("src.avszip_PlaneMinMax(maxthr=1.5)")
    with pytest.raises(AvsError, match="float chroma"):
        env.eval("srcf.avszip_PlaneMinMax(maxthr=0.1, planes=[0, 1])")
    with pytest.raises(AvsError, match="plane index out of range"):
        env.eval("src.avszip_PlaneMinMax(planes=[5])")
