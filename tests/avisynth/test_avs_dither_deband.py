"""Dither and Deband vs vszip."""

import pytest

import harness as H
from avsrt import AvsError


def same(avs_clip, vs_clip, tol=0.0, props=()):
    assert avs_clip.num_frames == vs_clip.num_frames
    for i, (x, y) in enumerate(zip(avs_clip.frames(), H.vs_frames(vs_clip))):
        H.compare_planes(x, y, tol, what=f"frame {i}")
        H.compare_props(x, y, props, what=f"frame {i}")


@pytest.mark.parametrize("mode", list(range(12)))
@pytest.mark.parametrize("conv", [("YUV420P16", 8), ("YUV444P10", 8), ("Y16", 10), ("RGBP16", 8), ("YUV420PS", 8), ("Y32", 16)])
def test_dither_down(env, mode, conv):
    fmt, bits = conv
    pair = H.Pair(env, fmt, H.make_frames(fmt, n=3))
    env.set_clip("src", pair.avs)
    a = env.eval(f"src.vszip_Dither({bits}, dither_type={mode})")
    v = H.vszip().Dither(pair.vs, bits, dither_type=mode)
    same(a, v)


@pytest.mark.parametrize("conv", [("YV12", 16, {}), ("YV12", 10, {}), ("Y10", 16, {}), ("YV24", 16, {"fulls": True, "fulld": True}),
                                  ("YV12", 32, {}), ("YUV420P16", 32, {}), ("RGBP", 16, {}), ("YV12", 16, {"fulls": True, "fulld": True, "dither_type": 0}),
                                  ("YUV444P16", 8, {"fulls": False, "fulld": True}), ("YUV420P10", 8, {"fulls": True})])
def test_dither_other(env, conv):
    fmt, bits, extra = conv
    pair = H.Pair(env, fmt, H.make_frames(fmt, n=2))
    env.set_clip("src", pair.avs)
    a = env.eval("src." + H.avs_call("Dither", extra).replace("vszip_Dither(", f"vszip_Dither({bits}" + (", " if extra else "")))
    vs_extra = {k: int(v) if isinstance(v, bool) else v for k, v in extra.items()}
    v = H.vszip().Dither(pair.vs, bits, **vs_extra)
    props = ["_ColorRange"] if ("fulls" in extra or "fulld" in extra) else []
    same(a, v, props=props)


def test_dither_errors(env):
    env.eval('src = BlankClip(pixel_type="YUV420P16", length=1)', want_clip=False)
    with pytest.raises(AvsError, match="16-bit float"):
        env.eval("src.vszip_Dither(16, sample_type=1)")
    with pytest.raises(AvsError, match="8, 10, 12, 14 or 16"):
        env.eval("src.vszip_Dither(9)")
    with pytest.raises(AvsError, match="dither_type must be 0-11"):
        env.eval("src.vszip_Dither(8, dither_type=12)")


DB_FMTS = ["YUV420P16", "YUV444P16", "Y16", "YUV420PS", "Y32", "RGBP16", "YV12", "YUV420P10", "Y8"]


@pytest.mark.parametrize("fmt", DB_FMTS)
@pytest.mark.parametrize("args", [
    {},
    {"range": 8, "thr": [2.0, 1.5], "grain": [0.0]},
    {"grain": [3.0, 1.0], "seed": 7},
    {"grain": [2.0], "dynamic_grain": True},
    {"sample_mode": 1}, {"sample_mode": 3}, {"sample_mode": 4}, {"sample_mode": 5},
    {"sample_mode": 6, "thr1": [3.0], "thr2": [0.5]},
    {"sample_mode": 7, "angle_boost": 2.0, "max_angle": 0.3},
    {"blur_first": False, "keep_tv_range": True},
    {"random_algo_ref": 2, "random_param_ref": 1.5, "random_algo_grain": 0, "grain": [1.0]},
])
def test_deband(env, fmt, args):
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=96, h=64, n=3, kind="smooth"))
    a, v = H.run_both(env, pair, "Deband", args)
    same(a, v)


@pytest.mark.parametrize("full", [False, True])
def test_deband_low_bits_tagged_range(env, full):
    fmt = "YV12"
    pair = H.Pair(env, fmt, H.make_frames(fmt, n=2, kind="smooth"), full_range=full)
    a, v = H.run_both(env, pair, "Deband", {"grain": [1.0]})
    same(a, v, props=["_ColorRange"])


def test_deband_cropped_input(env):
    fmt = "YUV444P16"
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=100, h=48, n=2, kind="smooth"))
    env.set_clip("src", pair.avs)
    a = env.eval("src.Crop(6, 0, 80, 0).vszip_Deband(grain=[1.0])")
    v = H.vszip().Deband(H.core.std.Crop(pair.vs, 6, 14), grain=[1.0])
    same(a, v)
