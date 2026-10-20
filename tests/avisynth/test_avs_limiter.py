import pytest

import harness as H
from avsrt import AvsError

FMTS = H.ALL


@pytest.mark.parametrize("fmt", FMTS)
@pytest.mark.parametrize("args", [
    {},
    {"tv_range": True},
    {"tv_range": True, "mask": True},
    {"tv_range": True, "planes": [0]},
    {"tv_range": True, "planes": [1, 2]},
])
def test_limiter_matches_vszip(env, fmt, args):
    if "planes" in args and fmt.startswith("Y") and not fmt.startswith("YUV") and not fmt.startswith("YV"):
        args = {k: v for k, v in args.items() if k != "planes"}
    pair = H.Pair(env, fmt, H.make_frames(fmt))
    H.assert_same(env, pair, "Limiter", args)


@pytest.mark.parametrize("fmt", ["YV12", "YUV444P10", "YUV420PS", "RGBP16", "Y8"])
def test_limiter_runtime_bounds(env, fmt):
    f = H.fmt_info(fmt)
    n = f.num_planes
    if f.sample_type == 1:
        lo, hi = [0.1, -0.2, -0.3][:n], [0.8, 0.25, 0.1][:n]
    else:
        peak = (1 << f.bits_per_sample) - 1
        lo, hi = [peak * 0.1, peak * 0.2, peak * 0.3][:n], [peak * 0.8, peak * 0.7, peak * 0.6][:n]
        lo, hi = [float(int(x)) for x in lo], [float(int(x)) for x in hi]
    pair = H.Pair(env, fmt, H.make_frames(fmt))
    H.assert_same(env, pair, "Limiter", {"min": lo, "max": hi})


def test_limiter_cropped_input(env):
    # Cropped source keeps its parent pitch: exercises the per-row path.
    pair = H.Pair(env, "YUV420P16", H.make_frames("YUV420P16", w=128, h=64))
    env.set_clip("src", pair.avs)
    a = env.eval("src.Crop(16, 2, -32, -4).vszip_Limiter(tv_range=true)")
    b = H.vszip().Limiter(H.core.std.Crop(pair.vs, 16, 32, 2, 4), tv_range=1)
    for x, y in zip(a.frames(), H.vs_frames(b)):
        H.compare_planes(x, y)


@pytest.mark.parametrize("args,msg", [
    ("min=[1, 2]", "same number of elements as planes"),
    ("min=[1, 2, 3]", "max array is not"),
    ("max=[1, 2, 3]", "min array is not"),
    ("min=[10, 10, 10], max=[5, 5, 5]", "less than or equal to max"),
    ("min=[0, 0, 0], max=[300, 300, 300]", "less than or equal to peak"),
    ("planes=[0, 0]", "plane specified twice"),
    ("planes=[3]", "plane index out of range"),
])
def test_limiter_errors(env, args, msg):
    env.eval('src = BlankClip(pixel_type="YV12", length=1)', want_clip=False)
    with pytest.raises(AvsError, match=msg):
        env.eval(f"src.vszip_Limiter({args})")
