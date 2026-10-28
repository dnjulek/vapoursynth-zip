"""BM3D / BM3Dv2 / VAggregate vs vszip: bit-exact across formats, parameters,
the fused temporal mode (any request order), the final estimate, chroma
(CBM3D), cropped sources, alpha passthrough and VAggregate."""

import random

import numpy as np
import pytest

import harness as H
from avsrt import AvsError


def same(avs_clip, vs_clip, order=None, planes=None):
    assert avs_clip.num_frames == vs_clip.num_frames
    ns = list(range(avs_clip.num_frames)) if order is None else order
    for n, x, y in zip(ns, [avs_clip.frame(n) for n in ns], H.vs_frames(vs_clip, ns)):
        H.compare_planes(x, y, 0.0, what=f"frame {n}", planes=planes)


@pytest.mark.parametrize("fmt", ["Y32", "YUV420PS", "YUV444PS", "RGBPS"])
@pytest.mark.parametrize("args", [
    {},
    {"sigma": [5.0, 2.0]},
    {"sigma": [3.0, 0.0]},
    {"block_step": [4, 6], "bm_range": [4, 12]},
    {"extractor_exp": 6},
    {"bm_error_s": "ssd/norm", "transform_2d_s": "haar", "transform_1d_s": ["wht", "bior1.5"]},
    {"bm_error_s": ["sad", "zssd"]},
])
def test_spatial(env, fmt, args):
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=72, h=64, n=2))
    a, v = H.run_both(env, pair, "BM3D", args)
    same(a, v)


@pytest.mark.parametrize("fmt", ["Y32", "YUV420PS"])
@pytest.mark.parametrize("args", [
    {"radius": 1},
    {"radius": 2, "ps_num": 3, "ps_range": 2},
    {"radius": 1, "extractor_exp": 8, "block_step": 6},
])
def test_temporal_any_order(env, fmt, args):
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=64, h=64, n=6))
    a, v = H.run_both(env, pair, "BM3D", args)
    same(a, v)
    order = list(range(6))
    random.Random(2).shuffle(order)
    same(a, v, order=order + [5, 0, 3, 3])


@pytest.mark.parametrize("radius", [0, 1])
def test_final_estimate(env, radius):
    fmt = "YUV444PS"
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=64, h=64, n=4, seed=1))
    ref = H.Pair(env, fmt, H.make_frames(fmt, w=64, h=64, n=4, seed=2, kind="smooth"))
    env.set_clip("r", ref.avs)
    a, v = H.run_both(env, pair, "BM3D", {"ref": H.Var("r", ref.vs), "radius": radius})
    same(a, v)


@pytest.mark.parametrize("radius", [0, 1])
def test_chroma(env, radius):
    fmt = "YUV444PS"
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=64, h=64, n=3))
    a, v = H.run_both(env, pair, "BM3D", {"chroma": True, "sigma": [3.0, 2.0, 0.0], "radius": radius})
    same(a, v)


def test_bm3dv2_matches_bm3d_and_passes_through(env):
    fmt = "Y32"
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=64, h=64, n=4))
    a, v = H.run_both(env, pair, "BM3Dv2", {"radius": 1})
    same(a, v)
    env.set_clip("src", pair.avs)
    same(env.eval("src.avszip_BM3Dv2(sigma=0.0)"), pair.vs)


def test_cropped_source(env):
    fmt = "YUV420PS"
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=160, h=80, n=3))
    env.set_clip("src", pair.avs)
    a = env.eval("src.Crop(10, 2, -54, -6).avszip_BM3D(radius=1)")
    v = H.vszip().BM3D(H.core.std.Crop(pair.vs, 10, 54, 2, 6), radius=1)
    same(a, v)


def test_alpha_passes_through(env):
    fmt = "YUV420PS"
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=64, h=64, n=2))
    env.set_clip("src", pair.avs)
    a = env.eval("src.AddAlphaPlane(0.25).avszip_BM3D(radius=1)")
    v = H.vszip().BM3D(pair.vs, radius=1)
    same(a, v, planes=[0, 1, 2])
    for n in range(a.num_frames):
        assert (a.frame(n).planes[3] == np.float32(0.25)).all()


@pytest.mark.parametrize("radius", [1, 2])
def test_vaggregate(env, radius):
    fmt = "YUV420PS"
    w, h, n = 64, 64, 5
    src = H.Pair(env, fmt, H.make_frames(fmt, w=w, h=h, n=n, seed=3))
    # a synthetic stacked clip: positive weights, any numerators
    stacked = H.make_frames(fmt, w=w, h=h * 2 * (2 * radius + 1), n=n, seed=4, kind="noise")
    for frame in stacked:
        for p, plane in enumerate(frame):
            ph = plane.shape[0] // (2 * (2 * radius + 1))
            for z in range(2 * radius + 1):
                plane[(2 * z + 1) * ph:(2 * z + 2) * ph] = np.abs(plane[(2 * z + 1) * ph:(2 * z + 2) * ph]) + 0.5
    st = H.Pair(env, fmt, stacked)
    env.set_clip("s", src.avs)
    env.set_clip("st", st.avs)
    for planes in ([0], [0, 1, 2], [2]):
        a = env.eval(f"avszip_VAggregate(st, s, planes={planes})")
        v = H.vszip().VAggregate(st.vs, src.vs, planes=planes)
        same(a, v)


def test_vaggregate_ignores_alpha_presence(env):
    """Only color planes are aggregated and the output alpha is the source's,
    so either input may carry alpha without the other."""
    fmt = "YUV420PS"
    src = H.Pair(env, fmt, H.make_frames(fmt, w=64, h=64, n=3, seed=3))
    env.set_clip("s", src.avs)
    env.eval("st = s.StackVertical(s, s, s, s, s).Expr(\"1\")", want_clip=False)
    want = env.eval("avszip_VAggregate(st, s, planes=[0, 1, 2])")
    for call, alpha in (("avszip_VAggregate(st, s.AddAlphaPlane(0.25), planes=[0, 1, 2])", 0.25),
                        ("avszip_VAggregate(st.AddAlphaPlane(0.5), s, planes=[0, 1, 2])", None),
                        ("avszip_VAggregate(st.AddAlphaPlane(0.5), s.AddAlphaPlane(0.25), planes=[0, 1, 2])", 0.25)):
        got = env.eval(call)
        for n in range(3):
            g, w = got.frame(n), want.frame(n)
            H.compare_planes(g, w, planes=[0, 1, 2], what=call)
            assert len(g.planes) == (4 if alpha is not None else 3), call
            if alpha is not None:
                assert (g.planes[3] == np.float32(alpha)).all(), call


@pytest.mark.parametrize("call", [
    "{c}.avszip_BM3D(radius=1)",
    "{c}.avszip_BM3D(radius=2)",
    "{c}.avszip_BM3Dv2(radius=1, ref={c})",
    "avszip_VAggregate({c}.StackVertical({c}, {c}, {c}, {c}, {c}, {c}, {c}, {c}, {c}), {c}, planes=[0])",
    "{c}.avszip_WNNM(radius=1, block_size=4)",
])
def test_last_frames_of_the_longest_clip(call):
    """Frame windows near frame 2^31 - 1 are summed without i32 overflow (a
    debug build panicked, a release build read past the frame list); the
    edge frames match a short clip's."""
    H.assert_long_clip_edges(call)


@pytest.mark.parametrize(("call", "msg"), [
    ("i8.avszip_BM3D()", "32 bit float"),
    ("small.avszip_BM3D()", "at least 64x64"),
    ("g.avszip_BM3D(radius=17)", "radius value 17 is out of range"),
    ("g.avszip_BM3D(block_step=9)", "block_step"),
    ("g.avszip_BM3D(ps_num=0)", "ps_num"),
    ("g.avszip_BM3D(sigma=[-1.0])", "sigma"),
    ("g.avszip_BM3D(sigma=0.0)", "nothing to process"),
    ("g.avszip_BM3D(chroma=true)", "YUV444"),
    ("y420.avszip_BM3D(chroma=true)", "YUV444"),
    ("g.avszip_BM3D(extractor_exp=128)", "extractor_exp"),
    ("g.avszip_BM3D(bm_error_s=\"nope\")", "invalid \"bm_error_s\""),
    ("g.avszip_BM3D(transform_2d_s=[\"dct\", \"dct\", \"dct\", \"dct\"])", "too many elements"),
    ("g.avszip_BM3D(ref=y420)", "same format"),
    ("g.avszip_BM3D(ref=g.Trim(0, 1))", "same number of frames"),
    ("avszip_VAggregate(g, g, planes=[0])", "not a BM3D stacked clip"),
    ("avszip_VAggregate(g.StackVertical(g, g, g, g, g), g)", "planes\" is required"),
    ("avszip_VAggregate(g.StackVertical(g, g, g, g, g), g, planes=[1])", "out-of-range plane index"),
])
def test_errors(env, call, msg):
    env.eval('g = BlankClip(pixel_type="Y32", length=3, width=64, height=64)\n'
             'small = BlankClip(pixel_type="Y32", length=3, width=48, height=64)\n'
             'i8 = BlankClip(pixel_type="Y8", length=3, width=64, height=64)\n'
             'y420 = BlankClip(pixel_type="YUV420PS", length=3, width=64, height=64)', want_clip=False)
    with pytest.raises(AvsError, match=msg) as e:
        env.eval(call)
    assert str(e.value).startswith(("avszip_BM3D:", "avszip_VAggregate:"))
