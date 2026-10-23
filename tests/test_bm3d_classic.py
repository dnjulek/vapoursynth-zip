"""BM3DBasic / BM3DFinal / BM3DVBasic / BM3DVFinal / BM3DVAggregate / RGB2OPP /
OPP2RGB: VapourSynth-BM3D r12 (`bm3d.*`) on the CPU.

Contract (src/vapoursynth/bm3d_classic.zig, src/filters/bm3d_classic.zig,
src/filters/bm3d_fftw.zig):
- 8-16 bit integer and 32-bit float Gray/YUV/RGB; arguments, defaults
  (profiles fast/lc/np/high/vn), matrix/range handling and error texts as in
  bm3d.*; output keeps the input format; RGB goes through OPP and back.
- Output is bit-identical to bm3d r12's x86-64 build, including its FFTW
  DCTs, for every block size and every group size except the composites
  18, 21, 22, 24, 26, 27, 28 and 30 (and sizes above 32): FFTW_PATIENT picks
  their Cooley-Tukey plan by timing, so bm3d itself is not reproducible
  there (two runs of the same script differed at group_size 30); those
  sizes agree to rounding.
- BM3DVBasic/BM3DVFinal aggregate internally: their output equals
  bm3d.VBasic/VFinal -> bm3d.VAggregate (sample=1 for float, 0 for 16 bit),
  RGB: -> VAggregate(sample=1) -> OPP2RGB.
- BM3DVAggregate / RGB2OPP / OPP2RGB equal their bm3d counterparts.

Nets:
  1. metamorphic properties (flat input, plane passthrough, crop vs repack,
     request order, V* with radius 1 vs the stack it replaces);
  2. numpy oracles for VAggregate and the OPP conversions;
  3. optional parity with bm3d r12 when it is installed;
  4. golden snapshots.
"""

import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import vapoursynth as vs

from golden import Case
from helpers import assert_same_clip, repack

HAS_BM3D = hasattr(vs.core, "bm3d") and hasattr(vs.core.bm3d, "VAggregate")
if HAS_BM3D:
    _bm3d = [p for p in vs.core.plugins() if p.namespace == "bm3d"][0]
    HAS_BM3D = (_bm3d.version.major, _bm3d.version.minor) == (12, 0)


@pytest.fixture(scope="module")
def clip(make_temporal_clip, temporal_rgb):
    """3 frames, 128x80, a crop of the shared temporal clip."""

    def make(fmt):
        if vs.core.get_video_format(fmt).color_family == vs.RGB:
            c = temporal_rgb if fmt == vs.RGB24 else temporal_rgb.resize.Point(format=fmt)
        else:
            c = make_temporal_clip(fmt)
        return c.std.CropAbs(128, 80, 200, 100)

    return make


def frames(clip, ns=None):
    out = []
    for n in range(clip.num_frames) if ns is None else ns:
        with clip.get_frame(n) as f:
            out.append([np.asarray(f[p]).copy() for p in range(f.format.num_planes)])
    return out


CASES = [
    Case(vs.GRAY8, args={}),
    Case(vs.GRAY16, "odd", {}),
    Case(vs.GRAYS, args={"profile": "lc"}),
    Case(vs.GRAYS, args={"profile": "vn"}),
    Case(vs.YUV420P8, args={"sigma": [10, 0, 0]}),
    Case(vs.YUV444P16, args={"sigma": [8, 4, 4]}),
    Case(vs.RGB24, args={}),
    Case(vs.RGBS, args={}),
    Case(vs.GRAYS, args={"block_size": 6, "block_step": 2, "group_size": 12, "bm_step": 2}),
    Case(vs.GRAYS, args={"block_size": 11, "group_size": 13, "th_mse": 900.0}),
    Case(vs.GRAY16, args={"final": True}, variant="final"),
    Case(vs.RGB48, args={"final": True}, variant="final"),
    Case(vs.GRAY16, args={"temporal": True}, variant="v"),
    Case(vs.YUV444PS, args={"temporal": True, "radius": 2, "sigma": [8, 4, 4]}, variant="v"),
    Case(vs.RGB48, args={"temporal": True, "final": True}, variant="v-final"),
]


def _run(src, args):
    args = dict(args)
    temporal = args.pop("temporal", False)
    if args.pop("final", False):
        basic = src.vszip.BM3DVBasic() if temporal else src.vszip.BM3DBasic()
        return (src.vszip.BM3DVFinal if temporal else src.vszip.BM3DFinal)(basic, **args)
    return (src.vszip.BM3DVBasic if temporal else src.vszip.BM3DBasic)(**args)


@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_cases(golden, clip, case):
    src = clip(case.fmt)
    if case.geometry == "odd":
        src = src.std.Crop(right=1 << src.format.subsampling_w, bottom=1 << src.format.subsampling_h)
    golden.check("bm3d_classic", case, _run(src, case.args), n=2)


# --- metamorphic -------------------------------------------------------------


@pytest.mark.parametrize("fn", ["BM3DBasic", "BM3DVBasic"])
def test_flat_input_stays_flat(core, fn):
    src = core.std.BlankClip(None, 64, 64, vs.GRAYS, length=3, color=0.4)
    for f in frames(getattr(src.vszip, fn)()):
        assert np.abs(f[0] - 0.4).max() < 2e-6


def test_unprocessed_planes_are_copied(clip):
    src = clip(vs.YUV420P16)
    out = frames(src.vszip.BM3DBasic(sigma=[10, 0, 0]))
    for fo, fs in zip(out, frames(src)):
        assert not np.array_equal(fo[0], fs[0])
        assert np.array_equal(fo[1], fs[1]) and np.array_equal(fo[2], fs[2])


def test_all_zero_sigma(clip):
    src = clip(vs.YUV444P8)
    assert_same_clip(src.vszip.BM3DBasic(sigma=0.0), src)
    with pytest.raises(vs.Error, match="no plane of the input clip would be processed"):
        src.vszip.BM3DVBasic(sigma=0.0)


def test_crop_matches_repacked_copy(clip):
    src = clip(vs.GRAY8).std.Crop(left=3, right=5, top=2, bottom=1)
    for fn in ("BM3DBasic", "BM3DVBasic"):
        assert_same_clip(getattr(src.vszip, fn)(), getattr(repack(src).vszip, fn)())


def test_temporal_request_order_and_threads(clip):
    src = clip(vs.GRAYS)
    src = src + src.std.FlipVertical() + src
    want = frames(src.vszip.BM3DVBasic(radius=2))
    order = list(range(src.num_frames))
    random.Random(3).shuffle(order)
    got = dict(zip(order, frames(src.vszip.BM3DVBasic(radius=2), order)))
    out = src.vszip.BM3DVBasic(radius=2)
    with ThreadPoolExecutor(8) as ex:
        conc = list(ex.map(lambda n: frames(out, [n])[0], [*order, *order]))
    for n in range(src.num_frames):
        assert np.array_equal(got[n][0], want[n][0])
    for n, f in zip([*order, *order], conc):
        assert np.array_equal(f[0], want[n][0])


def test_range_props(clip):
    # untagged Gray/YUV is limited range, _Range=0 too; _Range=1 is full
    src = clip(vs.GRAY8).std.RemoveFrameProps("_Range")
    untagged = frames(src.vszip.BM3DBasic())
    limited = frames(src.std.SetFrameProps(_Range=0).vszip.BM3DBasic())
    full = frames(src.std.SetFrameProps(_Range=1).vszip.BM3DBasic())
    assert all(np.array_equal(a[0], b[0]) for a, b in zip(untagged, limited))
    assert any(not np.array_equal(a[0], b[0]) for a, b in zip(untagged, full))


def test_opp_input_needs_matrix_100(clip):
    src = clip(vs.YUV444PS).std.SetFrameProps(BM3D_OPP=1)
    with pytest.raises(vs.Error, match=r"matrix=100"):
        src.vszip.BM3DBasic().get_frame(0)
    src.vszip.BM3DBasic(matrix=100).get_frame(0)


# --- OPP / VAggregate oracles ---------------------------------------------------


def test_rgb2opp_float_oracle(clip):
    src = clip(vs.RGBS)
    opp = frames(src.vszip.RGB2OPP(sample=1))
    for fo, fs in zip(opp, frames(src)):
        r, g, b = fs
        f = np.float32
        np.testing.assert_array_equal(fo[0], ((r + g) + b) * (f(1) / f(3)) + f(0))
        np.testing.assert_array_equal(fo[1], (r - b) * (f(1) / f(2)) + f(0))
        np.testing.assert_array_equal(fo[2], ((r - g * f(2)) + b) * (f(1) / f(4)) + f(0))


def test_opp_round_trip(clip):
    src = clip(vs.RGBS)
    back = frames(src.vszip.RGB2OPP(sample=1).vszip.OPP2RGB(sample=1))
    for fb, fs in zip(back, frames(src)):
        for p in range(3):
            assert np.abs(fb[p] - fs[p]).max() < 2e-6


def test_vaggregate_oracle(core):
    r, n_fr, w, h = 2, 5, 67, 64
    rng = np.random.default_rng(9)
    data = rng.random((n_fr, 2 * (2 * r + 1), h, w), dtype=np.float32) + np.float32(0.25)
    base = core.std.BlankClip(None, w, h * 2 * (2 * r + 1), vs.GRAYS, length=n_fr)

    def fill(n, f):
        fo = f.copy()
        np.asarray(fo[0])[:] = data[n].reshape(-1, w)
        fo.props["BM3D_V_radius"] = r
        fo.props["BM3D_V_process"] = [1, 1, 1]
        return fo

    stacked = base.std.ModifyFrame(base, fill)
    got = frames(core.vszip.BM3DVAggregate(stacked, sample=1))
    for n in range(n_fr):
        num = np.zeros((h, w), np.float32)
        den = np.zeros((h, w), np.float32)
        b, f = -min(n, r), min(n_fr - 1 - n, r)
        for i, k in enumerate(range(b, f + 1)):
            # stage n + k holds frame n in its slot r - k, oldest first
            num += data[n + k, 2 * (r - k)]
            den += data[n + k, 2 * (r - k) + 1]
        assert np.array_equal(got[n][0], num / den), f"frame {n}"


# --- validation --------------------------------------------------------------


@pytest.mark.parametrize(
    ("fn", "args", "msg"),
    [
        ("BM3DBasic", dict(profile="max"), "Unrecognized \"profile\""),
        ("BM3DBasic", dict(sigma=-1.0), "Invalid \"sigma\""),
        ("BM3DBasic", dict(block_size=0), r"block_size\" assigned, must be an integer in \[1, 64\]"),
        ("BM3DBasic", dict(block_size=65), r"\[1, 64\]"),
        ("BM3DBasic", dict(block_size=16, block_step=17), r"block_step\" assigned"),
        ("BM3DBasic", dict(group_size=257), r"group_size\" assigned"),
        ("BM3DBasic", dict(bm_range=0), "bm_range\" assigned"),
        ("BM3DBasic", dict(bm_step=10), "bm_step\" assigned"),
        ("BM3DBasic", dict(th_mse=-1.0), "th_mse\" assigned"),
        ("BM3DBasic", dict(hard_thr=0.0), "hard_thr\" assigned"),
        ("BM3DBasic", dict(matrix=3), "Unsupported \"matrix\""),
        ("BM3DVBasic", dict(radius=0), r"radius\" assigned, must be an integer in \[1, 16\]"),
        ("BM3DVBasic", dict(radius=17), "radius\" assigned"),
        ("BM3DVBasic", dict(ps_num=9), r"ps_num\" assigned"),
        ("BM3DVBasic", dict(ps_range=0), "ps_range\" assigned"),
        ("BM3DVBasic", dict(ps_range=2, ps_step=3), "ps_step\" assigned"),
    ],
)
def test_validation_errors(core, fn, args, msg):
    src = core.std.BlankClip(None, 64, 64, vs.GRAY8, length=1)
    with pytest.raises(vs.Error, match=f"{fn}: .*{msg}"):
        getattr(src.vszip, fn)(**args)


def test_format_errors(core):
    with pytest.raises(vs.Error, match="sub-sampled format is not supported"):
        core.std.BlankClip(None, 64, 64, vs.YUV420P8, length=1).vszip.BM3DBasic(sigma=[10, 5, 5])
    with pytest.raises(vs.Error, match="only 8-16 bit integer or 32 bit float"):
        core.std.BlankClip(None, 64, 64, vs.GRAYH, length=1).vszip.BM3DBasic()
    for w, h in ((64, 63), (63, 64)):
        with pytest.raises(vs.Error, match="at least 64x64"):
            core.std.BlankClip(None, w, h, vs.GRAY8, length=1).vszip.BM3DBasic()
    with pytest.raises(vs.Error, match="must be of the same format"):
        core.std.BlankClip(None, 64, 64, vs.GRAY8, length=1).vszip.BM3DFinal(core.std.BlankClip(None, 64, 64, vs.GRAY16, length=1))
    with pytest.raises(vs.Error, match="must be of RGB color family"):
        core.std.BlankClip(None, 64, 64, vs.GRAY8, length=1).vszip.RGB2OPP()
    with pytest.raises(vs.Error, match="BM3D_OPP=1\\\" is missing"):
        core.std.BlankClip(None, 64, 64, vs.YUV444P8, length=1).vszip.OPP2RGB().get_frame(0)


# --- optional: parity with bm3d r12 ----------------------------------------------

# group_size <= 17: every group size of these runs has an exact FFTW plan
PARITY = [
    (vs.GRAY8, {}),
    (vs.GRAY16, {"profile": "lc"}),
    (vs.GRAYS, {"profile": "np", "group_size": 16}),
    (vs.GRAYS, {"profile": "high", "group_size": 17}),
    (vs.YUV420P8, {"sigma": [10, 0, 0]}),
    (vs.YUV444P16, {"sigma": [8, 4, 4]}),
    (vs.YUV444PS, {"sigma": [0, 6, 6]}),
    (vs.RGB24, {}),
    (vs.RGBS, {"sigma": [10, 0, 6]}),
    (vs.GRAYS, {"block_size": 6, "block_step": 2, "group_size": 12, "bm_step": 2}),
    (vs.GRAYS, {"block_size": 11, "group_size": 17, "th_mse": 900.0}),
    (vs.GRAYS, {"group_size": 17, "matrix": 9}),
]


@pytest.mark.skipif(not HAS_BM3D, reason="bm3d r12 not installed")
@pytest.mark.parametrize(("fmt", "kw"), PARITY)
@pytest.mark.parametrize("fn", ["Basic", "Final"])
def test_matches_bm3d(clip, fmt, kw, fn):
    src = clip(fmt)
    if fn == "Final":
        ref = src.bm3d.Basic(**kw)
        a, b = src.vszip.BM3DFinal(ref, **kw), src.bm3d.Final(ref, **kw)
    else:
        a, b = src.vszip.BM3DBasic(**kw), src.bm3d.Basic(**kw)
    assert_same_clip(a, b)


@pytest.mark.skipif(not HAS_BM3D, reason="bm3d r12 not installed")
@pytest.mark.parametrize(
    ("fmt", "kw"),
    [(vs.GRAY16, {}), (vs.GRAYS, {"profile": "lc"}), (vs.YUV444PS, {"radius": 2, "ps_num": 3}), (vs.RGB48, {})],
)
@pytest.mark.parametrize("fn", ["VBasic", "VFinal"])
def test_v_matches_bm3d_stack(clip, fmt, kw, fn):
    src = clip(fmt)
    rgb = src.format.color_family == vs.RGB
    sample = 1 if (rgb or src.format.sample_type == vs.FLOAT) else 0
    out_sample = 1 if src.format.sample_type == vs.FLOAT else 0

    def agg(st):
        a = st.bm3d.VAggregate(sample=sample)
        return a.bm3d.OPP2RGB(sample=out_sample) if rgb else a

    if fn == "VFinal":
        ref = agg(src.bm3d.VBasic(**kw))
        a, b = src.vszip.BM3DVFinal(ref, **kw), agg(src.bm3d.VFinal(ref, **kw))
    else:
        a, b = src.vszip.BM3DVBasic(**kw), agg(src.bm3d.VBasic(**kw))
    for fa, fb in zip(frames(a), frames(b)):
        for pa, pb in zip(fa, fb):
            assert np.array_equal(pa, pb)


@pytest.mark.skipif(not HAS_BM3D, reason="bm3d r12 not installed")
@pytest.mark.parametrize("profile", ["np", "vn"])
@pytest.mark.parametrize("fn", ["Basic", "Final"])
def test_group32_is_close_to_bm3d(clip, profile, fn):
    """group sizes 18..30 (composite): FFTW_PATIENT picks their plan by
    timing, bm3d is not even reproducible against itself; the port agrees to
    rounding (a flipped threshold decision moves a group by ~1e-4)."""
    src = clip(vs.GRAYS)
    if fn == "Final":
        ref = src.bm3d.Basic(profile=profile)
        a, b = src.vszip.BM3DFinal(ref, profile=profile), src.bm3d.Final(ref, profile=profile)
    else:
        a, b = src.vszip.BM3DBasic(profile=profile), src.bm3d.Basic(profile=profile)
    for fa, fb in zip(frames(a), frames(b)):
        assert np.abs(fa[0] - fb[0]).max() < 2e-3


@pytest.mark.skipif(not HAS_BM3D, reason="bm3d r12 not installed")
def test_misc_match_bm3d(clip):
    st = clip(vs.YUV444PS).bm3d.VBasic(radius=1, sigma=[8, 4, 4])
    for sample in (0, 1):
        assert_same_clip(st.vszip.BM3DVAggregate(sample=sample), st.bm3d.VAggregate(sample=sample))
    for fmt in (vs.RGB24, vs.RGB48, vs.RGBS):
        src = clip(fmt)
        for sample in (0, 1):
            opp = src.bm3d.RGB2OPP(sample=sample)
            assert_same_clip(src.vszip.RGB2OPP(sample=sample), opp)
            for s2 in (0, 1):
                assert_same_clip(opp.vszip.OPP2RGB(sample=s2), opp.bm3d.OPP2RGB(sample=s2))
