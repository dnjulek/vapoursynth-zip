"""BM3D / BM3Dv2 / VAggregate: the CUDA-output BM3D (vszipcu's API) on the CPU.

Contract (src/vapoursynth/bm3d.zig, src/filters/bm3d_cuda.zig):
- 32-bit float only; output keeps the input format, size, length and props;
  planes with sigma < FLT_EPSILON are copied from the source.
- sigma/block_step/bm_range/ps_num/ps_range/bm_error_s/transform_*_s are
  per-plane arrays (the last given value repeats); chroma=True (CBM3D)
  needs YUV444 and matches on plane 0.
- radius > 0 aggregates internally (BM3D == BM3Dv2, no stacked clip); the
  temporal window is vszipcu's (frames clamped at the clip ends).
- BM3Dv2 passes a clip with nothing to process through; BM3D rejects it.
- Arithmetic follows vszipcu's sm_86 code (fma points, MUFU.RCP emulation);
  with extractor_exp set the GPU is deterministic and the CPU matches it
  bit for bit, except where ptxas fuses an inverse transform differently in
  context (observed: at most one pixel per frame, in some configurations).
- VAggregate aggregates stacked clips made by the GPU plugins.

Nets:
  1. metamorphic properties (flat input, sigma-0 plane passthrough, crop vs
     repack, request order and thread count, BM3D == BM3Dv2);
  2. a numpy oracle for VAggregate;
  3. optional parity with vszipcu (the CUDA plugin) when it is installed;
  4. golden snapshots.
"""

import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import vapoursynth as vs

from golden import Case
from helpers import assert_same_clip, repack

HAS_CU = hasattr(vs.core, "vszipcu")


@pytest.fixture(scope="module")
def clip(make_temporal_clip):
    """3 frames, 160x96, a crop of the shared temporal clip."""

    def make(fmt):
        return make_temporal_clip(fmt).std.CropAbs(160, 96, 200, 100)

    return make


def frames(clip, ns=None):
    out = []
    for n in range(clip.num_frames) if ns is None else ns:
        with clip.get_frame(n) as f:
            out.append([np.asarray(f[p]).copy() for p in range(f.format.num_planes)])
    return out


CASES = [
    Case(vs.GRAYS, args={}),
    Case(vs.GRAYS, "odd", {}),
    Case(vs.YUV420PS, args={"sigma": [3.0, 2.0]}),
    Case(vs.YUV444PS, args={"chroma": True, "sigma": [3.0, 2.0, 0.0]}),
    Case(vs.GRAYS, args={"radius": 1}),
    Case(vs.GRAYS, "odd", {"radius": 2, "ps_num": 3, "ps_range": 3}),
    Case(vs.GRAYS, args={"block_step": 3, "bm_range": 5}),
    Case(vs.GRAYS, args={"extractor_exp": 6}),
    Case(vs.GRAYS, args={"transform_2d_s": "haar", "transform_1d_s": "bior1.5", "radius": 1}),
    Case(vs.GRAYS, args={"bm_error_s": "zsad"}),
    Case(vs.GRAYS, args={"bm_error_s": "ssd/norm"}),
    Case(vs.GRAYS, args={"transform_2d_s": "wht"}),
    Case(vs.GRAYS, args={"final": True}, variant="final"),
    Case(vs.GRAYS, args={"final": True, "radius": 1}, variant="final"),
]


def _run(clip, args):
    args = dict(args)
    if args.pop("final", False):
        args["ref"] = clip.vszip.BM3D(radius=args.get("radius", 0))
    return clip.vszip.BM3D(**args)


@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_cases(golden, clip, case):
    src = clip(case.fmt)
    if case.geometry == "odd":
        src = src.std.Crop(right=1 << src.format.subsampling_w, bottom=1 << src.format.subsampling_h)
    # the last frame: its window is clamped
    golden.check("bm3d", case, _run(src, case.args), n=2)


# --- metamorphic -------------------------------------------------------------


@pytest.mark.parametrize("radius", [0, 1])
@pytest.mark.parametrize("final", [False, True])
def test_flat_input_stays_flat(core, radius, final):
    # every block is the same: the 3D spectrum is DC only (kept), every
    # group returns the block, and the weighted average is the value again
    # up to the transforms' rounding
    src = core.std.BlankClip(None, 64, 48, vs.GRAYS, length=3, color=0.4)
    kw = {"ref": src} if final else {}
    for f in frames(src.vszip.BM3D(radius=radius, **kw)):
        assert np.abs(f[0] - 0.4).max() < 2e-6


def test_zero_sigma_plane_is_copied(clip):
    src = clip(vs.YUV444PS)
    for chroma in (False, True):
        out = src.vszip.BM3D(sigma=[3.0, 0.0, 2.0], chroma=chroma)
        a, b = frames(out), frames(src)
        for fa, fb in zip(a, b):
            assert np.array_equal(fa[1], fb[1])
            assert not np.array_equal(fa[0], fb[0])
            assert not np.array_equal(fa[2], fb[2])


def test_per_plane_arguments_are_not_broadcast(clip):
    src = clip(vs.YUV444PS)
    base = frames(src.vszip.BM3D(sigma=[3.0, 3.0, 3.0]))
    alt = frames(src.vszip.BM3D(sigma=[3.0, 3.0, 3.0], block_step=[8, 8, 4]))
    for fb, fa in zip(base, alt):
        assert np.array_equal(fb[0], fa[0]) and np.array_equal(fb[1], fa[1])
        assert not np.array_equal(fb[2], fa[2])


def test_crop_matches_repacked_copy(clip):
    src = clip(vs.GRAYS).std.Crop(left=3, right=5, top=2, bottom=1)
    for kw in ({}, {"radius": 1}):
        assert_same_clip(src.vszip.BM3D(**kw), repack(src).vszip.BM3D(**kw))


@pytest.mark.parametrize("radius", [1, 2])
def test_request_order_and_threads_do_not_matter(core, clip, radius):
    src = clip(vs.GRAYS)
    src = src + src.std.FlipVertical() + src
    want = frames(src.vszip.BM3D(radius=radius))
    order = list(range(src.num_frames))
    random.Random(radius).shuffle(order)
    got = dict(zip(order, frames(src.vszip.BM3D(radius=radius), order)))
    out = src.vszip.BM3D(radius=radius)
    with ThreadPoolExecutor(8) as ex:
        conc = list(ex.map(lambda n: frames(out, [n])[0], [*order, *order]))
    for n in range(src.num_frames):
        assert np.array_equal(got[n][0], want[n][0])
    for n, f in zip([*order, *order], conc):
        assert np.array_equal(f[0], want[n][0])


def test_bm3dv2_is_bm3d(clip):
    src = clip(vs.YUV420PS)
    for kw in ({}, {"radius": 1}):
        assert_same_clip(src.vszip.BM3Dv2(**kw), src.vszip.BM3D(**kw))


def test_all_zero_sigma(clip):
    src = clip(vs.GRAYS)
    assert_same_clip(src.vszip.BM3Dv2(sigma=0.0), src)
    with pytest.raises(vs.Error, match="nothing to process"):
        src.vszip.BM3D(sigma=0.0)


# --- VAggregate --------------------------------------------------------------


def _stacked(core, w, h, r, nframes, seed):
    """A float stacked clip: slot z of frame m holds (num, den) planes."""
    rng = np.random.default_rng(seed)
    data = rng.random((nframes, 2 * (2 * r + 1), h, w), dtype=np.float32) + np.float32(0.25)
    base = core.std.BlankClip(None, w, h * 2 * (2 * r + 1), vs.GRAYS, length=nframes)

    def fill(n, f):
        fo = f.copy()
        np.asarray(fo[0])[:] = data[n].reshape(-1, w)
        return fo

    return base.std.ModifyFrame(base, fill), data


def test_vaggregate_oracle(core):
    r, n_fr, w, h = 2, 6, 37, 11
    stacked, data = _stacked(core, w, h, r, n_fr, 5)
    src = core.std.BlankClip(None, w, h, vs.GRAYS, length=n_fr)
    got = frames(core.vszip.VAggregate(stacked, src=src, planes=[0]))
    for n in range(n_fr):
        num = np.zeros((h, w), np.float32)
        den = np.zeros((h, w), np.float32)
        for i in range(2 * r + 1):
            m = min(max(n - r + i, 0), n_fr - 1)
            z = min(max(2 * r - i, n - n_fr + 1 + r), n + r)
            num += data[m, 2 * z]
            den += data[m, 2 * z + 1]
        assert np.array_equal(got[n][0], num / den), f"frame {n}"


def test_vaggregate_validation(core):
    src = core.std.BlankClip(None, 32, 16, vs.GRAYS, length=2)
    st = core.std.BlankClip(None, 32, 16 * 6, vs.GRAYS, length=2)
    with pytest.raises(vs.Error, match="out-of-range plane index"):
        core.vszip.VAggregate(st, src=src, planes=[1])
    with pytest.raises(vs.Error, match="duplicate plane index"):
        core.vszip.VAggregate(st, src=src, planes=[0, 0])
    with pytest.raises(vs.Error, match="not a BM3D stacked clip"):
        core.vszip.VAggregate(src, src=src, planes=[0])


# --- validation --------------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "msg"),
    [
        (dict(sigma=-1.0), "must be non-negative"),
        (dict(sigma=[1.0, 1.0, 1.0, 1.0]), "too many elements"),
        (dict(block_step=0), r"block_step\" must be in range \[1, 8\]"),
        (dict(block_step=9), r"block_step\" must be in range \[1, 8\]"),
        (dict(bm_range=0), "bm_range\" must be positive"),
        (dict(ps_num=9), r"ps_num\" must be in range \[1, 8\]"),
        (dict(ps_range=0), "ps_range\" must be positive"),
        (dict(radius=-1), "radius\" must be non-negative"),
        (dict(radius=17), "radius\" must be <= 16"),
        (dict(chroma=True), "YUV444"),
        (dict(bm_error_s="abc"), "invalid \"bm_error_s\""),
        (dict(transform_2d_s="fft"), "invalid \"transform_2d_s\""),
        (dict(transform_1d_s="fft"), "invalid \"transform_1d_s\""),
        (dict(extractor_exp=200), "extractor_exp"),
    ],
)
def test_validation_errors(core, args, msg):
    src = core.std.BlankClip(None, 64, 48, vs.YUV420PS, length=1)
    with pytest.raises(vs.Error, match=msg):
        src.vszip.BM3D(**args)


def test_format_and_ref_errors(core):
    with pytest.raises(vs.Error, match="32 bit float"):
        core.std.BlankClip(None, 64, 48, vs.GRAY16, length=1).vszip.BM3D()
    src = core.std.BlankClip(None, 64, 48, vs.GRAYS, length=2)
    with pytest.raises(vs.Error, match="same format"):
        src.vszip.BM3D(ref=core.std.BlankClip(None, 64, 48, vs.YUV444PS, length=2))
    with pytest.raises(vs.Error, match="same dimensions"):
        src.vszip.BM3D(ref=core.std.BlankClip(None, 64, 40, vs.GRAYS, length=2))
    with pytest.raises(vs.Error, match="same number of frames"):
        src.vszip.BM3D(ref=core.std.BlankClip(None, 64, 48, vs.GRAYS, length=3))
    with pytest.raises(vs.Error, match="at least 8x8"):
        core.std.BlankClip(None, 64, 7, vs.GRAYS, length=1).vszip.BM3D()


# --- optional: parity with vszipcu (the CUDA plugin) --------------------------


@pytest.mark.skipif(not HAS_CU, reason="vszipcu (CUDA BM3D) not installed")
@pytest.mark.parametrize(
    ("fmt", "kw", "max_diff_px"),
    [
        (vs.GRAYS, {}, 0),
        (vs.GRAYS, dict(radius=1), 0),
        (vs.GRAYS, dict(radius=2, ps_num=3, ps_range=6, block_step=5, bm_range=7), 0),
        (vs.GRAYS, dict(final=True), 2),
        (vs.GRAYS, dict(final=True, radius=1), 2),
        (vs.GRAYS, dict(transform_2d_s="haar", transform_1d_s="bior1.5", radius=1), 0),
        (vs.GRAYS, dict(transform_2d_s="wht", bm_error_s="zssd"), 0),
        (vs.YUV420PS, dict(radius=1), 2),
        (vs.YUV444PS, dict(chroma=True), 2),
    ],
)
def test_matches_cuda_plugin(clip, fmt, kw, max_diff_px):
    """With extractor_exp the GPU sums are exact and order-free, so the GPU is
    deterministic; extractor_exp=10 keeps every sum inside the exact range."""
    src = clip(fmt)
    kw = dict(kw, extractor_exp=10)
    if kw.pop("final", False):
        kw["ref"] = src.vszipcu.BM3Dv2(radius=kw.get("radius", 0), extractor_exp=10)
    a = frames(src.vszip.BM3D(**kw))
    b = frames(src.vszipcu.BM3Dv2(**kw))
    for n, (fa, fb) in enumerate(zip(a, b)):
        diff = sum(int(np.count_nonzero(pa != pb)) for pa, pb in zip(fa, fb))
        assert diff <= max_diff_px, f"frame {n}: {diff} pixels differ"
