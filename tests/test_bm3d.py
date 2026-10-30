"""BM3D on the CPU (vszipcu's algorithm and arguments).

Contract (src/vapoursynth/bm3d.zig, src/filters/bm3d.zig):
- 32-bit float only; output keeps the input format, size, length and props;
  planes with sigma < FLT_EPSILON are copied from the source, and a clip with
  nothing to process (every sigma 0) is returned unchanged.
- sigma/block_step/bm_range/ps_num/ps_range/bm_error_s/transform_*_s are
  per-plane arrays (the last given value repeats); chroma=True (CBM3D)
  needs YUV444 and matches on plane 0.
- radius > 0 aggregates internally (no stacked clip, no VAggregate step); the
  temporal window is vszipcu's (frames clamped at the clip ends).
- Same algorithm as vszipcu, not the GPU's rounding: its fast-math
  reciprocal/divide and its order of the separable transform passes are not
  reproduced. There is no extractor_exp (a GPU-only repeatability knob), so
  the parity test runs the GPU without it too.

Nets:
  1. metamorphic properties (flat input, sigma-0 plane passthrough, crop vs
     repack, request order and thread count);
  2. optional parity with vszipcu (the CUDA plugin) when it is installed;
  3. golden snapshots.
"""

import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import vapoursynth as vs

from golden import Case
from helpers import assert_long_clip_edges, assert_same_clip, repack

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
    src = core.std.BlankClip(None, 64, 64, vs.GRAYS, length=3, color=0.4)
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


@pytest.mark.parametrize("kw", [{}, {"radius": 1}, {"final": True}])
def test_all_zero_sigma_passes_through(clip, kw):
    """Nothing to process: the clip itself comes back, as with the other
    filters (including a given ref, which is then unused)."""
    src = clip(vs.YUV420PS)
    if kw.pop("final", False):
        kw["ref"] = src
    assert_same_clip(src.vszip.BM3D(sigma=0.0, **kw), src)
    assert_same_clip(src.vszip.BM3D(sigma=[0.0, 0.0, 0.0], **kw), src)


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
    ],
)
def test_validation_errors(core, args, msg):
    src = core.std.BlankClip(None, 64, 64, vs.YUV420PS, length=1)
    with pytest.raises(vs.Error, match=msg):
        src.vszip.BM3D(**args)


def test_format_and_ref_errors(core):
    with pytest.raises(vs.Error, match="32 bit float"):
        core.std.BlankClip(None, 64, 64, vs.GRAY16, length=1).vszip.BM3D()
    src = core.std.BlankClip(None, 64, 64, vs.GRAYS, length=2)
    with pytest.raises(vs.Error, match="same format"):
        src.vszip.BM3D(ref=core.std.BlankClip(None, 64, 64, vs.YUV444PS, length=2))
    with pytest.raises(vs.Error, match="same dimensions"):
        src.vszip.BM3D(ref=core.std.BlankClip(None, 64, 40, vs.GRAYS, length=2))
    with pytest.raises(vs.Error, match="same number of frames"):
        src.vszip.BM3D(ref=core.std.BlankClip(None, 64, 64, vs.GRAYS, length=3))
    for w, h in ((64, 63), (63, 64)):
        with pytest.raises(vs.Error, match="at least 64x64"):
            core.std.BlankClip(None, w, h, vs.GRAYS, length=1).vszip.BM3D()


# --- optional: parity with vszipcu (the CUDA plugin) --------------------------


@pytest.mark.skipif(not HAS_CU, reason="vszipcu (CUDA BM3D) not installed")
@pytest.mark.parametrize(
    ("fmt", "kw"),
    [
        (vs.GRAYS, {}),
        (vs.GRAYS, dict(radius=1)),
        (vs.GRAYS, dict(radius=2, ps_num=3, ps_range=6, block_step=5, bm_range=7)),
        (vs.GRAYS, dict(final=True)),
        (vs.GRAYS, dict(final=True, radius=1)),
        (vs.GRAYS, dict(transform_2d_s="haar", transform_1d_s="bior1.5", radius=1)),
        (vs.GRAYS, dict(transform_2d_s="wht", bm_error_s="zssd")),
        (vs.YUV420PS, dict(radius=1)),
        (vs.YUV444PS, dict(chroma=True)),
    ],
)
def test_matches_cuda_plugin(clip, fmt, kw):
    """Same algorithm as vszipcu, not the same rounding: the GPU's
    -use_fast_math reciprocal/divide (MUFU.RCP, +-1 ulp) and its order of the
    separable transform passes are not reproduced. The GPU sums with atomics
    in no fixed order, so its own output varies at float-rounding level from
    run to run (its extractor_exp, which makes it repeatable, quantizes the
    output far more than that, and vszip has no extractor). Both final
    estimates use the same basic estimate as ref."""
    src = clip(fmt)
    if kw.pop("final", False):
        kw["ref"] = src.vszip.BM3D(radius=kw.get("radius", 0))
    a = frames(src.vszip.BM3D(**kw))
    b = frames(src.vszipcu.BM3Dv2(**kw))
    for n, (fa, fb) in enumerate(zip(a, b)):
        d = np.concatenate([np.abs(pa.astype(np.float64) - pb).ravel() for pa, pb in zip(fa, fb)])
        # measured over these 27 frames: mean <= 4.1e-8, max <= 6.0e-7
        assert d.mean() < 1e-6, f"frame {n}: mean diff {d.mean():.3g}"
        assert d.max() < 1e-5, f"frame {n}: max diff {d.max():.3g}"


@pytest.mark.parametrize("call", [
    "{c}.vszip.BM3D(radius=1)",
    "{c}.vszip.BM3D(radius=2)",
    "{c}.vszip.BM3D(radius=1, ref={c})",
])
def test_last_frames_of_the_longest_clip(call):
    """Frame windows near frame 2^31 - 1 (radius 1 and 2, the final
    estimate) are formed without i32 overflow."""
    assert_long_clip_edges(call)