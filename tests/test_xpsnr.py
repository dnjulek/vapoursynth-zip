import math
import subprocess
import sys
from pathlib import Path

import pytest
import vapoursynth as vs

from golden import Case, grid, sweep
from helpers import assert_same_clip, props

TESTS_DIR = Path(__file__).resolve().parent


# --- golden snapshot coverage ----------------------------------------------
#
# XPSNR writes frame props (XPSNR_Y/_U/_V), not processed pixels, so the
# goldens record per-frame prop dicts via golden.check_value, not per-plane
# pixel stats. The second ("distorted") clip is a deterministic transform of
# the reference; the transform kind is carried in Case.variant and mapped to a
# concrete operation by `_distort` below.
#
# Geometry is pinned to "full". The XPSNR kernel indexes its temporal-state
# buffers (og_m1/og_m2, sized width*height) by frame STRIDE rather than width,
# so any clip with stride padding -- which "odd" and "tiny" both introduce --
# walks off the allocation and aborts the process (bounds panic in
# calcSquaredErrorAndWeight). Until that src bug is fixed the non-"full"
# geometries cannot be exercised here; coverage breadth instead comes from the
# distortion x temporal x format x per-frame axes.

DISTORTIONS = ("box2", "box5", "bright", "shift")


def _distort(clip: vs.VideoNode, kind: str) -> vs.VideoNode:
    """Deterministic distortion of `clip`. Every plane is perturbed so that
    XPSNR_U/_V stay finite (an untouched plane scores +inf, which is not
    golden-storable); each kind yields a measurably distinct score."""
    if kind == "box2":
        return clip.std.BoxBlur(hradius=2, vradius=2)
    if kind == "box5":
        return clip.std.BoxBlur(hradius=5, vradius=5)
    if kind == "bright":
        return clip.std.Expr("x 12 +")
    if kind == "shift":
        return clip.std.Expr("x 1 +")
    raise ValueError(f"unknown distortion {kind!r}")


CASES = (
    sweep(
        base_fmt=vs.YUV420P8,
        base_args=dict(temporal=True),
        # YUV 8/10 bit are the only accepted formats (see xpsnrCreate).
        formats=[vs.YUV420P8, vs.YUV420P10],
        # temporal on/off measurably changes the score (temporal weighting
        # folds in inter-frame activity).
        args=grid(temporal=[True, False]),
        variant="box2",
    )
    # distortion sweep: one Case per distortion x temporal on/off, all on the
    # 8-bit base format. Each distortion produces a distinct golden.
    + [
        Case(vs.YUV420P8, args=dict(temporal=t), variant=k)
        for k in DISTORTIONS
        for t in (True, False)
    ]
    # plus the same distortion sweep on 10-bit to lock the higher-depth path.
    + [
        Case(vs.YUV420P10, args=dict(temporal=t), variant=k)
        for k in DISTORTIONS
        for t in (True, False)
    ]
    # subsampling coverage: 422 and 444 take a different chroma-block path
    # (bx/by = b*w_pln/w in getWSSE) and a different whFromVi branch than 420.
    # The luma score is unchanged but XPSNR_U/_V genuinely differ per
    # subsampling, so each locks a distinct chroma golden. Both depths, both
    # temporal flags, on the box2 distortion (representative of the others).
    + [
        Case(fmt, args=dict(temporal=t), variant="box2")
        for fmt in (vs.YUV422P8, vs.YUV444P8, vs.YUV422P10, vs.YUV444P10)
        for t in (True, False)
    ]
)


@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_cases(golden, make_temporal_clip, case):
    ref = make_temporal_clip(case.fmt, case.geometry)
    dist = _distort(ref, case.variant)
    # verbose=False keeps the free-time stdout summary out of the test log; it
    # does not affect frame props.
    out = vs.core.vszip.XPSNR(ref, dist, verbose=False, **case.args)
    # Request frames strictly in order: the temporal path carries inter-frame
    # state, so out-of-order requests would change the scores.
    rel = 1e-6
    for n in range(out.num_frames):
        p = props(out, n)
        golden.check_value(
            "xpsnr",
            f"{case.id}|n{n}",
            {"Y": p["XPSNR_Y"], "U": p["XPSNR_U"], "V": p["XPSNR_V"]},
            rel=rel,
        )


@pytest.fixture(scope="module")
def pair(temporal_rgb):
    yuv = temporal_rgb.resize.Bilinear(format=vs.YUV420P8, matrix=1)
    return yuv, yuv.std.BoxBlur(hradius=2, vradius=2)


def ordered_props(clip: vs.VideoNode) -> list[dict]:
    """Request frames strictly in order: the temporal path keeps inter-frame
    state, so out-of-order requests change the scores."""
    return [props(clip, n) for n in range(clip.num_frames)]


def test_identical_clips_are_inf(pair):
    yuv, _ = pair
    p = props(vs.core.vszip.XPSNR(yuv, yuv, verbose=False))
    assert p["XPSNR_Y"] == p["XPSNR_U"] == p["XPSNR_V"] == math.inf


def test_golden_temporal(pair):
    yuv, dist = pair
    measured = ordered_props(vs.core.vszip.XPSNR(yuv, dist, temporal=True, verbose=False))
    assert [p["XPSNR_Y"] for p in measured] == pytest.approx(
        [26.879032147505455, 23.60520019665031, 23.599462216699855], rel=1e-6
    )
    assert [measured[0]["XPSNR_U"], measured[0]["XPSNR_V"]] == pytest.approx(
        [33.66546530912062, 28.232759256616703], rel=1e-6
    )


def test_golden_spatial(pair):
    yuv, dist = pair
    measured = ordered_props(vs.core.vszip.XPSNR(yuv, dist, temporal=False, verbose=False))
    assert [p["XPSNR_Y"] for p in measured] == pytest.approx(
        [22.84542330071783, 22.842022530697292, 22.835337286397316], rel=1e-6
    )


def test_temporal_differs_from_spatial(pair):
    yuv, dist = pair
    temporal = ordered_props(vs.core.vszip.XPSNR(yuv, dist, temporal=True, verbose=False))
    spatial = ordered_props(vs.core.vszip.XPSNR(yuv, dist, temporal=False, verbose=False))
    assert all(t["XPSNR_Y"] != s["XPSNR_Y"] for t, s in zip(temporal, spatial))


def test_output_frame_is_distorted_copy(pair):
    yuv, dist = pair
    assert_same_clip(vs.core.vszip.XPSNR(yuv, dist, verbose=False), dist)


def test_mixed_depth_aligns_to_highest(pair):
    yuv, dist = pair
    out = vs.core.vszip.XPSNR(yuv, dist.resize.Point(format=vs.YUV420P10), verbose=False)
    assert out.format.bits_per_sample == 10
    assert props(out)["XPSNR_Y"] != math.inf


def test_verbose_does_not_change_props(pair):
    """verbose only controls the free-time stdout summary; the emitted frame
    props are identical regardless of its value."""
    yuv, dist = pair
    quiet = props(vs.core.vszip.XPSNR(yuv, dist, temporal=False, verbose=False))
    loud = props(vs.core.vszip.XPSNR(yuv, dist, temporal=False, verbose=True))
    for k in ("XPSNR_Y", "XPSNR_U", "XPSNR_V"):
        assert quiet[k] == loud[k]


def test_rgb_error(core):
    src = core.std.BlankClip(None, 64, 32, vs.RGB24, length=1)
    with pytest.raises(vs.Error, match="only supports YUV format clips"):
        core.vszip.XPSNR(src, src)


def test_bit_depth_error(core):
    src = core.std.BlankClip(None, 64, 32, vs.YUV420P16, length=1)
    with pytest.raises(vs.Error, match="only supports 8 or 10 bit clips"):
        core.vszip.XPSNR(src, src)


def test_length_error(core):
    a = core.std.BlankClip(None, 64, 32, vs.YUV420P8, length=5)
    with pytest.raises(vs.Error, match="all input clips must have the same length"):
        core.vszip.XPSNR(a, a.std.Trim(0, 2))


def _err_clips(core, ref_fmt, dist_fmt=None, ref_len=3, dist_len=None, dist_size=None):
    ref = core.std.BlankClip(None, 64, 32, ref_fmt, length=ref_len)
    dw, dh = dist_size or (64, 32)
    dist = core.std.BlankClip(None, dw, dh, dist_fmt or ref_fmt, length=dist_len or ref_len)
    return ref, dist


# Every range / color-family / bit-depth / length check that xpsnrCreate can
# reach, in source order. The first two guards (color family, bit depth) run on
# the `reference` clip up front; the rest are the per-clip compareNodes checks
# applied to `distorted` (helper.compareNodes, SAME_LEN). The compareNodes
# bit-depth check is unreachable because xpsnrCreate aligns the two clips'
# depths before comparing, so it has no row here.
@pytest.mark.parametrize(
    ("ref_fmt", "dist_fmt", "ref_len", "dist_len", "dist_size", "msg"),
    [
        # non-YUV reference is rejected before any bit-depth check
        (vs.RGB24, None, 3, 3, None, "only supports YUV format clips"),
        (vs.GRAY8, None, 3, 3, None, "only supports YUV format clips"),
        # YUV reference outside {8,10} bit
        (vs.YUV420P16, None, 3, 3, None, "only supports 8 or 10 bit clips"),
        (vs.YUV420P12, None, 3, 3, None, "only supports 8 or 10 bit clips"),
        (vs.YUV444PS, None, 3, 3, None, "only supports 8 or 10 bit clips"),
        # distorted with mismatched dimensions (same family/depth/subsampling)
        (vs.YUV420P8, None, 3, 3, (48, 32), "must have the same width and height"),
        # distorted of a different color family but matching bit depth (so no
        # realignment masks the family mismatch)
        (vs.YUV420P8, vs.RGB24, 3, 3, None, "must have the same color family"),
        # distorted with a different subsampling
        (vs.YUV420P8, vs.YUV422P8, 3, 3, None, "must have the same subsampling"),
        # mismatched clip lengths
        (vs.YUV420P8, None, 5, 2, None, "all input clips must have the same length"),
        (vs.YUV420P10, None, 4, 3, None, "all input clips must have the same length"),
    ],
)
def test_validation_errors(core, ref_fmt, dist_fmt, ref_len, dist_len, dist_size, msg):
    ref, dist = _err_clips(core, ref_fmt, dist_fmt, ref_len, dist_len, dist_size)
    with pytest.raises(vs.Error, match=msg):
        core.vszip.XPSNR(ref, dist)


def test_no_teardown_corruption():
    """Regression guard: freeing an XPSNR node used to corrupt the heap
    (og_m1/og_m2 alignedAlloc/free alignment mismatch, fixed by typing them
    as aligned slices). Subprocess so a regression can't take pytest down."""
    script = (
        f"import sys; sys.path.insert(0, {str(TESTS_DIR)!r})\n"
        "import vapoursynth as vs\n"
        "from conftest import _load_vszip\n"
        "core = _load_vszip()\n"
        "a = core.std.BlankClip(None, 64, 64, vs.YUV420P8, length=1)\n"
        "core.vszip.XPSNR(a, a, verbose=False).get_frame(0)\n"
        "print('OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "OK" in r.stdout


def test_unaligned_width_no_stride_oob(temporal_rgb):
    """Regression: og_m1/og_m2 were sized width*height but indexed by the luma
    frame stride, so any clip whose stride exceeds its width (width not a
    multiple of the frame alignment) read/wrote out of bounds. width=100 pads
    the YUV420P8 luma stride above 100; this used to abort the process."""
    src = temporal_rgb.std.CropAbs(width=100, height=80, left=8, top=8).resize.Bilinear(format=vs.YUV420P8, matrix=1)
    dist = src.std.BoxBlur(hradius=2, vradius=2)
    out = vs.core.vszip.XPSNR(src, dist, temporal=True)
    scores = [props(out, n)["XPSNR_Y"] for n in range(out.num_frames)]
    assert all(math.isfinite(s) for s in scores)


def test_tiny_clip_no_block_divzero(temporal_rgb):
    """Regression: for clips with w*h below ~2025 the block size b rounds to 0,
    so the block-count division (w+b-1)/b divided by zero before reaching the
    b<4 plain-PSNR fallback. 32x32 hits b==0."""
    src = temporal_rgb.std.CropAbs(width=32, height=32, left=8, top=8).resize.Bilinear(format=vs.YUV420P8, matrix=1)
    dist = src.std.BoxBlur(hradius=1, vradius=1)
    assert math.isfinite(props(vs.core.vszip.XPSNR(src, dist), 0)["XPSNR_Y"])
