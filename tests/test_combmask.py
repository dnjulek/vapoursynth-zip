"""CombMask and CombMaskMT.

Two filters share the "combmask" golden store. Golden keys are method-prefixed
(CombMask|... and CombMaskMT|...) via the Case `variant` tag so the two CASES
lists never collide.

Both filters accept 8-bit integer clips only (the Create callback rejects
anything else). CombMask reacts to inter-frame motion when mthresh > 0, so its
golden cases use the 3-frame shifted temporal clip and stat frame n=1 (frame 0
has no previous frame -> empty motion mask). CombMaskMT is purely spatial and
uses the single-frame clip.
"""

import pytest
import vapoursynth as vs

from golden import Case, grid, sweep
from helpers import assert_binary, assert_has_gradient, avg, plane_stats

# CombMask accepts 8-bit int only; sweep the YUV/GRAY families the wiki lists.
CM_FORMATS = [vs.GRAY8, vs.YUV420P8, vs.YUV444P8]
MT_FORMATS = [vs.GRAY8, vs.YUV420P8, vs.YUV444P8]


# --- CombMask ----------------------------------------------------------------

# The 3-frame clip shifts by a full row per frame, so every pixel "moves":
# small mthresh values mark everything as motion and gate nothing (motion AND is
# a no-op). mthresh only becomes active once it exceeds the typical inter-frame
# delta (>= ~20 here), so the mthresh axis uses values that measurably differ.
CASES_CM = (
    sweep(
        base_fmt=vs.GRAY8,
        base_args=dict(cthresh=8, mthresh=50),
        formats=CM_FORMATS,
        # cthresh changes the spatial mask in both metric paths; mthresh gates
        # the motion path once above the inter-frame delta; expand/metric pick
        # distinct kernels.
        args=grid(cthresh=[4, 8, 16, 32])
        + grid(mthresh=[0, 50, 100, 150])
        + [
            dict(cthresh=8, mthresh=50, metric=1),
            dict(cthresh=8, mthresh=0, metric=1),
            dict(cthresh=8, mthresh=50, expand=False),
            dict(cthresh=8, mthresh=50, metric=1, expand=False),
            dict(cthresh=8, mthresh=0, expand=False),
            # metric=1 spatial-only (mthresh=0) with expand disabled: the eighth
            # and only remaining (metric_1, expand, motion) getFrame variant,
            # CombMask(true, false, false) at comb_mask.zig L137.
            dict(cthresh=8, mthresh=0, metric=1, expand=False),
            # metric=1 widens the cthresh range far past 255
            dict(cthresh=400, mthresh=50, metric=1),
        ],
        geometries=["odd", "tiny"],
        variant="CombMask",
    )
    + [
        # default args (cthresh=6, mthresh=9, expand=True, metric=0)
        Case(vs.GRAY8, args=dict(), variant="CombMask"),
        # metric=1 interacting with expand=False under active motion gating
        Case(vs.YUV420P8, args=dict(cthresh=16, mthresh=100, metric=1, expand=False), variant="CombMask"),
        # spatial-only, metric=1, no-motion path on subsampled chroma
        Case(vs.YUV420P8, args=dict(cthresh=8, mthresh=0, metric=1), variant="CombMask"),
    ]
)


@pytest.mark.parametrize("case", CASES_CM, ids=str)
def test_golden_cases_cm(golden, make_temporal_clip, case):
    src = make_temporal_clip(case.fmt, case.geometry)
    golden.check("combmask", case, src.vszip.CombMask(**case.args), n=1)


# --- CombMaskMT --------------------------------------------------------------

# Purely spatial: single-frame clip, default n=0.
CASES_MT = (
    sweep(
        base_fmt=vs.GRAY8,
        base_args=dict(thY1=30, thY2=30),
        formats=MT_FORMATS,
        # thY1==thY2 -> binary mask (same_thr path); thY1<thY2 -> gray gradient.
        args=[
            dict(thY1=10, thY2=10),
            dict(thY1=60, thY2=60),
            dict(thY1=100, thY2=100),
            dict(thY1=0, thY2=255),  # disable binarize, full gray range
            dict(thY1=10, thY2=200),
            dict(thY1=30, thY2=120),
            # edge values
            dict(thY1=0, thY2=0),
            dict(thY1=255, thY2=255),
            dict(thY1=0, thY2=30),
            dict(thY1=200, thY2=255),
        ],
        geometries=["odd", "tiny"],
        variant="CombMaskMT",
    )
    + [
        # default args (thY1=30, thY2=30)
        Case(vs.GRAY8, args=dict(), variant="CombMaskMT"),
        # gradient path on a subsampled-chroma format
        Case(vs.YUV420P8, args=dict(thY1=0, thY2=255), variant="CombMaskMT"),
        Case(vs.YUV444P8, args=dict(thY1=20, thY2=150), variant="CombMaskMT"),
    ]
)


@pytest.mark.parametrize("case", CASES_MT, ids=str)
def test_golden_cases_mt(golden, make_clip, case):
    src = make_clip(case.fmt, case.geometry)
    golden.check("combmask", case, src.vszip.CombMaskMT(**case.args), n=0)


# --- existing CombMask coverage ---------------------------------------------


@pytest.fixture(scope="module")
def src8(temporal_rgb):
    # Point resize preserves the dot-crawl-like detail the masks react to
    return temporal_rgb.resize.Point(format=vs.GRAY8, matrix=1).std.RemoveFrameProps("_Matrix")


# Frame-1 averages; integer mask output, so values are exact.
GOLDENS = [
    (dict(), 0.206181640625),
    (dict(cthresh=8, mthresh=2, metric=0), 0.1980615234375),
    (dict(cthresh=8, mthresh=2, metric=1), 0.2363623046875),
    (dict(cthresh=8, mthresh=100), 0.05046875),
    (dict(cthresh=8, mthresh=2, expand=False), 0.094482421875),
]


@pytest.mark.parametrize(("args", "expected"), GOLDENS)
def test_golden(src8, args, expected):
    assert avg(src8.vszip.CombMask(**args), n=1) == expected


def test_output_is_binary(src8):
    for metric in (0, 1):
        assert_binary(src8.vszip.CombMask(metric=metric), n=1)
    assert_binary(src8.vszip.CombMask(expand=False), n=1)


def test_first_frame_has_no_motion(src8):
    """With mthresh > 0 the first frame compares against itself, so the
    motion mask is empty and nothing is marked."""
    assert avg(src8.vszip.CombMask(cthresh=8, mthresh=2), n=0) == 0.0
    assert avg(src8.vszip.CombMask(cthresh=8, mthresh=0), n=0) == 0.196611328125


def test_expand_is_superset(src8):
    expanded = src8.vszip.CombMask(cthresh=8, mthresh=0)
    plain = src8.vszip.CombMask(cthresh=8, mthresh=0, expand=False)
    implies = vs.core.std.Expr([expanded, plain], "x y >= 255 0 ?")
    assert plane_stats(implies, n=1)["PlaneStatsMin"] == 255


def test_non_8bit_error(to_gray):
    with pytest.raises(vs.Error, match="only 8 bit int format supported"):
        to_gray(vs.GRAY16).vszip.CombMask()


@pytest.mark.parametrize(
    ("args", "msg"),
    [
        (dict(cthresh=256), "cthresh must be between 0 and 255 when metric = false"),
        (dict(cthresh=-1), "cthresh must be between 0 and 255 when metric = false"),
        (dict(cthresh=65026, metric=1), "cthresh must be between 0 and 65025 when metric = true"),
        (dict(cthresh=-1, metric=1), "cthresh must be between 0 and 65025 when metric = true"),
        (dict(mthresh=256), "mthresh must be between 0 and 255"),
        (dict(mthresh=-1), "mthresh must be between 0 and 255"),
    ],
)
def test_threshold_errors(src8, args, msg):
    with pytest.raises(vs.Error, match=msg):
        src8.vszip.CombMask(**args)


def test_metric1_allows_large_cthresh(src8):
    assert avg(src8.vszip.CombMask(cthresh=300, metric=1), n=1) > 0.0


# --- existing CombMaskMT coverage -------------------------------------------


def test_mt_golden(to_gray):
    src = to_gray(vs.GRAY8)
    assert avg(src.vszip.CombMaskMT()) == 0.1150439453125
    assert avg(src.vszip.CombMaskMT(0, 255)) == 0.10427868412990196


def test_mt_binarize(to_gray):
    src = to_gray(vs.GRAY8)
    # thY1 == thY2 collapses thr_diff to 0 -> the same_thr path, a binary mask.
    assert_binary(src.vszip.CombMaskMT())
    # thY1 != thY2 remaps the combing value over [thY1, thY2] to a 0..255 ramp,
    # so the in-between band yields real intermediate (gray) values, NOT a
    # binary mask. This is an intentional deviation from AviSynth MTCombMask
    # (which floors the in-between band to combing_value/256 == 0 for 8-bit).
    assert_has_gradient(src.vszip.CombMaskMT(0, 255))


def test_mt_non_8bit_error(to_gray):
    with pytest.raises(vs.Error, match="only 8 bit int format supported"):
        to_gray(vs.GRAY16).vszip.CombMaskMT()


@pytest.mark.parametrize(
    ("args", "msg"),
    [
        (dict(thY1=-1), r"thY1 value should be in range \[0;255\]"),
        (dict(thY1=256), r"thY1 value should be in range \[0;255\]"),
        (dict(thY2=-1), r"thY2 value should be in range \[0;255\]"),
        (dict(thY2=256), r"thY2 value should be in range \[0;255\]"),
        (dict(thY1=31, thY2=30), "thY1 can't be greater than thY2"),
        (dict(thY1=255, thY2=0), "thY1 can't be greater than thY2"),
    ],
)
def test_mt_threshold_errors(to_gray, args, msg):
    with pytest.raises(vs.Error, match=msg):
        to_gray(vs.GRAY8).vszip.CombMaskMT(**args)


@pytest.mark.parametrize("h", [1, 2])
def test_tiny_height_errors(core, h):
    """Planes shorter than the 3-row comb window underflowed the u32 row counter
    (h-3 / h-2) -> Debug panic / OOB read. Both Create callbacks now reject such
    clips up front (the kernels stay branch-free)."""
    src = core.std.BlankClip(None, 64, h, vs.GRAY8, length=2, color=100)
    builders = [
        lambda: src.vszip.CombMask(),
        lambda: src.vszip.CombMask(metric=1),
        lambda: src.vszip.CombMask(mthresh=10),
        lambda: src.vszip.CombMaskMT(),
    ]
    for build in builders:
        with pytest.raises(vs.Error, match="clip too small"):
            build()


def test_small_chroma_plane_errors(core):
    """The 3-row check is per plane: a YUV420 luma of 4 rows still leaves a
    chroma plane of only 2 (< 3), so both filters must reject the clip."""
    src = core.std.BlankClip(None, 64, 4, vs.YUV420P8, length=2, color=[100, 128, 128])
    for build in (lambda: src.vszip.CombMask(), lambda: src.vszip.CombMaskMT()):
        with pytest.raises(vs.Error, match="clip too small"):
            build()


@pytest.mark.parametrize("h", [3, 4])
def test_min_height_accepted(core, h):
    """Boundary: exactly 3 rows is the smallest plane the comb filters accept."""
    src = core.std.BlankClip(None, 64, h, vs.GRAY8, length=2, color=100)
    for out in (src.vszip.CombMask(), src.vszip.CombMaskMT()):
        assert (out.width, out.height) == (64, h)
        out.get_frame(1)
