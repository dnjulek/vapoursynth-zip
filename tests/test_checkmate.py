import pytest
import vapoursynth as vs

from golden import Case, grid, sweep
from helpers import assert_same_clip, avg, diff, plane_stats, repack

# Checkmate accepts any 8-bit integer format; the Create callback only checks
# sampleType==Integer and bitsPerSample==8. (RGB24 is also accepted by the
# filter but the make_temporal_clip fixture forces matrix=1 on its Point
# resize, which RGB output rejects, so only GRAY/YUV 8-bit are swept here.)
# The temporal blend path is reached only when tthr2 > 0 (use_tthr2=true), so
# positive tthr2 values are swept alongside the spatial thr/tmax controls.
# Goldens are measured at n=1 so the requested n-1/n+1 (and n-2/n+2 for tthr2)
# neighbours are real frames of the 3-frame shifted source, not clamped repeats.
CASES = (
    sweep(
        base_fmt=vs.GRAY8,
        base_args=dict(thr=12, tmax=12, tthr2=0),
        formats=[vs.GRAY8, vs.YUV420P8, vs.YUV422P8, vs.YUV444P8],
        args=grid(thr=[4, 12, 40], tmax=[1, 12, 64])
        + [
            # temporal blend path (tthr2 > 0); positive values pick different
            # pixels for the (p1+2*c+n1)/4 average vs the spatial weighting
            dict(thr=12, tmax=12, tthr2=4),
            dict(thr=12, tmax=12, tthr2=16),
            dict(thr=12, tmax=12, tthr2=64),
            dict(thr=4, tmax=4, tthr2=8),
            dict(thr=40, tmax=64, tthr2=32),
        ],
        geometries=["odd", "tiny"],
    )
    + [
        # thr/tmax interaction at the spatial extremes
        Case(vs.GRAY8, args=dict(thr=0, tmax=1, tthr2=0)),
        Case(vs.GRAY8, args=dict(thr=255, tmax=255, tthr2=0)),
        # temporal path on chroma-subsampled layouts
        Case(vs.YUV420P8, args=dict(thr=14, tmax=11, tthr2=4)),
        Case(vs.YUV422P8, args=dict(thr=14, tmax=11, tthr2=8)),
    ]
)


@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_cases(golden, make_temporal_clip, case):
    src = make_temporal_clip(case.fmt, case.geometry)
    golden.check("checkmate", case, src.vszip.Checkmate(**case.args), n=1)


# RGB24 is a distinct accepted-format path (3-plane RGB processing) that the
# make_temporal_clip fixture cannot build, because it forces matrix=1 on its
# Point resize and RGB output rejects YUV matrix coefficients. temporal_rgb is
# already an RGB24 clip, so the RGB family is golden-covered directly from it
# (spatial-only and temporal-blend branches, both reached on all three planes).
RGB_CASES = (
    Case(vs.RGB24, args=dict(thr=12, tmax=12, tthr2=0)),
    Case(vs.RGB24, args=dict(thr=14, tmax=11, tthr2=8)),
)


@pytest.mark.parametrize("case", RGB_CASES, ids=str)
def test_golden_rgb(golden, temporal_rgb, case):
    assert temporal_rgb.format.id == vs.RGB24
    golden.check("checkmate", case, temporal_rgb.vszip.Checkmate(**case.args), n=1)


# Frame-1 averages. tthr2=0 is the spatial path (carried from the old .vpy
# suite); the tthr2=4 value is the corrected temporal path (the old .vpy value
# 0.48719424019607843 encoded the row-0 temporal bug, see
# test_temporal_reads_correct_rows).
GOLDENS = [
    (dict(thr=12, tmax=12, tthr2=0), 0.4871367378982843),
    (dict(thr=14, tmax=11, tthr2=4), 0.48752056525735293),
]


@pytest.fixture(scope="module")
def src8(temporal_rgb):
    return temporal_rgb.resize.Bilinear(format=vs.GRAY8, matrix=1).std.RemoveFrameProps("_Matrix")


@pytest.mark.parametrize(("args", "expected"), GOLDENS)
def test_golden(src8, args, expected):
    assert avg(src8.vszip.Checkmate(**args), n=1) == pytest.approx(expected, rel=1e-6)


def test_temporal_blending_changes_output(src8):
    spatial_only = src8.vszip.Checkmate(thr=12, tmax=12, tthr2=0)
    temporal = src8.vszip.Checkmate(thr=12, tmax=12, tthr2=4)
    assert diff(spatial_only, temporal, n=1) > 0.0


def test_stride_handling(src8):
    cropped = src8.std.Crop(left=27)
    args = dict(thr=12, tmax=12, tthr2=4)
    assert_same_clip(cropped.vszip.Checkmate(**args), repack(cropped).vszip.Checkmate(**args))


def test_non_8bit_error(temporal_rgb):
    src16 = temporal_rgb.resize.Bilinear(format=vs.GRAY16, matrix=1)
    with pytest.raises(vs.Error, match="only 8 bit int format supported"):
        src16.vszip.Checkmate()


@pytest.mark.parametrize("tmax", [0, 256])
def test_tmax_range_error(src8, tmax):
    with pytest.raises(vs.Error, match=r"tmax value should be in range \[1;255\]"):
        src8.vszip.Checkmate(tmax=tmax)


def test_tthr2_negative_error(src8):
    with pytest.raises(vs.Error, match="tthr2 should be non-negative"):
        src8.vszip.Checkmate(tthr2=-1)


# Every range/bounds/format check in checkmateCreate, with the exact
# error-message substring from src/vapoursynth/checkmate.zig.
@pytest.mark.parametrize(
    ("args", "msg"),
    [
        (dict(tmax=0), r"tmax value should be in range \[1;255\]"),
        (dict(tmax=-5), r"tmax value should be in range \[1;255\]"),
        (dict(tmax=256), r"tmax value should be in range \[1;255\]"),
        (dict(tmax=1000), r"tmax value should be in range \[1;255\]"),
        (dict(tthr2=-1), "tthr2 should be non-negative"),
        (dict(tthr2=-100), "tthr2 should be non-negative"),
    ],
)
def test_validation_errors(src8, args, msg):
    with pytest.raises(vs.Error, match=msg):
        src8.vszip.Checkmate(**args)


@pytest.mark.parametrize("fmt", [vs.GRAY16, vs.GRAYS, vs.YUV420P10, vs.RGB30, vs.GRAYH])
def test_non_8bit_formats_rejected(temporal_rgb, fmt):
    f = vs.core.get_video_format(fmt)
    if f.color_family == vs.RGB:
        src = temporal_rgb.resize.Bilinear(format=fmt)
    else:
        src = temporal_rgb.resize.Bilinear(format=fmt, matrix=1)
        if f.color_family == vs.GRAY:
            src = src.std.RemoveFrameProps("_Matrix")
    with pytest.raises(vs.Error, match="only 8 bit int format supported"):
        src.vszip.Checkmate()


@pytest.mark.parametrize("tmax", [1, 255])
def test_tmax_bounds_accepted(src8, tmax):
    # boundary values of the [1;255] range must NOT raise
    out = src8.vszip.Checkmate(tmax=tmax)
    assert 0.0 < avg(out, n=1) < 1.0


def test_temporal_reads_correct_rows(core):
    """Regression: the tthr2 temporal path read srcp_p2/srcp_n2 from row 0 on
    every processed row (the pointers were never advanced by stride). Build a
    5-frame clip (measure n=2) where n-1 and n+1 are an identical flat 160,
    n is flat 100, and n-2/n+2 match n (100) on every interior row but carry a
    240 row 0. With correct rows the temporal gate holds on the interior, so
    each pixel collapses to (p1 + 2*src + n1) >> 2 = (160+200+160)>>2 = 130.
    The old row-0 bug compared the 240 row 0 against the interior, failed the
    threshold, and fell through to the spatial branch (which leaves flat 100
    unchanged) -> 100 instead of 130."""
    def flat(v, h=16):
        return core.std.BlankClip(None, 32, h, vs.GRAY8, length=1, color=v)

    src = flat(100)
    neighbor = flat(160)  # n-1 and n+1, identical so the |p1-n1| gate passes
    pn2 = core.std.StackVertical([flat(240, 1), flat(100, 15)])  # row 0 != interior
    five = pn2 + neighbor + src + neighbor + pn2
    out = five.vszip.Checkmate(thr=12, tmax=12, tthr2=16)
    interior = plane_stats(out.std.Crop(top=2, bottom=2), n=2)
    assert interior["PlaneStatsMin"] == interior["PlaneStatsMax"] == 130


@pytest.mark.parametrize("h", [1, 2, 3, 4])
def test_tiny_height_errors(core, h):
    """Planes with fewer than 5 rows have no interior row to filter; the loop
    bound h-2 underflowed and the 2-row memcpy read past the plane.
    checkmateCreate now rejects such clips up front (getFrame stays branch-free)."""
    src = core.std.BlankClip(None, 32, h, vs.GRAY8, length=3, color=120)
    for tthr2 in (0, 8):
        with pytest.raises(vs.Error, match="clip too small"):
            src.vszip.Checkmate(thr=12, tmax=12, tthr2=tthr2)


@pytest.mark.parametrize("w", [1, 2])
def test_tiny_width_errors(core, w):
    """The spatial branch reads columns x-2..x+2 and computes width-3, which
    underflowed for width < 3. checkmateCreate rejects narrow clips up front."""
    src = core.std.BlankClip(None, w, 16, vs.GRAY8, length=3, color=120)
    with pytest.raises(vs.Error, match="clip too small"):
        src.vszip.Checkmate()


def test_small_chroma_plane_errors(core):
    """The size check is per plane: a YUV420 luma of 8 rows still leaves a chroma
    plane of only 4 (< 5), so the clip must be rejected."""
    src = core.std.BlankClip(None, 32, 8, vs.YUV420P8, length=3, color=[120, 128, 128])
    with pytest.raises(vs.Error, match="clip too small"):
        src.vszip.Checkmate()


@pytest.mark.parametrize("h", [5, 6])
def test_min_height_accepted(core, h):
    """Boundary: exactly 5 rows is the smallest plane Checkmate accepts."""
    src = core.std.BlankClip(None, 32, h, vs.GRAY8, length=3, color=120)
    out = src.vszip.Checkmate(thr=12, tmax=12, tthr2=8)
    assert (out.width, out.height) == (32, h)
    out.get_frame(1)
