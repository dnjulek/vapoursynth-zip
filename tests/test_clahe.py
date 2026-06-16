import pytest
import vapoursynth as vs

from golden import Case, grid, sweep
from helpers import assert_same_clip, avg, plane_stats, repack

# CLAHE accepts any 8- or 16-bit integer clip: the Create callback checks only
# sampleType + bitsPerSample, not color family, and getFrame applies the same
# equalization to every plane. So Gray, YUV (all subsamplings) and RGB are all
# valid paths and are swept here. `limit` scales by tile_size_total / hist_size
# and is integer-truncated, so on 16-bit small limits collapse to the same
# clip_limit; the {2,4,10} sweep is therefore anchored on GRAY8 (where they're
# all distinct) and 16-bit limit coverage is supplied via the hand-picked cases
# with scaled limits.
CASES = (
    sweep(
        base_fmt=vs.GRAY8,
        base_args=dict(limit=4, tiles=3),
        formats=[
            vs.GRAY8,
            vs.GRAY16,
            vs.YUV420P8,
            vs.YUV444P8,
            vs.YUV420P16,
            vs.YUV444P16,
            vs.RGB24,
            vs.RGB48,
        ],
        args=grid(limit=[2, 4, 10])
        + [
            dict(tiles=2),
            dict(tiles=8),
            dict(tiles=[2, 4]),
            dict(tiles=[8, 2]),
            dict(tiles=[4, 8]),
        ],
        geometries=["odd", "tiny"],
    )
    + [
        # 16-bit limit coverage needs scaled limits to be distinguishable
        Case(vs.GRAY16, args=dict(limit=512, tiles=4)),
        Case(vs.GRAY16, args=dict(limit=1024, tiles=4)),
        Case(vs.GRAY16, args=dict(limit=2560, tiles=4)),
        # limit x non-square tiles interaction on 16-bit
        Case(vs.GRAY16, args=dict(limit=2560, tiles=[8, 2])),
        Case(vs.GRAY16, args=dict(limit=2560, tiles=[2, 8])),
        # asymmetric tiles whose mirror image differs (transpose sensitivity)
        Case(vs.GRAY8, args=dict(limit=4, tiles=[3, 2])),
        Case(vs.GRAY8, args=dict(limit=4, tiles=[2, 3])),
        Case(vs.GRAY8, args=dict(limit=4, tiles=4)),
        # limit x tiles interaction on 8-bit
        Case(vs.GRAY8, args=dict(limit=2, tiles=8)),
        # YUV with non-square tiles exercises chroma-plane processing
        Case(vs.YUV420P8, args=dict(limit=10, tiles=[4, 8])),
        Case(vs.YUV420P8, args=dict(limit=2, tiles=2)),
        Case(vs.YUV420P16, args=dict(limit=1024, tiles=[8, 2])),
        Case(vs.YUV444P8, args=dict(limit=2, tiles=[2, 4])),
        Case(vs.YUV444P16, args=dict(limit=2560, tiles=[2, 8])),
        # geometry variants on 16-bit and YUV (odd -> SIMD tail, tiny -> scalar)
        Case(vs.GRAY16, "odd", args=dict(limit=2560, tiles=4)),
        Case(vs.GRAY16, "tiny", args=dict(limit=4, tiles=3)),
        Case(vs.YUV420P16, "odd", args=dict(limit=4, tiles=3)),
        Case(vs.YUV420P16, "tiny", args=dict(limit=4, tiles=3)),
        Case(vs.YUV444P16, "odd", args=dict(limit=4, tiles=3)),
        Case(vs.YUV420P8, "tiny", args=dict(limit=4, tiles=3)),
        # RGB is an accepted color family too (Create checks only sampleType +
        # bitsPerSample); all 3 planes get equalized. Exercise limit/tiles moving
        # output on the multi-plane RGB path plus odd/tiny tails.
        Case(vs.RGB24, args=dict(limit=10, tiles=[4, 8])),
        Case(vs.RGB24, args=dict(limit=2, tiles=2)),
        Case(vs.RGB48, args=dict(limit=2560, tiles=[8, 2])),
        Case(vs.RGB24, "odd", args=dict(limit=4, tiles=3)),
        Case(vs.RGB48, "tiny", args=dict(limit=4, tiles=3)),
    ]
)


@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_cases(golden, make_clip, case):
    src = make_clip(case.fmt, case.geometry)
    golden.check("clahe", case, src.vszip.CLAHE(**case.args))


def test_golden_gray16(to_gray):
    out = to_gray(vs.GRAY16).vszip.CLAHE(limit=15, tiles=3)
    assert avg(out) == pytest.approx(0.6039824106244517, rel=1e-6)


def test_tiles_default_and_pair_equivalence(to_gray):
    """tiles defaults to 3 and an int equals the [n, n] pair."""
    src = to_gray(vs.GRAY16)
    ref = src.vszip.CLAHE(limit=15, tiles=3)
    assert_same_clip(src.vszip.CLAHE(limit=15), ref)
    assert_same_clip(src.vszip.CLAHE(limit=15, tiles=[3, 3]), ref)


@pytest.mark.parametrize("fmt", [vs.GRAY8, vs.GRAY16])
def test_formats_run(to_gray, fmt):
    src = to_gray(fmt)
    out = src.vszip.CLAHE()
    assert out.format.id == src.format.id
    assert 0.0 < avg(out) < 1.0


def test_equalization_increases_contrast(to_gray):
    """CLAHE on a low-contrast clip must widen the value range."""
    src = to_gray(vs.GRAY16).std.Expr("x 4 / 16384 +")  # squeeze into [16384, 32768)
    stats = plane_stats(src.vszip.CLAHE(limit=2560, tiles=4))
    assert stats["PlaneStatsMax"] > 60000
    assert stats["PlaneStatsMin"] < 4000


@pytest.mark.parametrize("fmt", [vs.GRAY10, vs.GRAY12, vs.GRAY32, vs.GRAYS])
def test_unsupported_formats_rejected(core, fmt):
    """Only 8- and 16-bit integer clips are accepted (10-bit used to be let
    through and was equalized to the 16-bit peak; 32-bit was reinterpreted)."""
    src = core.std.BlankClip(None, 64, 64, fmt, length=1)
    with pytest.raises(vs.Error, match="only 8 or 16 bit int formats supported"):
        src.vszip.CLAHE()


def test_stride_handling(to_gray):
    cropped = to_gray(vs.GRAY16).std.Crop(left=27)
    assert_same_clip(cropped.vszip.CLAHE(limit=15), repack(cropped).vszip.CLAHE(limit=15))


def test_tiles_too_many_error(to_gray):
    with pytest.raises(vs.Error, match="tiles array can't have more than 2 values"):
        to_gray(vs.GRAY16).vszip.CLAHE(tiles=[2, 2, 2])


@pytest.mark.parametrize(
    ("fmt", "args", "msg"),
    [
        # rejected sample types / bit depths (float + non-8/16 integer)
        (vs.GRAY10, dict(), "only 8 or 16 bit int formats supported"),
        (vs.GRAY12, dict(), "only 8 or 16 bit int formats supported"),
        (vs.GRAY14, dict(), "only 8 or 16 bit int formats supported"),
        (vs.GRAY32, dict(), "only 8 or 16 bit int formats supported"),
        (vs.GRAYH, dict(), "only 8 or 16 bit int formats supported"),
        (vs.GRAYS, dict(), "only 8 or 16 bit int formats supported"),
        (vs.YUV420P10, dict(), "only 8 or 16 bit int formats supported"),
        (vs.YUV444PS, dict(), "only 8 or 16 bit int formats supported"),
        # tiles array bound (max 2 values)
        (vs.GRAY16, dict(tiles=[2, 2, 2]), "tiles array can't have more than 2 values"),
        (vs.GRAY16, dict(tiles=[1, 2, 3, 4]), "tiles array can't have more than 2 values"),
    ],
)
def test_validation_errors(core, fmt, args, msg):
    src = core.std.BlankClip(None, 64, 64, fmt, length=1)
    with pytest.raises(vs.Error, match=msg):
        src.vszip.CLAHE(**args)


def test_clip_limit_large_frame_ok(core):
    """A sane limit on a large 16-bit frame computes clip_limit without overflow."""
    big = core.std.BlankClip(None, 1920, 1088, vs.GRAY16, length=1, color=30000)
    big.vszip.CLAHE(limit=7, tiles=[3]).get_frame(0)


def test_clip_limit_too_big_errors(core):
    """Regression: clip_limit = limit * tile_area / hist_size overflowed i32 on
    16-bit large frames with an absurd limit (Debug panic / OOB in Release). It is
    now rejected at Create instead of being handled per-frame."""
    big = core.std.BlankClip(None, 1920, 1088, vs.GRAY16, length=1, color=30000)
    with pytest.raises(vs.Error, match="limit too large"):
        big.vszip.CLAHE(limit=4_000_000_000, tiles=[3])
