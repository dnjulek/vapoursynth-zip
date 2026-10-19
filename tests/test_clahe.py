import pytest
import vapoursynth as vs

from golden import Case, grid, sweep
from helpers import assert_same_clip, avg, plane_stats, repack

# CLAHE accepts every 8..16-bit integer and f16/f32 clip (integer clips bin by
# their own bit depth; float is equalized on 16-bit bins, see clahe_ref.py),
# regardless of color family, and getFrame applies the same equalization to
# every plane. So Gray, YUV (all subsamplings) and RGB are all
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
            vs.GRAY10,
            vs.YUV420P10,
            vs.GRAYS,
            vs.YUV444PS,
            vs.RGBS,
            vs.GRAYH,
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


@pytest.mark.parametrize("fmt", [vs.GRAY8, vs.GRAY10, vs.GRAY12, vs.GRAY16, vs.GRAYH, vs.GRAYS])
def test_formats_run(to_gray, fmt):
    src = to_gray(fmt)
    out = src.vszip.CLAHE()
    assert out.format.id == src.format.id
    # std.PlaneStats has no f16 path: measure GRAYH through f32
    meas = out.resize.Point(format=vs.GRAYS) if fmt == vs.GRAYH else out
    assert 0.0 < avg(meas) < 1.0


@pytest.fixture(scope="module")
def np():
    return pytest.importorskip("numpy")


def _plane(np, clip, p):
    return np.asarray(clip.get_frame(0)[p]).copy()


ORACLE = [
    (vs.GRAY8, 8, dict(limit=4, tiles=3)),
    (vs.GRAY10, 10, dict(limit=4, tiles=3)),
    (vs.GRAY10, 10, dict(limit=40, tiles=[5, 2])),
    (vs.GRAY12, 12, dict(limit=7, tiles=[2, 4])),
    (vs.GRAY14, 14, dict(limit=300, tiles=4)),
    (vs.GRAY16, 16, dict(limit=2560, tiles=[3, 5])),
    (vs.GRAYS, 16, dict(limit=7, tiles=3)),
    (vs.GRAYS, 16, dict(limit=2560, tiles=[4, 2])),
    (vs.GRAYH, 16, dict(limit=7, tiles=3)),
]


@pytest.mark.parametrize(("fmt", "bits", "args"), ORACLE, ids=lambda v: str(v))
@pytest.mark.parametrize("geo", [(197, 113, 300, 100), (64, 37, 10, 20)], ids=["odd", "small"])
def test_matches_reference(np, src_rgb, fmt, bits, args, geo):
    """Bit-exact against the NumPy reference for every bin count (tails, tile
    remainder strips, 9..15-bit bins and float quantization included)."""
    from clahe_ref import clahe_plane

    w, h, x, y = geo
    src = src_rgb.std.CropAbs(width=w, height=h, left=x, top=y).resize.Bilinear(format=fmt, matrix=1).std.RemoveFrameProps("_Matrix")
    fobj = src.format
    is_float = fobj.sample_type == vs.FLOAT
    tiles = args["tiles"] if isinstance(args["tiles"], list) else [args["tiles"]] * 2
    got = _plane(np, src.vszip.CLAHE(**args), 0)
    want = clahe_plane(_plane(np, src, 0), bits, args["limit"], tiles, is_float=is_float, out_dtype=got.dtype)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("fmt", [vs.YUV444PS, vs.YUV420PS, vs.YUV420P10, vs.YUV444PH])
def test_matches_reference_yuv(np, src_rgb, fmt):
    """Every plane, with float chroma offset by +0.5 into the [0, 1] bins."""
    from clahe_ref import clahe_plane

    src = src_rgb.std.CropAbs(width=150, height=90, left=200, top=150).resize.Bilinear(format=fmt, matrix=1)
    fobj = src.format
    is_float = fobj.sample_type == vs.FLOAT
    bits = 16 if is_float else fobj.bits_per_sample
    out = src.vszip.CLAHE(limit=9, tiles=[3, 2])
    for p in range(3):
        got = _plane(np, out, p)
        want = clahe_plane(_plane(np, src, p), bits, 9, [3, 2], is_float=is_float, off=0.5 if (is_float and p) else 0.0, out_dtype=got.dtype)
        np.testing.assert_array_equal(got, want, err_msg=f"plane {p}")


def test_float_equals_16bit_path(np, to_gray):
    """f32 samples that sit exactly on 16-bit codes take the same bins and
    LUTs as the GRAY16 clip; the float output is the unrounded value of the
    integer one (within half a 16-bit step)."""
    g16 = to_gray(vs.GRAY16)
    gs = g16.std.Expr("x 65535 /", format=vs.GRAYS)
    i = _plane(np, g16.vszip.CLAHE(limit=15, tiles=[3, 4]), 0).astype(np.float64)
    f = _plane(np, gs.vszip.CLAHE(limit=15, tiles=[3, 4]), 0).astype(np.float64) * 65535
    # + f32 rounding of res / 65535 (~6e-8 relative)
    assert np.abs(f - i).max() <= 0.5 + 0.01


def test_stray_samples_above_peak_are_safe(np, to_gray):
    """10-bit clips holding values above 1023 (invalid but representable)
    bin as the peak instead of indexing past their LUT row."""
    src = to_gray(vs.GRAY10)
    stray = src.std.Expr("x 512 < x 60000 ?")  # half the plane far above peak
    out = _plane(np, stray.vszip.CLAHE(limit=4, tiles=3), 0)
    assert out.max() <= 1023
    np.testing.assert_array_equal(out, _plane(np, stray.std.Expr("x 1023 min").vszip.CLAHE(limit=4, tiles=3), 0))


def test_float_nan_and_out_of_range_are_safe(np, to_gray):
    """NaN bins as 0 and values outside [0, 1] clamp to the end bins."""
    src = to_gray(vs.GRAYS)
    odd = src.std.Expr("x 0.2 < 0 0 / x 0.8 > x 3 * x 0.5 < x 1 - x ? ? ?")
    out = _plane(np, odd.vszip.CLAHE(), 0)
    assert np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1


def test_equalization_increases_contrast(to_gray):
    """CLAHE on a low-contrast clip must widen the value range."""
    src = to_gray(vs.GRAY16).std.Expr("x 4 / 16384 +")  # squeeze into [16384, 32768)
    stats = plane_stats(src.vszip.CLAHE(limit=2560, tiles=4))
    assert stats["PlaneStatsMax"] > 60000
    assert stats["PlaneStatsMin"] < 4000


def test_unsupported_formats_rejected(core):
    """32-bit integer is the only rejected sample format."""
    src = core.std.BlankClip(None, 64, 64, vs.GRAY32, length=1)
    with pytest.raises(vs.Error, match="not supported Int format"):
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
        # rejected sample type (32-bit integer)
        (vs.GRAY32, dict(), "not supported Int format"),
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


NONDIV_GRIDS = [
    (vs.GRAY8, 100, 70, [3, 3]),
    (vs.GRAY8, 101, 67, [7, 9]),
    (vs.GRAY8, 65, 33, [64, 32]),  # 1-pixel tiles plus a 1-pixel remainder
    (vs.GRAY16, 99, 61, [8, 8]),
    (vs.GRAY16, 127, 63, [11, 13]),
    (vs.YUV420P8, 102, 70, [5, 7]),
    (vs.YUV444P16, 77, 41, [6, 4]),
]


@pytest.mark.parametrize(("fmt", "w", "h", "tiles"), NONDIV_GRIDS, ids=lambda v: str(v))
def test_tiles_not_dividing_frame(core, src_rgb, fmt, w, h, tiles):
    """Tile grids that leave right/bottom remainder strips (formerly an
    out-of-bounds read). Constant input must map to one constant everywhere,
    remainder strips included, and real content must run and be deterministic."""
    fobj = core.get_video_format(fmt)
    peak = (1 << fobj.bits_per_sample) - 1
    flat = core.std.BlankClip(None, w, h, fmt, length=1, color=[peak // 3] * fobj.num_planes)
    out = flat.vszip.CLAHE(limit=5, tiles=tiles)
    for p in range(fobj.num_planes):
        s = plane_stats(out, plane=p)
        assert s["PlaneStatsMin"] == s["PlaneStatsMax"], f"plane {p} not constant: {s}"

    real = src_rgb.std.CropAbs(width=w, height=h, left=300, top=200).resize.Bilinear(format=fmt, matrix=1)
    a = real.vszip.CLAHE(limit=5, tiles=tiles)
    b = real.vszip.CLAHE(limit=5, tiles=tiles)
    assert_same_clip(a, b)
