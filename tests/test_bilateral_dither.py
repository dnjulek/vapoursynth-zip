import pytest
import vapoursynth as vs

from golden import Case, grid, sweep
from helpers import assert_same_clip, diff, plane_stats, repack

# BilateralDither (Zig SIMD port of dither's Dither_bilateral16). Edge-preserving
# bilateral: each output pixel is a window average weighted by value closeness
# (thr/flat triangular-to-box weight) with a wmin flat-area floor. All math is
# f32 over a mirror-padded cache; results are deterministic (including the seeded
# blue-noise/spiral sub-sampling point lists), so goldens are stable. Supports
# constant-format 8..16 bit integer + 32 bit float, any color family; every
# plane is processed (subsampled chroma uses a reduced radius). Minimum size is
# 16x16 and width/height must be >= radius, so the "tiny" geometry doesn't apply.
CASES = (
    sweep(
        base_fmt=vs.GRAY8,
        base_args=dict(radius=8, thr=8.0, flat=0.4, subspl=2.0),  # dense full window
        formats=[vs.GRAY8, vs.GRAY16, vs.GRAYS, vs.YUV420P8, vs.YUV420P16, vs.YUV444PS, vs.RGB24],
        args=(
            grid(subspl=[0.0, 8.0])  # auto sub-sample (default) + explicit sub-sample
            + grid(flat=[0.0, 1.0])  # triangular peak vs box (hard threshold)
            + grid(thr=[2.5, 24.0])
            + grid(wmin=[0.5])  # flat-area protection
            + grid(radius=[4])
        ),
        geometries=["odd"],
    )
    + [
        # subsampled chroma drives the reduced chroma radius (auto subspl path)
        Case(vs.YUV420P16, args=dict(radius=8, thr=8.0, subspl=0.0)),
        # k >= 32 exercises the VoidAndCluster blue-noise point-list path
        Case(vs.GRAYS, args=dict(radius=12, thr=16.0, flat=0.0, subspl=16.0)),
        # RGB (no subsampling): all planes get the full radius
        Case(vs.RGBS, args=dict(radius=6, thr=8.0, subspl=2.0)),
        # per-plane arrays: distinct radius/thr/flat per plane (YUV444 so every
        # plane is full-size and the values apply directly)
        Case(vs.YUV444P16, args=dict(radius=[8, 4, 6], thr=[8.0, 16.0, 4.0], flat=[0.0, 0.4, 1.0], subspl=2.0)),
        # subsampled chroma with an explicit, reduced chroma radius
        Case(vs.YUV420P8, args=dict(radius=[8, 4, 4], thr=[8.0, 12.0, 12.0], subspl=2.0)),
        # plane selection: luma only / chroma only (the unprocessed planes are
        # copied straight from source)
        Case(vs.YUV420P16, args=dict(radius=8, thr=12.0, subspl=2.0, planes=[0])),
        Case(vs.YUV444PS, args=dict(radius=6, thr=16.0, subspl=2.0, planes=[1, 2])),
    ]
)


@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_cases(golden, make_clip, case):
    src = make_clip(case.fmt, case.geometry)
    golden.check("bilateral_dither", case, src.vszip.BilateralDither(**case.args))


@pytest.fixture(scope="module")
def gray16(make_clip):
    return make_clip(vs.GRAY16)


@pytest.fixture(scope="module")
def yuv16(make_clip):
    return make_clip(vs.YUV420P16)


# --- behavioral contract ----------------------------------------------------


def test_filter_changes_clip(gray16):
    assert diff(gray16.vszip.BilateralDither(radius=8, thr=16.0, subspl=2.0), gray16) > 0.0


def test_higher_thr_smooths_more(gray16):
    # a larger range threshold admits more neighbours into the average, pulling
    # the output further from the source
    lo = gray16.vszip.BilateralDither(radius=8, thr=1.0, subspl=2.0)
    hi = gray16.vszip.BilateralDither(radius=8, thr=64.0, subspl=2.0)
    assert diff(hi, gray16) > diff(lo, gray16)


def test_all_planes_processed(yuv16):
    out = yuv16.vszip.BilateralDither(radius=8, thr=16.0, subspl=2.0)
    for p in range(3):
        assert diff(out, yuv16, plane=p) > 0.0  # luma and both chroma planes filtered


def test_scalar_matches_uniform_array(yuv16):
    # a scalar arg must equal the same value given as a 3-element array
    scalar = dict(radius=8, thr=8.0, flat=0.4, wmin=0.5, subspl=2.0)
    array = dict(radius=[8, 8, 8], thr=[8.0, 8.0, 8.0], flat=[0.4, 0.4, 0.4], wmin=[0.5, 0.5, 0.5], subspl=[2.0, 2.0, 2.0])
    assert_same_clip(yuv16.vszip.BilateralDither(**scalar), yuv16.vszip.BilateralDither(**array))


def test_per_plane_radius_changes_chroma_only(yuv16):
    base = yuv16.vszip.BilateralDither(radius=8, thr=8.0, subspl=2.0)
    perp = yuv16.vszip.BilateralDither(radius=[8, 4, 4], thr=8.0, subspl=2.0)
    assert diff(base, perp, plane=0) == 0.0  # luma radius unchanged
    assert diff(base, perp, plane=1) > 0.0  # chroma radius differs
    assert diff(base, perp, plane=2) > 0.0


def test_per_plane_thr(yuv16):
    # tiny luma thr (near passthrough) but a strong chroma thr -> chroma moves more
    out = yuv16.vszip.BilateralDither(thr=[0.001, 24.0, 24.0], radius=6, subspl=2.0)
    assert diff(out, yuv16, plane=0) < diff(out, yuv16, plane=1)


def test_default_processes_all_planes(yuv16):
    # the default must equal explicitly selecting every plane
    assert_same_clip(
        yuv16.vszip.BilateralDither(radius=8, thr=8.0, subspl=2.0),
        yuv16.vszip.BilateralDither(radius=8, thr=8.0, subspl=2.0, planes=[0, 1, 2]),
    )


def test_planes_luma_only(yuv16):
    out = yuv16.vszip.BilateralDither(radius=8, thr=16.0, subspl=2.0, planes=[0])
    assert diff(out, yuv16, plane=0) > 0.0  # luma filtered
    assert diff(out, yuv16, plane=1) == 0.0  # chroma copied
    assert diff(out, yuv16, plane=2) == 0.0


def test_planes_chroma_only(yuv16):
    out = yuv16.vszip.BilateralDither(radius=8, thr=16.0, subspl=2.0, planes=[1, 2])
    assert diff(out, yuv16, plane=0) == 0.0  # luma copied
    assert diff(out, yuv16, plane=1) > 0.0
    assert diff(out, yuv16, plane=2) > 0.0


def test_unprocessed_plane_skips_radius_check(core):
    # a chroma radius too large for the (subsampled) chroma plane is rejected
    # only when chroma is actually processed
    src = core.std.BlankClip(width=32, height=32, format=vs.YUV420P8, length=1)  # chroma 16x16
    with pytest.raises(vs.Error, match="greater than"):
        src.vszip.BilateralDither(radius=20)
    src.vszip.BilateralDither(radius=20, planes=[0]).get_frame(0)  # luma only -> ok


def test_dense_vs_subsampled_differ(gray16):
    dense = gray16.vszip.BilateralDither(radius=8, thr=8.0, subspl=2.0)
    sub = gray16.vszip.BilateralDither(radius=8, thr=8.0, subspl=8.0)
    assert diff(dense, sub) > 0.0


def test_ref_equal_src_matches_no_ref(gray16):
    # weighting from the source itself is the classic bilateral (ref == src)
    assert_same_clip(
        gray16.vszip.BilateralDither(ref=gray16, radius=8, thr=8.0, subspl=2.0),
        gray16.vszip.BilateralDither(radius=8, thr=8.0, subspl=2.0),
    )


def test_ref_changes_output(gray16):
    # a blurred ref changes the local value differences that drive the weights,
    # so the output differs from the classic (ref == src) result
    ref = gray16.std.BoxBlur(hradius=8, vradius=8)
    out_ref = gray16.vszip.BilateralDither(ref=ref, radius=8, thr=8.0, subspl=2.0)
    out_noref = gray16.vszip.BilateralDither(radius=8, thr=8.0, subspl=2.0)
    assert diff(out_ref, out_noref) > 0.0


def test_stride_handling(gray16):
    cropped = gray16.std.Crop(left=19)
    args = dict(radius=8, thr=8.0, subspl=2.0)
    assert_same_clip(cropped.vszip.BilateralDither(**args), repack(cropped).vszip.BilateralDither(**args))


def test_float_chroma_stays_in_range(make_clip):
    # float output is a weighted average of in-range neighbours, so chroma stays
    # within [-0.5, 0.5] (and luma within [0, 1]) with no clamp needed
    out = make_clip(vs.YUV444PS).vszip.BilateralDither(radius=8, thr=24.0, subspl=2.0)
    for p in (1, 2):
        s = plane_stats(out, plane=p)
        assert s["PlaneStatsMin"] >= -0.5 - 1e-6
        assert s["PlaneStatsMax"] <= 0.5 + 1e-6
    luma = plane_stats(out, plane=0)
    assert luma["PlaneStatsMin"] >= -1e-6
    assert luma["PlaneStatsMax"] <= 1.0 + 1e-6


@pytest.mark.parametrize(
    "fmt",
    [vs.GRAY8, vs.GRAY10, vs.GRAY12, vs.GRAY14, vs.GRAY16, vs.GRAYS, vs.YUV422P10, vs.RGB24, vs.RGBS],
)
def test_all_formats_run(make_clip, fmt):
    make_clip(fmt).vszip.BilateralDither(radius=6, thr=8.0).get_frame(0)


# --- validation / format rejection ------------------------------------------


def test_too_small_rejected(core):
    src = core.std.BlankClip(width=15, height=15, format=vs.GRAY8, length=1)
    with pytest.raises(vs.Error, match="16x16 min"):
        src.vszip.BilateralDither()


def test_radius_bigger_than_picture_rejected(core):
    src = core.std.BlankClip(width=20, height=20, format=vs.GRAY8, length=1)
    with pytest.raises(vs.Error, match="greater than"):
        src.vszip.BilateralDither(radius=24)


def test_ref_format_mismatch_rejected(gray16, make_clip):
    with pytest.raises(vs.Error, match="same format and dimensions"):
        gray16.vszip.BilateralDither(ref=make_clip(vs.GRAY8))


@pytest.mark.parametrize(
    ("args", "msg"),
    [
        (dict(radius=1), "radius"),
        (dict(thr=-1.0), "thr"),
        (dict(flat=-0.1), "flat"),
        (dict(flat=1.1), "flat"),
        (dict(wmin=-1.0), "wmin"),
    ],
)
def test_validation_errors(gray16, args, msg):
    with pytest.raises(vs.Error, match=msg):
        gray16.vszip.BilateralDither(**args)


@pytest.mark.parametrize(
    ("args", "msg"),
    [
        # more than one value per plane (max 3) is rejected, not silently dropped
        (dict(radius=[8, 8, 8, 8]), "radius has too many elements"),
        (dict(thr=[1.0, 2.0, 3.0, 4.0]), "thr has too many elements"),
        # out-of-range in any array slot is rejected, not just slot 0
        (dict(radius=[8, 1, 8]), "radius value 1 is below minimum"),
        (dict(flat=[0.4, 1.5, 0.4]), "flat value 1.5 is above maximum"),
    ],
)
def test_array_validation_errors(gray16, args, msg):
    with pytest.raises(vs.Error, match=msg):
        gray16.vszip.BilateralDither(**args)


@pytest.mark.parametrize(
    ("args", "msg"),
    [
        (dict(planes=[3]), "plane index out of range"),
        (dict(planes=[0, 0]), "plane specified twice"),
    ],
)
def test_planes_validation_errors(yuv16, args, msg):
    with pytest.raises(vs.Error, match=msg):
        yuv16.vszip.BilateralDither(**args)
