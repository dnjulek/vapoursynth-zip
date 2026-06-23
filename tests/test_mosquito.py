import pytest
import vapoursynth as vs

from golden import Case, grid, sweep
from helpers import assert_same_clip, diff, plane_stats, repack

# MosquitoNR (Wataru Inariba's mosquito-noise reducer) processes luma only;
# chroma is copied untouched. Deterministic fixed-point/float DSP, so goldens
# are stable. Supported: constant-format 8..16 bit integer + 32 bit float,
# YUV or Gray, min 4x4. The 8-bit path is bit-exact to the original SSSE3
# plugin; higher depths are a generalized extension (different `bits` scaling,
# so each depth yields a distinct golden). odd/tiny geometries exercise the
# edge handling and the scalar tail of the hand-vectorized loops.
CASES = (
    sweep(
        base_fmt=vs.GRAY8,
        base_args=dict(strength=16, restore=128, radius=2),
        formats=[vs.GRAY8, vs.GRAY10, vs.GRAY16, vs.GRAYS, vs.YUV420P8, vs.YUV420P16, vs.YUV444PS],
        args=grid(strength=[8, 32]) + grid(restore=[0, 64]) + grid(radius=[1]),
        geometries=["odd", "tiny"],
    )
    + [
        # restore=128 vs the slightly different restore<128 blend, on YUV (chroma
        # copied) so the golden also pins the chroma-passthrough planes
        Case(vs.YUV420P8, args=dict(strength=16, restore=64, radius=1)),
        Case(vs.YUV444P16, args=dict(strength=24, restore=96, radius=2)),
        # 12/14 bit carriers exercise the remaining >8-bit scaling tables
        Case(vs.GRAY12, args=dict(strength=16, restore=128, radius=2)),
        Case(vs.GRAY14, args=dict(strength=32, restore=64, radius=1)),
        # planes selection: process every plane / chroma only (the default
        # leaves chroma as passthrough). The float case also pins the chroma
        # clamp to [-0.5, 0.5] instead of the luma [0, 1] range.
        Case(vs.YUV420P8, args=dict(strength=16, planes=[0, 1, 2])),
        Case(vs.YUV444P16, args=dict(strength=16, planes=[1, 2])),
        Case(vs.YUV444PS, args=dict(strength=24, planes=[0, 1, 2])),
        # per-plane strength/restore/radius arrays (distinct value per plane)
        Case(vs.YUV444P16, args=dict(strength=[16, 8, 24], restore=[128, 64, 96], radius=[2, 1, 2], planes=[0, 1, 2])),
    ]
)


@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_cases(golden, make_clip, case):
    src = make_clip(case.fmt, case.geometry)
    golden.check("mosquito", case, src.vszip.MosquitoNR(**case.args))


@pytest.fixture(scope="module")
def gray8(make_clip):
    return make_clip(vs.GRAY8)


@pytest.fixture(scope="module")
def yuv8(make_clip):
    return make_clip(vs.YUV420P8)


# --- behavioral contract ----------------------------------------------------


def test_strength0_is_exact_passthrough(gray8):
    assert_same_clip(gray8.vszip.MosquitoNR(strength=0), gray8)


def test_luma_processed_chroma_copied(yuv8):
    out = yuv8.vszip.MosquitoNR(strength=16)
    assert diff(out, yuv8, plane=0) > 0.0  # luma filtered
    assert diff(out, yuv8, plane=1) == 0.0  # chroma untouched
    assert diff(out, yuv8, plane=2) == 0.0


def test_default_planes_is_luma_only(yuv8):
    # the default must be identical to explicitly selecting plane 0
    assert_same_clip(yuv8.vszip.MosquitoNR(strength=16), yuv8.vszip.MosquitoNR(strength=16, planes=[0]))


def test_planes_all_processes_chroma(yuv8):
    out = yuv8.vszip.MosquitoNR(strength=16, planes=[0, 1, 2])
    assert diff(out, yuv8, plane=0) > 0.0
    assert diff(out, yuv8, plane=1) > 0.0
    assert diff(out, yuv8, plane=2) > 0.0


def test_planes_chroma_only_leaves_luma(yuv8):
    out = yuv8.vszip.MosquitoNR(strength=16, planes=[1, 2])
    assert diff(out, yuv8, plane=0) == 0.0  # luma copied
    assert diff(out, yuv8, plane=1) > 0.0
    assert diff(out, yuv8, plane=2) > 0.0


def test_per_plane_strength(yuv8):
    # strength is per-plane: filter luma, leave chroma untouched via strength 0
    out = yuv8.vszip.MosquitoNR(strength=[16, 0, 0], planes=[0, 1, 2])
    assert diff(out, yuv8, plane=0) > 0.0
    assert diff(out, yuv8, plane=1) == 0.0  # strength 0 -> passthrough
    assert diff(out, yuv8, plane=2) == 0.0


def test_scalar_matches_uniform_array(yuv8):
    # a scalar arg must equal the same value given as a 3-element array
    args = dict(planes=[0, 1, 2])
    scalar = yuv8.vszip.MosquitoNR(strength=16, restore=64, radius=1, **args)
    array = yuv8.vszip.MosquitoNR(strength=[16, 16, 16], restore=[64, 64, 64], radius=[1, 1, 1], **args)
    assert_same_clip(scalar, array)


def test_array_broadcast_fills_last(yuv8):
    # a short array fills the remaining planes with its last value
    short = yuv8.vszip.MosquitoNR(strength=[16, 8], planes=[0, 1, 2])
    full = yuv8.vszip.MosquitoNR(strength=[16, 8, 8], planes=[0, 1, 2])
    assert_same_clip(short, full)


def test_per_plane_radius_and_restore_change_chroma(yuv8):
    # different radius/restore per plane must actually reach the chroma planes
    a = yuv8.vszip.MosquitoNR(strength=16, radius=[2, 1, 1], restore=[128, 0, 0], planes=[0, 1, 2])
    b = yuv8.vszip.MosquitoNR(strength=16, radius=[2, 2, 2], restore=[128, 128, 128], planes=[0, 1, 2])
    assert diff(a, b, plane=0) == 0.0  # luma args identical
    assert diff(a, b, plane=1) > 0.0
    assert diff(a, b, plane=2) > 0.0


def test_float_chroma_clamped_to_range(make_clip):
    # float chroma is valid in [-0.5, 0.5]; the float core must clamp there,
    # not to the luma [0, 1] range (which would destroy negative chroma).
    out = make_clip(vs.YUV444PS).vszip.MosquitoNR(strength=32, planes=[0, 1, 2])
    for p in (1, 2):
        s = plane_stats(out, plane=p)
        assert s["PlaneStatsMin"] >= -0.5 - 1e-6
        assert s["PlaneStatsMax"] <= 0.5 + 1e-6
    luma = plane_stats(out, plane=0)
    assert luma["PlaneStatsMin"] >= -1e-6
    assert luma["PlaneStatsMax"] <= 1.0 + 1e-6


def test_radius_changes_output(gray8):
    assert diff(gray8.vszip.MosquitoNR(strength=16, radius=1), gray8.vszip.MosquitoNR(strength=16, radius=2)) > 0.0


def test_restore_changes_output(gray8):
    assert diff(gray8.vszip.MosquitoNR(strength=16, restore=0), gray8.vszip.MosquitoNR(strength=16, restore=128)) > 0.0


def test_stride_handling(gray8):
    cropped = gray8.std.Crop(left=27)
    args = dict(strength=16, restore=64, radius=2)
    assert_same_clip(cropped.vszip.MosquitoNR(**args), repack(cropped).vszip.MosquitoNR(**args))


# --- accepted boundary values (must not raise) ------------------------------


@pytest.mark.parametrize("strength", [0, 32])
def test_strength_bounds_accepted(gray8, strength):
    gray8.vszip.MosquitoNR(strength=strength).get_frame(0)


@pytest.mark.parametrize("restore", [0, 128])
def test_restore_bounds_accepted(gray8, restore):
    gray8.vszip.MosquitoNR(strength=16, restore=restore).get_frame(0)


@pytest.mark.parametrize("radius", [1, 2])
def test_radius_bounds_accepted(gray8, radius):
    gray8.vszip.MosquitoNR(strength=16, radius=radius).get_frame(0)


@pytest.mark.parametrize("fmt", [vs.GRAY8, vs.GRAY10, vs.GRAY12, vs.GRAY14, vs.GRAY16, vs.GRAYS])
def test_all_supported_depths_run(make_clip, fmt):
    make_clip(fmt).vszip.MosquitoNR(strength=20, radius=1).get_frame(0)


# --- validation / format rejection ------------------------------------------


@pytest.mark.parametrize("fmt", [vs.RGB24, vs.RGBS])
def test_rgb_rejected(make_clip, fmt):
    with pytest.raises(vs.Error, match="must be YUV or Gray"):
        make_clip(fmt).vszip.MosquitoNR()


@pytest.mark.parametrize("fmt", [vs.GRAYH, vs.YUV420PH])
def test_unsupported_depth_rejected(make_clip, fmt):
    # 16-bit float is neither an 8..16 bit integer nor 32-bit float
    with pytest.raises(vs.Error, match="8..16 bit integer or 32 bit float"):
        make_clip(fmt).vszip.MosquitoNR()


def test_too_small_rejected(core):
    src = core.std.BlankClip(width=3, height=3, format=vs.GRAY8, length=1)
    with pytest.raises(vs.Error, match="too small"):
        src.vszip.MosquitoNR()


def test_chroma_too_small_rejected(core):
    # YUV420 6x6 -> chroma planes are 3x3, below the 4x4 minimum; processing
    # chroma must be rejected, but luma-only on the same clip is fine.
    src = core.std.BlankClip(width=6, height=6, format=vs.YUV420P8, length=1)
    with pytest.raises(vs.Error, match="too small"):
        src.vszip.MosquitoNR(planes=[0, 1, 2])
    src.vszip.MosquitoNR().get_frame(0)


@pytest.mark.parametrize(
    ("args", "msg"),
    [
        (dict(planes=[3]), "out of range"),
        (dict(planes=[-1]), "out of range"),
        (dict(planes=[0, 0]), "specified twice"),
    ],
)
def test_planes_validation_errors(yuv8, args, msg):
    with pytest.raises(vs.Error, match=msg):
        yuv8.vszip.MosquitoNR(**args)


def test_plane_out_of_range_on_gray(gray8):
    with pytest.raises(vs.Error, match="out of range"):
        gray8.vszip.MosquitoNR(planes=[1])


@pytest.mark.parametrize(
    ("args", "msg"),
    [
        (dict(strength=-1), "strength value -1 is below minimum 0"),
        (dict(strength=33), "strength value 33 is above maximum 32"),
        (dict(restore=-1), "restore value -1 is below minimum 0"),
        (dict(restore=129), "restore value 129 is above maximum 128"),
        (dict(radius=0), "radius value 0 is below minimum 1"),
        (dict(radius=3), "radius value 3 is above maximum 2"),
        # out-of-range in any array slot is rejected, not just slot 0
        (dict(strength=[16, 33, 16]), "strength value 33 is above maximum 32"),
        (dict(radius=[2, 2, 0]), "radius value 0 is below minimum 1"),
        # more than 3 elements (one per plane) is rejected, not silently dropped
        (dict(strength=[16, 16, 16, 99]), "strength has too many elements"),
        (dict(restore=[0, 0, 0, 0]), "restore has too many elements"),
    ],
)
def test_validation_errors(gray8, args, msg):
    with pytest.raises(vs.Error, match=msg):
        gray8.vszip.MosquitoNR(**args)
