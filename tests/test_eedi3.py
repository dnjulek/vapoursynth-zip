import pytest
import vapoursynth as vs

from golden import Case, grid, sweep
from helpers import assert_same_clip, diff, max_abs_diff, repack

# EEDI3 / EEDI3H (Zig SIMD port of eedi3m's float EEDI3). Edge-directed
# interpolation: the missing field is reconstructed by a DP that finds the
# best non-crossing warping between neighbour lines. EEDI3 interpolates rows
# (vertical), EEDI3H interpolates columns (horizontal) by running the very same
# kernel over a transposed copy, so EEDI3H is bit-exact to T+EEDI3+T. The plugin
# is 32-bit-float only and works on any planar/gray format; every plane is
# interpolated independently. All math is deterministic f32, so goldens are
# stable. Output differs from eedi3m only by a documented, accepted edge-cost
# divergence (a handful of border pixels) and by `hp` actually doing half-pel
# (eedi3m's `hp` is a no-op).
#
# dh=False requires the interpolated axis (height for EEDI3, width for EEDI3H)
# to be mod 2, so the base 640x320 "full" geometry is used for goldens; the
# "odd"/"tiny" geometries would make that axis odd and are rejected.
FLOAT_FMTS = [vs.GRAYS, vs.YUV420PS, vs.YUV444PS, vs.RGBS]

CASES = (
    sweep(
        base_fmt=vs.GRAYS,
        base_args=dict(field=1),
        formats=FLOAT_FMTS,
        args=(
            grid(field=[0])
            + grid(dh=[True])  # double height; field stays 0/1
            + grid(nrad=[0, 3], mdis=[40])
            + grid(hp=[True])  # real half-pel (eedi3m ignores hp)
            + grid(vcheck=[0, 1, 3])
            + grid(alpha=[0.4], beta=[0.3], gamma=[40.0])
            + grid(gamma=[0.0])
        ),
    )
    + [
        # double-rate (bob): output has 2x frames, frame 0 is interpolated
        Case(vs.GRAYS, args=dict(field=2)),
        Case(vs.YUV420PS, args=dict(field=3, dh=False)),
        # strong edge connection
        Case(vs.GRAYS, args=dict(field=1, alpha=0.9, beta=0.05, gamma=2.0, mdis=30)),
    ]
)

# EEDI3H mirrors EEDI3 but on the width axis (640 is even, so dh=False is fine).
CASES_H = (
    sweep(
        base_fmt=vs.GRAYS,
        base_args=dict(field=1),
        formats=FLOAT_FMTS,
        args=(
            grid(field=[0])
            + grid(dh=[True])
            + grid(nrad=[3], mdis=[40])
            + grid(hp=[True])
            + grid(vcheck=[0, 3])
        ),
    )
    + [Case(vs.GRAYS, args=dict(field=2))]
)


@pytest.fixture(scope="module")
def grays(make_clip):
    return make_clip(vs.GRAYS)


@pytest.fixture(scope="module")
def yuv(make_clip):
    return make_clip(vs.YUV444PS)


@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_eedi3(golden, make_clip, case):
    src = make_clip(case.fmt, case.geometry)
    golden.check("eedi3", case, src.vszip.EEDI3(**case.args))


@pytest.mark.parametrize("case", CASES_H, ids=str)
def test_golden_eedi3h(golden, make_clip, case):
    src = make_clip(case.fmt, case.geometry)
    golden.check("eedi3h", case, src.vszip.EEDI3H(**case.args))


# --- behavioral contract ----------------------------------------------------


def test_field_doubles_height(grays):
    out = grays.vszip.EEDI3(field=1, dh=True)
    assert (out.width, out.height) == (grays.width, grays.height * 2)


def test_dh_false_keeps_dimensions(grays):
    out = grays.vszip.EEDI3(field=1)
    assert (out.width, out.height) == (grays.width, grays.height)


def test_double_rate_doubles_frames(grays):
    clip = grays.std.Loop(4)  # 4 frames -> 8 after bobbing
    out = clip.vszip.EEDI3(field=2)
    assert out.num_frames == clip.num_frames * 2


def test_eedi3h_doubles_width(grays):
    out = grays.vszip.EEDI3H(field=1, dh=True)
    assert (out.width, out.height) == (grays.width * 2, grays.height)


def test_eedi3h_matches_transpose_eedi3(grays):
    # EEDI3H runs the identical kernel on a transposed copy, so it is bit-exact
    # to Transpose -> EEDI3 -> Transpose for every option combination.
    for kw in (dict(field=1), dict(field=0, vcheck=0), dict(field=1, dh=True),
               dict(field=1, hp=True, vcheck=3), dict(field=1, nrad=3, mdis=40)):
        h = grays.vszip.EEDI3H(**kw)
        t = grays.std.Transpose().vszip.EEDI3(**kw).std.Transpose()
        assert max_abs_diff(h, t) == 0.0, kw


def test_all_planes_processed(yuv):
    out = yuv.vszip.EEDI3(field=1)
    for p in range(3):
        assert diff(out, yuv, plane=p) > 0.0  # every plane interpolated


def test_higher_mdis_changes_output(grays):
    lo = grays.vszip.EEDI3(field=1, mdis=1)
    hi = grays.vszip.EEDI3(field=1, mdis=40)
    assert max_abs_diff(lo, hi) > 0.0


def test_hp_is_implemented(grays):
    # unlike eedi3m (where hp is a no-op), vszip actually does half-pel steps
    assert max_abs_diff(grays.vszip.EEDI3(field=1, hp=True), grays.vszip.EEDI3(field=1, hp=False)) > 0.0


def test_vcheck_changes_output(grays):
    assert max_abs_diff(grays.vszip.EEDI3(field=1, vcheck=0), grays.vszip.EEDI3(field=1, vcheck=3)) > 0.0


def test_float_output_is_finite(make_clip):
    # the 4-tap cubic legitimately overshoots the nominal range (ringing), like
    # eedi3m, and is intentionally not clamped; just assert nothing blows up.
    out = make_clip(vs.YUV444PS).vszip.EEDI3(field=1)
    for p in range(3):
        s = out.std.PlaneStats(plane=p).get_frame(0).props
        assert -2.0 < s["PlaneStatsMin"] <= s["PlaneStatsMax"] < 2.0


def test_stride_handling(grays):
    # odd width (cropped) exercises the scalar tail of the per-line kernel; the
    # height stays even so dh=False is valid.
    cropped = grays.std.Crop(left=19)
    out_a = cropped.vszip.EEDI3(field=1, mdis=10)
    out_b = repack(cropped).vszip.EEDI3(field=1, mdis=10)
    assert_same_clip(out_a, out_b)


# --- sclip / mclip ----------------------------------------------------------


def test_sclip_changes_vcheck_output(grays):
    # with vcheck>0 the reliability check blends toward `cint`; sclip supplies a
    # custom cint, so a non-trivial sclip changes the result.
    sclip = grays.std.BoxBlur(hradius=4, vradius=4)
    with_sclip = grays.vszip.EEDI3(field=1, vcheck=3, sclip=sclip)
    no_sclip = grays.vszip.EEDI3(field=1, vcheck=3)
    assert max_abs_diff(with_sclip, no_sclip) > 0.0


def test_mclip_gray_accepted_and_masks(grays):
    # a gray mask restricts edge-directed interpolation to nonzero pixels; an
    # empty mask (all zero) forces plain cubic everywhere -> differs from the
    # full edge-directed result.
    edge = grays.std.Prewitt().std.Binarize(0.1).std.Maximum()
    empty = grays.std.BlankClip(color=[0.0])
    masked = grays.vszip.EEDI3(field=1, mclip=edge)
    cubic = grays.vszip.EEDI3(field=1, mclip=empty)
    assert max_abs_diff(masked, cubic) > 0.0


def test_mclip_float_gray_is_converted(grays):
    # a non-8-bit (float) gray mask is converted internally; it must run.
    edge = grays.std.Prewitt().std.Binarize(0.1)
    grays.vszip.EEDI3(field=1, mclip=edge).get_frame(0)


# --- validation / format rejection ------------------------------------------


def test_int_input_rejected(make_clip):
    with pytest.raises(vs.Error, match="32-bit float"):
        make_clip(vs.GRAY16).vszip.EEDI3(field=1).get_frame(0)


@pytest.mark.parametrize(
    ("args", "msg"),
    [
        (dict(field=4), "field must be"),
        (dict(field=-1), "field must be"),
        (dict(field=2, dh=True), "field must be 0 or 1 when dh"),
        (dict(field=1, alpha=1.5), "alpha"),
        (dict(field=1, beta=-0.1), "beta"),
        (dict(field=1, alpha=0.8, beta=0.8), "alpha . beta"),
        (dict(field=1, gamma=-1.0), "gamma"),
        (dict(field=1, nrad=4), "nrad"),
        (dict(field=1, mdis=0), "mdis"),
        (dict(field=1, mdis=41), "mdis"),
        (dict(field=1, vcheck=4), "vcheck"),
        (dict(field=1, vcheck=2, vthresh0=0.0), "vthresh"),
    ],
)
def test_param_validation(grays, args, msg):
    with pytest.raises(vs.Error, match=msg):
        grays.vszip.EEDI3(**args).get_frame(0)


def test_odd_axis_rejected_without_dh(make_clip):
    # EEDI3 needs even height, EEDI3H needs even width when dh=False
    odd_h = make_clip(vs.GRAYS).std.Crop(bottom=1)
    with pytest.raises(vs.Error, match="height must be mod 2"):
        odd_h.vszip.EEDI3(field=1).get_frame(0)
    odd_w = make_clip(vs.GRAYS).std.Crop(right=1)
    with pytest.raises(vs.Error, match="width must be mod 2"):
        odd_w.vszip.EEDI3H(field=1).get_frame(0)


def test_mclip_must_be_gray(yuv):
    with pytest.raises(vs.Error, match="mclip must be Gray"):
        yuv.vszip.EEDI3(field=1, mclip=yuv).get_frame(0)


def test_mclip_dimension_mismatch_rejected(grays):
    wrong = grays.std.AddBorders(right=2).std.Binarize(0.1)
    with pytest.raises(vs.Error, match="mclip's dimensions"):
        grays.vszip.EEDI3(field=1, mclip=wrong).get_frame(0)


def test_sclip_mismatch_rejected(grays):
    wrong = grays.std.AddBorders(right=2)
    with pytest.raises(vs.Error, match="sclip"):
        grays.vszip.EEDI3(field=1, vcheck=2, sclip=wrong).get_frame(0)


@pytest.mark.parametrize("fmt", [vs.GRAYS, vs.YUV420PS, vs.YUV422PS, vs.YUV444PS, vs.RGBS])
def test_all_float_formats_run(make_clip, fmt):
    make_clip(fmt).vszip.EEDI3(field=1).get_frame(0)
