import pytest
import vapoursynth as vs

from golden import Case, grid, sweep
from helpers import assert_same_clip, avg, diff, max_abs_diff, repack

# Snapshot-golden cases. Axis-sweep around a small symmetric base radius that
# stays on the comptime path (hradius == vradius, 1..22, single pass) and is
# safe on the tiny 12x6 geometry. The args axis (run on the full-size base
# clip) crosses into the runtime path: large radii (>22), hradius != vradius,
# and multi-pass blurs. The geometry axis only re-runs the base radius, so it
# never hits the runtime out-of-bounds that radius >= dimension would trigger.
CASES = (
    sweep(
        base_fmt=vs.GRAY16,
        base_args=dict(hradius=2, vradius=2),
        formats=[vs.GRAY8, vs.GRAY16, vs.GRAYH, vs.GRAYS, vs.YUV420P8, vs.YUV420P16, vs.RGBS],
        args=[
            # comptime path: radii straddling the 1..22 split point
            dict(hradius=1, vradius=1),
            dict(hradius=8, vradius=8),
            dict(hradius=22, vradius=22),
            # runtime path: radius > 22 forces it even when symmetric
            dict(hradius=23, vradius=23),
            dict(hradius=40, vradius=40),
            # runtime path via asymmetric radii (hradius != vradius)
            dict(hradius=4, vradius=9),
            dict(hradius=9, vradius=4),
            # h-only and v-only
            dict(hradius=7, vradius=0, vpasses=0),
            dict(hradius=0, hpasses=0, vradius=7),
            # multi-pass (runtime path); passes 1..3 each axis
            dict(hradius=5, vradius=5, hpasses=2, vpasses=1),
            dict(hradius=5, vradius=5, hpasses=1, vpasses=2),
            dict(hradius=5, vradius=5, hpasses=3, vpasses=3),
        ],
        geometries=["odd", "tiny"],
    )
    + [
        # per-plane subsets on subsampled YUV (chroma planes copied through)
        Case(vs.YUV420P16, args=dict(hradius=5, vradius=5, planes=[0])),
        Case(vs.YUV420P16, args=dict(hradius=5, vradius=5, planes=[1, 2])),
        # asymmetric multi-pass interaction on RGB (all planes, runtime path)
        Case(vs.RGBS, args=dict(hradius=6, vradius=3, hpasses=2, vpasses=3)),
        # f16 on the runtime path: the formats sweep only hits GRAYH on the
        # comptime kernel, so this is the sole golden over blurFloat<f16>.
        Case(vs.GRAYH, args=dict(hradius=6, vradius=3, hpasses=2, vpasses=2)),
    ]
)


@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_cases(golden, make_clip, case):
    src = make_clip(case.fmt, case.geometry)
    golden.check("boxblur", case, src.vszip.BoxBlur(**case.args))


# Golden averages carried over from the old .vpy suite (same source pipeline).
GOLDENS = [
    (vs.GRAYS, dict(hradius=30, vradius=60, hpasses=6, vpasses=8), 0.49595518544825606),
    (vs.GRAYS, dict(hradius=3, vradius=3), 0.49599070191539796),
    (vs.GRAY16, dict(hradius=30, vradius=33, hpasses=1, vpasses=3), 0.4867611337214847),
    (vs.GRAY16, dict(hradius=10, vradius=10), 0.4869014934022612),
]


@pytest.mark.parametrize(("fmt", "args", "expected"), GOLDENS)
def test_golden(to_gray, fmt, args, expected):
    assert avg(to_gray(fmt).vszip.BoxBlur(**args)) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("fmt", [vs.GRAY8, vs.GRAY16, vs.GRAYS])
@pytest.mark.parametrize("radius", [1, 8, 22, 23, 40])  # 1..22 comptime path, 23+ runtime path
def test_matches_std_boxblur(to_gray, fmt, radius):
    """Interior pixels match std.BoxBlur; borders are excluded because the two
    plugins use different edge-mirroring policies."""
    src = to_gray(fmt)
    margin = radius + 2
    interior = dict(left=margin, right=margin, top=margin, bottom=margin)
    ours = src.vszip.BoxBlur(hradius=radius, vradius=radius).std.Crop(**interior)
    ref = src.std.BoxBlur(hradius=radius, hpasses=1, vradius=radius, vpasses=1).std.Crop(**interior)
    tol = {vs.GRAY8: 2, vs.GRAY16: 16, vs.GRAYS: 1e-5}[fmt]  # fixed-point reciprocal rounding
    assert max_abs_diff(ours, ref) <= tol


@pytest.mark.parametrize("fmt", [vs.GRAY8, vs.GRAY16, vs.GRAYS])
def test_pass_composition(to_gray, fmt):
    """Two passes in one invocation equal two chained single-pass invocations."""
    src = to_gray(fmt)
    once = src.vszip.BoxBlur(hradius=7, hpasses=2, vradius=0, vpasses=0)
    single = dict(hradius=7, hpasses=1, vradius=0, vpasses=0)
    chained = src.vszip.BoxBlur(**single).vszip.BoxBlur(**single)
    assert_same_clip(once, chained)


def test_h_and_v_compose(to_gray):
    """hradius+vradius in one call equals separate h-only and v-only calls."""
    src = to_gray(vs.GRAY16)
    both = src.vszip.BoxBlur(hradius=4, vradius=9)
    split = src.vszip.BoxBlur(hradius=4, vradius=0, vpasses=0).vszip.BoxBlur(hradius=0, hpasses=0, vradius=9)
    assert_same_clip(both, split)


def test_f16_runs(to_gray):
    out = to_gray(vs.GRAYH).vszip.BoxBlur(hradius=5, vradius=5)
    assert out.format.id == vs.GRAYH
    # PlaneStats has no half support, so measure after converting to f32
    assert 0.0 < avg(out.resize.Point(format=vs.GRAYS)) < 1.0


def test_planes(to_yuv):
    src = to_yuv(vs.YUV420P16)
    out = src.vszip.BoxBlur(planes=[0], hradius=5, vradius=5)
    assert diff(out, src, plane=1) == 0.0  # untouched planes are copied
    assert diff(out, src, plane=2) == 0.0
    assert diff(out, src, plane=0) > 0.0
    y_out = out.std.ShufflePlanes(0, vs.GRAY)
    y_blur = src.std.ShufflePlanes(0, vs.GRAY).vszip.BoxBlur(hradius=5, vradius=5)
    assert_same_clip(y_out, y_blur)


@pytest.mark.parametrize("fmt", [vs.GRAY8, vs.GRAY16, vs.GRAYS])
@pytest.mark.parametrize("radius", [10, 30])  # comptime and runtime paths
def test_stride_handling(to_gray, fmt, radius):
    cropped = to_gray(fmt).std.Crop(left=27)  # odd width + offset plane pointer
    a = cropped.vszip.BoxBlur(hradius=radius, vradius=radius)
    b = repack(cropped).vszip.BoxBlur(hradius=radius, vradius=radius)
    assert_same_clip(a, b)


def test_nothing_to_do_error(to_gray):
    with pytest.raises(vs.Error, match="nothing to be performed"):
        to_gray(vs.GRAY8).vszip.BoxBlur(hradius=0, vradius=0, hpasses=0, vpasses=0)


def test_unsupported_format_error(core):
    src = core.std.BlankClip(format=vs.GRAY32, width=64, height=64)
    with pytest.raises(vs.Error, match="not supported Int format"):
        src.vszip.BoxBlur(hradius=1, vradius=1)


@pytest.mark.parametrize(
    ("args", "msg"),
    [
        # both blur axes inactive (radius or pass count zero on each)
        (dict(hradius=0, vradius=0, hpasses=0, vpasses=0), "nothing to be performed"),
        (dict(hradius=5, vradius=5, hpasses=0, vpasses=0), "nothing to be performed"),
        # mapGetPlanes bounds + uniqueness checks
        (dict(planes=[3]), "plane index out of range"),
        (dict(planes=[-1]), "plane index out of range"),
        (dict(planes=[0, 0]), "plane specified twice"),
    ],
)
def test_validation_errors(to_yuv, args, msg):
    with pytest.raises(vs.Error, match=msg):
        to_yuv(vs.YUV420P8).vszip.BoxBlur(**args)


def test_unsupported_int_format_error(core):
    # DataType.select rejects 4-byte integer (GRAY32) since enable_u32=false
    src = core.std.BlankClip(format=vs.GRAY32, width=64, height=64)
    with pytest.raises(vs.Error, match="not supported Int format"):
        src.vszip.BoxBlur(hradius=1, vradius=1)
