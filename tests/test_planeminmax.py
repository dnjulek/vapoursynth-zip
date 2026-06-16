import pytest
import vapoursynth as vs

from golden import Case, grid, sweep
from helpers import plane_stats, props

# Formats the Create callback accepts: DataType.select (enable_u32=False) takes
# 1/2-byte integer and 2/4-byte float, i.e. 8/16-bit int and f16/f32 across
# GRAY/YUV/RGB. 32-bit integer is rejected (U32 unreachable). thr is unusable on
# float YUV chroma, so float-YUV cases stay luma-only (planes=[0]).
CASES = (
    sweep(
        base_fmt=vs.GRAY16,
        base_args=dict(minthr=0.1, maxthr=0.1),
        formats=[
            vs.GRAY8,
            vs.GRAY16,
            vs.GRAYH,
            vs.GRAYS,
            vs.YUV420P8,
            vs.YUV420P16,
            vs.YUV444P16,
            vs.RGB24,
            vs.RGB48,
            vs.RGBH,
            vs.RGBS,
        ],
        # each minthr/maxthr value picks a different extreme; clipb adds psmDiff
        args=grid(minthr=[0, 0.1, 0.4], maxthr=[0, 0.1, 0.4])
        + [
            dict(minthr=0.1, maxthr=0.1, prop="mm"),
            dict(minthr=0.1, maxthr=0.1, variant_clipb=True),
        ],
        geometries=["odd", "tiny"],
    )
    + [
        # plane subsets on YUV: scalar vs list prop output
        Case(vs.YUV420P16, args=dict(minthr=0.1, maxthr=0.1, planes=[0, 1, 2])),
        Case(vs.YUV420P16, args=dict(minthr=0.1, maxthr=0.1, planes=[1, 2])),
        Case(vs.YUV444P16, args=dict(minthr=0.4, maxthr=0.1, planes=[0, 2])),
        # float YUV: thr only legal luma-only; full-plane no-thr path
        Case(vs.YUV420PS, args=dict(minthr=0.2, planes=[0])),
        Case(vs.YUV420PS, args=dict(planes=[0, 1, 2])),
        # RGB float keeps thr even across all planes (YUV-only restriction)
        Case(vs.RGBS, args=dict(minthr=0.2, maxthr=0.3, planes=[0, 1, 2])),
        # int + clipb diff across planes (psmDiff path)
        Case(vs.YUV420P16, args=dict(minthr=0.2, maxthr=0.3, planes=[0, 1, 2]), variant="ref"),
        Case(vs.RGB24, args=dict(minthr=0.1, maxthr=0.1, planes=[0, 1, 2]), variant="ref"),
    ]
)


def _planeminmax_props(clip: vs.VideoNode, prop: str = "psm") -> dict:
    """Collect the PlaneMinMax frame props that the case wrote, honoring the
    `prop` prefix so the rename case still captures real Min/Max values. Diff is
    only present when a clipb was supplied."""
    p = props(clip)
    out = {}
    for short in ("Min", "Max", "Diff"):
        key = prop + short
        if key in p:
            v = p[key]
            out[short] = list(v) if isinstance(v, (list, tuple)) else v
    return out


@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_cases(golden, make_clip, case):
    src = make_clip(case.fmt, case.geometry)
    kwargs = dict(case.args)
    use_clipb = kwargs.pop("variant_clipb", False) or case.variant == "ref"
    if use_clipb:
        kwargs["clipb"] = src.vszip.BoxBlur(hradius=1, vradius=1)
    out = src.vszip.PlaneMinMax(**kwargs)
    prop = kwargs.get("prop", "psm")
    rel = 1e-3 if src.format.bits_per_sample == 16 and src.format.sample_type == vs.FLOAT else 1e-6
    golden.check_value("planeminmax", case.id, _planeminmax_props(out, prop), rel=rel)


@pytest.fixture(scope="module")
def src16(src_rgb):
    return src_rgb.resize.Point(format=vs.YUV420P16, matrix=1)


@pytest.fixture(scope="module")
def src32(src_rgb):
    return src_rgb.resize.Point(format=vs.YUV420PS, matrix=1)


def test_thr0_matches_std_planestats(to_gray):
    for fmt in (vs.GRAY8, vs.GRAY16, vs.GRAYS):
        src = to_gray(fmt)
        p = props(src.vszip.PlaneMinMax())
        stats = plane_stats(src)
        assert p["psmMin"] == stats["PlaneStatsMin"]
        assert p["psmMax"] == stats["PlaneStatsMax"]


def test_threshold_drop_semantics(core):
    """minthr/maxthr drop that fraction of extreme pixels before picking."""
    zeros = core.std.BlankClip(None, 64, 8, vs.GRAY8, length=1, color=0)  # 25% of pixels
    rest = core.std.BlankClip(None, 64, 24, vs.GRAY8, length=1, color=200)
    src = core.std.StackVertical([zeros, rest])
    assert props(src.vszip.PlaneMinMax(minthr=0.2))["psmMin"] == 0  # drops 20% < 25%
    assert props(src.vszip.PlaneMinMax(minthr=0.3))["psmMin"] == 200  # drops 30% > 25%

    peak = core.std.BlankClip(None, 64, 8, vs.GRAY8, length=1, color=255)
    src2 = core.std.StackVertical([core.std.BlankClip(None, 64, 24, vs.GRAY8, length=1, color=100), peak])
    assert props(src2.vszip.PlaneMinMax(maxthr=0.2))["psmMax"] == 255
    assert props(src2.vszip.PlaneMinMax(maxthr=0.3))["psmMax"] == 100


def test_golden_int(src16):
    p = props(src16.vszip.PlaneMinMax(minthr=0.2, maxthr=0.3, planes=[0, 1, 2]))
    assert p["psmMin"] == [21177, 42248, 19079]
    assert p["psmMax"] == [38762, 52119, 34778]


def test_golden_float(src32):
    p = props(src32.vszip.PlaneMinMax(minthr=0.2, maxthr=0.3))
    assert p["psmMin"] == pytest.approx(0.30467689037323, rel=1e-7)
    assert p["psmMax"] == pytest.approx(0.618341326713562, rel=1e-7)


def test_float_no_thr_exact_minmax(src32):
    out = src32.std.Expr(["", "x 0.4 > 2 x ?", ""]).std.Expr(["", "x 0.3 < -2 x ?", ""])
    p = props(out.vszip.PlaneMinMax(planes=[0, 1, 2]))
    assert p["psmMin"] == pytest.approx([0.07218431681394577, -2.0, -0.47576838731765747], rel=1e-7)
    assert p["psmMax"] == pytest.approx([0.9401216506958008, 2.0, 0.34346699714660645], rel=1e-7)


def test_clipb_diff_golden(src16, src32):
    p16 = props(src16.vszip.PlaneMinMax(minthr=0.2, maxthr=0.3, clipb=src16.vszip.BoxBlur(hradius=1, vradius=1), planes=[0, 1, 2]))
    assert p16["psmDiff"] == pytest.approx([0.04060555502903982, 0.03211699557011521, 0.07426088248407339], rel=1e-6)
    p32 = props(src32.vszip.PlaneMinMax(minthr=0.2, maxthr=0.3, clipb=src32.vszip.BoxBlur(hradius=1, vradius=1)))
    assert p32["psmDiff"] == pytest.approx(0.04750444493987743, rel=1e-6)


def test_diff_ignores_thresholds(src16):
    """psmDiff is computed on all pixels; thr only affects min/max."""
    blur = src16.vszip.BoxBlur(hradius=1, vradius=1)
    d0 = props(src16.vszip.PlaneMinMax(minthr=0, maxthr=0, clipb=blur))["psmDiff"]
    d1 = props(src16.vszip.PlaneMinMax(minthr=0.2, maxthr=0.3, clipb=blur))["psmDiff"]
    assert d0 == d1


def test_prop_rename(core):
    src = core.std.BlankClip(None, 64, 32, vs.YUV420P16, length=1, color=[6777, 32768, 0])
    out = core.vszip.PlaneMinMax(clipa=src, minthr=0.2, maxthr=0.3)
    out = core.vszip.PlaneMinMax(clipa=out, minthr=0.2, maxthr=0.3, prop="mm_test")
    p = props(out)
    assert p["psmMin"] == p["mm_testMin"] == 6777
    assert p["psmMax"] == p["mm_testMax"] == 6777


def test_f16(core):
    src = core.std.BlankClip(None, 64, 32, vs.GRAYH, length=1, color=0.5)
    p = props(src.vszip.PlaneMinMax())
    assert p["psmMin"] == 0.5
    assert p["psmMax"] == 0.5


@pytest.mark.parametrize("args", [dict(minthr=1.5), dict(minthr=-0.1), dict(maxthr=2.0)])
def test_thr_range_error(to_gray, args):
    key = next(iter(args))
    with pytest.raises(vs.Error, match=f"{key} should be a float between 0.0 and 1.0"):
        to_gray(vs.GRAY16).vszip.PlaneMinMax(**args)


def test_float_chroma_thr_error(src32):
    with pytest.raises(vs.Error, match="you can't use maxthr/minthr with float chroma"):
        src32.vszip.PlaneMinMax(minthr=0.2, maxthr=0.3, planes=[0, 1, 2])


@pytest.mark.parametrize(
    ("args", "msg"),
    [
        # thr range (mapped through getThr); also exercised by test_thr_range_error
        (dict(minthr=1.5), "minthr should be a float between 0.0 and 1.0"),
        (dict(minthr=-0.1), "minthr should be a float between 0.0 and 1.0"),
        (dict(maxthr=2.0), "maxthr should be a float between 0.0 and 1.0"),
        (dict(maxthr=-0.5), "maxthr should be a float between 0.0 and 1.0"),
        # plane bounds / duplicate (mapGetPlanes)
        (dict(planes=[3]), "plane index out of range"),
        (dict(planes=[-1]), "plane index out of range"),
        (dict(planes=[0, 0]), "plane specified twice"),
    ],
)
def test_validation_errors(to_yuv, args, msg):
    with pytest.raises(vs.Error, match=msg):
        to_yuv(vs.YUV420P16).vszip.PlaneMinMax(**args)


def test_int_format_rejected(core):
    """DataType.select rejects 32-bit integer (the U32 branch is unreachable)."""
    src = core.std.BlankClip(None, 64, 32, vs.GRAY32, length=1, color=123)
    with pytest.raises(vs.Error, match="not supported Int format"):
        src.vszip.PlaneMinMax()


def test_clipb_shorter_error(core):
    a = core.std.BlankClip(None, 64, 32, vs.GRAY8, length=5)
    b = core.std.BlankClip(None, 64, 32, vs.GRAY8, length=3)
    with pytest.raises(vs.Error, match="second clip has less frames than input clip"):
        core.vszip.PlaneMinMax(clipa=a, clipb=b)


@pytest.mark.parametrize(
    ("clipb_fmt", "clipb_dims", "msg"),
    [
        # compareNodes checks, in source order: each clipb differs from clipa
        # (64x32 YUV420P16) in exactly one axis so the targeted message fires.
        (vs.YUV420P16, (32, 32), "all input clips must have the same width and height"),
        (vs.RGB48, (64, 32), "all input clips must have the same color family"),
        (vs.YUV444P16, (64, 32), "all input clips must have the same subsampling"),
        (vs.YUV420P8, (64, 32), "all input clips must have the same bit depth"),
    ],
)
def test_clipb_mismatch_errors(core, clipb_fmt, clipb_dims, msg):
    a = core.std.BlankClip(None, 64, 32, vs.YUV420P16, length=3)
    w, h = clipb_dims
    b = core.std.BlankClip(None, w, h, clipb_fmt, length=3)
    with pytest.raises(vs.Error, match=msg):
        core.vszip.PlaneMinMax(clipa=a, clipb=b)


@pytest.mark.parametrize("fmt", [vs.GRAY16, vs.GRAYS])
def test_thr_one_no_counter_overflow(core, fmt):
    """Regression: minthr/maxthr=1.0 on a 16-bit or float clip (hist_size=65536)
    overflowed the u16 histogram loop counter -> Debug panic / wrong min. The
    counter is now u32; dropping 100% yields peak as min and 0 as max."""
    src = core.std.BlankClip(None, 64, 64, fmt, length=1, color=0.5 if fmt == vs.GRAYS else 30000)
    pmin = props(src.vszip.PlaneMinMax(minthr=1.0))["psmMin"]
    pmax = props(src.vszip.PlaneMinMax(maxthr=1.0))["psmMax"]
    if fmt == vs.GRAY16:
        assert pmin == 65535 and pmax == 0
