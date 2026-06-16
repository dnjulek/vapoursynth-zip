import pytest
import vapoursynth as vs

from golden import Case, grid, sweep
from helpers import assert_same_clip, diff, plane_stats, write_png

# Limiter clamps each plane to [min, max]. The min/max array path (RT dispatch)
# requires both arrays, length == num_planes; tv_range/mask only bite on the
# comptime (no-min/max) path. Sweep formats the BPSType/DataType selectors
# accept: int 8/9/10/12/14/16/32 + float 16/32, across Gray/YUV/RGB. Each swept
# value below clamps a different window of the real image, so goldens differ.
CASES = (
    sweep(
        base_fmt=vs.GRAY16,
        base_args=dict(min=[10000], max=[50000]),
        # no format axis here: min/max bounds are format-specific (8-bit peak is
        # 255, float wants 0..1), so per-format goldens live in the explicit list
        # below. This sweep varies only the arg window and the geometry.
        args=[
            dict(min=[20000], max=[40000]),
            dict(min=[0], max=[30000]),
            dict(min=[30000], max=[65535]),
        ],
        geometries=["odd", "tiny"],
    )
    + [
        # gray formats across int + float (one case each, format-appropriate
        # clamp windows that actually clip the real image's value range)
        Case(vs.GRAY16, args=dict(min=[10000], max=[50000])),
        Case(vs.GRAY8, args=dict(min=[50], max=[200])),
        Case(vs.GRAYH, args=dict(min=[0.2], max=[0.8])),
        Case(vs.GRAYS, args=dict(min=[0.2], max=[0.8])),
        # the in-between int bit depths (U9/U12/U14) each select a distinct
        # BPSType -> distinct comptime range table in the tv_range/default
        # dispatch; the RT (min/max) path here also confirms the u16-carrier
        # selectors. Clamp windows are picked inside each format's measured
        # value range so the golden moves.
        Case(vs.GRAY9, args=dict(min=[100], max=[400])),
        Case(vs.YUV420P9, args=dict(min=[100, 300, 100], max=[400, 450, 380])),
        Case(vs.GRAY12, args=dict(min=[800, ], max=[3000])),
        Case(vs.YUV444P12, args=dict(min=[800, 2300, 600], max=[3000, 3600, 3000])),
        Case(vs.GRAY14, args=dict(min=[3000], max=[12000])),
        Case(vs.YUV422P14, args=dict(min=[3000, 9000, 2000], max=[12000, 14500, 12000])),
        # multi-plane formats with per-plane min/max arrays (int + float)
        Case(vs.YUV420P8, args=dict(min=[40, 20, 30], max=[200, 220, 190])),
        Case(vs.YUV420P10, args=dict(min=[200, 100, 100], max=[800, 900, 850])),
        Case(vs.YUV444P16, args=dict(min=[10000, 20000, 10000], max=[50000, 55000, 45000])),
        Case(vs.YUV420PS, args=dict(min=[0.1, -0.4, -0.4], max=[0.9, 0.4, 0.4])),
        Case(vs.RGB24, args=dict(min=[20, 20, 100], max=[180, 200, 250])),
        Case(vs.RGBS, args=dict(min=[0.1, 0.1, 0.1], max=[0.7, 0.7, 0.99])),
        Case(vs.RGBH, args=dict(min=[0.1, 0.1, 0.1], max=[0.7, 0.7, 0.99])),
        # per-plane arrays let one plane pass through while others clamp
        Case(vs.YUV444P16, args=dict(min=[8143, 0, 0], max=[56803, 65535, 65535])),
        # planes subset: unprocessed planes are copied (plane 0 stays input)
        Case(vs.YUV444P16, args=dict(min=[10000, 20000, 10000], max=[50000, 55000, 45000], planes=[0])),
        Case(vs.YUV444P16, args=dict(min=[10000, 20000, 10000], max=[50000, 55000, 45000], planes=[1, 2])),
        Case(vs.RGB24, args=dict(min=[20, 20, 100], max=[180, 200, 250], planes=[0, 2])),
        # comptime (no min/max) path: tv_range bites on RGB int (values outside
        # 16..235), mask bites on float YUV (negative chroma clamped to 0..1).
        # RGB int spans every int BPSType so each rgbN comptime range table is
        # exercised: rgb8 (RGB24), rgb9 (RGB27), rgb12 (RGB36), rgb16 (RGB48).
        Case(vs.RGB24, args=dict(tv_range=True)),
        Case(vs.RGB27, args=dict(tv_range=True)),
        Case(vs.RGB36, args=dict(tv_range=True)),
        Case(vs.RGB48, args=dict(tv_range=True)),
        # mask clamps float chroma to 0..1; on this path tv_range is bypassed
        # (mask forces the rgbf range), so tv_range+mask == mask alone -> omitted.
        Case(vs.YUV420PS, args=dict(mask=True)),
        Case(vs.YUV420PH, args=dict(mask=True)),
    ]
)


@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_cases(golden, make_clip, case):
    src = make_clip(case.fmt, case.geometry)
    golden.check("limiter", case, src.vszip.Limiter(**case.args))


def stacked(core: vs.Core, fmt: int) -> vs.VideoNode:
    """Top half all-peak, bottom half all-floor (floor is -2 for float)."""
    base = core.std.BlankClip(None, 64, 32, fmt, length=1)
    is_float = base.format.sample_type == vs.FLOAT
    peak = 2.0 if is_float else (1 << base.format.bits_per_sample) - 1
    floor = -2.0 if is_float else 0
    n = base.format.num_planes
    hi = base.std.BlankClip(color=[peak] * n)
    lo = base.std.BlankClip(color=[floor] * n)
    return core.std.StackVertical([hi, lo])


def min_max(clip: vs.VideoNode) -> tuple[list, list]:
    stats = [plane_stats(clip, plane=p) for p in range(clip.format.num_planes)]
    return [s["PlaneStatsMin"] for s in stats], [s["PlaneStatsMax"] for s in stats]


TV_RANGE = [
    (vs.YUV420P8, [16, 16, 16], [235, 240, 240]),
    (vs.YUV420P9, [32, 32, 32], [470, 480, 480]),
    (vs.YUV420P10, [64, 64, 64], [940, 960, 960]),
    (vs.YUV420P12, [256, 256, 256], [3760, 3840, 3840]),
    (vs.YUV420P14, [1024, 1024, 1024], [15040, 15360, 15360]),
    (vs.YUV420P16, [4096, 4096, 4096], [60160, 61440, 61440]),
    (vs.YUV420PS, [0.0, -0.5, -0.5], [1.0, 0.5, 0.5]),
]


@pytest.mark.parametrize(("fmt", "lo", "hi"), TV_RANGE)
def test_tv_range(core, fmt, lo, hi):
    out = stacked(core, fmt).vszip.Limiter(tv_range=True)
    assert min_max(out) == (lo, hi)


def test_mask_full_range_chroma(core):
    out = stacked(core, vs.YUV420PS).vszip.Limiter(tv_range=True, mask=True)
    assert min_max(out) == ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])


def test_float_default_clamps_full_range(core):
    out = stacked(core, vs.YUV420PS).vszip.Limiter()
    assert min_max(out) == ([0.0, -0.5, -0.5], [1.0, 0.5, 0.5])


def test_int_default_is_noop(core):
    src = stacked(core, vs.YUV420P16)
    assert_same_clip(src.vszip.Limiter(), src)


@pytest.mark.parametrize(("fmt", "lo", "hi"), TV_RANGE[:3])
def test_tv_range_equals_explicit_min_max(core, fmt, lo, hi):
    """The comptime tv_range path and the runtime min/max path must agree."""
    src = stacked(core, fmt)
    assert_same_clip(src.vszip.Limiter(tv_range=True), src.vszip.Limiter(min=lo, max=hi))


def test_explicit_min_max_pixels(core, tmp_path):
    ramp = [list(range(256)) for _ in range(2)]
    src = core.vszip.ImageRead(str(write_png(tmp_path / "ramp.png", ramp, color="gray")))
    out = src.vszip.Limiter(min=[10], max=[200])
    with out.get_frame(0) as f:
        row = [f[0][0, x] for x in range(256)]
    assert row == [min(max(x, 10), 200) for x in range(256)]


def test_planes(core):
    src = stacked(core, vs.YUV420P16)
    out = src.vszip.Limiter(tv_range=True, planes=[0])
    assert diff(out, src, plane=1) == 0.0  # untouched plane is copied
    assert diff(out, src, plane=2) == 0.0
    assert min_max(out)[0][0] == 4096


U32_FAMILIES = [
    (vs.GRAY, [3942645760], [3942645760]),
    (vs.YUV, [3942645760, 4026531840, 4026531840], [3942645760, 4026531840, 4026531840]),
    (vs.RGB, [3942645760, 3942645760, 3942645760], [3942645760, 3942645760, 3942645760]),
]


@pytest.mark.parametrize(("family", "tv_max", "_unused"), U32_FAMILIES)
def test_u32_comptime_paths(core, family, tv_max, _unused):
    """32-bit integer is its own BPSType (U32) with dedicated full32/yuv32/rgb32
    comptime range tables and a distinct Limiter(u32) instantiation. zimg has no
    32-bit-int pixel type, so make_clip/golden_stats (both resize/PlaneStats
    based) can't reach it; build the clip directly and read raw pixels. The
    explicit-min/max RT path is unreachable here because getPeakValue overflows
    for 32-bit int, so only the comptime default + tv_range paths are tested."""
    fmt = core.query_video_format(family, vs.INTEGER, 32, 0, 0).id
    base = core.std.BlankClip(None, 64, 32, fmt, length=1)
    n = base.format.num_planes
    peak = (1 << 32) - 1
    src = core.std.StackVertical([base.std.BlankClip(color=[peak] * n), base.std.BlankClip(color=[0] * n)])

    def mm(clip: vs.VideoNode) -> tuple[list, list]:
        # std.PlaneStats has no 32-bit-int support, so read raw uint32 planes.
        # The frame memoryview is already format 'I'; flatten via bytes cast.
        f = clip.get_frame(0)
        cols = [memoryview(f[p]).cast("B").cast("I") for p in range(clip.format.num_planes)]
        return [min(c) for c in cols], [max(c) for c in cols]

    # default: full32 table is [0, 4294967295] -> no-op on the [0, peak] clip
    assert mm(src.vszip.Limiter()) == ([0] * n, [peak] * n)
    # tv_range: floor clamps 0 -> 16<<24, ceil clamps peak -> the per-family max
    lo, hi = mm(src.vszip.Limiter(tv_range=True))
    assert lo == [268435456] * n
    assert hi == tv_max


@pytest.mark.parametrize(
    ("args", "msg"),
    [
        (dict(min=[0, 0, 0]), "min array is set but max array is not"),
        (dict(max=[255, 255, 255]), "max array is set but min array is not"),
        (dict(min=[0, 0], max=[255, 255, 255]), "min array must have the same number of elements as planes"),
        (dict(min=[0, 0, 0], max=[255, 255]), "max array must have the same number of elements as planes"),
        (dict(min=[-1, 0, 0], max=[255, 255, 255]), "min value must be greater than or equal to 0"),
        (dict(min=[0, 0, 0], max=[255, 255, 256]), "max value must be less than or equal to peak value"),
        (dict(planes=[3]), "plane index out of range"),
        (dict(planes=[-1]), "plane index out of range"),
        (dict(planes=[0, 0]), "plane specified twice"),
    ],
)
def test_validation_errors(core, args, msg):
    src = stacked(core, vs.YUV420P8)
    with pytest.raises(vs.Error, match=msg):
        src.vszip.Limiter(**args)
