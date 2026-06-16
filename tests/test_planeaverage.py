import pytest
import vapoursynth as vs

from golden import Case, grid, sweep
from helpers import plane_stats, props


# PlaneAverage writes frame props (psmAvg, and psmDiff when clipb is given), not
# processed pixels, so the golden cases below use golden.check_value() on the
# extracted prop values rather than golden.check()/per-plane pixel stats.
#
# The natural test image carries no flat regions, so exclude lists only shift the
# average when they hit values actually present. exclude=[-1] is a never-matching
# sentinel (plain PlaneStats); the active exclude values used on the args axis are
# 8-bit midtones verified to change the GRAY8 average.

# Formats accepted by DataType.select(enable_u32=true): int 8/16/32 and float
# 16/32, any color family. GRAY32 (int 32) is exercised separately because
# resize can't synthesize it from the image (see the "blank" variant case).
_FORMATS = [
    vs.GRAY8,
    vs.GRAY16,
    vs.GRAYH,
    vs.GRAYS,
    vs.YUV420P8,
    vs.YUV420P16,
    vs.YUV444PS,
    vs.RGB24,
    vs.RGBS,
]

CASES = (
    sweep(
        base_fmt=vs.GRAY8,
        base_args=dict(exclude=[-1]),
        formats=_FORMATS,
        args=grid(exclude=[[-1], [128], [100, 150, 200]])
        + [
            # prop rename combined with an active exclude (key changes, value too)
            dict(exclude=[128], prop="myavg"),
        ],
        geometries=["odd", "tiny"],
    )
    + [
        # planes subsets on a multi-plane clip (single float vs list output)
        Case(vs.YUV420P8, args=dict(exclude=[-1], planes=[0, 1, 2])),
        Case(vs.YUV420P8, args=dict(exclude=[-1], planes=[1])),
        Case(vs.YUV420P8, args=dict(exclude=[-1], planes=[0, 2])),
        Case(vs.YUV420P16, args=dict(exclude=[-1], planes=[0, 1, 2])),
        Case(vs.RGB24, args=dict(exclude=[-1], planes=[0, 1, 2])),
        Case(vs.RGBS, args=dict(exclude=[-1], planes=[0, 1, 2])),
        # active exclude interacting with multi-plane processing
        Case(vs.YUV420P8, args=dict(exclude=[128], planes=[0, 1, 2])),
        # clipb diff mode (psmDiff): different blur radii -> distinct diffs
        Case(vs.GRAY16, args=dict(exclude=[-1]), variant="ref1"),
        Case(vs.GRAY16, args=dict(exclude=[-1]), variant="ref3"),
        Case(vs.YUV420P8, args=dict(exclude=[-1], planes=[0, 1, 2]), variant="ref3"),
        # clipb diff on FLOAT clips: averageRef has a distinct float diff branch
        # (diffacc / total, no peak division) separate from the int path above.
        Case(vs.GRAYS, args=dict(exclude=[-1]), variant="ref3"),
        Case(vs.RGBS, args=dict(exclude=[-1], planes=[0, 1, 2]), variant="ref3"),
        # clipb diff + active exclude + renamed prop together
        Case(vs.GRAY16, args=dict(exclude=[5000], prop="myavg"), variant="ref3"),
        # NOTE: GRAY32 (int 32) is intentionally absent — exclude (required) is
        # rejected for 32-bit integer clips, so it has no golden path. See
        # test_gray32_exclude_rejected below.
    ]
)


def _ref_clip(src: vs.VideoNode, variant: str) -> vs.VideoNode:
    radius = {"ref1": 1, "ref3": 3}[variant]
    return src.std.BoxBlur(hradius=radius, vradius=radius)


def _case_clips(core, make_clip, case: Case):
    """Build (clipa, kwargs) for a golden case, honoring its variant."""
    kwargs = dict(case.args)
    src = make_clip(case.fmt, case.geometry)
    if case.variant.startswith("ref"):
        kwargs["clipb"] = _ref_clip(src, case.variant)
    return src, kwargs


def _prop_value(p: dict, case: Case) -> dict:
    """Extract the avg (and diff, in clipb mode) prop value(s) for a case.
    The value is a float for one plane and a list for several; the golden
    store handles both. Diff is present only when a clipb was given."""
    name = case.args.get("prop", "psm")
    out = {"avg": p[f"{name}Avg"]}
    if case.variant.startswith("ref"):
        out["diff"] = p[f"{name}Diff"]
    return out


@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_cases(golden, core, make_clip, case):
    src, kwargs = _case_clips(core, make_clip, case)
    out = src.vszip.PlaneAverage(**kwargs)
    rel = 1e-3 if out.format.sample_type == vs.FLOAT and out.format.bits_per_sample == 16 else 1e-6
    golden.check_value("planeaverage", case.id, _prop_value(props(out), case), rel=rel)


def two_tone(core: vs.Core, fmt: int, lo, hi) -> vs.VideoNode:
    a = core.std.BlankClip(None, 64, 32, fmt, length=1, color=lo)
    b = core.std.BlankClip(None, 64, 32, fmt, length=1, color=hi)
    return core.std.StackVertical([a, b])


def test_matches_std_planestats(to_gray):
    """With a never-matching exclude sentinel this is plain PlaneStats."""
    src = to_gray(vs.GRAY16)
    ours = props(src.vszip.PlaneAverage(exclude=[-1]))["psmAvg"]
    ref = plane_stats(src)["PlaneStatsAverage"]
    assert ours == pytest.approx(ref, rel=1e-12)


def test_exclude_exact(core):
    src = two_tone(core, vs.GRAY16, 1000, 3000)
    assert props(src.vszip.PlaneAverage(exclude=[1000]))["psmAvg"] == 3000 / 65535
    assert props(src.vszip.PlaneAverage(exclude=[3000]))["psmAvg"] == 1000 / 65535
    # excluding every pixel yields 0
    assert props(src.vszip.PlaneAverage(exclude=[1000, 3000]))["psmAvg"] == 0.0


def test_exclude_float_clip(core):
    src = two_tone(core, vs.GRAYS, 3.0, 1.0)
    assert props(src.vszip.PlaneAverage(exclude=[3]))["psmAvg"] == 1.0


def test_clipb_diff_matches_std(to_gray):
    src = to_gray(vs.GRAY16)
    blur = src.std.BoxBlur(hradius=3, vradius=3)
    p = props(src.vszip.PlaneAverage(exclude=[-1], clipb=blur))
    assert p["psmDiff"] == pytest.approx(plane_stats(src, blur)["PlaneStatsDiff"], rel=1e-12)


def test_planes_and_prop_rename(core):
    src = core.std.BlankClip(None, 64, 32, vs.YUV420P16, length=1, color=[6777, 32768, 0])
    out = core.vszip.PlaneAverage(clipa=src, exclude=[300, 5000])
    out = core.vszip.PlaneAverage(clipa=out, exclude=[300, 5000], prop="avg_test")
    p = props(out)
    assert p["psmAvg"] == 0.10341039139391164  # 6777 / 65535
    assert p["avg_testAvg"] == p["psmAvg"]

    multi = props(core.vszip.PlaneAverage(clipa=src, exclude=[-1], planes=[0, 1, 2]))["psmAvg"]
    assert multi == [6777 / 65535, 32768 / 65535, 0.0]


def test_gray32_exclude_rejected(core):
    # exclude is stored as i32 and cannot represent the upper half of the u32
    # sample range, so 32-bit integer clips are rejected (exclude is required).
    src = core.std.BlankClip(None, 64, 32, vs.GRAY32, length=1, color=123456)
    with pytest.raises(vs.Error, match="32-bit integer"):
        src.vszip.PlaneAverage(exclude=[-1])


def test_plane_errors(to_yuv):
    src = to_yuv(vs.YUV420P16)
    with pytest.raises(vs.Error, match="plane index out of range"):
        src.vszip.PlaneAverage(exclude=[-1], planes=[3])
    with pytest.raises(vs.Error, match="plane specified twice"):
        src.vszip.PlaneAverage(exclude=[-1], planes=[0, 0])


def test_exclude_required(to_gray):
    with pytest.raises(vs.Error, match="exclude"):
        to_gray(vs.GRAY16).vszip.PlaneAverage()


def test_clipb_shorter_error(core):
    a = core.std.BlankClip(None, 64, 32, vs.GRAY8, length=5)
    b = core.std.BlankClip(None, 64, 32, vs.GRAY8, length=3)
    with pytest.raises(vs.Error, match="second clip has less frames than input clip"):
        core.vszip.PlaneAverage(clipa=a, exclude=[-1], clipb=b)


def _validation_call(core, kind):
    """Build the (kwargs) that trips a single Create-callback check.
    `clipa` is a 64x32 YUV420P8 clip unless the check needs otherwise."""
    a = core.std.BlankClip(None, 64, 32, vs.YUV420P8, length=4)
    if kind == "plane_oob":
        return dict(clipa=a, exclude=[-1], planes=[3])
    if kind == "plane_twice":
        return dict(clipa=a, exclude=[-1], planes=[0, 0])
    if kind == "clipb_short":
        b = core.std.BlankClip(None, 64, 32, vs.YUV420P8, length=2)
        return dict(clipa=a, exclude=[-1], clipb=b)
    if kind == "clipb_size":
        b = core.std.BlankClip(None, 48, 32, vs.YUV420P8, length=4)
        return dict(clipa=a, exclude=[-1], clipb=b)
    if kind == "clipb_family":
        b = core.std.BlankClip(None, 64, 32, vs.RGB24, length=4)
        return dict(clipa=a, exclude=[-1], clipb=b)
    if kind == "clipb_subsampling":
        b = core.std.BlankClip(None, 64, 32, vs.YUV444P8, length=4)
        return dict(clipa=a, exclude=[-1], clipb=b)
    if kind == "clipb_depth":
        b = core.std.BlankClip(None, 64, 32, vs.YUV420P16, length=4)
        return dict(clipa=a, exclude=[-1], clipb=b)
    raise ValueError(kind)


@pytest.mark.parametrize(
    ("kind", "msg"),
    [
        ("plane_oob", "plane index out of range"),
        ("plane_twice", "plane specified twice"),
        ("clipb_short", "second clip has less frames than input clip"),
        ("clipb_size", "all input clips must have the same width and height"),
        ("clipb_family", "all input clips must have the same color family"),
        ("clipb_subsampling", "all input clips must have the same subsampling"),
        ("clipb_depth", "all input clips must have the same bit depth"),
    ],
)
def test_validation_errors(core, kind, msg):
    with pytest.raises(vs.Error, match=msg):
        core.vszip.PlaneAverage(**_validation_call(core, kind))
