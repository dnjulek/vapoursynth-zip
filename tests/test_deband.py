import pytest
import vapoursynth as vs

from golden import Case, grid, sweep
from helpers import assert_same_clip, avg, diff, plane_stats

# Base config: both debanding (thr) and grain active, with a fixed seed so the
# RNG-driven reference offsets and grain are deterministic. Deband samples
# references even on a smooth photo, so every thr/range/sample_mode/algo value
# yields a distinguishable golden.
CASES = (
    sweep(
        base_fmt=vs.GRAY16,
        base_args=dict(thr=48, grain=16, seed=7),
        formats=[vs.GRAY8, vs.GRAY16, vs.GRAYS, vs.YUV420P8, vs.YUV420P16, vs.YUV444PS],
        args=grid(sample_mode=[1, 2, 3, 4, 5, 6, 7])
        + grid(blur_first=[True, False])
        + grid(range=[1, 8, 31])
        + grid(random_algo_ref=[0, 1, 2])
        + grid(random_algo_grain=[0, 1, 2])
        + [
            # dynamic vs static grain (single frame still picks a frame-0 offset)
            dict(dynamic_grain=True),
            dict(dynamic_grain=False),
        ],
        geometries=["odd", "tiny"],
    )
    + [
        # asymmetric subsampling (ssw=1, ssh=0): exercises the chroma reference
        # LUT path where val>>ssw and val>>ssh diverge (neither YUV420 ssw==ssh==1
        # nor YUV444 ssw==ssh==0 reach it)
        Case(vs.YUV422P16, args=dict(thr=48, grain=16, seed=7)),
        Case(vs.YUV422P8, args=dict(thr=[48, 24], grain=[16, 8], seed=7)),
        # RGB family: accepted 3-plane paths (int + float). keep_tv_range is a
        # no-op on RGB (only YUV is clamped), so it is omitted here.
        Case(vs.RGB48, args=dict(thr=48, grain=16, seed=7)),
        Case(vs.RGBS, args=dict(thr=48, grain=16, seed=7)),
        # keep_tv_range only clamps YUV; on GRAY it is a no-op, so target YUV
        Case(vs.YUV420P16, args=dict(thr=48, grain=16, seed=7, keep_tv_range=True)),
        # detail-protection thresholds only bite on sample_mode 5/6/7
        Case(vs.GRAY16, args=dict(thr=48, grain=16, seed=7, sample_mode=5, thr1=80, thr2=20)),
        Case(vs.GRAY16, args=dict(thr=48, grain=16, seed=7, sample_mode=6, thr1=80, thr2=20)),
        Case(vs.GRAY16, args=dict(thr=48, grain=16, seed=7, sample_mode=7, thr1=80, thr2=20)),
        # angle check only applies to sample_mode 7
        Case(vs.GRAY16, args=dict(thr=48, grain=16, seed=7, sample_mode=7, angle_boost=4.0)),
        Case(vs.GRAY16, args=dict(thr=48, grain=16, seed=7, sample_mode=7, max_angle=0.5)),
        # per-plane thr/grain arrays (luma/chroma split) on YUV
        Case(vs.YUV420P16, args=dict(thr=[48, 24], grain=[16, 8], seed=7)),
        Case(vs.YUV444PS, args=dict(thr=[48, 24, 12], grain=[16, 8], seed=7)),
        # gaussian RNG with a non-default param
        Case(vs.GRAY16, args=dict(thr=48, grain=16, seed=7, random_algo_ref=2, random_param_ref=2.0)),
        Case(vs.GRAY16, args=dict(thr=48, grain=16, seed=7, random_algo_grain=2, random_param_grain=2.0)),
        # different seed -> different reference/grain pattern
        Case(vs.GRAY16, args=dict(thr=48, grain=16, seed=99)),
    ]
)


@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_cases(golden, make_clip, case):
    src = make_clip(case.fmt, case.geometry)
    golden.check("deband", case, src.vszip.Deband(**case.args))


@pytest.fixture(scope="module")
def banded16(to_gray):
    """Real image bit-crushed to 256 levels: strong banding in 16-bit."""
    return to_gray(vs.GRAY16).resize.Point(format=vs.GRAY8).resize.Point(format=vs.GRAY16)


@pytest.fixture(scope="module")
def banded_f32(banded16):
    return banded16.resize.Point(format=vs.GRAYS)


def test_passthrough_when_disabled(banded16, banded_f32):
    """thr=0 disables debanding, grain=0 disables grain: bit-exact passthrough."""
    assert_same_clip(banded16.vszip.Deband(thr=0, grain=0), banded16)
    assert_same_clip(banded_f32.vszip.Deband(thr=0, grain=0), banded_f32)


def test_deterministic(banded16):
    a = banded16.vszip.Deband(thr=48, grain=32, seed=7)
    b = banded16.vszip.Deband(thr=48, grain=32, seed=7)
    assert_same_clip(a, b)


SAMPLE_MODE_GOLDENS = {
    1: 0.48633140688181886,
    2: 0.4862626649881743,
    3: 0.4869006446936751,
    4: 0.48657496471351186,
    5: 0.48692404345006485,
    6: 0.4868622815903477,
    7: 0.4868109427303874,
}


@pytest.mark.parametrize("mode", range(1, 8))
def test_sample_modes_golden(banded16, mode):
    out = banded16.vszip.Deband(thr=48, sample_mode=mode, seed=7)
    assert avg(out) == pytest.approx(SAMPLE_MODE_GOLDENS[mode], rel=1e-6)
    assert diff(out, banded16) > 0.0


def test_float_golden(banded_f32):
    out = banded_f32.vszip.Deband(thr=48, sample_mode=2, seed=7)
    assert avg(out) == pytest.approx(0.4955954249984279, rel=1e-6)


def test_grain(banded16):
    out = banded16.vszip.Deband(thr=0, grain=64, seed=7)
    assert avg(out) == pytest.approx(0.48693080666878863, rel=1e-6)
    assert diff(out, banded16) > 0.0


def test_dynamic_grain(banded16):
    two = banded16 + banded16  # two identical frames
    static = two.vszip.Deband(thr=0, grain=64, seed=7, dynamic_grain=False)
    dynamic = two.vszip.Deband(thr=0, grain=64, seed=7, dynamic_grain=True)
    assert diff(static.std.Trim(0, 0), static.std.Trim(1, 1)) == 0.0
    assert diff(dynamic.std.Trim(0, 0), dynamic.std.Trim(1, 1)) > 0.0


def test_keep_tv_range(core):
    """Only applies to YUV: clamps luma to 60160 and chroma to 61440 max."""
    hi = core.std.BlankClip(None, 64, 32, vs.YUV420P16, length=1, color=[65000, 64000, 63000])
    lo = core.std.BlankClip(None, 64, 32, vs.YUV420P16, length=1, color=[64500, 63500, 62500])
    src = core.std.StackVertical([hi, lo])  # banding above the TV-range ceiling
    clamped = src.vszip.Deband(thr=48, seed=7, keep_tv_range=True)
    assert plane_stats(clamped, plane=0)["PlaneStatsMax"] == 60160
    assert plane_stats(clamped, plane=1)["PlaneStatsMax"] == 61440
    assert plane_stats(clamped, plane=2)["PlaneStatsMax"] == 61440
    free = src.vszip.Deband(thr=48, seed=7, keep_tv_range=False)
    assert plane_stats(free, plane=0)["PlaneStatsMax"] > 60160


@pytest.mark.parametrize("fmt", [vs.YUV444PS, vs.GRAYS, vs.RGBS])
def test_keep_tv_range_no_effect_on_float(core, fmt):
    """keep_tv_range is an integer-only option: on float formats it must be a pure
    no-op (float is always full-range, so there is nothing to clamp). Proven by
    bit-identical output with the flag on vs off."""
    color = {vs.YUV444PS: [1.0, 0.0, 0.0], vs.GRAYS: [1.0], vs.RGBS: [1.0, 1.0, 1.0]}[fmt]
    src = core.std.BlankClip(None, 64, 32, fmt, length=1, color=color)
    on = src.vszip.Deband(thr=48, grain=64, seed=7, keep_tv_range=True)
    off = src.vszip.Deband(thr=48, grain=64, seed=7, keep_tv_range=False)
    assert diff(on, off) == 0.0


def test_float_clamped_to_full_range(core):
    """Float deband/grain output is always clamped to full range (independent of
    keep_tv_range): luma/RGB to [0,1], chroma to [-0.5,0.5]. Pinning planes to the
    range edges and adding heavy grain would overflow without the clamp, so the
    output landing exactly on the edges proves the clamp engaged."""
    hi = core.std.BlankClip(None, 64, 32, vs.YUV444PS, length=1, color=[1.0, 0.5, 0.5])
    lo = core.std.BlankClip(None, 64, 32, vs.YUV444PS, length=1, color=[0.0, -0.5, -0.5])
    src = core.std.StackVertical([hi, lo])  # both range edges present in one plane
    out = src.vszip.Deband(thr=48, grain=96, seed=7)
    y, u, v = (plane_stats(out, plane=p) for p in (0, 1, 2))
    assert y["PlaneStatsMin"] == pytest.approx(0.0, abs=1e-9), y
    assert y["PlaneStatsMax"] == pytest.approx(1.0), y
    assert u["PlaneStatsMin"] == pytest.approx(-0.5) and u["PlaneStatsMax"] == pytest.approx(0.5), u
    assert v["PlaneStatsMin"] == pytest.approx(-0.5) and v["PlaneStatsMax"] == pytest.approx(0.5), v


def test_float_rgb_clamped_to_unit_range(core):
    """RGB float clamps every plane to [0,1] (no chroma planes)."""
    hi = core.std.BlankClip(None, 64, 32, vs.RGBS, length=1, color=[1.0, 1.0, 1.0])
    lo = core.std.BlankClip(None, 64, 32, vs.RGBS, length=1, color=[0.0, 0.0, 0.0])
    out = core.std.StackVertical([hi, lo]).vszip.Deband(thr=48, grain=96, seed=7)
    for p in (0, 1, 2):
        st = plane_stats(out, plane=p)
        assert st["PlaneStatsMin"] == pytest.approx(0.0, abs=1e-9), (p, st)
        assert st["PlaneStatsMax"] == pytest.approx(1.0), (p, st)


def test_8bit_roundtrip(banded16):
    """Sub-16-bit input is processed at 16 bits and dithered back."""
    src8 = banded16.resize.Point(format=vs.GRAY8)
    out = src8.vszip.Deband(thr=48, seed=7)
    assert out.format.id == vs.GRAY8
    assert avg(out) == pytest.approx(0.4881620902267157, rel=1e-3)


def test_random_algos_run(banded16):
    outs = [
        banded16.vszip.Deband(thr=48, grain=32, seed=7, random_algo_ref=a, random_algo_grain=a)
        for a in (0, 1, 2)
    ]
    assert diff(outs[0], outs[1]) > 0.0
    assert diff(outs[1], outs[2]) > 0.0


@pytest.mark.parametrize(
    ("args", "msg"),
    [
        (dict(sample_mode=8), r'parameter "sample_mode=8" out of range \[1\.\.7\]'),
        (dict(range=256), r'parameter "range=256" out of range \[0\.\.255\]'),
        (dict(max_angle=2.0), r'parameter "max_angle=2" out of range \[0\.\.1\]'),
        (dict(thr=[1, 2, 3, 4]), r'parameter "thr" has too many elements \(got 4, max 3\)'),
        (dict(grain=[1, 2, 3]), r'parameter "grain" has too many elements \(got 3, max 2\)'),
        (dict(thr=[300]), r'parameter "thr\[0\]=300" out of range \[0\.\.255\]'),
        # grain is capped at 127: higher values used to overflow the i16 grain buffer
        (dict(grain=200), r'parameter "grain\[0\]=200" out of range \[0\.\.127\]'),
        # thr1 / thr2 share the same array bounds as thr (max 3 elements, 0..255)
        (dict(thr1=[1, 2, 3, 4]), r'parameter "thr1" has too many elements \(got 4, max 3\)'),
        (dict(thr1=[300]), r'parameter "thr1\[0\]=300" out of range \[0\.\.255\]'),
        (dict(thr2=[1, 2, 3, 4]), r'parameter "thr2" has too many elements \(got 4, max 3\)'),
        (dict(thr2=[300]), r'parameter "thr2\[0\]=300" out of range \[0\.\.255\]'),
        (dict(thr=[-1]), r'parameter "thr\[0\]=-1" out of range \[0\.\.255\]'),
        (dict(grain=[-1]), r'parameter "grain\[0\]=-1" out of range \[0\.\.127\]'),
        # scalar range checks
        (dict(sample_mode=0), r'parameter "sample_mode=0" out of range \[1\.\.7\]'),
        (dict(range=-1), r'parameter "range=-1" out of range \[0\.\.255\]'),
        (dict(random_algo_ref=3), r'parameter "random_algo_ref=3" out of range \[0\.\.2\]'),
        (dict(random_algo_ref=-1), r'parameter "random_algo_ref=-1" out of range \[0\.\.2\]'),
        (dict(random_algo_grain=3), r'parameter "random_algo_grain=3" out of range \[0\.\.2\]'),
        (dict(random_param_ref=256), r'parameter "random_param_ref=256" out of range \[0\.\.255\]'),
        (dict(random_param_ref=-1), r'parameter "random_param_ref=-1" out of range \[0\.\.255\]'),
        (dict(random_param_grain=256), r'parameter "random_param_grain=256" out of range \[0\.\.255\]'),
        (dict(max_angle=-0.5), r'parameter "max_angle=-0.5" out of range \[0\.\.1\]'),
        (dict(angle_boost=-1.0), r'parameter "angle_boost=-1" out of range \[0\.\.65535\]'),
        (dict(angle_boost=70000.0), r'parameter "angle_boost=70000" out of range \[0\.\.65535\]'),
        (dict(random_param_grain=-1), r'parameter "random_param_grain=-1" out of range \[0\.\.255\]'),
        (dict(random_algo_grain=-1), r'parameter "random_algo_grain=-1" out of range \[0\.\.2\]'),
    ],
)
def test_validation_errors(banded16, args, msg):
    with pytest.raises(vs.Error, match=msg):
        banded16.vszip.Deband(**args)


def test_f16_error(core):
    src = core.std.BlankClip(None, 64, 64, vs.GRAYH, length=1)
    with pytest.raises(vs.Error, match="only 32-bit format is supported when float clip"):
        src.vszip.Deband()


def test_variable_format_error(core):
    a = core.std.BlankClip(None, 64, 64, vs.GRAY16, length=1)
    b = core.std.BlankClip(None, 64, 64, vs.GRAY8, length=1)
    variable = core.std.Splice([a, b], mismatch=True)
    with pytest.raises(vs.Error, match="clip must have constant format"):
        variable.vszip.Deband()


def test_grain_at_limit(core):
    """grain=127 is the largest allowed strength (127 * 257 still fits the
    i16 grain buffer) and must render without issue."""
    src = core.std.BlankClip(None, 64, 64, vs.GRAY16, length=1, color=32768)
    out = src.vszip.Deband(thr=0, grain=127, seed=7)
    assert diff(out, src) > 0.0
