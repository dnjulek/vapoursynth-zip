import pytest
import vapoursynth as vs

from golden import Case, grid, sweep
from helpers import max_abs_diff, props, write_png

# Reference implementation: kageru's Rust adaptivegrain plugin (`adg.Mask`),
# installed in the venv. Integer output must be bit-exact with it; float within
# ~1 ULP of the vectorized pow's ~1e-7 error bound. The Rust plugin reads the
# PlaneStatsAverage frame prop, ours computes the average internally, so a
# match also proves the internal average equals std.PlaneStats.
HAS_ADG = hasattr(vs.core, "adg")

CASES = (
    sweep(
        base_fmt=vs.GRAY8,
        base_args=dict(),
        # u8 / u16 (10 & 16 bit exercise the >>shift LUT index) / f32, plus
        # subsampled YUV inputs: only luma is read, output is Gray.
        formats=[vs.GRAY8, vs.GRAY10, vs.GRAY16, vs.GRAYS, vs.YUV420P8, vs.YUV420P10, vs.YUV444PS],
        # luma_scaling shapes the curve; negative blows up magnitudes.
        args=grid(luma_scaling=[0.5, 5.0, 30.0]) + [dict(luma_scaling=-5.0)],
        geometries=["odd", "tiny"],
    )
    + [
        Case(vs.GRAYS, geometry="odd"),
        Case(vs.GRAYS, geometry="tiny"),
        Case(vs.GRAY16, geometry="odd"),
        Case(vs.GRAYS, args=dict(luma_scaling=30.0)),
        Case(vs.GRAY16, args=dict(luma_scaling=-5.0)),
    ]
)


@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_cases(golden, make_clip, case):
    src = make_clip(case.fmt, case.geometry)
    out = src.vszip.AdaptiveGrainMask(**case.args)
    golden.check("adaptive_grain_mask", case, out)


def test_output_is_gray_same_depth(make_clip):
    for fmt, expect in [
        (vs.YUV420P8, vs.GRAY8),
        (vs.YUV420P10, vs.GRAY10),
        (vs.GRAY16, vs.GRAY16),
        (vs.YUV444PS, vs.GRAYS),
    ]:
        out = make_clip(fmt).vszip.AdaptiveGrainMask()
        assert out.format.id == expect, vs.PresetVideoFormat(out.format.id).name


def test_mask_is_monotonic_in_brightness(core, tmp_path):
    """Brighter pixels get lower mask values (the whole point of the filter)."""
    ramp = [list(range(256)) for _ in range(2)]
    src = core.vszip.ImageRead(str(write_png(tmp_path / "ramp.png", ramp, color="gray")))
    out = src.vszip.AdaptiveGrainMask()
    with out.get_frame(0) as f:
        row = [f[0][0, x] for x in range(256)]
    assert row[0] == 255
    assert all(a >= b for a, b in zip(row, row[1:]))


@pytest.mark.parametrize(
    "fmt", [vs.GRAY8, vs.GRAY10, vs.GRAY16, vs.YUV420P8, vs.YUV420P10],
    ids=lambda f: vs.PresetVideoFormat(f).name,
)
@pytest.mark.parametrize("luma_scaling", [0.5, 10.0, 30.0, -5.0])
@pytest.mark.skipif(not HAS_ADG, reason="Rust adg plugin not installed")
def test_rust_parity_int_bitexact(make_clip, fmt, luma_scaling):
    src = make_clip(fmt)
    ours = src.vszip.AdaptiveGrainMask(luma_scaling=luma_scaling)
    rust = src.std.PlaneStats().adg.Mask(luma_scaling=luma_scaling)
    assert ours.format.id == rust.format.id
    assert max_abs_diff(ours, rust) == 0


@pytest.mark.parametrize("fmt", [vs.GRAYS, vs.YUV444PS], ids=lambda f: vs.PresetVideoFormat(f).name)
@pytest.mark.parametrize("luma_scaling", [0.5, 10.0, 30.0, -5.0])
@pytest.mark.skipif(not HAS_ADG, reason="Rust adg plugin not installed")
def test_rust_parity_float(make_clip, fmt, luma_scaling):
    np = pytest.importorskip("numpy")
    src = make_clip(fmt)
    ours = src.vszip.AdaptiveGrainMask(luma_scaling=luma_scaling)
    rust = src.std.PlaneStats().adg.Mask(luma_scaling=luma_scaling)
    with ours.get_frame(0) as fa, rust.get_frame(0) as fb:
        a = np.asarray(fa[0]).astype(np.float64)
        b = np.asarray(fb[0]).astype(np.float64)
    # abs OR rel tolerance, like the adgz reference harness: the vectorized pow
    # is within ~1e-7 abs of libm; negative luma_scaling blows up magnitudes,
    # so large values are judged relatively.
    err = np.abs(a - b)
    ok = (err <= 1e-6) | (err <= 2e-6 * np.abs(b))
    assert ok.all(), f"max abs {err.max()}, worst rel {(err / np.maximum(np.abs(b), 1e-30)).max()}"


@pytest.mark.skipif(not HAS_ADG, reason="Rust adg plugin not installed")
def test_rust_parity_default_luma_scaling(make_clip):
    src = make_clip(vs.GRAY8)
    ours = src.vszip.AdaptiveGrainMask()
    rust = src.std.PlaneStats().adg.Mask()
    assert max_abs_diff(ours, rust) == 0


def test_props_passthrough(make_clip):
    out = make_clip(vs.YUV420P8).vszip.AdaptiveGrainMask()
    # output frame inherits the source frame's props (VS copies them on newVideoFrame)
    assert "_Matrix" in props(out)


@pytest.mark.parametrize(
    ("fmt", "msg"),
    [
        (vs.GRAYH, "half precision float input is not supported"),
        (vs.RGB24, "not supported"),  # RGB has no luma; DataType.select still
        # accepts it (int) — see explicit family check below if this changes
    ],
)
def test_validation_errors(make_clip, fmt, msg):
    if fmt == vs.RGB24:
        pytest.skip("RGB accepted like the Rust original (plane 0 used as luma)")
    src = make_clip(fmt)
    with pytest.raises(vs.Error, match=msg):
        src.vszip.AdaptiveGrainMask()
