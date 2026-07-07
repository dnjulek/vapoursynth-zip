import pytest
import vapoursynth as vs

from golden import Case, grid, sweep
from helpers import max_abs_diff, props

# Reference implementation: fmtconv's fmtc.bitdepth (C++). The fmtc dither modes
# must be bit-exact with it (the full sweep at port time was 1426/1426 exact).
# The oracle tests below run whenever the fmtc plugin is installed (autoloaded).
HAS_FMTC = hasattr(vs.core, "fmtc")

ALL_MODES = list(range(12))
# vszip dither_type -> fmtc dmode (fmtc has no equivalent for the zimg modes)
FMTC_DMODE = {0: 1, 4: 0, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9}

CASES = (
    sweep(
        base_fmt=vs.GRAY16,
        base_args=dict(bitdepth=8),  # default dither_type = zimg random (blue noise)
        # int depths both containers, f16/f32, subsampled + RGB inputs
        formats=[vs.GRAY8, vs.GRAY10, vs.GRAY14, vs.GRAY16, vs.GRAYS, vs.GRAYH,
                 vs.YUV420P10, vs.YUV444P16, vs.RGB24, vs.RGB48],
        args=grid(dither_type=ALL_MODES)
        + grid(bitdepth=[9, 10, 12, 14])  # incl. 14: fmtc cannot output it
        + [
            dict(bitdepth=32),                  # int -> f32 convert
            dict(bitdepth=16, sample_type=1),   # int -> f16 convert
            dict(bitdepth=8, fulls=1, fulld=1),   # full->full: float kernel
            dict(bitdepth=8, fulls=0, fulld=1),   # range conversion
            dict(bitdepth=8, fulls=1, fulld=0),
            dict(bitdepth=8, dither_type=9, fulls=1, fulld=0),  # errdiff float path
        ],
        geometries=["odd", "tiny"],
    )
    + [
        # upconversions: limited->limited is a pure shift, full->full dithers
        Case(vs.GRAY8, args=dict(bitdepth=16)),
        Case(vs.GRAY8, args=dict(bitdepth=16, fulls=1, fulld=1)),
        Case(vs.GRAY8, args=dict(bitdepth=16, dither_type=9, fulls=1, fulld=1)),
        # float passthrough / float<->float conversions
        Case(vs.GRAYS, args=dict(bitdepth=32)),
        Case(vs.GRAYH, args=dict(bitdepth=32)),
        Case(vs.GRAYS, args=dict(bitdepth=16, sample_type=1)),
        Case(vs.YUV444PS, args=dict(bitdepth=10)),
        # geometry x mode interactions: dst/src element strides differ at
        # tiny widths (u16 in, u8 out), zimg wavefront w<14 fallback,
        # serpentine on odd widths, chroma plane routing
        Case(vs.GRAY16, "tiny", dict(bitdepth=8, dither_type=4)),
        Case(vs.GRAY16, "tiny", dict(bitdepth=8, dither_type=3)),
        Case(vs.GRAY16, "odd", dict(bitdepth=8, dither_type=3)),
        Case(vs.GRAY16, "odd", dict(bitdepth=8, dither_type=11)),
        Case(vs.YUV420P10, "odd", dict(bitdepth=8, dither_type=11)),
        Case(vs.YUV420P10, "tiny", dict(bitdepth=8, dither_type=6)),
        Case(vs.YUV420P16, args=dict(bitdepth=10, dither_type=10)),
    ]
)


@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_cases(golden, make_clip, case):
    src = make_clip(case.fmt, case.geometry)
    out = src.vszip.Dither(**case.args)
    golden.check("dither", case, out)


@pytest.mark.parametrize(
    "fmt", [vs.GRAY8, vs.GRAY16, vs.GRAYS, vs.YUV420P8, vs.YUV420P10, vs.RGB24, vs.RGB48],
    ids=lambda f: vs.PresetVideoFormat(f).name,
)
@pytest.mark.parametrize("mode", list(range(12)), ids=lambda m: f"mode{m}")
def test_identity_noop(make_clip, fmt, mode):
    """An identity conversion (same depth + sample type + range) is a pure
    passthrough for EVERY mode — zimg's `pixel_in == pixel_out` no-op, applied
    before the dither dispatch. We follow zimg here, not fmtconv (which applies
    the dither pattern at equal bit depth); dithering to the same depth only
    adds noise."""
    src = make_clip(fmt)
    bits = src.format.bits_per_sample
    st = 1 if src.format.sample_type == vs.FLOAT else 0
    out = src.vszip.Dither(bitdepth=bits, sample_type=st, dither_type=mode)
    assert out.format.id == src.format.id
    assert max_abs_diff(src, out) == 0


# ---------------------------------------------------------------------------
# zimg-mode parity: dither_type 1/2/3 are ports of zimg's depth dither and must
# be bit-exact with VapourSynth resize.Point's ordered/random/error_diffusion
# (resize is core VS = zimg, always available). Per-pixel max_abs_diff==0 is
# far more sensitive than the golden per-plane stats, which can't see a
# few-row 1-LSB change. "odd" geometry gives h%8!=0, exercising the
# error-diffusion band->tail error carry (the case that diverged from zimg).
# ---------------------------------------------------------------------------

ZIMG_DT = {1: "ordered", 2: "random", 3: "error_diffusion"}


@pytest.mark.parametrize(
    "fmt",
    [vs.GRAY16, vs.GRAY10, vs.GRAYS, vs.YUV420P16, vs.YUV444P16, vs.YUV420P10, vs.RGB48],
    ids=lambda f: vs.PresetVideoFormat(f).name,
)
@pytest.mark.parametrize("mode", [1, 2, 3], ids=lambda m: ZIMG_DT[m])
@pytest.mark.parametrize("geometry", ["full", "odd"])
def test_zimg_parity(make_clip, core, fmt, mode, geometry):
    src = make_clip(fmt, geometry)
    out_fmt = core.query_video_format(
        src.format.color_family, vs.INTEGER, 8, src.format.subsampling_w, src.format.subsampling_h
    )
    ref = src.resize.Point(format=out_fmt.id, dither_type=ZIMG_DT[mode])
    got = src.vszip.Dither(bitdepth=8, dither_type=mode)
    assert got.format.id == ref.format.id
    assert max_abs_diff(ref, got) == 0


@pytest.mark.parametrize(
    "fmt", [vs.GRAY16, vs.YUV420P16, vs.RGB48], ids=lambda f: vs.PresetVideoFormat(f).name
)
def test_zimg_parity_upconv_and_f16(make_clip, core, fmt):
    """Non-dithered zimg paths: limited->limited upconvert is a pure shift, and
    int->f16 is a pure convert — both bit-exact with resize regardless of mode."""
    src = make_clip(fmt)
    # upconvert 16->... no; use an 8-bit source shifted up
    hf = core.query_video_format(src.format.color_family, vs.FLOAT, 16, src.format.subsampling_w, src.format.subsampling_h)
    f32 = core.query_video_format(src.format.color_family, vs.FLOAT, 32, src.format.subsampling_w, src.format.subsampling_h)
    for m in (1, 2, 3):
        ref = src.resize.Point(format=hf.id, dither_type=ZIMG_DT[m])
        got = src.vszip.Dither(bitdepth=16, sample_type=1, dither_type=m)
        assert got.format.id == ref.format.id
        # PlaneStats has no f16 support; widen both to f32 (lossless, identical
        # for both) before comparing.
        assert max_abs_diff(ref.resize.Point(format=f32.id), got.resize.Point(format=f32.id)) == 0


def test_output_formats(make_clip):
    for fmt, kwargs, expect in [
        (vs.GRAY16, dict(bitdepth=8), vs.GRAY8),
        (vs.GRAY16, dict(bitdepth=10), vs.GRAY10),
        (vs.GRAY16, dict(bitdepth=14), vs.GRAY14),
        (vs.GRAY16, dict(bitdepth=32), vs.GRAYS),
        (vs.GRAY16, dict(bitdepth=16, sample_type=1), vs.GRAYH),
        (vs.GRAYS, dict(bitdepth=16), vs.GRAY16),
        (vs.YUV420P10, dict(bitdepth=8), vs.YUV420P8),
        (vs.RGB48, dict(bitdepth=8), vs.RGB24),
    ]:
        out = make_clip(fmt).vszip.Dither(**kwargs)
        assert out.format.id == expect, out.format.name


def test_upconv_limited_is_pure_shift(make_clip):
    src = make_clip(vs.GRAY8)
    out = src.vszip.Dither(bitdepth=16)
    ref = src.std.Expr("x 256 *", format=vs.GRAY16)
    assert max_abs_diff(out, ref) == 0


def test_float_passthrough_is_exact(make_clip):
    src = make_clip(vs.GRAYS)
    out = src.vszip.Dither(bitdepth=32)
    assert out.format.id == vs.GRAYS
    assert max_abs_diff(out, src) == 0


def test_round_trip_f32(make_clip):
    """GRAY16 -> f32 -> GRAY16 round mode is lossless."""
    src = make_clip(vs.GRAY16)
    back = src.vszip.Dither(bitdepth=32).vszip.Dither(bitdepth=16, dither_type=0)
    assert max_abs_diff(src, back) == 0


def test_color_range_prop(make_clip):
    """fmtc semantics: _ColorRange is stamped only when fulls/fulld was given
    explicitly, otherwise the source prop passes through. Values are compared
    relative to the (limited) source so the test is agnostic to the VS
    _ColorRange/_Range encoding shim."""
    src = make_clip(vs.GRAY16)
    src_cr = props(src)["_ColorRange"]
    assert props(src.vszip.Dither(bitdepth=8))["_ColorRange"] == src_cr
    limited = props(src.vszip.Dither(bitdepth=8, fulld=0))["_ColorRange"]
    full = props(src.vszip.Dither(bitdepth=8, fulld=1))["_ColorRange"]
    assert limited == src_cr
    assert full != limited
    # fulls alone also stamps (fulld defaults to fulls)
    assert props(src.vszip.Dither(bitdepth=8, fulls=0))["_ColorRange"] == limited


def test_validation_errors(make_clip):
    src = make_clip(vs.GRAY16)
    with pytest.raises(vs.Error, match="bitdepth"):
        src.vszip.Dither(bitdepth=7)
    with pytest.raises(vs.Error, match="bitdepth"):
        src.vszip.Dither(bitdepth=17)
    with pytest.raises(vs.Error, match="bitdepth"):
        src.vszip.Dither(bitdepth=32, sample_type=0)
    with pytest.raises(vs.Error, match="bitdepth"):
        src.vszip.Dither(bitdepth=8, sample_type=1)
    with pytest.raises(vs.Error, match="dither_type"):
        src.vszip.Dither(bitdepth=8, dither_type=12)


# ---------------------------------------------------------------------------
# fmtconv oracle: bit-exact parity on everything fmtc can express
# ---------------------------------------------------------------------------

ORACLE_RANGES = [(None, None), (0, 0), (1, 1), (0, 1), (1, 0)]


@pytest.mark.skipif(not HAS_FMTC, reason="fmtconv plugin not available")
@pytest.mark.parametrize("fmt", [vs.GRAY16, vs.GRAYS, vs.YUV420P10, vs.RGB24],
                         ids=lambda f: vs.PresetVideoFormat(f).name)
@pytest.mark.parametrize("bits", [8, 10])
@pytest.mark.parametrize("mode", sorted(FMTC_DMODE), ids=lambda m: f"mode{m}")
def test_fmtc_parity_modes(make_clip, fmt, bits, mode):
    src = make_clip(fmt)
    if mode == 0 and src.format.sample_type == vs.FLOAT:
        pytest.skip("round on float sources uses the faster zimg-none kernel (test_round_mode_picks_fastest)")
    if src.format.sample_type == vs.INTEGER and bits == src.format.bits_per_sample and src.format.color_family == vs.RGB:
        pytest.skip("full-range same-format is a no-op for all modes (zimg design; test_identity_noop)")
    ref = src.fmtc.bitdepth(bits=bits, dmode=FMTC_DMODE[mode])
    got = src.vszip.Dither(bitdepth=bits, dither_type=mode)
    assert got.format.id == ref.format.id
    assert max_abs_diff(ref, got) == 0


@pytest.mark.skipif(not HAS_FMTC, reason="fmtconv plugin not available")
@pytest.mark.parametrize("fulls,fulld", ORACLE_RANGES, ids=lambda r: str(r))
# mode 0 (round) excluded: full-range / range-converting round goes through the
# faster zimg-none float kernel, not fmtc's — covered by test_round_mode_picks_fastest.
@pytest.mark.parametrize("mode", [5, 9, 11], ids=lambda m: f"mode{m}")
def test_fmtc_parity_ranges(make_clip, mode, fulls, fulld):
    src = make_clip(vs.GRAY16)
    kw = {}
    if fulls is not None:
        kw["fulls"] = fulls
    if fulld is not None:
        kw["fulld"] = fulld
    ref = src.fmtc.bitdepth(bits=8, dmode=FMTC_DMODE[mode], **kw)
    got = src.vszip.Dither(bitdepth=8, dither_type=mode, **kw)
    assert max_abs_diff(ref, got) == 0


@pytest.mark.skipif(not HAS_FMTC, reason="fmtconv plugin not available")
def test_round_mode_picks_fastest(make_clip, core):
    """Round (dither_type=0) uses the fastest kernel per input (measured
    2026-07-07): the fmtc integer kernel (round-half-up, 1.5-2.4x faster than the
    float path) for limited int->int reductions, and the zimg-style FMA float
    kernel (round-half-even == resize.Point 'none', 1.2-1.34x faster than fmtc's
    -32768 bias kernel) for float sources and full-range / range-converting
    paths. So round matches fmtc on the integer path and zimg-none elsewhere."""
    # limited int->int reduction -> fmtc round (fast integer kernel)
    for fmt, bits in ((vs.GRAY16, 8), (vs.YUV420P10, 8), (vs.GRAY16, 10)):
        src = make_clip(fmt)
        assert max_abs_diff(
            src.vszip.Dither(bitdepth=bits, dither_type=0), src.fmtc.bitdepth(bits=bits, dmode=1)
        ) == 0
    # float source & full-range int -> zimg 'none' (fast FMA float kernel)
    for fmt in (vs.GRAYS, vs.RGB48):
        src = make_clip(fmt)
        of = core.query_video_format(
            src.format.color_family, vs.INTEGER, 8, src.format.subsampling_w, src.format.subsampling_h
        )
        assert max_abs_diff(
            src.vszip.Dither(bitdepth=8, dither_type=0), src.resize.Point(format=of.id, dither_type="none")
        ) == 0


@pytest.mark.skipif(not HAS_FMTC, reason="fmtconv plugin not available")
@pytest.mark.parametrize("fmt", [vs.GRAY8, vs.GRAY16, vs.YUV420P10],
                         ids=lambda f: vs.PresetVideoFormat(f).name)
def test_fmtc_parity_float_out(make_clip, fmt):
    src = make_clip(fmt)
    ref = src.fmtc.bitdepth(bits=32, flt=1)
    got = src.vszip.Dither(bitdepth=32)
    assert got.format.id == ref.format.id
    assert max_abs_diff(ref, got) == 0


@pytest.mark.skipif(not HAS_FMTC, reason="fmtconv plugin not available")
def test_fmtc_parity_upconv(make_clip):
    src = make_clip(vs.GRAY8)
    for kw in [dict(), dict(fulls=1, fulld=1)]:
        ref = src.fmtc.bitdepth(bits=16, dmode=3, **kw)
        got = src.vszip.Dither(bitdepth=16, dither_type=5, **kw)
        assert max_abs_diff(ref, got) == 0
