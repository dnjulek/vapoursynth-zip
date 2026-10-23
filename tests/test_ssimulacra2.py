from concurrent.futures import ThreadPoolExecutor

import pytest
import vapoursynth as vs

from golden import Case, sweep
from helpers import props


@pytest.fixture(scope="module")
def src16(src_rgb):
    return src_rgb.resize.Bicubic(format=vs.YUV420P16, matrix=1)


def score(a: vs.VideoNode, b: vs.VideoNode, n: int = 0) -> float:
    return props(vs.core.vszip.SSIMULACRA2(a, b), n)["SSIMULACRA2"]


def distort(clip: vs.VideoNode, kind: str) -> vs.VideoNode:
    """Build a distorted copy of `clip`. Each kind degrades a different way so
    the resulting SSIMULACRA2 scores are mutually distinguishable."""
    if kind == "resize":  # bicubic 2x up then back down (ringing/blur)
        return clip.resize.Bicubic(clip.width * 2, clip.height * 2).resize.Bicubic(clip.width, clip.height)
    if kind == "blur1":
        return clip.std.BoxBlur(hradius=1, vradius=1)
    if kind == "blur3":
        return clip.std.BoxBlur(hradius=3, vradius=3)
    raise ValueError(f"unknown distortion {kind!r}")


# SSIMULACRA2 has no scalar args; the swept `dist` kwarg selects how the
# distorted clip is built (see distort()). The Create callback accepts any
# resize-convertible format: RGB (8-16 bit int via LUT, f32) is read natively,
# YUV (matrix) and GRAY (matrix gray->RGB) go through toRGBS first, and the
# sRGB decode always runs in the plugin. Sweep target formats spanning each
# path and the three distortions; GRAY is a distinct conversion path, so
# include GRAY8/GRAY16 below.
CASES = (
    sweep(
        base_fmt=vs.YUV420P16,
        base_args=dict(dist="blur1"),
        formats=[vs.YUV420P8, vs.YUV420P16, vs.RGB24, vs.RGBS, vs.GRAY8, vs.GRAY16],
        args=[dict(dist="resize"), dict(dist="blur1"), dict(dist="blur3")],
        geometries=["odd", "tiny"],
    )
    + [
        # hand-picked format x distortion interactions
        Case(vs.RGBS, args=dict(dist="resize")),
        Case(vs.RGB24, args=dict(dist="blur3")),
        Case(vs.YUV420P8, args=dict(dist="resize")),
        Case(vs.YUV420P16, args=dict(dist="blur3")),
        # GRAY color-family conversion path (gray->RGB via matrix)
        Case(vs.GRAY16, args=dict(dist="resize")),
        Case(vs.GRAY8, args=dict(dist="blur3")),
    ]
)


@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_cases(golden, make_clip, case):
    ref = make_clip(case.fmt, case.geometry)
    dist = distort(ref, case.args["dist"])
    s = props(vs.core.vszip.SSIMULACRA2(ref, dist))["SSIMULACRA2"]
    golden.check_value("ssimulacra2", case.id, s, rel=1e-3)


def test_identical_constant_clip(core):
    src = core.std.BlankClip(None, 64, 64, vs.YUV420P16, length=1, color=[30000, 20000, 40000])
    assert score(src, src) == 100.0


def test_identical_real_image(src16):
    assert score(src16, src16) > 99.9


def test_golden(src16):
    dist = src16.resize.Bicubic(src16.width * 2, src16.height * 2).resize.Bicubic(src16.width, src16.height)
    assert score(src16, dist) == pytest.approx(68.47905812796854, rel=1e-3)


def test_more_distortion_scores_lower(src16):
    blur1 = src16.std.BoxBlur(hradius=1, vradius=1)
    blur3 = src16.std.BoxBlur(hradius=3, vradius=3)
    s_ident, s1, s3 = score(src16, src16), score(src16, blur1), score(src16, blur3)
    assert s_ident > s1 > s3


def _rgb_pair(src_rgb, fmt):
    ref = src_rgb.resize.Point(format=fmt)
    return ref, src_rgb.std.BoxBlur(hradius=2, vradius=2).resize.Point(format=fmt)


def test_yuv_matches_manual_rgbs(src16):
    """YUV is converted to RGB inside the filter (zimg's steps: range, Bicubic
    chroma upsampling, matrix; values within ~1e-7 of resize), then the
    in-plugin sRGB decode: the score of converting to (still sRGB-encoded)
    RGBS by hand. Not bit-exact: 1e-7 of input noise alone moves a score by
    ~1e-4."""
    dist = src16.resize.Bicubic(src16.width * 2, src16.height * 2).resize.Bicubic(src16.width, src16.height)
    manual = score(src16.resize.Bicubic(format=vs.RGBS), dist.resize.Bicubic(format=vs.RGBS))
    assert score(src16, dist) == pytest.approx(manual, abs=2e-3)


@pytest.mark.parametrize("fmt", [vs.YUV420P8, vs.YUV422P10, vs.YUV444PS, vs.YUV440P8])
@pytest.mark.parametrize("loc", [None, 1, 2, 5])
@pytest.mark.parametrize("matrix", [None, 6, 9])
def test_yuv_conversion_matches_resize(src_rgb, fmt, loc, matrix):
    """The in-filter YUV -> RGB follows resize's frame props (_Matrix over the
    height rule, _ChromaLocation, _Range): scoring a YUV clip against its own
    resize.Bicubic RGBS conversion lands at the ~1e-7 noise floor (a 1e-7
    perturbation of an RGBS clip scores ~99.5)."""
    yuv = src_rgb.resize.Bicubic(format=fmt, matrix=1)
    props = {}
    if loc is not None:
        props["_ChromaLocation"] = loc
    if matrix is not None:
        props["_Matrix"] = matrix
    if props:
        yuv = yuv.std.SetFrameProps(**props)
    default = 1 if yuv.height > 650 else 6
    rgb = yuv.resize.Bicubic(format=vs.RGBS, matrix_in=default)
    assert score(yuv, rgb) > 99.4


@pytest.mark.parametrize("fmt", [vs.RGB24, vs.RGB30, vs.RGB48])
def test_native_int_rgb_matches_float_path(src_rgb, fmt):
    """Integer RGB is decoded through a sample -> linear LUT built from the
    float decode of v * (1 / peak) -- the exact f32 values zimg produces for
    the int -> RGBS conversion, so both paths give the identical score."""
    ref, dist = _rgb_pair(src_rgb, fmt)
    assert score(ref, dist) == score(ref.resize.Point(format=vs.RGBS), dist.resize.Point(format=vs.RGBS))


def test_out_of_range_float_is_clamped(src_rgb):
    """Encoded float values outside [0, 1] (YUV -> RGB overshoot) clamp
    before the decode, as libjxl's PNG input would, instead of taking its odd
    extension / out-of-fit polynomial."""
    ref, dist = _rgb_pair(src_rgb, vs.RGBS)
    ref, dist = ref.std.Expr("x 2 * 0.4 -"), dist.std.Expr("x 2 * 0.4 -")
    clamp = "x 0 max 1 min"
    # the input really leaves [0, 1] at both ends, so equality pins both bounds
    st = ref.std.PlaneStats().get_frame(0).props
    assert st["PlaneStatsMin"] < 0 and st["PlaneStatsMax"] > 1
    assert score(ref, dist) == score(ref.std.Expr(clamp), dist.std.Expr(clamp))


def test_linear_tagged_input_skips_decode(src_rgb):
    """_Transfer=8 (linear) input is used as is: zimg-linearized input lands
    within the zimg-vs-libjxl decode difference of the internal path, while
    decoding it a second time would be far off."""
    ref, dist = _rgb_pair(src_rgb, vs.RGBS)
    lin = [c.std.SetFrameProps(_Transfer=13).resize.Point(transfer=8) for c in (ref, dist)]
    internal = score(ref, dist)
    assert score(*lin) == pytest.approx(internal, abs=0.02)
    redecoded = [c.std.SetFrameProps(_Transfer=13) for c in lin]
    assert abs(score(*redecoded) - internal) > 0.5


def test_output_passes_input_through(src16, src_rgb):
    """Only the frame prop matters: the output is the reference frame -- RGB
    and YUV (converted inside the filter) as is, other families (GRAY) as the
    toRGBS clip."""
    rgb = src_rgb.resize.Point(format=vs.RGB24)
    assert vs.core.vszip.SSIMULACRA2(rgb, rgb).format.id == vs.RGB24
    out = vs.core.vszip.SSIMULACRA2(src16, src16)
    assert out.format.id == vs.YUV420P16
    assert (out.width, out.height, out.num_frames) == (src16.width, src16.height, src16.num_frames)
    gray = src16.std.ShufflePlanes(0, vs.GRAY)
    assert vs.core.vszip.SSIMULACRA2(gray, gray).format.id == vs.RGBS


def test_concurrent_requests_match_sequential(temporal_rgb):
    """Guard the filter's fmParallel contract with real concurrent requests."""
    ref = temporal_rgb.resize.Bicubic(format=vs.YUV420P16, matrix=1)
    distorted = ref.std.BoxBlur(hradius=2, vradius=2)
    sequential = vs.core.vszip.SSIMULACRA2(ref, distorted)
    concurrent = vs.core.vszip.SSIMULACRA2(ref, distorted)
    expected = [props(sequential, n)["SSIMULACRA2"] for n in range(ref.num_frames)]

    def get_score(n):
        return props(concurrent, n)["SSIMULACRA2"]

    with ThreadPoolExecutor(max_workers=ref.num_frames) as pool:
        actual = list(pool.map(get_score, reversed(range(ref.num_frames))))
    assert actual == pytest.approx(list(reversed(expected)), rel=1e-12)


def test_dimension_error(core):
    a = core.std.BlankClip(None, 64, 32, vs.YUV420P16, length=1)
    with pytest.raises(vs.Error, match="clips must have the same dimensions"):
        core.vszip.SSIMULACRA2(a, a.std.Crop(left=8))


def test_length_error(core):
    a = core.std.BlankClip(None, 64, 32, vs.YUV420P16, length=5)
    with pytest.raises(vs.Error, match="clips must have the same length"):
        core.vszip.SSIMULACRA2(a, a.std.Trim(0, 2))


# Every validation branch in ssimulacraCreate: mismatched dimensions and
# mismatched length are the only two checks (the filter accepts any
# resize-convertible format/bit-depth).
@pytest.mark.parametrize(
    ("distorted", "msg"),
    [
        (lambda a: a.std.Crop(left=8), "clips must have the same dimensions"),
        (lambda a: a.std.AddBorders(top=2), "clips must have the same dimensions"),
        (lambda a: a.std.Trim(0, 2), "clips must have the same length"),
    ],
)
def test_validation_errors(core, distorted, msg):
    a = core.std.BlankClip(None, 64, 32, vs.YUV420P16, length=5)
    with pytest.raises(vs.Error, match=msg):
        core.vszip.SSIMULACRA2(a, distorted(a))
