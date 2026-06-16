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
# resize-convertible format (everything is funneled through toRGBS+linearize),
# so sweep target formats spanning each accepted color family and the three
# distortions. toRGBS dispatches on color family: RGB (passthrough for RGBS,
# convert for RGB24/48), YUV (matrix), and GRAY (matrix gray->RGB) - the GRAY
# branch is a distinct conversion path, so include GRAY8/GRAY16 below.
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
    assert score(src16, dist) == pytest.approx(68.62493918303275, rel=1e-3)


def test_more_distortion_scores_lower(src16):
    blur1 = src16.std.BoxBlur(hradius=1, vradius=1)
    blur3 = src16.std.BoxBlur(hradius=3, vradius=3)
    s_ident, s1, s3 = score(src16, src16), score(src16, blur1), score(src16, blur3)
    assert s_ident > s1 > s3


def test_yuv_matches_prelinearized_rgbs(src16):
    """Internal YUV->RGBS->linear conversion equals doing it by hand."""
    dist = src16.resize.Bicubic(src16.width * 2, src16.height * 2).resize.Bicubic(src16.width, src16.height)
    rgbs = src16.resize.Bicubic(format=vs.RGBS).std.SetFrameProps(_Transfer=13)
    rgbs2 = dist.resize.Bicubic(format=vs.RGBS).std.SetFrameProps(_Transfer=13)
    linear = rgbs.resize.Bicubic(transfer=8)
    linear2 = rgbs2.resize.Bicubic(transfer=8)
    assert score(src16, dist) == score(linear, linear2)


def test_output_clip_is_rgbs(src16):
    out = vs.core.vszip.SSIMULACRA2(src16, src16)
    assert out.format.id == vs.RGBS
    assert (out.width, out.height) == (src16.width, src16.height)


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
