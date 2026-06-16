import pytest
import vapoursynth as vs

from golden import Case, grid, sweep
from helpers import assert_binary, avg, props, write_png


# clip2 is a required argument: the threshold rule is `src - clip2 <= -c`, so
# every golden case builds a blurred companion clip. The `variant` tag selects
# how that companion is produced (default BoxBlur(5) vs the wider "wide" blur),
# which measurably shifts the binarization relative to the base case.
def _clip2(src, variant: str = ""):
    r = 12 if variant == "wide" else 5
    return src.std.BoxBlur(hradius=r, vradius=r)


CASES = (
    sweep(
        base_fmt=vs.GRAY8,
        base_args=dict(c=3),
        # only 8-bit int is accepted; GRAY8 (luma) + YUV420P8 (multi-plane,
        # subsampled chroma) cover the per-plane loop on both color families.
        formats=[vs.GRAY8, vs.YUV420P8],
        # c walks the threshold offset; each value yields a distinct mask.
        args=grid(c=[0, 3, 6, 12]) + [dict(c=-5)],
        geometries=["odd", "tiny"],
    )
    + [
        # a wider clip2 blur shifts the local mean, changing which pixels pass.
        Case(vs.GRAY8, args=dict(c=3), variant="wide"),
        Case(vs.YUV420P8, args=dict(c=6), variant="wide"),
        # interacting combos: extreme c on the multi-plane format + geometries.
        Case(vs.YUV420P8, args=dict(c=0)),
        Case(vs.YUV420P8, args=dict(c=12)),
        Case(vs.YUV420P8, geometry="odd", args=dict(c=6)),
        Case(vs.YUV420P8, geometry="tiny", args=dict(c=6)),
        Case(vs.GRAY8, geometry="odd", args=dict(c=12)),
        Case(vs.GRAY8, geometry="tiny", args=dict(c=12)),
        # RGB24: the Create check only gates sampleType==Integer && bits==8, so
        # the RGB color family is also accepted. It exercises the third,
        # otherwise-uncovered numPlanes loop path (3 planes, no subsampling) and
        # the FULL color-range prop on an RGB frame.
        Case(vs.RGB24, args=dict(c=3)),
        Case(vs.RGB24, args=dict(c=6)),
        Case(vs.RGB24, args=dict(c=3), variant="wide"),
        Case(vs.RGB24, geometry="odd", args=dict(c=3)),
        Case(vs.RGB24, geometry="tiny", args=dict(c=3)),
    ]
)


@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_cases(golden, make_clip, case):
    src = make_clip(case.fmt, case.geometry)
    out = src.vszip.AdaptiveBinarize(_clip2(src, case.variant), **case.args)
    golden.check("adaptive_binarize", case, out)


def test_golden(to_gray):
    src = to_gray(vs.GRAY8)
    blur = src.std.BoxBlur(hradius=5, vradius=5)
    out = src.vszip.AdaptiveBinarize(blur)
    assert avg(out) == pytest.approx(0.45087890625, rel=1e-6)


def test_output_is_binary_and_full_range(to_gray):
    src = to_gray(vs.GRAY8)
    out = src.vszip.AdaptiveBinarize(src.std.BoxBlur(hradius=5, vradius=5))
    assert_binary(out)
    assert props(out)["_ColorRange"] == vs.Range.RANGE_FULL


@pytest.mark.parametrize("c", [0, 3, 10])
def test_threshold_rule_exact(core, tmp_path, c):
    """out = 255 where src <= clip2 - c, else 0 (OpenCV ADAPTIVE_THRESH_MEAN_C
    with THRESH_BINARY_INV)."""
    ramp = [list(range(256)) for _ in range(2)]
    src = core.vszip.ImageRead(str(write_png(tmp_path / "ramp.png", ramp, color="gray")))
    blur = core.std.BlankClip(src, color=128)
    out = src.vszip.AdaptiveBinarize(blur, c=c)
    with out.get_frame(0) as f:
        row = [f[0][0, x] for x in range(256)]
    assert row == [255 if x <= 128 - c else 0 for x in range(256)]


def test_higher_c_is_stricter(to_gray):
    src = to_gray(vs.GRAY8)
    blur = src.std.BoxBlur(hradius=5, vradius=5)
    a3 = avg(src.vszip.AdaptiveBinarize(blur, c=3))
    a10 = avg(src.vszip.AdaptiveBinarize(blur, c=10))
    assert a10 < a3  # larger c marks fewer pixels


def test_non_8bit_error(to_gray):
    src16 = to_gray(vs.GRAY16)
    with pytest.raises(vs.Error, match="only 8 bit int format supported"):
        src16.vszip.AdaptiveBinarize(src16)


@pytest.mark.parametrize(
    ("build", "msg"),
    [
        # only 8-bit integer formats are accepted (Create's explicit check)
        (lambda g, y: (g(vs.GRAY16), g(vs.GRAY16)), "only 8 bit int format supported"),
        (lambda g, y: (g(vs.GRAYS), g(vs.GRAYS)), "only 8 bit int format supported"),
        # compareNodes (BIGGER_THAN) clip vs clip2 consistency checks
        (lambda g, y: (g(vs.GRAY8), g(vs.GRAY8).std.Crop(right=2)),
         "all input clips must have the same width and height"),
        (lambda g, y: (g(vs.GRAY8), y(vs.YUV444P8)),
         "all input clips must have the same color family"),
        (lambda g, y: (y(vs.YUV420P8), y(vs.YUV444P8)),
         "all input clips must have the same subsampling"),
        (lambda g, y: (g(vs.GRAY8), g(vs.GRAY16)),
         "all input clips must have the same bit depth"),
        (lambda g, y: (g(vs.GRAY8) + g(vs.GRAY8), g(vs.GRAY8)),
         "second clip has less frames than input clip"),
    ],
)
def test_validation_errors(to_gray, to_yuv, build, msg):
    clip, clip2 = build(to_gray, to_yuv)
    with pytest.raises(vs.Error, match=msg):
        clip.vszip.AdaptiveBinarize(clip2).get_frame(0)
