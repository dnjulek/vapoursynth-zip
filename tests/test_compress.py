import pytest
import vapoursynth as vs

from golden import Case, grid, sweep
from helpers import assert_same_clip, avg, diff, repack

# Compress runs the FFmpeg intra pipeline (FDCT -> quantize -> dequantize ->
# IDCT) per 8x8 block, so the artifacts are deterministic and bit-exact: ideal
# for goldens. It accepts only 8-bit integer Gray/YUV. codec=0 is MPEG-2
# (qscale 1..31, dc_prec 0..3); codec=1 is JPEG (quality 1..100). chroma is a
# YUV-only toggle. Each swept value lands a different quantizer on the real
# image, so every golden differs; odd/tiny geometries exercise the non-mod8
# edge-block replication path.
YUV8 = [vs.YUV420P8, vs.YUV422P8, vs.YUV444P8]

MPEG_CASES = (
    sweep(
        base_fmt=vs.GRAY8,
        base_args=dict(codec=0, qscale=8),
        formats=[vs.GRAY8] + YUV8,
        args=grid(qscale=[1, 4, 20, 31]) + grid(dc_prec=[1, 2, 3]),
        geometries=["odd", "tiny"],
    )
    + [
        # chroma=False leaves the U/V planes untouched (luma-only artifacts)
        Case(vs.YUV420P8, args=dict(codec=0, qscale=20, chroma=False)),
        Case(vs.YUV444P8, args=dict(codec=0, qscale=20, chroma=False)),
    ]
)

JPEG_CASES = sweep(
    base_fmt=vs.GRAY8,
    base_args=dict(codec=1, quality=25),
    formats=[vs.GRAY8] + YUV8,
    args=grid(quality=[8, 50, 98]),
    geometries=["odd", "tiny"],
) + [
    Case(vs.YUV420P8, args=dict(codec=1, quality=25, chroma=False)),
]

CASES = MPEG_CASES + JPEG_CASES


@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_cases(golden, make_clip, case):
    src = make_clip(case.fmt, case.geometry)
    golden.check("compress", case, src.vszip.Compress(**case.args))


@pytest.fixture(scope="module")
def gray8(make_clip):
    return make_clip(vs.GRAY8)


# --- behavioral checks (ported from zmpeg's integration test) ---------------


def test_defaults_are_mpeg_qscale8(gray8):
    assert_same_clip(gray8.vszip.Compress(), gray8.vszip.Compress(codec=0, qscale=8))


def test_mpeg_coarser_qscale_more_error(gray8):
    q1 = diff(gray8.vszip.Compress(codec=0, qscale=1), gray8)
    q31 = diff(gray8.vszip.Compress(codec=0, qscale=31), gray8)
    assert q1 < q31


def test_jpeg_higher_quality_is_closer(gray8):
    q8 = diff(gray8.vszip.Compress(codec=1, quality=8), gray8)
    q98 = diff(gray8.vszip.Compress(codec=1, quality=98), gray8)
    assert q98 < q8


def test_brightness_preserved(gray8):
    src_avg = avg(gray8)
    for out in (gray8.vszip.Compress(codec=0, qscale=31), gray8.vszip.Compress(codec=1, quality=8)):
        # overall brightness within ~6 8-bit levels of the source
        assert avg(out) == pytest.approx(src_avg, abs=6 / 255)


def test_chroma_false_leaves_chroma_untouched(make_clip):
    src = make_clip(vs.YUV420P8)
    out = src.vszip.Compress(codec=0, qscale=20, chroma=False)
    assert diff(out, src, plane=0) > 0.0  # luma still processed
    assert diff(out, src, plane=1) == 0.0
    assert diff(out, src, plane=2) == 0.0


def test_chroma_true_modifies_chroma(make_clip):
    src = make_clip(vs.YUV420P8)
    out = src.vszip.Compress(codec=0, qscale=20, chroma=True)
    assert diff(out, src, plane=1) > 0.0
    assert diff(out, src, plane=2) > 0.0


def test_tiny_clip_edge_replication(make_clip):
    """Frames smaller than 8px must still work via pixel replication."""
    src = make_clip(vs.GRAY8, "tiny")
    out = src.vszip.Compress(codec=0, qscale=8)
    assert (out.width, out.height) == (src.width, src.height)
    out.get_frame(0)  # must not crash on the partial edge block


def test_stride_handling(gray8):
    cropped = gray8.std.Crop(left=27)
    args = dict(codec=0, qscale=8)
    assert_same_clip(cropped.vszip.Compress(**args), repack(cropped).vszip.Compress(**args))


# --- accepted boundary values (must not raise) ------------------------------


@pytest.mark.parametrize("qscale", [1, 31])
def test_qscale_bounds_accepted(gray8, qscale):
    gray8.vszip.Compress(codec=0, qscale=qscale).get_frame(0)


@pytest.mark.parametrize("quality", [1, 100])
def test_quality_bounds_accepted(gray8, quality):
    gray8.vszip.Compress(codec=1, quality=quality).get_frame(0)


@pytest.mark.parametrize("dc_prec", [0, 3])
def test_dc_prec_bounds_accepted(gray8, dc_prec):
    gray8.vszip.Compress(codec=0, qscale=8, dc_prec=dc_prec).get_frame(0)


# --- validation / format rejection ------------------------------------------


@pytest.mark.parametrize("fmt", [vs.RGB24, vs.RGB48, vs.GRAY16, vs.GRAYS, vs.YUV420P10, vs.YUV444PS])
def test_unsupported_format_rejected(make_clip, fmt):
    src = make_clip(fmt)
    with pytest.raises(vs.Error, match="only 8-bit integer Gray or YUV formats are supported"):
        src.vszip.Compress()


@pytest.mark.parametrize(
    ("args", "msg"),
    [
        (dict(codec=2), r"codec must be 0 \(mpeg2\) or 1 \(jpeg\)"),
        (dict(codec=-1), r"codec must be 0 \(mpeg2\) or 1 \(jpeg\)"),
        (dict(codec=0, qscale=0), "qscale must be between 1 and 31"),
        (dict(codec=0, qscale=32), "qscale must be between 1 and 31"),
        (dict(codec=0, dc_prec=-1), "dc_prec must be between 0 and 3"),
        (dict(codec=0, dc_prec=4), "dc_prec must be between 0 and 3"),
        (dict(codec=1, quality=0), "quality must be between 1 and 100"),
        (dict(codec=1, quality=101), "quality must be between 1 and 100"),
    ],
)
def test_validation_errors(gray8, args, msg):
    with pytest.raises(vs.Error, match=msg):
        gray8.vszip.Compress(**args)
