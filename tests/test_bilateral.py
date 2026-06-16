import subprocess
import sys
from pathlib import Path

import pytest
import vapoursynth as vs

from golden import Case, grid, sweep
from helpers import assert_same_clip, avg, diff, repack

TESTS_DIR = Path(__file__).resolve().parent

CASES = (
    sweep(
        base_fmt=vs.GRAY16,
        base_args=dict(sigmaS=2, sigmaR=2),
        formats=[vs.GRAY8, vs.GRAY16, vs.GRAYH, vs.GRAYS, vs.YUV420P8, vs.YUV420P16, vs.YUV444P16, vs.RGB24, vs.RGBS],
        args=grid(sigmaS=[0.8, 2, 5], sigmaR=[0.02, 2])
        + [
            dict(sigmaS=3, sigmaR=0.02, algorithm=2),
            # PBFICnum only affects the algorithm=1 (PBFIC) path
            dict(sigmaS=3, sigmaR=0.1, algorithm=1, PBFICnum=4),
            dict(sigmaS=3, sigmaR=0.1, algorithm=1, PBFICnum=32),
        ],
        geometries=["odd", "tiny"],
    )
    + [
        # per-plane argument arrays (luma/chroma split)
        Case(vs.YUV420P16, args=dict(sigmaS=[3, 1.5], sigmaR=[0.02, 0.05])),
        Case(vs.YUV420P16, args=dict(sigmaS=2, sigmaR=2, planes=[0])),
        # ref-clip (joint bilateral) variants
        Case(vs.GRAY16, args=dict(sigmaS=2, sigmaR=0.05), variant="ref"),
        Case(vs.YUV420P8, args=dict(sigmaS=2, sigmaR=0.05), variant="ref"),
    ]
)


@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_cases(golden, make_clip, case):
    src = make_clip(case.fmt, case.geometry)
    kwargs = dict(case.args)
    if case.variant == "ref":
        kwargs["ref"] = src.std.BoxBlur(hradius=5, vradius=5)
    golden.check("bilateral", case, src.vszip.Bilateral(**kwargs))


GOLDENS = [
    (vs.GRAYS, False, 0.4959264570310188),
    (vs.GRAYS, True, 0.4959947573716272),
    (vs.GRAY16, False, 0.4867585293312972),
    (vs.GRAY16, True, 0.4867979883572471),
    (vs.GRAY8, False, 0.48851139322916665),
]


@pytest.mark.parametrize(("fmt", "use_ref", "expected"), GOLDENS)
def test_golden(to_gray, fmt, use_ref, expected):
    src = to_gray(fmt)
    if use_ref:
        out = src.vszip.Bilateral(ref=src.std.BoxBlur(hradius=5, vradius=5))
    else:
        out = src.vszip.Bilateral(sigmaS=2, sigmaR=2)
    assert avg(out) == pytest.approx(expected, rel=1e-6)


def test_golden_algorithm2(to_gray):
    out = to_gray(vs.GRAY16).vszip.Bilateral(sigmaS=3, sigmaR=0.02, algorithm=2)
    assert avg(out) == pytest.approx(0.4867884865613317, rel=1e-6)


def test_planes(to_yuv):
    # Note: unlike the wiki claim, the implementation defaults to processing
    # ALL planes (Data.planes = {true, true, true}).
    src = to_yuv(vs.YUV420P16)
    out = src.vszip.Bilateral(sigmaS=2, sigmaR=2)
    for p in range(3):
        assert diff(out, src, plane=p) > 0.0
    luma_only = src.vszip.Bilateral(sigmaS=2, sigmaR=2, planes=[0])
    assert diff(luma_only, src, plane=0) > 0.0
    assert diff(luma_only, src, plane=1) == 0.0  # unprocessed planes copied
    assert diff(luma_only, src, plane=2) == 0.0


def test_sigma_zero_is_passthrough(to_gray):
    src = to_gray(vs.GRAY16)
    assert_same_clip(src.vszip.Bilateral(sigmaS=0), src)
    assert_same_clip(src.vszip.Bilateral(sigmaR=0), src)


def test_f16_runs(to_gray):
    out = to_gray(vs.GRAYH).vszip.Bilateral(sigmaS=2, sigmaR=2)
    assert out.format.id == vs.GRAYH
    assert 0.0 < avg(out.resize.Point(format=vs.GRAYS)) < 1.0


def test_stride_handling(to_gray):
    cropped = to_gray(vs.GRAY16).std.Crop(left=27)
    args = dict(sigmaS=2, sigmaR=2)
    assert_same_clip(cropped.vszip.Bilateral(**args), repack(cropped).vszip.Bilateral(**args))


@pytest.mark.parametrize(
    ("args", "msg"),
    [
        (dict(sigmaS=-1), 'Invalid "sigmaS" assigned'),
        (dict(PBFICnum=1), 'Invalid "PBFICnum" assigned'),
        (dict(PBFICnum=300), "PBFICnum value 300 is above maximum 256"),
        (dict(algorithm=3), "algorithm value 3 is above maximum 2"),
        (dict(sigmaR=-0.5), "sigmaR value -0.5 is below minimum 0"),
    ],
)
def test_validation_errors(to_gray, args, msg):
    with pytest.raises(vs.Error, match=msg):
        to_gray(vs.GRAY16).vszip.Bilateral(**args)


def test_algorithm1_does_not_crash():
    """Run in a subprocess because the failure mode is process death
    (STATUS_HEAP_CORRUPTION), which would take pytest down with it."""
    script = (
        f"import sys; sys.path.insert(0, {str(TESTS_DIR)!r})\n"
        "import vapoursynth as vs\n"
        "from conftest import _load_vszip\n"
        "core = _load_vszip()\n"
        "src = core.std.BlankClip(None, 64, 64, vs.GRAY16, length=1)\n"
        "src.vszip.Bilateral(sigmaS=3, sigmaR=0.1, algorithm=1).get_frame(0)\n"
        "print('OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "OK" in r.stdout


@pytest.mark.parametrize(("w", "h"), [(20, 4), (5, 20), (4, 4), (3, 30)])
@pytest.mark.parametrize("fmt", [vs.GRAY8, vs.GRAY16, vs.GRAYS])
def test_small_frame_errors(core, fmt, w, h):
    """A plane smaller than 2*radius on either axis has no algorithm-2 interior
    (radius=5 at the default sigmaS=3), and the kernel's (dim - radius) bounds
    used to underflow -> Debug panic / Release OOB read. bilateralCreate now
    rejects such clips up front rather than branching per-frame in the kernel."""
    color = 0.5 if fmt == vs.GRAYS else 100
    src = core.std.BlankClip(None, w, h, fmt, length=1, color=color)
    with pytest.raises(vs.Error, match="plane too small for the spatial radius"):
        src.vszip.Bilateral()


def test_small_frame_subsampled_chroma_errors(core):
    """The check is per processed plane. With algorithm=2 forced, luma 64x64 at
    sigmaS=2 (radius 3) has a fine interior, but the 32x32 chroma plane with a
    large chroma sigmaS=20 gives a radius far exceeding it -> the per-plane check
    must still reject (the failure is on chroma, not luma)."""
    src = core.std.BlankClip(None, 64, 64, vs.YUV420P8, length=1, color=[100, 128, 128])
    with pytest.raises(vs.Error, match="plane too small for the spatial radius"):
        src.vszip.Bilateral(sigmaS=[2, 20], algorithm=2)


@pytest.mark.parametrize(("w", "h"), [(5, 5), (4, 30), (8, 8)])
def test_small_frame_algorithm1_ok(core, w, h):
    """algorithm 1 (recursive Gaussian) is size-agnostic and exempt from the
    radius check: small frames must still produce output, not error."""
    out = core.std.BlankClip(None, w, h, vs.GRAY16, length=1, color=100).vszip.Bilateral(
        sigmaS=3, sigmaR=0.1, algorithm=1
    )
    assert (out.width, out.height) == (w, h)
    out.get_frame(0)
