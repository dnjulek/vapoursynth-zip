"""WNNM (Weighted Nuclear Norm Minimization denoiser) tests.

Three independent nets:
  1. golden sweep locking current behavior (tests/goldens/wnnm.json),
  2. per-pixel parity against the validated numpy reference (wnnm_ref.py),
  3. tolerance comparison against the compiled C++ oracle (WolframRhodium's
     VapourSynth-WNNM, the `vapoursynth-wnnm` wheel), skipped if absent.
"""

import numpy as np
import pytest
import vapoursynth as vs

from golden import Case, grid, sweep
from helpers import assert_same_clip, diff, repack
from wnnm_ref import wnnm_ref

# C++ oracle plugin (namespace `wnnm`), autoloaded from site-packages if the
# vapoursynth-wnnm wheel is installed.
HAS_WNNM_CPP = hasattr(vs.core, "wnnm")

# WNNM is float-only (32-bit). Geometry notes: every processed plane must be
# at least block_size on both axes, so `tiny` (GRAYS 13x7, YUV420PS 12x6 with
# 6x3 chroma) needs a small block_size; those combos are hand-picked below
# instead of swept.
CASES = (
    sweep(
        base_fmt=vs.GRAYS,
        base_args=dict(sigma=3),
        formats=[vs.GRAYS, vs.YUV444PS, vs.YUV420PS],
        args=grid(sigma=[1.5, 10])
        + grid(group_size=[4, 16])
        + [
            # block_size/block_step combos; (8, 3) exercises overlapping blocks
            dict(block_size=6, block_step=6),
            dict(block_size=4, block_step=4),
            dict(block_size=8, block_step=3),
        ]
        + grid(bm_range=[2])  # default 7 is the base case
        + [
            dict(residual=True),
            dict(adaptive_aggregation=False),
        ],
        geometries=["odd"],
    )
    + [
        # tiny frames force scalar tails and starved groups (fewer candidates
        # than group_size); block_size must fit the smallest processed plane
        Case(vs.GRAYS, "tiny", dict(sigma=3, block_size=4, block_step=4)),
        Case(vs.YUV444PS, "tiny", dict(sigma=3, block_size=6, block_step=6)),
        Case(vs.YUV420PS, "tiny", dict(sigma=3, block_size=3, block_step=3)),
        # luma-only processing lets a bigger block coexist with 6x3 chroma
        Case(vs.YUV420PS, "tiny", dict(sigma=[3, 0], block_size=6, block_step=6)),
        # per-plane sigma; the middle plane is skipped (copied through)
        Case(vs.YUV444PS, args=dict(sigma=[3, 0, 5])),
        # block matching on a blurred reference clip
        Case(vs.GRAYS, args=dict(sigma=3), variant="rclip"),
        # temporal regime (WNNMRaw + VAggregate chain) on a small 3-frame clip
        Case(vs.GRAYS, args=dict(sigma=3, radius=1), variant="temporal"),
        Case(vs.GRAYS, args=dict(sigma=3, radius=2), variant="temporal"),
    ]
)


@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_cases(golden, make_clip, make_temporal_clip, case):
    if case.variant == "temporal":
        src = make_temporal_clip(case.fmt).std.CropAbs(width=64, height=48, left=200, top=100)
        # frame 1 so both temporal directions contribute real neighbours
        golden.check("wnnm", case, src.vszip.WNNM(**case.args), n=1)
        return
    src = make_clip(case.fmt, case.geometry)
    kwargs = dict(case.args)
    if case.variant == "rclip":
        kwargs["rclip"] = src.std.BoxBlur(hradius=2, vradius=2)
    golden.check("wnnm", case, src.vszip.WNNM(**kwargs))


# --- shared clips -------------------------------------------------------------


@pytest.fixture(scope="module")
def small_gray(make_clip):
    """64x48 GRAYS crop from an interior region (real detail, not a flat corner)."""
    return make_clip(vs.GRAYS).std.CropAbs(width=64, height=48, left=200, top=100)


@pytest.fixture(scope="module")
def oracle_gray(make_clip):
    """64x48 GRAYS crop for the C++ oracle comparisons."""
    return make_clip(vs.GRAYS).std.CropAbs(width=64, height=48, left=400, top=200)


@pytest.fixture(scope="module")
def subpixel_temporal(make_clip):
    """3 frames of the same 64x48 region resampled at fractional per-frame
    offsets: genuine subpixel motion with no exactly-duplicated patches across
    frames. Exact block-error ties are where our stable selection and the C++
    std::partial_sort legitimately diverge, so the oracle temporal test needs a
    tie-free clip (make_temporal_clip shifts by whole rows, duplicating data)."""
    full = make_clip(vs.GRAYS)
    frames = [
        full.resize.Bicubic(64, 48, src_left=200 + 0.4 * n, src_top=100 + 0.6 * n, src_width=64, src_height=48)
        for n in range(3)
    ]
    return frames[0] + frames[1] + frames[2]


def _to_np(clip: vs.VideoNode, plane: int = 0) -> np.ndarray:
    """(T, H, W) f32 array of one plane of every frame."""
    out = []
    for n in range(clip.num_frames):
        with clip.get_frame(n) as f:
            out.append(np.asarray(f[plane]).copy())
    return np.stack(out)


# --- numpy-reference parity ----------------------------------------------------
# wnnm_ref mirrors the C++ algorithm but shares our accumulation order, stable
# tie-breaking and exact aggregation division, so parity is near float32-exact
# (measured headroom ~2.4e-7).


@pytest.mark.parametrize(
    ("kwargs", "use_rclip"),
    [
        (dict(sigma=3), False),
        (dict(sigma=5, block_size=6, block_step=3, group_size=16, residual=True), False),
        (dict(sigma=3, adaptive_aggregation=False), True),
    ],
    ids=["defaults", "bs6-step3-gs16-residual", "rclip-no-adaptive"],
)
def test_numpy_ref_parity_spatial(small_gray, kwargs, use_rclip):
    kw = dict(kwargs)
    ref_kw = dict(kwargs)
    if use_rclip:
        blur = small_gray.std.BoxBlur(hradius=2, vradius=2)
        kw["rclip"] = blur
        ref_kw["rclip"] = _to_np(blur)[0]
    ours = _to_np(small_gray.vszip.WNNM(**kw))[0]
    expected = wnnm_ref(_to_np(small_gray)[0], **ref_kw)
    err = float(np.abs(ours - expected).max())
    assert err <= 5e-7, f"max abs diff {err}"


def test_numpy_ref_parity_temporal(make_temporal_clip):
    """radius=1 over 3 frames, all frames compared (the reference reproduces
    our stable tie-breaking, so even the temporally-clamped edge frames match)."""
    src = make_temporal_clip(vs.GRAYS).std.CropAbs(width=64, height=48, left=200, top=100)
    ours = _to_np(src.vszip.WNNM(sigma=3, radius=1))
    expected = wnnm_ref(_to_np(src), sigma=3, radius=1)
    err = float(np.abs(ours - expected).max())
    assert err <= 5e-7, f"max abs diff {err}"


# --- C++ oracle tolerance -------------------------------------------------------
# The oracle's AVX2 build differs numerically by design: FMA'd block distances
# (can flip near-tie match order) and _mm256_rcp_ps in the radius==0 final
# aggregation (~2^-12 relative, i.e. a ~2.2e-4 floor on mid-gray values).


@pytest.mark.skipif(not HAS_WNNM_CPP, reason="C++ wnnm plugin not installed")
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(sigma=3),
        dict(sigma=1.5),
        dict(sigma=3, residual=True),
        dict(sigma=3, adaptive_aggregation=False),
        dict(sigma=3, block_size=6, block_step=3, group_size=16),
    ],
    ids=lambda kw: ",".join(f"{k}={v}" for k, v in kw.items()),
)
def test_cpp_oracle_spatial(oracle_gray, kwargs):
    ours = _to_np(oracle_gray.vszip.WNNM(**kwargs))[0].astype(np.float64)
    orac = _to_np(oracle_gray.wnnm.WNNM(**kwargs))[0].astype(np.float64)
    d = np.abs(ours - orac)
    assert d.max() <= 1e-3, f"max abs diff {d.max()}"
    p999 = float(np.percentile(d, 99.9))
    assert p999 <= 4e-4, f"p99.9 abs diff {p999}"


@pytest.mark.skipif(not HAS_WNNM_CPP, reason="C++ wnnm plugin not installed")
def test_cpp_oracle_yuv(make_clip):
    src = make_clip(vs.YUV444PS).std.CropAbs(width=64, height=48, left=400, top=200)
    kwargs = dict(sigma=[3, 2, 2])
    ours = src.vszip.WNNM(**kwargs)
    orac = src.wnnm.WNNM(**kwargs)
    for p in range(3):
        d = np.abs(_to_np(ours, p).astype(np.float64) - _to_np(orac, p).astype(np.float64))
        assert d.max() <= 1e-3, f"plane {p}: max abs diff {d.max()}"
        p999 = float(np.percentile(d, 99.9))
        assert p999 <= 4e-4, f"plane {p}: p99.9 abs diff {p999}"


@pytest.mark.skipif(not HAS_WNNM_CPP, reason="C++ wnnm plugin not installed")
def test_cpp_oracle_temporal_interior(subpixel_temporal):
    """radius=1: only the interior frame is compared. On the first/last
    `radius` frames temporal clamping duplicates whole frames, block errors tie
    exactly, and the tie winner (unspecified in C++ std::partial_sort, stable
    here) decides which temporal slab receives the patch — a documented
    divergence of up to ~3e-3. Interior frames have no duplicated data and
    agree to ~7e-7 (limit 1e-5 covers FMA/SVD rounding differences)."""
    ours = _to_np(subpixel_temporal.vszip.WNNM(sigma=3, radius=1))
    orac = _to_np(subpixel_temporal.wnnm.WNNM(sigma=3, radius=1))
    d = float(np.abs(ours[1].astype(np.float64) - orac[1].astype(np.float64)).max())
    assert d <= 1e-5, f"interior frame max abs diff {d}"


# --- behavior -------------------------------------------------------------------


def test_sigma_zero_is_passthrough(small_gray):
    """sigma=0 skips every plane: Create returns the input node itself."""
    assert_same_clip(small_gray.vszip.WNNM(sigma=0), small_gray)


def test_sigma_zero_plane_copied_through(make_clip):
    src = make_clip(vs.YUV444PS)
    out = src.vszip.WNNM(sigma=[3, 0, 5])
    assert diff(out, src, plane=0) > 0.0
    assert diff(out, src, plane=1) == 0.0  # unprocessed plane copied
    assert diff(out, src, plane=2) > 0.0


def test_deterministic(small_gray):
    a = small_gray.vszip.WNNM(sigma=3)
    b = small_gray.vszip.WNNM(sigma=3)
    assert_same_clip(a, b)


def test_stride_handling(small_gray):
    """CropAbs output shares the parent's wide stride; repack() reallocates
    compactly. Results must be bit-identical either way."""
    assert_same_clip(small_gray.vszip.WNNM(sigma=3), repack(small_gray).vszip.WNNM(sigma=3))


# --- validation errors ------------------------------------------------------------


@pytest.mark.parametrize("fmt", [vs.GRAY8, vs.GRAY16, vs.GRAYH, vs.YUV420P16], ids=lambda f: vs.PresetVideoFormat(f).name)
def test_non_f32_format_rejected(core, fmt):
    src = core.std.BlankClip(None, 64, 48, fmt, length=1)
    with pytest.raises(vs.Error, match="only constant format 32 bit float input supported"):
        src.vszip.WNNM()


@pytest.mark.parametrize(
    ("args", "msg"),
    [
        (dict(block_size=0), r'"block_size" must be in \[1, 64\]'),
        (dict(block_size=65), r'"block_size" must be in \[1, 64\]'),
        # 50 fits the 64px width but not the 48px height
        (dict(block_size=50), r'"block_size" must not exceed the dimensions of any processed plane'),
        (dict(block_size=16, group_size=128), r'min\(block_size\^2, group_size\) must be <= 64'),
        (dict(block_size=8, block_step=9), r'"block_step" must be positive and no larger than "block_size"'),
        (dict(block_step=0), r'"block_step" must be positive and no larger than "block_size"'),
        (dict(sigma=-1), r'"sigma" must be non-negative'),
        (dict(group_size=0), r'"group_size" must be in \[1, 256\]'),
        (dict(bm_range=-1), r'"bm_range" must be non-negative'),
        (dict(radius=16), r'"radius" must be in \[0, 15\]'),
        (dict(ps_num=0), r'"ps_num" must be in \[1, 256\]'),
        (dict(ps_range=-1), r'"ps_range" must be non-negative'),
    ],
)
def test_validation_errors(core, args, msg):
    src = core.std.BlankClip(None, 64, 48, vs.GRAYS, length=1)
    with pytest.raises(vs.Error, match=msg):
        src.vszip.WNNM(**args)


def test_block_size_checked_per_processed_plane(core):
    """YUV420PS 64x48 has 32x24 chroma: block_size=32 only fits luma. It must be
    rejected when chroma is processed and accepted when sigma skips chroma."""
    src = core.std.BlankClip(None, 64, 48, vs.YUV420PS, length=1)
    with pytest.raises(vs.Error, match=r'"block_size" must not exceed the dimensions of any processed plane'):
        src.vszip.WNNM(block_size=32)
    src.vszip.WNNM(block_size=32, sigma=[3, 0])  # luma-only: fine


def test_rclip_mismatch_rejected(core):
    a = core.std.BlankClip(None, 64, 48, vs.GRAYS, length=1)
    with pytest.raises(vs.Error, match=r'"rclip" must be of the same format, dimensions and number of frames'):
        a.vszip.WNNM(rclip=core.std.BlankClip(None, 48, 48, vs.GRAYS, length=1))
    with pytest.raises(vs.Error, match=r'"rclip" must be of the same format, dimensions and number of frames'):
        a.vszip.WNNM(rclip=a + a)


def test_max_radius_boundary(make_temporal_clip):
    """radius=15 fills the temporal frame-window arrays exactly (2*15+1 = 31
    frames per raw window and per VAggregate fold) and the 4-frame clip forces
    heavy temporal clamping. Guards the max_window-sized stack arrays in
    getFrameRaw/vagGetFrame against ever drifting from the radius limit."""
    src = make_temporal_clip(vs.GRAYS).std.CropAbs(width=32, height=24, left=200, top=100)[:4]
    out = src.vszip.WNNM(sigma=3, radius=15)
    for n in range(out.num_frames):
        with out.get_frame(n) as f:
            assert np.isfinite(np.asarray(f[0])).all()
