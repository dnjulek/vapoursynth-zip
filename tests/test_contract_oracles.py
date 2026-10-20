"""Independent oracles for gaps not protected by cross-host parity/goldens."""

import pytest
import vapoursynth as vs
import numpy as np

from contract_oracles import (GRAY_FORMATS, RGB_CASES, WIDTHS, VSHost, check_all_excluded,
                              check_limiter, check_rgb_metadata, check_selected_statistics, check_statistics)


@pytest.fixture
def host(core):
    return VSHost(core)


@pytest.mark.parametrize("name,fmt,bits", GRAY_FORMATS, ids=[x[0] for x in GRAY_FORMATS])
@pytest.mark.parametrize("width", WIDTHS)
def test_statistics_against_numpy(host, name, fmt, bits, width):
    check_statistics(host, name, fmt, bits, width)


@pytest.mark.parametrize("name,fmt,bits", GRAY_FORMATS, ids=[x[0] for x in GRAY_FORMATS])
def test_all_excluded_mean_and_unexcluded_difference(host, name, fmt, bits):
    check_all_excluded(host, name, fmt, bits)


@pytest.mark.parametrize("bits", [8, 10, 16, 32])
@pytest.mark.parametrize("width", [1, 7, 8, 9, 31, 32, 33, 65])
@pytest.mark.parametrize("default", [False, True], ids=["bounds", "default"])
def test_limiter_storage_bounds_and_selected_planes(host, bits, width, default):
    check_limiter(host, bits, width, default)


@pytest.mark.parametrize("name,fmt,bits,gray,matrix", RGB_CASES,
                         ids=[f"{c[0]}-matrix-{c[-1]}" for c in RGB_CASES])
def test_per_frame_yuv_conversion_against_equations(host, name, fmt, bits, gray, matrix):
    check_rgb_metadata(host, name, fmt, bits, matrix, gray)


def test_hd_unspecified_matrix_uses_height_fallback(host):
    check_rgb_metadata(host, "YUV444P16", vs.YUV444P16, 16, 2, height=651)


@pytest.mark.parametrize("bits", [8, 16, 32])
@pytest.mark.parametrize("family", ["rgb", "yuv"])
def test_statistics_plane_order_and_signed_chroma(host, bits, family):
    check_selected_statistics(host, bits, family)


def test_quantile_model_has_hand_checked_rank_and_zero_threshold_semantics():
    from contract_oracles import quantile_model
    samples = np.array([[0, 5, 10, 15]], np.uint8)
    assert quantile_model(samples, 8, 0.25, 0.5) == (5, 5)
    assert quantile_model(samples, 8, 1, 1) == (255, 0)
    floats = np.array([[0.1, 0.8]], np.float32)
    assert quantile_model(floats, 32, 0, 0) == (float(floats.min()), float(floats.max()))


def test_average_model_deduplicates_excludes_but_keeps_all_diff_samples():
    from contract_oracles import average_model
    a = np.array([[0, 10, 20]], np.uint8)
    b = np.array([[30, 10, 20]], np.uint8)
    avg, delta = average_model(a, b, 8, [0, 0])
    assert avg == 15 / 255
    assert delta == 10 / 255


def test_rgb_model_has_hand_checked_neutral_black_white_and_chroma():
    from contract_oracles import rgb_model
    y = np.array([[16, 235]], np.uint8)
    c = np.full_like(y, 128)
    for plane in rgb_model([y, c, c], 8, 1):
        np.testing.assert_allclose(plane, [[0, 1]], rtol=0, atol=1e-15)
    neutral = np.array([[0.5]], np.float32)
    chroma = np.array([[0.25]], np.float32)
    zero = np.zeros_like(neutral)
    red, _, blue = rgb_model([neutral, zero, chroma], 32, 1)
    assert red[0, 0] == pytest.approx(0.8937)
    assert blue[0, 0] == 0.5


def test_oracle_detects_one_code_error_in_selected_plane_scalar_tail(core):
    class CorruptedResult(VSHost):
        def frame(self, clip, n=0):
            planes, props = super().frame(clip, n)
            planes[1][-1, -1] ^= 1
            return planes, props

    # Simulate a vector-tail defect in observed output, without editing a
    # plugin or an expected value. Checking only plane 0/means would miss it.
    with pytest.raises(AssertionError):
        check_limiter(CorruptedResult(core), 16, 33, False)
