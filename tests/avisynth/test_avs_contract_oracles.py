"""AviSynth must satisfy independent models, even when both plugins agree."""

import pytest
import vapoursynth as vs

from contract_oracles import (GRAY_FORMATS, RGB_CASES, WIDTHS, check_all_excluded,
                              check_limiter, check_rgb_metadata, check_selected_statistics, check_statistics)


class AVSHost:
    def __init__(self, env):
        self.env = env

    def source(self, frames, name, fmt):
        return self.env.source(frames, name)

    def apply(self, name, clip, **kwargs):
        return self.env.invoke("vszip_" + name, clip, **kwargs)

    def frame(self, clip, n=0):
        f = clip.frame(n)
        return f.planes, f.props

    def tag(self, clip, **kwargs):
        for k, v in kwargs.items():
            clip = self.env.invoke("propSet", clip, k, v)
        return clip

    def sequence(self, clip, tags):
        self.env.set_clip("auditSource", clip)
        parts = [f"auditSource.Trim({n},-1)" + "".join(
            f'.propSet("{k}",{v})' for k, v in t.items()) for n, t in enumerate(tags)]
        return self.env.eval(" ++ ".join(parts))


@pytest.fixture
def host(env):
    return AVSHost(env)


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
