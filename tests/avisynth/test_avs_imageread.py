"""ImageRead vs vszip: pixels, alpha placement, props and errors."""

import pathlib
import shutil

import numpy as np
import pytest

import harness as H
from avsrt import AvsError
from imgutil import chrm_chunk, cicp_chunk, gama_chunk, srgb_chunk, write_bmp, write_png

REPO_IMAGE = H.REPO_ROOT / "tests" / "image.png"


def avs_read(env, *paths, extra=""):
    lst = ", ".join(f'"{p}"' for p in paths)
    return env.eval(f"avszip_ImageRead(path=[{lst}]{extra})")


def vs_read(*paths):
    return H.vszip().ImageRead([str(p) for p in paths])


def compare(env, *paths, alpha=None):
    a = avs_read(env, *[str(p).replace("\\", "/") for p in paths])
    v = vs_read(*paths)
    assert a.num_frames == v.num_frames and (a.width, a.height) == (v.width, v.height)
    for n in range(a.num_frames):
        fa = a.frame(n)
        fv = v.get_frame(n)
        vplanes = [np.array(fv[p], copy=True) for p in range(fv.format.num_planes)]
        for p, vp in enumerate(vplanes):
            assert np.array_equal(fa.planes[p], vp), f"frame {n} plane {p}"
        for k in ("zigimg_bits", "_Matrix", "_Transfer", "_Primaries"):
            if k in fv.props:
                assert fa.props[k] == fv.props[k], k
        for k in ("zigimg_format", "zigimg_file_path"):
            assert fa.props[k].decode().replace("\\", "/") == str(fv.props[k]).replace("\\", "/"), k
        if "_Alpha" in fv.props:
            va = np.array(fv.props["_Alpha"][0], copy=True)
            if alpha == "plane":
                assert np.array_equal(fa.planes[3], va), f"frame {n} alpha plane"
            else:
                assert fa.props["_Alpha"] == "<v>", "gray+alpha must carry an _Alpha frame prop"
    return a


def test_repo_image(env):
    a = compare(env, REPO_IMAGE)
    assert a.family == "rgb"


def test_gray8_gray16_rgb24_rgb48(env, tmp_path):
    compare(env, write_png(tmp_path / "g8.png", [[0, 128, 255], [7, 9, 200]], color="gray"))
    compare(env, write_png(tmp_path / "g16.png", [[0, 1000, 65535]], color="gray", bitdepth=16))
    compare(env, write_png(tmp_path / "rgb.png", [[(255, 0, 0), (0, 255, 0)], [(1, 2, 3), (9, 8, 7)]]))
    compare(env, write_png(tmp_path / "rgb48.png", [[(65535, 0, 1), (300, 40000, 7)]], bitdepth=16))


def test_rgba_alpha_is_native_plane(env, tmp_path):
    a = compare(env, write_png(tmp_path / "rgba.png", [[(255, 0, 0, 10), (0, 255, 0, 250)]], color="rgba"), alpha="plane")
    assert a.has_alpha


def test_gray_alpha_is_frame_prop(env, tmp_path):
    a = compare(env, write_png(tmp_path / "ga.png", [[(10, 20), (200, 255)]], color="graya"), alpha="prop")
    assert a.family == "gray"


@pytest.mark.parametrize(("bitdepth", "palette"), [
    (8, [(255, 0, 0), (0, 255, 0), (0, 0, 255)]),
    (1, [(255, 0, 0), (0, 255, 0)]),
    (2, [(10, 20, 30), (40, 50, 60), (70, 80, 90), (100, 110, 120)]),
    (4, [(i * 16, i * 16, i * 16) for i in range(16)]),
])
def test_palette(env, tmp_path, bitdepth, palette):
    rows = [[i % len(palette) for i in range(5)], [(i * 3) % len(palette) for i in range(5)]]
    compare(env, write_png(tmp_path / "pal.png", rows, color="palette", palette=palette, bitdepth=bitdepth), alpha="plane")


@pytest.mark.parametrize("bitdepth", [1, 2, 4])
def test_sub_byte_gray(env, tmp_path, bitdepth):
    peak = (1 << bitdepth) - 1
    compare(env, write_png(tmp_path / f"g{bitdepth}.png", [[0, peak, peak // 2 or 1, 0]], color="gray", bitdepth=bitdepth))


def test_bmp(env, tmp_path):
    compare(env, write_bmp(tmp_path / "a.bmp", [[(10, 20, 30), (40, 50, 60)], [(1, 2, 3), (4, 5, 6)]]))


@pytest.mark.parametrize("chunks", [(), (srgb_chunk(),), (gama_chunk(100000),), (gama_chunk(45455),),
                                    (chrm_chunk(31270, 32900, 70800, 29200, 17000, 79700, 13100, 4600),),
                                    (cicp_chunk(9, 16),), (cicp_chunk(200, 200),)])
def test_png_color_props(env, tmp_path, chunks):
    compare(env, write_png(tmp_path / "c.png", [[(255, 0, 0), (0, 255, 0)]], extra_chunks=chunks))


def test_sequence_and_validate(env, tmp_path):
    a = write_png(tmp_path / "a.png", [[1, 2], [3, 4]], color="gray")
    b = write_png(tmp_path / "b.png", [[5, 6], [7, 8]], color="gray")
    c = write_png(tmp_path / "c.png", [[5, 6, 7]], color="gray")
    clip = compare(env, a, b)
    assert clip.num_frames == 2 and clip.fps == (30, 1)
    with pytest.raises(AvsError, match="Dimensions do not match"):
        avs_read(env, str(a).replace("\\", "/"), str(c).replace("\\", "/"), extra=", validate=true")
    # Without validate a mismatching later frame fails at request time.
    lazy = avs_read(env, str(a).replace("\\", "/"), str(c).replace("\\", "/"))
    with pytest.raises(AvsError, match="the clip is 2x2"):
        lazy.frame(1)


def test_errors(env, tmp_path):
    with pytest.raises(AvsError, match="Couldn't open"):
        avs_read(env, str(tmp_path / "missing.png").replace("\\", "/"))
    with pytest.raises(AvsError, match="path is required"):
        env.eval("avszip_ImageRead()")


def test_source_filter_parity_and_audio_safe(env, tmp_path):
    p = write_png(tmp_path / "p.png", [[1, 2], [3, 4]], color="gray")
    env.set_clip("img", avs_read(env, str(p).replace("\\", "/")))
    # GetParity/HasAudio on a childless C filter must not crash.
    env.eval('Assert(!GetParity(img, 0), "parity")\nAssert(!HasAudio(img), "audio")', want_clip=False)
