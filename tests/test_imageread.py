import struct
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest
import vapoursynth as vs

from conftest import IMAGE
from helpers import (
    assert_same_clip,
    chrm_chunk,
    cicp_chunk,
    gama_chunk,
    pix,
    props,
    srgb_chunk,
    write_bmp,
    write_png,
)


def read(core: vs.Core, *paths) -> vs.VideoNode:
    return core.vszip.ImageRead([str(p) for p in paths])


# --- basic metadata ----------------------------------------------------------


def test_repo_image(core):
    clip = read(core, IMAGE)
    assert clip.format.id == vs.RGB24
    assert (clip.width, clip.height) == (1920, 1080)
    assert clip.num_frames == 1
    assert clip.fps == vs.core.std.BlankClip(fpsnum=30, fpsden=1).fps
    p = props(clip)
    assert p["zigimg_format"] == "rgb24"
    assert p["zigimg_bits"] == 8
    assert p["zigimg_file_path"] == str(IMAGE)


# --- pixel-exact decoding of generated images --------------------------------


def test_gray8(core, tmp_path):
    rows = [[0, 1, 127], [128, 200, 255]]
    clip = read(core, write_png(tmp_path / "g8.png", rows, color="gray"))
    assert clip.format.id == vs.GRAY8
    assert [[pix(clip, x, y) for x in range(3)] for y in range(2)] == rows
    assert props(clip)["zigimg_format"] == "grayscale8"


def test_gray16(core, tmp_path):
    rows = [[0, 300, 65535], [12345, 54321, 1]]
    clip = read(core, write_png(tmp_path / "g16.png", rows, color="gray", bitdepth=16))
    assert clip.format.id == vs.GRAY16
    assert [[pix(clip, x, y) for x in range(3)] for y in range(2)] == rows
    assert props(clip)["zigimg_bits"] == 16


def test_rgb24(core, tmp_path):
    rows = [[(10, 20, 30), (0, 255, 128)], [(1, 2, 3), (250, 240, 230)]]
    clip = read(core, write_png(tmp_path / "rgb.png", rows))
    assert clip.format.id == vs.RGB24
    for y, row in enumerate(rows):
        for x, (r, g, b) in enumerate(row):
            assert (pix(clip, x, y, 0), pix(clip, x, y, 1), pix(clip, x, y, 2)) == (r, g, b)


def test_rgb48(core, tmp_path):
    rows = [[(0, 30000, 65535), (1, 2, 3)]]
    clip = read(core, write_png(tmp_path / "rgb48.png", rows, bitdepth=16))
    assert clip.format.id == vs.RGB48
    assert (pix(clip, 0, 0, 0), pix(clip, 0, 0, 1), pix(clip, 0, 0, 2)) == (0, 30000, 65535)


def test_palette(core, tmp_path):
    palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    rows = [[0, 1, 2], [2, 1, 0]]
    clip = read(core, write_png(tmp_path / "pal.png", rows, color="palette", palette=palette))
    assert clip.format.id == vs.RGB24
    for y, row in enumerate(rows):
        for x, idx in enumerate(row):
            assert (pix(clip, x, y, 0), pix(clip, x, y, 1), pix(clip, x, y, 2)) == palette[idx]
    # indexed images always carry an alpha clip; opaque palette -> all 255
    alpha = clip.std.PropToClip(prop="_Alpha")
    assert alpha.format.id == vs.GRAY8
    assert pix(alpha, 0, 0) == 255


@pytest.mark.parametrize(
    ("bitdepth", "palette"),
    [
        (1, [(255, 0, 0), (0, 255, 0)]),
        (2, [(10, 20, 30), (40, 50, 60), (70, 80, 90), (100, 110, 120)]),
        (4, [(i * 16, i * 16, i * 16) for i in range(16)]),
    ],
)
def test_palette_sub_byte(core, tmp_path, bitdepth, palette):
    """Sub-byte indexed PNGs (indexed1/2/4) decode through the same
    copyPixelsIndexed(u8) path as indexed8 but exercise their own switch arms
    in both Create and getFrame. They also carry the opaque alpha clip that
    every indexed image gets (Create: pf.isIndexed())."""
    rows = [list(range(len(palette)))]
    clip = read(core, write_png(tmp_path / f"idx{bitdepth}.png", rows, color="palette", bitdepth=bitdepth, palette=palette))
    assert clip.format.id == vs.RGB24
    assert props(clip)["zigimg_format"] == f"indexed{bitdepth}"
    for x, idx in enumerate(rows[0]):
        assert (pix(clip, x, 0, 0), pix(clip, x, 0, 1), pix(clip, x, 0, 2)) == palette[idx]
    alpha = clip.std.PropToClip(prop="_Alpha")
    assert alpha.format.id == vs.GRAY8
    assert all(pix(alpha, x, 0) == 255 for x in range(len(palette)))


def test_rgba_alpha(core, tmp_path):
    rows = [[(10, 20, 30, 0), (40, 50, 60, 128)], [(70, 80, 90, 255), (1, 2, 3, 4)]]
    clip = read(core, write_png(tmp_path / "rgba.png", rows, color="rgba"))
    assert clip.format.id == vs.RGB24
    assert (pix(clip, 1, 0, 0), pix(clip, 1, 0, 1), pix(clip, 1, 0, 2)) == (40, 50, 60)
    alpha = clip.std.PropToClip(prop="_Alpha")
    assert [[pix(alpha, x, y) for x in range(2)] for y in range(2)] == [[0, 128], [255, 4]]
    assert props(alpha)["_ColorRange"] == vs.Range.RANGE_FULL


def test_gray_alpha(core, tmp_path):
    rows = [[(100, 200), (50, 25)]]
    clip = read(core, write_png(tmp_path / "ga.png", rows, color="graya"))
    assert clip.format.id == vs.GRAY8
    assert pix(clip, 0, 0) == 100
    alpha = clip.std.PropToClip(prop="_Alpha")
    assert (pix(alpha, 0, 0), pix(alpha, 1, 0)) == (200, 25)


def test_gray16_alpha(core, tmp_path):
    """16-bit grayscale+alpha (grayscale16Alpha): GRAY16 main + GRAY16 alpha."""
    rows = [[(1000, 60000), (40000, 25)]]
    clip = read(core, write_png(tmp_path / "ga16.png", rows, color="graya", bitdepth=16))
    assert clip.format.id == vs.GRAY16
    assert props(clip)["zigimg_format"] == "grayscale16Alpha"
    assert (pix(clip, 0, 0), pix(clip, 1, 0)) == (1000, 40000)
    alpha = clip.std.PropToClip(prop="_Alpha")
    assert alpha.format.id == vs.GRAY16
    assert (pix(alpha, 0, 0), pix(alpha, 1, 0)) == (60000, 25)
    assert props(alpha)["_ColorRange"] == vs.Range.RANGE_FULL


def test_rgba64_alpha(core, tmp_path):
    """16-bit RGBA (rgba64): RGB48 main + GRAY16 alpha."""
    rows = [[(1000, 30000, 65535, 40000), (1, 2, 3, 4)]]
    clip = read(core, write_png(tmp_path / "rgba64.png", rows, color="rgba", bitdepth=16))
    assert clip.format.id == vs.RGB48
    assert props(clip)["zigimg_format"] == "rgba64"
    assert (pix(clip, 0, 0, 0), pix(clip, 0, 0, 1), pix(clip, 0, 0, 2)) == (1000, 30000, 65535)
    alpha = clip.std.PropToClip(prop="_Alpha")
    assert alpha.format.id == vs.GRAY16
    assert (pix(alpha, 0, 0), pix(alpha, 1, 0)) == (40000, 4)
    assert props(alpha)["_ColorRange"] == vs.Range.RANGE_FULL


def test_gray_alpha_color_range_full(core, tmp_path):
    """8-bit grayscale+alpha carries the FULL range marker on its alpha clip."""
    clip = read(core, write_png(tmp_path / "ga.png", [[(100, 200), (50, 25)]], color="graya"))
    alpha = clip.std.PropToClip(prop="_Alpha")
    assert props(alpha)["_ColorRange"] == vs.Range.RANGE_FULL


@pytest.mark.parametrize(
    ("bitdepth", "values", "expected"),
    [
        (1, [0, 1, 1, 0, 1, 0, 1, 1], [0, 255, 255, 0, 255, 0, 255, 255]),
        (2, [0, 1, 2, 3], [0, 85, 170, 255]),
        (4, [0, 5, 10, 15], [0, 85, 170, 255]),
    ],
)
def test_sub_byte_gray_scaled(core, tmp_path, bitdepth, values, expected):
    """grayscale1/2/4 samples are expanded to the full 8-bit range."""
    clip = read(core, write_png(tmp_path / f"g{bitdepth}.png", [values], color="gray", bitdepth=bitdepth))
    assert clip.format.id == vs.GRAY8
    assert [pix(clip, x, 0) for x in range(len(values))] == expected


def test_bmp(core, tmp_path):
    rows = [
        [(10, 20, 30), (200, 100, 50), (1, 2, 3), (4, 5, 6)],
        [(0, 0, 0), (255, 255, 255), (9, 8, 7), (60, 70, 80)],
    ]
    clip = read(core, write_bmp(tmp_path / "img.bmp", rows))
    assert clip.format.id == vs.RGB24
    for y, row in enumerate(rows):
        for x, (r, g, b) in enumerate(row):
            assert (pix(clip, x, y, 0), pix(clip, x, y, 1), pix(clip, x, y, 2)) == (r, g, b)
    assert "_Transfer" not in props(clip)  # color props are PNG-only


# --- PNG color chunks -> frame props -----------------------------------------

RGB_ROWS = [[(255, 0, 0), (0, 255, 0)]]


def color_props(core, tmp_path, name, extra_chunks):
    clip = read(core, write_png(tmp_path / name, RGB_ROWS, extra_chunks=extra_chunks))
    p = props(clip)
    return int(p["_Matrix"]), int(p["_Transfer"]), int(p["_Primaries"])


def test_plain_png_defaults_to_srgb(core, tmp_path):
    matrix, transfer, primaries = color_props(core, tmp_path, "plain.png", ())
    assert (matrix, transfer, primaries) == (0, 13, 1)  # RGB, IEC 61966-2-1, BT.709


def test_srgb_chunk(core, tmp_path):
    assert color_props(core, tmp_path, "srgb.png", (srgb_chunk(),)) == (0, 13, 1)


def test_gama_linear(core, tmp_path):
    _, transfer, _ = color_props(core, tmp_path, "gama.png", (gama_chunk(100000),))
    assert transfer == 8  # LINEAR


def test_gama_470m(core, tmp_path):
    _, transfer, _ = color_props(core, tmp_path, "gama2.png", (gama_chunk(45455),))
    assert transfer == 4  # BT470_M


def test_chrm_bt2020(core, tmp_path):
    chrm = chrm_chunk(31270, 32900, 70800, 29200, 17000, 79700, 13100, 4600)
    _, _, primaries = color_props(core, tmp_path, "chrm.png", (gama_chunk(45455), chrm))
    assert primaries == 9  # BT2020


def test_gama_470bg(core, tmp_path):
    _, transfer, _ = color_props(core, tmp_path, "gama470bg.png", (gama_chunk(35714),))
    assert transfer == 5  # BT470_BG


def test_gama_unrecognized_is_unspecified(core, tmp_path):
    """A gamma matching none of the known values -> UNSPECIFIED transfer."""
    _, transfer, _ = color_props(core, tmp_path, "gama_un.png", (gama_chunk(22222),))
    assert transfer == 2  # UNSPECIFIED


def test_chrm_unmatched_is_unspecified(core, tmp_path):
    """Primaries that match no known cHRM candidate -> UNSPECIFIED."""
    chrm = chrm_chunk(11111, 22222, 33333, 44444, 55555, 11000, 22000, 33000)
    _, _, primaries = color_props(core, tmp_path, "chrm_un.png", (chrm,))
    assert primaries == 2  # UNSPECIFIED


def test_cicp_invalid_values_keep_defaults(core, tmp_path):
    """cICP code points outside the VS enums are ignored (defaults retained)."""
    chunks = (cicp_chunk(primaries=200, transfer=200),)
    _, transfer, primaries = color_props(core, tmp_path, "cicp_bad.png", chunks)
    assert (transfer, primaries) == (13, 1)  # IEC 61966-2-1, BT.709 defaults


def test_cicp_overrides(core, tmp_path):
    chunks = (cicp_chunk(primaries=9, transfer=16), srgb_chunk())
    _, transfer, primaries = color_props(core, tmp_path, "cicp.png", chunks)
    assert (transfer, primaries) == (16, 9)  # ST2084, BT2020


def test_gray_png_matrix_bt709(core, tmp_path):
    clip = read(core, write_png(tmp_path / "g.png", [[0, 255]], color="gray"))
    assert int(props(clip)["_Matrix"]) == 1  # gray gets BT709 instead of RGB


def test_gray_png_carries_transfer_and_primaries(core, tmp_path):
    """Gray PNGs also receive the sRGB transfer/primaries defaults, not just
    the BT709 matrix (the png_color branch runs for gray clips too)."""
    p = props(read(core, write_png(tmp_path / "g.png", [[0, 255]], color="gray")))
    assert (int(p["_Transfer"]), int(p["_Primaries"])) == (13, 1)  # IEC 61966-2-1, BT.709


def test_gray_png_cicp_overrides(core, tmp_path):
    """cICP on a gray PNG overrides transfer/primaries while keeping BT709 matrix."""
    chunks = (cicp_chunk(primaries=9, transfer=16),)
    p = props(read(core, write_png(tmp_path / "gc.png", [[0, 255]], color="gray", extra_chunks=chunks)))
    assert (int(p["_Matrix"]), int(p["_Transfer"]), int(p["_Primaries"])) == (1, 16, 9)


# --- multi-image clips and validation ----------------------------------------


def test_multiple_paths(core, tmp_path):
    a = write_png(tmp_path / "a.png", [[(1, 2, 3)]])
    b = write_png(tmp_path / "b.png", [[(4, 5, 6)]])
    clip = read(core, a, b)
    assert clip.num_frames == 2
    assert props(clip, 0)["zigimg_file_path"] == str(a)
    assert props(clip, 1)["zigimg_file_path"] == str(b)
    assert pix(clip, 0, 0, 0, n=0) == 1
    assert pix(clip, 0, 0, 0, n=1) == 4


def test_validate_dimension_mismatch(core, tmp_path):
    a = write_png(tmp_path / "a.png", [[(1, 2, 3)]])
    b = write_png(tmp_path / "b.png", [[(1, 2, 3), (4, 5, 6)]])
    with pytest.raises(vs.Error, match="Dimensions do not match"):
        core.vszip.ImageRead([str(a), str(b)], validate=True)


def test_validate_pixel_format_mismatch(core, tmp_path):
    a = write_png(tmp_path / "a.png", [[(1, 2, 3)]])
    b = write_png(tmp_path / "b.png", [[7]], color="gray")
    with pytest.raises(vs.Error, match="Pixel formats do not match"):
        core.vszip.ImageRead([str(a), str(b)], validate=True)


def test_missing_file_error(core, tmp_path):
    with pytest.raises(vs.Error, match="Couldn't open"):
        read(core, tmp_path / "nope.png")


def test_undecodable_file_error(core, tmp_path):
    """A file that exists but cannot be decoded fails at Create time (the
    decode-error branch, distinct from the file-not-found branch)."""
    bad = tmp_path / "garbage.png"
    bad.write_bytes(b"not an image at all, just text bytes padding padding padding")
    with pytest.raises(vs.Error, match="Couldn't open"):
        read(core, bad)


def test_per_frame_decode_error(core, tmp_path):
    """Without validate, a broken later image is only decoded when its frame is
    requested, so Create succeeds and the error surfaces from getFrame."""
    good = write_png(tmp_path / "good.png", [[(1, 2, 3)]])
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\n garbage trailing bytes that fail to decode")
    clip = read(core, good, bad)
    assert pix(clip, 0, 0, 0, n=0) == 1  # frame 0 decodes fine
    with pytest.raises(vs.Error, match="Couldn't open"):
        clip.get_frame(1)


def test_validate_happy_path(core, tmp_path):
    """validate=True with matching dimensions and pixel formats loads cleanly."""
    a = write_png(tmp_path / "a.png", [[(1, 2, 3), (4, 5, 6)]])
    b = write_png(tmp_path / "b.png", [[(7, 8, 9), (10, 11, 12)]])
    clip = core.vszip.ImageRead([str(a), str(b)], validate=True)
    assert clip.num_frames == 2
    assert (pix(clip, 0, 0, 0, n=0), pix(clip, 0, 0, 0, n=1)) == (1, 7)


def test_validate_missing_later_file(core, tmp_path):
    """validate=True eagerly opens every path, so a missing later file errors
    at Create (via validatePaths) rather than at frame time."""
    a = write_png(tmp_path / "a.png", [[(1, 2, 3)]])
    with pytest.raises(vs.Error, match="Couldn't open"):
        core.vszip.ImageRead([str(a), str(tmp_path / "nope.png")], validate=True)


def test_validate_single_path_is_noop(core, tmp_path):
    """validate=True with a single path skips validation (len <= 1 guard)."""
    a = write_png(tmp_path / "a.png", [[(1, 2, 3)]])
    clip = core.vszip.ImageRead([str(a)], validate=True)
    assert clip.num_frames == 1
    assert pix(clip, 0, 0, 0) == 1


# --- URL loading (local HTTP server, no external network) ---------------------


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def http_server(tmp_path_factory):
    directory = tmp_path_factory.mktemp("http")
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(directory)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield directory, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_url(core, http_server):
    directory, base = http_server
    rows = [[(11, 22, 33), (44, 55, 66)], [(77, 88, 99), (3, 2, 1)]]
    path = write_png(directory / "net.png", rows)
    from_url = core.vszip.ImageRead(f"{base}/net.png")
    assert_same_clip(from_url, read(core, path))
    assert props(from_url)["zigimg_file_path"] == f"{base}/net.png"


def test_url_404(core, http_server):
    _, base = http_server
    with pytest.raises(vs.Error, match="Couldn't open"):
        core.vszip.ImageRead(f"{base}/missing.png")
