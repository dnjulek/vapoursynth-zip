"""End-to-end XPSNR parity against FFmpeg's vf_xpsnr (the reference impl).

WHY THIS EXISTS
---------------
tests/test_xpsnr.py pins values with *self-referential* goldens (snapshots of
this build, regenerated via --update-goldens), and only ever exercises:
  * <=HD resolutions (<=640x320)  -> the `b_val==1` path only; the entire
    >HD high-pass-with-downsampling path (highds / diff1st / diff2nd) is never run;
  * the source image's default fps -> the 2nd-order temporal path (fps>=32) is
    never run and the fps==32 boundary is never tested;
  * even, "full" geometry;
  * never asserts the free-time "XPSNR average" summary.
So a behavioural change in any of those areas passes silently. This module
closes the gap by comparing, frame for frame, against FFmpeg on byte-identical
pixels (dumped to lossless y4m), across all three block-size regimes, both
supported depths, all subsamplings, and the 1st/2nd-order temporal boundary.

Role mapping (XPSNR's perceptual weight is asymmetric): vszip node1
("reference") == FFmpeg input #0 ("main") == the weight basis; node2
("distorted") == FFmpeg input #1. So `vszip.XPSNR(ref, dist)` corresponds to
`ffmpeg -i ref -i dist -lavfi xpsnr`.

These tests are skipped automatically where ffmpeg (with the xpsnr filter) or
the test image is unavailable. The non-ffmpeg regression tests at the bottom
always run.
"""
import math
import os
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest
import vapoursynth as vs

from helpers import props

# FFmpeg is slow and not always installed; this oracle is OPT-IN. The fast,
# always-on coverage (>HD path, fps boundary, 2nd-order temporal, odd-dim
# no-crash) now lives as goldens + structural asserts in test_xpsnr.py. Run this
# module to re-derive those values from FFmpeg after touching the kernel:
#     VSZIP_FFMPEG_ORACLE=1 pytest tests/test_xpsnr_ffmpeg.py
pytestmark = pytest.mark.skipif(
    os.environ.get("VSZIP_FFMPEG_ORACLE") != "1",
    reason="ffmpeg oracle is opt-in: set VSZIP_FFMPEG_ORACLE=1 to re-verify parity",
)

IMAGE = Path(__file__).resolve().parent / "image.png"

# 6 decimals via the metadata filter (vs 4 via stats_file); internal wsse is
# integer so any real divergence is >=1e-3, while print rounding is <=5e-7.
TOL = 1e-4


# --- environment detection --------------------------------------------------
@lru_cache(maxsize=1)
def _ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "-hide_banner", "-filters"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return None
    return exe if re.search(r"\bxpsnr\b", out) else None


requires_ffmpeg = pytest.mark.skipif(_ffmpeg() is None,
                                     reason="ffmpeg with xpsnr filter not available")
requires_image = pytest.mark.skipif(not IMAGE.is_file(), reason="tests/image.png missing")


# --- deterministic clip construction ---------------------------------------
@lru_cache(maxsize=1)
def _motion_rgb():
    """6-frame 1880x1040 RGB clip with deterministic vertical motion, cropped
    from the 1920x1080 test image (gives both spatial detail and temporal
    activity; enough headroom to upscale to >HD)."""
    img = vs.core.vszip.ImageRead(str(IMAGE))
    win_w, win_h, shift, k = 1880, 1040, 4, 6
    frames = [
        img.std.Crop(left=0, right=img.width - win_w, top=n * shift,
                     bottom=img.height - win_h - n * shift)
        for n in range(k)
    ]
    clip = frames[0]
    for f in frames[1:]:
        clip = clip + f
    return clip


def _build_pair(w: int, h: int, fmt: int, fps: int, frames: int = 5):
    """A (reference, distorted) pair at the requested geometry/format/fps.
    `distorted` perturbs every plane (BoxBlur) so chroma scores stay finite."""
    rgb = _motion_rgb().std.Trim(0, frames - 1).resize.Bilinear(width=w, height=h)
    ref = rgb.resize.Bilinear(format=fmt, matrix_s="709")
    ref = vs.core.std.AssumeFPS(ref, fpsnum=fps, fpsden=1)
    dist = ref.std.BoxBlur(hradius=2, vradius=2)
    return ref, dist


def _dump_y4m(clip: vs.VideoNode, path: Path):
    with open(path, "wb") as f:
        clip.output(f, y4m=True)


# --- ffmpeg oracle ----------------------------------------------------------
_META_RE = re.compile(r"lavfi\.xpsnr\.xpsnr\.([yuv])=(\S+)", re.I)


def _parse_val(v: str) -> float:
    v = v.lower()
    return math.inf if v == "inf" else (math.nan if v == "nan" else float(v))


def _ffmpeg_xpsnr(ref_path: Path, dist_path: Path) -> list[dict]:
    """Per-frame {y,u,v} from FFmpeg. input0 = ref = weight basis."""
    cmd = [_ffmpeg(), "-hide_banner", "-nostdin", "-i", str(ref_path), "-i", str(dist_path),
           "-lavfi", "[0:v][1:v]xpsnr[x];[x]metadata=mode=print:file=-", "-f", "null", "-"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    rows: list[dict] = []
    cur: dict | None = None
    for line in out.splitlines():
        if line.startswith("frame:"):
            cur = {}
            rows.append(cur)
        else:
            m = _META_RE.search(line)
            if m and cur is not None:
                cur[m.group(1).lower()] = _parse_val(m.group(2))
    return rows


def _vszip_xpsnr(ref: vs.VideoNode, dist: vs.VideoNode) -> list[dict]:
    """Per-frame scores, requested strictly in order (temporal state)."""
    out = vs.core.vszip.XPSNR(ref, dist, verbose=False)
    rows = []
    for n in range(out.num_frames):
        p = props(out, n)
        rows.append({"y": p["XPSNR_Y"], "u": p["XPSNR_U"], "v": p["XPSNR_V"]})
    return rows


def _assert_match(ffr: list[dict], zzr: list[dict], label: str):
    assert len(ffr) == len(zzr), f"{label}: frame count ff={len(ffr)} vszip={len(zzr)}"
    assert len(zzr) > 0, f"{label}: no frames"
    for n, (a, b) in enumerate(zip(ffr, zzr)):
        for c in "yuv":
            fv, zv = a.get(c), b[c]
            assert fv is not None, f"{label}: ffmpeg missing {c} at frame {n}"
            if math.isinf(fv) or math.isinf(zv):
                assert math.isinf(fv) and math.isinf(zv), \
                    f"{label}: inf mismatch n{n}.{c} ff={fv} vszip={zv}"
                continue
            assert abs(fv - zv) <= TOL, \
                f"{label}: n{n}.{c} ff={fv} vszip={zv} diff={abs(fv - zv):.2e} > {TOL:.0e}"


# Geometry regimes:
#   smooth -> w*h <= 640*480, exercises the in-line min-smoothing
#   hd     -> <=HD, b_val==1, large blocks, no smoothing
#   uhd    -> > 2048*1152, b_val==2 high-pass-with-downsampling (highds/diff*)
GEOMS = {"smooth": (512, 288), "hd": (1280, 720), "uhd": (2560, 1440)}
FMT_NAMES = {
    vs.YUV420P8: "420p8", vs.YUV420P10: "420p10",
    vs.YUV422P8: "422p8", vs.YUV444P8: "444p8",
    vs.YUV422P10: "422p10", vs.YUV444P10: "444p10",
}

# fps 24 -> 1st-order temporal; 32 -> boundary (FFmpeg uses 2nd-order at >=32).
_CASES = [
    (g, fmt, fps)
    for g in ("smooth", "hd", "uhd")
    for fmt in (vs.YUV420P8, vs.YUV420P10)
    for fps in (24, 32)
]
# subsampling coverage (different chroma-block path) incl. 2nd-order at fps 32
_CASES += [("hd", fmt, 32) for fmt in (vs.YUV422P8, vs.YUV444P8, vs.YUV422P10, vs.YUV444P10)]
# >HD high-pass path with subsampled chroma (even dims -> still bit-exact)
_CASES += [("uhd", fmt, 24) for fmt in (vs.YUV444P8, vs.YUV422P8)]


@requires_ffmpeg
@requires_image
@pytest.mark.usefixtures("core")
@pytest.mark.parametrize(
    "geom,fmt,fps", _CASES,
    ids=[f"{g}-{FMT_NAMES[f]}-fps{r}" for g, f, r in _CASES],
)
def test_ffmpeg_parity(tmp_path, geom, fmt, fps):
    w, h = GEOMS[geom]
    ref, dist = _build_pair(w, h, fmt, fps)
    rp, dp = tmp_path / "ref.y4m", tmp_path / "dist.y4m"
    _dump_y4m(ref, rp)
    _dump_y4m(dist, dp)
    ffr = _ffmpeg_xpsnr(rp, dp)
    zzr = _vszip_xpsnr(ref, dist)
    _assert_match(ffr, zzr, f"{geom}/{FMT_NAMES[fmt]}/fps{fps}")


@requires_ffmpeg
@requires_image
@pytest.mark.usefixtures("core")
def test_ffmpeg_average_and_frame_count(tmp_path):
    """The verbose free-time summary (processed-frame count + square-mean-root
    average) must match FFmpeg's overall average. Run vszip in a subprocess so
    the node frees and prints; it dumps the exact pixels it scored to y4m so
    FFmpeg sees byte-identical input."""
    rp, dp = tmp_path / "ref.y4m", tmp_path / "dist.y4m"
    script = f"""
import sys; sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
import vapoursynth as vs
from conftest import _load_vszip
core = _load_vszip()
import test_xpsnr_ffmpeg as T
ref, dist = T._build_pair(1280, 720, vs.YUV420P8, 50, frames=6)
T._dump_y4m(ref, {str(rp)!r}); T._dump_y4m(dist, {str(dp)!r})
out = core.vszip.XPSNR(ref, dist, verbose=True)
for n in range(out.num_frames):
    out.get_frame(n)
"""
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr
    m = re.search(r"XPSNR average,\s*(\d+)\s*frames\s+y:\s*([0-9.]+)\s+u:\s*([0-9.]+)\s+v:\s*([0-9.]+)",
                  r.stdout)
    assert m, f"no average line in vszip output:\n{r.stdout}\n{r.stderr}"
    n_frames = int(m.group(1))
    zz_avg = {"y": float(m.group(2)), "u": float(m.group(3)), "v": float(m.group(4))}
    assert n_frames == 6, f"processed-frame count {n_frames} != 6"

    cmd = [_ffmpeg(), "-hide_banner", "-nostdin", "-i", str(rp), "-i", str(dp),
           "-lavfi", "xpsnr", "-f", "null", "-"]
    ff_out = subprocess.run(cmd, capture_output=True, text=True).stderr
    fm = re.search(r"XPSNR\s+y:\s*([0-9.]+)\s+u:\s*([0-9.]+)\s+v:\s*([0-9.]+)", ff_out)
    assert fm, f"no ffmpeg average:\n{ff_out}"
    ff_avg = {"y": float(fm.group(1)), "u": float(fm.group(2)), "v": float(fm.group(3))}
    for c in "yuv":
        # both printed at 4 decimals -> agreement within ~1e-4
        assert abs(zz_avg[c] - ff_avg[c]) <= 2e-4, \
            f"average {c}: vszip={zz_avg[c]} ffmpeg={ff_avg[c]}"
