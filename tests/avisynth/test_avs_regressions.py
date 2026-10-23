"""Regressions from the review of the AviSynth port: combinations the format
and parameter matrices did not cover (foreign pitch + temporal windows, alpha
layouts, later ImageRead frames, RGB plane routing, extreme accepted ranges,
alpha pass-through).

Cases that used to abort the host run in a child process: a crash fails the
test instead of killing the suite, and the expected outcome (a frame, or a
specific script error) is asserted.
"""

import pathlib
import subprocess
import sys

import numpy as np
import pytest

import harness as H

HERE = pathlib.Path(__file__).resolve().parent

_CHILD = r"""
import sys
sys.path.insert(0, sys.argv[1])
import avsrt, harness
env = avsrt.Env(plugins=[harness.AVSZIP])
try:
    clip = env.eval(sys.argv[3])
    clip.frame(int(sys.argv[2]))
    print("OK")
except avsrt.AvsError as e:
    print("AVSERROR:", e)
"""


def isolated(script, n=0, timeout=120):
    """Evaluate `script` and request frame `n` in a fresh host process.
    Returns "OK" or "AVSERROR: <message>"; a dead process fails the test."""
    r = subprocess.run([sys.executable, "-c", _CHILD, str(HERE), str(n), script],
                       capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, f"host process died (exit {r.returncode}):\n{r.stderr[-600:]}"
    return r.stdout.strip().splitlines()[-1]


def foreign_pitch(clip):
    """Every plane keeps a pitch larger than its row (an aligned crop that
    retains the parent's pitch), so canonical copies must run."""
    return all(pitch != row for _, pitch, row, _ in clip.geometry(0))


# ---------------------------------------------------------------------------
# Scratch capacity: temporal windows x planes x foreign pitch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("script,n", [
    ('BlankClip(width=224, height=64, length=5, pixel_type="YV24").Crop(64, 0, 96, 64).avszip_Checkmate(tthr2=8)', 2),
    ('BlankClip(width=128, height=32, length=5, pixel_type="RGBPS").Crop(16, 0, 48, 32).avszip_WNNM(radius=1, block_size=4)', 2),
    ('BlankClip(width=128, height=32, length=5, pixel_type="Y32").Crop(16, 0, 48, 32).avszip_WNNM(radius=2, block_size=4)', 2),
    # maximum radius: a 31-frame window on three planes
    ('BlankClip(width=128, height=32, length=40, pixel_type="RGBPS").Crop(16, 0, 48, 32).avszip_WNNM(radius=15, block_size=4)', 20),
])
def test_temporal_foreign_pitch_does_not_abort(script, n):
    assert isolated(script, n) == "OK"


def test_checkmate_foreign_pitch_matches_vszip(env):
    pair = H.Pair(env, "YV24", H.make_frames("YV24", w=224, h=64, n=5))
    env.set_clip("src", pair.avs)
    a = env.eval("src.Crop(64, 0, 96, 64)")
    assert foreign_pitch(a)
    env.set_clip("cropped", a)
    out = env.eval("cropped.avszip_Checkmate(tthr2=8)")
    v = H.vszip().Checkmate(H.core.std.Crop(pair.vs, left=64, right=64), tthr2=8)
    for n, (x, y) in enumerate(zip(out.frames(), H.vs_frames(v))):
        H.compare_planes(x, y, what=f"frame {n}")


@pytest.mark.parametrize("fmt,radius", [("RGBPS", 1), ("Y32", 2)])
def test_wnnm_temporal_foreign_pitch_matches_vszip(env, fmt, radius):
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=128, h=32, n=5))
    env.set_clip("src", pair.avs)
    a = env.eval("src.Crop(16, 0, 48, 32)")
    assert foreign_pitch(a)
    env.set_clip("cropped", a)
    out = env.eval(f"cropped.avszip_WNNM(radius={radius}, block_size=4)")
    v = H.vszip().WNNM(H.core.std.Crop(pair.vs, left=16, right=64), radius=radius, block_size=4)
    for n, (x, y) in enumerate(zip(out.frames(), H.vs_frames(v))):
        H.compare_planes(x, y, what=f"frame {n}")


# ---------------------------------------------------------------------------
# Alpha layout of secondary clips
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fa,fb", [("RGBAP", "RGBP"), ("RGBP", "RGBAP"), ("YUVA444", "YV24"), ("YV24", "YUVA444")])
def test_adaptive_binarize_mixed_alpha_layouts(fa, fb):
    script = (f'a = BlankClip(width=32, height=16, length=1, pixel_type="{fa}")\n'
              f'b = BlankClip(width=32, height=16, length=1, pixel_type="{fb}")\n'
              'a.avszip_AdaptiveBinarize(clip2=b)')
    assert isolated(script) == "OK"  # color planes only; the primary's alpha passes through


@pytest.mark.parametrize("name,args", [("PlaneAverage", "exclude=[-1], "), ("PlaneMinMax", "")])
def test_stats_alpha_plane_requires_alpha_in_clipb(name, args):
    base = ('a = BlankClip(width=32, height=16, length=1, pixel_type="RGBAP")\n'
            'b = BlankClip(width=32, height=16, length=1, pixel_type="RGBP")\n')
    out = isolated(base + f"a.avszip_{name}({args}clipb=b, planes=[3])")
    assert out.startswith("AVSERROR:") and "alpha plane" in out
    assert isolated(base + f"a.avszip_{name}({args}clipb=b, planes=[0])") == "OK"


# ---------------------------------------------------------------------------
# ImageRead: later frames must decode to the first image's layout
# ---------------------------------------------------------------------------

def _png(path, mode, value):
    from PIL import Image
    Image.new(mode, (2, 2), value).save(path)
    return path.as_posix()


@pytest.mark.parametrize("first,second", [
    (("L", 100), ("RGB", (1, 2, 3))),        # family
    (("RGB", (1, 2, 3)), ("RGBA", (1, 2, 3, 4))),  # alpha
    (("L", 100), ("I;16", 1000)),             # depth
])
def test_imageread_later_format_change(tmp_path, first, second):
    a = _png(tmp_path / "a.png", *first)
    b = _png(tmp_path / "b.png", *second)
    out = isolated(f'avszip_ImageRead(path=["{a}", "{b}"], validate=false)', n=1)
    assert out.startswith("AVSERROR:") and "does not match the clip's first image" in out
    assert isolated(f'avszip_ImageRead(path=["{a}", "{b}"], validate=false)', n=0) == "OK"


def test_imageread_later_format_change_vszip(tmp_path):
    a = _png(tmp_path / "a.png", "L", 100)
    b = _png(tmp_path / "b.png", "RGB", (1, 2, 3))
    clip = H.vszip().ImageRead([a, b])
    clip.get_frame(0)
    with pytest.raises(Exception, match="does not match the clip's first image"):
        clip.get_frame(1)


# ---------------------------------------------------------------------------
# AdaptiveGrainMask reads logical plane 0 (R for planar RGB)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", ["RGBP", "RGBP16", "RGBPS"])
def test_adaptive_grain_mask_rgb_plane_routing(env, fmt):
    frames = H.make_frames(fmt, w=32, h=16, n=1)
    # very different channels, so reading G or B instead of R shows up
    peak = 1.0 if fmt == "RGBPS" else (255 if fmt == "RGBP" else 65535)
    for p, level in enumerate((0.08, 0.7, 0.2)):
        frames[0][p][:] = np.asarray(level * peak).astype(frames[0][p].dtype)
    pair = H.Pair(env, fmt, frames)
    a, v = H.run_both(env, pair, "AdaptiveGrainMask")
    H.compare_planes(a.frame(0), H.vs_frames(v)[0], what=fmt)


# ---------------------------------------------------------------------------
# Extreme accepted ranges must not overflow the search arithmetic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("script,n", [
    ('BlankClip(width=16, height=16, length=3, pixel_type="Y32").avszip_WNNM(radius=1, block_size=4, ps_range=2147483648)', 1),
    ('BlankClip(width=16, height=16, length=3, pixel_type="Y32").avszip_WNNM(radius=1, block_size=4, ps_range=4294967295)', 1),
    ('BlankClip(width=8, height=8, length=1, pixel_type="Y32").avszip_WNNM(block_size=4, bm_range=4294967295)', 0),
    ('BlankClip(width=8, height=8, length=1, pixel_type="Y32").avszip_WNNM(block_size=4, bm_range=2147483647)', 0),
])
def test_wnnm_extreme_ranges(script, n):
    assert isolated(script, n) == "OK"


def test_wnnm_extreme_ranges_match_plane_sized_ranges(env):
    # Ranges past the plane select the same windows as a plane-sized range.
    pair = H.Pair(env, "Y32", H.make_frames("Y32", w=24, h=16, n=3))
    env.set_clip("src", pair.avs)
    big = env.eval("src.avszip_WNNM(radius=1, block_size=4, bm_range=4294967295, ps_range=4294967295)")
    small = env.eval("src.avszip_WNNM(radius=1, block_size=4, bm_range=24, ps_range=24)")
    for n in range(3):
        H.compare_planes(big.frame(n), small.frame(n), what=f"frame {n}")


# ---------------------------------------------------------------------------
# Alpha passes through untouched
# ---------------------------------------------------------------------------

def _rgba_source(env, n, alpha_levels):
    rng = np.random.default_rng(3)
    frames = []
    for i in range(n):
        planes = [rng.integers(0, 256, (16, 32), dtype=np.uint8) for _ in range(3)]
        a = np.full((16, 32), alpha_levels[i], dtype=np.uint8)
        a[:, ::3] = (alpha_levels[i] + 7) % 256  # vary spatially too
        planes.append(a)
        frames.append(planes)
    return frames, env.source(frames, "RGBAP")


@pytest.mark.parametrize("call", [
    "avszip_CombMask()", "avszip_CombMaskMT()", "avszip_AdaptiveBinarize(clip2=src)", "avszip_Checkmate(tthr2=255)",
])
def test_alpha_passes_through(env, call):
    frames, src = _rgba_source(env, 5, [0, 100, 120, 130, 140])
    env.set_clip("src", src)
    out = env.eval(f"src.{call}")
    for n in range(5):
        assert np.array_equal(out.frame(n).planes[3], frames[n][3]), f"{call} frame {n}: alpha changed"


# ---------------------------------------------------------------------------
# SSIMULACRA2 converts YUV with each input's own _Matrix / _ColorRange
# ---------------------------------------------------------------------------

def _yuv444_pair(env, matrix, full_range, shift, h=64):
    fr = H.make_frames("YUV444P16", w=96, h=h, n=1, seed=1, kind="smooth")
    if shift:
        fr[0][1] = np.clip(fr[0][1].astype(np.int64) + shift, 0, 65535).astype(np.uint16)
    return H.Pair(env, "YUV444P16", fr, props={} if matrix is None else {"_Matrix": matrix}, full_range=full_range)


@pytest.mark.parametrize("m_ref,m_dist,full,h", [
    (1, 1, None, 64), (6, 6, None, 64), (9, 9, None, 64),  # Rec.709 / Rec.601 / Rec.2020 on SD
    (6, 6, None, 720),  # Rec.601 on HD
    (1, 6, None, 64),   # each input keeps its own tag
    (1, 1, True, 64),   # full range
    (None, None, None, 64), (None, None, None, 720),  # untagged: resolution fallback
])
def test_ssimulacra2_honors_matrix(env, m_ref, m_dist, full, h):
    r = _yuv444_pair(env, m_ref, full, 0, h)
    d = _yuv444_pair(env, m_dist, full, 2000, h)
    env.set_clip("ref", r.avs)
    env.set_clip("dist", d.avs)
    a = env.eval("avszip_SSIMULACRA2(ref, dist)").frame(0)
    v = H.vs_frames(H.vszip().SSIMULACRA2(r.vs, d.vs))[0]
    # The returned planes are the converted reference: a wrong matrix shows up
    # here far above float rounding (0.18-0.23 in the original report).
    H.compare_planes(a, v, tol=1e-5, what="converted reference")
    assert abs(a.props["SSIMULACRA2"] - v.props["SSIMULACRA2"]) < 0.02


# ---------------------------------------------------------------------------
# Internal depth conversions keep alpha (Dither converts it by rounding)
# ---------------------------------------------------------------------------

def _alpha_frames(pt, bits, w=48, h=32, n=3, seed=7):
    fam, _, _, ssw, ssh = avsrt_describe(pt)
    dt = np.uint8 if bits == 8 else np.uint16
    peak = (1 << bits) - 1
    base = H.make_frames({"RGBAP": "RGBP", "YUVA420": "YV12", "YUVA444P10": "YUV444P10", "YUVA420P10": "YUV420P10"}[pt], w=w, h=h, n=n, seed=seed)
    rng = np.random.default_rng(seed)
    return [planes + [rng.integers(0, peak + 1, (h, w)).astype(dt)] for planes in base]


def avsrt_describe(pt):
    import avsrt
    env = avsrt.Env()
    try:
        return avsrt.describe(env.eval(f'BlankClip(width=16, height=16, pixel_type="{pt}")').pixel_type)
    finally:
        env.close()


@pytest.mark.parametrize("pt,color,bits", [("RGBAP", "RGBP", 8), ("YUVA420", "YV12", 8), ("YUVA444P10", "YUV444P10", 10)])
def test_deband_low_bit_alpha(env, pt, color, bits):
    frames = _alpha_frames(pt, bits)
    src = env.source(frames, pt)
    env.set_clip("src", src)
    out = env.eval("src.avszip_Deband(grain=[2.0])")
    assert out.pixel_type == src.pixel_type
    pair = H.Pair(env, color, [f[:3] for f in frames])
    v = H.vszip().Deband(pair.vs, grain=[2.0])
    for n, (x, y) in enumerate(zip(out.frames(), H.vs_frames(v))):
        H.compare_planes(x, y, what=f"{pt} frame {n}", planes=range(3))
        assert np.array_equal(x.planes[3], frames[n][3]), f"{pt} frame {n}: alpha changed"


@pytest.mark.parametrize("pt,bits", [("RGBAP", 8), ("YUVA444P10", 10)])
def test_dither_alpha_round_trip(env, pt, bits):
    frames = _alpha_frames(pt, bits)
    env.set_clip("src", env.source(frames, pt))
    up = env.eval("src.avszip_Dither(bitdepth=16)")
    back = env.eval(f"src.avszip_Dither(bitdepth=16).avszip_Dither(bitdepth={bits}, dither_type=3)")
    peak = (1 << bits) - 1
    for n in range(3):
        a16 = up.frame(n).planes[3].astype(np.int64)
        want = np.round(frames[n][3].astype(np.float64) * 65535 / peak)
        assert np.array_equal(a16, want), "alpha is converted as full range, rounded"
        assert np.array_equal(back.frame(n).planes[3], frames[n][3]), "alpha survives the round trip"


@pytest.mark.parametrize("ref_pt,dist_pt", [("YUVA420", "YUV420P10"), ("YV12", "YUVA420P10"), ("YUVA420P10", "YV12")])
def test_xpsnr_mixed_depth_with_alpha(env, ref_pt, dist_pt):
    def clip(pt, seed):
        bits = 10 if "P10" in pt else 8
        color = {"YUVA420": "YV12", "YUVA420P10": "YUV420P10"}.get(pt, pt)
        fr = H.make_frames(color, w=64, h=32, n=2, seed=seed)
        if "A" in pt:
            dt = np.uint8 if bits == 8 else np.uint16
            fr = [f + [np.full((32, 64), 77, dtype=dt)] for f in fr]
        return env.source(fr, pt), env.source([f[:3] for f in fr], color)
    r_a, r_c = clip(ref_pt, 1)
    d_a, d_c = clip(dist_pt, 2)
    for name, c in (("ra", r_a), ("rc", r_c), ("da", d_a), ("dc", d_c)):
        env.set_clip(name, c)
    with_alpha = env.eval("avszip_XPSNR(ra, da, verbose=false)")
    without = env.eval("avszip_XPSNR(rc, dc, verbose=false)")
    for n in range(2):
        fa, fb = with_alpha.frame(n), without.frame(n)
        H.compare_planes(fa, fb, what=f"frame {n}", planes=range(3))
        for k in ("XPSNR_Y", "XPSNR_U", "XPSNR_V"):
            assert fa.props[k] == fb.props[k], k


# ---------------------------------------------------------------------------
# SSIMULACRA2: per-frame metadata and reference alpha
# ---------------------------------------------------------------------------

def _spliced(env, H_pair, name, tags):
    """`H_pair` with per-frame props `tags` (list of dicts, one per frame) on
    both hosts."""
    env.set_clip(name + "src", H_pair.avs)
    parts = []
    for n, t in enumerate(tags):
        parts.append(f"{name}src.Trim({n},-1)" + "".join(f'.propSet("{k}",{v})' for k, v in t.items()))
    env.set_clip(name, env.eval(" ++ ".join(parts)))
    vs_tags = []
    for t in tags:
        vt = {("_Range" if k == "_ColorRange" else k): (1 - v if k == "_ColorRange" else v) for k, v in t.items()}
        vs_tags.append(vt)
    return H.core.std.Splice([H_pair.vs[n].std.SetFrameProps(**t) if t else H_pair.vs[n] for n, t in enumerate(vs_tags)])


@pytest.mark.parametrize("ref_tags,dist_tags", [
    ([{"_Matrix": 6}, {"_Matrix": 1}], [{"_Matrix": 6}, {"_Matrix": 1}]),   # valid -> valid
    ([{}, {"_Matrix": 1}], [{}, {"_Matrix": 1}]),                             # missing -> valid
    ([{"_Matrix": 2}, {"_Matrix": 9}], [{"_Matrix": 2}, {"_Matrix": 9}]),     # unspecified -> valid
    ([{"_Matrix": 1}, {}], [{"_Matrix": 1}, {}]),                             # valid -> missing
    ([{"_Matrix": 1}, {"_Matrix": 1}], [{"_Matrix": 1}, {"_Matrix": 9}]),     # only the distorted changes
    ([{"_ColorRange": 0}, {"_ColorRange": 1}], [{"_ColorRange": 1}, {"_ColorRange": 0}]),  # range transitions
])
def test_ssimulacra2_per_frame_metadata(env, ref_tags, dist_tags):
    ref = H.make_frames("YUV444P16", w=96, h=64, n=2, seed=1, kind="smooth")
    ref[1] = [p.copy() for p in ref[0]]
    dist = [[p.copy() for p in f] for f in ref]
    for f in dist:
        f[1][:] = np.clip(f[1].astype(np.int64) + 2000, 0, 65535)
    r = H.Pair(env, "YUV444P16", ref)
    d = H.Pair(env, "YUV444P16", dist)
    rv = _spliced(env, r, "r", ref_tags)
    dv = _spliced(env, d, "d", dist_tags)
    a = env.eval("avszip_SSIMULACRA2(r, d)")
    v = H.vszip().SSIMULACRA2(rv, dv)
    vf = H.vs_frames(v)
    for order in ([0, 1], [1, 0, 1, 0]):  # repeated and reversed requests
        for n in order:
            x = a.frame(n)
            H.compare_planes(x, vf[n], tol=1e-5, what=f"frame {n}")
            assert abs(x.props["SSIMULACRA2"] - vf[n].props["SSIMULACRA2"]) < 0.02
    env.set_clip("m", a)
    mt = env.eval("m.Prefetch(4)")
    for n in (1, 0, 1):
        H.compare_planes(mt.frame(n), vf[n], tol=1e-5, what=f"Prefetch frame {n}")


def _yuva(env, pt, alpha_kind, seed=1):
    import avsrt
    fam, bits, has_alpha, ssw, ssh = avsrt_describe(pt)
    color = {"YUVA420": "YV12", "YUVA444P16": "YUV444P16", "YUVA444PS": "YUV444PS"}[pt]
    fr = H.make_frames(color, w=64, h=32, n=1, seed=seed, kind="smooth")
    if bits == 32:
        a = np.linspace(0, 1, 64 * 32, dtype=np.float32).reshape(32, 64) if alpha_kind == "ramp" else np.full((32, 64), 0.25, np.float32)
    else:
        peak = (1 << bits) - 1
        a = (np.arange(64 * 32).reshape(32, 64) * 7 % (peak + 1)).astype(np.uint8 if bits == 8 else np.uint16)
        if alpha_kind != "ramp":
            a[:] = peak // 3
    return fr, [f + [a] for f in fr], color, bits


@pytest.mark.parametrize("pt", ["YUVA420", "YUVA444P16", "YUVA444PS"])
def test_ssimulacra2_keeps_reference_alpha(env, pt):
    color_frames, frames, color, bits = _yuva(env, pt, "ramp")
    ref = env.source(frames, pt)
    plain = env.source(color_frames, color)
    env.set_clip("ra", ref)
    env.set_clip("rc", plain)
    out = env.eval("avszip_SSIMULACRA2(ra, rc)")
    # YUV(A) is converted inside the metric: the reference comes back as is
    assert out.has_alpha and out.family == "yuv"
    f = out.frame(0)
    for p in range(4):
        assert np.array_equal(f.planes[p], frames[0][p]), f"plane {p} unchanged"
    # color is unaffected by alpha: same score as the alpha-free call
    g = env.eval("avszip_SSIMULACRA2(rc, rc)").frame(0)
    assert f.props["SSIMULACRA2"] == g.props["SSIMULACRA2"]


def test_ssimulacra2_distorted_alpha_is_ignored(env):
    color_frames, frames, color, _ = _yuva(env, "YUVA444P16", "ramp")
    _, other_alpha, _, _ = _yuva(env, "YUVA444P16", "flat")
    env.set_clip("rc", env.source(color_frames, color))
    env.set_clip("da", env.source(frames, "YUVA444P16"))
    env.set_clip("db", env.source(other_alpha, "YUVA444P16"))
    out = env.eval("avszip_SSIMULACRA2(rc, da)")
    assert not out.has_alpha, "alpha on the distorted input alone adds no alpha plane"
    a = out.frame(0).props["SSIMULACRA2"]
    b = env.eval("avszip_SSIMULACRA2(rc, db)").frame(0).props["SSIMULACRA2"]
    c = env.eval("avszip_SSIMULACRA2(rc, rc)").frame(0).props["SSIMULACRA2"]
    assert a == b == c, "the distorted alpha does not change the score"
