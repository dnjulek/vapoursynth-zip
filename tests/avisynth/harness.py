"""Differential harness: run the same input through vszip (VapourSynth) and
avszip (AviSynth+) and compare planes and frame properties.

Inputs are numpy frames fed bit-identically to both hosts. Formats are named
by their AviSynth pixel_type; FORMATS maps them to VapourSynth formats.
"""

import os
import pathlib

import numpy as np
import vapoursynth as vs

import avsrt

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
# Both plugins come from the same `zig build -Davisynth=true` (zig-out/bin).
AVSZIP = os.environ.get("AVSZIP_LIB", str(REPO_ROOT / "zig-out" / "bin" / "avszip.dll"))
VSZIP = os.environ.get("VSZIP_LIB", str(REPO_ROOT / "zig-out" / "bin" / "vszip.dll"))

core = vs.core

# AviSynth pixel_type -> VapourSynth format
FORMATS = {
    "Y8": vs.GRAY8, "Y10": vs.GRAY10, "Y12": vs.GRAY12, "Y14": vs.GRAY14, "Y16": vs.GRAY16, "Y32": vs.GRAYS,
    "YV12": vs.YUV420P8, "YUV420P10": vs.YUV420P10, "YUV420P12": vs.YUV420P12, "YUV420P16": vs.YUV420P16,
    "YUV420PS": vs.YUV420PS,
    "YV16": vs.YUV422P8, "YUV422P10": vs.YUV422P10, "YUV422P16": vs.YUV422P16, "YUV422PS": vs.YUV422PS,
    "YV24": vs.YUV444P8, "YUV444P10": vs.YUV444P10, "YUV444P12": vs.YUV444P12, "YUV444P14": vs.YUV444P14,
    "YUV444P16": vs.YUV444P16, "YUV444PS": vs.YUV444PS,
    "RGBP": vs.RGB24, "RGBP10": vs.RGB30, "RGBP12": vs.RGB36, "RGBP16": vs.RGB48, "RGBPS": vs.RGBS,
}

INT8 = ["Y8", "YV12", "YV24", "RGBP"]
INT16 = ["Y16", "YUV420P16", "YUV444P16", "RGBP16"]
HIGH = ["Y10", "YUV420P10", "YUV422P10", "YUV444P12", "YUV444P14", "RGBP10"]
FLOAT = ["Y32", "YUV420PS", "YUV444PS", "RGBPS"]
ALL = INT8 + HIGH + INT16 + FLOAT

_vszip_loaded = False


def vszip():
    global _vszip_loaded
    if not _vszip_loaded:
        if not hasattr(core, "vszip"):
            core.std.LoadPlugin(VSZIP)
        _vszip_loaded = True
    return core.vszip


def fmt_info(name):
    f = core.get_video_format(FORMATS[name])
    return f


def plane_shapes(name, w, h):
    f = fmt_info(name)
    shapes = []
    for p in range(f.num_planes):
        if p == 0 or f.color_family != vs.YUV:
            shapes.append((h, w))
        else:
            shapes.append((h >> f.subsampling_h, w >> f.subsampling_w))
    return shapes


def make_frames(name, w=96, h=64, n=4, seed=1, kind="mixed"):
    """Deterministic content: noise + gradients + hard edges + flat areas,
    covering the full nominal range (and chroma around neutral)."""
    f = fmt_info(name)
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        planes = []
        for p, (ph, pw) in enumerate(plane_shapes(name, w, h)):
            yy, xx = np.mgrid[0:ph, 0:pw]
            grad = (xx / max(pw - 1, 1) * 0.6 + yy / max(ph - 1, 1) * 0.3 + 0.05 * i) % 1.0
            noise = rng.random((ph, pw))
            if kind == "noise":
                v = noise
            elif kind == "smooth":
                v = grad
            else:
                v = 0.55 * grad + 0.45 * noise
                v[: ph // 4, : pw // 4] = 0.5  # flat block
                v[ph // 2:, pw // 2: pw // 2 + 2] = 1.0  # hard edge
                v[ph - 1, :] = 0.0
            chroma = f.color_family == vs.YUV and p > 0
            if f.sample_type == vs.FLOAT:
                arr = (v - 0.5 if chroma else v).astype(np.float32)
            else:
                peak = (1 << f.bits_per_sample) - 1
                arr = np.round(v * peak).astype(np.uint8 if f.bits_per_sample == 8 else np.uint16)
            planes.append(arr)
        frames.append(planes)
    return frames


def vs_source(frames, name, props=None):
    h, w = frames[0][0].shape
    blank = core.std.BlankClip(format=FORMATS[name], width=w, height=h, length=len(frames), fpsnum=24, fpsden=1, keep=True)

    def sel(n, f):
        fout = f.copy()
        for p in range(fout.format.num_planes):
            np.asarray(fout[p])[:] = frames[n][p]
        return fout

    clip = core.std.ModifyFrame(blank, blank, sel)
    if props:
        clip = core.std.SetFrameProps(clip, **props)
    return clip


def vs_frames(clip, ns=None):
    out = []
    for n in range(clip.num_frames) if ns is None else ns:
        f = clip.get_frame(n)
        planes = [np.array(f[p], copy=True) for p in range(f.format.num_planes)]
        props = {k: f.props[k] for k in f.props.keys()}
        out.append(avsrt.Frame(planes, props))
    return out


class Pair:
    """The same input in both hosts. `full_range` tags both with the host's
    range property (AviSynth _ColorRange / VapourSynth _Range)."""

    def __init__(self, env, name, frames, props=None, full_range=None):
        self.name = name
        self.frames = frames
        vs_props = dict(props or {})
        props = dict(props or {})
        if full_range is not None:
            vs_props["_Range"] = 1 if full_range else 0
            props["_ColorRange"] = 0 if full_range else 1
        self.vs = vs_source(frames, name, vs_props)
        self.avs = env.source(frames, name)
        if props:
            env.set_clip("_tmp", self.avs)
            sets = "".join(f'.propSet("{k}", {v})' for k, v in props.items())
            self.avs = env.eval("_tmp" + sets)


def compare_planes(a, b, tol=0.0, what="", planes=None):
    assert len(a.planes) >= len(b.planes) or planes is not None, what
    for p in range(len(b.planes)) if planes is None else planes:
        x, y = a.planes[p], b.planes[p]
        assert x.shape == y.shape, f"{what} plane {p}: shape {x.shape} vs {y.shape}"
        assert x.dtype == y.dtype, f"{what} plane {p}: dtype {x.dtype} vs {y.dtype}"
        if tol == 0.0:
            if not np.array_equal(x, y):
                diff = np.argwhere(x != y)
                yy, xx = diff[0]
                raise AssertionError(
                    f"{what} plane {p}: {len(diff)} samples differ; first at (y={yy}, x={xx}): avs={x[yy, xx]} vs={y[yy, xx]}")
        else:
            d = np.abs(x.astype(np.float64) - y.astype(np.float64))
            assert np.nanmax(d) <= tol, f"{what} plane {p}: max diff {np.nanmax(d)} > {tol}"


def compare_props(a, b, keys, tol=0.0, what=""):
    for k in keys:
        assert k in a.props, f"{what}: avs frame lacks prop {k}"
        if k == "_ColorRange":
            # AviSynth+ _ColorRange (0 = full) <-> VapourSynth R4 _Range (1 = full)
            assert "_Range" in b.props, f"{what}: vs frame lacks prop _Range"
            assert a.props[k] == 1 - b.props["_Range"], f"{what}: _ColorRange={a.props[k]} vs _Range={b.props['_Range']}"
            continue
        assert k in b.props, f"{what}: vs frame lacks prop {k}"
        x, y = a.props[k], b.props[k]
        if isinstance(y, (list, tuple)):
            assert len(x) == len(y), f"{what} {k}: {x} vs {y}"
            pairs = list(zip(x, y))
        else:
            pairs = [(x, y)]
        for u, v in pairs:
            if isinstance(v, float) or isinstance(u, float):
                assert abs(u - v) <= tol or (np.isnan(u) and np.isnan(v)), f"{what} {k}: avs={u} vs={v}"
            else:
                if isinstance(v, str):
                    v = v.encode()
                assert u == v, f"{what} {k}: avs={u!r} vs={v!r}"


class Var:
    """A script variable reference (e.g. a second clip) in AviSynth args;
    `vs` is the matching VapourSynth value."""

    def __init__(self, name, vs_value=None):
        self.name = name
        self.vs = vs_value


def avs_call(name, args):
    """`args` dict -> AviSynth named-argument string (lists become arrays)."""
    parts = []
    for k, v in args.items():
        if isinstance(v, Var):
            parts.append(f"{k}={v.name}")
        elif isinstance(v, bool):
            parts.append(f"{k}={'true' if v else 'false'}")
        elif isinstance(v, (list, tuple)):
            parts.append(f"{k}=[{', '.join(repr(float(x)) if isinstance(x, float) else str(int(x)) for x in v)}]")
        elif isinstance(v, str):
            parts.append(f'{k}="{v}"')
        elif isinstance(v, float):
            parts.append(f"{k}={v!r}")
        else:
            parts.append(f"{k}={int(v)}")
    return f"vszip_{name}({', '.join(parts)})" if not parts else f"vszip_{name}({', '.join(parts)})"


def run_both(env, pair, name, avs_args=None, vs_args=None, clip_prefix=""):
    """Apply filter `name` in both hosts; returns (avs_clip, vs_clip)."""
    avs_args = dict(avs_args or {})
    vs_args = dict(avs_args if vs_args is None else vs_args)
    for k, v in list(vs_args.items()):
        if isinstance(v, bool):
            vs_args[k] = int(v)
        elif isinstance(v, Var):
            vs_args[k] = v.vs
    env.set_clip("src", pair.avs)
    call = avs_call(name, avs_args)
    a = env.eval(f"src.{call}" if not clip_prefix else clip_prefix + call)
    b = getattr(vszip(), name)(pair.vs, **vs_args)
    return a, b


def assert_same(env, pair, name, avs_args=None, vs_args=None, tol=0.0, props=(), prop_tol=0.0, frames=None):
    a, b = run_both(env, pair, name, avs_args, vs_args)
    assert a.num_frames == b.num_frames
    fa = a.frames(frames)
    fb = vs_frames(b, frames)
    for i, (x, y) in enumerate(zip(fa, fb)):
        compare_planes(x, y, tol, what=f"{name}{avs_args} {pair.name} frame {i}")
        compare_props(x, y, props, prop_tol, what=f"{name}{avs_args} {pair.name} frame {i}")
    return fa, fb
