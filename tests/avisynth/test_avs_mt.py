"""Concurrency and request-order contracts under AviSynth+ MT.

Every filter is MT_NICE_FILTER: output frame n must depend only on n.
Each case renders serially, then through Prefetch(4) with shuffled and
repeated requests, and the two must be identical. Also guards the harness
itself: a deliberate difference must be detected.
"""

import random

import numpy as np
import pytest

import harness as H

CASES = [
    ("Limiter", "YUV420P16", "tv_range=true"),
    ("BoxBlur", "YUV420P16", "hradius=3, vradius=3, vpasses=2"),
    ("PlaneAverage", "YV12", "exclude=[-1], planes=[0, 1, 2]"),
    ("PlaneMinMax", "YUV444P10", "minthr=0.01, maxthr=0.01"),
    ("Bilateral", "YUV420P16", "sigmaS=[2.0]"),
    ("BilateralDither", "YUV420P16", "radius=[4]"),
    ("MosquitoNR", "YV12", ""),
    ("Deband", "YV12", "grain=[2.0], dynamic_grain=true"),
    ("Deband", "YUV420PS", "sample_mode=7"),
    ("Dither", "YUV420P16", "bitdepth=8, dither_type=3"),
    ("CLAHE", "YUV420P10", ""),
    ("Compress", "YV12", "codec=1"),
    ("CombMask", "YV12", ""),
    ("Checkmate", "YV12", "tthr2=8"),
    ("EEDI3", "YV12", "field=3"),
    ("EEDI3H", "YUV420P16", "field=1, dh=true"),
    ("WNNM", "YUV420PS", "radius=1, block_size=4"),
    ("NLMeans", "YUV420P16", "d=2"),
    ("AdaptiveGrainMask", "YUV420P16", ""),
    ("LimitFilter", "YUV420P16", "src=src, dark_thr=[2.0]"),
    ("XPSNR", "YV12", "distorted=src.avszip_BoxBlur(), verbose=false"),
    ("SSIMULACRA2", "RGBPS", "distorted=src.avszip_BoxBlur()"),
    ("BM3D", "YUV420PS", "radius=1"),
    ("BM3D", "YUV444PS", "radius=2, ref=src, chroma=true"),
    ("Tonemap", "YUV420P16", "smoothing_period=2.0"),
    ("Tonemap", "RGBPS", "smoothing_period=1.0, contrast_recovery=0.5"),
]


def frames_equal(a, b, what):
    assert len(a.planes) == len(b.planes), what
    for p, (x, y) in enumerate(zip(a.planes, b.planes)):
        if not np.array_equal(x, y, equal_nan=True):
            raise AssertionError(f"{what}: plane {p} differs")
    for k, v in b.props.items():
        if k.startswith("_") and k not in ("_FieldBased", "_ColorRange", "_DurationNum", "_DurationDen"):
            continue
        assert a.props.get(k) == v, f"{what}: prop {k}: {a.props.get(k)} vs {v}"


@pytest.mark.parametrize("name,fmt,args", CASES, ids=[f"{c[0]}-{c[1]}" for c in CASES])
def test_prefetch_matches_serial(env, name, fmt, args):
    pair = H.Pair(env, fmt, H.make_frames(fmt, w=96, h=64, n=8, kind="noise"))
    env.set_clip("src", pair.avs)
    call = f"src.avszip_{name}({args})"
    serial = env.eval(call)
    ref = [serial.frame(n) for n in range(serial.num_frames)]

    mt = env.eval(f"{call}.Prefetch(4)")
    order = list(range(mt.num_frames))
    random.Random(7).shuffle(order)
    order += order[:5] + [0, mt.num_frames - 1, 0]
    for n in order:
        frames_equal(mt.frame(n), ref[n], f"{name} frame {n}")


def test_harness_detects_a_one_lsb_difference(env):
    fmt = "YUV420P16"
    pair = H.Pair(env, fmt, H.make_frames(fmt))
    a, v = H.run_both(env, pair, "Limiter", {"tv_range": True})
    fa, fv = a.frame(0), H.vs_frames(v, [0])[0]
    fa.planes[1][3, 5] ^= 1
    with pytest.raises(AssertionError, match="samples differ"):
        H.compare_planes(fa, fv)
    fa.props["x"] = 1.0
    fv.props["x"] = 1.0 + 1e-9
    with pytest.raises(AssertionError):
        H.compare_props(fa, fv, ["x"])
