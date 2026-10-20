"""Cold output graphs, changing frames, repeated edges, and exclusive scratch."""

from concurrent.futures import ThreadPoolExecutor
import random

import numpy as np
import vapoursynth as vs

from contract_oracles import VSHost, pattern


SCHEDULE_CASES = [
    ("Checkmate", "YV24", vs.YUV444P8, 8, {"tthr2": 8}),
    ("CombMask", "YV24", vs.YUV444P8, 8, {"mthresh": 9}),
    ("Deband", "YUV444P16", vs.YUV444P16, 16,
     {"grain": [2.0], "dynamic_grain": True, "seed": 17}),
    ("EEDI3", "YV24", vs.YUV444P8, 8, {"field": 3, "mdis": 4}),
    ("WNNM", "RGBPS", vs.RGBS, 32,
     {"sigma": [2.0], "radius": 1, "block_size": 4, "bm_range": 2, "ps_range": 2}),
    ("PlaneAverage", "RGBPS", vs.RGBS, 32, {"exclude": [-1], "planes": [0, 1, 2]}),
    ("PlaneMinMax", "YUV444P16", vs.YUV444P16, 16, {"planes": [0, 1, 2], "minthr": 0.1}),
]


def check_schedule(host, case):
    name, avs_fmt, fmt, bits, kwargs = case
    frames = [[pattern(bits, 32, 24, plane=p, frame=n) for p in range(3)] for n in range(5)]

    def graph():
        source = host.source(frames, avs_fmt, fmt)
        source = host.sequence(source, [{"AuditFrame": n} for n in range(5)])
        # Output filters are distinct. The concurrent graph has never been
        # rendered by the sequential path, so its cache cannot fake coverage.
        return host.apply(name, source, **kwargs)

    serial, parallel = graph(), graph()
    count = 10 if name == "EEDI3" else 5
    assert serial.num_frames == parallel.num_frames == count
    expected = [host.frame(serial, n) for n in range(count)]
    order = list(range(count))
    random.Random(73).shuffle(order)
    order += [count-1, 0, count//2, count-1, 0]
    if isinstance(host, VSHost):
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(lambda n: host.frame(parallel, n), order))
    else:
        host.env.set_clip("auditParallel", parallel)
        parallel = host.env.eval("auditParallel.Prefetch(4)")
        actual = [host.frame(parallel, n) for n in order]
    for n, (pixels, props) in zip(order, actual):
        ref_pixels, ref_props = expected[n]
        assert len(pixels) == len(ref_pixels)
        for p, (a, b) in enumerate(zip(pixels, ref_pixels)):
            assert np.isfinite(a).all(), (name, n, p)
            np.testing.assert_array_equal(a, b, err_msg=f"{name} frame {n} plane {p}")
        assert props["AuditFrame"] == (n//2 if name == "EEDI3" else n)
        for key, value in ref_props.items():
            if key.startswith(("psm", "SSIMULACRA", "XPSNR")):
                assert props[key] == value, (name, n, key)
