"""Subprocess isolation for legal minimum geometries, not just rejected sizes."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


TINY_CASES = [
    ("Checkmate", 3, 5, 8, {"tthr2": 8}, 120),
    ("CombMask", 1, 3, 8, {"metric": 0, "mthresh": 0}, 0),
    ("CombMask", 1, 3, 8, {"metric": 1, "mthresh": 9}, 0),
    ("CombMaskMT", 1, 3, 8, {}, 0),
    ("EEDI3", 1, 2, 32, {"field": 1}, 0.25),
    ("EEDI3H", 2, 1, 32, {"field": 1}, 0.25),
    ("EEDI3", 1, 1, 32, {"field": 1, "dh": True}, 0.25),
    ("EEDI3H", 1, 1, 32, {"field": 1, "dh": True}, 0.25),
    ("WNNM", 1, 1, 32, {"block_size": 1, "group_size": 1, "radius": 0}, 0.25),
    ("WNNM", 1, 1, 32, {"block_size": 1, "group_size": 1, "radius": 1}, 0.25),
    ("Bilateral", 1, 1, 32, {"sigmaS": [1.0], "sigmaR": [0.02], "algorithm": [1]}, 0.25),
    ("Bilateral", 33, 2, 32, {"sigmaS": [1.0], "sigmaR": [0.02], "algorithm": [1]}, 0.25),
    ("Bilateral", 17, 2, 8, {"sigmaS": [1.0], "sigmaR": [0.02], "algorithm": [1]}, 120),
    ("Bilateral", 1, 3, 32, {"sigmaS": [1.0], "sigmaR": [0.02], "algorithm": [1]}, 0.25),
    ("Deband", 1, 1, 32, {"thr": [0.0], "grain": [0.0]}, 0.25),
    # 4:2:0: the chroma planes are half as tall as the clip, so the 4x4 cases
    # run algorithm 1 on 2-row chroma (review.4 R1), processed or skipped.
    ("Bilateral", 4, 4, 8, {"sigmaS": [1.0], "sigmaR": [0.02], "algorithm": [1]}, 120, "yuv420"),
    ("Bilateral", 4, 4, 8, {"sigmaS": [1.0], "sigmaR": [0.02], "algorithm": [1], "planes": [0]}, 120, "yuv420"),
    ("Bilateral", 8, 6, 32, {"sigmaS": [1.0], "sigmaR": [0.02], "algorithm": [1]}, 0.25, "yuv420"),
]


def _family(case):
    return case[6] if len(case) > 6 else "gray"


TINY_IDS = [f"{c[0]}-{_family(c)}-{c[1]}x{c[2]}-{i}" for i, c in enumerate(TINY_CASES)]
TINY_PARAMETERS = [pytest.param(c, id=cid) for c, cid in zip(TINY_CASES, TINY_IDS)]
CHILD = r'''
import json, pathlib, sys
import numpy as np
root, backend, recipe = sys.argv[1:]
sys.path.insert(0, root)
name, w, h, bits, kwargs, expected, *rest = json.loads(recipe)
family = rest[0] if rest else "gray"
yuv = family == "yuv420"
shapes = [(h, w)] + ([(h // 2, w // 2)] * 2 if yuv else [])
if backend == "vs":
    from conftest import _load_vszip
    import vapoursynth as vs
    core = _load_vszip()
    fmt = {(False, 8): vs.GRAY8, (False, 32): vs.GRAYS, (True, 8): vs.YUV420P8, (True, 32): vs.YUV420PS}[(yuv, bits)]
    src = core.std.BlankClip(width=w, height=h, format=fmt, length=3,
                             color=[expected] * len(shapes))
    create = lambda: getattr(core.vszip, name)(src, **kwargs)
    def planes(n):
        with out.get_frame(n) as f:
            return [np.array(f[p], copy=True) for p in range(f.format.num_planes)]
else:
    sys.path.insert(0, str(pathlib.Path(root) / "avisynth"))
    import avsrt, harness
    env = avsrt.Env(plugins=[harness.AVSZIP])
    pt = {(False, 8): "Y8", (False, 32): "Y32", (True, 8): "YV12", (True, 32): "YUV420PS"}[(yuv, bits)]
    src = env.source([[np.full(s, expected, dtype=np.uint8 if bits==8 else np.float32) for s in shapes]
                      for _ in range(3)], pt)
    create = lambda: env.invoke("avszip_" + name, src, **kwargs)
    def planes(n):
        return out.frame(n).planes
# Every recipe is a legal geometry: it must evaluate, not be rejected.
out = create()
for n in [2, 0, 1, 2]:
    for p in planes(n):
        assert np.isfinite(p).all(), "nonfinite output"
        np.testing.assert_allclose(p, expected, rtol=0, atol=1e-6)
print("PASSED")
'''


def check_tiny(backend, recipe):
    root = Path(__file__).resolve().parent
    try:
        result = subprocess.run([sys.executable, "-c", CHILD, str(root), backend, json.dumps(recipe)],
                                cwd=root, env=os.environ.copy(), capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired as e:
        raise AssertionError(f"{backend} {recipe}: child timed out after 30 seconds") from e
    assert result.returncode == 0, f"{backend} {recipe}: exit {result.returncode}\n{result.stderr[-2500:]}"
    assert result.stdout.strip().splitlines()[-1] == "PASSED"
