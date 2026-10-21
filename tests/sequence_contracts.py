"""Isolated lazy-decoding checks shared by the ImageRead host suites."""

import json
import os
from pathlib import Path
import subprocess
import sys

from helpers import write_png


BAD_SEQUENCES = ["depth", "family", "alpha", "dimensions", "missing", "truncated", "replace-first"]
CHILD = r'''
import json, pathlib, sys
tests, backend, first, second, kind, eager = sys.argv[1:]
sys.path.insert(0, tests)
from helpers import write_png
if backend == "vs":
    from conftest import _load_vszip
    import vapoursynth as vs
    core = _load_vszip()
    create = lambda: core.vszip.ImageRead([first, second], validate=int(eager))
    read = lambda c, n: c.get_frame(n)
    error = vs.Error
else:
    sys.path.insert(0, str(pathlib.Path(tests) / "avisynth"))
    import avsrt, harness
    env = avsrt.Env(plugins=[harness.AVSZIP])
    create = lambda: env.invoke("avszip_ImageRead", path=[first, second], validate=bool(int(eager)))
    read = lambda c, n: c.frame(n)
    error = avsrt.AvsError
phase = "create"
try:
    clip = create()
    if kind == "replace-first":
        write_png(pathlib.Path(first), [[(10,20,30)] * 4 for _ in range(3)])
    phase = "frame"
    read(clip, 0 if kind == "replace-first" else 1)
    print(json.dumps({"result":"accepted", "phase":phase}))
except error as e:
    print(json.dumps({"result":"rejected", "phase":phase, "message":str(e)}))
'''


def rejected_sequence(tmp_path, backend, kind, eager):
    first = write_png(tmp_path / "first.png", [[10] * 4 for _ in range(3)], color="gray")
    second = tmp_path / "second.png"
    if kind == "depth":
        write_png(second, [[1000] * 4 for _ in range(3)], color="gray", bitdepth=16)
    elif kind == "family":
        write_png(second, [[(10, 20, 30)] * 4 for _ in range(3)])
    elif kind == "alpha":
        write_png(second, [[(10, 20)] * 4 for _ in range(3)], color="graya")
    elif kind == "dimensions":
        write_png(second, [[10] * 5 for _ in range(3)], color="gray")
    elif kind == "truncated":
        second.write_bytes(first.read_bytes()[:30])
    elif kind == "replace-first":
        second.write_bytes(first.read_bytes())
    elif kind != "missing":
        raise ValueError(kind)
    root = Path(__file__).resolve().parent
    proc = subprocess.run([sys.executable, "-c", CHILD, str(root), backend,
                           str(first), str(second), kind, str(int(eager))],
                          env=os.environ.copy(), cwd=root,
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f"{backend}/{kind}: exit {proc.returncode}\n{proc.stderr[-2000:]}"
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["result"] == "rejected", result
    expected_phase = "create" if eager and kind != "replace-first" else "frame"
    assert result["phase"] == expected_phase, result
    assert result.get("message"), result
