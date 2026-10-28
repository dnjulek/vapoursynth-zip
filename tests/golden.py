"""Snapshot-golden infrastructure.

Each test file declares a list of `Case`s (format x geometry x filter args).
Golden per-plane stats live in tests/goldens/<filter>.json, keyed by the case
id. `pytest --update-goldens` regenerates the JSON from the current build;
normal runs compare against it. Review `git diff tests/goldens/` after every
regeneration: goldens lock in *current* behavior, they don't prove it correct.

Partial runs (-k, -m, node ids, --lf) merge into the existing JSON. A *full*
run additionally checks for stale data: a stored key that no test checked, or
a golden file no test module refers to, fails the session; a full
`pytest --update-goldens` prunes them instead. Golden files used by a module
that had skipped tests in the run are exempt (their keys may simply not have
been reached).
"""

import json
import math
import re
from pathlib import Path

import pytest
import vapoursynth as vs

from helpers import plane_stats

TESTS_DIR = Path(__file__).resolve().parent
GOLDENS_DIR = TESTS_DIR / "goldens"


def fmt_name(fmt: int) -> str:
    return vs.PresetVideoFormat(int(fmt)).name


def _fmt_val(v) -> str:
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_fmt_val(x) for x in v) + "]"
    if isinstance(v, float):
        return format(v, "g")
    return str(v)


class Case:
    """One golden case: input format + geometry variant + filter kwargs.

    `variant` tags cases whose clip setup differs beyond plain kwargs (e.g.
    "ref" for a second-clip argument the test file builds itself)."""

    def __init__(self, fmt: int, geometry: str = "full", args: dict | None = None, variant: str = ""):
        self.fmt = int(fmt)
        self.geometry = geometry
        self.args = dict(args or {})
        self.variant = variant

    @property
    def id(self) -> str:
        argstr = ",".join(f"{k}={_fmt_val(v)}" for k, v in sorted(self.args.items())) or "default"
        s = f"{fmt_name(self.fmt)}|{self.geometry}|{argstr}"
        return f"{s}|{self.variant}" if self.variant else s

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        return f"Case({self.id})"


def grid(**axes) -> list[dict]:
    """Cartesian product of the given axes only: grid(a=[1,2], b=[3]) ->
    [{a:1,b:3}, {a:2,b:3}]."""
    out = [{}]
    for key, values in axes.items():
        out = [{**d, key: v} for d in out for v in values]
    return out


def sweep(
    *,
    base_fmt: int,
    base_args: dict | None = None,
    base_geometry: str = "full",
    formats: tuple = (),
    args: tuple = (),
    geometries: tuple = (),
    variant: str = "",
) -> list[Case]:
    """Axis-sweep composition: vary one axis at a time around the base config
    (formats x base_args, base_fmt x each args dict, geometries x base_args).
    Deliberately not a full cartesian product - that explodes case counts
    without adding much coverage. Hand-pick interacting combos separately."""
    base_args = dict(base_args or {})
    out: list[Case] = []
    seen: set[str] = set()

    def add(case: Case) -> None:
        if case.id not in seen:
            seen.add(case.id)
            out.append(case)

    for f in formats:
        add(Case(f, base_geometry, base_args, variant))
    for a in args:
        add(Case(base_fmt, base_geometry, {**base_args, **a}, variant))
    for g in geometries:
        add(Case(base_fmt, g, base_args, variant))
    return out


def golden_stats(clip: vs.VideoNode, n: int = 0) -> dict:
    """Per-plane {avg, min, max}. f16 is measured after a Point resize to f32
    because std.PlaneStats has no half support."""
    if clip.format.sample_type == vs.FLOAT and clip.format.bits_per_sample == 16:
        clip = clip.resize.Point(format=clip.format.replace(bits_per_sample=32).id)
    out = {}
    for p in range(clip.format.num_planes):
        s = plane_stats(clip, plane=p, n=n)
        st = {"avg": s["PlaneStatsAverage"], "min": s["PlaneStatsMin"], "max": s["PlaneStatsMax"]}
        for k, v in st.items():
            assert math.isfinite(v), f"plane {p} {k} is not finite: {v}"
        # avg is normalized to 0-1 for int formats while min/max stay raw, so
        # only min <= max can be asserted format-independently
        assert st["min"] <= st["max"], f"plane {p}: min > max: {st}"
        out[f"p{p}"] = st
    return out


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        assert math.isfinite(value), f"non-finite value not storable as golden: {value}"
        return float(value)
    raise TypeError(f"not golden-storable: {value!r} ({type(value).__name__})")


def _assert_close(expected, actual, rel: float, abs_: float, path: str, key_abs: dict | None = None) -> None:
    """key_abs: an absolute tolerance per leaf key (e.g. {"min": 8}), used
    instead of `abs_` for values stored under that key."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected dict, got {type(actual).__name__}"
        assert set(expected) == set(actual), f"{path}: keys {sorted(actual)} != {sorted(expected)}"
        for k in expected:
            if key_abs and k in key_abs and not isinstance(expected[k], (dict, list)):
                _assert_close(expected[k], actual[k], rel, key_abs[k], f"{path}.{k}")
            else:
                _assert_close(expected[k], actual[k], rel, abs_, f"{path}.{k}", key_abs)
    elif isinstance(expected, list):
        assert isinstance(actual, (list, tuple)), f"{path}: expected list, got {type(actual).__name__}"
        assert len(expected) == len(actual), f"{path}: length {len(actual)} != {len(expected)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            _assert_close(e, a, rel, abs_, f"{path}[{i}]", key_abs)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert actual == pytest.approx(expected, rel=rel, abs=abs_), (
            f"{path}: {actual!r} != golden {expected!r} (rel={rel}, abs={abs_})"
        )
    else:
        assert actual == expected, f"{path}: {actual!r} != golden {expected!r}"


class GoldenStore:
    def __init__(self, update: bool):
        self.update = update
        self._data: dict[str, dict] = {}
        self._dirty: set[str] = set()
        self._seen: dict[str, set[str]] = {}

    def _file(self, name: str) -> Path:
        return GOLDENS_DIR / f"{name}.json"

    def _get(self, name: str) -> dict:
        if name not in self._data:
            f = self._file(name)
            self._data[name] = json.loads(f.read_text()) if f.is_file() else {}
        return self._data[name]

    def check_value(self, filter_name: str, key, value, rel: float = 1e-6, abs_: float = 1e-9, key_abs: dict | None = None) -> None:
        """Compare (or record, with --update-goldens) an arbitrary
        JSON-serializable value: number, list, or dict of numbers.
        `key_abs`: absolute tolerances for leaves under those keys."""
        key = str(key)
        self._seen.setdefault(filter_name, set()).add(key)
        value = _jsonable(value)
        data = self._get(filter_name)
        if self.update:
            data[key] = value
            self._dirty.add(filter_name)
            return
        if key not in data:
            pytest.fail(f"no golden for {filter_name}[{key}]; run `pytest --update-goldens`")
        _assert_close(data[key], value, rel, abs_, f"{filter_name}[{key}]", key_abs)

    def check(self, filter_name: str, case: Case, clip: vs.VideoNode, n: int = 0, rel: float | None = None,
              extrema_abs: float | None = None, avg_abs: float = 1e-9) -> None:
        """Compare per-plane output stats against the stored golden.
        `extrema_abs`: an absolute tolerance for the per-plane min/max (the
        averages keep `rel`), for filters whose single extreme pixels move by
        a code between FMA and no-FMA builds. `avg_abs`: the averages'
        absolute floor (they are normalized to 0-1; `rel` alone is too tight
        for an average near 0, such as centered chroma)."""
        if rel is None:
            rel = 1e-6
        key_abs = {"min": extrema_abs, "max": extrema_abs} if extrema_abs is not None else None
        self.check_value(filter_name, case, golden_stats(clip, n), rel=rel, abs_=avg_abs, key_abs=key_abs)

    def stale(self, skipped_modules: set[str]) -> dict[str, list[str] | None]:
        """Golden data a full run did not use: {file name: stale keys}, or
        {file name: None} for a file no test module refers to at all.

        Which modules use a file is found statically (a quoted
        `"<name>"` literal in a tests/test_*.py file), so a module that was
        skipped entirely still protects its goldens."""
        users: dict[str, set[str]] = {}
        sources = {f.name: f.read_text(encoding="utf-8") for f in TESTS_DIR.glob("test_*.py")}
        out: dict[str, list[str] | None] = {}
        for f in sorted(GOLDENS_DIR.glob("*.json")):
            name = f.stem
            pat = re.compile(r"[\"']" + re.escape(name) + r"[\"']")
            users[name] = {mod for mod, text in sources.items() if pat.search(text)}
            if not users[name]:
                out[name] = None
                continue
            if users[name] & skipped_modules:
                continue
            keys = set(self._get(name))
            unseen = sorted(keys - self._seen.get(name, set()))
            if unseen:
                out[name] = unseen
        return out

    def prune(self, stale: dict[str, list[str] | None]) -> None:
        """--update-goldens on a full run: drop stale keys / orphan files."""
        for name, keys in stale.items():
            if keys is None:
                self._file(name).unlink()
                self._data.pop(name, None)
                self._dirty.discard(name)
                continue
            data = self._get(name)
            for k in keys:
                data.pop(k, None)
            self._dirty.add(name)

    def save(self) -> None:
        if not self._dirty:
            return
        GOLDENS_DIR.mkdir(exist_ok=True)
        for name in sorted(self._dirty):
            data = {k: self._data[name][k] for k in sorted(self._data[name])}
            self._file(name).write_text(json.dumps(data, indent=1) + "\n")
