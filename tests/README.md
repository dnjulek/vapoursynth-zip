# vszip test suite

The suite combines independent oracles, invariants, external references, and
golden regression snapshots. See [TESTING_GUIDE.md](TESTING_GUIDE.md) before
adding a filter or changing an optimized kernel.

`tests/avisynth/` is the AviSynth+ plugin's differential suite (avszip vs vszip
on identical input; see [src/avisynth/README.md](../src/avisynth/README.md)). It
runs only when `zig build -Davisynth=true` produced `zig-out/bin/avszip.dll` and
AviSynth+ is installed; otherwise it is skipped at collection.

---

## Quick start

```sh
# from the repo root
zig build                           # build the debug plugin
pytest -v                           # run the whole suite
pytest tests/test_boxblur.py        # one file
```

The suite needs `vapoursynth` importable (activate the virtualenv where it's
installed).

Run the suite against **both** build modes before trusting a change; tolerances
are wide enough that both stay green:

```sh
zig build -Doptimize=debug   && pytest -v
zig build -Doptimize=fast    && pytest -v
```

---

## Golden snapshot tests

Most filters are regression-tested with **golden snapshots**: a stored summary of
the current output that the suite compares against on every run.

### Regenerating

```sh
pytest --update-goldens      # rewrite tests/goldens/*.json from the CURRENT build
git diff tests/goldens/      # ALWAYS review the result
```

A **full** run (no `-k`, `-m`, `--lf` or node-id arguments) also fails on stale
golden data: stored keys no test checked, or a JSON file no test module refers
to. A full `pytest --update-goldens` prunes them; partial runs never do. Files
used by a module that had skipped tests in the run are exempt.
---

## Notes & gotchas

- **Build before testing.** The suite tests `zig-out/`, not the source — rebuild
  after any Zig change.
- **debug vs fast.** Both must stay green. If a filter ever needs a wider
  tolerance, widen only that filter and note why.
- **Golden tolerance** defaults to `rel=1e-6`. Direct oracle tests should be
  bit-exact where the algorithm permits it; otherwise use the narrowest
  format-specific tolerance and document its numerical reason.
- **Sensitivity check** when adding goldens: tweak one stored value by ~1% and
  confirm the matching test fails, then revert.
- The golden harness asserts basic invariants on every check (stats finite,
  `min <= max`).
