# vszip test suite

AI generated pytest tests.

---

## Quick start

```sh
# from the repo root
zig build                           # build the Debug plugin
pytest -v                           # run the whole suite
pytest tests/test_boxblur.py        # one file
```

The suite needs `vapoursynth` importable (activate the virtualenv where it's
installed).

Run the suite against **both** build modes before trusting a change; tolerances
are wide enough that both stay green:

```sh
zig build                          && pytest -v
zig build -Doptimize=ReleaseFast   && pytest -v
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
---

## Notes & gotchas

- **Build before testing.** The suite tests `zig-out/`, not the source — rebuild
  after any Zig change.
- **Debug vs ReleaseFast.** Both must stay green. If a filter ever needs a wider
  tolerance, widen only that filter and note why.
- **Tolerance** is `rel=1e-6` for every format (f16 included — its output is
  deterministic enough across Debug/Release that no looser bound is needed). A
  filter that genuinely needs a wider bound can pass `rel=` to `golden.check`.
- **Sensitivity check** when adding goldens: tweak one stored value by ~1% and
  confirm the matching test fails, then revert.
- The golden harness asserts basic invariants on every check (stats finite,
  `min <= max`).
