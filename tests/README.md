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

## Independent interaction tests

`test_contract_oracles.py` and `avisynth/test_avs_contract_oracles.py` run the
same independent NumPy equations against each plugin separately. They cover
statistics at SIMD boundaries, exclusions, quantile ranks, property isolation,
selected-plane clipping, out-of-peak integer carriers, and per-frame YUV/Gray
conversion. Shared models live in `contract_oracles.py` and have hand-checked
anchors; plugin output and goldens never generate their expected pixels.

The additional scheduling, image-sequence, and tiny-input modules check cold
graphs under concurrency, late decode/layout failures, actual gray-alpha frame
contents, and legal minimum geometries. Dangerous native calls run in bounded
child processes. `avisynth/test_avs_hostile_args.py` reuses the VS hostile-float
manifest with real V11 doubles/longs, rather than sanitized script literals.
Only the plugin's own validation counts as a rejection there: a call the host
refuses before the plugin runs (wrong type, unknown argument) reports CRASH,
so a broken case cannot pass as "rejected".

Both loaders verify the loaded VS plugin path against the requested build
(`VSZIP_LIB` or `zig-out`), so an autoloaded wheel cannot silently substitute for
the checkout. The AviSynth DLL is selected with `AVSZIP_LIB`.

The tiny cases include Bilateral algorithm 1 on planes under three rows,
which used to underflow the reverse vertical recurrence (review.4 R1): gray
1- and 2-row clips, and 4:2:0 clips whose 2-row chroma is processed or skipped
(`planes=[0]`). They must evaluate and preserve the constant input.

These checks extend the coverage matrix; they do not prove the absence of all
bugs or replace the host-free allocation-failure tests and platform/CPU runs.

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
