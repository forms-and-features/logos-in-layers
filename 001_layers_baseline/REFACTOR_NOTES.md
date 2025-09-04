# Refactor Plan for 001_layers_baseline/run.py

Goal: reduce `run.py` size and complexity without changing behavior. We will extract small, well‑scoped units with minimal coupling first, add tests when prudent, and verify via the existing CPU test suite. Each move should be reversible and low risk.

## Principles

- Preserve behavior and outputs byte‑for‑byte where feasible.
- Prefer extracting leaf utilities/classes first; avoid deep orchestration changes early.
- Keep interfaces explicit; pass dependencies instead of using module globals/closures.
- Add or reuse unit tests to guard semantics before touching larger flows.

## Refactor steps (ranked by safety/impact)

1) ✅ WindowManager (very low risk, small surface)
- Status: completed
- What: the small rolling window helper class for copy‑collapse detection inside `run.py`.
- Why: self‑contained logic; used in three passes (orig, no_filler, control); easy to unit test; reduces duplication.
- Target: `layers_core/windows.py` (new) and export via `layers_core/__init__.py`.
 Implementation:
  - Added `layers_core/windows.py` with `WindowManager` (same API and behavior).
  - Exported via `layers_core/__init__.py`.
  - Removed inline class from `run.py` and imported `WindowManager`.
  - Added unit test `001_layers_baseline/tests/test_windows.py`.
  - Included the unit test in `scripts/run_cpu_tests.sh`.
- Validation:
  - Ran `bash scripts/test.sh` — CPU-only suite passed.
 Behavior change: none; only code relocation and test addition.

2) ✅ Head transforms detection (low risk)
- Status: completed
- What: `_detect_head_transforms()` local helper that probes `model`/`cfg` for scale/softcap.
- Why: purely introspective, no side effects; makes `run.py` slimmer.
- Target: `layers_core/head_transforms.py` (new) with a single `detect_head_transforms(model) -> tuple[Optional[float], Optional[float]]`.
 Implementation:
  - Added `layers_core/head_transforms.py`.
  - Exported via `layers_core/__init__.py`.
  - Removed inline class from `run.py` and imported `head_transforms`.
  - Added unit test `001_layers_baseline/tests/test_head_transforms.py`.
  - Included the unit test in `scripts/run_cpu_tests.sh`.
- Validation:
  - Ran `bash scripts/test.sh` — CPU-only suite passed.
 Behavior change: none; only code relocation and test addition.

3) Unembedding helper + fp32 shadow selection (low–medium risk)
- What: `_unembed_mm` closure + block that promotes analysis unembedding weights to fp32 when needed.
- Why: reused in many places; centralizes dtype/device handling; reduces clutter in `run.py`.
- Target: `layers_core/unembed.py` (new):
  - `prepare_unembed_weights(W_U, b_U, force_fp32: bool) -> (W, b, dtype_str)`
  - `unembed_mm(X, W, b, cache=None) -> logits`

4) Record construction (medium risk)
- What: `print_summary()` record builder and JSON appends.
- Why: consolidate record shaping into pure functions returning row dictionaries; easier to test.
- Target: `layers_core/records.py` (new): `make_record(...) -> dict` and `make_pure_record(...) -> dict`.

5) Pure next‑token emission (medium–higher risk)
- What: `emit_pure_next_token_record(...)` (currently a large nested helper).
- Why: concentrates metrics, flags, and optional raw‑lens sample; but has many inputs and call‑site context.
- Target: later step after (1–4) so we can pass smaller, typed inputs and reuse helpers. Keep signature explicit; return the record and a small summary for `collected_pure_records`.

6) Prism sidecar emitters (higher risk)
- What: sidecar per‑position and pure‑next‑token writing (duplicated across passes).
- Why: sizeable block; already has supporting `prism.py` utils and tests; can be isolated once (1–5) are stable.
- Target: `layers_core/prism_sidecar.py` (new) with mirror interfaces to baseline emitters.

7) CLI/launcher separation (low–medium risk)
- What: `parse_cli()` + `main()` orchestration/rotation/launching.
- Why: trims the file; reduces import‑time side effects by isolating the launcher.
- Target: `001_layers_baseline/launcher.py` (later); keep `run.py` as the worker.

## Notes on Safety and Backward Compatibility

- We will not change CLI, outputs, or schema in early steps; the JSON/CSV writers and file rotation stay untouched.
- Each extraction will keep function/class names and signatures stable at call sites; only imports move.
- Where behavior must remain byte‑identical (e.g., `rest_mass` rounding), we will reuse existing helpers.
- We will avoid refactoring the multi‑pass orchestration until leaf utilities are stabilized and tested.
