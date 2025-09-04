# Refactor Plan for 001_layers_baseline/run.py

Goal: reduce `run.py` size and complexity without changing behavior. We will extract small, well‑scoped units with minimal coupling first, add tests when prudent, and verify via the existing CPU test suite. Each move should be reversible and low risk.

## Instructions to the coding assistant

- Approach each step separately, never attempt to do multiple steps at the same time.
- After the user tells you to implement a step, plan the implementation and report back to the user to run it by them.
- If the user tells you to adjust the approach, rethink it and report back to the user.
- If the user tells you that you may proceed, implement the change as planned, including a test or a test case in an existing test file located in `001_layers_baseline/tests/`.
- If you introduced a new test file, also add it to the test-running shell script in `scripts/run_cpu_tests.sh`.
- Don't run the test — just tell the user that the implementation is complete, and the user can run the test; the user will run the suite themselves.
- If something broke in the tests, the user will tell you and possibly provide details, and you'll investigate the problem, immediately fix it, and ask the user to re-run the tests.
- After the user tells you that the tests completed fine, mark the step as completed in `001_layers_baseline/REFACTOR_NOTES.md` in the same format as the previous steps (checkbox emoji in the heading of the step, status line, brief notes on the implementation etc).
- After updating the refactor plan, commit the changes (don't wait for the user to tell you to do so — do it immediately after changing the refactor plan); use a multiline commit message; the top line should have a concise description of the step (don't add anything like "step 3" etc — this is irrelevant from the point of view of commit history; describe the change itself; no need to mention refactor notes — they are also irrelevant from the point of view of commit history).

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

3) ✅ Unembedding helper + fp32 shadow selection (low–medium risk)
- Status: completed
- What: extracted the fp32-shadow selection and the unembedding matmul into reusable helpers; replaced the inline `_unembed_mm` closure and ad‑hoc promotion logic in `run.py`.
- Why: logic is reused across passes and in Prism sidecar; centralizing dtype/device handling reduces duplication and future drift.
- Target: `layers_core/unembed.py` (new):
  - `prepare_unembed_weights(W_U, b_U, force_fp32: bool) -> (W, b)`
  - `unembed_mm(X, W, b, cache=None) -> logits`
  - Per‑device tiny cache to avoid repeated host↔device transfers.
  - No mutation of model parameters; helpers return analysis‑only tensors.
 Implementation:
  - Added `layers_core/unembed.py` with the two helpers.
  - Exported via `layers_core/__init__.py`.
  - Refactored `run.py` to use `prepare_unembed_weights` and `unembed_mm`; preserved console messages and diagnostics (`unembed_dtype`, `use_fp32_unembed`).
  - Kept `safe_cast_for_unembed` usage unchanged; maintained Prism paths.
  - Added unit test `001_layers_baseline/tests/test_unembed.py` and wired it into `scripts/run_cpu_tests.sh`.
 Validation:
  - User ran `scripts/run_cpu_tests.sh` — CPU-only suite passed.
  - Behavior change: none; outputs and schemas unchanged.

4) ✅ Record construction (medium risk)
- Status: completed
- What: centralized record shaping into pure helpers and refactored `run.py`'s `print_summary` to delegate.
- Why: consolidates duplicated schema logic; makes behavior easier to test and maintain.
- Target: `layers_core/records.py` (new): `make_record(...) -> dict` and `make_pure_record(...) -> dict`.
 Implementation:
  - Added `layers_core/records.py` with `_pack_topk`, `make_record`, and `make_pure_record` (preserve `[token, prob]` schema and `.item()` extraction).
  - Exported via `layers_core/__init__.py`.
  - Updated `run.py` `print_summary(...)` to call these helpers and append to `json_data`.
  - Added unit test `001_layers_baseline/tests/test_records.py`; wired into `scripts/run_cpu_tests.sh`.
 Validation:
  - User ran `scripts/run_cpu_tests.sh` — CPU-only suite passed.
  - Behavior change: none; JSON/CSV schemas and console output unchanged.

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
