# Refactor Plan for 001_layers_baseline/run.py

Goal: reduce `run.py` size and complexity without changing behavior.
We will extract well‑scoped units with minimal coupling first, add tests when prudent, and verify via the existing CPU test suite.
Each move should be reversible and low risk.

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
- Prefer extracting leaf utilities/classes first; deep orchestration - later.
- Keep interfaces explicit; pass dependencies instead of using module globals/closures.
- Add or reuse unit tests to guard semantics before touching larger flows.

## Refactor steps (ranked by safety/impact)

The plan below is incremental and behavior-preserving. Each step is small, reversible, and adds or reuses tests. Only proceed to the next step after the previous is merged and tests are green.

Legend: [ ] pending · [x] completed

1) [x] Consolidate Prism sidecar usage
- Scope: Replace inline Prism record assembly in run.py with calls to prism_sidecar.append_prism_record and append_prism_pure_next_token everywhere they're duplicated.
- Rationale: Remove obvious duplication and keep one implementation of sidecar logic.
- Deliverables: run.py updated to use helpers; no behavior/schema change; zero numeric changes.
- Tests: Add an equivalence unit test that compares inline vs helper outputs on deterministic tensors (no model/HF).
- Rollback: Revert to previous inline blocks.
- Status: completed 2025-09-06
- Notes:
  - Replaced manual Prism rows with helpers in orig/ablation/control paths (records and pure next-token).
  - Fixed control-pass Prism variable/placement bugs (use prism_enabled_ctl, prism_Q_ctl; guard placement with try/except).
  - prism_sidecar entropy now uses bits_entropy_from_logits to match baseline numerics.
  - Added tests/test_prism_sidecar_equivalence.py and updated scripts/run_cpu_tests.sh.

2) [x] Extract get_residual_safely to layers_core.hooks
- Scope: Move the inline helper from run.py into hooks.get_residual_safely(cache, layer) and reuse.
- Rationale: Centralize residual lookup and uniform error messages.
- Deliverables: New helper with docstring; run.py uses it; no behavior change.
- Tests: Extend tests/test_hooks.py with success and failure cases.
- Rollback: Inline the function again.
 - Status: completed 2025-09-06
 - Notes:
   - Added hooks.get_residual_safely with helpful KeyError including nearby keys.
   - run.py imports and uses the helper; removed inline closure.
   - tests/test_hooks.py covers success and failure; included in scripts/run_cpu_tests.sh already.

3) [x] Extract last-layer consistency computation
- Scope: Move temperature/KL and head-transform diagnostics into a new layers_core/consistency.py: compute_last_layer_consistency(...).
- Rationale: Isolate numerics; easier unit testing and reuse across lenses later.
- Deliverables: New module + run.py call site update; identical JSON fields.
- Tests: New tests/test_consistency.py with deterministic tensors; assert keys and values.
- Rollback: Inline the logic in run.py again.
 - Status: completed 2025-09-06
 - Notes:
   - Added layers_core/consistency.py with compute_last_layer_consistency.
   - Replaced inline block in run.py with call; preserved outputs and thresholds.
   - Ensured lens_top1_id derives from logits via topk(1) to match original logic.
   - Added tests/test_consistency.py; updated scripts/run_cpu_tests.sh.

4) [x] Introduce lens adapters and adopt NormLens immediately
- Scope: Add layers_core/lenses/ with a minimal base interface and implement NormLensAdapter that exactly reproduces the current normalization + unembed path.
- Rationale: Establish a pluggable lens boundary without duplication. The adapter replaces the inline path right away.
- Deliverables:
  - base interface + NormLensAdapter;
  - run.py calls the adapter instead of the inline path;
  - no unused code left behind;
  - no behavior/schema change.
- Tests: New tests/test_lenses_basic.py to verify identical logits on deterministic tensors (adapter vs inline baseline snapshot).
- Rollback: Revert this commit; no unused code left behind.
- Status: completed 2025-09-07
- Notes:
  - Added `layers_core/lenses/base.py` (LensAdapter) and `layers_core/lenses/norm.py` (NormLensAdapter), with `layers_core/lenses/__init__.py` re-export.
  - Replaced inline normalization+unembed in `run.py` (orig/ablation/control; L0 and post-block) with NormLensAdapter calls.
  - Adapter always receives the pre-normalization residual; explicit normalization retained only for Prism/diagnostics.
  - Standardized variable names to `resid_raw` and `resid_norm`; Prism whitening consistently uses `resid_norm`.
  - Added `tests/test_lenses_basic.py`; updated `scripts/run_cpu_tests.sh` to include it.

5) [x] Create pass-level runner for a single prompt/variant
- Scope: Add layers_core/passes.py with run_prompt_pass(...), encapsulating:
  - layer-0 decode, post-block sweep, pure-next-token emission
  - per-lens handling (baseline lens + optional sidecar lenses)
  - optional raw-vs-norm sampling integration via existing raw_lens helpers
- Adoption in run.py: use the runner only for the positive/orig pass.
- Rationale: Reduce monolith; keep changes localized.
- Deliverables:
  - new module;
  - run.py calls run_prompt_pass for orig;
  - no unused code left behind;
  - identical records/JSON.
- Tests: Add tests/test_pass_runner_minimal.py with a mock model and deterministic tensors.
- Rollback: Call sites revert to inlined loops.
 - Status: completed 2025-09-07
 - Notes:
   - Added layers_core/passes.py with run_prompt_pass covering hooks, L0 and post-block decode via NormLensAdapter, per-position records, pure next-token, raw-vs-norm sampling, and last-layer consistency.
   - run.py imports and delegates the positive/orig pass to run_prompt_pass; merges returned prism diag deltas into diagnostics.prism_summary.
   - Restored keep_residuals parity: L0 and per-layer residuals saved with clean model name; used os.path.join; dtype fallback matches baseline.
   - Prism parity: unembed path uses unembed_mm with cache; pass-wide enable flag decided at L0 and carried forward.
   - Test added and strengthened (tests/test_pass_runner_minimal.py) and included in scripts/run_cpu_tests.sh.

6) [x] Adopt pass runner for ablation and control
- Scope: Replace duplicated loops for no_filler and control with run_prompt_pass.
- Rationale: Eliminate repetition; unify code paths.
- Deliverables:
  - run.py simplified;
  - identical ablation deltas and control summary (margin, first_pos);
  - no unused code left behind.
- Tests: Extend pass runner test to cover control margin wiring and ablation variant tagging.
- Rollback: Reintroduce inline loops.
 - Status: completed 2025-09-07
 - Notes:
   - Migrated both no_filler (ablation) and control passes to layers_core/passes.run_prompt_pass.
   - Added enable_raw_lens_sampling flag to the runner; sampling disabled for ablation/control to preserve orig-only QA summary.
   - Restored parity for residual saving: control pass does not save residuals (avoid overwrite).
   - Runner now emits Prism L0 per-position sidecar rows for parity with baseline coverage.
   - Removed dead helpers and unused imports in run.py; redundant variant reset removed.
   - Added control_ids threading in runner for control margin; tests cover control margin and runner returns.

7) [x] Convert Prism path to a lens adapter and route through the pass runner
- Scope: Implement PrismLensAdapter under the same lenses/ interface and switch run.py to use it via the pass runner.
- Rationale: Unify all lenses behind one interface; reduce special-casing in run.py.
- Deliverables:
   - adapter that wraps whitening + Q + unembed;
   - sidecar writer invoked via the common lens flow;
   - filenames remain unchanged (e.g., -records-prism.csv);
   - no unused code left behind.
- Tests: Extend tests/test_lenses_basic.py to cover Prism adapter on synthetic tensors and ensure sidecar CSV schemas match existing outputs.
 - Rollback: Revert adapter wiring; keep Prism via helpers.
 - Status: completed 2025-09-12
 - Notes:
   - Added lenses/prism.py (PrismLensAdapter) and exported via lenses/__init__.py.
   - Refactored layers_core/passes.py to call the adapter for L0 and post‑block; sidecars emitted via helpers.
   - Fixed residual saving bug; behavior‑preserving policy: normalized when Prism enabled, else raw (L0 and post‑block).
   - Applied safe top‑k clamping in passes.py and run.py debug paths for small‑vocab stubs.
   - Tests: extended test_lenses_basic (adapter parity), new test_prism_placement_failure (placement error disables adapter; no sidecars), extended test_pass_runner_minimal (Prism sidecar rows, keep‑residuals policy).
   - Increased verbosity and added __main__ shims for Prism/lenses tests to run under plain‑python harness; adjusted prism sidecar smoke test to use TemporaryDirectory (no repo pollution).

8) [x] Extract lightweight probes
- Scope: Move test prompt emission and temperature exploration to layers_core/probes.py:
  - emit_test_prompts(model, prompts, decode_id)
  - emit_temperature_exploration(model, prompt, decode_id)
- Rationale: Reduce run.py size; keep probes orthogonal to main pass.
- Deliverables:
  - new module;
  - run.py delegates;
  - identical JSON lists;
  - no unused code left behind.
- Tests: tests/test_probes.py with fixed logits to assert shapes/keys.
- Rollback: Inline back into run.py.
 - Status: completed 2025-09-12
 - Notes:
   - Added layers_core/probes.py with emit_test_prompts and emit_temperature_exploration; wired run.py to delegate; outputs and schemas unchanged.
   - Added tests/test_probes.py and updated scripts/run_cpu_tests.sh; tests verified under the CPU suite.
   - Polish: wrapped emit_test_prompts in torch.no_grad() and widened decode_id type hints to accept tensor or int without mismatch.

9) [x] Introduce small context objects to reduce closures
- Scope: Add tiny dataclasses:
  - UnembedContext: {W, b, force_fp32, cache}
  - PrismContext: {stats, Q, active, placement}
- Rationale: Replace hard-to-test closures with explicit dependencies.
- Deliverables: Contexts passed into pass runner and used by lenses; no behavior change.
- Tests: Type/shape sanity tests in existing modules (no new public API).
- Rollback: Use local variables again.
 - Status: completed 2025-09-12
 - Notes:
   - Added layers_core/contexts.py with UnembedContext and PrismContext.
   - Refactored layers_core/passes.run_prompt_pass to accept `unembed_ctx` and `prism_ctx` and to mirror any Prism placement errors into both `diag_delta` and `prism_ctx.placement_error`.
   - Updated run.py to construct contexts once and pass them to all runs; updated tests (test_pass_runner_minimal.py, test_prism_placement_failure.py) accordingly. No output/schema changes.

10) [x] Optional cleanup (no behavior change)
- Scope: Tidy decode_id into a tiny util; gate prints; minor docstrings.
- Rationale: Readability improvements after structure is stable.
- Deliverables: No schema change; logs preserved by default.
- Tests: None beyond lint/format; keep behavior identical.
- Rollback: N/A; keep minimal.
 - Status: completed 2025-09-12
 - Notes:
   - Added layers_core/token_utils.make_decode_id and switched run.py to use it everywhere (removed ad‑hoc lambdas in gold-alignment fallbacks).
   - Introduced _vprint and CLI_ARGS.quiet to gate info/debug prints; default remains verbose; errors/warnings unchanged.
   - Added tests/test_token_utils.py and wired into scripts/run_cpu_tests.sh; all tests pass.
   - Probes annotations aligned with decode helper (Callable[[Any], str]).

---

## Polish / Backlog

- [x] Prism placement helper (completed 2025-09-12)
  - Implemented as layers_core/prism_utils.ensure_prism_Q_on and used by PrismLensAdapter._ensure_Q_on.
  - Consolidates placement behavior in one utility, surfaces a consistent error string, and keeps per-pass enablement unchanged. No external API or behavior change.

---

## Tuned Lens Readiness (minimal adapter seam) ✅

Purpose: provide a safe, tested seam for a future Tuned Lens without touching orchestration or changing baseline outputs.

Status: completed 2025-09-13

Scope implemented (behavior-neutral):
- Added `layers_core/lenses/tuned.py` with `TunedLensAdapter` implementing the standard lens interface.
  - Uses architecture‑aware normalization (`get_correct_norm_module` + `apply_norm_or_skip`).
  - Decodes via supplied per‑layer tuned heads `(W, b)`; returns logits as float32.
  - Strict vs non‑strict missing‑head behavior: `strict=True` raises; `strict=False` returns `None` and records `diag.missing_layers`.
  - Explicit guards: floating dtypes only (rejects quantized integer weights), shape checks for `W`/`b` vs `d_model`/`vocab`.
  - Notes in docstring to use a distinct cache per lens to avoid collisions.
- Exposed the adapter from `layers_core/lenses/__init__.py`.
- Tests extended in `tests/test_lenses_basic.py`:
  - Deterministic parity vs inline reference for pre‑norm/post‑norm and pre/post‑block cases.
  - Edge cases: missing head (non‑strict → None), integer weights rejected, shape mismatch rejected.

Intentionally deferred (to avoid scope creep and preserve byte‑identical outputs):
- No artifact loader (`load_tuned_lens_artifacts`), no provenance or device/dtype placement policies.
- No pass‑runner integration, no tuned sidecar buffers/CSVs, no diagnostics.
- No CLI flags in launcher/worker.

Rationale:
- Keep baseline outputs and orchestration untouched while establishing the extension point.
- Address reviewer concerns early (shape/dtype guards, missing‑head behavior) without committing to artifact formats.
- Enable incremental adoption later (loader, sidecar, CLI) with minimal surface area change.

## Verification Matrix

Existing tests to rely on:
- numerics, metrics, collapse_rules, norm_utils, csv_io, raw_lens, prism modules
- orchestrator smoke test (no network) to ensure outputs exist

New tests to add as steps land:
- Prism sidecar equivalence (Step 1)
- hooks.get_residual_safely behavior (Step 2)
- consistency.compute_last_layer_consistency (Step 3)
- lenses basic shape/typing (Step 4)
- pass runner minimal end-to-end (Step 5/6)
- probes emitters (Step 8)
- light sanity for context objects (Step 9)

All tests are CPU-only and avoid network/HF; follow scripts/run_cpu_tests.sh.
