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

1) [ ] Consolidate Prism sidecar usage
- Scope: Replace inline Prism record assembly in run.py with calls to prism_sidecar.append_prism_record and append_prism_pure_next_token everywhere they're duplicated.
- Rationale: Remove obvious duplication and keep one implementation of sidecar logic.
- Deliverables: run.py updated to use helpers; no behavior/schema change; zero numeric changes.
- Tests: Add an equivalence unit test that compares inline vs helper outputs on deterministic tensors (no model/HF).
- Rollback: Revert to previous inline blocks.

2) [ ] Extract get_residual_safely to layers_core.hooks
- Scope: Move the inline helper from run.py into hooks.get_residual_safely(cache, layer) and reuse.
- Rationale: Centralize residual lookup and uniform error messages.
- Deliverables: New helper with docstring; run.py uses it; no behavior change.
- Tests: Extend tests/test_hooks.py with success and failure cases.
- Rollback: Inline the function again.

3) [ ] Extract last-layer consistency computation
- Scope: Move temperature/KL and head-transform diagnostics into a new layers_core/consistency.py: compute_last_layer_consistency(...).
- Rationale: Isolate numerics; easier unit testing and reuse across lenses later.
- Deliverables: New module + run.py call site update; identical JSON fields.
- Tests: New tests/test_consistency.py with deterministic tensors; assert keys and values.
- Rollback: Inline the logic in run.py again.

4) [ ] Introduce lens adapters and adopt NormLens immediately
- Scope: Add layers_core/lenses/ with a minimal base interface and implement NormLensAdapter that exactly reproduces the current normalization + unembed path.
- Rationale: Establish a pluggable lens boundary without duplication. The adapter replaces the inline path right away.
- Deliverables: Base interface + NormLensAdapter; run.py calls the adapter instead of the inline path; no behavior/schema change.
- Tests: New tests/test_lenses_basic.py to verify identical logits on deterministic tensors (adapter vs inline baseline snapshot).
- Rollback: Revert this commit; no unused code left behind.

5) [ ] Create pass-level runner for a single prompt/variant
- Scope: Add layers_core/passes.py with run_prompt_pass(...), encapsulating:
  - layer-0 decode, post-block sweep, pure-next-token emission
  - per-lens handling (baseline lens + optional sidecar lenses)
  - optional raw-vs-norm sampling integration via existing raw_lens helpers
- Adoption: Use only for the positive/orig pass in run.py.
- Rationale: Reduce monolith; keep changes localized.
- Deliverables: New module; run.py calls run_prompt_pass for orig; identical records/JSON.
- Tests: Add tests/test_pass_runner_minimal.py with a mock model and deterministic tensors.
- Rollback: Call sites revert to inlined loops.

6) [ ] Adopt pass runner for ablation and control
- Scope: Replace duplicated loops for no_filler and control with run_prompt_pass.
- Rationale: Eliminate repetition; unify code paths.
- Deliverables: run.py simplified; identical ablation deltas and control summary (margin, first_pos).
- Tests: Extend pass runner test to cover control margin wiring and ablation variant tagging.
- Rollback: Reintroduce inline loops.

7) [ ] Convert Prism path to a lens adapter and route through the pass runner
- Scope: Implement PrismLensAdapter under the same lenses/ interface and switch run.py to use it via the pass runner.
- Rationale: Unify all lenses behind one interface; reduce special-casing in run.py.
- Deliverables: Adapter that wraps whitening + Q + unembed; sidecar writer invoked via the common lens flow; filenames remain unchanged (e.g., -records-prism.csv).
- Tests: Extend tests/test_lenses_basic.py to cover Prism adapter on synthetic tensors and ensure sidecar CSV schemas match existing outputs.
- Rollback: Revert adapter wiring; keep Prism via helpers.

8) [ ] Extract lightweight probes
- Scope: Move test prompt emission and temperature exploration to layers_core/probes.py:
  - emit_test_prompts(model, prompts, decode_id)
  - emit_temperature_exploration(model, prompt, decode_id)
- Rationale: Reduce run.py size; keep probes orthogonal to main pass.
- Deliverables: New module; run.py delegates; identical JSON lists.
- Tests: tests/test_probes.py with fixed logits to assert shapes/keys.
- Rollback: Inline back into run.py.

9) [ ] Introduce small context objects to reduce closures
- Scope: Add tiny dataclasses:
  - UnembedContext: {W, b, force_fp32, cache}
  - PrismContext: {stats, Q, active, placement}
- Rationale: Replace hard-to-test closures with explicit dependencies.
- Deliverables: Contexts passed into pass runner and lenses; no behavior change.
- Tests: Type/shape sanity tests in existing modules (no new public API).
- Rollback: Use local variables again.

10) [ ] Optional cleanup (no behavior change)
- Scope: Tidy decode_id into a tiny util; gate prints; minor docstrings.
- Rationale: Readability improvements after structure is stable.
- Deliverables: No schema change; logs preserved by default.
- Tests: None beyond lint/format; keep behavior identical.
- Rollback: N/A; keep minimal.

---

## Tuned Lens Readiness (no changes yet)

Purpose: Ensure the refactor allows adding a Tuned Lens without touching core orchestration or changing baseline outputs.

- Pluggable lenses: The lens adapters (Step 4) form a stable boundary. We will later add a TunedLensAdapter that consumes per-layer heads and produces logits.
- Sidecar generalization: Keep baseline CSVs/JSON byte-identical. Additional lenses emit parallel sidecar CSVs with identical schemas (e.g., -records-tuned.csv), mirroring the current Prism flow.
- Metrics semantics: KL-to-final and cos_to_final continue to reference the model’s final head. For Tuned Lens, additional lens-relative diagnostics can live in JSON sidecars without altering public CSV schemas.
- Artifact management: Mirror Prism loaders with tuned_lens.load_tuned_lens_artifacts(art_dir) and provenance. Respect device/dtype and analysis-only weights.
- QA hooks: The extracted “last-layer consistency” utility can compare any two logits (e.g., Norm vs Tuned). Raw-vs-norm sampling remains intact.

No code is added for Tuned Lens in this plan; the refactor merely creates the seams where it fits cleanly later.

---

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
