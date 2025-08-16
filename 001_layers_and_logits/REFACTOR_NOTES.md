# Refactor Notes

Goal: Iterative module extraction with immediate unit tests; preserve behavior and outputs exactly while making core pieces reusable and testable.

## Iteration Slices (Checklist)

- [x] norm_utils (apply_norm_or_skip, detect_model_architecture, get_correct_norm_module) — moved to `layers_core/`; tests green
- [x] numerics (bits_entropy_from_logits, safe_cast_for_unembed) — moved to `layers_core/`; tests green
- [x] csv_io (records CSV, pure next-token CSV) — moved to `layers_core/`; tests green
- [x] collapse_rules (copy-collapse, semantic-collapse) — moved to `layers_core/`; tests green
- [x] device_policy (dtype choice, unembed promotion) — moved to `layers_core/`; tests green
- [x] hooks (attach/detach residual hooks) — moved to `layers_core/`; tests green
- [x] run_dir (run-latest rotation) — moved to `layers_core/`; tests green
- [x] update kl_sanity_test imports
- [ ] cleanup re-exports in run.py

Status counters: In Progress 0 · Done 8 · Pending 0

## Slice 1 — norm_utils (Done)

Scope:
- apply_norm_or_skip (LayerNorm and RMSNorm, epsilon inside sqrt)
- _get_rms_scale (robust getter for RMS scale param)
- detect_model_architecture (structural, ln2 vs mlp order)
- get_correct_norm_module (architecture-aware γ selection for probe-before/after)

Invariants:
- RMSNorm epsilon is inside sqrt; LayerNorm applies γ and β faithfully.
- Architecture detection uses block child order; GPT-J/Falcon/NeoX are post-norm.
- Norm selection rules:
  - Pre-norm: after block → next block ln1; last layer → ln_final; before block → current ln1
  - Post-norm: after block → current ln2; before block → current ln1

Tests (added here):
- Epsilon placement correctness w.r.t. manual computation.
- Architecture detection and norm selection for mock pre/post blocks.

Notes:
- Introduced `layers_core/` subpackage and moved `norm_utils.py` to `layers_core/norm_utils.py`.
- Added `conftest.py` to make `layers_core` importable from repo root during pytest runs.
- Updated `run.py` and tests to import from `layers_core.norm_utils` using absolute imports.
- Tests for this slice pass with the project venv.

## Invariants (Global)

- CSV schemas and field order remain identical; `rest_mass` retained.
- Entropy computation identical (bits from full softmax; no NaN/neg-zero).
- Copy-collapse thresholds and margin semantics unchanged.
- Device/dtype policy unchanged (CUDA fp16; Gemma bf16; MPS fp16; CPU fp32).

## API Contracts (selected)

- `apply_norm_or_skip(residual: Tensor, norm_module: nn.Module|None) -> Tensor`
  - Applies model’s own norm (LN or RMS), preserving dtype/device; returns input if `norm_module` is None.
- `detect_model_architecture(model) -> Literal['pre_norm','post_norm']`
- `get_correct_norm_module(model, layer_idx: int, probe_after_block: bool) -> nn.Module|None`
- `_get_rms_scale(norm_mod) -> Tensor|None`

## Test Matrix (per module)

- norm_utils: ε placement; scaling equivalence (unit-normalize then γ); pre/post detection; norm selection for before/after and last layer.
- numerics: uniform vs peaked entropy; stability on extremes; casting across fp16/bf16/fp32/int8; explicit `force_fp32_unembed` parameter.
- csv_io: header, row length, rest_mass correctness, quoting.
- collapse_rules: threshold+margin correctness; entropy fallback toggle; tokenization edge-cases; semantic equality via trimmed text.
- device_policy: dtype table; Gemma bf16 override; FP32 unembed toggling.
- hooks: only intended hooks attached; tensors detached; cleanup works.
- run_dir: rotation with/without timestamp; deterministic via injected `now`.

## Slice 3 — csv_io (Done)

Scope:
- `write_csv_files` for records and pure next-token CSVs; schema preserved (rest_mass, flags).

Notes:
- `run.py` updated to pass `TOP_K_VERBOSE` explicitly to writer.
- Tests added in `test_csv_io.py` verifying headers, row lengths, and rest_mass.

## Slice 4 — collapse_rules (Done)

Scope:
- `detect_copy_collapse` with threshold + margin + entropy fallback.
- `is_semantic_top1` for trimmed string equality against ground truth.

Notes:
- `run.py` updated to use these helpers in both Layer 0 and per-layer pure next-token checks; behavior identical.
- Tests added in `test_collapse_rules.py` for margin/threshold, entropy fallback, and semantic matching.

## Slice 2 — numerics (Done)

Scope:
- `bits_entropy_from_logits` (numerically safe entropy in bits)
- `safe_cast_for_unembed` (dtype casting policy; preserves quantized kernels)

Invariants:
- Entropy equals log2(V) for uniform logits and ~0 for near-delta.
- Casting does not upcast for quantized `W_U` (e.g., int8) and only forces fp32 when explicitly requested.

Notes:
- Introduced explicit parameter `force_fp32_unembed` (decouples helpers from CLI globals); updated `run.py` call sites accordingly.
- Tests added in `test_numerics.py` (CPU-only) and run green with project venv.
- csv_io: header, row length, rest_mass correctness, quoting.
- collapse_rules: threshold+margin correctness; entropy fallback toggle; tokenization edge-cases.
- device_policy: dtype table; Gemma bf16 override; FP32 unembed toggling.
- hooks: only intended hooks attached; tensors detached; cleanup works.
- run_dir: rotation with/without timestamp; deterministic via injected `now`.

## Decisions Log

- 2025-08-16: Adopted iterative extraction (module + tests per slice); keep behavior identical; no output changes.
- 2025-08-16: Created `layers_core/` subpackage for reusable helpers; added `conftest.py` to support absolute imports during tests.
- 2025-08-16: Extracted numerics helpers; added explicit `force_fp32_unembed` flag and updated usage in `run.py`.
- 2025-08-16: Extracted CSV I/O helpers; kept CSV schema stable and added unit tests.
- 2025-08-16: Extracted collapse rules; centralized thresholds/margins and fallback semantics; added unit tests.
- 2025-08-16: Extracted device policy helpers; codified dtype selection and auto unembed promotion; added unit tests.
- 2025-08-16: Extracted hooks helpers; centralized attachment and cleanup; added unit tests with mocks.

## Next Steps

Repo Structure and Reuse Across Experiments
- Keep `001_layers_and_logits/layers_core` as-is for now; when starting `002_*`, lift it to a top-level `experiments_core/` (or `logos_core/`).
- Provide temporary shims: re-export `experiments_core.*` from `001_layers_and_logits/layers_core/__init__.py` so 001 keeps working while 002 imports the new core directly.
- Keep each experiment self-contained (CLI, prompts, results), depending only on shared core.

Anticipating PROJECT_NOTES.md (future experiments)
- Adapters (later): introduce a minimal `ModelAdapter` abstraction for tokenize/forward/hooks/unembed; start with HookedTransformer, add HF-only/multimodal later. Don’t change runtime yet; prototype adapters in tests first.
- Probes (later): keep probes as small functions/classes returning tensors/records (residual lens, ablations, patching). Avoid plugin frameworks; simple callables suffice.
- Metrics (later): use a simple dict registry `{name: fn}` with a tiny signature contract; no heavy plugin system.

Testing Strategy (scalable)
- Unit: CPU-only with mocks; add a tiny `MockModelAdapter` to test probes/metrics without network/GPUs.
- Contract: golden CSV schema tests per experiment; JSON sanity checks (e.g., `L_copy`/`L_semantic`).
- Marks: decorate network/download tests; default `-m "not network"` in pytest.ini.
- CLI smoke: one tiny path test per experiment that uses mocks (no downloads) to catch wiring regressions.

Migration Path (low risk)
- Do nothing now; when `experiments_core/` is created, move modules and add re-export shims in `001_layers_and_logits/layers_core`.
- Update new experiments to import from `experiments_core`. Flip 001 imports later once stable; remove shims at the end.

Avoid Over-Engineering
- No global configuration system; prefer small per-experiment dataclasses and explicit parameters (as done with `force_fp32_unembed`).
- No plugin frameworks; simple callables and dict registries are enough.
- Don’t unify reporting yet; just stabilize CSV schemas so downstream dashboards can evolve separately.

Quick Wins (small, high-value)
- Stabilize test harness: add `pytest.ini`, make CWD-agnostic self-test, and a `scripts/run_cpu_tests.sh` runner using `venv/bin/python` per test.
- Tighten public API: re-export helpers in `layers_core/__init__.py`; add minimal docstrings/type hints (mypy optional).
- CLI/Config: introduce a small `config.py` dataclass and thread it through `run_experiment_for_model` (no behavior change).
- Docs: update README to mention `layers_core` and testing strategy; add “add a slice” guidance here.
- Guardrails: add a lightweight orchestrator smoke test using mocks; consider pre-commit (black/isort/ruff) later.

## Quick Run

- Run only this slice’s tests (CPU): `pytest -q 001_layers_and_logits -k norm_utils -x`

## Environment Notes

- Prefer invoking the project’s interpreter directly: use `venv/bin/python ...` (or `venv/bin/pytest`) instead of relying on `source venv/bin/activate && python ...`. In sandboxes/CI, activation may not persist; the direct interpreter path avoids “ModuleNotFoundError: torch”.
- Unit tests are CPU-only and do not require network/model downloads.
- Running tests from repo root is supported: `pytest -q 001_layers_and_logits` (a `conftest.py` adds the folder to `sys.path` so `layers_core` resolves).
- For manual runs inside the folder: `cd 001_layers_and_logits && ../venv/bin/python test_norm_utils.py`.

Known gotcha:
- `test_refactored_self_test.py` calls `run.py --help` via a relative path. It passes when the CWD is `001_layers_and_logits`, e.g. `cd 001_layers_and_logits && ../venv/bin/python test_refactored_self_test.py`. Running from repo root may fail this specific check unless the subprocess path is adjusted.

Suggested “all CPU-only tests” run locally:
- From repo root:
  - `venv/bin/python 001_layers_and_logits/test_norm_utils.py`
  - `venv/bin/python 001_layers_and_logits/test_normalization.py`
  - `cd 001_layers_and_logits && ../venv/bin/python test_refactored_self_test.py && cd -`

## Testing Strategy

- In sandboxed/CI environments, avoid chaining multiple test invocations in a single shell command (e.g., using `&&`). Long, chained commands are prone to timeouts and state loss; prefer running each test as a separate command.
- Prefer direct interpreter invocations over shell activation: `venv/bin/python <test>.py` or `venv/bin/pytest ...`.
- Path-sensitive tests should set CWD explicitly or resolve target paths relative to `__file__` to avoid failures when running from repo root.
- For quick iteration: run slice-specific tests only (e.g., `venv/bin/python 001_layers_and_logits/test_numerics.py`). Use pytest locally for broader runs: `venv/bin/pytest -q 001_layers_and_logits -k "norm_utils or numerics"`.
## Slice 5 — device_policy (Done)

Scope:
- `choose_dtype(device, model_id)` with CUDA Gemma bf16 override; mps fp16; cpu fp32.
- `should_auto_promote_unembed(compute_dtype)` returns True only for fp32 models (CPU default).

Notes:
- `run.py` updated to use `choose_dtype` and `should_auto_promote_unembed`; behavior unchanged.
- Tests added in `test_device_policy.py` cover dtype table, Gemma override, and auto-promotion rule.
## Slice 6 — hooks (Done)

Scope:
- `build_cache_hook` to store detached tensors keyed by hook.name.
- `attach_residual_hooks` to register hooks for embeddings, optional positional embeddings, and resid_post per layer.
- `detach_hooks` to remove all handles safely.

Notes:
- `run.py` updated to use these helpers; behavior unchanged.
- Tests added in `test_hooks.py` with minimal HookPoint/Handle mocks; verifies cache keys, handle removal, and positional hook presence.
- Network/model-dependent tests (e.g., full `--self-test` with downloads) should be run locally with authenticated access; keep CPU-only unit tests fast and offline by default.
## Slice 7 — run_dir (Done)

Scope:
- `setup_run_latest_directory(script_dir, now_fn=datetime.now)` moves the rotation logic out of run.py; supports testable injected clock.

Notes:
- Tests in `test_run_dir.py` cover initial creation, rotation using existing timestamp file, and fallback rotation without timestamp (uses `-rotated` suffix).
- `run.py` imports and uses the helper unchanged semantically.
## Slice 8 — kl_sanity_test imports (Done)

Scope:
- Point kl_sanity_test.py at `layers_core.norm_utils` instead of importing from `run.py` to avoid CLI side effects and tighten dependencies.

Notes:
- `test_refactored_self_test.py` remains compatible since `run.py` still exposes the same function names via imports; help text check unchanged.
