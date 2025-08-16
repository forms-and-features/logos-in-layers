# Refactor Notes

Goal: Iterative module extraction with immediate unit tests; preserve behavior and outputs exactly while making core pieces reusable and testable.

## Iteration Slices (Checklist)

- [x] norm_utils (apply_norm_or_skip, detect_model_architecture, get_correct_norm_module) — moved to `layers_core/`; tests green
- [x] numerics (bits_entropy_from_logits, safe_cast_for_unembed) — moved to `layers_core/`; tests green
- [ ] csv_io (records CSV, pure next-token CSV)
- [ ] collapse_rules (copy-collapse, semantic-collapse)
- [ ] device_policy (dtype choice, unembed promotion)
- [ ] hooks (attach/detach residual hooks)
- [ ] run_dir (run-latest rotation)
- [ ] update kl_sanity_test imports
- [ ] cleanup re-exports in run.py

Status counters: In Progress 0 · Done 2 · Pending 6

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
- collapse_rules: threshold+margin correctness; entropy fallback toggle; tokenization edge-cases.
- device_policy: dtype table; Gemma bf16 override; FP32 unembed toggling.
- hooks: only intended hooks attached; tensors detached; cleanup works.
- run_dir: rotation with/without timestamp; deterministic via injected `now`.

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
