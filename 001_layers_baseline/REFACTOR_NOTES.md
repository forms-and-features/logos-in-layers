# Refactor & Bugfix Plan for run.py

This document tracks targeted fixes for 001_layers_baseline/run.py. Scope is limited to correctness and robustness; no feature work. Implementation will proceed in phases.

Note: Line numbers are approximate; search by the cited code fragments when applying fixes.

## Phase 1: Must‑Fix

- Control‑pass per‑layer computation runs outside loop due to indentation; all control layers use final‑layer logits. Also isolate rolling windows per lens and per prompt variant to prevent copy‑detection contamination.

### Bug 1 — Control Pass Loop Indentation
- Summary: In the control prompt branch, the unembed and logits computation for post‑block residuals are dedented out of the `for layer in range(n_layers)` loop.
- Evidence/Location: Around the control pass, e.g. lines ~1424–1453:
  ```python
  for layer in range(n_layers):
      resid = residual_cache[f'blocks.{layer}.hook_resid_post']
      if USE_NORM_LENS:
          norm_module = get_correct_norm_module(...)
          resid = apply_norm_or_skip(resid, norm_module)
  casted = safe_cast_for_unembed(resid[0, :, :], ...)   # WRONG INDENT
  layer_logits = _unembed_mm(casted, ...)               # WRONG INDENT
  ```
- Impact:
  - Every control layer record is computed from the same final residual/logits.
  - `control_margin` timelines and first‑positive positions are invalid.
- Proposed fix:
  ```python
  for layer in range(n_layers):
      resid = residual_cache[f'blocks.{layer}.hook_resid_post']
      resid_raw_tensor = resid.detach().clone() if RAW_LENS_MODE != "off" else None
      if USE_NORM_LENS:
          norm_module = get_correct_norm_module(model, layer, probe_after_block=True)
          resid = apply_norm_or_skip(resid, norm_module)
      # These lines must be indented inside the loop
      casted = safe_cast_for_unembed(resid[0, :, :], analysis_W_U,
                                     force_fp32_unembed=(config.fp32_unembed or USE_FP32_UNEMBED))
      layer_logits = _unembed_mm(casted, analysis_W_U, analysis_b_U).float()
      # ...followed by emit_pure_next_token_record and optional Prism block
  ```
- Validation:
  - Control pure CSV shows changing logits across layers.
  - `control_summary.first_control_margin_pos` becomes non‑None and plausible.

Status: Implemented
- Change: Moved control pass per-layer unembed/logits computation and Prism sidecar emission inside the `for layer in range(n_layers)` loop (was wrongly dedented).
- File: 001_layers_baseline/run.py
- Lines: around 1426–1497 (casted/layer_logits, emit_pure_next_token_record, and Prism per-layer block now correctly indented inside the loop).
- Next: Validate on a small run that control-layer logits vary and `first_control_margin_pos` is populated when appropriate.

### Bug 2 — Rolling Window Contamination (lens + variants)
- Summary: The same rolling window list is mutated by both the normalized‑lens path and the Prism sidecar. Additionally, the control and ablation variants reuse the positive prompt’s window history.
- Evidence/Location: Updates to a shared `window_ids` occur in both main emit and Prism branches (e.g., ~420, ~720, ~870), and the control pass continues using `window_ids` from the positive pass.
- Impact:
  - Cross‑contamination causes false copy‑collapse detections.
  - Variant windows inherit unrelated history, skewing `L_copy`.
- Proposed fix (robust window management):
  - Introduce a tiny window manager with per‑(lens, prompt, variant) keys and bounded size:
    ```python
    class WindowManager:
        def __init__(self, window_k: int):
            self.window_k = window_k
            self.windows: dict[tuple[str, str, str], list[int]] = {}
        def append_and_trim(self, lens_type: str, prompt_id: str, variant: str, token_id: int) -> list[int]:
            key = (lens_type, prompt_id, variant)
            wl = self.windows.setdefault(key, [])
            wl.append(token_id)
            if len(wl) > self.window_k:
                wl.pop(0)
            return wl.copy()
        def reset_variant(self, prompt_id: str, variant: str):
            # reset both lens windows for this (prompt, variant)
            for lens in ("norm", "prism"):
                self.windows.pop((lens, prompt_id, variant), None)
    ```
  - Lifecycle rules:
    - Initialize/reset windows at the start of each variant pass (`pos/orig`, `pos/no_filler`, `ctl/orig`).
    - No cross‑variant or cross‑lens sharing.
    - Use a single `window_k` from config for both lenses (future: allow per‑lens tuning without changing behavior now).
  - Replace direct list mutations with window manager calls in both norm and Prism paths.
- Validation:
  - `L_copy` per lens/variant stable whether Prism is on/off.
  - Windows never exceed `window_k`; no cross‑variant pollution.

Status: Implemented
- Change: Introduced `WindowManager` (per-lens/per-variant rolling windows), replaced shared list mutations. Norm-lens `emit_pure_next_token_record` now uses the manager; all Prism branches updated to use separate windows. Windows are reset at the start of each prompt variant (`pos/orig`, `pos/no_filler`, `ctl/orig`).
- File: 001_layers_baseline/run.py
- Touch points: helper definition after prompt_id/variant; helper signature updated; call sites updated in orig/NF/CTL passes; Prism L0 and per-layer windows now isolated.
- Next: Validate `L_copy` stability with Prism toggled and confirm no cross-variant leakage.

## Phase 2: Should Fix

### Bug 3 — Prism Device Placement Safety

Status: Implemented (per pass) — one-time placement with guarded fallback; sidecar gated by local prism_enabled*. Placement errors recorded in diagnostics.
- Summary: `prism_Q.to(Xw.device)` is called inline across layers; can thrash memory or fail on constrained devices.
- Impact:
  - Repeated device transfers; potential OOM/runtime errors; perf loss.
- Proposed fix:
  - One‑time per‑pass placement with guarded fallback and cleanup:
    ```python
    try:
        target = Xw.device  # established once at L0
        if prism_Q.device != target:
            prism_Q = prism_Q.to(target)
        # reuse prism_Q for subsequent layers in this pass
    except RuntimeError as e:
        print(f"Prism device placement failed; disabling prism for this pass: {e}")
        prism_active = False
        # Diagnostics: record reason in prism_summary.error
    finally:
        # no per-layer .to(...); free only at end of pass if moved
        pass
    ```
  - Fallback policy: If placement fails, skip writing Prism sidecar rows for this prompt variant; main (non‑Prism) CSVs remain unaffected for data consistency.
  - Cleanup: If we moved tensors to a new device, rely on pass‑end cleanup and empty_cache/synchronize.
- Validation: No device mismatch; reduced transfers; stable memory footprint.

### Bug 4 — Fragile Hook Key Access

Status: Implemented — added get_residual_safely() and used in per-layer loops.
- Summary: Directly indexes `residual_cache[f'blocks.{layer}.hook_resid_post']` without validation.
- Proposed fix:
  - Add a tiny accessor with a clear error message and use it in all three passes:
    ```python
    def get_residual_safely(cache: dict[str, torch.Tensor], layer: int):
        key = f"blocks.{layer}.hook_resid_post"
        if key not in cache:
            candidates = [k for k in cache.keys() if f"blocks.{layer}" in k]
            raise KeyError(f"Missing '{key}'. Available near layer {layer}: {candidates}")
        return cache[key]
    ```
- Validation: Clear failure mode if hooks drift; easier debugging.

### Bug 5 — Over‑Broad Exception Handling (Systematic)

Status: Partially implemented — narrowed exceptions for control margin and gold alignment fallbacks; remaining generic handlers kept for outermost paths.
- Summary: Many `except Exception:` blocks swallow real errors across model loading, tokenization, metric computation, device transfers, and control margin.
- Proposed fix:
  - Sweep and narrow excepts with a helper for warn‑and‑default:
    ```python
    def warn_and_none(context: str, e: Exception):
        print(f"Warning in {context}: {e}")
        return None
    ```
  - Targeted areas:
    - Model loading fallback (initial load + move‑to‑device).
    - Tokenization/gold alignment fallbacks (L0 + control + ablation).
    - Metric computation (`compute_next_token_metrics`, control margin math).
    - Device transfers (Prism placement; unembed casts).
    - JSON writing (sanitize/signal unexpected types).
  - Pattern: catch expected `(IndexError, ValueError, TypeError, KeyError, RuntimeError)` as appropriate; warn and continue. For unexpected exceptions: print context and re‑raise.
- Validation: Legitimate mistakes surface; expected edge cases remain handled gracefully.

### Bug 6 — Temperature Exploration Memory Pressure

Status: Implemented — wrapped in no_grad, added explicit cleanup and CUDA cache/sync.
- Summary: The temperature probe allocates tensors in a loop without explicit no_grad/cleanup; on GPU this can retain intermediates.
- Proposed fix:
  - Wrap the block in `with torch.no_grad():`.
  - Explicitly `del` loop temporaries (`scaled_logits`, `temp_top_indices`, `temp_full_probs`, `temp_top_probs`).
  - After loop, call `torch.cuda.empty_cache(); torch.cuda.synchronize()` when CUDA is available.
- Validation: No progressive memory growth during the loop.

### Bug 7 — JSON Data Safety (NaN/Inf, Python types)

Status: Implemented (targeted) — applied finite coercion to last-layer consistency metrics.
- Summary: Tensor operations can produce NaN/Inf; JSON serialization expects finite Python types.
- Proposed fix:
  - Introduce helpers to coerce/sanitize floats:
    ```python
    def to_finite_float(x: float | int | torch.Tensor) -> float | None:
        v = float(x) if not isinstance(x, torch.Tensor) else float(x.item())
        if math.isfinite(v):
            return v
        print("Warning: non-finite value encountered; writing None")
        return None
    ```
  - Apply when building diagnostics summaries (not the large per‑token records which aren’t dumped into JSON meta).
- Validation: JSON dumps without errors; no NaN/Inf present.

### Bug 8 — Model Loading Fallback Robustness

Status: Implemented — degrade to CPU if move fails; update effective device.
- Summary: If direct device load fails, code loads on CPU and then attempts to move to the same device; this can trigger the same failure again without clear degradation.
- Proposed fix:
  - Fallback algorithm:
    1) Try direct load on requested device.
    2) If it fails: load on CPU.
    3) Try move to requested device; on failure, log a diagnostic (e.g., `degraded_to_cpu_after_move_fail=True`) and continue on CPU, updating the effective `device` used for diagnostics and downstream logic.
- Validation: Runs proceed on CPU with a clear log when device moves fail.

### Promoted to Phase 2 (from Phase 3)

Status: Unembed cache + cleanup ordering implemented; CLI globals deferred.
- Pre‑move analysis unembed weights: Create device‑local shadow `W_U`/`b_U` once per pass (fp32 if promoted) and reuse across layers.
- Cleanup ordering in hot paths: Prefer `torch.cuda.empty_cache(); torch.cuda.synchronize()` before `gc.collect()` for better allocator behavior.
- CLI global tightening: Reduce reliance on module‑level `CLI_ARGS` by threading `cli_args` to key functions (`run_single_model`, runner), to improve testability (minimal change, no behavior alteration).

## Phase 3: Nice to Fix (Future Refactor)

These are non‑blocking improvements to consider after Phase 1–2 land.

- Determinism env placement: Set `CUBLAS_WORKSPACE_CONFIG` before any CUDA checks/context creation.
- Duplicate return in self‑test fail path: Remove repeated `return {"error": "Self-test failed"}`.
- Residual save dtype: When `--keep-residuals`, prefer `resid.dtype` or explicit `torch.float32` instead of `model.cfg.dtype` (which may be a string).
- Unify `trust_remote_code` usage across load paths (or drop if not used by TransformerLens), for parity.
- Argmax conversions: Optional guard against NaN/Inf; use direct `.item()` from `argmax_tensor` with a finite check.
- Configurability: Move `entropy_collapse_threshold` from magic `1.0` to `ExperimentConfig`.

## Validation, Rollback, and Testing Strategy

- Functional checks:
  - Control pass: layerwise logits vary; `first_control_margin_pos` non‑None when appropriate.
  - Copy detection: `L_copy` unchanged when toggling Prism on/off (no lens contamination).
- Integrity checks:
  - Finite‑value assertions before JSON dump; warn on sanitization.
  - Residual key accessor raises meaningful error on missing keys.
- Performance checks:
  - One‑time device placement for Prism; no repeated `.to(...)` per layer.
  - Memory usage stable during temperature exploration; improved cleanup ordering verified on constrained devices.
- Rollback/Monitoring:
  - Each fix is localized and revertible. Add diagnostics fields (e.g., `diagnostics.degraded_to_cpu_after_move_fail`, Prism `error` reason) to detect fallback behavior.
  - Keep a small before/after sweep to diff CSV schemas (unchanged) and metrics (only control/copy expected to change due to bug fixes).
- Tests:
  - Unit: WindowManager append/trim; safe residual accessor failure path.
  - Integration: Small model run exercising orig/no_filler/control passes; assert sane control summary and copy flags.

Implementation will start with Phase 1 (indentation + window manager). Then Phase 2 (Prism placement guard, safe residual accessor, exception‑handling sweep, temp exploration cleanup, JSON safety, model fallback, promoted perf items). Phase 3 remains as follow‑ups during the next maintenance window.
