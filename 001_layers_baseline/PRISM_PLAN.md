# 1.10 · Logit Prism (Shared Decoder) — Implementation Plan

This document is the working plan for adding a “Logit Prism” shared decoder to the 001_layers_baseline experiment. It is intended to guide implementation and provide persistent context across sessions.

## Summary
- Goal: add a calibration decoder that projects whitened residuals through a shared, structure‑aware mapping aligned with the model’s unembedding. This provides a more stable, architecture‑agnostic readout early/mid‑depth without changing the baseline outputs.
- Approach: separate “fit” script to build Prism artifacts per model; optional sidecar decode in `run.py` that loads artifacts and writes CSV sidecars.
- Defaults first: good defaults for layers, token budget, and rank; minimal flags; same model list as baseline.

## Status Update (2025‑08‑31)
- Completed:
  - Core utilities in `layers_core/prism.py` (running moments; whitening; reservoir basis; Procrustes fit; artifact I/O; orthogonality check).
  - Shared model registry `001_layers_baseline/models.py` exposing `CANDIDATE_MODELS` (imported by `run.py`).
  - `run.py` integration with prism=auto|on|off, `--prism-dir`, sidecar CSV emission, and JSON `prism_summary`.
  - CPU‑only tests for whitening/Procrustes/artifacts and sidecar smoke; wired into `scripts/run_cpu_tests.sh` and passing locally.
- Pending:
  - None for 1.10 core; next optional step is to fit artifacts for selected models and evaluate Prism metrics.
- Notes (implementation differences from initial sketch):
  - Running moments provided as a `RunningMoments` class (batched Welford) rather than a single function.
  - Left singular subspace derived via eigh on `W_U W_Uᵀ` (float32) with QR polish; equivalent to SVD for our purposes.
  - Added explicit helpers: `whiten_apply`, `orthogonality_error`, and `_left_singular_vectors_WU` for clarity.

## Scope and Non‑Goals
- In scope:
  - A separate fitter script: `001_layers_baseline/prism_fit.py`.
  - Core numerics in `layers_core/prism.py` (whitening, Procrustes, artifact I/O).
  - Optional sidecar decode path in `run.py` producing `*-records-prism.csv` and `*-pure-next-token-prism.csv` (schemas identical to baseline CSVs).
  - Shared model registry to keep the fit/run model lists aligned.
- Out of scope (for 1.10):
  - Tuned Lens (learned per‑layer heads).
  - Changing baseline outputs or schemas.
  - Networked corpora or large dataset ingestion. Default prompts will be local/minimal.

## Design Overview
1) Data capture (fit phase)
   - Collect residuals at a small set of depths (default: 0, ≈25%, ≈50%, ≈75%).
   - Use pure next‑token residual vectors (last position) across a modest prompt set until a token budget is met.
   - Track per‑feature mean/var in fp32 for whitening (running moments; no full matrix).
   - Keep a thin reservoir sample of whitened residuals for subspace alignment (optional, bounded memory).

2) Decoder fit
   - Compute SVD of the model’s unembedding `W_U = U Σ Vᵀ` (fp32). Take `U_k` with k ≤ d_model (default k = min(512, d_model)).
   - Compute an orthogonal map `Q ∈ R^{d×d}` aligning the whitened residual subspace to `U_k` via Orthogonal Procrustes:
     - Let `E_k` be an orthonormal basis of the whitened residual sample (PCA or QR); solve `min_Q ||Q E_k − U_k||_F` → `Q = U Vᵀ` from `svd(U_k E_kᵀ)`.
   - Prism decode at runtime: `logits_prism = (whiten(resid) @ Q) @ W_U + b_U` (no change to `W_U`).
   - Fallback (simple mode): diagonal‑whiten + identity `Q` (sanity path if fit fails); optional ridge decoder if Procrustes is numerically unstable.

3) Artifacts per model (on disk)
   - Location: `001_layers_baseline/prisms/<clean_model_name>/`.
   - `whiten.pt`: `{ mean: (d,), var: (d,), eps: float }` (fp32).
   - `Q_prism.pt`: `(d, d)` fp32 orthogonal map.
   - `provenance.json`: `{ method: "procrustes", k, layers, tokens_seen, seed, date, code_sha, d_model, d_vocab, notes }`.
   - Note: We avoid storing full `W_prism (d×vocab)` to keep size reasonable; we always reuse the model’s `W_U` at decode time.

4) Integration in 001 (sidecar only)
   - Flags (minimal): `--use-prism` to emit `*-prism.csv` alongside baseline CSVs; `--prism-dir` to override artifact path.
   - No changes to baseline outputs without `--use-prism`.
   - JSON adds a small `prism_summary` block (fit present, k, layers sampled).

## CLI and Defaults
### prism_fit.py (per‑model fitter)
- Usage (defaults cover most cases):
  - `python prism_fit.py` → fits all baseline models on `device=auto` with defaults below.
  - `python prism_fit.py <MODEL_ID>` → fits a single model.
- Defaults:
  - Layers to sample: `0, ⌊n/4⌋, ⌊n/2⌋, ⌊3n/4⌋`.
  - Token budget: `tokens=200_000` (pure next‑token residuals; repeats local prompts as needed).
  - Rank k: `k = min(512, d_model)`.
  - Device/dtype: same policy as baseline (`device_policy.choose_dtype`, `select_best_device`).
  - Seed/determinism: reuse `SEED=316` and deterministic algorithms.

### run.py (sidecar decode)
- Default policy: prism=auto
  - If compatible artifacts are found → emit `*-records-prism.csv` and `*-pure-next-token-prism.csv` sidecars.
  - If missing or incompatible → print one-line notice and proceed baseline-only.
- Flags/env:
  - `--prism {auto|on|off}` (default `auto`).
    - `on`: require artifacts; error if missing/incompatible.
    - `off`: disable sidecar entirely.
    - `auto`: opportunistic sidecar (recommended default).
  - `--prism-dir PATH` (optional): override artifact search path.
  - Env alias: `LOGOS_PRISM=auto|on|off` (CLI takes precedence).

## Shared Model Registry
- Create `001_layers_baseline/models.py` (or `layers_core/models.py`) exposing `CANDIDATE_MODELS` used by both `run.py` and `prism_fit.py`.
- Replace local lists in both with imports from this module.

## Fitting Numerics (details)
- Whitening: per‑feature `(x − μ) / sqrt(var + eps)` with `eps=1e-8` in fp32.
- Residual basis `E_k`: compute via PCA on the reservoir sample or via incremental QR. Start with PCA on a sampled matrix (N×d, N≲50k).
- Procrustes: `svd(U_k E_kᵀ) → U Σ Vᵀ`, `Q = U Vᵀ` (orthogonal). Store `Q` only.
- Stability: clamp tiny variances; ensure orthogonality via re‑SVD on `Q` if needed (Q ← U Vᵀ of `svd(Q)`).

## Artifacts and Provenance
- `prisms/<model>/whiten.pt` (torch.save), `prisms/<model>/Q_prism.pt`, `provenance.json`.
- Provenance fields: model_id, d_model, d_vocab, layers, tokens_seen, k, method, seed, date, commit, device, dtype.
- Size: two `(d,)` vectors + a `(d,d)` matrix; for d≈4096 that’s ~64 MB in fp32 — acceptable for a small set of models. If needed, add `--q-fp16` to compress.

## Sidecar CSVs
- Filenames: `output-<model>-records-prism.csv`, `output-<model>-pure-next-token-prism.csv`.
- Schemas identical to baseline CSVs (no extra columns); consumers can infer from filename.
- JSON `diagnostics` gains `prism_summary` with `{ present: bool, k, layers }`.

## Testing Plan
1) Unit tests in `001_layers_baseline/tests/`:
   - `test_prism_whiten.py`: whitening/unwhitening round‑trip on synthetic data.
   - `test_prism_procrustes.py`: Procrustes recovers known rotation (synthetic).
   - `test_prism_artifacts.py`: save/load round‑trip; shape/dtype checks.
   - `test_prism_sidecar.py`: mock model path—ensure sidecar CSVs are written with `--use-prism`.
2) Orchestrator smoke: extend existing smoke test to simulate Prism decode via mock artifacts.
3) Numerical sanity (optional): KL(P_prism || P_final) at sampled depths is ≤ KL(baseline || P_final) on a tiny synthetic model.

## Implementation Steps (checklist)
- [x] Create `layers_core/prism.py` with:
  - [x] Running moments (batched Welford) via `RunningMoments` class.
  - [x] Diagonal whitening apply via `whiten_apply`.
  - [x] `compute_reservoir_basis` (PCA/QR) for E_k.
  - [x] `_left_singular_vectors_WU` (eigh on `W_U W_Uᵀ`) and `fit_prism_Q(W_U, E_k)` (Procrustes with orthogonality polish).
  - [x] `save_prism_artifacts(...)` / `load_prism_artifacts(...)`, plus `orthogonality_error`.
- [x] Add `001_layers_baseline/models.py` exporting `CANDIDATE_MODELS`.
- [x] Implement `prism_fit.py`:
  - [x] CLI with good defaults; device auto‑select; shared model registry.
  - [x] Residual capture at default depths; running moments + reservoir sample.
  - [x] Basis extraction, Procrustes fit, artifact save.
- [x] Integrate in `run.py` (sidecar decode):
  - [x] Flags: `--prism {auto|on|off}` (default auto), `--prism-dir` (optional); env `LOGOS_PRISM`.
  - [x] Auto behavior: load artifacts if present/compatible; otherwise warn-once and skip (prints single-line notice).
  - [x] Compute prism logits per layer and prompts; write sidecar CSVs (`-records-prism.csv`, `-pure-next-token-prism.csv`).
  - [x] Add `prism_summary` in JSON diagnostics (present, compatible, k, layers, artifact_path, error).
- [x] Tests (CPU‑only) as above; hook into `scripts/run_cpu_tests.sh`.

## Validation Metrics and Heuristics
- Report at fit time: dimensions, k, layers sampled, tokens_seen, spectral gap of `W_U` around k, orthogonality error `||QᵀQ − I||_F`.
- Optional quick check: on a handful of prompts, `KL(P_prism || P_final)` ≤ `KL(P_norm || P_final)` at early/mid layers; warn otherwise.

## Performance and Memory
- Running moments are O(d). Reservoir size `N_keep` default 50k rows (d×N_keep matrix in fp32); if memory tight, reduce to 10k.
- SVD on `W_U` is the dominant cost but done once per model; you can truncate via randomized SVD.
- Decode overhead at run time: one extra `(d×d)` and `(d×vocab)` matmul per layer’s last position — negligible vs baseline.

## Failure Modes and Mitigations
- Unstable Procrustes due to poor basis: fall back to identity `Q` (warn) or to ridge decoder; persist provenance.
- Artifact mismatch (d_model differs): refuse to load and print a clear error.
- No artifacts found with `--use-prism`: continue baseline, log “prism disabled (no artifacts)”.

## Open Questions / Future Work
- Switch from diagonal‑whiten to full‑cov whitening when feasible (store Cholesky instead of var).
- Compare Procrustes vs ridge in KL metrics; optionally support both via `method` in provenance.
- Compress `Q` to fp16 to reduce disk footprint with guardrails.
- Later 002: Tuned Lens integration; initialize from Prism for stability.

---
Owner: (this assistant)
Seed: 316 (match baseline)
Status: 1.10 implemented end‑to‑end (core utils, fitter, sidecar integration, tests). Ready to generate artifacts and analyze Prism vs baseline if desired.
