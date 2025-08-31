#!/usr/bin/env python3
"""
Prism fitter: builds shared decoder artifacts per model.

Defaults are designed so you can just run:
  cd 001_layers_baseline && python prism_fit.py

This fits artifacts for the shared model list using an auto-selected device,
captures residuals at a few depths, computes whitening + an orthogonal map Q,
and saves artifacts under prisms/<clean_model_name>/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch

from transformer_lens import HookedTransformer

from layers_core.device_policy import select_best_device, choose_dtype
from layers_core.hooks import attach_residual_hooks, detach_hooks, build_cache_hook
from layers_core.prism import (
    RunningMoments,
    whiten_apply,
    compute_reservoir_basis,
    fit_prism_Q,
    save_prism_artifacts,
    orthogonality_error,
)
from models import CANDIDATE_MODELS


# Deterministic bootstrap (aligns with run.py)
SEED = 316
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def clean_model_name(model_id: str) -> str:
    return model_id.split("/")[-1]


def default_depth_indices(n_layers: int) -> List[int]:
    """Return post-block layer indices to sample: ≈25%, 50%, 75% (0-based)."""
    picks_1b = {n_layers // 4, n_layers // 2, (3 * n_layers) // 4}  # 1-based notion
    # Convert to 0-based block indices (exclude 0 values)
    idxs = sorted({p - 1 for p in picks_1b if 1 <= p <= n_layers})
    # Ensure unique and within range
    return [i for i in idxs if 0 <= i < n_layers]


def _iter_default_prompts() -> Iterable[str]:
    """Small built-in prompt set; cycles as needed."""
    prompts = [
        "The capital of Germany is",
        "The capital of France is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The largest ocean is",
        "2 + 2 =",
        "Complete the sentence: The sky is",
        "Name a prime number after 10:",
        "Compute 3 * 7 =",
        "Python is a",
    ]
    i = 0
    while True:
        yield prompts[i % len(prompts)]
        i += 1


def _iter_prompts_from_file(path: Path) -> Iterable[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return _iter_default_prompts()
    i = 0
    while True:
        yield lines[i % len(lines)]
        i += 1


def fit_single_model(
    model_id: str,
    *,
    device: str,
    dtype: torch.dtype,
    tokens_budget: int,
    reservoir_cap: int,
    rank_k: Optional[int],
    prism_root: Path,
    prompts_iter: Iterable[str],
) -> Tuple[bool, Optional[str]]:
    """Fit Prism artifacts for one model. Returns (success, message)."""

    print(f"\n{'='*80}\nPRISM FIT: {model_id}\n{'='*80}")
    # Load model
    os.environ['TRANSFORMERS_BATCH_FIRST'] = 'False'
    try:
        print(f"Loading {model_id} on {device} (dtype={dtype}) …")
        model = HookedTransformer.from_pretrained_no_processing(
            model_id,
            device=device,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False, f"load_failed: {e}"

    model.eval()

    # Model dims
    d_model = int(model.cfg.d_model)
    n_layers = int(model.cfg.n_layers)
    W_U = model.unembed.W_U  # (d×vocab)
    d_vocab = int(W_U.shape[1])

    # Rank k default
    k = int(rank_k) if rank_k is not None else min(512, d_model)
    layers_idx = default_depth_indices(n_layers)

    # Running moments and reservoir
    moments = RunningMoments(dim=d_model, eps=1e-8)
    reservoir: List[torch.Tensor] = []  # list of (d,) float32 CPU tensors
    total_seen = 0

    # Capture loop
    with torch.no_grad():
        prompts_used = 0
        while total_seen < tokens_budget:
            prompt = next(prompts_iter)
            prompts_used += 1

            # Hook cache
            residual_cache = {}
            cache_hook = build_cache_hook(residual_cache)
            hooks, has_pos_embed = attach_residual_hooks(model, cache_hook)
            try:
                tokens = model.to_tokens(prompt)
                _ = model(tokens)
                # Build batch rows: layer 0 and selected post-block indices
                rows: List[torch.Tensor] = []
                # Layer 0
                resid0 = residual_cache['hook_embed']
                if has_pos_embed and 'hook_pos_embed' in residual_cache:
                    resid0 = resid0 + residual_cache['hook_pos_embed']
                rows.append(resid0[0, -1, :].detach().float().cpu())
                # Post-blocks
                for li in layers_idx:
                    resid = residual_cache[f'blocks.{li}.hook_resid_post']
                    rows.append(resid[0, -1, :].detach().float().cpu())
                # Update moments
                batch = torch.stack(rows, dim=0)  # (L_sel, d)
                moments.update(batch)
                # Reservoir sampling per row
                for row in rows:
                    total_seen += 1
                    if len(reservoir) < reservoir_cap:
                        reservoir.append(row)
                    else:
                        j = random.randint(0, total_seen - 1)
                        if j < reservoir_cap:
                            reservoir[j] = row
            finally:
                detach_hooks(hooks)
                residual_cache.clear()
                hooks.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if prompts_used % 50 == 0:
                print(f"  … progress: {total_seen}/{tokens_budget} samples; reservoir={len(reservoir)}")

    print(f"Collected {total_seen} samples across {prompts_used} prompts; reservoir={len(reservoir)}")

    # Finalize whitening stats and compute basis
    stats = moments.finalize()
    X = torch.stack(reservoir, dim=0)  # (N_keep, d)
    Xw = whiten_apply(X, stats)
    print(f"Computing reservoir basis (N={Xw.shape[0]}, d={Xw.shape[1]}, k={k}) …")
    E_k = compute_reservoir_basis(Xw, k=k)

    # Fit Q via Procrustes against W_U
    print("Fitting orthogonal Q via Procrustes …")
    Q = fit_prism_Q(W_U, E_k)
    ortho_err = orthogonality_error(Q)
    print(f"  Orthogonality error ||QᵀQ−I||_F ≈ {ortho_err:.3e}")

    # Save artifacts
    out_dir = prism_root / clean_model_name(model_id)
    prov = {
        "method": "procrustes",
        "k": k,
        "layers": ["embed"] + layers_idx,
        "tokens_seen": total_seen,
        "reservoir": len(reservoir),
        "seed": SEED,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "dtype": str(dtype),
        "d_model": d_model,
        "d_vocab": d_vocab,
        "n_layers": n_layers,
        "orthogonality_error": ortho_err,
        "notes": "whiten=diag(mean/var), U_k from W_U W_Uᵀ, QR-polished",
    }
    w_path, q_path, p_path = save_prism_artifacts(out_dir, stats=stats, Q=Q, provenance=prov)
    print(f"Saved prism artifacts to: {out_dir}\n - {w_path.name}\n - {q_path.name}\n - {p_path.name}")

    return True, None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit Logit Prism artifacts (shared decoder) for baseline models")
    p.add_argument("model_id", nargs="?", help="Optional single MODEL_ID to fit; default fits all baseline models")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"], help="Compute device (default auto)")
    p.add_argument("--tokens", type=int, default=200_000, help="Total sample budget (number of residual vectors)")
    p.add_argument("--reservoir", type=int, default=50_000, help="Reservoir sample cap for basis (rows)")
    p.add_argument("--k", type=int, default=None, help="Rank k for subspace alignment (default min(512, d_model))")
    p.add_argument("--prism-dir", default="prisms", help="Artifacts root (relative to this script dir)")
    p.add_argument("--prompts-file", type=str, default=None, help="Optional path to a prompts file (one prompt per line)")
    return p.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).parent
    prism_root = script_dir / args.prism_dir

    def run_for(model_id: str) -> Tuple[bool, Optional[str]]:
        # Device selection
        if args.device == "auto":
            sel = select_best_device(model_id)
            if sel is None:
                print(f"⛔ Skipping {model_id}: no device fits (estimates)")
                return False, "no_fit"
            dev, dtype, debug = sel
        else:
            dev = args.device
            dtype = choose_dtype(dev, model_id)
        # Prompts iterator
        if args.prompts_file:
            prompts_iter = _iter_prompts_from_file(Path(args.prompts_file))
        else:
            prompts_iter = _iter_default_prompts()
        return fit_single_model(
            model_id,
            device=dev,
            dtype=dtype,
            tokens_budget=int(args.tokens),
            reservoir_cap=int(args.reservoir),
            rank_k=(int(args.k) if args.k is not None else None),
            prism_root=prism_root,
            prompts_iter=prompts_iter,
        )

    if args.model_id:
        ok, msg = run_for(args.model_id)
        raise SystemExit(0 if ok else 1)

    # All baseline models
    results: List[Tuple[str, str]] = []
    for model_id in CANDIDATE_MODELS:
        ok, msg = run_for(model_id)
        results.append((model_id, "OK" if ok else (msg or "FAILED")))

    print("\nSummary:")
    for model_id, status in results:
        print(f" - {clean_model_name(model_id)}: {status}")


if __name__ == "__main__":
    main()

