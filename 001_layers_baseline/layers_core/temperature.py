"""Norm-lens temperature calibration helpers (001_LAYERS_BASELINE_PLAN §1.16).

Runs entirely on CPU to avoid device-specific op gaps (e.g., MPS logspace).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import math
import torch

from .hooks import build_cache_hook, attach_residual_hooks, detach_hooks, get_residual_safely


@torch.no_grad()
def _collect_layer_logits(model, prompt: str, norm_lens, unembed_ctx) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Return per-layer logits at last position and final probs for a prompt."""
    tokens = model.to_tokens(prompt)
    logits = model(tokens)
    final_logits = logits[0, -1, :].float()
    final_probs = torch.softmax(final_logits, dim=0)

    cache: Dict[str, torch.Tensor] = {}
    hook = build_cache_hook(cache)
    handles, has_pos = attach_residual_hooks(model, hook)
    try:
        _ = model(tokens)
    finally:
        detach_hooks(handles)

    last_pos = tokens.shape[1] - 1
    n_layers = model.cfg.n_layers
    out: List[torch.Tensor] = []

    # L0
    resid_raw = cache['hook_embed'] + (cache.get('hook_pos_embed', 0) if has_pos else 0)
    logits_l0 = norm_lens.forward(
        model,
        0,
        resid_raw,
        probe_after_block=False,
        W_U=unembed_ctx.W,
        b_U=unembed_ctx.b,
        force_fp32_unembed=unembed_ctx.force_fp32,
        cache=unembed_ctx.cache,
    )
    out.append(logits_l0[last_pos].detach().float())

    # post-block layers
    for layer in range(n_layers):
        resid = get_residual_safely(cache, layer)
        logits_all = norm_lens.forward(
            model,
            layer,
            resid,
            probe_after_block=True,
            W_U=unembed_ctx.W,
            b_U=unembed_ctx.b,
            force_fp32_unembed=unembed_ctx.force_fp32,
            cache=unembed_ctx.cache,
        )
        out.append(logits_all[last_pos].detach().float())

    cache.clear()
    return out, final_probs


def fit_norm_temperatures(
    model,
    prompts: Sequence[str],
    norm_lens,
    unembed_ctx,
    *,
    tau_min: float = 0.2,
    tau_max: float = 5.0,
    grid_points: int = 25,
) -> List[float]:
    """Fit per-layer scalar τ to minimize CE(P(z/τ), P_final); returns τ vector."""
    cpu = torch.device('cpu')
    per_layer_logits: Dict[int, List[torch.Tensor]] = {}
    per_layer_targets: Dict[int, List[torch.Tensor]] = {}

    for p in prompts:
        logits_list, final_probs = _collect_layer_logits(model, p, norm_lens, unembed_ctx)
        for idx, vec in enumerate(logits_list):
            per_layer_logits.setdefault(idx, []).append(vec.to(cpu, dtype=torch.float32))
            per_layer_targets.setdefault(idx, []).append(final_probs.to(cpu, dtype=torch.float32))

    n = model.cfg.n_layers + 1  # include L0
    taus = [1.0 for _ in range(n)]
    grid = torch.logspace(math.log10(tau_min), math.log10(tau_max), steps=max(5, grid_points), device=cpu)

    for idx in range(n):
        Ls = per_layer_logits.get(idx)
        Ts = per_layer_targets.get(idx)
        if not Ls or not Ts:
            taus[idx] = 1.0
            continue
        best_tau = 1.0
        best = float('inf')
        for tau in grid:
            tv = float(tau.item())
            loss = 0.0
            for z, t in zip(Ls, Ts):
                P = torch.softmax(z / tv, dim=0)
                # CE(t || P)
                ce = -(t * (P + 1e-30).log()).sum().item()
                loss += ce
            loss /= len(Ls)
            if loss < best:
                best = loss
                best_tau = tv
        taus[idx] = best_tau

    return taus

