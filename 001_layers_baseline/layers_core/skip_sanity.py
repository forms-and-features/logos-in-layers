"""Skip-layer sanity diagnostics for tuned lens (PROJECT_NOTES §1.17)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import math
import torch

from .hooks import build_cache_hook, attach_residual_hooks, detach_hooks, get_residual_safely


@torch.no_grad()
def evaluate_skip_layers(
    model,
    tuned_adapter,
    unembed_ctx,
    prompts: Sequence[Tuple[str, Optional[int]]],
    m_values: Iterable[int] = (2, 4, 8),
) -> Dict[str, Optional[float]]:
    """Compare ppl delta when replacing last m blocks by tuned logits at L−m.

    Returns mapping like {"m=2": delta, ...}; delta is relative increase.
    """
    if tuned_adapter is None:
        return {}

    n_layers = model.cfg.n_layers
    results: Dict[int, List[Tuple[float, float]]] = {}

    for prompt, ans_id in prompts:
        if ans_id is None:
            continue
        tokens = model.to_tokens(prompt)
        logits = model(tokens)
        final_probs = torch.softmax(logits[0, -1, :].float(), dim=0)
        base_loss = -math.log(float(final_probs[int(ans_id)].item()) + 1e-12)

        cache: Dict[str, torch.Tensor] = {}
        hook = build_cache_hook(cache)
        handles, _ = attach_residual_hooks(model, hook)
        try:
            _ = model(tokens)
        finally:
            detach_hooks(handles)

        last_pos = tokens.shape[1] - 1
        for m in m_values:
            if m <= 0 or m > n_layers:
                continue
            lidx = n_layers - m
            try:
                resid = get_residual_safely(cache, lidx)
            except KeyError:
                continue
            tl_logits_all = tuned_adapter.forward(
                model,
                lidx,
                resid,
                probe_after_block=True,
                W_U=unembed_ctx.W,
                b_U=unembed_ctx.b,
                force_fp32_unembed=unembed_ctx.force_fp32,
                cache=unembed_ctx.cache,
            )
            if tl_logits_all is None:
                continue
            P = torch.softmax(tl_logits_all[last_pos].float(), dim=0)
            tuned_loss = -math.log(float(P[int(ans_id)].item()) + 1e-12)
            results.setdefault(m, []).append((base_loss, tuned_loss))

        cache.clear()

    out: Dict[str, Optional[float]] = {}
    for m, pairs in results.items():
        key = f"m={m}"
        if not pairs:
            out[key] = None
            continue
        base = sum(b for b, _ in pairs) / len(pairs)
        tuned = sum(t for _, t in pairs) / len(pairs)
        base_ppl = math.exp(base)
        tuned_ppl = math.exp(tuned)
        out[key] = float((tuned_ppl - base_ppl) / base_ppl)
    return out

