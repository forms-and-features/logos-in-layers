from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .numerics import kl_bits
from .metrics import compute_next_token_metrics


def _finite(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        try:
            v = float(x.item())  # torch scalar
        except Exception:
            return None
    if not torch.isfinite(torch.tensor(v)):
        return None
    return v


def compute_last_layer_consistency(
    *,
    last_logits: torch.Tensor,
    final_probs: torch.Tensor,
    final_top1_id: int,
    first_ans_id: Optional[int],
    head_scale_cfg: Optional[float],
    head_softcap_cfg: Optional[float],
    topk_cum: int = 5,
) -> Dict[str, Any]:
    """Compute diagnostics comparing the last lens layer to the final head.

    Mirrors the inline logic previously in run.py:
    - Probability metrics vs final head
    - Scalar temperature probe (grid search) that minimizes KL(P(z/s)||final)
    - Family-reported head transforms (scale, softcap) and KL after applying them
    Returns a dictionary with identical keys/semantics.
    """
    zs = last_logits.float()
    last_full_probs = torch.softmax(zs, dim=0)
    # Preserve original behavior: derive top-1 from raw logits via topk(1)
    _, _idx1 = torch.topk(zs, 1, largest=True, sorted=True)
    lens_top1_id = int(_idx1[0].item())

    # Metrics relative to final head
    m = compute_next_token_metrics(
        last_full_probs, lens_top1_id, final_probs, first_ans_id, topk_cum=topk_cum
    )

    # Temperature probe s ∈ [0.1, 10] (log-space) for minimal KL(P(z/s)||final)
    best_s: Optional[float] = None
    best_kl: Optional[float] = None
    try:
        s_values = torch.logspace(-1, 1, steps=25, dtype=torch.float32).tolist()
        best_s_v = float("nan")
        best_kl_v = float("inf")
        for s in s_values:
            s_f = float(s)
            P = torch.softmax(zs / s_f, dim=0)
            kl = float(kl_bits(P, final_probs))
            if kl < best_kl_v:
                best_kl_v = kl
                best_s_v = s_f
        best_s = best_s_v
        best_kl = best_kl_v
    except Exception:
        best_s = None
        best_kl = None

    # KL after simple family-specific transforms (last layer only)
    kl_after_scale = None
    kl_after_softcap = None
    kl_after_scale_then_softcap = None
    cfg_transform = {"scale": head_scale_cfg, "softcap": head_softcap_cfg}

    try:
        if head_scale_cfg is not None and head_scale_cfg > 0:
            P = torch.softmax(zs / float(head_scale_cfg), dim=0)
            kl = float(kl_bits(P, final_probs))
            kl_after_scale = kl
        if head_softcap_cfg is not None and head_softcap_cfg > 0:
            c = float(head_softcap_cfg)
            zs_c = torch.tanh(zs / c) * c
            P = torch.softmax(zs_c, dim=0)
            kl_after_softcap = float(kl_bits(P, final_probs))
        if (
            head_scale_cfg is not None
            and head_scale_cfg > 0
            and head_softcap_cfg is not None
            and head_softcap_cfg > 0
        ):
            c = float(head_softcap_cfg)
            s = float(head_scale_cfg)
            zs_sc = torch.tanh((zs / s) / c) * c
            P = torch.softmax(zs_sc, dim=0)
            kl_after_scale_then_softcap = float(kl_bits(P, final_probs))
    except Exception:
        pass

    kl_to_final_bits = _finite(m.get("kl_to_final_bits"))
    kl_after_temp_bits = _finite(best_kl)
    threshold_bits = 0.25
    delta_temp_bits = None
    if kl_to_final_bits is not None and kl_after_temp_bits is not None:
        delta_temp_bits = float(kl_to_final_bits - kl_after_temp_bits)

    warn_gate = False
    if kl_to_final_bits is not None and kl_to_final_bits >= threshold_bits:
        warn_gate = True
    if delta_temp_bits is not None and delta_temp_bits >= threshold_bits:
        warn_gate = True

    out = {
        "kl_to_final_bits": kl_to_final_bits,
        "top1_agree": bool(lens_top1_id == int(final_top1_id)),
        "p_top1_lens": _finite(m.get("p_top1")),
        "p_top1_model": _finite(final_probs[int(final_top1_id)].item()),
        "p_answer_lens": _finite(m.get("p_answer")),
        "answer_rank_lens": m.get("answer_rank"),
        # Temperature probe: best scalar s and KL after rescale
        "temp_est": _finite(best_s),
        "kl_after_temp_bits": kl_after_temp_bits,
        # Config-reported head transforms and KL after applying them
        "cfg_transform": cfg_transform,
        "kl_after_transform_bits": {
            "scale": _finite(kl_after_scale),
            "softcap": _finite(kl_after_softcap),
            "scale_then_softcap": _finite(kl_after_scale_then_softcap),
        },
        # Advisory warning for family-agnostic visibility
        "warn_high_last_layer_kl": bool(warn_gate),
        "gates": {
            "warn_high_last_layer_kl": bool(warn_gate),
            "threshold_bits": threshold_bits,
        },
    }
    if delta_temp_bits is not None:
        out["gates"]["delta_kl_temp_bits"] = delta_temp_bits
    return out
