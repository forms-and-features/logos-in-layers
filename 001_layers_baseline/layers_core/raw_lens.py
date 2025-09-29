import os
from typing import Dict, Any, Optional, Iterable, List

import torch

from .numerics import safe_cast_for_unembed, kl_bits
from .metrics import compute_next_token_metrics
from .collapse_rules import is_semantic_top1
from .unembed import unembed_mm


def get_raw_lens_mode(self_test: bool) -> str:
    """Resolve raw-lens mode from env, defaulting to 'sample'.

    - Accepts: off | sample | full (case-insensitive)
    - In self-test, upgrade 'sample' to 'full' to exercise the path thoroughly.
    """
    mode = os.environ.get("LOGOS_RAW_LENS", "sample").strip().lower()
    if mode not in ("off", "sample", "full"):
        mode = "sample"
    if self_test and mode == "sample":
        mode = "full"
    return mode


def init_raw_lens_check(mode: str) -> Dict[str, Any]:
    return {"mode": mode, "samples": [], "summary": None}


def should_sample_layer(mode: str, n_layers: int, layer_one_indexed: int) -> bool:
    """Return True iff this post-block layer should be sampled for raw-vs-norm.

    - layer_one_indexed: 1..n
    - In 'sample', choose ~25%, 50%, 75% depth (rounded via integer division).
    """
    if mode == "off":
        return False
    if mode == "full":
        return 1 <= layer_one_indexed <= n_layers
    # sample
    candidates = {n_layers // 4, n_layers // 2, (3 * n_layers) // 4}
    # convert 0-based to 1-based
    candidate_one_indexed = {c + 1 for c in candidates if 0 <= c < n_layers}
    return layer_one_indexed in candidate_one_indexed


def record_dual_lens_sample(
    out_block: Dict[str, Any],
    *,
    layer_out_idx: int,
    last_logits_norm: torch.Tensor,
    resid_raw_last_vec: torch.Tensor,
    W_U: torch.Tensor,
    b_U: Optional[torch.Tensor],
    force_fp32_unembed: bool,
    tokenizer,
    final_probs: torch.Tensor,
    first_ans_id: Optional[int],
    ground_truth: str,
) -> None:
    """Compute and append a raw-vs-norm sample for the last position.

    Appends a dict with KL, agreement, p_top1, p_answer, and answer_rank for both lenses.
    Ignores any internal errors (best-effort QA signal).
    """
    try:
        # Raw logits from pre-norm residual
        raw_last_vec_cast = safe_cast_for_unembed(
            resid_raw_last_vec, W_U, force_fp32_unembed=force_fp32_unembed
        )
        logits_raw_last = (raw_last_vec_cast @ W_U)
        if b_U is not None:
            logits_raw_last = logits_raw_last + b_U
        logits_raw_last = logits_raw_last.float()

        P_norm = torch.softmax(last_logits_norm.float(), dim=0)
        P_raw = torch.softmax(logits_raw_last, dim=0)

        # Top-1 ids
        top1_norm = int(torch.argmax(P_norm).item())
        top1_raw = int(torch.argmax(P_raw).item())

        # Metrics vs final head
        m_norm = compute_next_token_metrics(P_norm, top1_norm, final_probs, first_ans_id, topk_cum=5)
        m_raw = compute_next_token_metrics(P_raw, top1_raw, final_probs, first_ans_id, topk_cum=5)

        # KL between lenses (bits)
        kl_nr_bits = kl_bits(P_norm, P_raw)

        # is_answer flags (fallback to string equality if id unknown)
        if m_norm.get("answer_rank") is not None:
            is_ans_norm = (m_norm["answer_rank"] == 1)
            is_ans_raw = (m_raw["answer_rank"] == 1)
        else:
            tok_norm = tokenizer.decode([top1_norm])
            tok_raw = tokenizer.decode([top1_raw])
            is_ans_norm = is_semantic_top1(tok_norm, ground_truth)
            is_ans_raw = is_semantic_top1(tok_raw, ground_truth)

        out_block["samples"].append({
            "layer": layer_out_idx,
            "kl_norm_vs_raw_bits": kl_nr_bits,
            "top1_agree": bool(top1_norm == top1_raw),
            "p_top1_norm": m_norm.get("p_top1"),
            "p_top1_raw": m_raw.get("p_top1"),
            "p_answer_norm": m_norm.get("p_answer"),
            "p_answer_raw": m_raw.get("p_answer"),
            "answer_rank_norm": m_norm.get("answer_rank"),
            "answer_rank_raw": m_raw.get("answer_rank"),
            "is_answer_norm": is_ans_norm,
            "is_answer_raw": is_ans_raw,
        })
    except Exception:
        # best-effort, do not fail the run
        pass


def summarize_raw_lens_check(samples: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary fields from collected samples.

    Returns keys: first_norm_only_semantic_layer, max_kl_norm_vs_raw_bits, lens_artifact_risk.
    Risk heuristic: high if any norm-only semantics or max KL≥1.0; medium if KL≥0.5; else low.
    """
    samples_list = list(samples)
    if not samples_list:
        return {
            "first_norm_only_semantic_layer": None,
            "max_kl_norm_vs_raw_bits": None,
            "lens_artifact_risk": None,
        }
    max_kl = max(s.get("kl_norm_vs_raw_bits", 0.0) for s in samples_list)
    first_norm_only = None
    for s in samples_list:
        if s.get("is_answer_norm") and not s.get("is_answer_raw"):
            first_norm_only = s.get("layer")
            break
    if first_norm_only is not None or max_kl >= 1.0:
        risk = "high"
    elif max_kl >= 0.5:
        risk = "medium"
    else:
        risk = "low"
    return {
        "first_norm_only_semantic_layer": first_norm_only,
        "max_kl_norm_vs_raw_bits": max_kl,
        "lens_artifact_risk": risk,
    }


def compute_windowed_raw_norm(
    *,
    radius: int,
    center_layers: Iterable[int],
    norm_logits_map: Dict[int, torch.Tensor],
    raw_resid_map: Dict[int, torch.Tensor],
    collected_map: Dict[int, Dict[str, Any]],
    final_probs: torch.Tensor,
    W_U: torch.Tensor,
    b_U: Optional[torch.Tensor],
    force_fp32_unembed: bool,
    decode_id_fn,
    first_ans_token_id: Optional[int],
    ground_truth: str,
    prompt_id: str,
    prompt_variant: str,
    n_layers: int,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Compute windowed raw-vs-norm diagnostics around candidate collapse layers."""

    available_layers = set(norm_logits_map.keys()) & set(raw_resid_map.keys())
    clean_centers = sorted({int(c) for c in center_layers if isinstance(c, int) and -1 <= c <= (n_layers + 1)})
    filtered_centers = [c for c in clean_centers if 0 <= c <= n_layers and (not available_layers or c in available_layers)]
    summary = {
        "radius": int(radius),
        "center_layers": filtered_centers,
        "layers_checked": [],
        "norm_only_semantics_layers": [],
        "max_kl_norm_vs_raw_bits_window": None,
        "mode": "window",
    }

    if not norm_logits_map or not raw_resid_map or not available_layers:
        return summary, []

    radius = max(0, int(radius))
    layers_to_check: List[int] = []
    for center in summary["center_layers"]:
        for layer in range(max(0, center - radius), min(n_layers, center + radius) + 1):
            if layer in available_layers:
                layers_to_check.append(layer)
    layers_to_check = sorted(dict.fromkeys(layers_to_check))

    if not layers_to_check:
        return summary, []

    records: List[Dict[str, Any]] = []
    norm_only_layers: List[int] = []
    max_kl_window: Optional[float] = None

    final_probs_cpu = final_probs.detach().float().cpu()

    for layer in layers_to_check:
        norm_logits = norm_logits_map.get(layer)
        raw_vec = raw_resid_map.get(layer)
        if norm_logits is None or raw_vec is None:
            continue

        try:
            norm_logits_cpu = norm_logits.detach().float().cpu()
        except Exception:
            norm_logits_cpu = torch.tensor([], dtype=torch.float32)
        if norm_logits_cpu.numel() == 0:
            continue

        norm_probs = torch.softmax(norm_logits_cpu, dim=0)
        top1_norm = int(torch.argmax(norm_probs).item())
        metrics_norm = compute_next_token_metrics(norm_probs, top1_norm, final_probs_cpu, first_ans_token_id, topk_cum=5)
        top1_norm_str = decode_id_fn(top1_norm)

        try:
            raw_vec_cpu = raw_vec.detach().float().cpu()
        except Exception:
            raw_vec_cpu = torch.tensor([], dtype=torch.float32)
        if raw_vec_cpu.numel() == 0:
            continue

        device = W_U.device if hasattr(W_U, "device") else torch.device("cpu")
        resid_vec = raw_vec_cpu.to(device).unsqueeze(0)
        resid_cast = safe_cast_for_unembed(resid_vec, W_U, force_fp32_unembed=force_fp32_unembed)
        logits_raw = unembed_mm(resid_cast, W_U, b_U).squeeze(0).float()
        raw_probs = torch.softmax(logits_raw.cpu(), dim=0)
        top1_raw = int(torch.argmax(raw_probs).item())
        metrics_raw = compute_next_token_metrics(raw_probs, top1_raw, final_probs_cpu, first_ans_token_id, topk_cum=5)
        top1_raw_str = decode_id_fn(top1_raw)

        try:
            kl_window = float(kl_bits(norm_probs, raw_probs))
        except Exception:
            kl_window = None

        if kl_window is not None:
            max_kl_window = max(kl_window, max_kl_window) if max_kl_window is not None else kl_window

        is_answer_norm = bool(metrics_norm.get("answer_rank") == 1)
        if metrics_norm.get("answer_rank") is None and collected_map.get(layer):
            is_answer_norm = bool(collected_map[layer].get("is_answer"))
        is_answer_raw = bool(metrics_raw.get("answer_rank") == 1)
        if metrics_raw.get("answer_rank") is None and first_ans_token_id is None:
            # fall back to string comparison
            is_answer_raw = is_semantic_top1(top1_raw_str, ground_truth)

        if is_answer_norm and not is_answer_raw:
            norm_only_layers.append(layer)

        for lens_tag, metrics, top1_id, top1_str in (
            (
                "norm",
                metrics_norm,
                top1_norm,
                top1_norm_str,
            ),
            (
                "raw",
                metrics_raw,
                top1_raw,
                top1_raw_str,
            ),
        ):
            records.append(
                {
                    "prompt_id": prompt_id,
                    "prompt_variant": prompt_variant,
                    "layer": layer,
                    "lens": lens_tag,
                    "p_top1": metrics.get("p_top1"),
                    "top1_token_id": top1_id,
                    "top1_token_str": top1_str,
                    "p_answer": metrics.get("p_answer"),
                    "answer_rank": metrics.get("answer_rank"),
                    "kl_norm_vs_raw_bits": kl_window,
                }
            )

        summary["layers_checked"].append(layer)

    summary["layers_checked"] = sorted(dict.fromkeys(summary["layers_checked"]))
    summary["norm_only_semantics_layers"] = sorted(dict.fromkeys(norm_only_layers))
    summary["max_kl_norm_vs_raw_bits_window"] = max_kl_window

    return summary, records
