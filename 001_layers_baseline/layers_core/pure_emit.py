from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple
import math

import torch

from .numerics import bits_entropy_from_logits
from .collapse_rules import (
    detect_copy_collapse_id_subseq,
    is_pure_whitespace_or_punct,
    is_semantic_top1,
    is_id_subseq,
)
from .metrics import compute_next_token_metrics


def compute_pure_next_token_info(
    *,
    layer_out_idx: int,
    logits_all: torch.Tensor,
    tokens_tensor: torch.Tensor,
    ctx_ids_list: Iterable[int],
    window_manager,
    lens_type: str,
    final_probs_tensor: torch.Tensor,
    first_ans_token_id: Optional[int],
    final_dir_vec: torch.Tensor,
    copy_threshold: float,
    copy_margin: float,
    copy_strict_label: str,
    copy_soft_threshold: float,
    copy_soft_window_ks: Iterable[int],
    copy_soft_labels: Dict[int, str],
    copy_soft_extra_labels: Dict[Tuple[int, float], str],
    entropy_collapse_threshold: float,
    decode_id_fn,
    ground_truth: str,
    top_k_record: int,
    prompt_id: str,
    prompt_variant: str,
    control_ids: Optional[Tuple[Optional[int], Optional[int]]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Compute pure next-token metrics and summary structures.

    Returns:
    - view dict: { pos, token_str, entropy_bits, top_tokens, top_probs, record_extra }
    - collected dict: minimal fields for L_copy/L_sem summaries
    - dual_lens ctx dict: fields needed for optional raw-lens sampling
    """
    last_pos = tokens_tensor.shape[1] - 1
    last_logits = logits_all[last_pos]
    last_entropy_bits = bits_entropy_from_logits(last_logits)
    last_full_probs = torch.softmax(last_logits, dim=0)
    token_str = "⟨NEXT⟩"
    _, last_top_indices = torch.topk(last_logits, top_k_record, largest=True, sorted=True)
    last_top_probs = last_full_probs[last_top_indices]
    last_top_tokens = [decode_id_fn(idx) for idx in last_top_indices]

    # Update rolling window (per lens, per variant)
    top1_id = int(last_top_indices[0].item())
    window_manager.append_and_trim(lens_type, prompt_id, prompt_variant, top1_id)
    window_strict = window_manager.get_window(lens_type, prompt_id, prompt_variant, window_manager.window_k)

    # Copy / semantic flags and metrics
    copy_collapse = detect_copy_collapse_id_subseq(
        last_logits,
        list(ctx_ids_list),
        list(window_strict),
        copy_threshold=copy_threshold,
        copy_margin=copy_margin,
    )
    if copy_collapse and is_pure_whitespace_or_punct(last_top_tokens[0]):
        copy_collapse = False
    entropy_collapse = last_entropy_bits <= float(entropy_collapse_threshold)
    # Defer final is_answer decision until rank is known; keep string fallback
    is_answer_fallback = is_semantic_top1(last_top_tokens[0], ground_truth)

    metrics = compute_next_token_metrics(
        last_full_probs, top1_id, final_probs_tensor, first_ans_token_id, topk_cum=5
    )
    p_top1_value = metrics.get("p_top1", 0.0)
    # Prefer rank-based ID check when available; fallback to string match
    is_answer = (
        (metrics.get("answer_rank") == 1)
        if metrics.get("answer_rank") is not None
        else is_answer_fallback
    )

    # Cosine to final direction
    _curr_norm = torch.norm(last_logits) + 1e-12
    cos_to_final = torch.dot((last_logits / _curr_norm), final_dir_vec).item()

    # Control margin
    control_margin = None
    if control_ids is not None and all(x is not None for x in control_ids):
        paris_id, berlin_id = control_ids  # type: ignore
        try:
            control_margin = float(last_full_probs[int(paris_id)]) - float(
                last_full_probs[int(berlin_id)]
            )
        except Exception:
            control_margin = None

    # Soft copy detections (base + extra thresholds)
    soft_hits: Dict[int, bool] = {}
    for k in copy_soft_window_ks:
        k_int = int(k)
        label = copy_soft_labels.get(k_int)
        if not label:
            continue
        window_soft = window_manager.get_window(lens_type, prompt_id, prompt_variant, k_int)
        soft_hits[k_int] = (
            len(window_soft) >= k_int
            and is_id_subseq(window_soft, list(ctx_ids_list))
            and float(p_top1_value) > float(copy_soft_threshold)
        )

    soft_extra_hits: Dict[Tuple[int, float], bool] = {}
    for (k_int, th), label in copy_soft_extra_labels.items():
        window_soft = window_manager.get_window(lens_type, prompt_id, prompt_variant, k_int)
        soft_extra_hits[(int(k_int), float(th))] = (
            len(window_soft) >= int(k_int)
            and is_id_subseq(window_soft, list(ctx_ids_list))
            and float(p_top1_value) > float(th)
        )

    # Teacher entropy (final distribution at the NEXT position), in bits
    try:
        teacher_entropy_bits = float(-(final_probs_tensor * (final_probs_tensor + 1e-30).log()).sum().item() / math.log(2))
    except Exception:
        teacher_entropy_bits = None

    record_extra = {
        "copy_collapse": copy_collapse,
        copy_strict_label: copy_collapse,
        "entropy_collapse": entropy_collapse,
        "is_answer": is_answer,
        **metrics,
        "cos_to_final": cos_to_final,
        "control_margin": control_margin,
    }
    if teacher_entropy_bits is not None:
        record_extra["teacher_entropy_bits"] = teacher_entropy_bits
    for k_int, hit in soft_hits.items():
        label = copy_soft_labels.get(k_int)
        if label and label not in record_extra:
            record_extra[label] = hit
    for (k_int, th), hit in soft_extra_hits.items():
        label = copy_soft_extra_labels.get((k_int, th))
        if label and label not in record_extra:
            record_extra[label] = hit

    collected = {
        "layer": layer_out_idx,
        "copy_collapse": copy_collapse,
        "entropy_collapse": entropy_collapse,
        "is_answer": is_answer,
        "kl_to_final_bits": metrics["kl_to_final_bits"],
        "answer_rank": metrics["answer_rank"],
        "copy_soft_hits": soft_hits,
    }

    dual_ctx = {
        "layer": layer_out_idx,
        "last_pos": last_pos,
        "last_logits_norm": last_logits,
        "final_probs": final_probs_tensor,
        "first_ans_id": first_ans_token_id,
        "ground_truth": ground_truth,
    }

    view = {
        "pos": last_pos,
        "token_str": token_str,
        "entropy_bits": last_entropy_bits,
        "top_tokens": last_top_tokens,
        "top_probs": last_top_probs,
        "record_extra": record_extra,
    }
    return view, collected, dual_ctx
