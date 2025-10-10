from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple
import math

import torch

from .numerics import bits_entropy_from_logits, kl_bits
from .collapse_rules import (
    detect_copy_collapse_id_subseq,
    is_pure_whitespace_or_punct,
    is_semantic_top1,
    is_id_subseq,
)
from .metrics import compute_next_token_metrics
from .surface import (
    compute_surface_masses,
    compute_geometric_cosines,
    compute_topk_prompt_mass,
)


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
    # Surface/geom diagnostics
    prompt_vocab_ids: Optional[Iterable[int]] = None,
    decoder_weight: Optional[torch.Tensor] = None,
    geom_vec: Optional[torch.Tensor] = None,
    topk_prompt_mass_k: int = 50,
    geom_gamma: float = 0.02,
    # Norm-lens per-layer temperature (norm-only)
    norm_temp_tau: Optional[float] = None,
    # Strict-copy threshold sweep (k=1 window, same margin).
    # When None, defaults to (0.70, 0.80, 0.90, 0.95) and will emit flags
    # labeled via format_copy_strict_label(). Pass an empty iterable to disable.
    copy_strict_thresholds: Optional[Iterable[float]] = None,
    bias_tensor: Optional[torch.Tensor] = None,
    raw_resid_vec: Optional[torch.Tensor] = None,
    norm_resid_vec: Optional[torch.Tensor] = None,
    p_uniform: Optional[float] = None,
    semantic_margin_delta: float = 0.002,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Compute pure next-token metrics and summary structures.

    Returns:
    - view dict: { pos, token_str, entropy_bits, top_tokens, top_probs, record_extra }
    - collected dict: minimal fields for L_copy/L_sem summaries
    - dual_lens ctx dict: fields needed for optional raw-lens sampling
    """
    last_pos = tokens_tensor.shape[1] - 1
    last_logits = logits_all[last_pos]
    bias_free_logits = None
    if geom_vec is not None and decoder_weight is not None:
        try:
            bias_free_logits = torch.matmul(
                geom_vec.detach().to(dtype=torch.float32),
                decoder_weight.detach().to(dtype=torch.float32),
            )
        except Exception:
            bias_free_logits = None
    if bias_free_logits is None:
        bias_free_logits = last_logits
        if bias_tensor is not None:
            try:
                bias_free_logits = bias_free_logits - bias_tensor.to(
                    device=last_logits.device,
                    dtype=last_logits.dtype,
                )
            except Exception:
                pass
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
    cos_to_final = None
    try:
        logits_vec = bias_free_logits.to(dtype=torch.float32)
        denom = torch.norm(logits_vec) + 1e-12
        if torch.isfinite(denom) and denom > 0:
            cos_to_final = torch.dot(logits_vec / denom, final_dir_vec.to(dtype=torch.float32)).item()
    except Exception:
        cos_to_final = None

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

    # Uniform semantics margin
    vocab_size = int(last_logits.shape[-1])
    inferred_uniform = None
    if vocab_size > 0:
        inferred_uniform = 1.0 / float(vocab_size)
    uniform_baseline = p_uniform if p_uniform is not None else inferred_uniform
    answer_minus_uniform = None
    if uniform_baseline is not None and metrics.get("p_answer") is not None:
        try:
            answer_minus_uniform = float(metrics["p_answer"]) - float(uniform_baseline)
        except Exception:
            answer_minus_uniform = None
    if answer_minus_uniform is None and metrics.get("p_answer") is not None:
        try:
            answer_minus_uniform = float(metrics["p_answer"])
        except Exception:
            answer_minus_uniform = None

    semantic_margin_ok: Optional[bool] = None
    if lens_type == "norm":
        rank_val = metrics.get("answer_rank")
        if rank_val is None or answer_minus_uniform is None:
            semantic_margin_ok = None
        else:
            try:
                semantic_margin_ok = bool(int(rank_val) == 1 and float(answer_minus_uniform) >= float(semantic_margin_delta))
            except Exception:
                semantic_margin_ok = None

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

    # Strict-copy threshold sweep (k=1; ID membership + prob threshold + margin)
    # Default thresholds when not explicitly provided
    strict_tau_list: Tuple[float, ...]
    if copy_strict_thresholds is None:
        strict_tau_list = (0.70, 0.80, 0.90, 0.95)
    else:
        strict_tau_list = tuple(sorted({float(t) for t in copy_strict_thresholds}))

    # Compute base stats to avoid recomputation
    # top-2 from last position for margin check
    _vals2, _idx2 = torch.topk(last_logits, 2, largest=True, sorted=True)
    _full = last_full_probs
    _p1 = float(_full[_idx2[0]].item()) if _full is not None else None
    _p2 = float(_full[_idx2[1]].item()) if _full is not None else None
    strict_hits: Dict[str, bool] = {}
    # k=1 window is last top-1 id
    window_k1 = window_manager.get_window(lens_type, prompt_id, prompt_variant, 1)
    # Heuristic: ignore whitespace/punct-only top-1 tokens
    _top1_str = last_top_tokens[0] if last_top_tokens else None
    k1_ok = not is_pure_whitespace_or_punct(_top1_str)
    if k1_ok and len(window_k1) >= 1:
        in_ctx = is_id_subseq(window_k1, list(ctx_ids_list))
        for tau in strict_tau_list:
            label = f"copy_strict@{format(tau, '.2f').rstrip('0').rstrip('.')}"
            hit = False
            try:
                hit = (
                    in_ctx
                    and _p1 is not None and _p2 is not None
                    and (_p1 > float(tau))
                    and ((_p1 - _p2) > float(copy_margin))
                )
            except Exception:
                hit = False
            strict_hits[label] = bool(hit)
    else:
        for tau in strict_tau_list:
            label = f"copy_strict@{format(tau, '.2f').rstrip('0').rstrip('.')}"
            strict_hits[label] = False

    answer_logit_gap = None
    answer_vs_top1_gap = None
    if first_ans_token_id is not None and 0 <= int(first_ans_token_id) < last_logits.shape[-1]:
        ans_id = int(first_ans_token_id)
        ans_logit = float(last_logits[ans_id].item())
        top1_current = int(_idx2[0].item())
        answer_rank = metrics.get("answer_rank")
        if answer_rank == 1 and top1_current == ans_id:
            if len(_vals2) > 1:
                answer_logit_gap = ans_logit - float(_vals2[1].item())
        elif answer_rank is not None and answer_rank > 1:
            answer_vs_top1_gap = ans_logit - float(_vals2[0].item())

    # Teacher entropy (final distribution at the NEXT position), in bits
    try:
        teacher_entropy_bits = float(-(final_probs_tensor * (final_probs_tensor + 1e-30).log()).sum().item() / math.log(2))
    except Exception:
        teacher_entropy_bits = None

    # Surface masses
    echo_mass_prompt = None
    answer_mass = None
    mass_ratio = None
    if prompt_vocab_ids is not None:
        echo_mass_prompt, answer_mass, mass_ratio = compute_surface_masses(
            last_full_probs, prompt_vocab_ids, first_ans_token_id
        )
    answer_minus_echo = None
    if answer_mass is not None and echo_mass_prompt is not None:
        answer_minus_echo = float(answer_mass - echo_mass_prompt)

    # Geometric cosines
    cos_to_answer = None
    cos_to_prompt_max = None
    if geom_vec is not None and decoder_weight is not None:
        cos_to_answer, cos_to_prompt_max = compute_geometric_cosines(
            geom_vec, decoder_weight, prompt_vocab_ids or [], first_ans_token_id
        )
    geom_crossover = None
    if cos_to_answer is not None and cos_to_prompt_max is not None:
        geom_crossover = bool(cos_to_answer >= (cos_to_prompt_max + geom_gamma))

    # Top-K prompt mass
    topk_prompt_mass = None
    if prompt_vocab_ids is not None:
        topk_prompt_mass = compute_topk_prompt_mass(
            last_full_probs, prompt_vocab_ids, topk_prompt_mass_k
        )

    resid_norm_ratio = None
    delta_resid_cos = None
    if raw_resid_vec is not None and norm_resid_vec is not None:
        try:
            raw_vec = raw_resid_vec.detach().to(dtype=torch.float32)
            norm_vec = norm_resid_vec.detach().to(dtype=torch.float32)
            raw_norm = torch.norm(raw_vec) + 1e-12
            norm_norm = torch.norm(norm_vec)
            if torch.isfinite(raw_norm) and raw_norm > 0:
                resid_norm_ratio = float((norm_norm + 1e-12) / raw_norm)
            denom = (torch.norm(raw_vec) * torch.norm(norm_vec)) + 1e-12
            if torch.isfinite(denom) and denom > 0:
                dot_val = torch.dot(raw_vec.flatten(), norm_vec.flatten())
                delta_resid_cos = float(torch.clamp(dot_val / denom, -1.0, 1.0))
        except Exception:
            resid_norm_ratio = None
            delta_resid_cos = None

    # Norm temperature KL to teacher (norm-only): KL(P(z/τ) || P_final)
    kl_norm_temp_bits = None
    if norm_temp_tau is not None and norm_temp_tau > 0:
        P = torch.softmax(last_logits / float(norm_temp_tau), dim=0)
        try:
            kl_norm_temp_bits = kl_bits(P, final_probs_tensor)
        except Exception:
            kl_norm_temp_bits = None

    record_extra = {
        "copy_collapse": copy_collapse,
        copy_strict_label: copy_collapse,
        "entropy_collapse": entropy_collapse,
        "is_answer": is_answer,
        "top1_token_id": top1_id,
        "top1_token_str": last_top_tokens[0] if last_top_tokens else None,
        **metrics,
        "cos_to_final": cos_to_final,
        "control_margin": control_margin,
        # Surface and geom
        "echo_mass_prompt": echo_mass_prompt,
        "answer_mass": answer_mass,
        "mass_ratio_ans_over_prompt": mass_ratio,
        "answer_minus_echo_mass": answer_minus_echo,
        "cos_to_answer": cos_to_answer,
        "cos_to_prompt_max": cos_to_prompt_max,
        "geom_crossover": geom_crossover,
        "topk_prompt_mass@50": topk_prompt_mass,
        "kl_to_final_bits_norm_temp": kl_norm_temp_bits,
        "resid_norm_ratio": resid_norm_ratio,
        "delta_resid_cos": delta_resid_cos,
        "answer_logit_gap": answer_logit_gap,
        "answer_vs_top1_gap": answer_vs_top1_gap,
        "answer_minus_uniform": answer_minus_uniform,
    }
    record_extra["entropy_bits"] = float(last_entropy_bits)
    if lens_type == "norm":
        record_extra["semantic_margin_ok"] = semantic_margin_ok
    # Add strict-sweep flags to the flat record, if any
    for k_label, hit in strict_hits.items():
        # Avoid duplicating the base strict label if identical formatting
        if k_label != copy_strict_label:
            record_extra[k_label] = hit
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
        "top1_token_id": top1_id,
        "cos_to_final": cos_to_final,
        # Surface and geom collected for summaries
        "echo_mass_prompt": echo_mass_prompt,
        "answer_mass": answer_mass,
        "mass_ratio_ans_over_prompt": mass_ratio,
        "answer_minus_echo_mass": answer_minus_echo,
        "cos_to_answer": cos_to_answer,
        "cos_to_prompt_max": cos_to_prompt_max,
        "geom_crossover": geom_crossover,
        "topk_prompt_mass@50": topk_prompt_mass,
        "kl_to_final_bits_norm_temp": kl_norm_temp_bits,
    }
    collected["entropy_bits"] = float(last_entropy_bits)
    collected["teacher_entropy_bits"] = teacher_entropy_bits
    if teacher_entropy_bits is not None:
        try:
            collected["entropy_gap_bits"] = float(last_entropy_bits) - float(teacher_entropy_bits)
        except Exception:
            collected["entropy_gap_bits"] = None
    if strict_hits:
        collected["copy_strict_hits"] = strict_hits
    collected["answer_minus_uniform"] = answer_minus_uniform
    collected["p_answer"] = metrics.get("p_answer")
    if lens_type == "norm":
        collected["semantic_margin_ok"] = semantic_margin_ok

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
