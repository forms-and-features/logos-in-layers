from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import torch

from .records import make_record, make_pure_record
from .numerics import bits_entropy_from_logits
from .pure_emit import compute_pure_next_token_info


def append_prism_record(
    buf: Dict[str, Any],
    *,
    prompt_id: str,
    prompt_variant: str,
    layer: int,
    pos: int,
    token: str,
    logits_pos: torch.Tensor,
    decode_id_fn,
    top_k: int,
) -> None:
    probs = torch.softmax(logits_pos, dim=0)
    # Use the shared helper for entropy in bits to match baseline numerics
    ent_bits = bits_entropy_from_logits(logits_pos)
    _, idx = torch.topk(logits_pos, top_k, largest=True, sorted=True)
    top_probs = probs[idx]
    top_tokens = [decode_id_fn(i) for i in idx]
    rec = make_record(
        prompt_id=prompt_id,
        prompt_variant=prompt_variant,
        layer=layer,
        pos=pos,
        token=token,
        entropy=ent_bits,
        top_tokens=top_tokens,
        top_probs=top_probs,
    )
    buf.setdefault("records", []).append(rec)


def append_prism_pure_next_token(
    buf: Dict[str, Any],
    *,
    layer_out_idx: int,
    prism_logits_all: torch.Tensor,
    tokens_tensor: torch.Tensor,
    ctx_ids_list: Iterable[int],
    window_manager,
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
) -> None:
    flag_labels = [copy_strict_label] + [copy_soft_labels.get(int(k)) for k in copy_soft_window_ks if copy_soft_labels.get(int(k))]
    flag_labels.extend(copy_soft_extra_labels.values())
    existing_cols = buf.setdefault("copy_flag_columns", [])
    for label in flag_labels:
        if label and label not in existing_cols:
            existing_cols.append(label)

    view, collected, _ = compute_pure_next_token_info(
        layer_out_idx=layer_out_idx,
        logits_all=prism_logits_all,
        tokens_tensor=tokens_tensor,
        ctx_ids_list=ctx_ids_list,
        window_manager=window_manager,
        lens_type="prism",
        final_probs_tensor=final_probs_tensor,
        first_ans_token_id=first_ans_token_id,
        final_dir_vec=final_dir_vec,
        copy_threshold=copy_threshold,
        copy_margin=copy_margin,
        copy_strict_label=copy_strict_label,
        copy_soft_threshold=copy_soft_threshold,
        copy_soft_window_ks=copy_soft_window_ks,
        copy_soft_labels=copy_soft_labels,
        copy_soft_extra_labels=copy_soft_extra_labels,
        entropy_collapse_threshold=entropy_collapse_threshold,
        decode_id_fn=decode_id_fn,
        ground_truth=ground_truth,
        top_k_record=top_k_record,
        prompt_id=prompt_id,
        prompt_variant=prompt_variant,
        control_ids=control_ids,
        copy_strict_thresholds=(),
    )
    rec = make_pure_record(
        prompt_id=prompt_id,
        prompt_variant=prompt_variant,
        layer=layer_out_idx,
        pos=view["pos"],
        token=view["token_str"],
        entropy=view["entropy_bits"],
        top_tokens=view["top_tokens"],
        top_probs=view["top_probs"],
        extra=view["record_extra"],
    )
    buf.setdefault("pure_next_token_records", []).append(rec)
