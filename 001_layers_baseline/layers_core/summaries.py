from typing import List, Dict, Any, Optional, Sequence


def summarize_pure_records(
    pure_records: List[Dict[str, Any]],
    *,
    copy_threshold: float,
    copy_window_k: int,
    copy_soft_threshold: float,
    copy_soft_window_ks: Sequence[int],
    copy_match_level: str = "id_subsequence",
) -> Dict[str, Any]:
    """Summarize collapse and threshold indices from per-layer pure next-token records.

    Expects each record to include keys: layer, copy_collapse, entropy_collapse,
    is_answer, kl_to_final_bits (float), and answer_rank (int or None).
    Returns a dict suitable for diag.update in run.py.
    """
    L_copy: Optional[int] = None
    L_copy_H: Optional[int] = None
    L_sem: Optional[int] = None
    soft_window_set = list(dict.fromkeys(int(k) for k in copy_soft_window_ks if int(k) > 0))
    if not soft_window_set:
        soft_window_set = [copy_window_k]
    L_copy_soft: Dict[int, Optional[int]] = {k: None for k in soft_window_set}
    first_kl_below_0_5: Optional[int] = None
    first_kl_below_1_0: Optional[int] = None
    first_rank_le_1: Optional[int] = None
    first_rank_le_5: Optional[int] = None
    first_rank_le_10: Optional[int] = None

    for rec in pure_records:
        layer = rec.get("layer")
        if L_copy is None and rec.get("copy_collapse"):
            L_copy = layer
        if L_copy_H is None and rec.get("entropy_collapse"):
            L_copy_H = layer
        if L_sem is None and rec.get("is_answer"):
            L_sem = layer

        if isinstance(rec.get("copy_soft_hits"), dict):
            for k in soft_window_set:
                if L_copy_soft[k] is None and rec["copy_soft_hits"].get(k):
                    L_copy_soft[k] = layer

        kl_bits = rec.get("kl_to_final_bits")
        if kl_bits is not None:
            if first_kl_below_0_5 is None and kl_bits <= 0.5:
                first_kl_below_0_5 = layer
            if first_kl_below_1_0 is None and kl_bits <= 1.0:
                first_kl_below_1_0 = layer

        ar = rec.get("answer_rank")
        if ar is not None:
            if first_rank_le_1 is None and ar <= 1:
                first_rank_le_1 = layer
            if first_rank_le_5 is None and ar <= 5:
                first_rank_le_5 = layer
            if first_rank_le_10 is None and ar <= 10:
                first_rank_le_10 = layer

    delta_soft: Dict[int, Optional[int]] = {}
    for k, layer_idx in L_copy_soft.items():
        if L_sem is None or layer_idx is None:
            delta_soft[k] = None
        else:
            delta_soft[k] = L_sem - layer_idx

    summary = {
        "L_copy": L_copy,
        "L_copy_H": L_copy_H,
        "L_semantic": L_sem,
        "delta_layers": None if (L_copy is None or L_sem is None) else (L_sem - L_copy),
        # provenance for copy detector
        "copy_thresh": copy_threshold,
        "copy_window_k": copy_window_k,
        "copy_match_level": copy_match_level,
        # §1.3 summary thresholds
        "first_kl_below_0.5": first_kl_below_0_5,
        "first_kl_below_1.0": first_kl_below_1_0,
        "first_rank_le_1": first_rank_le_1,
        "first_rank_le_5": first_rank_le_5,
        "first_rank_le_10": first_rank_le_10,
        "copy_soft_threshold": copy_soft_threshold,
        "copy_soft_window_ks": soft_window_set,
        "L_copy_soft": L_copy_soft,
        "delta_layers_soft": delta_soft,
        "copy_detector": {
            "strict": {
                "thresh": copy_threshold,
                "k": copy_window_k,
                "L_copy_strict": L_copy,
            },
            "soft": {
                "thresh": copy_soft_threshold,
                "window_ks": soft_window_set,
                "L_copy_soft": {f"k{k}": L_copy_soft[k] for k in soft_window_set},
            },
            "deltas": {
                "Δ_sem_minus_copy_strict": None if (L_copy is None or L_sem is None) else (L_sem - L_copy),
                "Δ_sem_minus_copy_soft": {f"k{k}": delta_soft[k] for k in soft_window_set},
            },
        },
    }

    return summary
