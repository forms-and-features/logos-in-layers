from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Set


SemMilestone = Tuple[Optional[str], Optional[int]]


def _to_int(value: Any) -> Optional[int]:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def resolve_preferred_semantic_milestone(diag: Dict[str, Any]) -> SemMilestone:
    """Return (name, layer) for the preferred semantic milestone.

    Priority order (§1.54): L_semantic_strong_run2 → L_semantic_strong →
    L_semantic_confirmed → L_semantic_norm.
    """
    if not isinstance(diag, dict):
        return (None, None)

    sem_gate = diag.get("semantic_gate") or {}
    candidates = (
        ("L_semantic_strong_run2", sem_gate.get("L_semantic_strong_run2")),
        ("L_semantic_strong", sem_gate.get("L_semantic_strong")),
        ("L_semantic_confirmed", diag.get("L_semantic_confirmed")),
        ("L_semantic_norm", diag.get("L_semantic")),
    )
    for name, raw_layer in candidates:
        layer = _to_int(raw_layer)
        if layer is not None:
            return (name, layer)
    return (None, None)


def _record_matches(
    record: Dict[str, Any],
    *,
    prompt_id: str,
    prompt_variant: str,
    target_layer: int,
    fact_index: Optional[int],
) -> bool:
    if record.get("prompt_id") != prompt_id:
        return False
    if record.get("prompt_variant") != prompt_variant:
        return False
    rec_layer = _to_int(record.get("layer"))
    if rec_layer is None or rec_layer != target_layer:
        return False
    if fact_index is None:
        return True
    rec_idx = record.get("fact_index")
    if rec_idx is None:
        return fact_index == 0
    rec_idx_int = _to_int(rec_idx)
    return rec_idx_int is not None and rec_idx_int == fact_index


def _find_record_for_layer(
    records: Sequence[Dict[str, Any]],
    *,
    prompt_id: str,
    prompt_variant: str,
    target_layer: int,
    fact_index: Optional[int],
) -> Optional[Dict[str, Any]]:
    for rec in records:
        if _record_matches(
            rec,
            prompt_id=prompt_id,
            prompt_variant=prompt_variant,
            target_layer=target_layer,
            fact_index=fact_index,
        ):
            return rec
    return None


def _topk_token_set(record: Dict[str, Any], k: int) -> Set[str]:
    tokens: Set[str] = set()
    topk = record.get("topk") or []
    for entry in topk[: max(0, int(k))]:
        if isinstance(entry, (list, tuple)) and entry:
            tok = entry[0]
        else:
            tok = entry
        if tok is None:
            continue
        tokens.add(str(tok))
    return tokens


def _jaccard(set_a: Set[str], set_b: Set[str]) -> Optional[float]:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return None
    intersection = set_a & set_b
    return float(len(intersection)) / float(len(union))


def build_repeatability_forward_summary(
    *,
    pass1_diag: Dict[str, Any],
    pass2_diag: Optional[Dict[str, Any]],
    pass1_records: Sequence[Dict[str, Any]],
    pass2_records: Optional[Sequence[Dict[str, Any]]],
    prompt_id: str,
    prompt_variant: str,
    fact_index: Optional[int],
    tolerance_layers: int,
    topk_k: int,
    min_jaccard: float = 0.5,
) -> Dict[str, Any]:
    """Assemble summary payload for forward-of-two repeatability gate."""
    summary: Dict[str, Any] = {
        "milestones": {
            "primary": None,
            "pass1": {"layer": None},
            "pass2": {"layer": None},
            "delta_layers": None,
        },
        "topk_jaccard_at_primary_layer": None,
        "gate": {"repeatability_forward_pass": None},
    }

    primary_name1, layer1 = resolve_preferred_semantic_milestone(pass1_diag or {})
    summary["milestones"]["primary"] = primary_name1
    summary["milestones"]["pass1"]["layer"] = layer1

    if pass2_diag is None or pass2_records is None:
        return summary

    primary_name2, layer2 = resolve_preferred_semantic_milestone(pass2_diag)
    summary["milestones"]["pass2"]["layer"] = layer2

    if layer1 is not None and layer2 is not None:
        summary["milestones"]["delta_layers"] = abs(layer2 - layer1)

    if layer1 is not None:
        rec1 = _find_record_for_layer(
            pass1_records,
            prompt_id=prompt_id,
            prompt_variant=prompt_variant,
            target_layer=layer1,
            fact_index=fact_index,
        )
        rec2 = _find_record_for_layer(
            pass2_records,
            prompt_id=prompt_id,
            prompt_variant=prompt_variant,
            target_layer=layer1,
            fact_index=fact_index,
        )
        if rec1 is not None and rec2 is not None:
            set_a = _topk_token_set(rec1, topk_k)
            set_b = _topk_token_set(rec2, topk_k)
            summary["topk_jaccard_at_primary_layer"] = _jaccard(set_a, set_b)

    gate_result: Optional[bool]
    if primary_name1 is None or layer1 is None:
        gate_result = None
    elif primary_name2 is None or layer2 is None:
        gate_result = False
    else:
        name_match = primary_name1 == primary_name2
        delta = summary["milestones"]["delta_layers"]
        delta_ok = isinstance(delta, int) and abs(int(delta)) <= max(0, int(tolerance_layers))
        jaccard = summary["topk_jaccard_at_primary_layer"]
        jaccard_ok = (jaccard is not None) and (jaccard >= min_jaccard)
        gate_result = bool(name_match and delta_ok and jaccard_ok)
    summary["gate"]["repeatability_forward_pass"] = gate_result
    return summary

