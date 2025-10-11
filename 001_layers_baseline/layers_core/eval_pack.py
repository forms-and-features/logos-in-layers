from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple


def _filter_records(
    records: Iterable[Dict[str, Any]],
    *,
    prompt_id: str = "pos",
    prompt_variant: str = "orig",
    fact_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    return [
        rec
        for rec in records
        if rec.get("prompt_id") == prompt_id and rec.get("prompt_variant") == prompt_variant
        and (
            fact_index is None
            or (
                rec.get("fact_index") is None and fact_index == 0
            )
            or (
                rec.get("fact_index") is not None
                and _matches_int(rec.get("fact_index"), fact_index)
            )
        )
    ]


def _matches_int(value: Any, target: int) -> bool:
    try:
        return int(value) == int(target)
    except Exception:
        return False


def _layer_row_map(records: List[Dict[str, Any]]) -> Dict[int, int]:
    layer_to_row: Dict[int, int] = {}
    for idx, rec in enumerate(records):
        layer = rec.get("layer")
        if isinstance(layer, int) and layer not in layer_to_row:
            layer_to_row[layer] = idx
    return layer_to_row


def _frac(layer: Optional[int], n_layers: Optional[int]) -> Optional[float]:
    if not isinstance(layer, int):
        return None
    if not isinstance(n_layers, int) or n_layers <= 0:
        return None
    return round(float(layer) / float(n_layers), 4)


def _sanitise_float(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except Exception:
        return None


def _sanitise_int(value: Any) -> Optional[int]:
    try:
        return None if value is None else int(value)
    except Exception:
        return None


def _collect(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for val in values:
        try:
            if val is None:
                continue
            out.append(float(val))
        except Exception:
            continue
    return out


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    pct = max(0.0, min(1.0, float(pct)))
    vals_sorted = sorted(values)
    rank = pct * (len(vals_sorted) - 1)
    low = int(rank)
    high = min(len(vals_sorted) - 1, low + 1)
    if low == high:
        return float(vals_sorted[low])
    frac = rank - low
    return float(vals_sorted[low] * (1.0 - frac) + vals_sorted[high] * frac)


def _soft_copy_entry(diag: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    soft_block = ((diag.get("copy_detector") or {}).get("soft") or {}).get("L_copy_soft", {})
    candidates: List[Tuple[int, Optional[int]]] = []
    if isinstance(soft_block, dict):
        for key, value in soft_block.items():
            if not isinstance(value, int):
                continue
            try:
                k = int(str(key).lstrip("k"))
            except Exception:
                k = None
            candidates.append((value, k))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: (item[0], item[1] if item[1] is not None else 999))
    best_layer, best_k = candidates[0]
    return best_layer, best_k


def _repeatability_flag(repeat_diag: Dict[str, Any]) -> str:
    status = repeat_diag.get("status")
    if status not in {"ok", "skipped", "unavailable"}:
        status = "ok" if repeat_diag else "unavailable"
    if status != "ok":
        return str(status)
    try:
        max_dev = float(repeat_diag.get("max_rank_dev", 0.0) or 0.0)
        flip_rate = float(repeat_diag.get("top1_flip_rate", 0.0) or 0.0)
    except Exception:
        max_dev, flip_rate = 0.0, 0.0
    if max_dev > 5.0 or flip_rate > 0.02:
        return "high_variance"
    return "ok"


def _prepare_milestone_rows(
    *,
    pure_records: List[Dict[str, Any]],
    layer_to_row: Dict[int, int],
    milestones: Dict[str, Any],
    confirmed_source: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def _row_payload(layer: Optional[int]) -> Dict[str, Any]:
        rec = None
        if isinstance(layer, int):
            idx = layer_to_row.get(layer)
            if idx is not None and 0 <= idx < len(pure_records):
                rec = pure_records[idx]
        payload = {
            "layer": layer if isinstance(layer, int) else "",
            "answer_rank": rec.get("answer_rank") if rec else "",
            "p_answer": rec.get("p_answer") if rec else "",
            "kl_to_final_bits": rec.get("kl_to_final_bits") if rec else "",
            "entropy_bits": rec.get("entropy_bits", rec.get("entropy")) if rec else "",
        }
        return payload

    copy_strict_layer = milestones.get("L_copy_strict")
    copy_soft_info = milestones.get("L_copy_soft", {}) or {}
    copy_soft_layer = copy_soft_info.get("layer")
    copy_soft_k = copy_soft_info.get("k")
    semantic_layer = milestones.get("L_semantic_norm")
    confirmed_block = milestones.get("L_semantic_confirmed", {}) or {}
    confirmed_layer = confirmed_block.get("layer")

    row_strict = {
        **_row_payload(copy_strict_layer),
        "is_copy_strict": bool(isinstance(copy_strict_layer, int)),
        "is_copy_soft_k": "",
        "is_semantic_norm": False,
        "is_semantic_confirmed": False,
        "lens": "norm",
    }
    rows.append(row_strict)

    row_soft = {
        **_row_payload(copy_soft_layer),
        "is_copy_strict": False,
        "is_copy_soft_k": copy_soft_k if isinstance(copy_soft_layer, int) else "",
        "is_semantic_norm": False,
        "is_semantic_confirmed": False,
        "lens": "norm",
    }
    rows.append(row_soft)

    row_semantic = {
        **_row_payload(semantic_layer),
        "is_copy_strict": False,
        "is_copy_soft_k": "",
        "is_semantic_norm": bool(isinstance(semantic_layer, int)),
        "is_semantic_confirmed": False,
        "lens": "norm",
    }
    rows.append(row_semantic)

    row_confirmed = {
        **_row_payload(confirmed_layer if isinstance(confirmed_layer, int) else None),
        "is_copy_strict": False,
        "is_copy_soft_k": "",
        "is_semantic_norm": False,
        "is_semantic_confirmed": bool(isinstance(confirmed_layer, int)),
        "lens": confirmed_source or "none",
    }
    rows.append(row_confirmed)

    return rows


def _artifact_rows(
    *,
    artifact_summary: Dict[str, Any],
    raw_full_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    score_block = artifact_summary.get("score", {}) if isinstance(artifact_summary, dict) else {}
    js_pct = (artifact_summary.get("js_divergence_percentiles") or {}) if isinstance(artifact_summary, dict) else {}
    l1_pct = (artifact_summary.get("l1_prob_diff_percentiles") or {}) if isinstance(artifact_summary, dict) else {}
    overlap = (artifact_summary.get("topk_overlap") or {}) if isinstance(artifact_summary, dict) else {}
    kl_rn_values = _collect(rec.get("kl_raw_to_norm_bits") for rec in raw_full_rows)
    kl_rn_p50 = _percentile(kl_rn_values, 0.50)

    rows.append(
        {
            "layer": "_summary",
            "js_divergence": _sanitise_float(js_pct.get("p50")),
            "kl_raw_to_norm_bits": kl_rn_p50,
            "l1_prob_diff": _sanitise_float(l1_pct.get("p50")),
            "topk_jaccard_raw_norm@50": _sanitise_float(overlap.get("jaccard_raw_norm_p50")),
            "lens_artifact_score_v2": _sanitise_float(score_block.get("lens_artifact_score_v2")),
            "risk_tier": score_block.get("tier"),
        }
    )

    for rec in raw_full_rows:
        rows.append(
            {
                "layer": rec.get("layer"),
                "js_divergence": rec.get("js_divergence"),
                "kl_raw_to_norm_bits": rec.get("kl_raw_to_norm_bits"),
                "l1_prob_diff": rec.get("l1_prob_diff"),
                "topk_jaccard_raw_norm@50": rec.get("topk_jaccard_raw_norm@50"),
                "lens_artifact_score_v2": "",
                "risk_tier": "",
            }
        )

    return rows


def build_evaluation_pack(
    *,
    model_name: str,
    n_layers: Optional[int],
    json_data: Dict[str, Any],
    json_data_tuned: Optional[Dict[str, Any]],
    diag: Dict[str, Any],
    measurement_guidance: Optional[Dict[str, Any]],
    tuned_audit_summary: Optional[Dict[str, Any]],
    tuned_audit_data: Optional[Dict[str, Any]],
    clean_name: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    pure_norm_records = _filter_records(
        json_data.get("pure_next_token_records", []),
        fact_index=0,
    )
    layer_to_row = _layer_row_map(pure_norm_records)
    raw_full_records = _filter_records(
        json_data.get("raw_lens_full_records", []),
        fact_index=0,
    )

    mg = measurement_guidance or {}
    preferred_lens = mg.get("preferred_lens_for_reporting") or "norm"
    use_confirmed = bool(mg.get("use_confirmed_semantics", False))

    L_copy_strict = diag.get("L_copy") if isinstance(diag.get("L_copy"), int) else None
    copy_soft_layer, copy_soft_k = _soft_copy_entry(diag)
    L_semantic_norm = diag.get("L_semantic") if isinstance(diag.get("L_semantic"), int) else None
    confirmed_block = diag.get("confirmed_semantics") or {}
    confirmed_layer = confirmed_block.get("L_semantic_confirmed") if isinstance(confirmed_block.get("L_semantic_confirmed"), int) else None
    confirmed_source = confirmed_block.get("confirmed_source", "none")

    base_copy_layer = L_copy_strict if isinstance(L_copy_strict, int) else copy_soft_layer

    milestones = {
        "L_copy_strict": L_copy_strict,
        "L_copy_soft": {"k": copy_soft_k, "layer": copy_soft_layer},
        "L_semantic_norm": L_semantic_norm,
        "L_semantic_confirmed": {"layer": confirmed_layer, "source": confirmed_source},
        "depth_fractions": {
            "copy_strict_frac": _frac(L_copy_strict, n_layers),
            "semantic_frac": _frac(L_semantic_norm, n_layers),
            "delta_hat": None,
        },
    }
    if isinstance(L_semantic_norm, int) and isinstance(base_copy_layer, int) and isinstance(n_layers, int) and n_layers > 0:
        milestones["depth_fractions"]["delta_hat"] = round(float(L_semantic_norm - base_copy_layer) / float(n_layers), 4)

    artifact_summary = diag.get("raw_lens_full") or {}
    score_block = artifact_summary.get("score") or {}
    topk_overlap = artifact_summary.get("topk_overlap") or {}
    artifact = {
        "lens_artifact_score": _sanitise_float(score_block.get("lens_artifact_score")),
        "lens_artifact_score_v2": _sanitise_float(score_block.get("lens_artifact_score_v2")),
        "js_divergence_p50": _sanitise_float((artifact_summary.get("js_divergence_percentiles") or {}).get("p50")),
        "l1_prob_diff_p50": _sanitise_float((artifact_summary.get("l1_prob_diff_percentiles") or {}).get("p50")),
        "first_js_le_0.1": _sanitise_int(artifact_summary.get("first_js_le_0.1")),
        "first_l1_le_0.5": _sanitise_int(artifact_summary.get("first_l1_le_0.5")),
        "jaccard_raw_norm_p50": _sanitise_float(topk_overlap.get("jaccard_raw_norm_p50")),
        "first_jaccard_raw_norm_ge_0.5": _sanitise_int(topk_overlap.get("first_jaccard_raw_norm_ge_0.5")),
        "pct_layers_kl_ge_1.0": _sanitise_float(artifact_summary.get("pct_layers_kl_ge_1.0")),
        "n_norm_only_semantics_layers": artifact_summary.get("n_norm_only_semantics_layers"),
        "earliest_norm_only_semantic": artifact_summary.get("earliest_norm_only_semantic"),
        "risk_tier": score_block.get("tier"),
    }

    repeat_diag = diag.get("repeatability") or {}
    repeatability = {
        "max_rank_dev": _sanitise_float(repeat_diag.get("max_rank_dev")),
        "p95_rank_dev": _sanitise_float(repeat_diag.get("p95_rank_dev")),
        "top1_flip_rate": _sanitise_float(repeat_diag.get("top1_flip_rate")),
        "flag": _repeatability_flag(repeat_diag),
    }

    alignment = {
        "gold_alignment_rate": _sanitise_float(diag.get("gold_alignment_rate")),
        "variant": (diag.get("gold_alignment") or {}).get("variant"),
    }

    norm_traj = diag.get("norm_trajectory") or {}
    entropy_block = diag.get("entropy_gap_bits_percentiles") or {}

    pack = {
        "model": model_name,
        "n_layers": n_layers,
        "preferred_lens_for_reporting": preferred_lens,
        "use_confirmed_semantics": use_confirmed,
        "milestones": milestones,
        "artifact": artifact,
        "repeatability": repeatability,
        "alignment": alignment,
        "norm_trajectory": {
            "shape": norm_traj.get("shape"),
            "slope": _sanitise_float(norm_traj.get("slope")),
            "r2": _sanitise_float(norm_traj.get("r2")),
            "n_spikes": norm_traj.get("n_spikes"),
        },
        "entropy": {
            "entropy_gap_bits_p25": _sanitise_float(entropy_block.get("p25")),
            "entropy_gap_bits_p50": _sanitise_float(entropy_block.get("p50")),
            "entropy_gap_bits_p75": _sanitise_float(entropy_block.get("p75")),
        },
        "tuned_audit": tuned_audit_summary or {},
    }

    citations_layers = {
        "L_copy_strict_row": layer_to_row.get(L_copy_strict) if isinstance(L_copy_strict, int) else None,
        "L_copy_soft_row": layer_to_row.get(copy_soft_layer) if isinstance(copy_soft_layer, int) else None,
        "L_semantic_norm_row": layer_to_row.get(L_semantic_norm) if isinstance(L_semantic_norm, int) else None,
        "L_semantic_confirmed_row": layer_to_row.get(confirmed_layer) if isinstance(confirmed_layer, int) else None,
    }

    has_tuned = bool(json_data_tuned and ((json_data_tuned.get("pure_next_token_records") or []) or (json_data_tuned.get("records") or [])))
    has_tuned_variants = bool((tuned_audit_data or {}).get("variant_rows"))
    has_positional = bool((tuned_audit_data or {}).get("positional_rows"))
    has_raw_full = bool(raw_full_records)

    citations_files = {
        "pure_csv": f"output-{clean_name}-pure-next-token.csv",
        "tuned_pure_csv": f"output-{clean_name}-pure-next-token-tuned.csv" if has_tuned else None,
        "rawlens_full_csv": f"output-{clean_name}-pure-next-token-rawlens.csv" if has_raw_full else None,
        "tuned_variants_csv": f"output-{clean_name}-pure-next-token-tuned-variants.csv" if has_tuned_variants else None,
        "positions_audit_csv": f"output-{clean_name}-positions-tuned-audit.csv" if has_positional else None,
    }

    pack["citations"] = {
        "layers": citations_layers,
        "files": citations_files,
    }

    milestone_rows = _prepare_milestone_rows(
        pure_records=pure_norm_records,
        layer_to_row=layer_to_row,
        milestones={
            "L_copy_strict": L_copy_strict,
            "L_copy_soft": {"k": copy_soft_k, "layer": copy_soft_layer},
            "L_semantic_norm": L_semantic_norm,
            "L_semantic_confirmed": {"layer": confirmed_layer},
        },
        confirmed_source=confirmed_source,
    )

    # Attach CSV row indices to milestone rows for downstream consumption
    for row in milestone_rows:
        layer = row.get("layer")
        if isinstance(layer, int):
            row_index = layer_to_row.get(layer)
        else:
            row_index = None
        row["row_index"] = row_index

    artifact_rows = _artifact_rows(
        artifact_summary=artifact_summary if isinstance(artifact_summary, dict) else {},
        raw_full_rows=raw_full_records,
    )

    micro_diag = diag.get("micro_suite") if isinstance(diag, dict) else None
    if isinstance(micro_diag, dict):
        diag_facts = micro_diag.get("facts") or []
        diag_aggs = micro_diag.get("aggregates") or {}
        diag_citations = (micro_diag.get("citations") or {}).get("fact_rows") or {}
        fact_entries: List[Dict[str, Any]] = []
        for fact_entry in diag_facts:
            if not isinstance(fact_entry, dict):
                continue
            fact_entries.append(
                {
                    "fact_key": fact_entry.get("fact_key"),
                    "fact_index": fact_entry.get("fact_index"),
                    "L_copy_strict": fact_entry.get("L_copy_strict"),
                    "L_semantic_norm": fact_entry.get("L_semantic_norm"),
                    "L_semantic_confirmed": fact_entry.get("L_semantic_confirmed"),
                    "L_semantic_margin_ok_norm": fact_entry.get("L_semantic_margin_ok_norm"),
                    "delta_hat": fact_entry.get("delta_hat"),
                }
            )
        pack["micro_suite"] = {
            "facts": fact_entries,
            "aggregates": {
                "L_semantic_confirmed_median": diag_aggs.get("L_semantic_confirmed_median"),
                "delta_hat_median": diag_aggs.get("delta_hat_median"),
                "n_missing": diag_aggs.get("n_missing"),
                "n": diag_aggs.get("n"),
            },
            "citations": {
                "fact_rows": diag_citations,
            },
        }

    return pack, milestone_rows, artifact_rows


__all__ = ["build_evaluation_pack"]
