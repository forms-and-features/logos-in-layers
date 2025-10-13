from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import math
import torch

from .numerics import kl_bits


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return float(vals[0])
    q = max(0.0, min(1.0, float(q)))
    rank = q * (len(vals) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(vals[lower])
    weight = rank - lower
    return float(vals[lower] * (1 - weight) + vals[upper] * weight)


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


def summarize_rotation_temperature(variant_rows: Sequence[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    rot_vals = _collect(row.get("delta_kl_bits_rot_only") for row in variant_rows)
    temp_vals = _collect(row.get("delta_kl_bits_temp_only") for row in variant_rows)
    tuned_vals = _collect(row.get("delta_kl_bits_tuned") for row in variant_rows)
    interaction_vals = _collect(row.get("delta_kl_bits_interaction") for row in variant_rows)

    summary = {
        "delta_kl_rot_p25": _percentile(rot_vals, 0.25),
        "delta_kl_rot_p50": _percentile(rot_vals, 0.50),
        "delta_kl_rot_p75": _percentile(rot_vals, 0.75),
        "delta_kl_temp_p25": _percentile(temp_vals, 0.25),
        "delta_kl_temp_p50": _percentile(temp_vals, 0.50),
        "delta_kl_temp_p75": _percentile(temp_vals, 0.75),
        "delta_kl_interaction_p50": _percentile(interaction_vals, 0.50),
        "delta_kl_tuned_p50": _percentile(tuned_vals, 0.50),
    }
    return summary


def summarize_positional(
    positional_rows: Sequence[Dict[str, Any]],
    pos_grid: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    grid = [float(entry.get("pos_frac", 0.0)) for entry in pos_grid]
    in_dist = _collect(
        row.get("delta_kl_bits_tuned")
        for row in positional_rows
        if row.get("pos_frac") is not None and float(row["pos_frac"]) <= 0.92
    )
    ood = _collect(
        row.get("delta_kl_bits_tuned")
        for row in positional_rows
        if row.get("pos_frac") is not None and float(row["pos_frac"]) >= 0.96
    )
    pos_summary = {
        "pos_grid": grid,
        "pos_in_dist_le_0.92": _percentile(in_dist, 0.50),
        "pos_ood_ge_0.96": _percentile(ood, 0.50),
        "pos_ood_gap": None,
    }
    a = pos_summary["pos_ood_ge_0.96"]
    b = pos_summary["pos_in_dist_le_0.92"]
    if a is not None and b is not None:
        pos_summary["pos_ood_gap"] = float(a) - float(b)
    return pos_summary


def compute_tau_star_modelcal(
    tuned_logits: torch.Tensor,
    final_probs: torch.Tensor,
    *,
    tau_min: float = 0.2,
    tau_max: float = 5.0,
    coarse_steps: int = 61,
    fine_steps: int = 41,
) -> Tuple[Optional[float], Optional[float]]:
    if tuned_logits is None or final_probs is None:
        return None, None
    try:
        logits = tuned_logits.detach().to(dtype=torch.float32)
        target = final_probs.detach().to(dtype=torch.float32)
    except Exception:
        return None, None
    if logits.ndim != 1 or target.ndim != 1 or logits.shape[0] != target.shape[0]:
        return None, None

    def _kl_for_tau(tau: float) -> float:
        scaled = logits / tau
        probs = torch.softmax(scaled, dim=0)
        return kl_bits(probs, target)

    taus = torch.logspace(math.log10(tau_min), math.log10(tau_max), steps=coarse_steps)
    evals: List[Tuple[float, float]] = []
    for tau in taus:
        tau_f = float(tau.item())
        evals.append((tau_f, _kl_for_tau(tau_f)))
    evals.append((1.0, _kl_for_tau(1.0)))
    best_tau, best_val = min(evals, key=lambda item: item[1])

    left = max(tau_min, best_tau * 0.5)
    right = min(tau_max, best_tau * 1.5)
    fine_taus = torch.linspace(left, right, steps=fine_steps)
    for tau in fine_taus:
        tau_f = float(tau.item())
        evals.append((tau_f, _kl_for_tau(tau_f)))
    best_tau, best_val = min(evals, key=lambda item: item[1])
    return best_tau, best_val


def summarize_head_mismatch(head_payload: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    if not isinstance(head_payload, dict):
        return {
            "kl_bits_tuned_final": None,
            "kl_bits_tuned_final_after_tau_star": None,
            "tau_star_modelcal": None,
        }
    tuned_logits = head_payload.get("tuned_logits_last")
    final_probs = head_payload.get("final_probs")
    kl_tuned = head_payload.get("kl_bits_tuned_final")
    tau_star, kl_after = compute_tau_star_modelcal(tuned_logits, final_probs)
    return {
        "kl_bits_tuned_final": float(kl_tuned) if kl_tuned is not None else None,
        "kl_bits_tuned_final_after_tau_star": kl_after,
        "tau_star_modelcal": tau_star,
    }


def build_tuned_audit_summary(
    audit_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(audit_data, dict):
        return {
            "rotation_vs_temperature": None,
            "positional": None,
            "head_mismatch": summarize_head_mismatch(None),
            "tuned_is_calibration_only": None,
            "preferred_semantics_lens_hint": "norm",
        }

    variant_rows = audit_data.get("variant_rows") or []
    positional_rows = audit_data.get("positional_rows") or []
    pos_grid = audit_data.get("pos_grid") or []

    rotation_summary = summarize_rotation_temperature(variant_rows)
    positional_summary = summarize_positional(positional_rows, pos_grid)
    head_summary = summarize_head_mismatch(audit_data.get("head_mismatch"))

    rot_p50 = rotation_summary.get("delta_kl_rot_p50")
    temp_p50 = rotation_summary.get("delta_kl_temp_p50")
    tuned_p50 = rotation_summary.get("delta_kl_tuned_p50")

    tuned_is_calibration_only = None
    if (
        rot_p50 is not None
        and temp_p50 is not None
        and tuned_p50 is not None
    ):
        tuned_is_calibration_only = (rot_p50 < 0.2) and (temp_p50 >= 0.8 * tuned_p50)

    preferred_hint = "norm"
    if tuned_is_calibration_only:
        preferred_hint = "tuned_for_calibration_only"
    elif tuned_p50 is not None and tuned_p50 > 0.2:
        preferred_hint = "tuned"

    return {
        "rotation_vs_temperature": rotation_summary,
        "positional": positional_summary,
        "head_mismatch": head_summary,
        "tuned_is_calibration_only": tuned_is_calibration_only,
        "preferred_semantics_lens_hint": preferred_hint,
    }


def build_provenance_snapshot(provenance: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(provenance, dict):
        return None

    translator = provenance.get("translator", {})
    training = provenance.get("training", {})
    dataset = (training.get("dataset") or {})

    temps = _collect(translator.get("temperatures", []) or [])
    temp_stats = None
    if temps:
        temp_stats = {
            "min": min(temps),
            "p25": _percentile(temps, 0.25),
            "p50": _percentile(temps, 0.50),
            "p75": _percentile(temps, 0.75),
            "max": max(temps),
        }

    precond = translator.get("preconditioner") or {}
    precond_summary = {
        "whiten": bool(translator.get("has_preconditioner", False)),
        "orthogonal_rotation": bool(precond.get("rotation")),
    }

    tokens_per_step = training.get("tokens_per_step")
    total_steps = training.get("total_steps")
    fit_tokens = None
    try:
        if tokens_per_step is not None and total_steps is not None:
            fit_tokens = float(tokens_per_step) * float(total_steps)
    except Exception:
        fit_tokens = None

    pos_range = dataset.get("position_fraction_range") or []
    if isinstance(pos_range, (list, tuple)) and len(pos_range) >= 2:
        train_pos_window = [float(pos_range[0]), float(pos_range[1])]
    else:
        train_pos_window = None

    snapshot = {
        "dataset_id": dataset.get("repo_id") or dataset.get("name"),
        "dataset_revision": dataset.get("revision"),
        "content_hash": dataset.get("content_hash"),
        "train_pos_window": train_pos_window,
        "sampled_layers_count": training.get("layers_sampled_per_step"),
        "sampled_positions_count": dataset.get("positions_per_seq") or training.get("positions_per_step"),
        "rank": translator.get("rank"),
        "preconditioner": precond_summary,
        "temperatures_stats": temp_stats,
        "final_layer_identity": bool(translator.get("final_identity", False)),
        "fit_total_tokens_est": fit_tokens,
        "optimizer": training.get("optimizer"),
        "schedule": training.get("schedule"),
    }
    return snapshot


def compute_position_window_stability(
    positional_rows: Sequence[Dict[str, Any]],
    pos_grid_entries: Sequence[Dict[str, Any]],
    *,
    semantic_layer: Optional[int],
    run2_layer: Optional[int] = None,
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Return summary stats for position-grid stability at semantic onset.

    Args:
        positional_rows: Rows emitted by the tuned positions audit.
        pos_grid_entries: Grid metadata with `pos_index` / `pos_frac`.
        semantic_layer: Baseline semantic onset layer (norm lens).
        run2_layer: Starting layer for the semantic run-of-two band.

    Returns:
        (summary_dict or None, low_stability_flag)
    """

    def _to_int(val: Any) -> Optional[int]:
        try:
            if val is None:
                return None
            return int(val)
        except (TypeError, ValueError):
            return None

    def _is_rank_one(row: Dict[str, Any]) -> bool:
        return _to_int(row.get("answer_rank_baseline")) == 1

    if not isinstance(semantic_layer, int):
        return None, False

    grid_fractions: List[float] = []
    grid_positions: set[int] = set()
    for entry in pos_grid_entries or []:
        pos_idx = _to_int(entry.get("pos_index"))
        frac = entry.get("pos_frac")
        if pos_idx is None:
            continue
        grid_positions.add(pos_idx)
        try:
            if frac is not None:
                grid_fractions.append(float(frac))
        except (TypeError, ValueError):
            continue
    if grid_fractions:
        grid_fractions = sorted({float(f) for f in grid_fractions})
    else:
        grid_fractions = []

    rows_at_semantic = [
        row for row in positional_rows or []
        if _to_int(row.get("layer")) == semantic_layer
    ]
    by_pos: Dict[int, Dict[str, Any]] = {}
    for row in rows_at_semantic:
        pos_idx = _to_int(row.get("pos_index"))
        if pos_idx is None:
            continue
        by_pos[pos_idx] = row

    if grid_positions:
        # Restrict to known grid positions when available.
        by_pos = {idx: row for idx, row in by_pos.items() if idx in grid_positions}

    n_positions = len(by_pos)
    rank1_frac: Optional[float] = None
    if n_positions > 0:
        passes = sum(1 for row in by_pos.values() if _is_rank_one(row))
        rank1_frac = float(passes) / float(n_positions)

    rank1_frac_run2: Optional[float] = None
    if isinstance(run2_layer, int):
        run2_layers = {int(run2_layer), int(run2_layer) + 1}
        per_pos_pass: Dict[int, bool] = {}
        for row in positional_rows or []:
            layer_val = _to_int(row.get("layer"))
            if layer_val not in run2_layers:
                continue
            pos_idx = _to_int(row.get("pos_index"))
            if pos_idx is None:
                continue
            if grid_positions and pos_idx not in grid_positions:
                continue
            prev = per_pos_pass.get(pos_idx, False)
            per_pos_pass[pos_idx] = prev or _is_rank_one(row)

        if per_pos_pass:
            denominator = len(per_pos_pass)
            passes = sum(1 for flag in per_pos_pass.values() if flag)
            rank1_frac_run2 = float(passes) / float(denominator)

    summary = {
        "grid": grid_fractions,
        "L_semantic_norm": int(semantic_layer),
        "rank1_frac": rank1_frac,
        "n_positions": n_positions,
        "rank1_frac_strong_run2": rank1_frac_run2,
    }

    low_stability = (rank1_frac is not None) and (rank1_frac < 0.50)
    return summary, low_stability


__all__ = [
    "build_tuned_audit_summary",
    "summarize_rotation_temperature",
    "summarize_positional",
    "summarize_head_mismatch",
    "compute_tau_star_modelcal",
    "build_provenance_snapshot",
    "compute_position_window_stability",
]
