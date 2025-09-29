from typing import List, Dict, Any, Optional, Sequence


def summarize_pure_records(
    pure_records: List[Dict[str, Any]],
    *,
    copy_threshold: float,
    copy_window_k: int,
    copy_soft_threshold: float,
    copy_soft_window_ks: Sequence[int],
    copy_match_level: str = "id_subsequence",
    lens_tag: Optional[str] = None,
    surface_delta: float = 0.05,
    geom_gamma: float = 0.02,
    topk_prompt_tau: float = 0.33,
    n_layers: Optional[int] = None,
    cos_thresholds: Sequence[float] = (0.2, 0.4, 0.6),
) -> Dict[str, Any]:
    """Summarize collapse and threshold indices from per-layer pure next-token records.

    Args:
        pure_records: Collected per-layer metrics for a given lens.
        copy_threshold: Strict copy probability threshold.
        copy_window_k: Strict copy window size (tokens).
        copy_soft_threshold: Soft copy probability threshold.
        copy_soft_window_ks: Iterable of soft copy window sizes.
        copy_match_level: Description of copy detection mode (informational).
        lens_tag: Lens identifier (``"norm"``/``"tuned"``/etc.).
        surface_delta: Echo→answer crossover margin (§1.13).
        geom_gamma: Geometric crossover margin (§1.14).
        topk_prompt_tau: Prompt-mass decay threshold (§1.15).
        n_layers: Total transformer layers; enables normalized depth fractions (§1.20).
        cos_thresholds: Cosine milestones to track for `cos_to_final` (§1.20).

    Returns:
        Dict[str, Any]: Collapse indices, KL/rank milestones, cosine milestones,
        surface/geom diagnostics, and normalized depth fractions where available.
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

    # Surface/geom/top-k summaries
    L_surface_to_meaning: Optional[int] = None
    answer_mass_at_L_surface: Optional[float] = None
    echo_mass_at_L_surface: Optional[float] = None
    delta_mass_at_L_surface: Optional[float] = None

    L_geom: Optional[int] = None
    cos_answer_at_L_geom: Optional[float] = None
    cos_prompt_at_L_geom: Optional[float] = None

    L_topk_decay: Optional[int] = None
    topk_mass_at_L: Optional[float] = None

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

        # Surface mass crossover
        a_mass = rec.get("answer_mass")
        e_mass = rec.get("echo_mass_prompt")
        if (
            L_surface_to_meaning is None
            and a_mass is not None
            and e_mass is not None
            and a_mass >= (e_mass + surface_delta)
        ):
            L_surface_to_meaning = layer
            answer_mass_at_L_surface = a_mass
            echo_mass_at_L_surface = e_mass
            dm = rec.get("answer_minus_echo_mass")
            if dm is None:
                dm = a_mass - e_mass
            delta_mass_at_L_surface = dm

        # Geometric crossover
        ca = rec.get("cos_to_answer")
        cp = rec.get("cos_to_prompt_max")
        if (
            L_geom is None
            and ca is not None
            and cp is not None
            and ca >= (cp + geom_gamma)
        ):
            L_geom = layer
            cos_answer_at_L_geom = ca
            cos_prompt_at_L_geom = cp

        # Top-K decay
        tk = rec.get("topk_prompt_mass@50")
        if L_topk_decay is None and tk is not None and tk <= topk_prompt_tau:
            L_topk_decay = layer
            topk_mass_at_L = tk

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

    # Surface/geom/top-k extras
    suffix = f"_{lens_tag}" if lens_tag else ""
    summary[f"L_surface_to_meaning{suffix}"] = L_surface_to_meaning
    summary[f"answer_mass_at_L_surface{suffix}"] = answer_mass_at_L_surface
    summary[f"echo_mass_at_L_surface{suffix}"] = echo_mass_at_L_surface
    summary[f"L_geom{suffix}"] = L_geom
    summary[f"cos_to_answer_at_L_geom{suffix}"] = cos_answer_at_L_geom
    summary[f"cos_to_prompt_max_at_L_geom{suffix}"] = cos_prompt_at_L_geom
    summary[f"L_topk_decay{suffix}"] = L_topk_decay
    summary[f"topk_prompt_mass_at_L{suffix}"] = topk_mass_at_L
    summary[f"topk_prompt_tau{suffix}"] = topk_prompt_tau
    summary[f"surface_delta{suffix}"] = surface_delta
    summary[f"geom_gamma{suffix}"] = geom_gamma

    # Cosine milestones (per lens)
    lens_key = lens_tag or "norm"
    thresholds: List[float] = []
    try:
        thresholds = sorted({float(t) for t in cos_thresholds})
    except Exception:
        thresholds = [0.2, 0.4, 0.6]

    def _format_thresh(value: float) -> str:
        return f"{value:.3f}".rstrip('0').rstrip('.')

    milestone_labels = [f"ge_{_format_thresh(t)}" for t in thresholds]
    milestones = {label: None for label in milestone_labels}
    # Iterate in layer order to capture earliest milestone crossings
    ordered_records = sorted(
        [rec for rec in pure_records if isinstance(rec.get("layer"), int)],
        key=lambda rec: rec["layer"],
    )
    for rec in ordered_records:
        cos_val = rec.get("cos_to_final")
        if cos_val is None:
            continue
        try:
            cos_float = float(cos_val)
        except (TypeError, ValueError):
            continue
        layer_idx = rec.get("layer")
        for thresh, label in zip(thresholds, milestone_labels):
            if milestones[label] is None and cos_float >= thresh:
                milestones[label] = layer_idx

    summary.setdefault("cos_milestones", {})[lens_key] = milestones

    # Normalized depth fractions (baseline lens only)
    if lens_key == "norm" and n_layers and n_layers > 0:
        denom = float(n_layers)

        def _frac(val: Optional[Any]) -> Optional[float]:
            if val is None:
                return None
            try:
                return round(float(val) / denom, 3)
            except (TypeError, ValueError):
                return None

        depth_fractions: Dict[str, Optional[float]] = {}
        depth_fractions["L_semantic_frac"] = _frac(summary.get("L_semantic"))
        depth_fractions["first_rank_le_5_frac"] = _frac(summary.get("first_rank_le_5"))

        strict_layer = summary.get("copy_detector", {}).get("strict", {}).get("L_copy_strict")
        depth_fractions["L_copy_strict_frac"] = _frac(strict_layer)

        soft_layers_map = summary.get("copy_detector", {}).get("soft", {}).get("L_copy_soft", {}) or {}
        soft_windows = sorted({int(k) for k in copy_soft_window_ks if int(k) > 0})
        for k in soft_windows:
            key = f"k{k}"
            depth_fractions[f"L_copy_soft_k{k}_frac"] = _frac(soft_layers_map.get(key))

        summary["depth_fractions"] = depth_fractions

    return summary


# --- Unified sidecar summaries (PROJECT_NOTES §1.21) ------------------------

from typing import Tuple


def _filter_pure_records(records: List[Dict[str, Any]], *, prompt_id: str, prompt_variant: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in records or []:
        if rec.get("prompt_id") != prompt_id:
            continue
        if rec.get("prompt_variant") != prompt_variant:
            continue
        if not isinstance(rec.get("layer"), int):
            continue
        out.append(rec)
    # ensure increasing layer order
    out.sort(key=lambda r: r.get("layer"))
    return out


def _rank_milestones(records: List[Dict[str, Any]]) -> Dict[str, Optional[int]]:
    # earliest layer where answer_rank ≤ {10,5,1}
    res = {"le_10": None, "le_5": None, "le_1": None}
    for rec in records:
        ar = rec.get("answer_rank")
        if ar is None:
            continue
        try:
            ar_i = int(ar)
        except Exception:
            continue
        L = rec.get("layer")
        if res["le_10"] is None and ar_i <= 10:
            res["le_10"] = L
        if res["le_5"] is None and ar_i <= 5:
            res["le_5"] = L
        if res["le_1"] is None and ar_i <= 1:
            res["le_1"] = L
        if res["le_10"] is not None and res["le_5"] is not None and res["le_1"] is not None:
            break
    return res


def _kl_bits_at_percentiles(
    records: List[Dict[str, Any]],
    n_layers: int,
    *,
    percents: Tuple[float, float, float] = (0.25, 0.50, 0.75),
) -> Dict[str, Optional[float]]:
    # Map each target layer index to the rec's kl_to_final_bits
    out: Dict[str, Optional[float]] = {"p25": None, "p50": None, "p75": None}
    if not isinstance(n_layers, int) or n_layers < 0:
        return out
    # Build lookup by layer
    by_layer = {int(rec.get("layer")): rec for rec in records if isinstance(rec.get("layer"), int)}
    layer_targets = [max(0, min(n_layers, int(round(n_layers * p)))) for p in percents]
    keys = ["p25", "p50", "p75"]
    for key, L in zip(keys, layer_targets):
        rec = by_layer.get(L)
        try:
            out[key] = None if rec is None else float(rec.get("kl_to_final_bits"))
        except Exception:
            out[key] = None
    return out


def _first_kl_le_threshold(records: List[Dict[str, Any]], *, threshold: float = 1.0) -> Optional[int]:
    for rec in records:
        try:
            val = rec.get("kl_to_final_bits")
            v = None if val is None else float(val)
        except Exception:
            v = None
        if v is not None and v <= float(threshold):
            return rec.get("layer")
    return None


def build_unified_lens_metrics(
    *,
    baseline_records: List[Dict[str, Any]],
    alt_records: List[Dict[str, Any]] | None,
    n_layers: int,
    alt_label: str,
    prompt_id: str = "pos",
    prompt_variant: str = "orig",
) -> Dict[str, Any]:
    """Compute unified sidecar metrics for a lens vs baseline (PROJECT_NOTES §1.21).

    Returns a dict with keys: rank_milestones, kl_bits_at_percentiles, first_kl_le_1.0
    where each contains baseline/alt/delta entries. The alt subkey is named by
    `alt_label` (e.g., "prism" or "tuned").
    """
    # Filter to the primary prompt/variant
    base = _filter_pure_records(baseline_records, prompt_id=prompt_id, prompt_variant=prompt_variant)
    alt = _filter_pure_records(alt_records or [], prompt_id=prompt_id, prompt_variant=prompt_variant)

    # Rank milestones
    rm_b = _rank_milestones(base)
    rm_a = _rank_milestones(alt)
    rm_delta = {k: (None if rm_b.get(k) is None or rm_a.get(k) is None else (int(rm_a[k]) - int(rm_b[k]))) for k in rm_b}

    # KL@percentiles (bits)
    kl_b = _kl_bits_at_percentiles(base, n_layers)
    kl_a = _kl_bits_at_percentiles(alt, n_layers)
    kl_delta = {}
    for k in kl_b.keys():
        vb, va = kl_b.get(k), kl_a.get(k)
        kl_delta[k] = (None if vb is None or va is None else float(vb) - float(va))

    # First KL ≤ 1.0
    fk_b = _first_kl_le_threshold(base, threshold=1.0)
    fk_a = _first_kl_le_threshold(alt, threshold=1.0)
    fk_delta = None if fk_b is None or fk_a is None else int(fk_a) - int(fk_b)

    return {
        "rank_milestones": {
            "baseline": rm_b,
            alt_label: rm_a,
            "delta": rm_delta,
        },
        "kl_bits_at_percentiles": {
            "baseline": kl_b,
            alt_label: kl_a,
            "delta": kl_delta,
        },
        "first_kl_le_1.0": {
            "baseline": fk_b,
            alt_label: fk_a,
            "delta": fk_delta,
        },
    }
