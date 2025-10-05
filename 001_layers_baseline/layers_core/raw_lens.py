import math
import math
import os
from typing import Dict, Any, Optional, Iterable, List, Tuple

import torch

from .numerics import safe_cast_for_unembed, kl_bits
from .metrics import compute_next_token_metrics
from .collapse_rules import is_semantic_top1, is_pure_whitespace_or_punct
from .unembed import unembed_mm


TOPK_JACCARD_K = 50
TOPK_JACCARD_CROSS_KEY = f"topk_jaccard_raw_norm@{TOPK_JACCARD_K}"
TOPK_JACCARD_CONSEC_KEY = f"topk_jaccard_consecutive@{TOPK_JACCARD_K}"


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
    ctx_ids_list,
    first_ans_token_id: Optional[int],
    ground_truth: str,
    prompt_id: str,
    prompt_variant: str,
    n_layers: int,
    copy_tau_list: tuple[float, ...] = (0.70, 0.80, 0.90, 0.95),
    copy_margin: float = 0.10,
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

        # Strict-copy sweep flags per lens for this layer (k=1)
        def _strict_flags(probs, top1_id, top1_str):
            try:
                top2_vals, top2_idx = torch.topk(probs, 2, largest=True, sorted=True)
                p1 = float(probs[top2_idx[0]].item())
                p2 = float(probs[top2_idx[1]].item())
            except Exception:
                p1, p2 = None, None
            in_ctx = int(top1_id) in set(int(t) for t in (ctx_ids_list or []))
            flags: Dict[str, Any] = {}
            # Guard trivial spacing tokens via decode string
            skip = is_pure_whitespace_or_punct(top1_str)
            for tau in copy_tau_list:
                label = f"copy_strict@{format(tau, '.2f').rstrip('0').rstrip('.')}"
                hit = False
                if not skip and in_ctx and p1 is not None and p2 is not None:
                    hit = (p1 > float(tau)) and ((p1 - p2) > float(copy_margin))
                flags[label] = bool(hit)
            return flags

        flags_norm = _strict_flags(norm_probs, top1_norm, top1_norm_str)
        flags_raw = _strict_flags(raw_probs, top1_raw, top1_raw_str)

        for lens_tag, metrics, top1_id, top1_str, _flags in (
            (
                "norm",
                metrics_norm,
                top1_norm,
                top1_norm_str,
                flags_norm,
            ),
            (
                "raw",
                metrics_raw,
                top1_raw,
                top1_raw_str,
                flags_raw,
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
                    **_flags,
                }
            )

        summary["layers_checked"].append(layer)

    summary["layers_checked"] = sorted(dict.fromkeys(summary["layers_checked"]))
    summary["norm_only_semantics_layers"] = sorted(dict.fromkeys(norm_only_layers))
    summary["max_kl_norm_vs_raw_bits_window"] = max_kl_window

    return summary, records


def compute_full_raw_norm(
    *,
    norm_logits_map: Dict[int, torch.Tensor],
    raw_resid_map: Dict[int, torch.Tensor],
    collected_map: Dict[int, Dict[str, Any]],
    final_probs: torch.Tensor,
    W_U: torch.Tensor,
    b_U: Optional[torch.Tensor],
    force_fp32_unembed: bool,
    decode_id_fn,
    ctx_ids_list,
    first_ans_token_id: Optional[int],
    ground_truth: str,
    prompt_id: str,
    prompt_variant: str,
    n_layers: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Compute full-depth raw-vs-norm metrics (one row per post-block layer).

    Returns (summary, rows) where rows carry both raw and norm fields for the
    pure next-token position, and summary aggregates lens-artefact indicators.
    """
    records: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "pct_layers_kl_ge_1.0": None,
        "pct_layers_kl_ge_0.5": None,
        "n_norm_only_semantics_layers": 0,
        "earliest_norm_only_semantic": None,
        "max_kl_norm_vs_raw_bits": None,
        "mode": "full",
    }

    if not norm_logits_map or not raw_resid_map:
        return summary, records

    final_probs_cpu = final_probs.detach().float().cpu()
    available_layers = sorted(set(norm_logits_map.keys()) & set(raw_resid_map.keys()))
    if not available_layers:
        return summary, records

    kl_ge_1 = 0
    kl_ge_0_5 = 0
    norm_only_layers: List[int] = []
    max_kl_val: Optional[float] = None
    js_values: List[float] = []
    l1_values: List[float] = []
    jaccard_cross_values: List[float] = []
    first_js_le_0_1: Optional[int] = None
    first_l1_le_0_5: Optional[int] = None
    first_jaccard_ge_0_5: Optional[int] = None
    per_layer_cross: Dict[int, Optional[float]] = {}
    per_layer_consecutive_norm: Dict[int, Optional[float]] = {}
    prev_topk_norm: Optional[set] = None

    def _entropy_bits_from_probs(probs: torch.Tensor) -> Optional[float]:
        try:
            p32 = probs.to(dtype=torch.float32)
            log_p = torch.log(p32 + 1e-30)
            ent_nats = -torch.sum(p32 * log_p)
            return float(ent_nats.detach().cpu().item() / math.log(2))
        except Exception:
            return None

    def _topk_id_set(probs: torch.Tensor, k: int) -> set:
        k_eff = int(max(1, min(k, probs.shape[0])))
        try:
            topk_idx = torch.topk(probs, k_eff, largest=True, sorted=False).indices
        except Exception:
            return set()
        return {int(idx) for idx in topk_idx.tolist()}

    def _jaccard(a: set, b: set) -> Optional[float]:
        if not a and not b:
            return 1.0
        union = a | b
        if not union:
            return None
        return float(len(a & b) / len(union))

    teacher_entropy_bits = _entropy_bits_from_probs(final_probs_cpu)

    for layer in available_layers:
        norm_logits = norm_logits_map.get(layer)
        raw_vec = raw_resid_map.get(layer)
        if norm_logits is None or raw_vec is None:
            continue
        try:
            norm_logits_cpu = norm_logits.detach().float().cpu()
        except Exception:
            continue
        if norm_logits_cpu.numel() == 0:
            continue
        norm_probs = torch.softmax(norm_logits_cpu, dim=0)
        top1_norm = int(torch.argmax(norm_probs).item())
        metrics_norm = compute_next_token_metrics(norm_probs, top1_norm, final_probs_cpu, first_ans_token_id, topk_cum=5)
        top1_norm_str = decode_id_fn(top1_norm)

        # Raw projection for last-position vector
        try:
            raw_vec_cpu = raw_vec.detach().float().cpu()
        except Exception:
            continue
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

        # KL and norm-only semantics
        try:
            kl_nr = float(kl_bits(norm_probs, raw_probs))
        except Exception:
            kl_nr = None
        try:
            kl_rn = float(kl_bits(raw_probs, norm_probs))
        except Exception:
            kl_rn = None

        js_divergence = None
        try:
            midpoint = 0.5 * (norm_probs + raw_probs)
            midpoint = midpoint / midpoint.sum()
            kl_nm = kl_bits(norm_probs, midpoint)
            kl_rm = kl_bits(raw_probs, midpoint)
            js_divergence = 0.5 * (kl_nm + kl_rm)
        except Exception:
            js_divergence = None

        entropy_norm_bits = _entropy_bits_from_probs(norm_probs)
        entropy_raw_bits = _entropy_bits_from_probs(raw_probs)
        entropy_gap_bits = None
        if entropy_norm_bits is not None and teacher_entropy_bits is not None:
            entropy_gap_bits = entropy_norm_bits - teacher_entropy_bits

        l1_prob_diff = None
        try:
            l1_prob_diff = float(torch.sum(torch.abs(norm_probs - raw_probs)).item())
        except Exception:
            l1_prob_diff = None

        norm_topk = _topk_id_set(norm_probs, TOPK_JACCARD_K)
        raw_topk = _topk_id_set(raw_probs, TOPK_JACCARD_K)
        jaccard_cross = _jaccard(norm_topk, raw_topk)
        if jaccard_cross is not None:
            jaccard_cross_values.append(jaccard_cross)
            if first_jaccard_ge_0_5 is None and jaccard_cross >= 0.5:
                first_jaccard_ge_0_5 = layer
        if js_divergence is not None:
            js_values.append(float(js_divergence))
            if first_js_le_0_1 is None and js_divergence <= 0.1:
                first_js_le_0_1 = layer

        if l1_prob_diff is not None:
            l1_values.append(float(l1_prob_diff))
            if first_l1_le_0_5 is None and l1_prob_diff <= 0.5:
                first_l1_le_0_5 = layer

        jaccard_consecutive_norm = None
        if prev_topk_norm is not None:
            jaccard_consecutive_norm = _jaccard(prev_topk_norm, norm_topk)
        per_layer_cross[layer] = jaccard_cross
        per_layer_consecutive_norm[layer] = jaccard_consecutive_norm
        prev_topk_norm = norm_topk

        if kl_nr is not None:
            max_kl_val = kl_nr if max_kl_val is None else max(max_kl_val, kl_nr)
            if kl_nr >= 1.0:
                kl_ge_1 += 1
            if kl_nr >= 0.5:
                kl_ge_0_5 += 1

        is_answer_norm = bool(metrics_norm.get("answer_rank") == 1)
        if metrics_norm.get("answer_rank") is None and collected_map.get(layer):
            is_answer_norm = bool(collected_map[layer].get("is_answer"))
        is_answer_raw = bool(metrics_raw.get("answer_rank") == 1)
        if metrics_raw.get("answer_rank") is None and first_ans_token_id is None:
            is_answer_raw = is_semantic_top1(top1_raw_str, ground_truth)
        norm_only = bool(is_answer_norm and not is_answer_raw)
        if norm_only:
            norm_only_layers.append(layer)

        records.append(
            {
                "prompt_id": prompt_id,
                "prompt_variant": prompt_variant,
                "layer": layer,
                # raw
                "p_top1_raw": metrics_raw.get("p_top1"),
                "top1_token_id_raw": top1_raw,
                "top1_token_str_raw": top1_raw_str,
                "p_answer_raw": metrics_raw.get("p_answer"),
                "answer_rank_raw": metrics_raw.get("answer_rank"),
                # norm
                "p_top1_norm": metrics_norm.get("p_top1"),
                "top1_token_id_norm": top1_norm,
                "top1_token_str_norm": top1_norm_str,
                "p_answer_norm": metrics_norm.get("p_answer"),
                "answer_rank_norm": metrics_norm.get("answer_rank"),
                # cross
                "kl_norm_vs_raw_bits": kl_nr,
                "kl_raw_to_norm_bits": kl_rn,
                "js_divergence": js_divergence,
                "entropy_bits_norm": entropy_norm_bits,
                "entropy_bits_raw": entropy_raw_bits,
                "entropy_gap_bits": entropy_gap_bits,
                "l1_prob_diff": l1_prob_diff,
                TOPK_JACCARD_CROSS_KEY: jaccard_cross,
                TOPK_JACCARD_CONSEC_KEY: jaccard_consecutive_norm,
                "norm_only_semantics": norm_only,
            }
        )

    total_layers = len(available_layers)
    if total_layers > 0:
        summary["pct_layers_kl_ge_1.0"] = float(kl_ge_1) / float(total_layers)
        summary["pct_layers_kl_ge_0.5"] = float(kl_ge_0_5) / float(total_layers)
    summary["n_norm_only_semantics_layers"] = len(norm_only_layers)
    summary["earliest_norm_only_semantic"] = (min(norm_only_layers) if norm_only_layers else None)
    summary["max_kl_norm_vs_raw_bits"] = max_kl_val

    def _percentile(values: List[float], pct: float) -> Optional[float]:
        if not values:
            return None
        if len(values) == 1:
            return float(values[0])
        vals_sorted = sorted(values)
        rank = (len(vals_sorted) - 1) * pct
        low = int(math.floor(rank))
        high = int(math.ceil(rank))
        if low == high:
            return float(vals_sorted[low])
        fraction = rank - low
        return float(vals_sorted[low] + (vals_sorted[high] - vals_sorted[low]) * fraction)

    summary["js_divergence_percentiles"] = {
        "p25": _percentile(js_values, 0.25),
        "p50": _percentile(js_values, 0.50),
        "p75": _percentile(js_values, 0.75),
    }
    summary["l1_prob_diff_percentiles"] = {
        "p25": _percentile(l1_values, 0.25),
        "p50": _percentile(l1_values, 0.50),
        "p75": _percentile(l1_values, 0.75),
    }
    summary["first_js_le_0.1"] = first_js_le_0_1
    summary["first_l1_le_0.5"] = first_l1_le_0_5
    summary["teacher_entropy_bits"] = teacher_entropy_bits
    summary["topk_overlap"] = {
        "K": TOPK_JACCARD_K,
        "jaccard_raw_norm_p50": _percentile(jaccard_cross_values, 0.50),
        "first_jaccard_raw_norm_ge_0.5": first_jaccard_ge_0_5,
        "per_layer_raw_norm": per_layer_cross,
        "per_layer_consecutive_norm": per_layer_consecutive_norm,
        "cross_key": TOPK_JACCARD_CROSS_KEY,
        "consecutive_key": TOPK_JACCARD_CONSEC_KEY,
    }

    return summary, records
