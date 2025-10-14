from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, List, Set

import math
import torch
import os

from .hooks import build_cache_hook, attach_residual_hooks, detach_hooks, get_residual_safely
from .prism_sidecar import append_prism_record, append_prism_pure_next_token
from .records import make_record, make_pure_record
from .pure_emit import compute_pure_next_token_info
from .metrics import compute_next_token_metrics
from .numerics import bits_entropy_from_logits, safe_cast_for_unembed
from .norm_utils import (
    detect_model_architecture,
    get_correct_norm_module,
    apply_norm_or_skip,
    describe_norm_origin,
)
from .raw_lens import should_sample_layer, record_dual_lens_sample, compute_windowed_raw_norm, compute_full_raw_norm
from .summaries import summarize_pure_records
from .consistency import compute_last_layer_consistency
from .lenses import PrismLensAdapter
from .contexts import UnembedContext, PrismContext
from .surface import build_prompt_vocab_ids
from .unembed import unembed_mm
from .decoding_point import compare_decoding_strategies
from .repeat_forward import build_repeatability_forward_summary, resolve_preferred_semantic_milestone

SURFACE_DELTA = 0.05
GEOM_GAMMA = 0.02
TOPK_PROMPT_MASS_K = 50
TOPK_DECAY_TAU = 0.33
SEMANTIC_MARGIN_DELTA = 0.002
POS_GRID_FRACTIONS: Tuple[float, ...] = (0.20, 0.40, 0.60, 0.80, 0.92, 0.96, 0.98, 1.00)


def _is_verbose_position(pos: int, token_str: str, seq_len: int, important_words: Sequence[str]) -> bool:
    if pos == seq_len - 1:
        return True
    for w in important_words:
        if w.lower() in token_str.lower().strip(".,!?;:"):
            return True
    return False


def run_prompt_pass(
    *,
    model: Any,
    context_prompt: str,
    ground_truth: str,
    prompt_id: str,
    prompt_variant: str,
    window_manager,
    norm_lens,
    unembed_ctx: UnembedContext,
    copy_threshold: float,
    copy_margin: float,
    copy_soft_threshold: float,
    copy_soft_window_ks: Sequence[int],
    copy_strict_label: str,
    copy_soft_labels: Dict[int, str],
    copy_soft_extra_labels: Dict[Tuple[int, float], str],
    entropy_collapse_threshold: float,
    top_k_record: int,
    top_k_verbose: int,
    keep_residuals: bool,
    out_dir: Optional[str],
    RAW_LENS_MODE: str,
    json_data: Dict[str, Any],
    json_data_prism: Dict[str, Any],
    prism_ctx: PrismContext,
    decode_id_fn,
    ctx_ids_list: Sequence[int],
    first_ans_token_id: Optional[int],
    important_words: Sequence[str],
    head_scale_cfg: Optional[float],
    head_softcap_cfg: Optional[float],
    clean_model_name: Optional[str] = None,
    control_ids: Optional[Tuple[Optional[int], Optional[int]]] = None,
    enable_raw_lens_sampling: bool = True,
    tuned_spec: Optional[Dict[str, Any]] = None,
    norm_temp_taus: Optional[Sequence[Optional[float]]] = None,
    copy_strict_thresholds: Optional[Sequence[float]] = None,
    enable_repeatability_check: bool = True,
    fact_key: Optional[str] = None,
    fact_index: Optional[int] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], str, Dict[str, Any]]:
    """Run a single prompt pass and append outputs into json_data structures.

    Returns (summary_diag, last_layer_consistency, detected_architecture, diag_delta).
    """
    window_manager.reset_variant(prompt_id, prompt_variant)

    prompt_vocab_ids = build_prompt_vocab_ids(ctx_ids_list, decode_id_fn) if ctx_ids_list else []
    decoder_weight = getattr(unembed_ctx, 'W', None)

    def _tau_for(idx: int) -> Optional[float]:
        if norm_temp_taus is None:
            return None
        if idx < 0 or idx >= len(norm_temp_taus):
            return None
        return norm_temp_taus[idx]

    def _metrics_for_logits(
        logits_vec: torch.Tensor,
        final_probs_vec: torch.Tensor,
        answer_id: Optional[int],
    ) -> Tuple[Dict[str, Any], float, int]:
        probs = torch.softmax(logits_vec, dim=0)
        top1_idx = int(torch.argmax(probs).item())
        metrics = compute_next_token_metrics(
            probs,
            top1_idx,
            final_probs_vec,
            answer_id,
        )
        entropy_bits_val = bits_entropy_from_logits(logits_vec)
        return metrics, entropy_bits_val, top1_idx

    def _float_or_none(val: Any) -> Optional[float]:
        try:
            return None if val is None else float(val)
        except Exception:
            return None

    def _int_or_none(val: Any) -> Optional[int]:
        try:
            return None if val is None else int(val)
        except Exception:
            return None

    def _rank_shift(base: Optional[int], variant: Optional[int]) -> Optional[int]:
        if base is None or variant is None:
            return None
        return int(base) - int(variant)

    tuned_enabled = bool(tuned_spec and tuned_spec.get("adapter"))
    if tuned_enabled:
        tuned_adapter = tuned_spec["adapter"]
        tuned_json_data = tuned_spec["json_data"]
        tuned_window_manager = tuned_spec["window_manager"]
        tuned_window_manager.reset_variant(prompt_id, prompt_variant)
        tuned_cache = tuned_spec.setdefault("cache", {})
        tuned_collected_records: List[Dict[str, Any]] = []
        tuned_summaries = tuned_spec.setdefault("summaries", [])
    else:
        tuned_adapter = None
        tuned_json_data = None
        tuned_window_manager = None
        tuned_cache = None
        tuned_collected_records = []
        tuned_summaries = None

    collect_audit = bool(tuned_enabled and prompt_id == "pos" and prompt_variant == "orig")
    audit_data: Optional[Dict[str, Any]] = None
    if collect_audit and tuned_spec is not None:
        audit_data = {
            "variant_rows": [],
            "positional_rows": [],
            "pos_grid": [],
            "head_mismatch": None,
        }
        tuned_spec["audit_data"] = audit_data

    last_layer_consistency: Optional[Dict[str, Any]] = None
    detected_architecture: str = "unknown"

    norm_logits_window: Dict[int, torch.Tensor] = {}
    raw_resid_window: Dict[int, torch.Tensor] = {}
    tuned_logits_window: Dict[int, torch.Tensor] = {}
    collected_by_layer: Dict[int, Dict[str, Any]] = {}
    norm_arch = detect_model_architecture(model)
    norm_strategy = "post_ln2" if norm_arch == "post_norm" else "next_ln1"
    norm_provenance_entries: List[Dict[str, Any]] = []
    normalization_effects: List[Dict[str, Any]] = []
    layer_map_entries: List[Dict[str, Any]] = []
    numeric_tracker = {
        "any_nan": False,
        "any_inf": False,
        "max_abs_logit": [],
        "min_prob": [],
        "layers_flagged": [],
    }
    fact_extra: Dict[str, Any] = {
        "fact_key": fact_key,
        "fact_index": fact_index,
    }

    def _norm_effect_metrics(raw_vec: torch.Tensor | None, norm_vec: torch.Tensor | None):
        if raw_vec is None or norm_vec is None:
            return None, None, None
        try:
            raw_f = raw_vec.detach().to(dtype=torch.float32)
            norm_f = norm_vec.detach().to(dtype=torch.float32)
            raw_norm = torch.norm(raw_f) + 1e-12
            norm_norm = torch.norm(norm_f)
            ratio = None
            if torch.isfinite(raw_norm) and raw_norm > 0:
                ratio = float((norm_norm + 1e-12) / raw_norm)
            denom = (torch.norm(raw_f) * torch.norm(norm_f)) + 1e-12
            cos_val = None
            if torch.isfinite(denom) and denom > 0:
                cos_val = float(torch.clamp(torch.dot(raw_f.flatten(), norm_f.flatten()) / denom, -1.0, 1.0))
            raw_norm_val = None
            try:
                raw_norm_val = float(torch.norm(raw_f).item())
            except Exception:
                raw_norm_val = None
            return ratio, cos_val, raw_norm_val
        except Exception:
            return None, None, None

    def _update_numeric(layer_idx: int, resid_vec: torch.Tensor | None, norm_vec: torch.Tensor | None,
                        logits_vec: torch.Tensor, probs_vec: torch.Tensor | None):
        has_nan = False
        has_inf = False
        try:
            if resid_vec is not None:
                has_nan = has_nan or bool(torch.isnan(resid_vec).any())
                has_inf = has_inf or bool(torch.isinf(resid_vec).any())
        except Exception:
            pass
        try:
            if norm_vec is not None:
                has_nan = has_nan or bool(torch.isnan(norm_vec).any())
                has_inf = has_inf or bool(torch.isinf(norm_vec).any())
        except Exception:
            pass
        try:
            has_nan = has_nan or bool(torch.isnan(logits_vec).any())
            has_inf = has_inf or bool(torch.isinf(logits_vec).any())
        except Exception:
            pass
        if has_nan or has_inf:
            numeric_tracker["layers_flagged"].append({
                "layer": layer_idx,
                "lens": "norm",
                "any_nan": has_nan,
                "any_inf": has_inf,
            })
        numeric_tracker["any_nan"] = numeric_tracker["any_nan"] or has_nan
        numeric_tracker["any_inf"] = numeric_tracker["any_inf"] or has_inf
        try:
            numeric_tracker["max_abs_logit"].append(float(torch.max(torch.abs(logits_vec)).item()))
        except Exception:
            pass
        if probs_vec is not None:
            try:
                numeric_tracker["min_prob"].append(float(torch.min(probs_vec).item()))
            except Exception:
                pass

    def _percentile(values: List[float], pct: float) -> Optional[float]:
        if not values:
            return None
        vals = sorted(values)
        if len(vals) == 1:
            return float(vals[0])
        rank = pct * (len(vals) - 1)
        lower = int(math.floor(rank))
        upper = int(math.ceil(rank))
        if lower == upper:
            return float(vals[lower])
        weight = rank - lower
        return float(vals[lower] * (1 - weight) + vals[upper] * weight)

    def _redecode_layer_logits(layer_idx: int, raw_vec_cpu: torch.Tensor) -> Optional[torch.Tensor]:
        if raw_vec_cpu is None:
            return None
        try:
            target_device = getattr(unembed_ctx.W, "device", raw_vec_cpu.device)
        except Exception:
            target_device = raw_vec_cpu.device
        vec = raw_vec_cpu.to(device=target_device)
        resid = vec.unsqueeze(0).unsqueeze(0)
        if layer_idx <= 0:
            norm_module = get_correct_norm_module(model, 0, probe_after_block=False)
        else:
            norm_module = get_correct_norm_module(model, layer_idx - 1, probe_after_block=True)
        normed = apply_norm_or_skip(resid, norm_module)
        norm_vec = normed.squeeze(0).squeeze(0)
        cast_vec = safe_cast_for_unembed(norm_vec.unsqueeze(0), unembed_ctx.W, force_fp32_unembed=unembed_ctx.force_fp32)
        logits = unembed_mm(cast_vec, unembed_ctx.W, unembed_ctx.b, cache=unembed_ctx.cache).squeeze(0)
        return logits.detach().to(device=torch.device("cpu"))

    def _compute_repeatability_metrics() -> Dict[str, Any]:
        if not enable_repeatability_check:
            return {"status": "skipped", "reason": "disabled"}
        det_env = bool(torch.are_deterministic_algorithms_enabled())

        layer_keys = sorted(set(norm_logits_window.keys()) & set(raw_resid_window.keys()))
        if not layer_keys:
            result: Dict[str, Any] = {"status": "unavailable"}
            if det_env:
                result["deterministic_algorithms"] = True
            return result

        rank_devs: List[float] = []
        layers_count = 0
        flips = 0

        for layer_idx in layer_keys:
            orig_logits = norm_logits_window.get(layer_idx)
            raw_vec = raw_resid_window.get(layer_idx)
            if orig_logits is None or raw_vec is None:
                continue
            try:
                new_logits = _redecode_layer_logits(layer_idx, raw_vec)
            except Exception:
                continue
            if new_logits is None:
                continue
            try:
                orig_logits_f = orig_logits.to(dtype=torch.float32)
            except Exception:
                orig_logits_f = torch.tensor([])
            if orig_logits_f.numel() == 0:
                continue
            new_logits_f = new_logits.to(dtype=torch.float32)
            layers_count += 1
            orig_top1 = int(torch.argmax(orig_logits_f).item())
            new_top1 = int(torch.argmax(new_logits_f).item())
            if new_top1 != orig_top1:
                flips += 1
            try:
                rank = int((new_logits_f > new_logits_f[orig_top1]).sum().item()) + 1
            except Exception:
                rank = 1
            rank_devs.append(abs(rank - 1))

        if layers_count == 0:
            result = {"status": "unavailable"}
            if det_env:
                result["deterministic_algorithms"] = True
            return result

        max_dev = max(rank_devs) if rank_devs else 0.0
        p95_dev = _percentile(rank_devs, 0.95) if rank_devs else 0.0
        flip_rate = float(flips) / float(layers_count)

        result = {
            "status": "ok",
            "layers_checked": layers_count,
            "max_rank_dev": float(max_dev),
            "p95_rank_dev": float(p95_dev if p95_dev is not None else 0.0),
            "top1_flip_rate": float(flip_rate),
        }
        if det_env:
            result["deterministic_algorithms"] = True
        return result

    p_uniform: Optional[float] = None

    with torch.no_grad():
        residual_cache: Dict[str, Any] = {}
        cache_hook = build_cache_hook(residual_cache)
        hooks, has_pos_embed = attach_residual_hooks(model, cache_hook)
        try:
            tokens = model.to_tokens(context_prompt)
            str_tokens = model.to_str_tokens(context_prompt)
            logits = model(tokens)

            final_logits = logits[0, -1, :].float()
            final_probs = torch.softmax(final_logits, dim=0)
            if unembed_ctx.b is not None:
                try:
                    final_logits_bias_free = final_logits - unembed_ctx.b.to(
                        device=final_logits.device,
                        dtype=final_logits.dtype,
                    )
                except Exception:
                    final_logits_bias_free = final_logits
            else:
                final_logits_bias_free = final_logits
            _final_norm = torch.norm(final_logits_bias_free) + 1e-12
            final_dir = (final_logits_bias_free / _final_norm)
            vocab_size = int(final_logits.shape[-1])
            p_uniform = None
            if vocab_size > 0:
                p_uniform = 1.0 / float(vocab_size)

            final_logits_all = logits[0].detach().to(dtype=torch.float32)
            final_probs_by_pos: Dict[int, torch.Tensor] = {}
            positional_target_indices: Set[int] = set()
            pos_frac_lookup: Dict[int, float] = {}
            if audit_data is not None:
                seq_len = tokens.shape[1]
                if seq_len <= 0:
                    audit_data = None
                    if tuned_spec is not None:
                        tuned_spec["audit_data"] = None
                else:
                    max_index = max(0, seq_len - 1)
                    pos_map: Dict[int, float] = {}
                    for frac in POS_GRID_FRACTIONS:
                        if not math.isfinite(frac):
                            continue
                        idx = int(round(max_index * float(frac)))
                        idx = max(0, min(max_index, idx))
                        pos_map.setdefault(idx, float(frac))
                    pos_grid_entries = [
                        {"pos_index": idx, "pos_frac": frac}
                        for idx, frac in sorted(pos_map.items())
                    ]
                    audit_data["pos_grid"] = pos_grid_entries
                    pos_frac_lookup = {entry["pos_index"]: entry["pos_frac"] for entry in pos_grid_entries}
                    positional_target_indices = {entry["pos_index"] for entry in pos_grid_entries}
                    for idx in positional_target_indices:
                        final_probs_by_pos[idx] = torch.softmax(final_logits_all[idx], dim=0)
            else:
                pos_grid_entries: List[Dict[str, Any]] = []

            # ---- Layer 0 (embeddings) ----
            if has_pos_embed:
                resid_raw = residual_cache['hook_embed'] + residual_cache['hook_pos_embed']
            else:
                resid_raw = residual_cache['hook_embed']

            # Compute norm-lens logits for L0 from the raw residual
            norm_logits_all = norm_lens.forward(
                model,
                0,
                resid_raw,
                probe_after_block=False,
                W_U=unembed_ctx.W,
                b_U=unembed_ctx.b,
                force_fp32_unembed=unembed_ctx.force_fp32,
                cache=unembed_ctx.cache,
            )
            tuned_logits_all_L0 = None
            if tuned_adapter is not None:
                tuned_logits_all_L0 = tuned_adapter.forward(
                    model,
                    0,
                    resid_raw,
                    probe_after_block=False,
                    W_U=unembed_ctx.W,
                    b_U=unembed_ctx.b,
                    force_fp32_unembed=unembed_ctx.force_fp32,
                    cache=tuned_cache,
                )
            # Pass-wide Prism enablement is decided at L0 and carried forward
            # to match baseline behavior (no per-layer re-enabling attempts).
            prism_lens = PrismLensAdapter(prism_ctx.stats, prism_ctx.Q, prism_ctx.active)
            diag_delta: Dict[str, Any] = {}
            prism_logits_all_L0 = None
            if prism_lens.enabled:
                prism_logits_all_L0 = prism_lens.forward(
                    model,
                    0,
                    resid_raw,
                    probe_after_block=False,
                    W_U=unembed_ctx.W,
                    b_U=unembed_ctx.b,
                    force_fp32_unembed=unembed_ctx.force_fp32,
                    cache=unembed_ctx.cache,
                )
                if prism_lens.diag.get("placement_error"):
                    err = prism_lens.diag["placement_error"]
                    diag_delta["placement_error"] = err
                    # Mirror into context once for completeness
                    if getattr(prism_ctx, "placement_error", None) is None:
                        prism_ctx.placement_error = err

            # Per-position records at L0
            last_pos = tokens.shape[1] - 1
            norm_module_pre = get_correct_norm_module(model, 0, probe_after_block=False)
            resid_norm_pre = apply_norm_or_skip(resid_raw, norm_module_pre)
            raw_vec_L0 = resid_raw[0, last_pos, :].detach()
            geom_vec_norm_pre = resid_norm_pre[0, last_pos, :].detach()
            try:
                raw_resid_window[0] = raw_vec_L0.cpu()
            except Exception:
                pass

            ratio0, cos0, raw_norm0 = _norm_effect_metrics(raw_vec_L0, geom_vec_norm_pre)
            ln_source0, eps_inside0, scale_used0 = describe_norm_origin(model, 0, probe_after_block=False)
            norm_provenance_entries.append({
                "layer": 0,
                "ln_source": ln_source0,
                "eps_inside_sqrt": bool(eps_inside0),
                "scale_gamma_used": bool(scale_used0),
                "resid_norm_ratio": ratio0,
                "delta_resid_cos": cos0,
                "raw_resid_norm": raw_norm0,
            })
            block_label0 = "blocks[0]" if getattr(model, "blocks", []) else "blocks[0]"
            layer_map_entries.append({
                "layer": 0,
                "block": block_label0,
                "stream": "pre_block",
                "norm": ln_source0,
            })
            normalization_effects.append({
                "layer": 0,
                "resid_norm_ratio": ratio0,
                "delta_resid_cos": cos0,
                "raw_resid_norm": raw_norm0,
            })
            layer0_logits = norm_logits_all[last_pos]
            layer0_probs = torch.softmax(layer0_logits, dim=0)
            _update_numeric(0, raw_vec_L0, geom_vec_norm_pre, layer0_logits, layer0_probs)

            for pos in range(tokens.shape[1]):
                layer_logits = norm_logits_all[pos]
                entropy_bits = bits_entropy_from_logits(layer_logits)
                token_str = str_tokens[pos]
                verbose = _is_verbose_position(pos, token_str, tokens.shape[1], important_words)
                k = top_k_verbose if verbose else top_k_record
                k_eff = min(int(k), int(layer_logits.shape[-1]))
                _, top_indices_k = torch.topk(layer_logits, k_eff, largest=True, sorted=True)
                full_probs = torch.softmax(layer_logits, dim=0)
                top_probs_k = full_probs[top_indices_k]
                top_tokens_k = [decode_id_fn(idx) for idx in top_indices_k]
                rec = make_record(
                    prompt_id=prompt_id,
                    prompt_variant=prompt_variant,
                    layer=0,
                    pos=pos,
                    token=token_str,
                    entropy=entropy_bits,
                    top_tokens=top_tokens_k,
                    top_probs=top_probs_k,
                    extra=fact_extra,
                )
                json_data["records"].append(rec)
                # Emit Prism per-position record at L0 for sidecar parity
                if prism_logits_all_L0 is not None:
                    append_prism_record(
                        json_data_prism,
                        prompt_id=prompt_id,
                        prompt_variant=prompt_variant,
                        layer=0,
                        pos=pos,
                        token=token_str,
                        logits_pos=prism_logits_all_L0[pos],
                        decode_id_fn=decode_id_fn,
                        top_k=k_eff,
                        extra=fact_extra,
                    )

                if tuned_logits_all_L0 is not None and tuned_json_data is not None and tuned_window_manager is not None:
                    tuned_layer_logits = tuned_logits_all_L0[pos]
                    tuned_entropy_bits = bits_entropy_from_logits(tuned_layer_logits)
                    tuned_full_probs = torch.softmax(tuned_layer_logits, dim=0)
                    _, tuned_top_indices = torch.topk(tuned_layer_logits, k_eff, largest=True, sorted=True)
                    tuned_top_tokens = [decode_id_fn(idx) for idx in tuned_top_indices]
                    tuned_top_probs = tuned_full_probs[tuned_top_indices]
                    tuned_json_data["records"].append(
                        make_record(
                            prompt_id=prompt_id,
                            prompt_variant=prompt_variant,
                            layer=0,
                            pos=pos,
                            token=token_str,
                            entropy=tuned_entropy_bits,
                            top_tokens=tuned_top_tokens,
                            top_probs=tuned_top_probs,
                            extra=fact_extra,
                        )
                    )

            # Pure next-token (L0)
            collected_pure_records = []
            view, collected, dual_ctx = compute_pure_next_token_info(
                layer_out_idx=0,
                logits_all=norm_logits_all,
                tokens_tensor=tokens,
                ctx_ids_list=ctx_ids_list,
                window_manager=window_manager,
                lens_type="norm",
                final_probs_tensor=final_probs,
                first_ans_token_id=first_ans_token_id,
                final_dir_vec=final_dir,
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
                prompt_vocab_ids=prompt_vocab_ids,
                decoder_weight=decoder_weight,
                geom_vec=geom_vec_norm_pre,
                topk_prompt_mass_k=TOPK_PROMPT_MASS_K,
                geom_gamma=GEOM_GAMMA,
                norm_temp_tau=_tau_for(0),
                copy_strict_thresholds=copy_strict_thresholds,
                bias_tensor=unembed_ctx.b,
                raw_resid_vec=raw_vec_L0,
                norm_resid_vec=geom_vec_norm_pre,
                p_uniform=p_uniform,
                semantic_margin_delta=SEMANTIC_MARGIN_DELTA,
            )
            combined_extra = dict(fact_extra)
            combined_extra.update(view["record_extra"])
            json_data["pure_next_token_records"].append(
                make_pure_record(
                    prompt_id=prompt_id,
                    prompt_variant=prompt_variant,
                    layer=0,
                    pos=view["pos"],
                    token=view["token_str"],
                    entropy=view["entropy_bits"],
                    top_tokens=view["top_tokens"],
                    top_probs=view["top_probs"],
                    extra=combined_extra,
                )
            )
            collected_pure_records.append(collected)
            try:
                norm_logits_window[0] = dual_ctx["last_logits_norm"].detach().cpu()
            except Exception:
                pass
            collected_by_layer[0] = collected

            if tuned_logits_all_L0 is not None and tuned_json_data is not None and tuned_window_manager is not None:
                tuned_view, tuned_collected, _ = compute_pure_next_token_info(
                    layer_out_idx=0,
                    logits_all=tuned_logits_all_L0,
                    tokens_tensor=tokens,
                    ctx_ids_list=ctx_ids_list,
                    window_manager=tuned_window_manager,
                    lens_type="tuned",
                    final_probs_tensor=final_probs,
                    first_ans_token_id=first_ans_token_id,
                    final_dir_vec=final_dir,
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
                    prompt_vocab_ids=prompt_vocab_ids,
                    decoder_weight=decoder_weight,
                    geom_vec=None,
                    topk_prompt_mass_k=TOPK_PROMPT_MASS_K,
                    geom_gamma=GEOM_GAMMA,
                    norm_temp_tau=None,
                    copy_strict_thresholds=(),
                    bias_tensor=unembed_ctx.b,
                )
                tuned_extra = dict(fact_extra)
                tuned_extra.update(tuned_view["record_extra"])
                tuned_json_data["pure_next_token_records"].append(
                    make_pure_record(
                        prompt_id=prompt_id,
                        prompt_variant=prompt_variant,
                        layer=0,
                        pos=tuned_view["pos"],
                        token=tuned_view["token_str"],
                        entropy=tuned_view["entropy_bits"],
                        top_tokens=tuned_view["top_tokens"],
                        top_probs=tuned_view["top_probs"],
                        extra=tuned_extra,
                    )
                )
                tuned_collected_records.append(tuned_collected)
                try:
                    tuned_logits_window[0] = tuned_logits_all_L0[last_pos].detach().cpu()
                except Exception:
                    pass

            # Optional raw-vs-norm sample at L0
            if enable_raw_lens_sampling and RAW_LENS_MODE != "off":
                record_dual_lens_sample(
                    json_data["raw_lens_check"],
                    layer_out_idx=dual_ctx["layer"],
                    last_logits_norm=dual_ctx["last_logits_norm"],
                    resid_raw_last_vec=resid_raw[0, int(dual_ctx["last_pos"]), :],
                    W_U=unembed_ctx.W,
                    b_U=unembed_ctx.b,
                    force_fp32_unembed=unembed_ctx.force_fp32,
                    tokenizer=model.tokenizer,
                    final_probs=dual_ctx["final_probs"],
                    first_ans_id=dual_ctx["first_ans_id"],
                    ground_truth=dual_ctx["ground_truth"],
                )

            if prism_logits_all_L0 is not None:
                append_prism_pure_next_token(
                    json_data_prism,
                    layer_out_idx=0,
                    prism_logits_all=prism_logits_all_L0,
                    tokens_tensor=tokens,
                    ctx_ids_list=ctx_ids_list,
                    window_manager=window_manager,
                    final_probs_tensor=final_probs,
                    first_ans_token_id=first_ans_token_id,
                    final_dir_vec=final_dir,
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
                    p_uniform=p_uniform,
                    semantic_margin_delta=SEMANTIC_MARGIN_DELTA,
                    extra=fact_extra,
                )

            # Save residual if requested. Behavior-preserving policy:
            # - If Prism is enabled, save the normalized residual for this probe point
            # - Otherwise, save the raw residual
            if keep_residuals:
                name_root = clean_model_name or model.cfg.__dict__.get("model_name", "model")
                resid_filename = f"{name_root}_00_resid.pt"
                resid_path = os.path.join(out_dir or os.getcwd(), resid_filename)
                try:
                    if prism_lens.enabled:
                        _nm = get_correct_norm_module(model, 0, probe_after_block=False)
                        resid_to_save = apply_norm_or_skip(resid_raw, _nm)
                    else:
                        resid_to_save = resid_raw
                    torch.save(resid_to_save.to(dtype=resid_to_save.dtype).cpu(), resid_path)
                except Exception:
                    # Best-effort save; do not fail the pass on save errors
                    pass

            # ---- Post-block layers ----
            detected_architecture = detect_model_architecture(model)
            n_layers = model.cfg.n_layers

            def _should_sample(one_indexed: int) -> bool:
                return should_sample_layer(RAW_LENS_MODE, n_layers, one_indexed)

            for layer in range(n_layers):
                resid_raw = get_residual_safely(residual_cache, layer)

                norm_module_post = get_correct_norm_module(model, layer, probe_after_block=True)
                resid_norm_post = apply_norm_or_skip(resid_raw, norm_module_post)
                raw_vec_layer = resid_raw[0, last_pos, :].detach()
                geom_vec_norm_post = resid_norm_post[0, last_pos, :].detach()
                try:
                    raw_resid_window[layer + 1] = raw_vec_layer.cpu()
                except Exception:
                    pass

                ratio_layer, cos_layer, raw_norm_layer = _norm_effect_metrics(raw_vec_layer, geom_vec_norm_post)
                ln_source_layer, eps_inside_layer, scale_used_layer = describe_norm_origin(
                    model, layer, probe_after_block=True
                )
                norm_provenance_entries.append({
                    "layer": layer + 1,
                    "ln_source": ln_source_layer,
                    "eps_inside_sqrt": bool(eps_inside_layer),
                    "scale_gamma_used": bool(scale_used_layer),
                    "resid_norm_ratio": ratio_layer,
                    "delta_resid_cos": cos_layer,
                    "raw_resid_norm": raw_norm_layer,
                })
                block_label = f"blocks[{layer}]"
                stream_label = "post_block"
                if ln_source_layer == "ln_final":
                    block_label = "final"
                    stream_label = "unembed_head"
                layer_map_entries.append({
                    "layer": layer + 1,
                    "block": block_label,
                    "stream": stream_label,
                    "norm": ln_source_layer,
                })
                normalization_effects.append({
                    "layer": layer + 1,
                    "resid_norm_ratio": ratio_layer,
                    "delta_resid_cos": cos_layer,
                    "raw_resid_norm": raw_norm_layer,
                })

                logits_all = norm_lens.forward(
                    model,
                    layer,
                    resid_raw,
                    probe_after_block=True,
                    W_U=unembed_ctx.W,
                    b_U=unembed_ctx.b,
                    force_fp32_unembed=unembed_ctx.force_fp32,
                    cache=unembed_ctx.cache,
                )
                layer_logits_last = logits_all[last_pos]
                layer_probs_last = torch.softmax(layer_logits_last, dim=0)
                _update_numeric(layer + 1, raw_vec_layer, geom_vec_norm_post, layer_logits_last, layer_probs_last)

                tuned_logits_all = None
                rot_logits_all = None
                temp_only_logits_all = None
                tau_value: Optional[float] = None
                geom_vec_tuned_post = None
                if tuned_adapter is not None:
                    tuned_logits_all = tuned_adapter.forward(
                        model,
                        layer,
                        resid_raw,
                        probe_after_block=True,
                        W_U=unembed_ctx.W,
                        b_U=unembed_ctx.b,
                        force_fp32_unembed=unembed_ctx.force_fp32,
                        cache=tuned_cache,
                    )
                    try:
                        tau_tensor = tuned_adapter.translator.temperature(layer)
                        tau_value = float(tau_tensor.detach().cpu().item())
                        if tau_value <= 0 or not math.isfinite(tau_value):
                            tau_value = None
                    except Exception:
                        tau_value = None
                    translated_seq = None
                    try:
                        translated_seq = tuned_adapter.translator(
                            resid_norm_post[0, :, :], layer
                        )
                        geom_vec_tuned_post = translated_seq[last_pos, :].detach()
                    except Exception:
                        translated_seq = None
                        geom_vec_tuned_post = None
                    if translated_seq is not None:
                        try:
                            translated_cast = safe_cast_for_unembed(
                                translated_seq,
                                unembed_ctx.W,
                                force_fp32_unembed=unembed_ctx.force_fp32,
                            )
                            rot_logits_all = unembed_mm(
                                translated_cast,
                                unembed_ctx.W,
                                unembed_ctx.b,
                                cache=tuned_cache,
                            ).float()
                        except Exception:
                            rot_logits_all = None
                    if tau_value is not None and tau_value > 0:
                        try:
                            temp_only_logits_all = (logits_all / float(tau_value)).to(dtype=torch.float32)
                        except Exception:
                            temp_only_logits_all = None
                    else:
                        try:
                            temp_only_logits_all = logits_all.to(dtype=torch.float32)
                        except Exception:
                            temp_only_logits_all = logits_all

                # Prism sidecar logits via adapter for per-position record emission
                prism_logits_all = None
                if prism_lens.enabled:
                    prism_logits_all = prism_lens.forward(
                        model,
                        layer,
                        resid_raw,
                        probe_after_block=True,
                        W_U=unembed_ctx.W,
                        b_U=unembed_ctx.b,
                        force_fp32_unembed=unembed_ctx.force_fp32,
                        cache=unembed_ctx.cache,
                    )
                    if prism_lens.diag.get("placement_error"):
                        err = prism_lens.diag["placement_error"]
                        diag_delta.setdefault("placement_error", err)  # keep first occurrence
                        if getattr(prism_ctx, "placement_error", None) is None:
                            prism_ctx.placement_error = err

                # Per-position records for this layer
                for pos in range(tokens.shape[1]):
                    layer_logits = logits_all[pos]
                    full_probs = torch.softmax(layer_logits, dim=0)
                    entropy_bits = bits_entropy_from_logits(layer_logits)
                    token_str = str_tokens[pos]
                    verbose = _is_verbose_position(pos, token_str, tokens.shape[1], important_words)
                    k = top_k_verbose if verbose else top_k_record
                    k_eff = min(int(k), int(layer_logits.shape[-1]))
                    _, top_indices_k = torch.topk(layer_logits, k_eff, largest=True, sorted=True)
                    top_probs_k = full_probs[top_indices_k]
                    top_tokens_k = [decode_id_fn(idx) for idx in top_indices_k]
                    json_data["records"].append(
                        make_record(
                            prompt_id=prompt_id,
                            prompt_variant=prompt_variant,
                            layer=layer + 1,
                            pos=pos,
                            token=token_str,
                            entropy=entropy_bits,
                            top_tokens=top_tokens_k,
                            top_probs=top_probs_k,
                            extra=fact_extra,
                        )
                    )
                    if prism_logits_all is not None:
                        append_prism_record(
                            json_data_prism,
                            prompt_id=prompt_id,
                            prompt_variant=prompt_variant,
                            layer=layer + 1,
                            pos=pos,
                            token=token_str,
                            logits_pos=prism_logits_all[pos],
                            decode_id_fn=decode_id_fn,
                            top_k=k_eff,
                            extra=fact_extra,
                        )

                    if tuned_logits_all is not None and tuned_json_data is not None and tuned_window_manager is not None:
                        tuned_layer_logits = tuned_logits_all[pos]
                        tuned_entropy_bits = bits_entropy_from_logits(tuned_layer_logits)
                        tuned_full_probs = torch.softmax(tuned_layer_logits, dim=0)
                        _, tuned_top_indices = torch.topk(tuned_layer_logits, k_eff, largest=True, sorted=True)
                        tuned_top_tokens = [decode_id_fn(idx) for idx in tuned_top_indices]
                        tuned_top_probs = tuned_full_probs[tuned_top_indices]
                        tuned_json_data["records"].append(
                            make_record(
                                prompt_id=prompt_id,
                                prompt_variant=prompt_variant,
                                layer=layer + 1,
                                pos=pos,
                                token=token_str,
                                entropy=tuned_entropy_bits,
                                top_tokens=tuned_top_tokens,
                                top_probs=tuned_top_probs,
                                extra=fact_extra,
                            )
                        )

                        if (
                            audit_data is not None
                            and pos in positional_target_indices
                        ):
                            final_probs_pos = final_probs_by_pos.get(pos)
                            pos_frac = pos_frac_lookup.get(pos)
                            if final_probs_pos is not None and pos_frac is not None:
                                ans_id_pos = first_ans_token_id if pos == last_pos else None
                                base_metrics_pos, entropy_base_pos, _ = _metrics_for_logits(
                                    layer_logits,
                                    final_probs_pos,
                                    ans_id_pos,
                                )
                                tuned_metrics_pos, entropy_tuned_pos, _ = _metrics_for_logits(
                                    tuned_layer_logits,
                                    final_probs_pos,
                                    ans_id_pos,
                                )
                                kl_base_pos = _float_or_none(base_metrics_pos.get("kl_to_final_bits"))
                                kl_tuned_pos = _float_or_none(tuned_metrics_pos.get("kl_to_final_bits"))
                                delta_pos = None
                                if kl_base_pos is not None and kl_tuned_pos is not None:
                                    delta_pos = kl_base_pos - kl_tuned_pos
                                rank_base_pos = _int_or_none(base_metrics_pos.get("answer_rank"))
                                rank_tuned_pos = _int_or_none(tuned_metrics_pos.get("answer_rank"))
                                pos_row = {
                                    "pos_frac": pos_frac,
                                    "pos_index": pos,
                                    "layer": layer + 1,
                                    "kl_bits_baseline": kl_base_pos,
                                    "kl_bits_tuned": kl_tuned_pos,
                                    "delta_kl_bits_tuned": delta_pos,
                                    "answer_rank_baseline": rank_base_pos,
                                    "answer_rank_tuned": rank_tuned_pos,
                                    "rank_shift_tuned": _rank_shift(rank_base_pos, rank_tuned_pos),
                                    "entropy_bits_baseline": float(entropy_base_pos),
                                    "entropy_bits_tuned": float(entropy_tuned_pos),
                                }
                                audit_data["positional_rows"].append(pos_row)

                # Pure next-token for this layer
                view, collected, dual_ctx = compute_pure_next_token_info(
                    layer_out_idx=layer + 1,
                    logits_all=logits_all,
                    tokens_tensor=tokens,
                    ctx_ids_list=ctx_ids_list,
                    window_manager=window_manager,
                    lens_type="norm",
                    final_probs_tensor=final_probs,
                    first_ans_token_id=first_ans_token_id,
                    final_dir_vec=final_dir,
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
                    prompt_vocab_ids=prompt_vocab_ids,
                    decoder_weight=decoder_weight,
                    geom_vec=geom_vec_norm_post,
                    topk_prompt_mass_k=TOPK_PROMPT_MASS_K,
                    geom_gamma=GEOM_GAMMA,
                    norm_temp_tau=_tau_for(layer + 1),
                    copy_strict_thresholds=copy_strict_thresholds,
                    bias_tensor=unembed_ctx.b,
                    raw_resid_vec=raw_vec_layer,
                    norm_resid_vec=geom_vec_norm_post,
                    p_uniform=p_uniform,
                    semantic_margin_delta=SEMANTIC_MARGIN_DELTA,
                )
                layer_extra = dict(fact_extra)
                layer_extra.update(view["record_extra"])
                json_data["pure_next_token_records"].append(
                    make_pure_record(
                        prompt_id=prompt_id,
                        prompt_variant=prompt_variant,
                        layer=layer + 1,
                        pos=view["pos"],
                        token=view["token_str"],
                        entropy=view["entropy_bits"],
                        top_tokens=view["top_tokens"],
                        top_probs=view["top_probs"],
                        extra=layer_extra,
                    )
                )
                collected_pure_records.append(collected)
                try:
                    norm_logits_window[layer + 1] = dual_ctx["last_logits_norm"].detach().cpu()
                except Exception:
                    pass
                collected_by_layer[layer + 1] = collected

                if tuned_logits_all is not None and tuned_json_data is not None and tuned_window_manager is not None:
                    tuned_view, tuned_collected, _ = compute_pure_next_token_info(
                        layer_out_idx=layer + 1,
                        logits_all=tuned_logits_all,
                        tokens_tensor=tokens,
                        ctx_ids_list=ctx_ids_list,
                        window_manager=tuned_window_manager,
                        lens_type="tuned",
                        final_probs_tensor=final_probs,
                        first_ans_token_id=first_ans_token_id,
                        final_dir_vec=final_dir,
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
                        prompt_vocab_ids=prompt_vocab_ids,
                        decoder_weight=decoder_weight,
                        geom_vec=geom_vec_tuned_post,
                        topk_prompt_mass_k=TOPK_PROMPT_MASS_K,
                        geom_gamma=GEOM_GAMMA,
                        norm_temp_tau=None,
                        bias_tensor=unembed_ctx.b,
                        p_uniform=p_uniform,
                        semantic_margin_delta=SEMANTIC_MARGIN_DELTA,
                    )
                    tuned_layer_extra = dict(fact_extra)
                    tuned_layer_extra.update(tuned_view["record_extra"])
                    tuned_json_data["pure_next_token_records"].append(
                        make_pure_record(
                            prompt_id=prompt_id,
                            prompt_variant=prompt_variant,
                            layer=layer + 1,
                            pos=tuned_view["pos"],
                            token=tuned_view["token_str"],
                            entropy=tuned_view["entropy_bits"],
                            top_tokens=tuned_view["top_tokens"],
                            top_probs=tuned_view["top_probs"],
                            extra=tuned_layer_extra,
                        )
                    )
                    tuned_collected_records.append(tuned_collected)
                    try:
                        tuned_logits_window[layer + 1] = tuned_logits_all[last_pos].detach().cpu()
                    except Exception:
                        pass

                    if audit_data is not None and tuned_logits_all is not None:
                        kl_baseline = _float_or_none(collected.get("kl_to_final_bits"))
                        kl_tuned = _float_or_none(tuned_collected.get("kl_to_final_bits"))
                        entropy_baseline = _float_or_none(collected.get("entropy_bits"))
                        entropy_tuned = _float_or_none(tuned_collected.get("entropy_bits"))
                        rank_baseline = _int_or_none(collected.get("answer_rank"))
                        rank_tuned = _int_or_none(tuned_collected.get("answer_rank"))

                        rot_metrics = None
                        rot_entropy = None
                        if rot_logits_all is not None:
                            try:
                                rot_vec = rot_logits_all[last_pos]
                                rot_metrics, rot_entropy, _ = _metrics_for_logits(
                                    rot_vec,
                                    final_probs,
                                    first_ans_token_id,
                                )
                            except Exception:
                                rot_metrics = None
                                rot_entropy = None

                        temp_metrics = None
                        temp_entropy = None
                        if temp_only_logits_all is not None:
                            try:
                                temp_vec = temp_only_logits_all[last_pos]
                                temp_metrics, temp_entropy, _ = _metrics_for_logits(
                                    temp_vec,
                                    final_probs,
                                    first_ans_token_id,
                                )
                            except Exception:
                                temp_metrics = None
                                temp_entropy = None

                        def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
                            if a is None or b is None:
                                return None
                            return a - b

                        kl_rot = _float_or_none((rot_metrics or {}).get("kl_to_final_bits"))
                        kl_temp = _float_or_none((temp_metrics or {}).get("kl_to_final_bits"))
                        rank_rot = _int_or_none((rot_metrics or {}).get("answer_rank"))
                        rank_temp = _int_or_none((temp_metrics or {}).get("answer_rank"))

                        delta_tuned = _delta(kl_baseline, kl_tuned)
                        delta_rot = _delta(kl_baseline, kl_rot)
                        delta_temp = _delta(kl_baseline, kl_temp)
                        delta_interaction = None
                        if delta_tuned is not None and delta_rot is not None and delta_temp is not None:
                            delta_interaction = delta_tuned - (delta_rot + delta_temp)

                        variant_row = {
                            "layer": layer + 1,
                            "kl_bits_baseline": kl_baseline,
                            "kl_bits_tuned": kl_tuned,
                            "kl_bits_rot_only": kl_rot,
                            "kl_bits_temp_only": kl_temp,
                            "delta_kl_bits_tuned": delta_tuned,
                            "delta_kl_bits_rot_only": delta_rot,
                            "delta_kl_bits_temp_only": delta_temp,
                            "delta_kl_bits_interaction": delta_interaction,
                            "answer_rank_baseline": rank_baseline,
                            "answer_rank_tuned": rank_tuned,
                            "answer_rank_rot_only": rank_rot,
                            "answer_rank_temp_only": rank_temp,
                            "rank_shift_tuned": _rank_shift(rank_baseline, rank_tuned),
                            "rank_shift_rot_only": _rank_shift(rank_baseline, rank_rot),
                            "rank_shift_temp_only": _rank_shift(rank_baseline, rank_temp),
                            "entropy_bits_baseline": entropy_baseline,
                            "entropy_bits_tuned": entropy_tuned,
                        }
                        audit_data["variant_rows"].append(variant_row)

                        if layer == (n_layers - 1):
                            try:
                                tuned_logits_last = tuned_logits_all[last_pos].detach().to(device="cpu", dtype=torch.float32)
                            except Exception:
                                tuned_logits_last = None
                            audit_data["head_mismatch"] = {
                                "kl_bits_tuned_final": kl_tuned,
                                "tuned_logits_last": tuned_logits_last,
                                "final_probs": final_probs.detach().to(device="cpu", dtype=torch.float32),
                            }

                # Prism pure next-token row for this layer, if available
                if prism_logits_all is not None:
                    append_prism_pure_next_token(
                        json_data_prism,
                        layer_out_idx=layer + 1,
                        prism_logits_all=prism_logits_all,
                        tokens_tensor=tokens,
                        ctx_ids_list=ctx_ids_list,
                        window_manager=window_manager,
                        final_probs_tensor=final_probs,
                        first_ans_token_id=first_ans_token_id,
                        final_dir_vec=final_dir,
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
                    p_uniform=p_uniform,
                    semantic_margin_delta=SEMANTIC_MARGIN_DELTA,
                    extra=fact_extra,
                )

                # Optional raw-vs-norm sample
                if enable_raw_lens_sampling and RAW_LENS_MODE != "off" and _should_sample(layer + 1):
                    record_dual_lens_sample(
                        json_data["raw_lens_check"],
                        layer_out_idx=dual_ctx["layer"],
                        last_logits_norm=dual_ctx["last_logits_norm"],
                        resid_raw_last_vec=resid_raw[0, int(dual_ctx["last_pos"]), :],
                        W_U=unembed_ctx.W,
                        b_U=unembed_ctx.b,
                        force_fp32_unembed=unembed_ctx.force_fp32,
                        tokenizer=model.tokenizer,
                        final_probs=dual_ctx["final_probs"],
                        first_ans_id=dual_ctx["first_ans_id"],
                        ground_truth=dual_ctx["ground_truth"],
                    )

                # Last layer consistency snapshot
                if layer == (n_layers - 1):
                    last_pos = tokens.shape[1] - 1
                    last_logits = logits_all[last_pos]
                    final_top1_id = int(torch.argmax(final_probs).item())
                    last_layer_consistency = compute_last_layer_consistency(
                        last_logits=last_logits,
                        final_probs=final_probs,
                        final_top1_id=final_top1_id,
                        first_ans_id=first_ans_token_id,
                        head_scale_cfg=head_scale_cfg,
                        head_softcap_cfg=head_softcap_cfg,
                        topk_cum=5,
                    )

                # Save residual per layer if requested (policy mirrors L0)
                if keep_residuals:
                    name_root = clean_model_name or model.cfg.__dict__.get("model_name", "model")
                    resid_filename = f"{name_root}_{layer+1:02d}_resid.pt"
                    save_dtype = getattr(getattr(model, 'cfg', object()), 'dtype', torch.float32)
                    resid_path = os.path.join(out_dir or os.getcwd(), resid_filename)
                    try:
                        if prism_lens.enabled:
                            _nm = get_correct_norm_module(model, layer, probe_after_block=True)
                            _normed = apply_norm_or_skip(resid_raw, _nm)
                            resid_to_save = _normed
                        else:
                            resid_to_save = resid_raw
                        try:
                            resid_to_save_tensor = resid_to_save.to(dtype=save_dtype).cpu()
                        except Exception:
                            resid_to_save_tensor = resid_to_save.float().cpu()
                        torch.save(resid_to_save_tensor, resid_path)
                    except Exception:
                        # Best-effort save
                        pass

                # Free layer residual and keep memory flat
                try:
                    del residual_cache[f'blocks.{layer}.hook_resid_post']
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                import gc as _gc
                _gc.collect()

            # Summarize this pass
            summary_diag = summarize_pure_records(
                collected_pure_records,
                copy_threshold=copy_threshold,
                copy_window_k=getattr(window_manager, "window_k", 1),
                copy_soft_threshold=copy_soft_threshold,
                copy_soft_window_ks=copy_soft_window_ks,
                copy_match_level="id_subsequence",
                lens_tag="norm",
                surface_delta=SURFACE_DELTA,
                geom_gamma=GEOM_GAMMA,
                topk_prompt_tau=TOPK_DECAY_TAU,
                n_layers=int(model.cfg.n_layers),
                semantic_margin_delta=SEMANTIC_MARGIN_DELTA,
                p_uniform=p_uniform,
            )
            summary_diag.setdefault("answer_margin_unit", "logit")

            # Gate-stability under small temperature rescalings (PLAN §1.50)
            # Only for the primary positive/original pass; no extra forwards.
            try:
                if prompt_id == "pos" and prompt_variant == "orig" and isinstance(first_ans_token_id, int):
                    scales = [0.90, 0.95, 1.05, 1.10]

                    def _layer_gate_pass_fractions(layer_idx: int) -> Optional[Dict[str, float]]:
                        if not isinstance(layer_idx, int):
                            return None
                        z = norm_logits_window.get(layer_idx)
                        if z is None or z.numel() == 0:
                            return None
                        try:
                            z_f = z.detach().to(dtype=torch.float32)
                        except Exception:
                            z_f = z.float()
                        vocab = int(z_f.shape[-1]) if z_f.ndim == 1 else int(z_f.view(-1).shape[0])
                        # Uniform baseline
                        _p_uniform = p_uniform if (p_uniform is not None) else (1.0 / float(vocab) if vocab > 0 else None)
                        if _p_uniform is None:
                            return None
                        # Base runner-up logit among non-answer ids (conservative; hold fixed across scales)
                        ans_id = int(first_ans_token_id)
                        try:
                            # Build masked view to get the best competing logit
                            mask = torch.ones_like(z_f, dtype=torch.bool)
                            if 0 <= ans_id < z_f.shape[-1]:
                                mask[ans_id] = False
                            else:
                                return None
                            rival_vals = z_f[mask]
                            if rival_vals.numel() == 0:
                                return None
                            rival_max = float(torch.max(rival_vals).item())
                        except Exception:
                            return None
                        z_ans = float(z_f[ans_id].item())
                        base_gap = z_ans - rival_max
                        # Thresholds
                        delta_abs = float(SEMANTIC_MARGIN_DELTA)
                        delta_top2 = 0.5

                        um_pass = 0
                        t2_pass = 0
                        both_pass = 0
                        for s in scales:
                            try:
                                z_s = z_f * float(s)
                                P_s = torch.softmax(z_s, dim=0)
                                p_ans = float(P_s[ans_id].item())
                                um_ok = (p_ans - float(_p_uniform)) >= delta_abs
                                t2_ok = (float(s) * base_gap) >= delta_top2
                                if um_ok:
                                    um_pass += 1
                                if t2_ok:
                                    t2_pass += 1
                                if um_ok and t2_ok:
                                    both_pass += 1
                            except Exception:
                                # Count as fail on any numeric issue
                                pass
                        n = float(len(scales)) if scales else 1.0
                        return {
                            "uniform_margin_pass_frac": (float(um_pass) / n),
                            "top2_gap_pass_frac": (float(t2_pass) / n),
                            "both_gates_pass_frac": (float(both_pass) / n),
                        }

                    per_target: Dict[str, Dict[str, float] | Dict[str, Any]] = {}
                    # Targets available at this stage (confirmed may be added later in run.py)
                    L_sem = summary_diag.get("L_semantic")
                    if isinstance(L_sem, int):
                        fracs = _layer_gate_pass_fractions(L_sem)
                        if fracs is not None:
                            per_target["L_semantic_norm"] = {"layer": int(L_sem), **fracs}

                    sem_gate = summary_diag.get("semantic_gate") or {}
                    L_strong = sem_gate.get("L_semantic_strong")
                    if isinstance(L_strong, int):
                        fracs = _layer_gate_pass_fractions(L_strong)
                        if fracs is not None:
                            per_target["L_semantic_strong"] = {"layer": int(L_strong), **fracs}

                    L_run2 = sem_gate.get("L_semantic_strong_run2")
                    if isinstance(L_run2, int):
                        # Require both L and L+1 to pass at the same scale
                        fr_L = _layer_gate_pass_fractions(L_run2)
                        fr_L1 = _layer_gate_pass_fractions(L_run2 + 1)
                        if fr_L is not None and fr_L1 is not None:
                            # Combine conservatively: min of per-scale pass indicators → via fractions
                            # For independence of s, recompute with AND across scales
                            # Repeat the check using raw logits to ensure strict AND across L and L+1
                            z0 = norm_logits_window.get(L_run2)
                            z1 = norm_logits_window.get(L_run2 + 1)
                            if z0 is not None and z1 is not None:
                                try:
                                    z0f = z0.detach().float(); z1f = z1.detach().float()
                                except Exception:
                                    z0f = z0.float(); z1f = z1.float()
                                ans_id = int(first_ans_token_id)
                                mask0 = torch.ones_like(z0f, dtype=torch.bool); mask1 = torch.ones_like(z1f, dtype=torch.bool)
                                if 0 <= ans_id < z0f.shape[-1] and 0 <= ans_id < z1f.shape[-1]:
                                    mask0[ans_id] = False; mask1[ans_id] = False
                                    r0 = float(torch.max(z0f[mask0]).item()); r1 = float(torch.max(z1f[mask1]).item())
                                    base_gap0 = float(z0f[ans_id].item()) - r0
                                    base_gap1 = float(z1f[ans_id].item()) - r1
                                    delta_abs = float(SEMANTIC_MARGIN_DELTA)
                                    delta_top2 = 0.5
                                    # Uniform baseline (recompute if needed)
                                    try:
                                        vocab = int(z0f.shape[-1])
                                    except Exception:
                                        vocab = 0
                                    puni = p_uniform if (p_uniform is not None) else (1.0 / float(vocab) if vocab > 0 else None)
                                    if puni is None:
                                        fr_run2 = None
                                    else:
                                        both_both = 0
                                        for s in scales:
                                            try:
                                                P0 = torch.softmax(z0f * float(s), dim=0)
                                                P1 = torch.softmax(z1f * float(s), dim=0)
                                                um0 = float(P0[ans_id].item()) - float(puni) >= delta_abs
                                                um1 = float(P1[ans_id].item()) - float(puni) >= delta_abs
                                                t20 = float(s) * base_gap0 >= delta_top2
                                                t21 = float(s) * base_gap1 >= delta_top2
                                                if um0 and um1 and t20 and t21:
                                                    both_both += 1
                                            except Exception:
                                                pass
                                        n = float(len(scales)) if scales else 1.0
                                        fr_run2 = float(both_both) / n
                                else:
                                    fr_run2 = None
                            else:
                                fr_run2 = None
                            if fr_run2 is not None:
                                per_target["L_semantic_strong_run2"] = {"layer": int(L_run2),
                                                                         "uniform_margin_pass_frac": None,
                                                                         "top2_gap_pass_frac": None,
                                                                         "both_gates_pass_frac": fr_run2}

                    if per_target:
                        try:
                            min_both = None
                            for entry in per_target.values():
                                val = entry.get("both_gates_pass_frac") if isinstance(entry, dict) else None
                                if val is None:
                                    continue
                                min_both = val if min_both is None else min(min_both, float(val))
                        except Exception:
                            min_both = None
                        summary_diag["gate_stability_small_scale"] = {
                            "scales": scales,
                            "per_target": per_target,
                            "min_both_gates_pass_frac": min_both,
                        }
            except Exception:
                # Best-effort diagnostic; ignore on failure
                pass

            raw_summary_block = (json_data.get("raw_lens_check") or {}).get("summary") or {}
            radius = 4
            try:
                max_kl_sample = raw_summary_block.get("max_kl_norm_vs_raw_bits")
                if raw_summary_block.get("lens_artifact_risk") == "high" or (
                    max_kl_sample is not None and float(max_kl_sample) >= 1.0
                ):
                    radius = 8
            except Exception:
                radius = 4

            center_candidates: List[int] = []
            for key in ("L_semantic", "first_rank_le_5", "first_kl_below_1.0"):
                val = summary_diag.get(key)
                if isinstance(val, int):
                    center_candidates.append(val)
            strict_copy_layer = summary_diag.get("copy_detector", {}).get("strict", {}).get("L_copy_strict")
            if isinstance(strict_copy_layer, int):
                center_candidates.append(strict_copy_layer)
            soft_map = summary_diag.get("L_copy_soft") or {}
            if isinstance(soft_map, dict):
                soft_layers = [v for v in soft_map.values() if isinstance(v, int)]
                if soft_layers:
                    center_candidates.append(min(soft_layers))

            window_summary, window_records = compute_windowed_raw_norm(
                radius=radius,
                center_layers=center_candidates,
                norm_logits_map=norm_logits_window,
                raw_resid_map=raw_resid_window,
                collected_map=collected_by_layer,
                final_probs=final_probs.detach().float().cpu(),
                W_U=unembed_ctx.W,
                b_U=unembed_ctx.b,
                force_fp32_unembed=unembed_ctx.force_fp32,
                decode_id_fn=decode_id_fn,
                ctx_ids_list=ctx_ids_list,
                first_ans_token_id=first_ans_token_id,
                ground_truth=ground_truth,
                prompt_id=prompt_id,
                prompt_variant=prompt_variant,
                n_layers=int(model.cfg.n_layers),
            )
            if window_summary:
                summary_diag["raw_lens_window"] = window_summary
            if window_records:
                for row in window_records:
                    row.update(fact_extra)
                json_data.setdefault("raw_lens_window_records", []).extend(window_records)

            # Full dual-lens sweep across all available layers (001_LAYERS_BASELINE_PLAN §1.24)
            try:
                full_summary, full_rows = compute_full_raw_norm(
                    norm_logits_map=norm_logits_window,
                    raw_resid_map=raw_resid_window,
                    collected_map=collected_by_layer,
                    final_probs=final_probs.detach().float().cpu(),
                    W_U=unembed_ctx.W,
                    b_U=unembed_ctx.b,
                    force_fp32_unembed=unembed_ctx.force_fp32,
                    decode_id_fn=decode_id_fn,
                    ctx_ids_list=ctx_ids_list,
                    first_ans_token_id=first_ans_token_id,
                    ground_truth=ground_truth,
                    prompt_id=prompt_id,
                    prompt_variant=prompt_variant,
                    n_layers=int(model.cfg.n_layers),
                )
                if full_summary:
                    summary_diag["raw_lens_full"] = full_summary
                if full_rows:
                    for row in full_rows:
                        row.update(fact_extra)
                    json_data.setdefault("raw_lens_full_records", []).extend(full_rows)
                    overlap_info = (full_summary or {}).get("topk_overlap") or {}
                    per_layer_cross = overlap_info.get("per_layer_raw_norm") or {}
                    per_layer_consec = overlap_info.get("per_layer_consecutive_norm") or {}
                    cross_key = overlap_info.get("cross_key", "topk_jaccard_raw_norm@50")
                    consec_key = overlap_info.get("consecutive_key", "topk_jaccard_consecutive@50")
                    if per_layer_cross or per_layer_consec:
                        for rec in json_data.get("pure_next_token_records", []):
                            if rec.get("prompt_id") != prompt_id or rec.get("prompt_variant") != prompt_variant:
                                continue
                            layer_idx = rec.get("layer")
                            if layer_idx in per_layer_cross:
                                rec[cross_key] = per_layer_cross[layer_idx]
                            if layer_idx in per_layer_consec:
                                rec[consec_key] = per_layer_consec[layer_idx]
            except Exception as e:
                # Best-effort: surface minimal error signal for debugging
                try:
                    summary_diag["raw_lens_full_error"] = str(e)
                except Exception:
                    pass

            # Cross-validate strict-copy earliest layers across thresholds against raw lens (001_LAYERS_BASELINE_PLAN §1.23)
            try:
                ct_block = summary_diag.setdefault("copy_thresholds", {})
                L_map = (ct_block.get("L_copy_strict") or {})
                if isinstance(L_map, dict) and window_records:
                    # Build lookup of raw strict flags by layer
                    raw_by_layer: Dict[int, Dict[str, Any]] = {}
                    for rec in window_records:
                        if rec.get("lens") != "raw":
                            continue
                        lyr = rec.get("layer")
                        if not isinstance(lyr, int):
                            continue
                        raw_by_layer[lyr] = rec
                    # Determine window bounds from summary
                    radius_w = int((window_summary or {}).get("radius", 4))
                    flags = {}
                    for key, L in L_map.items():
                        if not isinstance(L, int):
                            flags[str(key)] = None
                            continue
                        # any raw strict copy within ±radius?
                        found = False
                        for lyr in range(L - radius_w, L + radius_w + 1):
                            row = raw_by_layer.get(lyr)
                            if row is None:
                                continue
                            try:
                                if bool(row.get(f"copy_strict@{key}")):
                                    found = True
                                    break
                            except Exception:
                                continue
                        # norm-only if not found
                        flags[str(key)] = (False if found else True)
                    ct_block["norm_only_flags"] = flags
            except Exception:
                pass

            summary_diag["normalization_provenance"] = {
                "arch": norm_arch,
                "strategy": norm_strategy,
                "per_layer": norm_provenance_entries,
            }
            summary_diag["layer_map"] = layer_map_entries
            numeric_summary = {
                "any_nan": bool(numeric_tracker["any_nan"]),
                "any_inf": bool(numeric_tracker["any_inf"]),
                "max_abs_logit_p99": _percentile(numeric_tracker["max_abs_logit"], 0.99),
                "min_prob_p01": _percentile(numeric_tracker["min_prob"], 0.01),
                "layers_flagged": numeric_tracker["layers_flagged"],
            }
            summary_diag["numeric_health"] = numeric_summary

            summary_diag["repeatability"] = _compute_repeatability_metrics()

            decoding_point_block: Dict[str, Any] = {
                "arch": norm_arch,
                "tested": False,
                "strategies": ["post_ln2", "next_ln1"],
                "targets": [],
                "per_target": [],
                "gate": {"decoding_point_consistent": None},
            }

            def _to_int_or_none(value: Any) -> Optional[int]:
                try:
                    return None if value is None else int(value)
                except (TypeError, ValueError):
                    return None

            if (
                prompt_id == "pos"
                and prompt_variant == "orig"
                and norm_arch == "pre_norm"
            ):
                n_layers_int = int(n_layers)
                target_entries: List[Dict[str, Any]] = []

                def _append_target(name: str, value: Any) -> None:
                    layer_idx = _to_int_or_none(value)
                    if layer_idx is None:
                        return
                    if layer_idx < 0 or layer_idx > n_layers_int:
                        return
                    entry_key = (name, layer_idx)
                    for existing in target_entries:
                        if existing.get("name") == name and existing.get("layer") == layer_idx:
                            return
                    target_entries.append({"name": name, "layer": layer_idx})

                _append_target("L_semantic_norm", summary_diag.get("L_semantic"))
                sem_gate_block = summary_diag.get("semantic_gate") or {}
                _append_target("L_semantic_strong", sem_gate_block.get("L_semantic_strong"))
                L_strong_run2 = _to_int_or_none(sem_gate_block.get("L_semantic_strong_run2"))
                if L_strong_run2 is not None:
                    _append_target("L_semantic_strong_run2", L_strong_run2)
                    if L_strong_run2 + 1 <= n_layers_int:
                        _append_target("L_semantic_strong_run2_plus1", L_strong_run2 + 1)
                _append_target("first_rank_le_5", summary_diag.get("first_rank_le_5"))

                decoding_point_block["targets"] = target_entries

                per_target_entries: List[Dict[str, Any]] = []
                if target_entries:
                    final_probs_cpu = final_probs.detach().to(dtype=torch.float32, device="cpu")
                    answer_id_int = _to_int_or_none(first_ans_token_id)
                    for target in target_entries:
                        layer_idx = int(target["layer"])
                        entry: Dict[str, Any] = {"name": target["name"], "layer": layer_idx}
                        raw_vec = raw_resid_window.get(layer_idx)
                        baseline_logits = norm_logits_window.get(layer_idx)
                        if raw_vec is None:
                            entry["status"] = "missing_residual"
                            per_target_entries.append(entry)
                            continue
                        if baseline_logits is None:
                            entry["status"] = "missing_baseline_logits"
                            per_target_entries.append(entry)
                            continue
                        block_idx = layer_idx - 1
                        if block_idx < 0 or block_idx >= len(model.blocks):
                            entry["status"] = "missing_block"
                            per_target_entries.append(entry)
                            continue
                        ln2_module = getattr(model.blocks[block_idx], "ln2", None)
                        if ln2_module is None:
                            entry["status"] = "missing_ln2"
                            per_target_entries.append(entry)
                            continue
                        if unembed_ctx.W is None:
                            entry["status"] = "missing_unembed"
                            per_target_entries.append(entry)
                            continue
                        try:
                            with torch.no_grad():
                                try:
                                    norm_device = next(ln2_module.parameters()).device  # type: ignore[attr-defined]
                                except (StopIteration, AttributeError):
                                    norm_device = getattr(unembed_ctx.W, "device", torch.device("cpu"))
                                raw_vec_device = raw_vec.detach().to(dtype=torch.float32, device=norm_device)
                                resid_tensor = raw_vec_device.unsqueeze(0).unsqueeze(0)
                                norm_tensor = apply_norm_or_skip(resid_tensor, ln2_module)
                                norm_vec = norm_tensor[0, 0, :]
                                cast_vec = safe_cast_for_unembed(
                                    norm_vec.unsqueeze(0),
                                    unembed_ctx.W,
                                    force_fp32_unembed=unembed_ctx.force_fp32,
                                )
                                logits_same_ln2 = unembed_mm(
                                    cast_vec,
                                    unembed_ctx.W,
                                    unembed_ctx.b,
                                    cache=unembed_ctx.cache,
                                ).squeeze(0)
                                logits_next_ln1 = baseline_logits.detach()
                            metrics = compare_decoding_strategies(
                                logits_same_ln2=logits_same_ln2,
                                logits_next_ln1=logits_next_ln1,
                                final_probs=final_probs_cpu,
                                answer_token_id=answer_id_int,
                            )
                            entry.update(metrics)
                            entry["status"] = "ok"
                        except Exception as exc:
                            entry["status"] = "error"
                            entry["error"] = str(exc)
                        per_target_entries.append(entry)

                    decoding_point_block["per_target"] = per_target_entries
                    if per_target_entries:
                        successful = [e for e in per_target_entries if e.get("status") == "ok"]
                        decoding_point_block["tested"] = bool(successful)
                        gate_value: Optional[bool] = None
                        if successful:
                            rank_values = [
                                e.get("rank1_agree")
                                for e in successful
                                if e.get("rank1_agree") is not None
                            ]
                            if rank_values and not all(bool(val) for val in rank_values):
                                gate_value = False
                            else:
                                pref_name, pref_layer = resolve_preferred_semantic_milestone(summary_diag)
                                decoding_point_block["preferred_target"] = {
                                    "name": pref_name,
                                    "layer": pref_layer,
                                }
                                preferred_entry = None
                                if pref_layer is not None:
                                    for entry in successful:
                                        if _to_int_or_none(entry.get("layer")) == int(pref_layer):
                                            preferred_entry = entry
                                            break
                                if preferred_entry is None and successful:
                                    preferred_entry = successful[0]
                                if preferred_entry is not None:
                                    jaccard_pref = preferred_entry.get("jaccard@10")
                                    if jaccard_pref is None:
                                        gate_value = False
                                    else:
                                        try:
                                            gate_value = bool(float(jaccard_pref) >= 0.5)
                                        except (TypeError, ValueError):
                                            gate_value = False
                                else:
                                    gate_value = None
                    decoding_point_block["gate"]["decoding_point_consistent"] = gate_value
                else:
                    decoding_point_block["tested"] = False
                    decoding_point_block["gate"]["decoding_point_consistent"] = None
            summary_diag["decoding_point"] = decoding_point_block
            if (
                decoding_point_block.get("tested")
                and norm_arch == "pre_norm"
            ):
                strategy_val = summary_diag["normalization_provenance"].get("strategy")
                if isinstance(strategy_val, dict):
                    new_strategy = dict(strategy_val)
                elif isinstance(strategy_val, str) and strategy_val.strip():
                    new_strategy = {"primary": strategy_val}
                else:
                    new_strategy = {"primary": norm_strategy}
                new_strategy["ablation"] = "post_ln2_vs_next_ln1@targets"
                summary_diag["normalization_provenance"]["strategy"] = new_strategy

            if prompt_id == "pos" and prompt_variant == "orig":
                def _softmax_cpu(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
                    if tensor is None:
                        return None
                    try:
                        vec = tensor.detach().to(dtype=torch.float32, device="cpu")
                    except Exception:
                        return None
                    if vec.numel() == 0:
                        return None
                    return torch.softmax(vec, dim=0)

                def _raw_probs_for_layer(layer_idx: int) -> Optional[torch.Tensor]:
                    raw_vec = raw_resid_window.get(layer_idx)
                    if raw_vec is None or unembed_ctx.W is None:
                        return None
                    try:
                        device = unembed_ctx.W.device if hasattr(unembed_ctx.W, "device") else torch.device("cpu")
                        resid_vec = raw_vec.to(dtype=torch.float32, device=device).unsqueeze(0)
                        resid_cast = safe_cast_for_unembed(resid_vec, unembed_ctx.W, force_fp32_unembed=unembed_ctx.force_fp32)
                        logits = unembed_mm(resid_cast, unembed_ctx.W, unembed_ctx.b).squeeze(0).float().cpu()
                    except Exception:
                        return None
                    if logits.numel() == 0:
                        return None
                    return torch.softmax(logits, dim=0)

                def _tuned_probs_for_layer(layer_idx: int) -> Optional[torch.Tensor]:
                    return _softmax_cpu(tuned_logits_window.get(layer_idx))

                def _norm_probs_for_layer(layer_idx: int) -> Optional[torch.Tensor]:
                    return _softmax_cpu(norm_logits_window.get(layer_idx))

                def _topk_id_set(probs: Optional[torch.Tensor], k: int) -> Optional[Set[int]]:
                    if probs is None or probs.numel() == 0:
                        return None
                    k_eff = max(1, min(int(k), probs.shape[0]))
                    try:
                        indices = torch.topk(probs, k_eff, largest=True, sorted=False).indices
                    except Exception:
                        return None
                    return {int(idx) for idx in indices.tolist()}

                def _jaccard(a: Optional[Set[int]], b: Optional[Set[int]]) -> Optional[float]:
                    if a is None or b is None:
                        return None
                    if not a and not b:
                        return 1.0
                    if not a or not b:
                        return 0.0
                    union = a | b
                    if not union:
                        return None
                    return float(len(a & b) / len(union))

                def _spearman_topk(norm_probs: Optional[torch.Tensor], other_probs: Optional[torch.Tensor], k: int = 50) -> Optional[float]:
                    if norm_probs is None or other_probs is None:
                        return None
                    vocab = norm_probs.shape[0]
                    if vocab == 0:
                        return None
                    k_eff = max(1, min(int(k), vocab))
                    try:
                        topk_norm = torch.topk(norm_probs, k_eff, largest=True, sorted=True)
                    except Exception:
                        return None
                    indices = topk_norm.indices
                    if indices.numel() < 2:
                        return None
                    ranks_norm = torch.arange(1, indices.numel() + 1, dtype=torch.float32)
                    try:
                        other_order = torch.argsort(other_probs, descending=True)
                    except Exception:
                        return None
                    if other_order.numel() == 0:
                        return None
                    rank_lookup = {int(idx): int(pos + 1) for pos, idx in enumerate(other_order.tolist())}
                    ranks_other = torch.tensor([rank_lookup.get(int(idx), vocab) for idx in indices], dtype=torch.float32)
                    if ranks_other.numel() < 2:
                        return None
                    x = ranks_norm
                    y = ranks_other
                    x_mean = torch.mean(x)
                    y_mean = torch.mean(y)
                    x_diff = x - x_mean
                    y_diff = y - y_mean
                    denom = torch.sqrt(torch.sum(x_diff * x_diff) * torch.sum(y_diff * y_diff))
                    if denom <= 0 or not torch.isfinite(denom):
                        return None
                    spearman = torch.sum(x_diff * y_diff) / denom
                    return float(torch.clamp(spearman, -1.0, 1.0).item())

                def _median(values: List[float]) -> Optional[float]:
                    vals = [float(v) for v in values if v is not None]
                    if not vals:
                        return None
                    vals.sort()
                    n = len(vals)
                    mid = n // 2
                    if n % 2 == 1:
                        return vals[mid]
                    return (vals[mid - 1] + vals[mid]) / 2.0

                candidate_pairs: List[Tuple[str, int]] = []

                def _add_target(name: str, value: Any):
                    if isinstance(value, int):
                        candidate_pairs.append((name, value))

                _add_target("first_rank_le_10", summary_diag.get("first_rank_le_10"))
                _add_target("first_rank_le_5", summary_diag.get("first_rank_le_5"))
                _add_target("L_semantic_norm", summary_diag.get("L_semantic"))
                sem_gate_block = summary_diag.get("semantic_gate") or {}
                _add_target("L_semantic_strong", sem_gate_block.get("L_semantic_strong"))
                _add_target("L_semantic_strong_run2", sem_gate_block.get("L_semantic_strong_run2"))

                if candidate_pairs:
                    norm_vs_raw_entries: List[Dict[str, Optional[float]]] = []
                    norm_vs_tuned_entries: List[Dict[str, Optional[float]]] = []

                    for _, layer_idx in candidate_pairs:
                        norm_probs = _norm_probs_for_layer(layer_idx)
                        raw_probs = _raw_probs_for_layer(layer_idx)
                        tuned_probs = _tuned_probs_for_layer(layer_idx) if tuned_enabled else None

                        norm_top10 = _topk_id_set(norm_probs, 10)
                        norm_top50 = _topk_id_set(norm_probs, 50)
                        raw_top10 = _topk_id_set(raw_probs, 10)
                        raw_top50 = _topk_id_set(raw_probs, 50)
                        tuned_top10 = _topk_id_set(tuned_probs, 10)
                        tuned_top50 = _topk_id_set(tuned_probs, 50)

                        norm_vs_raw_entries.append({
                            "layer": layer_idx,
                            "jaccard@10": _jaccard(norm_top10, raw_top10),
                            "jaccard@50": _jaccard(norm_top50, raw_top50),
                            "spearman_top50": _spearman_topk(norm_probs, raw_probs, 50),
                        })

                        if tuned_enabled:
                            norm_vs_tuned_entries.append({
                                "layer": layer_idx,
                                "jaccard@10": _jaccard(norm_top10, tuned_top10),
                                "jaccard@50": _jaccard(norm_top50, tuned_top50),
                                "spearman_top50": _spearman_topk(norm_probs, tuned_probs, 50),
                            })

                    def _build_p50(entries: List[Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
                        metrics = {"jaccard@10": [], "jaccard@50": [], "spearman_top50": []}
                        for entry in entries:
                            for key in metrics:
                                val = entry.get(key)
                                if isinstance(val, float):
                                    metrics[key].append(val)
                        return {metric: _median(vals) for metric, vals in metrics.items()}

                    lens_consistency_block: Dict[str, Any] = {
                        "targets": [name for name, _ in candidate_pairs],
                        "norm_vs_raw": {
                            "at_targets": norm_vs_raw_entries,
                            "p50": _build_p50(norm_vs_raw_entries),
                        },
                    }
                    if tuned_enabled and norm_vs_tuned_entries:
                        lens_consistency_block["norm_vs_tuned"] = {
                            "at_targets": norm_vs_tuned_entries,
                            "p50": _build_p50(norm_vs_tuned_entries),
                        }
                    summary_diag["lens_consistency"] = lens_consistency_block

            if tuned_enabled and tuned_summaries is not None:
                summary_tuned = summarize_pure_records(
                    tuned_collected_records,
                    copy_threshold=copy_threshold,
                    copy_window_k=getattr(tuned_window_manager, "window_k", 1) if tuned_window_manager is not None else 1,
                    copy_soft_threshold=copy_soft_threshold,
                    copy_soft_window_ks=copy_soft_window_ks,
                    copy_match_level="id_subsequence",
                    lens_tag="tuned",
                    surface_delta=SURFACE_DELTA,
                    geom_gamma=GEOM_GAMMA,
                    topk_prompt_tau=TOPK_DECAY_TAU,
                    n_layers=int(model.cfg.n_layers),
                    semantic_margin_delta=SEMANTIC_MARGIN_DELTA,
                    p_uniform=p_uniform,
                )
                tuned_summaries.append(summary_tuned)
            return summary_diag, last_layer_consistency, detected_architecture, diag_delta
        finally:
            detach_hooks(hooks)
            residual_cache.clear()
            hooks.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
