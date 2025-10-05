from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, List

import math
import torch
import os

from .hooks import build_cache_hook, attach_residual_hooks, detach_hooks, get_residual_safely
from .prism_sidecar import append_prism_record, append_prism_pure_next_token
from .records import make_record, make_pure_record
from .pure_emit import compute_pure_next_token_info
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

SURFACE_DELTA = 0.05
GEOM_GAMMA = 0.02
TOPK_PROMPT_MASS_K = 50
TOPK_DECAY_TAU = 0.33


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

    last_layer_consistency: Optional[Dict[str, Any]] = None
    detected_architecture: str = "unknown"

    norm_logits_window: Dict[int, torch.Tensor] = {}
    raw_resid_window: Dict[int, torch.Tensor] = {}
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

    def _norm_effect_metrics(raw_vec: torch.Tensor | None, norm_vec: torch.Tensor | None):
        if raw_vec is None or norm_vec is None:
            return None, None
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
            return ratio, cos_val
        except Exception:
            return None, None

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
        if torch.are_deterministic_algorithms_enabled():
            return {"status": "skipped", "reason": "deterministic_env"}

        layer_keys = sorted(set(norm_logits_window.keys()) & set(raw_resid_window.keys()))
        if not layer_keys:
            return {"status": "unavailable"}

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
            return {"status": "unavailable"}

        max_dev = max(rank_devs) if rank_devs else 0.0
        p95_dev = _percentile(rank_devs, 0.95) if rank_devs else 0.0
        flip_rate = float(flips) / float(layers_count)

        return {
            "status": "ok",
            "layers_checked": layers_count,
            "max_rank_dev": float(max_dev),
            "p95_rank_dev": float(p95_dev if p95_dev is not None else 0.0),
            "top1_flip_rate": float(flip_rate),
        }

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

            ratio0, cos0 = _norm_effect_metrics(raw_vec_L0, geom_vec_norm_pre)
            ln_source0, eps_inside0, scale_used0 = describe_norm_origin(model, 0, probe_after_block=False)
            norm_provenance_entries.append({
                "layer": 0,
                "ln_source": ln_source0,
                "eps_inside_sqrt": bool(eps_inside0),
                "scale_gamma_used": bool(scale_used0),
                "resid_norm_ratio": ratio0,
                "delta_resid_cos": cos0,
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
            )
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
                    extra=view["record_extra"],
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
                        extra=tuned_view["record_extra"],
                    )
                )
                tuned_collected_records.append(tuned_collected)

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

                ratio_layer, cos_layer = _norm_effect_metrics(raw_vec_layer, geom_vec_norm_post)
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
                        translated_seq = tuned_adapter.translator(
                            resid_norm_post[0, :, :], layer
                        )
                        geom_vec_tuned_post = translated_seq[last_pos, :].detach()
                    except Exception:
                        geom_vec_tuned_post = None

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
                            )
                        )

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
                )
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
                        extra=view["record_extra"],
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
                )
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
                            extra=tuned_view["record_extra"],
                        )
                    )
                    tuned_collected_records.append(tuned_collected)

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
            )
            summary_diag.setdefault("answer_margin_unit", "logit")

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
                )
                tuned_summaries.append(summary_tuned)
            return summary_diag, last_layer_consistency, detected_architecture, diag_delta
        finally:
            detach_hooks(hooks)
            residual_cache.clear()
            hooks.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
