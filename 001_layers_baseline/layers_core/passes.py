from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, List

import torch
import os

from .hooks import build_cache_hook, attach_residual_hooks, detach_hooks, get_residual_safely
from .prism_sidecar import append_prism_record, append_prism_pure_next_token
from .records import make_record, make_pure_record
from .pure_emit import compute_pure_next_token_info
from .numerics import bits_entropy_from_logits
from .norm_utils import detect_model_architecture, get_correct_norm_module, apply_norm_or_skip
from .raw_lens import should_sample_layer, record_dual_lens_sample
from .summaries import summarize_pure_records
from .consistency import compute_last_layer_consistency
from .lenses import PrismLensAdapter
from .contexts import UnembedContext, PrismContext


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
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], str, Dict[str, Any]]:
    """Run a single prompt pass and append outputs into json_data structures.

    Returns (summary_diag, last_layer_consistency, detected_architecture, diag_delta).
    """
    window_manager.reset_variant(prompt_id, prompt_variant)

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
            _final_norm = torch.norm(final_logits) + 1e-12
            final_dir = (final_logits / _final_norm)

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
            )
            if tuned_enabled and tuned_summaries is not None:
                summary_tuned = summarize_pure_records(
                    tuned_collected_records,
                    copy_threshold=copy_threshold,
                    copy_window_k=getattr(tuned_window_manager, "window_k", 1) if tuned_window_manager is not None else 1,
                    copy_soft_threshold=copy_soft_threshold,
                    copy_soft_window_ks=copy_soft_window_ks,
                    copy_match_level="id_subsequence",
                )
                tuned_summaries.append(summary_tuned)
            return summary_diag, last_layer_consistency, detected_architecture, diag_delta
        finally:
            detach_hooks(hooks)
            residual_cache.clear()
            hooks.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
