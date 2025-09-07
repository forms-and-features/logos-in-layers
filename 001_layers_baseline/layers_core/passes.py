from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import os

from .hooks import build_cache_hook, attach_residual_hooks, detach_hooks, get_residual_safely
from .prism import whiten_apply
from .prism_sidecar import append_prism_record, append_prism_pure_next_token
from .records import make_record, make_pure_record
from .pure_emit import compute_pure_next_token_info
from .numerics import bits_entropy_from_logits
from .norm_utils import detect_model_architecture, get_correct_norm_module, apply_norm_or_skip
from .raw_lens import should_sample_layer, record_dual_lens_sample
from .summaries import summarize_pure_records
from .consistency import compute_last_layer_consistency
from .unembed import unembed_mm


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
    analysis_W_U: torch.Tensor,
    analysis_b_U: Optional[torch.Tensor],
    force_fp32_unembed: bool,
    mm_cache: Dict[str, Any],
    copy_threshold: float,
    copy_margin: float,
    entropy_collapse_threshold: float,
    top_k_record: int,
    top_k_verbose: int,
    keep_residuals: bool,
    out_dir: Optional[str],
    RAW_LENS_MODE: str,
    json_data: Dict[str, Any],
    json_data_prism: Dict[str, Any],
    prism_active: bool,
    prism_stats: Any,
    prism_Q: Any,
    decode_id_fn,
    ctx_ids_list: Sequence[int],
    first_ans_token_id: Optional[int],
    important_words: Sequence[str],
    head_scale_cfg: Optional[float],
    head_softcap_cfg: Optional[float],
    clean_model_name: Optional[str] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], str, Dict[str, Any]]:
    """Run a single prompt pass and append outputs into json_data structures.

    Returns (summary_diag, last_layer_consistency, detected_architecture).
    """
    window_manager.reset_variant(prompt_id, prompt_variant)

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

            resid_norm = resid_raw
            # Normalized residual only for Prism and optional saving
            norm_logits_all = norm_lens.forward(
                model,
                0,
                resid_raw,
                probe_after_block=False,
                W_U=analysis_W_U,
                b_U=analysis_b_U,
                force_fp32_unembed=force_fp32_unembed,
                cache=mm_cache,
            )
            # Pass-wide Prism enablement is decided at L0 and carried forward
            # to match baseline behavior (no per-layer re-enabling attempts).
            prism_enabled = prism_active
            prism_Q_use = prism_Q
            diag_delta: Dict[str, Any] = {}
            if prism_active:
                # align with run.py behavior: normalize before Prism whitening for L0
                norm_module = get_correct_norm_module(model, 0, probe_after_block=False)
                resid_norm = apply_norm_or_skip(resid_norm, norm_module)
                Xw_L0 = whiten_apply(resid_norm[0], prism_stats)
                try:
                    if hasattr(prism_Q_use, 'device') and prism_Q_use.device != Xw_L0.device:
                        prism_Q_use = prism_Q_use.to(Xw_L0.device)
                except RuntimeError:
                    prism_enabled = False
                    diag_delta["placement_error"] = "prism Q placement failed at L0"
                if prism_enabled:
                    Xp_L0 = Xw_L0 @ prism_Q_use
                    Wp = analysis_W_U.float() if analysis_W_U.dtype != torch.float32 else analysis_W_U
                    bp = (analysis_b_U.float() if (analysis_b_U is not None and analysis_b_U.dtype != torch.float32) else analysis_b_U)
                    prism_logits_all_L0 = unembed_mm(Xp_L0, Wp, bp, cache=mm_cache).float()

            # Per-position records at L0
            for pos in range(tokens.shape[1]):
                layer_logits = norm_logits_all[pos]
                entropy_bits = bits_entropy_from_logits(layer_logits)
                token_str = str_tokens[pos]
                verbose = _is_verbose_position(pos, token_str, tokens.shape[1], important_words)
                k = top_k_verbose if verbose else top_k_record
                _, top_indices_k = torch.topk(layer_logits, k, largest=True, sorted=True)
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
                entropy_collapse_threshold=entropy_collapse_threshold,
                decode_id_fn=decode_id_fn,
                ground_truth=ground_truth,
                top_k_record=top_k_record,
                prompt_id=prompt_id,
                prompt_variant=prompt_variant,
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

            # Optional raw-vs-norm sample at L0
            if RAW_LENS_MODE != "off":
                record_dual_lens_sample(
                    json_data["raw_lens_check"],
                    layer_out_idx=dual_ctx["layer"],
                    last_logits_norm=dual_ctx["last_logits_norm"],
                    resid_raw_last_vec=resid_raw[0, int(dual_ctx["last_pos"]), :],
                    W_U=analysis_W_U,
                    b_U=analysis_b_U,
                    force_fp32_unembed=force_fp32_unembed,
                    tokenizer=model.tokenizer,
                    final_probs=dual_ctx["final_probs"],
                    first_ans_id=dual_ctx["first_ans_id"],
                    ground_truth=dual_ctx["ground_truth"],
                )

            if prism_enabled:
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
                    entropy_collapse_threshold=entropy_collapse_threshold,
                    decode_id_fn=decode_id_fn,
                    ground_truth=ground_truth,
                    top_k_record=top_k_record,
                    prompt_id=prompt_id,
                    prompt_variant=prompt_variant,
                )

            # Save normalized residual if requested
            if keep_residuals:
                name_root = clean_model_name or model.cfg.__dict__.get("model_name", "model")
                resid_filename = f"{name_root}_00_resid.pt"
                resid_path = os.path.join(out_dir or os.getcwd(), resid_filename)
                torch.save(resid_norm.to(dtype=resid_norm.dtype).cpu(), resid_path)

            # ---- Post-block layers ----
            detected_architecture = detect_model_architecture(model)
            n_layers = model.cfg.n_layers

            def _should_sample(one_indexed: int) -> bool:
                return should_sample_layer(RAW_LENS_MODE, n_layers, one_indexed)

            for layer in range(n_layers):
                resid_raw = get_residual_safely(residual_cache, layer)
                resid_norm = resid_raw
                # For Prism whitening, normalize post-block residual where applicable
                # Carry the pass-wide Prism enable flag; do not retry per layer
                prism_enabled_layer = prism_enabled
                if prism_enabled_layer:
                    norm_module = get_correct_norm_module(model, layer, probe_after_block=True)
                    resid_norm = apply_norm_or_skip(resid_norm, norm_module)

                logits_all = norm_lens.forward(
                    model,
                    layer,
                    resid_raw,
                    probe_after_block=True,
                    W_U=analysis_W_U,
                    b_U=analysis_b_U,
                    force_fp32_unembed=force_fp32_unembed,
                    cache=mm_cache,
                )

                # Prism sidecar logits for per-position record emission
                if prism_enabled_layer:
                    Xw = whiten_apply(resid_norm[0], prism_stats)
                    Xp = Xw @ prism_Q_use
                    Wp = analysis_W_U.float() if analysis_W_U.dtype != torch.float32 else analysis_W_U
                    bp = (analysis_b_U.float() if (analysis_b_U is not None and analysis_b_U.dtype != torch.float32) else analysis_b_U)
                    prism_logits_all = unembed_mm(Xp, Wp, bp, cache=mm_cache).float()

                # Per-position records for this layer
                for pos in range(tokens.shape[1]):
                    layer_logits = logits_all[pos]
                    full_probs = torch.softmax(layer_logits, dim=0)
                    entropy_bits = bits_entropy_from_logits(layer_logits)
                    token_str = str_tokens[pos]
                    verbose = _is_verbose_position(pos, token_str, tokens.shape[1], important_words)
                    k = top_k_verbose if verbose else top_k_record
                    _, top_indices_k = torch.topk(layer_logits, k, largest=True, sorted=True)
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
                    if prism_enabled_layer:
                        append_prism_record(
                            json_data_prism,
                            prompt_id=prompt_id,
                            prompt_variant=prompt_variant,
                            layer=layer + 1,
                            pos=pos,
                            token=token_str,
                            logits_pos=prism_logits_all[pos],
                            decode_id_fn=decode_id_fn,
                            top_k=k,
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
                    entropy_collapse_threshold=entropy_collapse_threshold,
                    decode_id_fn=decode_id_fn,
                    ground_truth=ground_truth,
                    top_k_record=top_k_record,
                    prompt_id=prompt_id,
                    prompt_variant=prompt_variant,
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

                # Optional raw-vs-norm sample
                if RAW_LENS_MODE != "off" and _should_sample(layer + 1):
                    record_dual_lens_sample(
                        json_data["raw_lens_check"],
                        layer_out_idx=dual_ctx["layer"],
                        last_logits_norm=dual_ctx["last_logits_norm"],
                        resid_raw_last_vec=resid_raw[0, int(dual_ctx["last_pos"]), :],
                        W_U=analysis_W_U,
                        b_U=analysis_b_U,
                        force_fp32_unembed=force_fp32_unembed,
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

                # Save residual per layer if requested
                if keep_residuals:
                    name_root = clean_model_name or model.cfg.__dict__.get("model_name", "model")
                    resid_filename = f"{name_root}_{layer+1:02d}_resid.pt"
                    # Match original dtype behavior when available
                    save_dtype = getattr(getattr(model, 'cfg', object()), 'dtype', torch.float32)
                    try:
                        resid_to_save = resid_norm.to(dtype=save_dtype).cpu()
                    except Exception:
                        resid_to_save = resid_norm.float().cpu()
                    resid_path = os.path.join(out_dir or os.getcwd(), resid_filename)
                    torch.save(resid_to_save, resid_path)

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
                copy_window_k=getattr(window_manager, "k", 1),
                copy_match_level="id_subsequence",
            )
            return summary_diag, last_layer_consistency, detected_architecture, diag_delta
        finally:
            detach_hooks(hooks)
            residual_cache.clear()
            hooks.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
