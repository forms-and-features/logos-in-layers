from transformer_lens import HookedTransformer
import torch
import torch.nn as nn
from datetime import datetime
import os
import subprocess
import sys
import math
import json
import argparse
import gc  # For garbage collection

# --- deterministic bootstrap -------------------------------------------------
import random, numpy as np

SEED = 316
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.use_deterministic_algorithms(True)   # PyTorch 2.x+
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # harmless on CPU, required for CUDA
torch.set_num_threads(1)  # optional; comment out if you need full CPU speed
# -----------------------------------------------------------------------------

# Top-k settings for record emission
TOP_K_RECORD = 5    # number of tokens to record for non-verbose slots
TOP_K_VERBOSE = 20  # number of tokens to record for verbose slots and answer position

# Layer-by-layer prediction analysis with LayerNorm lens correction
# Toggle USE_NORM_LENS for raw vs normalized residual stream analysis

# Candidate models (small → large). Device fit is decided per model at runtime.
CANDIDATE_MODELS = [
    "mistralai/Mistral-7B-v0.1",
    "meta-llama/Meta-Llama-3-8B",
    "Qwen/Qwen3-8B",
    "google/gemma-2-9b",
    "Qwen/Qwen3-14B",
    "google/gemma-2-27b",
    "01-ai/Yi-34B",
    "meta-llama/Meta-Llama-3-70B",
    "mistralai/Mistral-Small-24B-Base-2501",
    "Qwen/Qwen2.5-72B",
]

# Backward-compatible name used throughout this script
CONFIRMED_MODELS = CANDIDATE_MODELS

# --- helpers (extracted to norm_utils) --------------------------------------
from layers_core.norm_utils import (
    _get_rms_scale,
    apply_norm_or_skip,
    detect_model_architecture,
    get_correct_norm_module,
)
from layers_core.numerics import (
    bits_entropy_from_logits,
    safe_cast_for_unembed,
)
from layers_core.metrics import compute_next_token_metrics
from layers_core.csv_io import write_csv_files
from layers_core.collapse_rules import (
    is_semantic_top1,
    detect_copy_collapse_id_subseq,
    is_pure_whitespace_or_punct,
)
from layers_core.device_policy import (
    choose_dtype,
    should_auto_promote_unembed,
    select_best_device,
)
from layers_core.hooks import build_cache_hook, attach_residual_hooks, detach_hooks
from layers_core.run_dir import setup_run_latest_directory
from layers_core.config import ExperimentConfig
from layers_core.raw_lens import (
    get_raw_lens_mode,
    init_raw_lens_check,
    should_sample_layer,
    record_dual_lens_sample,
    summarize_raw_lens_check,
)
from layers_core.summaries import summarize_pure_records

def clean_model_name(model_id):
    """Extract clean model name for filename"""
    # Remove organization prefix (everything before last '/')
    clean_name = model_id.split('/')[-1]
    return clean_name

 


def run_experiment_for_model(model_id, output_files, config: ExperimentConfig):
    """Run the complete experiment for a single model and write results to files"""
    
    def evaluate_model():
        """The actual experiment code - all prints go to console"""
        print(f"\n{'='*60}")
        print(f"EVALUATING MODEL: {model_id}")
        print(f"{'='*60}")
        
        # Variable to store detected architecture
        detected_architecture = None
        
        # ---- device & dtype ---------------------------------------------------
        device = config.device
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA requested but not available; falling back to CPU.")
            device = "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            print("⚠️  MPS requested but not available; falling back to CPU.")
            device = "cpu"

        dtype = choose_dtype(device, model_id)

        # ---- load model -------------------------------------------------------
        print(f"Loading model on [{device}] ...")
        
        # Clear any existing CUDA cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Set batch_first=False for models that need it (TransformerLens expectation)
        os.environ['TRANSFORMERS_BATCH_FIRST'] = 'False'
            
        # Load model
        try:
            print("Loading directly to target device…")
            if device == "cpu":
                # Explicitly keep all weights on CPU – avoids Accelerate placing them on MPS
                model = HookedTransformer.from_pretrained_no_processing(
                    model_id,
                    device="cpu",
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
            else:
                model = HookedTransformer.from_pretrained_no_processing(
                    model_id,
                    device=device,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
        except Exception as e:
            print(f"Direct loading to {device} failed: {e}")
            print("Falling back to CPU loading...")
            # Fallback: load on CPU then move
            model = HookedTransformer.from_pretrained_no_processing(
                model_id,
                device="cpu",
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            # Clear cache before moving
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Moving model to {device}...")
            model = model.to(device)
            
        model.eval()  # Hygiene: avoid dropout etc.
        
        # Run KL sanity test if requested
        if config.self_test:
            try:
                from kl_sanity_test import run_kl_sanity_test
                # Prefer the model's own tokenizer when available
                tokenizer = getattr(model, 'tokenizer', None)
                if not hasattr(model, 'lm_head'):
                    print("ℹ️ Skipping KL sanity test: requires HF model interface (lm_head/hidden_states). Use kl_sanity_test.py standalone if needed.")
                else:
                    test_passed = run_kl_sanity_test(model, tokenizer)
                    if not test_passed:
                        print("❌ Self-test failed - normalization scaling is incorrect!")
                        print("❌ ABORTING: Cannot trust analysis results with incorrect scaling!")
                        return {"error": "Self-test failed"}
                    print("✅ Self-test passed - continuing with normal evaluation...\n")
            except ImportError as e:
                print(f"❌ Could not import KL sanity test: {e}")
                return {"error": "Self-test import failed"}
        
        # Clear cache after loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Debug: inspect model parameter devices
        unique_param_devices = {p.device for p in model.parameters()}
        print(f"[DEBUG MODEL] Unique parameter devices: {unique_param_devices}")
        
        # Toggle for using normalized lens (recommended for accurate interpretation)
        USE_NORM_LENS = True
        
        # Toggle for FP32 unembedding (recommended for research-grade precision)
        # Prevents under-resolving logit gaps < 1e-5 with minimal memory overhead.
        USE_FP32_UNEMBED = should_auto_promote_unembed(dtype)

        # Shadow copies for analysis-only unembedding (do NOT mutate model params)
        analysis_W_U = model.unembed.W_U  # default: use model parameter dtype
        analysis_b_U = getattr(model.unembed, 'b_U', None)
        UNEMBED_DTYPE = analysis_W_U.dtype  # initial report

        # If enabled, create FP32 shadow weights for unembedding without touching the model's forward path
        if USE_FP32_UNEMBED and analysis_W_U.dtype != torch.float32:
            print(f"🔬 Using FP32 shadow unembed weights for analysis (was {analysis_W_U.dtype})")
            analysis_W_U = analysis_W_U.float()
            if analysis_b_U is not None:
                analysis_b_U = analysis_b_U.float()
            UNEMBED_DTYPE = torch.float32

        # Apply CLI-based FP32 unembed promotion if requested (analysis-only)
        if config.fp32_unembed and UNEMBED_DTYPE != torch.float32:
            print(f"🔬 CLI: Forcing FP32 shadow unembed weights for analysis (was {UNEMBED_DTYPE})")
            analysis_W_U = analysis_W_U.float()
            if analysis_b_U is not None:
                analysis_b_U = analysis_b_U.float()
            UNEMBED_DTYPE = torch.float32
        
        context_prompt = "Give the city name only, plain text. The capital of Germany is called simply"
        ground_truth = "Berlin"  # For display/comparison
        
        first_block_ln1_type = type(model.blocks[0].ln1).__name__ if hasattr(model, 'blocks') and len(model.blocks) > 0 else None
        final_ln_type = type(model.ln_final).__name__ if hasattr(model, 'ln_final') else None
        
        # Check if LayerNorm bias fix will be applied
        uses_layernorm = (hasattr(model, 'blocks') and len(model.blocks) > 0 and 
                         isinstance(model.blocks[0].ln1, nn.LayerNorm))
        layernorm_bias_fix = "active" if uses_layernorm else "not_needed_rms_model"
        
        # Check normalization alignment for post-block residuals
        has_ln2 = (hasattr(model, 'blocks') and len(model.blocks) > 0 and 
                   hasattr(model.blocks[0], 'ln2'))
        if has_ln2:
            ln2_type = type(model.blocks[0].ln2).__name__
            if isinstance(model.blocks[0].ln2, nn.LayerNorm):
                norm_alignment_fix = "using_ln2_layernorm_for_post_block"
            else:
                norm_alignment_fix = "using_ln2_rmsnorm_for_post_block"
        else:
            norm_alignment_fix = "fallback_to_ln1"
        
        # Check layer-0 normalization approach
        layer0_norm_fix = "using_real_ln1_on_embeddings" if USE_NORM_LENS else "no_normalization"
        
        # Check mixed precision fix
        mixed_precision_fix = "casting_to_fp32_before_unembed"
        
        # Check positional embedding type for layer-0 interpretation
        has_additive_pos_embed = 'hook_pos_embed' in model.hook_dict
        layer0_position_info = "additive_pos_embed_included" if has_additive_pos_embed else "token_only_rotary_model"
        
        # Raw-vs-Norm dual-lens mode (env-controlled; default: sampled checks)
        RAW_LENS_MODE = get_raw_lens_mode(config.self_test)

        diag = {
            "type": "diagnostics",
            "model": model_id,
            "device": device,
            "use_norm_lens": USE_NORM_LENS,
            "use_fp32_unembed": USE_FP32_UNEMBED,
            "unembed_dtype": str(UNEMBED_DTYPE),
            "first_block_ln1_type": first_block_ln1_type,
            "final_ln_type": final_ln_type,
            "layernorm_bias_fix": layernorm_bias_fix,
            "norm_alignment_fix": norm_alignment_fix,
            "layer0_norm_fix": layer0_norm_fix,
            "mixed_precision_fix": mixed_precision_fix,
            "layer0_position_info": layer0_position_info,
            "context_prompt": context_prompt,
            "target_prediction": "first unseen token (likely 'Berlin')"
        }
        # Collect data for JSON output
        json_data = {
            "prompt": {"type": "prompt", "context_prompt": context_prompt, "ground_truth": ground_truth},
            "records": [],
            "pure_next_token_records": [],
            "test_prompts": [],
            "temperature_exploration": [],
            "diagnostics": None,
            "final_prediction": None,
            "model_stats": None,
            # Raw-vs-Norm sanity block (PROJECT_NOTES §1.4)
            "raw_lens_check": init_raw_lens_check(RAW_LENS_MODE),
        }
        IMPORTANT_WORDS = ["Germany", "Berlin", "capital", "Answer", "word", "simply"]

        def is_verbose_position(pos, token_str, seq_len):
            # Final position (predicting next token) always verbose
            if pos == seq_len - 1:
                return True
            # Any prompt word in token string
            for w in IMPORTANT_WORDS:
                if w.lower() in token_str.lower().strip(".,!?;:"):
                    return True
            return False

        def print_summary(layer_idx, pos, token_str, entropy_bits, top_tokens, top_probs, is_pure_next_token=False, extra=None):
            # Collect record data for JSON output
            record = {
                "type": "pure_next_token_record" if is_pure_next_token else "record",
                "layer": layer_idx,
                "pos": pos,
                "token": token_str,
                "entropy": entropy_bits,
                "topk": [[tok, prob.item()] for tok, prob in zip(top_tokens, top_probs)]
            }
            if extra:
                record.update(extra)
            
            # Add to appropriate collection
            if is_pure_next_token:
                json_data["pure_next_token_records"].append(record)
            else:
                json_data["records"].append(record)
        
        # Tokenize the context prompt (without "Answer:" to avoid teacher-forcing)
        tokens = model.to_tokens(context_prompt)      # let Accelerate move it

        # Prepare prompt token IDs (ordered) for sub-word-aware copy detection
        # Prefer HF-style tokenizer; fall back to model.to_tokens for mock/test models
        try:
            ctx_ids = model.tokenizer(context_prompt, add_special_tokens=False)["input_ids"]
        except Exception:
            try:
                ctx_ids = model.to_tokens(context_prompt)[0].tolist()
            except Exception:
                ctx_ids = []
        # Gold answer first-token id (ID-level matching; see PROJECT_NOTES §1.3)
        try:
            ans_ids_full = model.tokenizer(context_prompt + " " + ground_truth, add_special_tokens=False)["input_ids"]
            first_ans_id = ans_ids_full[len(ctx_ids)] if len(ans_ids_full) > len(ctx_ids) else None
        except Exception:
            try:
                ans_tokens = model.to_tokens(context_prompt + " " + ground_truth)[0].tolist()
                first_ans_id = ans_tokens[len(ctx_ids)] if len(ans_tokens) > len(ctx_ids) else None
            except Exception:
                first_ans_id = None
        # Rolling window of the last k top-1 IDs
        window_ids: list[int] = []
        
        # Storage to collect pure_next_token_records for L_copy/L_semantic computation
        collected_pure_records = []
        
        # Begin capturing residual streams
        with torch.no_grad():
            # MEMORY EFFICIENT: Use targeted caching instead of run_with_cache
            # (removed human-readable progress print)
            
            # Storage for only the residual streams we need
            residual_cache = {}
            # Build hook closure
            cache_hook = build_cache_hook(residual_cache)
            # Attach hooks and record handles
            hooks, has_pos_embed = attach_residual_hooks(model, cache_hook)
            
            try:
                # Run forward pass with targeted hooks
                logits = model(tokens)

                # Cache final head reference distribution once (for KL-to-final)
                final_logits = logits[0, -1, :].float()
                final_probs = torch.softmax(final_logits, dim=0)
                # Representation-drift baseline direction (PROJECT_NOTES §1.5)
                _final_norm = torch.norm(final_logits) + 1e-12
                final_dir = (final_logits / _final_norm)
                
                # Debug: print device placements of cached activations and tokens
                for name, t in residual_cache.items():
                    print(f"[DEBUG CACHE] {name} device: {t.device}")
                print(f"[DEBUG TOKENS] tokens device: {tokens.device}")
                
                # Show top predictions at different layers
                print(f"\nLayer-by-layer analysis of context: '{context_prompt}'")
                print("→ Predicting the first unseen token (what comes after the context)")
                if USE_NORM_LENS:
                    # Check if we'll actually be applying norms
                    if hasattr(model, 'blocks') and len(model.blocks) > 0:
                        first_norm = model.blocks[0].ln1
                        norm_type = type(first_norm).__name__
                        
                        if isinstance(first_norm, nn.LayerNorm):
                            print("Using NORMALIZED residual stream (LayerNorm applied - more accurate)")
                        elif 'RMS' in norm_type:
                            if _get_rms_scale(first_norm) is None:
                                print("Using NORMALIZED residual stream (RMS, no learnable scale)")
                            else:
                                print("Using NORMALIZED residual stream (RMS + learned scale)")
                        else:
                            print("Using RAW residual stream (unsupported normalization, skipping to avoid distortion)")
                    else:
                        print("Using RAW residual stream (no normalization layers found)")
                else:
                    print("Using RAW residual stream (normalization disabled)")
                print("Note: Shown probabilities are from full softmax (calibrated and comparable)")
                print(f"copy-collapse: top-1 ID-window in prompt & p>{config.copy_threshold}")
                if config.keep_residuals:
                    print("💾 Saving residual tensors to disk (--keep-residuals enabled)")
                print("-" * 60)
                
                # Get string representations of tokens for labeling output
                str_tokens = model.to_str_tokens(context_prompt)

                # Layer 0: embeddings (+ positional embeddings if available)
                print("Layer  0 (embeddings):")
                if has_pos_embed:
                    resid = (residual_cache['hook_embed'] +
                             residual_cache['hook_pos_embed'])
                else:
                    resid = residual_cache['hook_embed']
                    print("[diagnostic] No separate positional embedding hook found (as expected for rotary models).")
                    print("[diagnostic] Layer 0 contains TOKEN information only; positional info is injected inside attention layers.")
                # Keep a raw copy before normalization for dual-lens sanity
                resid_raw_L0 = resid
                # FIXED: Apply first real normalizer to embeddings if using norm-lens
                # This gives us the normalized embeddings that the model actually sees
                if USE_NORM_LENS:
                    # Use the actual first normalization layer instead of synthetic γ=1
                    print("[diagnostic] Applying real ln1 normalization to embeddings (not synthetic γ=1)")
                    if 'detected_architecture' not in locals():
                        detected_architecture = detect_model_architecture(model)
                    norm_module = get_correct_norm_module(model, 0, probe_after_block=False)
                    resid = apply_norm_or_skip(resid, norm_module)
                
                # Vectorized unembedding for all positions using analysis shadow weights
                resid_cast = safe_cast_for_unembed(resid[0], analysis_W_U, force_fp32_unembed=(config.fp32_unembed or USE_FP32_UNEMBED))
                logits_all = (resid_cast @ analysis_W_U)
                if analysis_b_U is not None:
                    logits_all = logits_all + analysis_b_U
                logits_all = logits_all.float()  # downstream numerics in fp32
                
                # Save residuals if requested
                if config.keep_residuals:
                    clean_name = clean_model_name(model_id)
                    resid_filename = f"{clean_name}_00_resid.pt"
                    # Use configured output directory; meta_filepath is not available here
                    resid_path = os.path.join(config.out_dir or os.getcwd(), resid_filename)
                    resid_cpu = resid.to(dtype=model.cfg.dtype if hasattr(model.cfg, 'dtype') else torch.float32).cpu()
                    torch.save(resid_cpu, resid_path)
                    del resid_cpu
                
                for pos in range(tokens.shape[1]):
                    layer_logits = logits_all[pos]
                    # Compute entropy in bits via centralized helper
                    entropy_bits = bits_entropy_from_logits(layer_logits)  # Prevent negative zero

                    token_str = str_tokens[pos]
                    # Decide verbosity for this position
                    verbose = is_verbose_position(pos, token_str, tokens.shape[1])
                    # Choose k based on verbosity
                    k = TOP_K_VERBOSE if verbose else TOP_K_RECORD
                    # Get top-k indices from raw logits
                    _, top_indices_k = torch.topk(layer_logits, k, largest=True, sorted=True)
                    full_probs  = torch.softmax(layer_logits, dim=0)
                    top_probs_k = full_probs[top_indices_k]
                    top_tokens_k = [model.tokenizer.decode([idx]) for idx in top_indices_k]
                    print_summary(0, pos, token_str, entropy_bits, top_tokens_k, top_probs_k)
                    # Verbose console output removed to reduce noise - data still captured in files
                
                # FIXED: Also emit pure next-token record (last position only)
                # This avoids deflated entropy from tokens the model has already seen
                last_pos = tokens.shape[1] - 1
                last_logits = logits_all[last_pos]
                last_entropy_bits = bits_entropy_from_logits(last_logits)
                last_full_probs   = torch.softmax(last_logits, dim=0)
                last_token_str = "⟨NEXT⟩"  # Pure next-token prediction, not the last prompt token
                _, last_top_indices = torch.topk(last_logits, TOP_K_RECORD, largest=True, sorted=True)
                last_top_probs = last_full_probs[last_top_indices]
                last_top_tokens = [model.tokenizer.decode([idx]) for idx in last_top_indices]
                # Maintain rolling window (k recent top-1 IDs)
                top1_id = last_top_indices[0].item()
                window_ids.append(top1_id)
                if len(window_ids) > config.copy_window_k:
                    window_ids.pop(0)
                
                # ------------------------------------------------------------------- #
                #  1.  Is this layer "copy-collapsed" (prompt echo)?   -> L_copy
                #  2.  Does top-1 equal the ground-truth answer token? -> L_semantic
                # ------------------------------------------------------------------- #
                copy_collapse = detect_copy_collapse_id_subseq(
                    last_logits,
                    ctx_ids,
                    window_ids,
                    copy_threshold=config.copy_threshold,
                    copy_margin=config.copy_margin,
                )
                if copy_collapse and is_pure_whitespace_or_punct(last_top_tokens[0]):
                    copy_collapse = False
                entropy_collapse = last_entropy_bits <= 1.0
                is_answer = is_semantic_top1(last_top_tokens[0], "Berlin")
                # Additional §1.3 metrics (factored helper)
                metrics = compute_next_token_metrics(
                    last_full_probs, top1_id, final_probs, first_ans_id, topk_cum=5
                )

                # Cosine to final direction (PROJECT_NOTES §1.5)
                _curr_norm = torch.norm(last_logits) + 1e-12
                cos_to_final = torch.dot((last_logits / _curr_norm), final_dir).item()

                record_extra = {
                    "copy_collapse": copy_collapse,
                    "entropy_collapse": entropy_collapse,
                    "is_answer": is_answer,
                    **metrics,
                    "cos_to_final": cos_to_final,
                }
                
                # Collect for L_copy/L_semantic computation
                collected_pure_records.append({
                    "layer": 0,
                    "copy_collapse": copy_collapse,
                    "entropy_collapse": entropy_collapse,
                    "is_answer": is_answer,
                    "kl_to_final_bits": metrics["kl_to_final_bits"],
                    "answer_rank": metrics["answer_rank"],
                })
                
                print_summary(0, last_pos, last_token_str, last_entropy_bits, last_top_tokens, last_top_probs, is_pure_next_token=True, extra=record_extra)

                # Dual-lens sanity at layer 0 (always if not "off")
                if RAW_LENS_MODE != "off":
                    # Append a dual-lens sample using helper
                    record_dual_lens_sample(
                        json_data["raw_lens_check"],
                        layer_out_idx=0,
                        last_logits_norm=last_logits,
                        resid_raw_last_vec=resid_raw_L0[0, last_pos, :],
                        W_U=analysis_W_U,
                        b_U=analysis_b_U,
                        force_fp32_unembed=(config.fp32_unembed or USE_FP32_UNEMBED),
                        tokenizer=model.tokenizer,
                        final_probs=final_probs,
                        first_ans_id=first_ans_id,
                        ground_truth=ground_truth,
                    )
                
                # --- free Layer-0 residual to keep host RAM flat ---------------------
                del resid
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Layers 1 to n_layers: after each transformer block
                # Detect architecture once for efficiency
                detected_architecture = detect_model_architecture(model)
                print(f"Detected architecture: {detected_architecture}")
                
                # Determine number of layers for iteration (no longer set by hook attachment)
                n_layers = model.cfg.n_layers
                # Decide which layers to sample for raw-vs-norm (1-indexed for post-block layers)
                # Build a closure to test sampling quickly
                def _should_sample(one_indexed: int) -> bool:
                    return should_sample_layer(RAW_LENS_MODE, n_layers, one_indexed)

                for layer in range(n_layers):
                    print(f"Layer {layer + 1:2d} (after transformer block {layer}):")
                    # Get residual stream after this layer's block
                    resid = residual_cache[f'blocks.{layer}.hook_resid_post']
                    resid_raw = resid  # keep pre-norm for raw lens
                    
                    # Apply normalization if requested
                    if USE_NORM_LENS:
                        # Use the correct normalization module based on probe timing and architecture
                        norm_module = get_correct_norm_module(model, layer, probe_after_block=True)
                        resid = apply_norm_or_skip(resid, norm_module)
                    
                    # Vectorized unembedding for all positions using analysis shadow weights
                    resid_cast = safe_cast_for_unembed(resid[0], analysis_W_U, force_fp32_unembed=(config.fp32_unembed or USE_FP32_UNEMBED))
                    logits_all = (resid_cast @ analysis_W_U)
                    if analysis_b_U is not None:
                        logits_all = logits_all + analysis_b_U
                    logits_all = logits_all.float()  # downstream numerics in fp32
                    
                    # Save residuals if requested
                    if config.keep_residuals:
                        clean_name = clean_model_name(model_id)
                        resid_filename = f"{clean_name}_{layer+1:02d}_resid.pt"
                        # Use configured output directory; meta_filepath is not available here
                        resid_path = os.path.join(config.out_dir or os.getcwd(), resid_filename)
                        resid_cpu = resid.to(dtype=model.cfg.dtype if hasattr(model.cfg, 'dtype') else torch.float32).cpu()
                        torch.save(resid_cpu, resid_path)
                        del resid_cpu
                    
                    for pos in range(tokens.shape[1]):
                        layer_logits = logits_all[pos]
                        # fresh per-token probabilities for THIS layer
                        full_probs  = torch.softmax(layer_logits, dim=0)
                        entropy_bits = bits_entropy_from_logits(layer_logits)

                        token_str = str_tokens[pos]
                        # Decide verbosity for this position
                        verbose = is_verbose_position(pos, token_str, tokens.shape[1])
                        # Choose k based on verbosity
                        k = TOP_K_VERBOSE if verbose else TOP_K_RECORD
                        # Get top-k indices from raw logits
                        _, top_indices_k = torch.topk(layer_logits, k, largest=True, sorted=True)
                        top_probs_k = full_probs[top_indices_k]
                        top_tokens_k = [model.tokenizer.decode([idx]) for idx in top_indices_k]
                        print_summary(layer + 1, pos, token_str, entropy_bits, top_tokens_k, top_probs_k)
                        # Verbose console output removed to reduce noise - data still captured in files
                    
                    # FIXED: Also emit pure next-token record (last position only) 
                    # This avoids deflated entropy from tokens the model has already seen
                    last_pos = tokens.shape[1] - 1
                    last_logits = logits_all[last_pos]
                    last_entropy_bits = bits_entropy_from_logits(last_logits)
                    last_full_probs   = torch.softmax(last_logits, dim=0)
                    last_token_str = "⟨NEXT⟩"  # Pure next-token prediction, not the last prompt token
                    _, last_top_indices = torch.topk(last_logits, TOP_K_RECORD, largest=True, sorted=True)
                    last_top_probs = last_full_probs[last_top_indices]
                    last_top_tokens = [model.tokenizer.decode([idx]) for idx in last_top_indices]
                    # Update rolling window
                    top1_id = last_top_indices[0].item()
                    window_ids.append(top1_id)
                    if len(window_ids) > config.copy_window_k:
                        window_ids.pop(0)
                    
                    # ------------------------------------------------------------------- #
                    #  1.  Is this layer "copy-collapsed" (prompt echo)?   -> L_copy
                    #  2.  Does top-1 equal the ground-truth answer token? -> L_semantic
                    # ------------------------------------------------------------------- #
                    copy_collapse = detect_copy_collapse_id_subseq(
                        last_logits,
                        ctx_ids,
                        window_ids,
                        copy_threshold=config.copy_threshold,
                        copy_margin=config.copy_margin,
                    )
                    if copy_collapse and is_pure_whitespace_or_punct(last_top_tokens[0]):
                        copy_collapse = False
                    entropy_collapse = last_entropy_bits <= 1.0
                    is_answer = is_semantic_top1(last_top_tokens[0], "Berlin")
                    # Additional §1.3 metrics (factored helper)
                    metrics = compute_next_token_metrics(
                        last_full_probs, top1_id, final_probs, first_ans_id, topk_cum=5
                    )

                    # Cosine to final direction (PROJECT_NOTES §1.5)
                    _curr_norm = torch.norm(last_logits) + 1e-12
                    cos_to_final = torch.dot((last_logits / _curr_norm), final_dir).item()

                    record_extra = {
                        "copy_collapse": copy_collapse,
                        "entropy_collapse": entropy_collapse,
                        "is_answer": is_answer,
                        **metrics,
                        "cos_to_final": cos_to_final,
                    }
                    
                    # Collect for L_copy/L_semantic computation
                    collected_pure_records.append({
                        "layer": layer + 1,
                        "copy_collapse": copy_collapse,
                        "entropy_collapse": entropy_collapse,
                        "is_answer": is_answer,
                        "kl_to_final_bits": metrics["kl_to_final_bits"],
                        "answer_rank": metrics["answer_rank"],
                    })
                    
                    print_summary(layer + 1, last_pos, last_token_str, last_entropy_bits, last_top_tokens, last_top_probs, is_pure_next_token=True, extra=record_extra)

                    # Dual-lens sanity at selected layers
                    if (RAW_LENS_MODE != "off") and _should_sample(layer + 1):
                        record_dual_lens_sample(
                            json_data["raw_lens_check"],
                            layer_out_idx=layer + 1,
                            last_logits_norm=last_logits,
                            resid_raw_last_vec=resid_raw[0, last_pos, :],
                            W_U=analysis_W_U,
                            b_U=analysis_b_U,
                            force_fp32_unembed=(config.fp32_unembed or USE_FP32_UNEMBED),
                            tokenizer=model.tokenizer,
                            final_probs=final_probs,
                            first_ans_id=first_ans_id,
                            ground_truth=ground_truth,
                        )
                    
                    # --- free residual for this layer -----------------------------------
                    del resid
                    del residual_cache[f'blocks.{layer}.hook_resid_post']
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Summarize collapse and thresholds via helper
                diag.update(
                    summarize_pure_records(
                        collected_pure_records,
                        copy_threshold=config.copy_threshold,
                        copy_window_k=getattr(config, "copy_window_k", 1),
                        copy_match_level="id_subsequence",
                    )
                )
                json_data["diagnostics"] = diag

                # Summarize raw-vs-norm sanity samples (PROJECT_NOTES §1.4)
                try:
                    json_data["raw_lens_check"]["summary"] = summarize_raw_lens_check(
                        json_data["raw_lens_check"]["samples"]
                    )
                except Exception:
                    pass
                
            finally:
                # Clean up hooks and cache
                detach_hooks(hooks)
                
                # Aggressively free memory
                residual_cache.clear()  # free the dict but keep the name alive
                hooks.clear()  # Keep the variable, just empty it
                gc.collect()  # Force garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Ensure all CUDA operations are complete
            
            # === stop capturing large activations – probes don't need them =======


            # Let's also see what the actual model would predict (final layer)
            print("=" * 60)
            print("ACTUAL MODEL PREDICTION (for comparison):")

            # final_logits already computed above; ensure float
            final_logits = final_logits.float()
            _, final_top_indices = torch.topk(final_logits, 20, largest=True, sorted=True)
            final_full_probs = torch.softmax(final_logits, dim=0)
            final_top_probs = final_full_probs[final_top_indices]
            
            # Calculate final entropy via zero-safe helper
            final_entropy_bits = bits_entropy_from_logits(final_logits)
            
            # Collect final prediction data
            final_record = {
                "type": "final_prediction",
                "entropy": final_entropy_bits,
                "topk": [[model.tokenizer.decode([idx]), prob.item()] for prob, idx in zip(final_top_probs, final_top_indices)]
            }
            json_data["final_prediction"] = final_record
            
            # Emit additional probing records for test prompts
            # Process test prompts in smaller batches to reduce memory usage
            test_prompts = [
                "Berlin is the capital of",
                "Germany's capital city is called simply",
                "The capital city of Germany is named simply",
                "Germany has its capital at the city called simply",
                "In Germany the capital city is simply",            
                "Germany's capital city is called",
                "The capital city of Germany is named",
                "Germany has its capital at",
                "In Germany the capital city is known as",
                "Give the country name only, plain text. Berlin is the capital of",
                "Give the city name only, plain text. Germany's capital city is called",
                "Give the city name only, plain text. The capital city of Germany is named simply",
                "Give the city name only, plain text. Germany has its capital at",
                "Give the city name only, plain text. In Germany, the capital city is known as",
            ]
            
            # Process test prompts one at a time to minimize memory usage
            for test_prompt in test_prompts:
                test_tokens = model.to_tokens(test_prompt)      # let Accelerate move it
                
                test_logits = model(test_tokens)
                _, test_top_indices = torch.topk(test_logits[0, -1, :], 10, largest=True, sorted=True)
                test_full_probs = torch.softmax(test_logits[0, -1, :], dim=0)
                test_top_probs = test_full_probs[test_top_indices]
                
                # Calculate entropy for this prompt efficiently (convert to bits)
                test_log_probs = torch.log_softmax(test_logits[0, -1, :], dim=0).to(torch.float32)
                test_entropy_nats = -torch.sum(torch.exp(test_log_probs) * test_log_probs)
                test_entropy_bits = max(test_entropy_nats.item() / math.log(2), 0.0)  # Prevent negative zero
                # Collect test prompt data
                probe_record = {
                    "type": "test_prompt",
                    "prompt": test_prompt,
                    "entropy": test_entropy_bits,
                    "topk": [[model.tokenizer.decode([idx]), prob.item()] for prob, idx in zip(test_top_probs, test_top_indices)]
                }
                json_data["test_prompts"].append(probe_record)
                
                # Clean up tensors immediately
                del test_tokens, test_logits, test_top_indices, test_full_probs, test_top_probs, test_log_probs
            
            # Emit temperature exploration records
            # Note: Using consistent prompt for temperature exploration to maintain comparability
            temp_test_prompt = "Give the city name only, plain text. The capital of Germany is called simply"
            temp_tokens = model.to_tokens(temp_test_prompt)      # let Accelerate move it
            
            # Single forward pass - then rescale for different temperatures
            base_logits = model(temp_tokens)[0, -1, :]
            
            temperatures = [0.1, 2.0]
            
            for temp in temperatures:
                # Compute for temperature and emit JSONL record
                
                # Rescale existing logits instead of new forward pass (cast to float32 for numerical stability)
                scaled_logits = (base_logits / temp).float()
                _, temp_top_indices = torch.topk(scaled_logits, 15, largest=True, sorted=True)
                temp_full_probs = torch.softmax(scaled_logits, dim=0)
                temp_top_probs = temp_full_probs[temp_top_indices]
                
                # Calculate entropy at this temperature efficiently (convert to bits)
                temp_log_probs = torch.log_softmax(scaled_logits, dim=0).to(torch.float32)
                temp_entropy_nats = -torch.sum(torch.exp(temp_log_probs) * temp_log_probs)
                temp_entropy_bits = max(temp_entropy_nats.item() / math.log(2), 0.0)  # Prevent negative zero
                # Collect temperature exploration data
                temp_record = {
                    "type": "temperature_exploration",
                    "temperature": temp,
                    "entropy": temp_entropy_bits,
                    "topk": [[model.tokenizer.decode([idx]), prob.item()] for prob, idx in zip(temp_top_probs, temp_top_indices)]
                }
                json_data["temperature_exploration"].append(temp_record)
            
            # Clean up temperature exploration tensors
            del temp_tokens, base_logits
        
        print("=== END OF INSPECTING ==============\n")
        
        # Collect model stats data (architecture already detected above)
        stats_record = {
            "type": "model_stats",
            "num_layers": model.cfg.n_layers,
            "d_model": model.cfg.d_model,
            "n_heads": model.cfg.n_heads,
            "d_vocab": model.cfg.d_vocab,
            "n_ctx": model.cfg.n_ctx,
            "architecture": detected_architecture or 'unknown'
        }
        json_data["model_stats"] = stats_record
        
        # Clean up model to free memory (though process will end anyway)
        del model
        gc.collect()  # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return json_data

    try:
        # Run the experiment and let output print normally
        json_data = evaluate_model()

        # In self-test mode, do not write JSON/CSV artifacts
        if config.self_test:
            return json_data

        # Extract file paths
        meta_filepath, csv_filepath, pure_csv_filepath = output_files

        # Write CSV files FIRST (they need the full record lists)
        write_csv_files(json_data, csv_filepath, pure_csv_filepath, TOP_K_VERBOSE)

        # Strip bulky per-token records from JSON to keep it compact
        json_data_compact = {k: v for k, v in json_data.items()
                             if k not in ("records", "pure_next_token_records")}

        # Write compact JSON metadata
        with open(meta_filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data_compact, f, ensure_ascii=False, indent=2)

        return json_data_compact
        
    except Exception as e:
        error_msg = f"ERROR evaluating {model_id}: {str(e)}"
        print(error_msg)
        raise

## csv writing helpers moved to layers_core.csv_io

def run_single_model(model_id):
    """Run experiment for a single model - used when called as subprocess"""
    print(f"\n{'='*80}")
    print(f"🚀 Starting subprocess for model: {model_id}")
    print(f"{'='*80}")
    
    # Set memory limits BEFORE any CUDA operations
    if torch.cuda.is_available():
        try:
            # Use 85% of GPU memory (increased from 80% since we're managing memory better)
            # torch.cuda.set_per_process_memory_fraction(0.85)
            # Also set environment variable for better memory management
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        except AttributeError:
            pass
    
    # Generate filename components
    clean_name = clean_model_name(model_id)
    meta_filename = f"output-{clean_name}.json"
    csv_filename  = f"output-{clean_name}-records.csv"
    pure_csv_filename = f"output-{clean_name}-pure-next-token.csv"

    # Determine output directory (parent may pass --out_dir)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if CLI_ARGS.out_dir:
        out_dir = CLI_ARGS.out_dir
    else:
        # For self-test, avoid rotating/creating run-latest; write nowhere
        if CLI_ARGS.self_test:
            out_dir = script_dir
        else:
            # Stand-alone invocation: create its own run-latest directory
            out_dir = setup_run_latest_directory(script_dir)
    # Ensure directory exists
    os.makedirs(out_dir, exist_ok=True)

    meta_filepath = os.path.join(out_dir, meta_filename)
    csv_filepath  = os.path.join(out_dir, csv_filename)
    pure_csv_filepath = os.path.join(out_dir, pure_csv_filename)
    
    try:
        # Resolve device: auto-pick if requested
        chosen_device = CLI_ARGS.device
        debug_info = None
        if CLI_ARGS.device == "auto":
            sel = select_best_device(model_id)
            if sel is None:
                print(f"⛔ No suitable device fits the model: {model_id}")
                # Write a minimal meta file noting skip
                with open(meta_filepath, 'w', encoding='utf-8') as f:
                    f.write(f"SKIPPED (no device fit) FOR {model_id}\n")
                    f.write(f"Timestamp: {datetime.now().strftime('%Y%m%d%H%M%S')}\n")
                return False
            dev, dtype, debug_info = sel
            chosen_device = dev
            print("📐 Device decision:")
            print(f"   device={dev} dtype={dtype} est_peak={debug_info.get('est_peak')}B available={debug_info.get('available')}B")

        # Run the experiment
        cfg = ExperimentConfig(
            device=chosen_device,
            fp32_unembed=CLI_ARGS.fp32_unembed,
            keep_residuals=CLI_ARGS.keep_residuals,
            copy_threshold=CLI_ARGS.copy_threshold,
            copy_margin=CLI_ARGS.copy_margin,
            out_dir=out_dir,
            self_test=CLI_ARGS.self_test,
        )
        data = run_experiment_for_model(model_id, (meta_filepath, csv_filepath, pure_csv_filepath), cfg)

        if CLI_ARGS.self_test:
            print("✅ Self-test complete. No artifacts written (by design).")
        else:
            print(f"✅ Experiment complete. JSON metadata saved to: {meta_filepath}")
            print(f"✅ Records CSV saved to: {csv_filepath}")
            print(f"✅ Pure next-token CSV saved to: {pure_csv_filepath}")
        return True
        
    except Exception as e:
        error_msg = f"❌ Failed to evaluate {model_id}: {str(e)}"
        print(error_msg)
        
        # Still save error output
        with open(meta_filepath, 'w', encoding='utf-8') as f:
            f.write(f"EXPERIMENT FAILED FOR {model_id}\n")
            f.write(f"Error: {str(e)}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y%m%d%H%M%S')}\n")
        return False

# CLI -----------------------------------------------------------------------
def parse_cli():
    p = argparse.ArgumentParser(description="Layer-by-layer logit-lens sweep")
    p.add_argument("--device",
                   default="auto",
                   choices=["auto", "cuda", "mps", "cpu"],
                   help="compute device to run on (default: auto picks best fit)")
    p.add_argument("--fp32-unembed",
                   action="store_true",
                   help="Use FP32 shadow unembedding for analysis-only decoding (do not mutate model params)")
    p.add_argument("--keep-residuals",
                   action="store_true",
                   help="Dump full residual tensors; if absent, keep only per-layer logits")
    p.add_argument("--copy-threshold", type=float, default=0.95,
                   help="Minimum P(top-1) for copy collapse")
    p.add_argument("--copy-margin", type=float, default=0.10,
                   help="Require P(top-1) − P(top-2) > margin for copy collapse")
    p.add_argument("model_id", nargs="?", default=None,
                   help="Model ID for single-run (when invoking as subprocess)")
    p.add_argument("--out_dir",
                   default=None,
                   help="Output directory to save CSV & JSON results (default: current script directory or value forwarded by parent launcher)")
    p.add_argument("--self-test",
                   action="store_true",
                   help="Run KL sanity test to validate normalization scaling (PROJECT_NOTES.md section 1.1). Can also run standalone: python kl_sanity_test.py MODEL_ID")
    return p.parse_args()


# Parse once and promote to module-global so run_single_model and main can see it
CLI_ARGS = parse_cli()

def main():
    """Main function to launch separate processes for each model"""
    # If a model_id was provided, run in single-model mode
    if CLI_ARGS.model_id:
        success = run_single_model(CLI_ARGS.model_id)
        sys.exit(0 if success else 1)
    
    # Main process - launch subprocess for each model
    print(f"🎯 Starting experiment launcher for {len(CONFIRMED_MODELS)} models...")
    print("Each model will run in a separate process for clean memory isolation.")

    script_path = os.path.abspath(__file__)

    # Set up run-latest directory with automatic rotation
    run_dir = setup_run_latest_directory(os.path.dirname(script_path))

    # Create empty markdown files for evaluation reports
    print(f"📝 Creating empty evaluation markdown files...")
    for model_id in CONFIRMED_MODELS:
        clean_name = clean_model_name(model_id)
        eval_md_path = os.path.join(run_dir, f"evaluation-{clean_name}.md")
        with open(eval_md_path, 'w', encoding='utf-8') as f:
            f.write(f"# Evaluation Report: {model_id}\n\n")
            f.write(f"*Run executed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

        print(f"   📄 Created: evaluation-{clean_name}.md")

    results = []
    launched_models = []
    
    for i, model_id in enumerate(CONFIRMED_MODELS, 1):
        print(f"\n{'='*80}")
        print(f"📋 Launching process {i}/{len(CONFIRMED_MODELS)}: {model_id}")
        print(f"{'='*80}")
        
        try:
            # Decide device per model (auto by default)
            if CLI_ARGS.device == "auto":
                sel = select_best_device(model_id)
                if sel is None:
                    print(f"⛔ Skipping {model_id}: no device fits (estimates)")
                    results.append((model_id, "SKIPPED_NO_FIT"))
                    continue
                dev, dtype, debug = sel
                print(f"📐 Decision for {model_id}: device={dev} dtype={dtype} est_peak={debug.get('est_peak')} avail={debug.get('available')}")
                chosen_device = dev
            else:
                chosen_device = CLI_ARGS.device

            # Launch subprocess
            cmd = [
                sys.executable,
                script_path,
                "--device", chosen_device,   # forward the per-model device
                "--out_dir", run_dir,        # ensure all subprocesses share same dir
            ]
            if CLI_ARGS.fp32_unembed:
                cmd.append("--fp32-unembed")   # forward the fp32-unembed flag
            if CLI_ARGS.keep_residuals:
                cmd.append("--keep-residuals") # forward the keep-residuals flag
            # Forward copy-collapse parameters
            cmd.extend(["--copy-threshold", str(CLI_ARGS.copy_threshold)])
            cmd.extend(["--copy-margin", str(CLI_ARGS.copy_margin)])
            cmd.append(model_id)
            
            result = subprocess.run(cmd, capture_output=False, text=True, check=False)

            if result.returncode == 0:
                print(f"✅ Process {i} completed successfully")
                results.append((model_id, "SUCCESS"))
                launched_models.append(model_id)
            else:
                print(f"❌ Process {i} failed with return code {result.returncode}")
                results.append((model_id, "FAILED"))
                
        except Exception as e:
            error_msg = f"Failed to launch subprocess for {model_id}: {str(e)}"
            print(f"❌ {error_msg}")
            results.append((model_id, f"LAUNCH_FAILED: {str(e)}"))
    
    # Summary
    print(f"\n{'='*80}")
    print("🎉 All model processes completed!")
    print(f"📁 Output files saved in: {run_dir}")
    
    print("\n📊 Results Summary:")
    for model_id, status in results:
        clean_name = clean_model_name(model_id)
        status_emoji = "✅" if status == "SUCCESS" else "❌"
        print(f"   {status_emoji} {clean_name}: {status}")
    
    print(f"\n📄 Expected output files:")
    for model_id in (launched_models if len(launched_models) > 0 else CONFIRMED_MODELS):
        clean_name = clean_model_name(model_id)
        print(f"   {os.path.join(run_dir, f'output-{clean_name}.json')}")
        print(f"   {os.path.join(run_dir, f'output-{clean_name}-records.csv')}")
        print(f"   {os.path.join(run_dir, f'output-{clean_name}-pure-next-token.csv')} ")
        print(f"   {os.path.join(run_dir, f'evaluation-{clean_name}.md')}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
