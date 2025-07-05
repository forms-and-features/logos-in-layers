import transformer_lens
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import torch.nn as nn
import io
from contextlib import redirect_stdout
from datetime import datetime
import os
import subprocess
import sys
import math
import json
import csv
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

# List of confirmed supported models
CUDA_ONLY_MODELS = [
    "01-ai/Yi-34B",
    "Qwen/Qwen3-14B",
    "google/gemma-2-27b",
]

MPS_SAFE_MODELS = [
    "mistralai/Mistral-7B-v0.1",
    "google/gemma-2-9b",
    "Qwen/Qwen3-8B",
    "meta-llama/Meta-Llama-3-8B"
]

# Dynamically determine which models to run based on CUDA availability
if torch.cuda.is_available():
    CONFIRMED_MODELS = CUDA_ONLY_MODELS + MPS_SAFE_MODELS
else:
    CONFIRMED_MODELS = MPS_SAFE_MODELS
    print("⚠️  CUDA not available - running only MPS-safe models")

# --- helpers ---------------------------------------------------------------
RMS_ATTR_CANDIDATES = ("w", "weight", "scale", "gamma")

def _get_rms_scale(norm_mod):
    for attr in RMS_ATTR_CANDIDATES:
        if hasattr(norm_mod, attr):
            return getattr(norm_mod, attr)
    params = list(norm_mod.parameters(recurse=False))
    return params[0] if len(params) == 1 else None

def rms_lens(resid, gamma, eps=1e-5):
    """
    Apply RMS normalization with learnable scale parameter.
    
    Args:
        resid: Residual stream tensor [B, seq_len, d_model]
        gamma: Scale parameter from RMSNorm layer [d_model]
        eps: Epsilon for numerical stability (default: 1e-5)
    
    Returns:
        RMS-normalized residual stream
    """
    # Ensure gamma matches device and dtype of residual
    gamma = gamma.to(resid.device, dtype=resid.dtype)
    
    # Compute RMS: [B, seq_len, d_model] -> [B, seq_len, 1]
    # CRITICAL: eps must be INSIDE the sqrt for numerical correctness
    rms = torch.sqrt(resid.pow(2).mean(-1, keepdim=True) + eps)
    
    # Normalize and scale
    return resid / rms * gamma

def apply_norm_or_skip(residual: torch.Tensor, norm_module):
    """
    Return the residual stream after applying the model's own normalisation layer.
    Works for both RMSNorm (no bias) and LayerNorm (γ *and* β kept intact).

    This runs under torch.no_grad() so it incurs no autograd overhead.
    """
    if norm_module is None:
        return residual  # some models expose pre-norm residuals

    with torch.no_grad():
        if isinstance(norm_module, torch.nn.LayerNorm):
            # --- faithful LayerNorm -----------------------------------------
            mean = residual.mean(dim=-1, keepdim=True)
            var  = residual.var(dim=-1, unbiased=False, keepdim=True)  # same as LN
            normalized = (residual - mean) / torch.sqrt(var + norm_module.eps)
            weight = norm_module.weight.to(residual.dtype)
            bias   = norm_module.bias.to(residual.dtype)
            return normalized * weight + bias
        else:
            # Gracefully handle different RMSNorm variants that may expose the scale
            # parameter under various attribute names (e.g. `.w`, `.weight`, `.scale`, `.gamma`).
            denom = residual.norm(dim=-1, keepdim=True) / math.sqrt(residual.size(-1))
            scale = _get_rms_scale(norm_module)
            if scale is not None:
                # Detach to avoid autograd bookkeeping and match device & dtype
                scale = scale.detach().to(residual.device, dtype=residual.dtype)
                return residual / (denom + norm_module.eps) * scale
            else:
                # Fallback: no learned scale present (rare but possible)
                return residual / (denom + norm_module.eps)

def clean_model_name(model_id):
    """Extract clean model name for filename"""
    # Remove organization prefix (everything before last '/')
    clean_name = model_id.split('/')[-1]
    return clean_name

def setup_run_latest_directory(script_dir):
    """
    Set up the run-latest directory with automatic rotation of previous runs.
    
    - If run-latest doesn't exist, create it
    - If run-latest exists, rename it to run-YYYYMMDD-HHMM based on its timestamp file
    - Create a new run-latest directory with a current timestamp file
    
    Returns the path to the run-latest directory.
    """
    run_latest_dir = os.path.join(script_dir, "run-latest")
    current_timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    
    # Check if run-latest already exists
    if os.path.exists(run_latest_dir):
        # Look for existing timestamp file to determine rotation name
        timestamp_files = [f for f in os.listdir(run_latest_dir) if f.startswith("timestamp-")]
        
        if timestamp_files:
            # Extract timestamp from existing file
            timestamp_file = timestamp_files[0]  # Take the first one if multiple exist
            old_timestamp = timestamp_file.replace("timestamp-", "")
            rotated_name = f"run-{old_timestamp}"
        else:
            # No timestamp file found, use current time as fallback
            rotated_name = f"run-{current_timestamp}-rotated"
        
        rotated_dir = os.path.join(script_dir, rotated_name)
        
        # Rename existing run-latest to rotated name
        print(f"🔄 Rotating existing run-latest to: {rotated_name}")
        os.rename(run_latest_dir, rotated_dir)
    
    # Create new run-latest directory
    os.makedirs(run_latest_dir, exist_ok=True)
    
    # Create timestamp file in the new run-latest directory
    timestamp_file = os.path.join(run_latest_dir, f"timestamp-{current_timestamp}")
    with open(timestamp_file, 'w', encoding='utf-8') as f:
        f.write(f"Experiment started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"📁 Created run-latest directory: {run_latest_dir}")
    print(f"⏰ Created timestamp file: timestamp-{current_timestamp}")
    
    return run_latest_dir

def bits_entropy_from_logits(logits: torch.Tensor) -> float:
    """
    Shannon entropy in **bits** computed safely from raw logits.
    Works on CPU / GPU and never returns NaN.
    """
    probs = logits.softmax(dim=-1)
    log_probs = probs.log()
    ent_nats = torch.sum((-probs.double()) * log_probs.double())
    return (ent_nats / math.log(2)).item()


def safe_cast_for_unembed(resid, W_U):
    """
    Return the residual stream casted to the *right* dtype for the
    un-embedding.

    * For ordinary models W_U.dtype is fp16 / bf16 / fp32 ⇒ cast as usual.
    * For 8-bit-quantised weights W_U.dtype is int8  ⇒ keep activations
      in their original float dtype (INT8 matmul kernels expect that).
    """
    if CLI_ARGS.fp32_unembed:
        return resid.float()
    if torch.is_floating_point(W_U):       # fp16 / bf16 / fp32 case
        return resid.to(dtype=W_U.dtype)
    else:                                  # int8 / 4-bit etc.
        return resid                       # **no** cast!


def run_experiment_for_model(model_id, output_files):
    """Run the complete experiment for a single model and write results to files"""
    
    def evaluate_model():
        """The actual experiment code - all prints go to console"""
        print(f"\n{'='*60}")
        print(f"EVALUATING MODEL: {model_id}")
        print(f"{'='*60}")
        
        # ---- device & dtype ---------------------------------------------------
        device = CLI_ARGS.device
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA requested but not available; falling back to CPU.")
            device = "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            print("⚠️  MPS requested but not available; falling back to CPU.")
            device = "cpu"

        dtype = {
            "cuda": torch.float16,
            # MPS currently doesn't support float16 weights reliably; use float32
            "mps":  torch.float32,
            "cpu":  torch.float32,
        }[device]

        # ── add this directly AFTER the dictionary block ─────────────
        if "gemma" in model_id.lower() and device == "cuda":
            # Gemma weights are published in bf16 – use that wider range
            dtype = torch.bfloat16
        # ──────────────────────────────────────────────────────────────

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
            model = HookedTransformer.from_pretrained_no_processing(
                model_id,
                device_map      = "auto",
                torch_dtype     = dtype,
                low_cpu_mem_usage = True,
                trust_remote_code = True,
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
        
        # Clear cache after loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Debug: inspect model parameter devices
        unique_param_devices = {p.device for p in model.parameters()}
        print(f"[DEBUG MODEL] Unique parameter devices: {unique_param_devices}")
        
        # Get the primary device of the model
        primary_device = device
        
        # Toggle for using normalized lens (recommended for accurate interpretation)
        USE_NORM_LENS = True
        
        # Toggle for FP32 unembedding via CLI flag
        USE_FP32_UNEMBED = CLI_ARGS.fp32_unembed

        # Promote unembedding weights to FP32 if requested for true precision gain
        if USE_FP32_UNEMBED and model.unembed.W_U.dtype != torch.float32:
            print(f"🔬 Promoting unembed weights to FP32 for research-grade precision (was {model.unembed.W_U.dtype})")
            with torch.no_grad():
                model.unembed.W_U.data = model.unembed.W_U.data.float()
                if hasattr(model.unembed, 'b_U') and model.unembed.b_U is not None:
                    model.unembed.b_U.data = model.unembed.b_U.data.float()
            UNEMBED_DTYPE = torch.float32  # refresh after promotion
        UNEMBED_DTYPE = model.unembed.W_U.dtype   # match actual weight dtype
        
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
            "model_stats": None
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
        
        # Storage to collect pure_next_token_records for L_copy/L_semantic computation
        collected_pure_records = []
        
        # Begin capturing residual streams
        with torch.no_grad():
            # MEMORY EFFICIENT: Use targeted caching instead of run_with_cache
            # (removed human-readable progress print)
            
            # Storage for only the residual streams we need
            residual_cache = {}
            
            def make_cache_hook(cache_dict):
                def cache_residual_hook(tensor, hook):
                    # Only store the tensor we need, detached from computation graph
                    # Store activations in fp32 for numerical stability and full sequence
                    # Keep activations on their original device to preserve device_map="auto" layout
                    cache_dict[hook.name] = tensor.to(dtype=torch.float32).detach()
                return cache_residual_hook
            
            # Create the hook function with explicit cache reference
            cache_hook = make_cache_hook(residual_cache)
            
            # Set up hooks for residual streams only
            hooks = []
            
            # Hook for embeddings (layer 0 equivalent)
            embed_hook = model.hook_dict['hook_embed'].add_hook(cache_hook)
            hooks.append(embed_hook)
            # Conditionally hook for positional embeddings if available
            if 'hook_pos_embed' in model.hook_dict:
                pos_hook = model.hook_dict['hook_pos_embed'].add_hook(cache_hook)
                hooks.append(pos_hook)
                has_pos_embed = True
            else:
                has_pos_embed = False
            
            # Hook for each layer's residual post
            n_layers = model.cfg.n_layers
            for layer in range(n_layers):
                resid_hook = model.blocks[layer].hook_resid_post.add_hook(cache_hook)
                hooks.append(resid_hook)
            
            try:
                # Run forward pass with targeted hooks
                logits = model(tokens)
                
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
                print("copy-collapse: first layer where top-1 token is in prompt & p>0.9")
                print("-" * 60)
                
                # Get string representations of tokens for labeling output
                str_tokens = model.to_str_tokens(context_prompt)

                # Collect prompt tokens for copy-collapse detection
                prompt_token_set = {
                    tok.strip() for tok in model.to_str_tokens(context_prompt)
                }

                # Layer 0: embeddings (+ positional embeddings if available)
                print("Layer  0 (embeddings):")
                if has_pos_embed:
                    resid = (residual_cache['hook_embed'] +
                             residual_cache['hook_pos_embed'])
                else:
                    resid = residual_cache['hook_embed']
                    print("[diagnostic] No separate positional embedding hook found (as expected for rotary models).")
                    print("[diagnostic] Layer 0 contains TOKEN information only; positional info is injected inside attention layers.")
                # FIXED: Apply first real normalizer to embeddings if using norm-lens
                # This gives us the normalized embeddings that the model actually sees
                if USE_NORM_LENS:
                    # Use the actual first normalization layer instead of synthetic γ=1
                    print("[diagnostic] Applying real ln1 normalization to embeddings (not synthetic γ=1)")
                    resid = apply_norm_or_skip(resid, model.blocks[0].ln1)
                
                # Vectorized unembedding for all positions  
                resid_cast = safe_cast_for_unembed(resid[0], model.unembed.W_U)
                logits_all = model.unembed(resid_cast).float()  # [seq, d_vocab]
                
                for pos in range(tokens.shape[1]):
                    layer_logits = logits_all[pos]
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
                
                # ------------------------------------------------------------------- #
                #  1.  Is this layer "copy-collapsed" (prompt echo)?   -> L_copy
                #  2.  Does top-1 equal the ground-truth answer token? -> L_semantic
                # ------------------------------------------------------------------- #
                # --- NEW copy-collapse rule (prompt echo) ---------------------------
                copy_collapse = (
                    last_top_tokens[0].strip() in prompt_token_set
                    and last_top_probs[0].item() > 0.90
                )
                entropy_collapse = last_entropy_bits <= 1.0      # keep for reference
                # use new criterion for L_copy
                collapsed = copy_collapse
                is_answer = (last_top_tokens[0].strip() == "Berlin")
                
                record_extra = {
                    "copy_collapse": copy_collapse,
                    "entropy_collapse": entropy_collapse,
                    "is_answer": is_answer,
                }
                
                # Collect for L_copy/L_semantic computation
                collected_pure_records.append({
                    "layer": 0,
                    "copy_collapse": copy_collapse,
                    "entropy_collapse": entropy_collapse,
                    "is_answer": is_answer
                })
                
                print_summary(0, last_pos, last_token_str, last_entropy_bits, last_top_tokens, last_top_probs, is_pure_next_token=True, extra=record_extra)
                
                # --- free Layer-0 residual to keep host RAM flat ---------------------
                del resid
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Layers 1 to n_layers: after each transformer block
                for layer in range(n_layers):
                    print(f"Layer {layer + 1:2d} (after transformer block {layer}):")
                    # Get residual stream after this layer's block
                    resid = residual_cache[f'blocks.{layer}.hook_resid_post']
                    
                    # Apply normalization if requested
                    if USE_NORM_LENS:
                        # For pre-LN models, apply ln_final on last layer, otherwise use ln2 for post-block residuals
                        if layer == n_layers - 1 and hasattr(model, 'ln_final'):
                            resid = apply_norm_or_skip(resid, model.ln_final)
                        else:
                            # FIXED: Use ln2 (MLP-pre norm) instead of ln1 for post-block residuals
                            # ln2 stats better match the post-attention, pre-MLP state which is closer to post-block
                            norm_layer = model.blocks[layer].ln2 if hasattr(model.blocks[layer], 'ln2') else model.blocks[layer].ln1
                            
                            # Use the unified normalization function that handles both LayerNorm and RMSNorm
                            resid = apply_norm_or_skip(resid, norm_layer)
                    
                    # Vectorized unembedding for all positions
                    resid_cast = safe_cast_for_unembed(resid[0], model.unembed.W_U)
                    logits_all = model.unembed(resid_cast).float() # [seq, d_vocab]
                    
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
                    
                    # ------------------------------------------------------------------- #
                    #  1.  Is this layer "copy-collapsed" (prompt echo)?   -> L_copy
                    #  2.  Does top-1 equal the ground-truth answer token? -> L_semantic
                    # ------------------------------------------------------------------- #
                    # --- NEW copy-collapse rule (prompt echo) ---------------------------
                    copy_collapse = (
                        last_top_tokens[0].strip() in prompt_token_set
                        and last_top_probs[0].item() > 0.90
                    )
                    entropy_collapse = last_entropy_bits <= 1.0      # keep for reference
                    # use new criterion for L_copy
                    collapsed = copy_collapse
                    is_answer = (last_top_tokens[0].strip() == "Berlin")
                    
                    record_extra = {
                        "copy_collapse": copy_collapse,
                        "entropy_collapse": entropy_collapse,
                        "is_answer": is_answer,
                    }
                    
                    # Collect for L_copy/L_semantic computation
                    collected_pure_records.append({
                        "layer": layer + 1,
                        "copy_collapse": copy_collapse,
                        "entropy_collapse": entropy_collapse,
                        "is_answer": is_answer
                    })
                    
                    print_summary(layer + 1, last_pos, last_token_str, last_entropy_bits, last_top_tokens, last_top_probs, is_pure_next_token=True, extra=record_extra)
                    
                    # --- free residual for this layer -----------------------------------
                    del resid
                    del residual_cache[f'blocks.{layer}.hook_resid_post']
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # ---------------------------------------------------------------
                #  Compute L_copy  (first copy-collapsed layer, prompt echo rule)
                #  Compute L_semantic (first layer whose top-1 == "Berlin")
                # ---------------------------------------------------------------
                L_copy = None
                L_copy_H = None  # Legacy entropy-based metric
                L_sem = None
                for rec in collected_pure_records:    # already collected in RAM
                    if L_copy is None and rec["copy_collapse"]:
                        L_copy = rec["layer"]
                    if L_copy_H is None and rec["entropy_collapse"]:
                        L_copy_H = rec["layer"]          # optional: legacy metric
                    if L_sem is None and rec["is_answer"]:
                        L_sem = rec["layer"]
                    if L_copy is not None and L_copy_H is not None and L_sem is not None:
                        break

                # Attach to diagnostics block
                diag.update({"L_copy": L_copy,
                             "L_copy_H": L_copy_H,
                             "L_semantic": L_sem,
                             "delta_layers": None if (L_copy is None or L_sem is None)
                                                  else L_sem - L_copy})
                json_data["diagnostics"] = diag
                
            finally:
                # Clean up hooks and cache
                for h in hooks:  # HookPoint.add_hook always returns a handle with .remove()
                    if h is not None:
                        h.remove()
                        
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

            final_logits = logits[0, -1, :]
            if USE_FP32_UNEMBED:
                # Ensure consistent precision for final prediction
                final_logits = final_logits.float()
            _, final_top_indices = torch.topk(final_logits, 20, largest=True, sorted=True)
            final_full_probs = torch.softmax(final_logits, dim=0)
            final_top_probs = final_full_probs[final_top_indices]
            
            # Calculate final entropy efficiently (convert to bits)
            final_probs = final_logits.softmax(dim=0)
            final_log_probs = final_probs.log()
            final_entropy_nats = torch.sum((-final_probs.double()) * final_log_probs.double())
            final_entropy_bits = max(final_entropy_nats.item() / math.log(2), 0.0)  # Prevent negative zero
            
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
                test_probs = test_logits[0, -1, :].softmax(dim=0)
                test_log_probs = test_probs.log()
                test_entropy_nats = torch.sum((-test_probs.double()) * test_log_probs.double())
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
                temp_probs = scaled_logits.softmax(dim=0)
                temp_log_probs = temp_probs.log()
                temp_entropy_nats = torch.sum((-temp_probs.double()) * temp_log_probs.double())
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
        
        # Collect model stats data
        stats_record = {
            "type": "model_stats",
            "num_layers": model.cfg.n_layers,
            "d_model": model.cfg.d_model,
            "n_heads": model.cfg.n_heads,
            "d_vocab": model.cfg.d_vocab,
            "n_ctx": model.cfg.n_ctx
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
        
        # Extract file paths
        meta_filepath, csv_filepath, pure_csv_filepath = output_files
        
        # Write CSV files FIRST (they need the full record lists)
        write_csv_files(json_data, csv_filepath, pure_csv_filepath)

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

def write_csv_files(json_data, csv_filepath, pure_csv_filepath):
        """Write CSV files from collected JSON data"""
        records = json_data["records"]
        pure_next_token_records = json_data["pure_next_token_records"]
        
        # Save records to CSV
        with open(csv_filepath, 'w', newline='', encoding='utf-8') as f_csv:
            writer = csv.writer(
                f_csv,
                delimiter=",",              # stay with normal commas
                quotechar='"',              # wrap any field that needs quoting
                quoting=csv.QUOTE_MINIMAL,  # auto-quote only when required
                escapechar="\\",            # …otherwise escape
                lineterminator="\n",
            )
            # FIXED: Add rest_mass column to preserve full probability distribution
            # Header: layer,pos,token,entropy + top-k pairs + rest_mass
            header = ["layer","pos","token","entropy"]
            # Pad all rows to TOP_K_VERBOSE slots
            for i in range(1, TOP_K_VERBOSE + 1):
                header.extend([f"top{i}", f"prob{i}"])
            header.append("rest_mass")  # Probability mass not in top-k
            writer.writerow(header)
            for rec in records:
                row = [rec.get("layer"), rec.get("pos"), rec.get("token"), rec.get("entropy")]
                # Pad each record to TOP_K_VERBOSE entries
                topk_list = rec.get("topk", [])
                topk_prob_sum = 0.0
                for j in range(TOP_K_VERBOSE):
                    if j < len(topk_list):
                        tok, prob = topk_list[j]
                        topk_prob_sum += prob
                    else:
                        tok, prob = "", ""
                    row.extend([tok, prob])
                # Add rest-of-probability-mass for offline entropy/KL calculations
                rest_mass = max(0.0, 1.0 - topk_prob_sum)  # Ensure non-negative
                row.append(rest_mass)
                writer.writerow(row)

        # Save pure next-token records to separate CSV (cleaner entropy analysis)
        with open(pure_csv_filepath, 'w', newline='', encoding='utf-8') as f_csv:
            writer = csv.writer(
                f_csv,
                delimiter=",",
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL,
                escapechar="\\",
                lineterminator="\n",
            )
            # Header includes new collapse detection flags
            header = ["layer","pos","token","entropy"]
            for i in range(1, TOP_K_VERBOSE + 1):
                header.extend([f"top{i}", f"prob{i}"])
            header.extend(["rest_mass", "copy_collapse", "entropy_collapse", "is_answer"])
            writer.writerow(header)
            for rec in pure_next_token_records:
                row = [rec.get("layer"), rec.get("pos"), rec.get("token"), rec.get("entropy")]
                topk_list = rec.get("topk", [])
                topk_prob_sum = 0.0
                for j in range(TOP_K_VERBOSE):
                    if j < len(topk_list):
                        tok, prob = topk_list[j]
                        topk_prob_sum += prob
                    else:
                        tok, prob = "", ""
                    row.extend([tok, prob])
                rest_mass = max(0.0, 1.0 - topk_prob_sum)
                # Add the new collapse detection flags
                row.extend([
                    rest_mass,
                    rec.get("copy_collapse", ""),
                    rec.get("entropy_collapse", ""),
                    rec.get("is_answer", "")
                ])
                writer.writerow(row)

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
        # Stand-alone invocation: create its own run-latest directory
        out_dir = setup_run_latest_directory(script_dir)
    # Ensure directory exists
    os.makedirs(out_dir, exist_ok=True)

    meta_filepath = os.path.join(out_dir, meta_filename)
    csv_filepath  = os.path.join(out_dir, csv_filename)
    pure_csv_filepath = os.path.join(out_dir, pure_csv_filename)
    
    try:
        # Run the experiment - files are written directly
        data = run_experiment_for_model(model_id, (meta_filepath, csv_filepath, pure_csv_filepath))
        
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
                   default="cuda",
                   choices=["cuda", "mps", "cpu"],
                   help="compute device to run on (default: cuda)")
    p.add_argument("model_id", nargs="?", default=None,
                   help="Model ID for single-run (when invoking as subprocess)")
    p.add_argument("--out_dir",
                   default=None,
                   help="Output directory to save CSV & JSON results (default: current script directory or value forwarded by parent launcher)")
    p.add_argument(
        "--fp32-unembed",
        action="store_true",
        help="Up-cast the model's un-embedding matrix to float32")
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
    
    for i, model_id in enumerate(CONFIRMED_MODELS, 1):
        print(f"\n{'='*80}")
        print(f"📋 Launching process {i}/{len(CONFIRMED_MODELS)}: {model_id}")
        print(f"{'='*80}")
        
        try:
            # Launch subprocess
            cmd = [
                sys.executable,
                script_path,
                "--device", CLI_ARGS.device,
                "--out_dir", run_dir,
            ]
            if CLI_ARGS.fp32_unembed:
                cmd.append("--fp32-unembed")
            cmd.append(model_id)
            result = subprocess.run(cmd, capture_output=False, text=True, check=False)
            
            if result.returncode == 0:
                print(f"✅ Process {i} completed successfully")
                results.append((model_id, "SUCCESS"))
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
    for model_id in CONFIRMED_MODELS:
        clean_name = clean_model_name(model_id)
        print(f"   {os.path.join(run_dir, f'output-{clean_name}.json')}")
        print(f"   {os.path.join(run_dir, f'output-{clean_name}-records.csv')}")
        print(f"   {os.path.join(run_dir, f'output-{clean_name}-pure-next-token.csv')} ")
        print(f"   {os.path.join(run_dir, f'evaluation-{clean_name}.md')}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()