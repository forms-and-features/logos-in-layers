import transformer_lens
from transformer_lens import HookedTransformer
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

# Top-k settings for record emission
TOP_K_RECORD = 5    # number of tokens to record for non-verbose slots
TOP_K_VERBOSE = 20  # number of tokens to record for verbose slots and answer position

# Layer-by-layer prediction analysis with LayerNorm lens correction
# Toggle USE_NORM_LENS for raw vs normalized residual stream analysis

# List of confirmed supported models
CONFIRMED_MODELS = [
    "meta-llama/Meta-Llama-3-8B",      # 8B - Llama 3 Base
    "mistralai/Mistral-7B-v0.1",       # 7B - Mistral
    "google/gemma-2-9b",               # 9B - Gemma 2
    "Qwen/Qwen3-8B"                    # 8B - Qwen3
]

# --- helpers ---------------------------------------------------------------
RMS_ATTR_CANDIDATES = ("w", "weight", "scale", "gamma")

def _get_rms_scale(norm_mod):
    for attr in RMS_ATTR_CANDIDATES:
        if hasattr(norm_mod, attr):
            return getattr(norm_mod, attr)
    params = list(norm_mod.parameters(recurse=False))
    return params[0] if len(params) == 1 else None

def is_safe_layernorm(norm_mod):
    """Check if a normalization module is vanilla LayerNorm that's safe to apply"""
    return isinstance(norm_mod, nn.LayerNorm)

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

def apply_norm_or_skip(resid, norm_mod, layer_info=""):
    """
    Apply normalization safely, with support for both LayerNorm and RMSNorm.
    
    Args:
        resid: Residual stream tensor
        norm_mod: Normalization module (LayerNorm, RMSNorm, etc.)
        layer_info: String for debugging/warning messages
    
    Returns:
        Normalized residual stream (or original if skipped)
    Note: All supported norm types are handled above; fallback below is unreachable.
    """
    # Vanilla LayerNorm - apply as before
    if isinstance(norm_mod, nn.LayerNorm):
        return norm_mod(resid)
    
    # RMSNorm - robust scale detection and normalization
    norm_type = type(norm_mod).__name__
    if "RMS" in norm_type:
        # 1. epsilon (HF sometimes calls it variance_epsilon)
        eps = getattr(norm_mod, "eps", getattr(norm_mod, "variance_epsilon", 1e-5))
        # 2. scale (may be None for RMSNormPre)
        gamma = _get_rms_scale(norm_mod)
        if gamma is None:
            if (not hasattr(norm_mod, "_lens_ones")
                or norm_mod._lens_ones.device != resid.device
                or norm_mod._lens_ones.dtype  != resid.dtype):
                norm_mod._lens_ones = torch.ones(
                    resid.shape[-1], device=resid.device, dtype=resid.dtype
                )
            gamma = norm_mod._lens_ones
        return rms_lens(resid, gamma, eps)
    # Fallback - this shouldn't happen but be safe (unreachable)
    return resid

def clean_model_name(model_id):
    """Extract clean model name for filename"""
    # Remove organization prefix (everything before last '/')
    clean_name = model_id.split('/')[-1]
    return clean_name

def run_experiment_for_model(model_id):
    """Run the complete experiment for a single model and return output as string"""
    
    def evaluate_model():
        """The actual experiment code - all prints will be captured"""
        print(f"\n{'='*60}")
        print(f"EVALUATING MODEL: {model_id}")
        print(f"{'='*60}")
        
        # Load model with TransformerLens
        print(f"Loading model: {model_id}...")
        
        # Load with explicit device and dtype to avoid CPU float16 issues
        model = HookedTransformer.from_pretrained(
            model_id,
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        model.eval()  # Hygiene: avoid dropout etc.
        
        # Toggle for using normalized lens (recommended for accurate interpretation)
        USE_NORM_LENS = True
        
        # Toggle for FP32 unembedding (recommended for research-grade precision)
        # Prevents under-resolving logit gaps < 1e-5 at cost of ~50MB memory
        USE_FP32_UNEMBED = True
        
        # Promote unembedding weights to FP32 if requested for true precision gain
        if USE_FP32_UNEMBED and model.unembed.W_U.dtype != torch.float32:
            print(f"🔬 Promoting unembed weights to FP32 for research-grade precision (was {model.unembed.W_U.dtype})")
            model.unembed.W_U = torch.nn.Parameter(model.unembed.W_U.float(), requires_grad=False)
            if hasattr(model.unembed, 'b_U'):
                model.unembed.b_U = model.unembed.b_U.float()
            UNEMBED_DTYPE = torch.float32  # refresh after promotion
        else:
            UNEMBED_DTYPE = torch.float32 if USE_FP32_UNEMBED else model.unembed.W_U.dtype
        
        # Assemble prompt and normalization diagnostics
        prompt = "Question: What is the capital of Germany? Answer:"
        first_block_ln1_type = type(model.blocks[0].ln1).__name__ if hasattr(model, 'blocks') and len(model.blocks) > 0 else None
        final_ln_type = type(model.ln_final).__name__ if hasattr(model, 'ln_final') else None
        diag = {
            "type": "diagnostics",
            "model": model_id,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "use_norm_lens": USE_NORM_LENS,
            "use_fp32_unembed": USE_FP32_UNEMBED,
            "unembed_dtype": str(UNEMBED_DTYPE),
            "first_block_ln1_type": first_block_ln1_type,
            "final_ln_type": final_ln_type,
            "prompt": prompt
        }
        print(json.dumps(diag, ensure_ascii=False))
        # Emit prompt record
        print(json.dumps({"type": "prompt", "prompt": prompt}, ensure_ascii=False))
        IMPORTANT_WORDS = ["Germany", "Berlin", "capital"]

        def is_verbose_position(pos, token_str, seq_len):
            # Answer slot always verbose
            if pos == seq_len - 1:
                return True
            # Any prompt word in token string
            for w in IMPORTANT_WORDS:
                if w.lower() in token_str.lower().strip(".,!?;:"):
                    return True
            return False

        def print_summary(layer_idx, pos, token_str, entropy_bits, top_tokens, top_probs):
            # Emit a JSON Lines record for this layer/position
            record = {
                "type": "record",
                "layer": layer_idx,
                "pos": pos,
                "token": token_str,
                "entropy": entropy_bits,
                "topk": [[tok, prob.item()] for tok, prob in zip(top_tokens, top_probs)]
            }
            print(json.dumps(record, ensure_ascii=False))
        
        # Tokenize the prompt (cache for reuse)
        tokens = model.to_tokens(prompt).to(model.cfg.device)
        
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
                    # TODO: For batch runs, consider storing cache on GPU to avoid repeated .to(device) copies
                    cache_dict[hook.name] = tensor.cpu().float().detach()
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
                
                # Show top predictions at different layers
                print(f"\nTop predictions for next token at each position in '{prompt}':")
                if USE_NORM_LENS:
                    # Check if we'll actually be applying norms
                    if hasattr(model, 'blocks') and len(model.blocks) > 0:
                        first_norm = model.blocks[0].ln1
                        norm_type = type(first_norm).__name__
                        
                        if is_safe_layernorm(first_norm):
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
                print("-" * 60)
                
                # Get string representations of tokens for labeling output
                str_tokens = model.to_str_tokens(prompt)

                # Layer 0: embeddings (+ positional embeddings if available)
                print("Layer  0 (embeddings):")
                if has_pos_embed:
                    resid = (residual_cache['hook_embed'] +
                             residual_cache['hook_pos_embed']).to(model.cfg.device)
                else:
                    resid = residual_cache['hook_embed'].to(model.cfg.device)
                    print("[diagnostic] No separate positional embedding hook found (as expected for some models); using token embeddings for layer 0.")
                # Apply first block's LayerNorm to embeddings if using norm-lens
                if USE_NORM_LENS:
                    resid = rms_lens(
                        resid,
                        torch.ones(resid.shape[-1], device=resid.device),
                        eps=getattr(model.blocks[0].ln1, "eps", 1e-5)
                    )
                
                # Vectorized unembedding for all positions
                logits_all = model.unembed(resid[0].to(UNEMBED_DTYPE)).float()  # [seq, d_vocab]
                
                for pos in range(tokens.shape[1]):
                    layer_logits = logits_all[pos]
                    # Compute log-probs for entropy and selective probabilities
                    log_probs = torch.log_softmax(layer_logits, dim=0).to(torch.float32)
                    entropy_nats = -torch.sum(torch.exp(log_probs) * log_probs)
                    entropy_bits = entropy_nats.item() / math.log(2)

                    token_str = str_tokens[pos]
                    # Decide verbosity for this position
                    verbose = is_verbose_position(pos, token_str, tokens.shape[1])
                    # Choose k based on verbosity
                    k = TOP_K_VERBOSE if verbose else TOP_K_RECORD
                    # Get top-k indices from raw logits
                    _, top_indices_k = torch.topk(layer_logits, k, largest=True, sorted=True)
                    top_probs_k = torch.exp(log_probs[top_indices_k])
                    top_tokens_k = [model.tokenizer.decode([idx]) for idx in top_indices_k]
                    print_summary(0, pos, token_str, entropy_bits, top_tokens_k, top_probs_k)
                    if verbose:
                        # Print full verbose list (TOP_K_VERBOSE), reusing initial topk when possible
                        if k == TOP_K_VERBOSE:
                            verbose_indices = top_indices_k
                            verbose_probs = top_probs_k
                        else:
                            _, verbose_indices = torch.topk(layer_logits, TOP_K_VERBOSE, largest=True, sorted=True)
                            verbose_probs = torch.exp(log_probs[verbose_indices])
                        for i, (prob, idx) in enumerate(zip(verbose_probs, verbose_indices)):
                            tok = model.tokenizer.decode([idx])
                            print(f"    {i+1:2d}. '{tok}' ({prob.item():.6f})")
                        print()
                
                # Layers 1 to n_layers: after each transformer block
                for layer in range(n_layers):
                    print(f"Layer {layer + 1:2d} (after transformer block {layer}):")
                    # Get residual stream after this layer's block (move from CPU to GPU)
                    resid = residual_cache[f'blocks.{layer}.hook_resid_post'].to(model.cfg.device)
                    
                    # Apply normalization if requested
                    if USE_NORM_LENS:
                        # For pre-LN models, apply ln_final on last layer, otherwise a simple RMS lens
                        if layer == n_layers - 1 and hasattr(model, 'ln_final'):
                            resid = apply_norm_or_skip(resid, model.ln_final, f"layer {layer + 1} (final ln)")
                        else:
                            gamma = _get_rms_scale(model.blocks[layer].ln1) \
                                    or torch.ones(resid.shape[-1], device=resid.device)
                            eps = getattr(model.blocks[layer].ln1, "eps", 1e-5)
                            resid = rms_lens(resid, gamma, eps)
                    
                    # Vectorized unembedding for all positions
                    logits_all = model.unembed(resid[0].to(UNEMBED_DTYPE)).float() # [seq, d_vocab]
                    
                    for pos in range(tokens.shape[1]):
                        layer_logits = logits_all[pos]
                        # Compute log-probs for entropy and selective probabilities
                        log_probs = torch.log_softmax(layer_logits, dim=0).to(torch.float32)
                        entropy_nats = -torch.sum(torch.exp(log_probs) * log_probs)
                        entropy_bits = entropy_nats.item() / math.log(2)

                        token_str = str_tokens[pos]
                        # Decide verbosity for this position
                        verbose = is_verbose_position(pos, token_str, tokens.shape[1])
                        # Choose k based on verbosity
                        k = TOP_K_VERBOSE if verbose else TOP_K_RECORD
                        # Get top-k indices from raw logits
                        _, top_indices_k = torch.topk(layer_logits, k, largest=True, sorted=True)
                        top_probs_k = torch.exp(log_probs[top_indices_k])
                        top_tokens_k = [model.tokenizer.decode([idx]) for idx in top_indices_k]
                        print_summary(layer + 1, pos, token_str, entropy_bits, top_tokens_k, top_probs_k)
                        if verbose:
                            # Print full verbose list (TOP_K_VERBOSE), reusing initial topk when possible
                            if k == TOP_K_VERBOSE:
                                verbose_indices = top_indices_k
                                verbose_probs = top_probs_k
                            else:
                                _, verbose_indices = torch.topk(layer_logits, TOP_K_VERBOSE, largest=True, sorted=True)
                                verbose_probs = torch.exp(log_probs[verbose_indices])
                            for i, (prob, idx) in enumerate(zip(verbose_probs, verbose_indices)):
                                tok = model.tokenizer.decode([idx])
                                print(f"    {i+1:2d}. '{tok}' ({prob.item():.6f})")
                            print()
                
            finally:
                # Clean up hooks and cache
                for h in hooks:  # HookPoint.add_hook always returns a handle with .remove()
                    if h is not None:
                        h.remove()
                del residual_cache, hooks
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
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
            final_log_probs = torch.log_softmax(final_logits, dim=0).to(torch.float32)
            final_entropy_nats = -torch.sum(torch.exp(final_log_probs) * final_log_probs)
            final_entropy_bits = final_entropy_nats.item() / math.log(2)
            
            # Emit final prediction as a JSON Lines record
            final_record = {
                "type": "final_prediction",
                "entropy": final_entropy_bits,
                "topk": [[model.tokenizer.decode([idx]), prob.item()] for prob, idx in zip(final_top_probs, final_top_indices)]
            }
            print(json.dumps(final_record, ensure_ascii=False))
            
            # Emit additional probing records for test prompts
            # Cache tokenized test prompts to avoid redundant tokenization
            test_prompts = [
                "Germany's capital is", 
                "Berlin is the capital of",
                "Respond in one word: which city is the capital of Germany?"
            ]
            test_tokens_cache = {prompt: model.to_tokens(prompt).to(model.cfg.device) for prompt in test_prompts}
            
            for test_prompt in test_prompts:
                test_tokens = test_tokens_cache[test_prompt]
                
                test_logits = model(test_tokens)
                _, test_top_indices = torch.topk(test_logits[0, -1, :], 10, largest=True, sorted=True)
                test_full_probs = torch.softmax(test_logits[0, -1, :], dim=0)
                test_top_probs = test_full_probs[test_top_indices]
                
                # Calculate entropy for this prompt efficiently (convert to bits)
                test_log_probs = torch.log_softmax(test_logits[0, -1, :], dim=0).to(torch.float32)
                test_entropy_nats = -torch.sum(torch.exp(test_log_probs) * test_log_probs)
                test_entropy_bits = test_entropy_nats.item() / math.log(2)
                # Emit JSON Lines record for this test prompt
                probe_record = {
                    "type": "test_prompt",
                    "prompt": test_prompt,
                    "entropy": test_entropy_bits,
                    "topk": [[model.tokenizer.decode([idx]), prob.item()] for prob, idx in zip(test_top_probs, test_top_indices)]
                }
                print(json.dumps(probe_record, ensure_ascii=False))
            
            # Emit temperature exploration records
            test_prompt = "Question: What is the capital of Germany? Answer:"
            # Reuse cached tokens if available, otherwise tokenize once
            if test_prompt not in test_tokens_cache:
                test_tokens_cache[test_prompt] = model.to_tokens(test_prompt).to(model.cfg.device)
            test_tokens = test_tokens_cache[test_prompt]
            
            # Single forward pass - then rescale for different temperatures
            base_logits = model(test_tokens)[0, -1, :]
            
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
                temp_entropy_bits = temp_entropy_nats.item() / math.log(2)
                # Emit JSON Lines record for this temperature
                temp_record = {
                    "type": "temperature_exploration",
                    "temperature": temp,
                    "entropy": temp_entropy_bits,
                    "topk": [[model.tokenizer.decode([idx]), prob.item()] for prob, idx in zip(temp_top_probs, temp_top_indices)]
                }
                print(json.dumps(temp_record, ensure_ascii=False))
        
        print("=== END OF INSPECTING ==============\n")
        
        # Emit model stats as a JSON Lines record
        stats_record = {
            "type": "model_stats",
            "num_layers": model.cfg.n_layers,
            "d_model": model.cfg.d_model,
            "n_heads": model.cfg.n_heads,
            "d_vocab": model.cfg.d_vocab,
            "n_ctx": model.cfg.n_ctx
        }
        print(json.dumps(stats_record, ensure_ascii=False))
        
        # Clean up model to free memory (though process will end anyway)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    try:
        # Capture output while still showing it in console
        output_buffer = io.StringIO()
        
        # Use tee-like behavior: capture to buffer AND show in console
        with redirect_stdout(output_buffer):
            evaluate_model()
        
        # Get all captured output (including human-readable logs and JSONL records)
        raw = output_buffer.getvalue()
        # Print full raw output to console for diagnostics
        print(raw, end='')
        # Parse only the JSONL records into Python objects
        lines = [line for line in raw.splitlines() if line.strip().startswith('{')]
        objs = [json.loads(line) for line in lines]
        # Partition into sections
        diagnostics = next(o for o in objs if o.get('type') == 'diagnostics')
        prompt_rec = next(o for o in objs if o.get('type') == 'prompt')
        records = [o for o in objs if o.get('type') == 'record']
        final_pred = next(o for o in objs if o.get('type') == 'final_prediction')
        test_prompts = [o for o in objs if o.get('type') == 'test_prompt']
        temp_expl = [o for o in objs if o.get('type') == 'temperature_exploration']
        stats = next(o for o in objs if o.get('type') == 'model_stats')
        # Assemble full JSON output
        output_dict = {
            "diagnostics": diagnostics,
            "prompt": prompt_rec.get('prompt'),
            "records": records,
            "final_prediction": final_pred,
            "test_prompts": test_prompts,
            "temperature_exploration": temp_expl,
            "model_stats": stats
        }
        return json.dumps(output_dict, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_msg = f"ERROR evaluating {model_id}: {str(e)}"
        print(error_msg)
        return error_msg

def run_single_model(model_id):
    """Run experiment for a single model - used when called as subprocess"""
    print(f"\n{'='*80}")
    print(f"🚀 Starting subprocess for model: {model_id}")
    print(f"{'='*80}")
    
    # Set memory limit BEFORE any CUDA operations
    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of GPU memory
        except AttributeError:
            pass
    
    # Generate filename
    clean_name = clean_model_name(model_id)
    meta_filename = f"output-{clean_name}.json"
    csv_filename  = f"output-{clean_name}-records.csv"
    
    # Save in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    meta_filepath = os.path.join(script_dir, meta_filename)
    csv_filepath  = os.path.join(script_dir, csv_filename)
    
    try:
        # Generate full JSON output from the experiment
        json_str = run_experiment_for_model(model_id)
        data = json.loads(json_str)
        records = data.pop("records", [])

        # Save JSON metadata (without records)
        with open(meta_filepath, 'w', encoding='utf-8') as f_json:
            json.dump(data, f_json, ensure_ascii=False, indent=2)

        # Save records to CSV
        with open(csv_filepath, 'w', newline='', encoding='utf-8') as f_csv:
            writer = csv.writer(f_csv)
            # Header: layer,pos,token,entropy
            header = ["layer","pos","token","entropy"]
            # Pad all rows to TOP_K_VERBOSE slots
            for i in range(1, TOP_K_VERBOSE + 1):
                header.extend([f"top{i}", f"prob{i}"])
            writer.writerow(header)
            for rec in records:
                row = [rec.get("layer"), rec.get("pos"), rec.get("token"), rec.get("entropy")]
                # Pad each record to TOP_K_VERBOSE entries
                topk_list = rec.get("topk", [])
                for j in range(TOP_K_VERBOSE):
                    if j < len(topk_list):
                        tok, prob = topk_list[j]
                    else:
                        tok, prob = "", ""
                    row.extend([tok, prob])
                writer.writerow(row)

        print(f"✅ Experiment complete. JSON metadata saved to: {meta_filepath}")
        print(f"✅ Records CSV saved to: {csv_filepath}")
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

def main():
    """Main function to launch separate processes for each model"""
    if len(sys.argv) > 1:
        # We're being called as subprocess for a single model
        model_id = sys.argv[1]
        success = run_single_model(model_id)
        sys.exit(0 if success else 1)
    
    # Main process - launch subprocess for each model
    print(f"🎯 Starting experiment launcher for {len(CONFIRMED_MODELS)} models...")
    print("Each model will run in a separate process for clean memory isolation.")
    
    script_path = os.path.abspath(__file__)
    results = []
    
    for i, model_id in enumerate(CONFIRMED_MODELS, 1):
        print(f"\n{'='*80}")
        print(f"📋 Launching process {i}/{len(CONFIRMED_MODELS)}: {model_id}")
        print(f"{'='*80}")
        
        try:
            # Launch subprocess
            result = subprocess.run([
                sys.executable, script_path, model_id
            ], capture_output=False, text=True, check=False)
            
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"📁 Output files saved in: {script_dir}")
    
    print("\n📊 Results Summary:")
    for model_id, status in results:
        clean_name = clean_model_name(model_id)
        status_emoji = "✅" if status == "SUCCESS" else "❌"
        print(f"   {status_emoji} {clean_name}: {status}")
    
    print(f"\n📄 Expected output files:")
    for model_id in CONFIRMED_MODELS:
        clean_name = clean_model_name(model_id)
        print(f"   output-{clean_name}.json")
        print(f"   output-{clean_name}-records.csv")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

