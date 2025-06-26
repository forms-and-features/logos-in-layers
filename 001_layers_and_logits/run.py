from transformers import AutoTokenizer, AutoModelForCausalLM
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
import warnings
import math

# Layer-by-layer prediction analysis with LayerNorm lens correction
# Toggle USE_NORM_LENS for raw vs normalized residual stream analysis

# List of confirmed supported models
CONFIRMED_MODELS = [
    "meta-llama/Meta-Llama-3-8B",      # 8B - Llama 3 Base
    "mistralai/Mistral-7B-v0.1",       # 7B - Mistral
    "google/gemma-2-9b",               # 9B - Gemma 2
    "Qwen/Qwen3-8B"                    # 8B - Qwen3
]

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
    """
    # Vanilla LayerNorm - apply as before
    if isinstance(norm_mod, nn.LayerNorm):
        return norm_mod(resid)
    
    # RMSNorm - use RMS-aware lens if scale parameter is available
    norm_type = type(norm_mod).__name__
    if 'RMS' in norm_type:
        # Try to get scale parameter (weight or scale attribute)
        gamma = getattr(norm_mod, 'weight', getattr(norm_mod, 'scale', None))
        if gamma is not None:
            # Try to get epsilon from the norm module, fallback to default
            eps = getattr(norm_mod, 'eps', 1e-5)
            return rms_lens(resid, gamma, eps)
    
        # RMSNorm without accessible scale parameter - skip with warning
        else:
            warnings.warn(f"Skipping RMS lens at {layer_info} - no accessible weight/scale parameter")
            return resid
    
    # Any other normalization type - skip
    if norm_type != 'LayerNorm':
        warnings.warn(f"Skipping norm-lens at {layer_info} because {norm_type} "
                     f"is not supported (only LayerNorm and RMSNorm supported)")
        return resid
    
    # Fallback - this shouldn't happen but be safe
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
        
        # Toggle for using normalized lens (recommended for accurate interpretation)
        USE_NORM_LENS = True
        
        # Toggle for FP32 unembedding (recommended for research-grade precision)
        # Prevents under-resolving logit gaps < 1e-5 at cost of ~50MB memory
        USE_FP32_UNEMBED = True
        
        # Promote unembedding weights to FP32 if requested for true precision gain
        if USE_FP32_UNEMBED and model.unembed.W_U.dtype != torch.float32:
            print(f"🔬 Promoting unembed weights to FP32 for research-grade precision (was {model.unembed.W_U.dtype})")
            model.unembed.W_U = torch.nn.Parameter(model.unembed.W_U.float(), requires_grad=False)
        
        # Diagnostic: Check normalization types in the model
        print("\n=== NORMALIZATION ANALYSIS ========")
        if hasattr(model, 'blocks') and len(model.blocks) > 0:
            first_block_ln1_type = type(model.blocks[0].ln1).__name__
            print(f"Block normalization type: {first_block_ln1_type}")
            
            if is_safe_layernorm(model.blocks[0].ln1):
                print("✅ LayerNorm detected - standard norm-lens will be applied")
            elif 'RMS' in first_block_ln1_type:
                first_norm = model.blocks[0].ln1
                gamma = getattr(first_norm, 'weight', getattr(first_norm, 'scale', None))
                if gamma is not None:
                    param_name = 'weight' if hasattr(first_norm, 'weight') else 'scale'
                    print(f"✅ RMSNorm detected with {param_name} parameter - RMS-aware lens will be applied")
                else:
                    print("⚠️  RMSNorm detected but no weight/scale parameter - norm-lens will be skipped")
            else:
                print(f"⚠️  Unsupported norm type ({first_block_ln1_type}) - norm-lens will be skipped")
        
        if hasattr(model, 'ln_final'):
            final_ln_type = type(model.ln_final).__name__
            print(f"Final normalization type: {final_ln_type}")
        print("=== END NORMALIZATION ANALYSIS ====\n")
        
        # Inspect a short prompt - using Q&A format that works best across models
        prompt = "Question: What is the capital of Germany? Answer:"
        
        print("\n=== PROMPT =========================")
        print(prompt)
        print("=== END OF PROMPT =================\n")
        
        print("\n=== INSPECTING ====================")
        
        # Tokenize the prompt (cache for reuse)
        tokens = model.to_tokens(prompt).to(model.cfg.device)
        
        print(f"Input tokens: {model.to_str_tokens(prompt)}")
        
        # Use no_grad to disable gradient computation for memory efficiency
        with torch.no_grad():
            # MEMORY EFFICIENT: Use targeted caching instead of run_with_cache
            print("Computing layer-wise predictions (memory-efficient targeted caching)...")
            
            # Storage for only the residual streams we need
            residual_cache = {}
            
            def make_cache_hook(cache_dict):
                def cache_residual_hook(tensor, hook):
                    # Only store the tensor we need, detached from computation graph
                    # Store only last token position on CPU to save VRAM (keep original dtype to avoid mismatches)
                    cache_dict[hook.name] = tensor[:, -1:, :].cpu().detach()
                return cache_residual_hook
            
            # Create the hook function with explicit cache reference
            cache_hook = make_cache_hook(residual_cache)
            
            # Set up hooks for residual streams only
            hooks = []
            
            # Hook for embeddings (layer 0 equivalent)
            embed_hook = model.hook_dict['hook_embed'].add_hook(cache_hook)
            hooks.append(embed_hook)
            
            # Hook for each layer's residual post
            n_layers = model.cfg.n_layers
            for layer in range(n_layers):
                resid_hook = model.blocks[layer].hook_resid_post.add_hook(cache_hook)
                hooks.append(resid_hook)
            
            try:
                # Run forward pass with targeted hooks
                logits = model(tokens)
                
                # Show top predictions at different layers for the last token position
                print(f"\nTop predictions for next token after '{prompt}':")
                if USE_NORM_LENS:
                    # Check if we'll actually be applying norms
                    if hasattr(model, 'blocks') and len(model.blocks) > 0:
                        first_norm = model.blocks[0].ln1
                        norm_type = type(first_norm).__name__
                        
                        if is_safe_layernorm(first_norm):
                            print("Using NORMALIZED residual stream (LayerNorm applied - more accurate)")
                        elif 'RMS' in norm_type and hasattr(first_norm, 'weight'):
                            print("Using NORMALIZED residual stream (RMS-aware lens applied - more accurate)")
                        else:
                            print("Using RAW residual stream (unsupported normalization, skipping to avoid distortion)")
                    else:
                        print("Using RAW residual stream (no normalization layers found)")
                else:
                    print("Using RAW residual stream (normalization disabled)")
                print("Note: Shown probabilities are from full softmax (calibrated and comparable)")
                print("-" * 60)
                
                # Look at the last position (after "Answer:")
                last_pos = -1
                
                # Layer indexing scheme: 
                # Layer 0 = embeddings (before any transformer blocks)
                # Layer N = after transformer block N-1 (N = 1, 2, ..., n_layers)
                
                # Layer 0: embeddings
                print("Layer  0 (embeddings):")
                resid = residual_cache['hook_embed'].to(model.cfg.device)  # Move from CPU to GPU
                
                # Apply first block's LayerNorm to embeddings if using norm-lens
                if USE_NORM_LENS:
                    resid = apply_norm_or_skip(resid, model.blocks[0].ln1, "layer 0 (embeddings + block 0 ln1)")
                
                # Apply the unembedding to get logits (resid is now [1, 1, d_model] from caching last token only)
                # Get weight dtype safely (TransformerLens Unembed uses .W_U)
                weight_dtype = getattr(model.unembed, "weight", model.unembed.W_U).dtype
                if USE_FP32_UNEMBED:
                    # Use float32 throughout for research-grade precision (weights already promoted)
                    layer_logits = model.unembed(resid[0, 0, :].float())
                else:
                    # Cast to weight dtype for matmul, then to float32 for calculations
                    layer_logits = model.unembed(resid[0, 0, :].to(weight_dtype)).float()
                
                # Get top-k indices, then compute calibrated probabilities from full softmax
                _, top_indices = torch.topk(layer_logits, 20, largest=True, sorted=True)
                full_probs = torch.softmax(layer_logits, dim=0)
                top_probs = full_probs[top_indices]
                
                # Calculate entropy efficiently using log_softmax (convert to bits)
                log_probs = torch.log_softmax(layer_logits, dim=0)
                entropy_nats = -torch.sum(torch.exp(log_probs) * log_probs)
                entropy_bits = entropy_nats.item() / math.log(2)
                
                print(f"  (entropy: {entropy_bits:.3f} bits):")
                for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                    token = model.tokenizer.decode([idx])
                    print(f"  {i+1:2d}. '{token}' ({prob.item():.6f})")
                print()
                
                # Layers 1 to n_layers: after each transformer block
                for layer in range(n_layers):
                    # Get residual stream after this layer's block (move from CPU to GPU)
                    resid = residual_cache[f'blocks.{layer}.hook_resid_post'].to(model.cfg.device)
                    
                    # Apply LayerNorm if requested (with safety checks)
                    if USE_NORM_LENS:
                        if layer < n_layers - 1:
                            # Apply the LayerNorm that the next block would use
                            resid = apply_norm_or_skip(resid, model.blocks[layer + 1].ln1, f"layer {layer + 1} (block {layer + 1} ln1)")
                        else:
                            # For the final layer, apply the final LayerNorm if it exists
                            final_ln = getattr(model, 'ln_final', None)
                            if final_ln is not None:
                                resid = apply_norm_or_skip(resid, final_ln, f"layer {layer + 1} (final ln)")
                    
                    # Apply the unembedding to get logits (resid is [1, 1, d_model], cast to float32 for stability)
                    # Get weight dtype safely (TransformerLens Unembed uses .W_U)
                    weight_dtype = getattr(model.unembed, "weight", model.unembed.W_U).dtype
                    if USE_FP32_UNEMBED:
                        # Use float32 throughout for research-grade precision (weights already promoted)
                        layer_logits = model.unembed(resid[0, 0, :].float())
                    else:
                        # Cast to weight dtype for matmul, then to float32 for calculations
                        layer_logits = model.unembed(resid[0, 0, :].to(weight_dtype)).float()
                    
                    # Get top-k indices, then compute calibrated probabilities from full softmax
                    _, top_indices = torch.topk(layer_logits, 20, largest=True, sorted=True)
                    full_probs = torch.softmax(layer_logits, dim=0)
                    top_probs = full_probs[top_indices]
                    
                    # Calculate entropy efficiently using log_softmax (convert to bits)
                    log_probs = torch.log_softmax(layer_logits, dim=0)
                    entropy_nats = -torch.sum(torch.exp(log_probs) * log_probs)
                    entropy_bits = entropy_nats.item() / math.log(2)
                    
                    print(f"Layer {layer + 1:2d} (after transformer block {layer}) (entropy: {entropy_bits:.3f} bits):")
                    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                        token = model.tokenizer.decode([idx])
                        print(f"  {i+1:2d}. '{token}' ({prob.item():.6f})")
                    print()
                
            finally:
                # Clean up hooks and cache
                for hook in hooks:
                    if hook is not None:
                        hook.remove()
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
            final_log_probs = torch.log_softmax(final_logits, dim=0)
            final_entropy_nats = -torch.sum(torch.exp(final_log_probs) * final_log_probs)
            final_entropy_bits = final_entropy_nats.item() / math.log(2)
            
            print(f"Model's final prediction (entropy: {final_entropy_bits:.3f} bits):")
            for i, (prob, idx) in enumerate(zip(final_top_probs, final_top_indices)):
                token = model.tokenizer.decode([idx])
                print(f"  {i+1:2d}. '{token}' ({prob.item():.6f})")
            
            # Let's probe the model's knowledge a bit more
            print("=" * 60)
            print("ADDITIONAL PROBING:")
            
            # Cache tokenized test prompts to avoid redundant tokenization
            test_prompts = [
                "Germany's capital is", 
                "Berlin is the capital of",
                "Respond in one word: which city is the capital of Germany?"
            ]
            test_tokens_cache = {prompt: model.to_tokens(prompt).to(model.cfg.device) for prompt in test_prompts}
            
            for test_prompt in test_prompts:
                print(f"\nPrompt: '{test_prompt}'")
                test_tokens = test_tokens_cache[test_prompt]
                
                test_logits = model(test_tokens)
                _, test_top_indices = torch.topk(test_logits[0, -1, :], 10, largest=True, sorted=True)
                test_full_probs = torch.softmax(test_logits[0, -1, :], dim=0)
                test_top_probs = test_full_probs[test_top_indices]
                
                # Calculate entropy for this prompt efficiently (convert to bits)
                test_log_probs = torch.log_softmax(test_logits[0, -1, :], dim=0)
                test_entropy_nats = -torch.sum(torch.exp(test_log_probs) * test_log_probs)
                test_entropy_bits = test_entropy_nats.item() / math.log(2)
                print(f"  (entropy: {test_entropy_bits:.3f} bits)")
                
                for i, (prob, idx) in enumerate(zip(test_top_probs, test_top_indices)):
                    token = model.tokenizer.decode([idx])
                    print(f"  {i+1:2d}. '{token}' ({prob.item():.6f})")
            
            # OPTIMIZED: Single forward pass for temperature exploration
            print("=" * 60)
            print("TEMPERATURE EXPLORATION:")
            print("(Temperature controls randomness: low=confident, high=creative)")
            
            test_prompt = "Question: What is the capital of Germany? Answer:"
            # Reuse cached tokens if available, otherwise tokenize once
            if test_prompt not in test_tokens_cache:
                test_tokens_cache[test_prompt] = model.to_tokens(test_prompt).to(model.cfg.device)
            test_tokens = test_tokens_cache[test_prompt]
            
            # Single forward pass - then rescale for different temperatures
            base_logits = model(test_tokens)[0, -1, :]
            
            temperatures = [0.1, 2.0]
            
            for temp in temperatures:
                print(f"\nTemperature {temp}:")
                
                # Rescale existing logits instead of new forward pass (cast to float32 for numerical stability)
                scaled_logits = (base_logits / temp).float()
                _, temp_top_indices = torch.topk(scaled_logits, 15, largest=True, sorted=True)
                temp_full_probs = torch.softmax(scaled_logits, dim=0)
                temp_top_probs = temp_full_probs[temp_top_indices]
                
                # Calculate entropy at this temperature efficiently (convert to bits)
                temp_log_probs = torch.log_softmax(scaled_logits, dim=0)
                temp_entropy_nats = -torch.sum(torch.exp(temp_log_probs) * temp_log_probs)
                temp_entropy_bits = temp_entropy_nats.item() / math.log(2)
                print(f"  (entropy: {temp_entropy_bits:.3f} bits)")
                
                for i, (prob, idx) in enumerate(zip(temp_top_probs, temp_top_indices)):
                    token = model.tokenizer.decode([idx])
                    print(f"  {i+1:2d}. '{token}' ({prob.item():.6f})")
            
        print("=== END OF INSPECTING ==============\n")
        
        # Show some basic statistics about the model
        print("\n=== MODEL STATS ===============")
        print(f"Number of layers: {model.cfg.n_layers}")
        print(f"Model dimension: {model.cfg.d_model}")
        print(f"Number of heads: {model.cfg.n_heads}")
        print(f"Vocab size: {model.cfg.d_vocab}")
        print(f"Context length: {model.cfg.n_ctx}")
        print("=== END OF MODEL STATS ========\n")
        
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
        
        # Also print to console by temporarily redirecting back
        captured_output = output_buffer.getvalue()
        print(captured_output, end='')  # Print captured output to console too
        
        return captured_output
        
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
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of GPU memory
    
    # Generate filename
    clean_name = clean_model_name(model_id)
    filename = f"output-{clean_name}.txt"
    
    # Save in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)
    
    try:
        output = run_experiment_for_model(model_id)
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(output)
        
        print(f"✅ Experiment complete. Output saved to: {filepath}")
        return True
        
    except Exception as e:
        error_msg = f"❌ Failed to evaluate {model_id}: {str(e)}"
        print(error_msg)
        
        # Still save error output
        with open(filepath, 'w', encoding='utf-8') as f:
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
        print(f"   output-{clean_name}.txt")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

