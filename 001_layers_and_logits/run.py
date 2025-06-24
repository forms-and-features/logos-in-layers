from transformers import AutoTokenizer, AutoModelForCausalLM
import transformer_lens
from transformer_lens import HookedTransformer
import torch
import io
from contextlib import redirect_stdout
from datetime import datetime
import os
import subprocess
import sys

# Layer-by-layer prediction analysis with LayerNorm lens correction
# Toggle USE_NORM_LENS for raw vs normalized residual stream analysis

# List of confirmed supported models
CONFIRMED_MODELS = [
    "meta-llama/Meta-Llama-3-8B",      # 8B - Llama 3 Base
    "mistralai/Mistral-7B-v0.1",       # 7B - Mistral
    "google/gemma-2-9b",               # 9B - Gemma 2
    "Qwen/Qwen3-8B"                    # 8B - Qwen3
]

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
        model = HookedTransformer.from_pretrained(model_id)
        
        # Toggle for using normalized lens (recommended for accurate interpretation)
        USE_NORM_LENS = True
        
        # Inspect a short prompt - using Q&A format that works best across models
        prompt = "Question: What is the capital of Germany? Answer:"
        
        print("\n=== PROMPT =========================")
        print(prompt)
        print("=== END OF PROMPT =================\n")
        
        print("\n=== INSPECTING ====================")
        
        # Tokenize the prompt
        tokens = model.to_tokens(prompt)
        
        print(f"Input tokens: {model.to_str_tokens(prompt)}")
        
        # Run model and get activations
        logits, cache = model.run_with_cache(tokens)
        
        # Show top predictions at different layers for the last token position
        print(f"\nTop predictions for next token after '{prompt}':")
        if USE_NORM_LENS:
            print("Using NORMALIZED residual stream (LayerNorm applied - more accurate)")
        else:
            print("Using RAW residual stream (no LayerNorm - may be less accurate)")
        print("-" * 60)
        
        # Check all layers to capture when the answer crystallizes
        n_layers = model.cfg.n_layers
        
        # Look at the last position (after "Answer:")
        last_pos = -1
        
        for layer in range(n_layers):
            # Get residual stream at this layer
            if layer == 0:
                # Use embeddings for layer 0
                resid = cache["embed"]
            else:
                resid = cache["resid_post", layer]
            
            # Apply LayerNorm if requested
            if USE_NORM_LENS:
                if layer == 0:
                    # For layer 0, apply the first block's LayerNorm to embeddings
                    resid = model.blocks[0].ln1(resid)
                elif layer < n_layers - 1:
                    # Apply the LayerNorm that the next block would use
                    resid = model.blocks[layer + 1].ln1(resid)
                else:
                    # For the final layer, apply the final LayerNorm
                    resid = model.ln_final(resid)
            
            # Apply the unembedding to get logits
            layer_logits = model.unembed(resid[0, last_pos, :])
            
            # Convert to probabilities and get top 20
            probs = torch.softmax(layer_logits, dim=0)
            top_probs, top_indices = torch.topk(probs, 20)
            
            # Calculate entropy (uncertainty measure)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            
            print(f"Layer {layer:2d} (entropy: {entropy.item():.3f}):")
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                token = model.tokenizer.decode([idx])
                print(f"  {i+1:2d}. '{token}' ({prob.item():.6f})")
            print()
        
        # Let's also see what the actual model would predict (final layer)
        print("=" * 60)
        print("ACTUAL MODEL PREDICTION (for comparison):")
        final_logits = logits[0, -1, :]
        final_probs = torch.softmax(final_logits, dim=0)
        top_probs, top_indices = torch.topk(final_probs, 20)
        
        # Calculate final entropy
        final_entropy = -torch.sum(final_probs * torch.log(final_probs + 1e-10))
        
        print(f"Model's final prediction (entropy: {final_entropy.item():.3f}):")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            token = model.tokenizer.decode([idx])
            print(f"  {i+1:2d}. '{token}' ({prob.item():.6f})")
        
        # Let's probe the model's knowledge a bit more
        print("=" * 60)
        print("ADDITIONAL PROBING:")
        
        # Test some variations to understand the model's knowledge
        test_prompts = [
            "Germany's capital is", 
            "Berlin is the capital of",
            "Respond in one word: which city is the capital of Germany?"
        ]
        
        for test_prompt in test_prompts:
            print(f"\nPrompt: '{test_prompt}'")
            test_tokens = model.to_tokens(test_prompt)
            
            test_logits = model(test_tokens)
            test_probs = torch.softmax(test_logits[0, -1, :], dim=0)
            top_probs, top_indices = torch.topk(test_probs, 10)
            
            # Calculate entropy for this prompt
            test_entropy = -torch.sum(test_probs * torch.log(test_probs + 1e-10))
            print(f"  (entropy: {test_entropy.item():.3f})")
            
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                token = model.tokenizer.decode([idx])
                print(f"  {i+1:2d}. '{token}' ({prob.item():.6f})")
        
        # Let's explore how temperature affects the predictions
        print("=" * 60)
        print("TEMPERATURE EXPLORATION:")
        print("(Temperature controls randomness: low=confident, high=creative)")
        
        test_prompt = "Question: What is the capital of Germany? Answer:"
        test_tokens = model.to_tokens(test_prompt)
        
        temperatures = [0.1, 2.0]
        
        for temp in temperatures:
            print(f"\nTemperature {temp}:")
            
            test_logits = model(test_tokens)
            
            # Apply temperature scaling
            scaled_logits = test_logits[0, -1, :] / temp
            test_probs = torch.softmax(scaled_logits, dim=0)
            top_probs, top_indices = torch.topk(test_probs, 15)
            
            # Calculate entropy at this temperature
            temp_entropy = -torch.sum(test_probs * torch.log(test_probs + 1e-10))
            print(f"  (entropy: {temp_entropy.item():.3f})")
            
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
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
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
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
    
    # Generate timestamp and filename
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    clean_name = clean_model_name(model_id)
    filename = f"{clean_name}-{timestamp}.txt"
    
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
            f.write(f"Timestamp: {timestamp}\n")
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
        print(f"   {clean_name}-YYYYMMDDHHMMSS.txt")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

