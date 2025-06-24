from transformers import AutoTokenizer, AutoModelForCausalLM
import transformer_lens
from transformer_lens import HookedTransformer
import torch

# Layer-by-layer prediction analysis with LayerNorm lens correction
# Toggle USE_NORM_LENS for raw vs normalized residual stream analysis

# 1. Pick a model that is ALREADY in HF format

# ✅ CONFIRMED SUPPORTED:
# 8B - Llama 3 Base
# model_id = "meta-llama/Meta-Llama-3-8B"
# 7B - Mistral
# model_id = "mistralai/Mistral-7B-v0.1"
# 9B - Gemma 2
model_id = "google/gemma-2-9b"
# 8B - Qwen3
# model_id = "Qwen/Qwen3-8B"


# 2. Load model with TransformerLens
# TransformerLens handles tokenizer and model loading automatically
model = HookedTransformer.from_pretrained(model_id)

# Toggle for using normalized lens (recommended for accurate interpretation)
USE_NORM_LENS = True
# Note: Raw residual analysis still useful for activation patching and causal interventions

# 4. Inspect a short prompt - using Q&A format that works best across models
prompt = "Question: What is the capital of Germany? Answer:"

print("\n=== PROMPT =========================")
print(prompt)
print("=== END OF PROMPT =================\n")

print("\n=== INSPECTING ====================")
# Use TransformerLens to analyze layer-by-layer predictions
import torch

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

# Sample key layers across the model's layers
n_layers = model.cfg.n_layers
layers_to_check = [0, n_layers//6, n_layers//3, n_layers//2, 2*n_layers//3, 5*n_layers//6, n_layers-1]

# Look at the last position (after "Answer:")
last_pos = -1

for layer in layers_to_check:
    if layer < n_layers:
        # Get residual stream at this layer
        if layer == 0:
            # Use embeddings for layer 0
            resid = cache["embed"]
        else:
            resid = cache["resid_post", layer]
        
        # Apply LayerNorm if requested and not the final layer
        if USE_NORM_LENS and layer < n_layers - 1:
            # Apply the LayerNorm that the next block would use
            resid = model.blocks[layer + 1].ln1(resid)
        elif USE_NORM_LENS and layer == n_layers - 1:
            # For the final layer, apply the final LayerNorm
            resid = model.ln_final(resid)
        
        # Apply the unembedding to get logits
        layer_logits = model.unembed(resid[0, last_pos, :])
        
        # Convert to probabilities and get top 5
        probs = torch.softmax(layer_logits, dim=0)
        top_probs, top_indices = torch.topk(probs, 5)
        
        print(f"Layer {layer:2d}:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            token = model.to_string(idx.unsqueeze(0))
            print(f"  {i+1}. '{token}' ({prob.item():.6f})")
        print()

# Let's also see what the actual model would predict (final layer)
print("=" * 60)
print("ACTUAL MODEL PREDICTION (for comparison):")
final_logits = logits[0, -1, :]
final_probs = torch.softmax(final_logits, dim=0)
top_probs, top_indices = torch.topk(final_probs, 5)

print("Model's final prediction:")
for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
    token = model.to_string(idx.unsqueeze(0))
    print(f"  {i+1}. '{token}' ({prob.item():.6f})")

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
    top_probs, top_indices = torch.topk(test_probs, 3)
    
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        token = model.to_string(idx.unsqueeze(0))
        print(f"  {i+1}. '{token}' ({prob.item():.6f})")

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
    top_probs, top_indices = torch.topk(test_probs, 5)
    
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        token = model.to_string(idx.unsqueeze(0))
        print(f"  {i+1}. '{token}' ({prob.item():.6f})")

print("=== END OF INSPECTING ==============\n")

# 5. Show some basic statistics about the model
print("\n=== MODEL STATS ===============")
print(f"Number of layers: {model.cfg.n_layers}")
print(f"Model dimension: {model.cfg.d_model}")
print(f"Number of heads: {model.cfg.n_heads}")
print(f"Vocab size: {model.cfg.d_vocab}")
print(f"Context length: {model.cfg.n_ctx}")
print("=== END OF MODEL STATS ========\n")

