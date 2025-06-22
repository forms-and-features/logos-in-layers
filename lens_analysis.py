from transformers import AutoTokenizer, AutoModelForCausalLM
from tuned_lens import TunedLens
import torch

# 1. Pick a model that is ALREADY in HF format

# ✅ CONFIRMED WORKING with tuned_lens:
# 345M - WORKS, but biased (Frankfurt > Berlin)
# model_id = "microsoft/DialoGPT-large"
# 8B - WORKS, correct knowledge (Berlin)
model_id = "meta-llama/Meta-Llama-3-8B"

# 🧪 UNTESTED (cutting-edge models worth trying):
# 7B - Older Gemma 1 (Feb 2024), might work
# model_id = "google/gemma-7b"

# ❌ CONFIRMED NOT WORKING with tuned_lens:
# Mistral (all versions) - "Unknown model type" errors
# Gemma 3 (all versions) - "Unknown model type" errors
# Qwen3 (all versions) - "Unknown model type" errors
# DeepSeek-R1-Distill-Qwen (all versions) - "Unknown model type" errors (Qwen2 base)

# 2. Load tokenizer + model on the M-series GPU (no quantization needed)
tok = AutoTokenizer.from_pretrained(model_id)
# Load model with automatic device mapping to GPU where possible
# Use half precision for memory efficiency on Apple Silicon
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
).eval()

# 3. Build a tuned lens (projects every layer through the LM head)
# Create tuned lens - this is instant, no training required
lens = TunedLens.from_model(model)
# Move lens to same device as model
lens = lens.to(model.device)

# 4. Inspect a short prompt
prompt = "The capital of Germany is"

print("\n=== PROMPT =========================")
print(prompt)
print("=== END OF PROMPT =================\n")

print("\n=== INSPECTING ====================")
# Use PredictionTrajectory to analyze layer-by-layer predictions
from tuned_lens.plotting import PredictionTrajectory
import torch

# Tokenize the prompt
input_ids = tok.encode(prompt, return_tensors="pt")
targets = input_ids[0, 1:].tolist() + [tok.eos_token_id]

print(f"Input tokens: {tok.convert_ids_to_tokens(input_ids[0])}")
print(f"Predicting next tokens for: {targets}")

# Create prediction trajectory
trajectory = PredictionTrajectory.from_lens_and_model(
    lens, 
    model, 
    tokenizer=tok,
    input_ids=input_ids[0].tolist(),
    targets=targets
)

# Show top predictions at different layers for the last token position
print(f"\nTop predictions for next token after '{prompt}':")
print("-" * 60)

# Sample key layers across the model's 37 layers (0-36)
layers_to_check = [0, 6, 12, 18, 24, 30, 36]
# Add the final layer since we have 37 layers (0-36)
if trajectory.log_probs.shape[0] > 36:
    # Add the very last layer
    layers_to_check.append(trajectory.log_probs.shape[0] - 1)
# Look at the last position (after "is")
last_pos = -1

for layer in layers_to_check:
    if layer < trajectory.log_probs.shape[0]:
        # Get log probabilities for this layer and position
        layer_logits = trajectory.log_probs[layer, last_pos, :]
        
        # Convert to PyTorch tensor if it's numpy
        if not isinstance(layer_logits, torch.Tensor):
            layer_logits = torch.from_numpy(layer_logits)
        
        # Convert to probabilities and get top 5
        probs = torch.softmax(layer_logits, dim=0)
        top_probs, top_indices = torch.topk(probs, 5)
        
        print(f"Layer {layer:2d}:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            token = tok.decode([idx.item()])
            print(f"  {i+1}. '{token}' ({prob.item():.3f})")
        print()

# Let's also see what the actual model would predict (without the lens)
print("=" * 60)
print("ACTUAL MODEL PREDICTION (for comparison):")
with torch.no_grad():
    model_output = model(input_ids.to(model.device))
    # Last position logits
    final_logits = model_output.logits[0, -1, :]
    final_probs = torch.softmax(final_logits, dim=0)
    top_probs, top_indices = torch.topk(final_probs, 5)
    
    print("Model's final prediction:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        token = tok.decode([idx.item()])
        print(f"  {i+1}. '{token}' ({prob.item():.3f})")

# Let's probe the model's knowledge a bit more
print("=" * 60)
print("ADDITIONAL PROBING:")

# Test some variations to understand the model's knowledge
test_prompts = [
    "The capital city of Germany is",
    "Germany's capital is", 
    "Berlin is the capital of",
    "Frankfurt is the capital of",
    "The current capital of Germany is"
]

for test_prompt in test_prompts:
    print(f"\nPrompt: '{test_prompt}'")
    test_ids = tok.encode(test_prompt, return_tensors="pt")
    
    with torch.no_grad():
        test_output = model(test_ids.to(model.device))
        test_logits = test_output.logits[0, -1, :]
        test_probs = torch.softmax(test_logits, dim=0)
        top_probs, top_indices = torch.topk(test_probs, 3)
        
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            token = tok.decode([idx.item()])
            print(f"  {i+1}. '{token}' ({prob.item():.3f})")

# Let's explore how temperature affects the predictions
print("=" * 60)
print("TEMPERATURE EXPLORATION:")
print("(Temperature controls randomness: low=confident, high=creative)")

test_prompt = "The capital of Germany is"
test_ids = tok.encode(test_prompt, return_tensors="pt")

temperatures = [0.1, 0.5, 1.0, 1.5, 2.0]

for temp in temperatures:
    print(f"\nTemperature {temp}:")
    
    with torch.no_grad():
        test_output = model(test_ids.to(model.device))
        test_logits = test_output.logits[0, -1, :]
        
        # Apply temperature scaling
        scaled_logits = test_logits / temp
        test_probs = torch.softmax(scaled_logits, dim=0)
        top_probs, top_indices = torch.topk(test_probs, 5)
        
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            token = tok.decode([idx.item()])
            print(f"  {i+1}. '{token}' ({prob.item():.3f})")

print("=== END OF INSPECTING ==============\n")

# 5. Show some basic statistics about the trajectory
print("\n=== TRAJECTORY STATS ===============")
print(f"Number of layers: {trajectory.log_probs.shape[0]}")
print(f"Sequence length: {trajectory.log_probs.shape[1]}")
print(f"Vocab size: {trajectory.log_probs.shape[2]}")
print("=== END OF TRAJECTORY STATS ========\n")

# Optional: show() opens an interactive HTML if you're in Jupyter
