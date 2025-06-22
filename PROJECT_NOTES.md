# Interpretability Project Notes

## Goal
Experiment with interpretability of open-weights LLMs.

## Recent Progress
- models/ folder contains .gguf of quantized llama 4 scout
- chat.py has working version of a demo chat with llama 4 scout
- lens_analysis.py uses TransformerLens which is confirmed to work with DialoGPT, Llama 3, and Mistral 7B models
- Implemented layer-by-layer prediction analysis showing how knowledge emerges
- Discovered fascinating model differences in geographical knowledge:
  - DialoGPT: consistently predicts Frankfurt > Cologne > Berlin (wrong, confident)
  - Llama 3: correctly predicts Berlin with proper layer evolution (0% → 67% → 79%)
  - Mistral 7B: correctly predicts Berlin with clean layer progression (0.4% → 2.2% → 22.6%)
- Temperature scaling reveals different behaviors across architectures
- Reverse direction works consistently: "Berlin is the capital of" → "Germany" (high confidence)

## Next Steps
- Test additional cutting-edge models: Gemma, Qwen2/3, DeepSeek
- Explore advanced TransformerLens features (attention patterns, circuit analysis)
- Compare factual knowledge across more model families

## Key Findings
- existing libraries that are useful for interpretability do not work with .gguf files and require raw transformer files
- llama 4 scout transformers size is too large for this machine, and there is no pre-quantized version
- TransformerLens provides vastly superior model support compared to tuned_lens (the downside, however, is the speed):
  - WORKS: GPT-2, GPT-Neo, GPT-NeoX, OPT, Pythia, Llama 3, Mistral, Gemma, Qwen2/3, and many more
- Model architecture dramatically affects factual knowledge:
  - DialoGPT: confidently wrong about German capital (Frankfurt bias)
  - Llama 3: correct knowledge with clear layer evolution showing how answer emerges
  - Mistral 7B: correct knowledge with clean progression, competes with generic responses
- Interpretability insights from layer analysis:
  - Early layers: generic/random predictions
  - Middle layers: correct answer starts emerging  
  - Final layers: confident correct prediction
  - **Pattern holds across architectures**: DialoGPT, Llama 3, and Mistral all show similar layer evolution
- Directional knowledge bias exists across models: "Berlin is capital of Germany" works better than "capital of Germany is Berlin"

## Technical Issues Resolved
- Fixed BitsAndBytesConfig compatibility by removing quantization (not needed on Apple Silicon)
- Solved model compatibility issues by switching from tuned_lens to TransformerLens
- Simplified codebase by using TransformerLens's unified model loading and tokenization
- TransformerLens handles device management and tensor operations

## Chat Context
Assume the user is a software engineer with no python background.
The code is running on Macbook Pro M2 Max 64Gb.
