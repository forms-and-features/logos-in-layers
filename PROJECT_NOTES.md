# Interpretability Project Notes

## Goal
Experiment with interpretability of open-weights LLMs.

## Recent Progress
- models/ folder contains .gguf of quantized llama 4 scout
- chat.py has working version of a demo chat with llama 4 scout
- lens_analysis.py now working with DialoGPT-large, GPT-J, and Llama 3 models
- Successfully implemented tuned lens analysis showing layer-by-layer predictions
- Discovered fascinating model differences in geographical knowledge:
  - DialoGPT: consistently predicts Frankfurt > Cologne > Berlin (wrong, confident)
  - Llama 3: correctly predicts Berlin with proper layer evolution (0% → 67% → 79%)
- Temperature scaling reveals different behaviors: DialoGPT stays wrong, Llama 3 stays correct
- Reverse direction works for both models: "Berlin is the capital of" → "Germany"

## Next Steps
- Try with cutting-edge models (Llama, Mistral) once tuned_lens compatibility is resolved
- Explore other interpretability techniques for models not supported by tuned_lens

## Key Findings
- existing libraries that are useful for interpretability do not work with .gguf files and require raw transformer files
- llama 4 scout transformers size is too large for this machine, and there is no pre-quantized version
- tuned_lens library support varies by architecture:
  - WORKS: GPT-2, GPT-Neo, GPT-NeoX, OPT, Pythia, Llama 3
  - FAILS: Mistral, Gemma (all versions), Qwen2/3, DeepSeek-R1-Distill-Qwen (throws "Unknown model type" errors)
  - UNTESTED: None remaining - all major 2024+ model families tested and failed
- Model architecture dramatically affects factual knowledge:
  - DialoGPT: confidently wrong about German capital (Frankfurt bias)
  - Llama 3: correct knowledge with clear layer evolution showing how answer emerges
- Interpretability insights from layer analysis:
  - Early layers: generic/random predictions
  - Middle layers: correct answer starts emerging  
  - Final layers: confident correct prediction
- Directional knowledge bias exists in some models: "Berlin is capital of Germany" works better than "capital of Germany is Berlin"

## Technical Issues Resolved
- Fixed BitsAndBytesConfig compatibility by removing quantization (not needed on Apple Silicon)
- Fixed device mismatch by moving lens to same device as model
- Fixed missing tuned_lens API methods (inspect doesn't exist, use PredictionTrajectory instead)
- Fixed tensor/numpy conversion issues in trajectory processing

## Chat Context
Assume the user is a software engineer with no python background.
The code is running on Macbook Pro M2 Max 64Gb.
