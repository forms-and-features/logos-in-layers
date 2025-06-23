# Interpretability Project Notes

## Goal
Experiment with interpretability of open-weights LLMs.

## Recent Progress
- Switched from tuned_lens to TransformerLens for better model support
- lens_analysis.py now works with Llama 3, Mistral 7B, and Gemma 2 9B models
- Implemented layer-by-layer prediction analysis showing how knowledge emerges through transformer layers
- **Prompt format discovery**: Q&A format ("Question: What is the capital of Germany? Answer:") works better than completion format for many models
- Tested model differences in factual knowledge:
  - Llama 3: correctly predicts Berlin with both completion and Q&A formats (high confidence)
  - Mistral 7B: correctly predicts Berlin with both formats, shows interesting layer evolution patterns
  - Gemma 2 9B: conversational model that works best with Q&A format (79.1% confidence for Berlin)
- **Technical insight**: Found discrepancy between final residual stream and actual model output, revealing additional processing layers
- Temperature scaling reveals model confidence patterns across architectures

## Next Steps
- Test additional cutting-edge models: Qwen2/3, DeepSeek, Phi-4
- Investigate the residual stream vs final output discrepancy across different models
- Explore advanced TransformerLens features (attention patterns, circuit analysis)
- Compare prompt format sensitivity across more model families

## Key Findings
- Existing libraries that are useful for interpretability do not work with .gguf files and require raw transformer files
- Llama 4 Scout transformers size is too large for this machine, and there is no pre-quantized version
- TransformerLens provides vastly superior model support compared to tuned_lens (the downside, however, is the speed): Llama 3, Mistral, Gemma, Qwen2/3, and many more
- Model behavior varies by architecture and prompting:
  - Llama 3: correct knowledge (Berlin) with both completion and Q&A formats
  - Mistral 7B: correct knowledge (Berlin) with both formats, shows clean layer evolution
  - Gemma 2 9B: conversational model that works best with Q&A format (79% confidence vs generic responses)
- Interpretability insights from layer analysis:
  - Early layers: generic/random predictions across all models
  - Middle layers: correct answer starts emerging gradually
  - Final layers: confident prediction (but final residual stream differs from actual model output)
  - **Pattern holds across architectures**: Layer evolution consistent across Llama 3, Mistral, and Gemma
- Directional knowledge bias exists across models: "Berlin is capital of Germany" works better than "capital of Germany is Berlin"



## Chat Context
Assume the user is a software engineer with no python background.
The code is running on Macbook Pro M2 Max 64Gb.
