# LLM Interpretability Project

Experiment with interpretability of open-weights Large Language Models using tuned lens analysis and other techniques.

## Overview

This project explores the internal workings of transformer models by analyzing how predictions evolve through different layers. We use the `tuned_lens` library to peek inside models and understand how they process information.

## What This Does

- **Layer-by-layer prediction analysis** using tuned lens
- **Model comparison** across different architectures (DialoGPT, Llama 3)
- **Temperature exploration** to understand prediction confidence
- **Bias detection** and knowledge representation analysis
- **Model compatibility testing** for various transformer architectures

## Hardware Requirements

- **Apple Silicon Mac** (M1/M2/M3) with Metal GPU support
- **64GB+ RAM** recommended for larger models
- **50GB+ free disk space** for model downloads

## Setup

### 1. Clone and Setup Environment

```bash
git clone <your-repo-url>
cd interpretability
python -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```

### 2. Hugging Face Authentication

```bash
huggingface-cli login
```

You'll need to accept the license agreements for gated models (Llama, etc.).

### 3. Run Analysis

```bash
python lens_analysis.py
```

## Supported Models

### ✅ Working with tuned_lens
- **GPT-2, GPT-Neo, GPT-NeoX** - EleutherAI models
- **OPT, Pythia** - Meta and EleutherAI models  
- **Llama 3** - Meta's latest models (confirmed working)
- **DialoGPT** - Microsoft's conversational models

### ❌ Not Supported
- **Mistral** - Architecture not supported by tuned_lens
- **Gemma** - Entire architecture family not supported (including Gemma 1/7B)
- **Qwen2/3** - Latest architecture not supported yet
- **DeepSeek-R1-Distill-Qwen** - Based on Qwen2, not supported
- **GGUF files** - Require raw transformer format

## Key Findings

### Model Bias Analysis
- **DialoGPT**: Confidently wrong about German capital (Frankfurt > Cologne > Berlin)
- **Llama 3**: Correct knowledge with clear layer evolution (0% → 67% → 79% confidence for Berlin)
- **Temperature scaling**: Reveals deep-seated biases vs. surface-level uncertainty

### Interpretability Insights
- **Early layers**: Generic/random predictions
- **Middle layers**: Correct answer starts emerging
- **Final layers**: Confident correct prediction
- **Directional bias**: Some models better at "Berlin is capital of ___" than "capital of Germany is ___"



## Technical Notes

- **Device Management**: Automatic MPS (Metal) GPU detection for Apple Silicon
- **Memory Optimization**: Half-precision (float16) loading to reduce memory usage
- **Quantization**: Avoided due to Apple Silicon compatibility issues
- **Model Compatibility**: Automatic fallback for unsupported architectures



## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **tuned_lens** library for interpretability tools
- **Hugging Face** for model hosting and transformers library
- **EleutherAI, Meta** for open-weight models
- **Apple** for Metal GPU acceleration

### AI-Assisted Development
This research was guided by **OpenAI ChatGPT o3** for conceptual direction and implemented with **Anthropic Claude 4 Sonnet** via **Cursor IDE** for code development and analysis. 