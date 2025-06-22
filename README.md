# LLM Interpretability Project

Experiment with interpretability of open-weights Large Language Models using TransformerLens and other cutting-edge techniques.

## Overview

This project explores the internal workings of transformer models by analyzing how predictions evolve through different layers. We use the `TransformerLens` library to peek inside models and understand how they process information, with support for the latest model architectures.

## What This Does

- **Layer-by-layer prediction analysis** using TransformerLens
- **Model comparison** across different architectures (DialoGPT, Llama 3, Mistral 7B)
- **Temperature exploration** to understand prediction confidence
- **Bias detection** and knowledge representation analysis
- **Cutting-edge model support** for Mistral, Gemma, Qwen, and more

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

### ✅ Working with TransformerLens
- **GPT-2, GPT-Neo, GPT-NeoX** - EleutherAI models
- **OPT, Pythia** - Meta and EleutherAI models  
- **Llama 3** - Meta's latest models (confirmed working)
- **Mistral 7B** - Mistral AI models (confirmed working)
- **DialoGPT** - Microsoft's conversational models
- **Gemma** - Google's Gemma model family (supported)
- **Qwen2/3** - Alibaba's latest models (supported)
- **DeepSeek** - DeepSeek model family (supported)

### ❌ Not Supported
- **GGUF files** - Require raw transformer format
- **Extremely large models** - Limited by hardware constraints

## Key Findings

### Model Bias Analysis
- **DialoGPT**: Confidently wrong about German capital (Frankfurt > Cologne > Berlin)
- **Llama 3**: Correct knowledge with clear layer evolution (0% → 67% → 79% confidence for Berlin)
- **Mistral 7B**: Correct knowledge with clean progression (0.4% → 2.2% → 22.6% confidence for Berlin)
- **Temperature scaling**: Reveals deep-seated biases vs. surface-level uncertainty across architectures

### Interpretability Insights
- **Early layers**: Generic/random predictions
- **Middle layers**: Correct answer starts emerging
- **Final layers**: Confident correct prediction
- **Universal pattern**: Layer evolution holds across DialoGPT, Llama 3, and Mistral architectures
- **Directional bias**: Models consistently better at "Berlin is capital of ___" than "capital of Germany is ___"



## Technical Notes

- **Device Management**: Automatic MPS (Metal) GPU detection for Apple Silicon
- **Memory Optimization**: Half-precision (float16) loading to reduce memory usage
- **Quantization**: Avoided due to Apple Silicon compatibility issues
- **Model Compatibility**: Automatic fallback for unsupported architectures



## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **TransformerLens** library for comprehensive interpretability tools
- **Hugging Face** for model hosting and transformers library
- **EleutherAI, Meta, Mistral AI** for open-weight models
- **Apple** for Metal GPU acceleration

### AI-Assisted Development
This research was guided by **OpenAI ChatGPT o3** for conceptual direction and implemented with **Anthropic Claude 4 Sonnet** via **Cursor IDE** for code development and analysis. 