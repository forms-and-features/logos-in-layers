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
- **Llama 3** - Meta's latest models (confirmed working with both completion and Q&A formats)
- **Mistral 7B** - Mistral AI models (confirmed working with both formats)
- **Gemma 2** - Google's latest models (confirmed working, prefers Q&A format)
- **Qwen2/3** - Alibaba's latest models (supported, untested)
- **DeepSeek** - DeepSeek model family (supported, untested)
- **Phi-4** - Microsoft's latest models (supported, untested)

### ❌ Not Supported
- **GGUF files** - Require raw transformer format
- **Extremely large models** - Limited by hardware constraints

## Key Findings

### Model Analysis Results
- **Llama 3**: Correct knowledge (Berlin) with both completion and Q&A formats - robust across prompting styles
- **Mistral 7B**: Correct knowledge (Berlin) with both formats - shows clean layer evolution and strong final confidence
- **Gemma 2 9B**: Conversational model that works best with Q&A format (79.1% confidence for Berlin vs generic responses with completion)
- **Prompt format sensitivity**: Q&A format ("Question: ... Answer:") more reliable than completion format across models
- **Temperature scaling**: Reveals model confidence patterns - low temp shows true beliefs, high temp shows robustness

### Interpretability Insights
- **Early layers**: Generic/random predictions across all tested models
- **Middle layers**: Correct answer starts emerging gradually
- **Final layers**: Confident correct prediction (but see technical note below)
- **Universal pattern**: Layer evolution holds across Llama 3, Mistral, and Gemma architectures
- **Directional bias**: Models consistently better at "Berlin is capital of ___" than "capital of Germany is ___"
- **Technical discovery**: Final residual stream differs from actual model output, revealing additional processing steps



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