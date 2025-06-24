# LLM Interpretability Project

Exploring how transformer models process information layer-by-layer, using TransformerLens to analyze how predictions evolve through the network.

## Overview

This project examines how predictions change through different layers of transformer models. We analyze how several models handle a factual question to understand their internal processing patterns.

## Experiments

### 001: Layer-by-Layer Logit Analysis
**Files**: `001_layers_and_logits.py` + `001_layers_and_logits.md`

Analysis of how the prediction for "What is the capital of Germany?" evolves through the layers of four different models:

- **Qwen3-8B** (36 layers): Shows a distinctive "Germany → Berlin" transition pattern
- **Meta-Llama-3-8B** (32 layers): More direct path to the correct answer
- **Mistral-7B-v0.1** (32 layers): Early emergence of German-related tokens
- **Gemma-2-9B** (42 layers): Later convergence on the factual answer

**Finding**: All models converge on the correct answer around 80-85% through their layers, followed by confidence calibration in the final layers.

## Setup

### Requirements
- **Apple Silicon Mac** (M1/M2/M3) with Metal GPU support
- **64GB+ RAM** recommended for larger models
- **50GB+ free disk space** for model downloads

### Installation

```bash
git clone <your-repo-url>
cd interpretability
python -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```

### Authentication

```bash
huggingface-cli login
```

Accept license agreements for gated models (Llama, etc.).

### Running Experiments

```bash
# Run the layer-by-layer analysis
python 001_layers_and_logits.py
```

## Supported Models

### ✅ Confirmed Working
- **Llama 3** (Meta)
- **Mistral 7B** (Mistral AI)  
- **Gemma 2** (Google)
- **Qwen3** (Alibaba)

### ❌ Not Supported
- **GGUF files** - Require raw transformer format
- **Extremely large models** - Hardware constraints

## Key Insights

### Patterns Identified
- **Convergence timing**: Models consistently arrive at the correct answer around 80-85% through their depth
- **Processing phases**: Early layers show noise, middle layers develop the answer, final layers calibrate confidence
- **Internal confidence**: Models exhibit higher internal certainty than their final output probabilities suggest
- **Directional asymmetry**: "Berlin is capital of Germany" produces higher confidence than "Germany's capital is Berlin"

### Technical Notes
- **LayerNorm correction**: Important to apply the same normalization the model uses internally
- **Final layer adjustments**: Last few layers seem to moderate the confidence for more natural responses
- **Temperature effects**: Lower temperature shows what the model "really thinks"



## Possible Next Steps

- **Attention patterns**: Look at which parts of the input different layers pay attention to
- **Other types of questions**: Try math problems, historical facts, etc.
- **Intervention experiments**: Try changing intermediate layers to see what happens
- **Newer/larger models**: Test the same patterns on bigger models when possible

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- **TransformerLens** for comprehensive interpretability tools
- **Hugging Face** for model hosting and ecosystem
- **Model Creators**: Meta, Mistral AI, Google, Alibaba for open-weight models
- **Apple** for Metal GPU acceleration

### AI-Assisted Development
Research guided by **OpenAI ChatGPT o3** for conceptual direction, implemented with **Anthropic Claude 4 Sonnet** via **Cursor IDE** for code development and analysis. 