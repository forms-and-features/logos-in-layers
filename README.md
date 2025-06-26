# LLM Interpretability Project

Exploring how transformer models process information layer-by-layer, using TransformerLens to analyze how predictions evolve through the network.

## Overview

This project examines how predictions change through different layers of transformer models. We analyze how several models handle a factual question to understand their internal processing patterns.

## Experiments

### 000: Basic Chat
**Directory**: `000_basic_chat/`
**Files**: `run.py`

Basic chat interface for testing model responses and getting familiar with the models.

### 001: Layer-by-Layer Analysis
**Directory**: `001_layers_and_logits/`
**Files**: `run.py`, individual model evaluations (`evaluation-*.md`), cross-model analysis (`interpretability/001_layers_and_logits/evaluation-cross-model.md`), raw outputs (`output-*.txt`), and evaluation prompts (`prompt-*.txt`)

Layer-by-layer analysis of how the prediction for "What is the capital of Germany?" evolves through four different models:

- **Qwen3-8B** (36 layers): Shows distinctive "Germany → Berlin" transition with template-driven behavior
- **Meta-Llama-3-8B** (32 layers): More direct path with anomalous junk tokens in mid-layers  
- **Mistral-7B-v0.1** (32 layers): Early emergence of German-related tokens with formatting bias
- **Gemma-2-9B** (42 layers): Later convergence with early over-confidence on punctuation


**Cross-Model Findings**: 
- All models converge on the correct answer around 75-85% through their layers
- Consistent "category-before-instance" pattern (generic "capital" before specific "Berlin")
- Model-specific anomalies reveal spurious features and template biases
- Asymmetric factual recall (reverse relations easier than forward)

## Setup

### Requirements
- **Apple Silicon Mac** (M1/M2/M3) with Metal GPU support
- **64GB+ RAM** recommended for larger models
- **50GB+ free disk space** for model downloads

### Installation

```bash
git clone <your-repo-url>
cd tinycave
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
# Run the basic chat interface
cd 000_basic_chat
python run.py

# Run the layer-by-layer analysis (all models)
cd 001_layers_and_logits
python run.py
```

## Supported Models

### ✅ Confirmed Working
- **Llama 3** (Meta) - 32 layers
- **Mistral 7B** (Mistral AI) - 32 layers  
- **Gemma 2** (Google) - 42 layers
- **Qwen3** (Alibaba) - 36 layers

### ❌ Not Supported
- **GGUF files** - Require raw transformer format
- **Extremely large models** - Hardware constraints

## Key Insights

### Interpretability Patterns
- **Late-binding factual knowledge**: Correct answers emerge consistently at 75-85% network depth
- **Hierarchical processing**: Abstract categories ("capital") before specific instances ("Berlin")
- **Model-specific artifacts**: Each architecture shows unique spurious features and biases
- **Surface-form sensitivity**: Strong dependence on prompt formatting and direction

## Technical Implementation

### Normalization Handling
- **Automatic detection**: Script identifies RMSNorm vs LayerNorm architectures
- **Safe application**: Only applies normalization lens to vanilla LayerNorm to avoid distortion
- **Raw mode fallback**: Maintains interpretability for non-vanilla architectures

### Memory Management
- **Targeted caching**: Only stores required residual streams instead of full activations
- **Device optimization**: Automatic GPU/CPU management with appropriate precision
- **Efficient computation**: Top-k selection before softmax for performance

### Analysis Pipeline
- **Individual evaluation**: Detailed per-model analysis with anomaly detection
- **Cross-model comparison**: Systematic comparison identifying universal patterns
- **AI-assisted insights**: Expert-level evaluation using specialized interpretability prompts

## File Structure

```
001_layers_and_logits/
├── run.py                           # Main experiment script
├── evaluation-[model].md            # Individual model analyses  
├── output-[model].txt               # Raw experimental outputs
├── prompt-*.txt                     # Evaluation prompts
└── interpretability/
    └── 001_layers_and_logits/
        └── evaluation-cross-model.md # Cross-model comparative analysis
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- **TransformerLens** for comprehensive interpretability tools
- **Hugging Face** for model hosting and ecosystem
- **Model Creators**: Meta, Mistral AI, Google, Alibaba for open-weight models
- **Apple** for Metal GPU acceleration

### AI-Assisted Development
Research guided by **OpenAI o3** for conceptual direction, implemented with **Anthropic Claude 4 Sonnet** via **Cursor IDE** for code development and analysis. Individual model evaluations and cross-model analysis generated using OpenAI o3. 