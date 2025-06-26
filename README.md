# LLM Interpretability Project

- Systematic probes of open-weight LLMs to track **where** and **how** concepts crystallise inside transformer layers — informing a long-term study of nominalism ↔ realism.

## Overview

This project runs **iterative, self-contained experiments** (see the `000_`, `001_`, … directories) that dissect transformer behaviour from multiple angles.  
The first published iteration (`001_layers_and_logits/`) introduces a memory-efficient logit-lens to measure token-level entropy across layers. It examines four models — Gemma-2-9B, Qwen-3-8B, Mistral-7B-v0.1 and Llama-3-8B — revealing a shared entropy **collapse → rebound** signature around factual recall. 

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
- All four models exhibit a sharp entropy **collapse** to near-deterministic predictions roughly 75-85 % into their layer stack. Gemma collapses twice: an early syntactic ':' placeholder, then a later semantic collapse onto "Berlin" (≈ 83 %).
- A consistent 1–2 bit **entropy rebound** appears at the final unembedding layer in every model.
- Mid-stack **meta-token fixation** (e.g. "Answer"/"answer") shows up in three models, hinting at a symbolic slot-filling stage before concrete entity resolution.
- Model-specific quirks—Gemma's colon-spam, Qwen's underscore phase, Mistral's late Washington distraction—highlight spurious template biases.

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
- **Entropy collapse & rebound**: Near-zero entropy collapse followed by a modest (≈ 1–2 bit) rebound at the unembedding step.
- **Mid-stack meta tokens**: "Answer"-style labels frequently dominate before the model commits to the factual entity.
- **Late-binding factual knowledge**: Correct answers emerge consistently at 75-85% network depth
- **Hierarchical processing**: Abstract categories ("capital") before specific instances ("Berlin")
- **Model-specific artifacts**: Each architecture shows unique spurious features and biases
- **Surface-form sensitivity**: Strong dependence on prompt formatting and direction

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

## Further Reading

For implementation details, developer-focused toggles, and methodology notes, see `PROJECT_NOTES.md`. 