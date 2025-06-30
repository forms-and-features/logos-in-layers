# λόγος in layers

Systematic probing of open-weight LLMs to trace **where** and **how** concepts crystallise across transformer layers - groundwork for the nominalism ↔ realism debate.

## Overview

This project runs **iterative, self-contained experiments** (see the `000_`, `001_`, … directories) that dissect transformer behaviour from multiple angles.  
The first published iteration (`001_layers_and_logits/`) introduces a memory-efficient logit-lens to measure token-level entropy across layers. It examines four models — Gemma-2-9B, Qwen-3-8B, Mistral-7B-v0.1 and Llama-3-8B — revealing a shared entropy **collapse → rebound** signature around factual recall. 

## Experiments

### 000: Basic Chat
**Directory**: `000_basic_chat/`
**Files**: `run.py`

Basic chat interface for testing model responses and getting familiar with the models.

### 001: Layer-by-Layer Analysis
Detailed experiment documentation and findings have moved to `001_layers_and_logits/NOTES.md`.

## Setup

### Requirements
- **Apple Silicon Mac** (M1/M2/M3) with Metal GPU support
- **64GB+ RAM** recommended for larger models
- **50GB+ free disk space** for model downloads

### Installation

```bash
git clone https://github.com/forms-and-features/logos-in-layers.git
cd logos-in-layers
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

## Further Reading

For implementation details, developer-focused toggles, and methodology notes, see `PROJECT_NOTES.md`. 

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- **TransformerLens** for comprehensive interpretability tools
- **Hugging Face** for model hosting and ecosystem
- **Model Creators**: Meta, Mistral AI, Google, Alibaba for open-weight models
- **Apple** for Metal GPU acceleration

## AI-Assisted Development
- Conceptual direction: **OpenAI o3 pro**
- Implementation: **Anthropic Claude 4 Sonnet** and **OpenAI o4-mini** via **Cursor IDE** 
- Individual model evaluations and cross-model analysis: **OpenAI o3**

