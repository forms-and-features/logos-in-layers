# λόγος in layers

An experiment in LLM interpretability to provide empirical evidence for nominalism vs realism debate.

## Overview

The first experimental suite (001_layers_and_logits/) introduces a lightweight logit-lens that tracks per-layer token-entropy in seven open-weight models.

Across all four, we see the typical "copy plateau, then sharp entropy drop" that coincides with factual recall. These measurements form the baseline for the causal and cross-modal probes planned in later stages.

## Experiments

### 001: Layer-by-Layer Analysis

See `001_layers_and_logits/run-latest/*.md` for evaluation reports of the latest run, and `001_layers_and_logits/NOTES.md` for technical notes.

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

For details and methodology notes, see `PROJECT_NOTES.md`. 

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

