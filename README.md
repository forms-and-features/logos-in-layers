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
**Directory**: `001_layers_and_logits/`
**Files**: `run.py`, individual model evaluations (`evaluation-*.md`), cross-model analysis (`evaluation-cross-model.md`), structured outputs (`output-*.json`, `output-*-records.csv`, `output-*-pure-next-token.csv`), and evaluation prompts (`prompt-*.txt`)

Layer-by-layer analysis of how the prediction for "What is the capital of Germany?" evolves through four different models:

- **Qwen3-8B** (36 layers): Shows distinctive "Germany → Berlin" transition with template-driven behavior
- **Meta-Llama-3-8B** (32 layers): More direct path with anomalous junk tokens in mid-layers  
- **Mistral-7B-v0.1** (32 layers): Early emergence of German-related tokens with formatting bias
- **Gemma-2-9B** (42 layers): Later convergence with early over-confidence on punctuation


**Cross-Model Findings** (see `evaluation-cross-model.md` for full analysis):

– A depth-normalised **entropy collapse**: the correct answer becomes near-deterministic at ~0.78 ± 0.05 of each model's layer stack (e.g. Llama & Mistral L 25/32, Qwen L 28/36, Gemma L 35/42).

– **Concept-before-entity** progression: generic tokens like "capital" (or placeholders like "Answer/") peak 3-5 layers before "Berlin" dominates.

– Universal **entropy rebound** of ≈1–2 bits after `ln_final` + unembed, indicating a calibration step rather than new evidence.

– **Mid-stack category plateau**: all checkpoints first converge on a *generic answer class* (e.g. the word "city" or the placeholder "____") before the specific referent appears.  This plateau spans ~5–10 layers and reaches entropy well below 4 bits.

– **Late formatting override**: after "Berlin" peaks, probability mass drifts back to surface-form tokens (`<strong>`, numerals, full-width punctuation).  The factual state is still recoverable at low temperature (τ = 0.1) but is suppressed in generation-ready logits.

– **Early-layer heterogeneity**: Gemma is over-confident on punctuation (entropy < 10⁻⁶ bits on ':'), whereas the others emit high-entropy junk or multilingual shards, revealing tokeniser noise when semantics are undeveloped.

– Model-specific artefacts persist (colon-spam, underscore phase, Washington detour), underscoring template and corpus biases.

## Setup

### Requirements
- **Apple Silicon Mac** (M1/M2/M3) with Metal GPU support
- **64GB+ RAM** recommended for larger models
- **50GB+ free disk space** for model downloads

### Installation

```bash
git clone <your-repo-url>
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

## Supported Models

### ✅ Confirmed Working
- **Llama 3** (Meta) - 32 layers
- **Mistral 7B** (Mistral AI) - 32 layers  
- **Gemma 2** (Google) - 42 layers
- **Qwen3** (Alibaba) - 36 layers

### ❌ Not Supported
- **GGUF files** - Require raw transformer format
- **Extremely large models** - Hardware constraints


## File Structure

```
001_layers_and_logits/
├── run.py                         # Main experiment script
├── evaluation-[model].md          # Per-model analyses  
├── evaluation-cross-model.md      # Cross-model analysis
├── output-[model].json            # JSON metadata (per model)
├── output-[model]-records.csv     # Layer-wise records
├── output-[model]-pure-next-token.csv  # Clean entropy (first unseen token only)
├── prompt-*.txt                   # Evaluation prompts
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

