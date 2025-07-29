# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an LLM interpretability research project investigating the philosophical debate between nominalism and realism through empirical analysis of transformer models. The project uses logit lens techniques to analyze layer-by-layer token entropy and factual recall patterns across multiple open-weight models.

## Project Structure

- `000_basic_chat/` - Simple chat interface using llama-cpp for basic model interaction
- `001_layers_and_logits/` - Main experimental suite with layer-by-layer analysis
  - `run.py` - Core analysis script that evaluates multiple models
  - `run-latest/` - Latest experimental results and evaluation reports
  - `NOTES.md` - Technical implementation notes
- `models/` - Local model storage (GGUF format)
- `PROJECT_NOTES.md` - Comprehensive philosophical context and development roadmap

## Development Environment

### Requirements
- Python 3.10+ with virtual environment
- Apple Silicon Mac (M1/M2/M3) with Metal GPU support recommended
- 64GB+ RAM for larger models
- 50GB+ disk space for model downloads

### Setup Commands
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
huggingface-cli login  # Required for gated models
```

### Running Experiments
```bash
# Basic chat interface
cd 000_basic_chat && python run.py

# Layer-by-layer analysis (all models)
cd 001_layers_and_logits && python run.py

# Single model analysis
cd 001_layers_and_logits && python run.py --models meta-llama/Meta-Llama-3-8B

# Run KL sanity test to validate normalization scaling
cd 001_layers_and_logits && python run.py --self-test meta-llama/Meta-Llama-3-8B
# Or run standalone:
cd 001_layers_and_logits && python kl_sanity_test.py meta-llama/Meta-Llama-3-8B
```

## Code Architecture

### Model Support
The codebase distinguishes between:
- **MPS_SAFE_MODELS**: Can run on Apple Silicon GPU (Mistral-7B, Gemma-2-9B, Qwen3-8B, Meta-Llama-3-8B)
- **CUDA_ONLY_MODELS**: Require NVIDIA GPU (Yi-34B, Qwen3-14B, Gemma-2-27b)

### Key Components
- **Logit Lens Pipeline**: `001_layers_and_logits/run.py` implements layer-by-layer token prediction analysis
- **KL Sanity Test**: `001_layers_and_logits/kl_sanity_test.py` validates normalization scaling correctness
- **Deterministic Execution**: SEED=316 with torch deterministic algorithms for reproducible results
- **Output Formats**: 
  - JSON metadata files with run configuration
  - CSV files with per-layer metrics (`*-pure-next-token.csv`, `*-records.csv`)
  - Markdown evaluation reports

### Core Metrics
- **Copy-collapse detection**: Identifies when models echo prompt tokens
- **Semantic collapse**: Layer where correct answer becomes top-1 prediction
- **Entropy tracking**: Per-layer uncertainty measurement
- **Top-k analysis**: Configurable via `TOP_K_RECORD` and `TOP_K_VERBOSE`

### Recent Improvements (Section 1.1 Fix - COMPLETE & VERIFIED)
- **✅ Fixed RMSNorm epsilon placement**: Epsilon now correctly placed inside sqrt as per official formula
- **✅ Architecture-aware γ selection**: 
  - Pre-norm models (Llama, Mistral, Gemma): Use **next block's ln1** (or ln_final for last layer)
  - Post-norm models (GPT-J, GPT-Neo): Use **current block's ln2**
- **✅ Robust architecture detection**: `detect_model_architecture()` examines block structure to correctly distinguish pre/post-norm
- **✅ Multi-layer KL sanity test**: `--self-test` validates γ=1 vs learned γ across 25%, 50%, 75% depth layers
- **✅ Comprehensive unit tests**: `test_normalization.py` validates both pre-norm AND post-norm logic
- **✅ Eliminated scaling artifacts**: Both pre-norm and post-norm models now have correct γ scaling

**Critical fix**: This addresses the scaling bug where pre-norm models were using wrong γ (inflating/deflating logits by γ_{L+1}/γ_L ratios), which could create spurious "early semantic meaning" and undermine the philosophical claims about nominalism vs realism. Post-norm models are also now correctly handled.

## Philosophical Context

This project targets the centuries-old dispute between nominalism (only particular tokens exist) and realism (mind-independent universals exist). The experimental approach:

1. **Stage 1**: Establish robust logit-lens baselines across models
2. **Stage 2**: Add causal interventions and cross-modal probes
3. **Stage 3**: Discriminate between metalinguistic nominalism and realism

See `PROJECT_NOTES.md` for detailed philosophical framework and planned experimental variations.

## Model Integration

The project uses transformer-lens for interpretability tools and supports both:
- Direct Hugging Face model loading with quantization options
- Local GGUF models via llama-cpp (see `000_basic_chat/run.py`)

Authentication required for gated models (Llama family) via `huggingface-cli login`.