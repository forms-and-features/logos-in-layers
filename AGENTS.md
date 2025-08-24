# AGENT.md

This file provides guidance to AI agents when working with code in this repository.

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

### Virtualenv & CLI Tips (for agents)
- Always activate the venv before running project scripts: `source venv/bin/activate && <command>`; for example: `source venv/bin/activate && scripts/run_cpu_tests.sh`. This is known to work.
- DO NOT ASSSUME THAT PACKAGES ARE MISSING - ANY ISSUES YOU ENCOUNTER ARE BECAUSE YOU DID NOT ACTIVATE THE ENVIRONMENT PROPERLY. FOLLOW THE INSTRUCTION.
- Prefer invoking helper scripts that already honor the venv, e.g. `scripts/run_cpu_tests.sh` (it uses `PYTHON_BIN` and defaults to `venv/bin/python`).
- If `venv/bin/python` is a symlink to a host-managed interpreter and fails in a sandbox, use `source venv/bin/activate` instead of calling the path directly.
- If activation still fails due to a broken interpreter path, recreate the venv inside the workspace and reinstall deps: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`.
- Network-restricted runs: CPU-only tests and most unit tests do not require network. The KL self-test may need network/HF auth; ask for approval to run without sandbox if blocked.

### Running Experiments
```bash
# Basic chat interface
cd 000_basic_chat && python run.py

# Layer-by-layer analysis (all models)
cd 001_layers_and_logits && python run.py

# Single model analysis
cd 001_layers_and_logits && python run.py meta-llama/Meta-Llama-3-8B

# Run KL sanity test to validate normalization scaling
cd 001_layers_and_logits && python run.py --self-test meta-llama/Meta-Llama-3-8B
# Or run standalone:
cd 001_layers_and_logits && python kl_sanity_test.py meta-llama/Meta-Llama-3-8B
```

## Code Architecture

### Model Support
Device selection is dynamic. The runner estimates whether a model will fit on each available device and auto-picks the best fit in order `cuda → mps → cpu`. The curated model list remains advisory; models that do not fit on any device are skipped with a clear log.

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

### Recent Improvements (Section 1.1 Fix - FINAL & VERIFIED)
- **✅ Fixed RMSNorm epsilon placement**: Epsilon now correctly placed inside sqrt as per official formula
- **✅ Architecture-aware γ selection**: 
  - Pre-norm models (Llama, Mistral, Gemma): Use **next block's ln1** (or ln_final for last layer)
  - Post-norm models (GPT-J, GPT-Neo, Falcon): Use **current block's ln2**
- **✅ Structural architecture detection**: `detect_model_architecture()` examines **last child module** to reliably distinguish architectures
- **✅ Multi-layer KL sanity test**: `kl_sanity_test.py` validates γ=1 vs learned γ across multiple depth layers (25%, 50%, 75%)
- **✅ Critical unit test assertions**: `test_normalization.py` includes explicit post-norm detection and γ selection validation
- **✅ Fail-fast behavior**: CLI `--self-test` aborts analysis if scaling validation fails

**Critical achievement**: The architecture detector now uses **structural analysis** (examining if the last child module is a normalization layer) rather than attribute presence, correctly identifying GPT-J/Falcon/NeoX as post-norm. This eliminates the γ_{L+1}/γ_L scaling artifacts that could create spurious "early semantic meaning" across **all** supported model families.

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
