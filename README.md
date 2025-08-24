# λόγος in layers

An experiment in LLM interpretability to provide empirical evidence for nominalism vs realism debate.

## Overview

The first experimental suite (001_layers_and_logits/) introduces a lightweight logit-lens that tracks per-layer token-entropy in seven open-weight models.

Across all four, we see the typical "copy plateau, then sharp entropy drop" that coincides with factual recall. These measurements form the baseline for the causal and cross-modal probes planned in later stages.

## Experiments

### 001: Layer-by-Layer Analysis

See `001_layers_and_logits/README.md` for detailed usage, outputs, testing, and internals. Evaluation reports for the latest run live in `001_layers_and_logits/run-latest/*.md`; additional implementation notes are in `001_layers_and_logits/NOTES.md`.
Device notes: the script now auto-selects the best device per model (prefers `cuda` → `mps` → `cpu`) based on a conservative memory‑fit estimate. You can still override with `--device {cuda|mps|cpu}` when needed.

Precision policy: on CPU, models ≤27B use fp32 by default; ≥30B use bf16 to fit comfortably on 256 GiB hosts. When the compute dtype is bf16/fp16, the unembedding matrix is automatically promoted to fp32 and logits are decoded in fp32 for stability. LayerNorm/RMSNorm statistics are computed in fp32 internally and cast back. No flags needed; defaults remain unchanged for ≤27B.

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

This creates a fresh `run-latest/` (rotating any previous one to `run-YYYYMMDD-HHMM/`) and writes:

- `output-<model>.json`: compact metadata (diagnostics, final prediction, model stats)
- `output-<model>-records.csv`: per-layer/per-position top‑k with rest_mass
- `output-<model>-pure-next-token.csv`: per-layer next-token top‑k with collapse flags

Run a single model to a custom directory:

```bash
python run.py --device cpu --out_dir ./some_dir mistralai/Mistral-7B-v0.1
```

### Testing and Self-Test

- All CPU-only tests (no downloads): `scripts/run_cpu_tests.sh`
- Single test: `venv/bin/python 001_layers_and_logits/tests/test_numerics.py`
- KL self-test (network+HF auth may be required):
  - Default: `scripts/self_test.sh`
  - Custom: `scripts/self_test.sh <MODEL_ID> <DEVICE>` (e.g., `mps`)

Notes:
- In `--self-test` mode, the script validates scaling and prints results but does not write JSON/CSV artifacts or rotate `run-latest/`.

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
- Conceptual direction: **OpenAI GPT-5 pro**, **o3 pro**
- Implementation: **OpenAI GPT-5** via **codex-cli**; **Anthropic Claude 4 Sonnet** and **OpenAI o4-mini** via **Cursor IDE**
- Individual model evaluations and cross-model analysis: **OpenAI GPT-5**; **OpenAI o3**
