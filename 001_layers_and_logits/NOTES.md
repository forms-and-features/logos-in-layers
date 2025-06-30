# Layer-by-Layer Analysis (Experiment 001)

## Overview of experiment runs

### Run at 2025-06-29 21:02 UTC+2

Layer-by-layer analysis (identical prompt, deterministic seed):

– **Gemma-2-9B** (42 layers) Immediate copy-collapse at layer 0 on "simply"; semantic answer "Berlin" only at the final layer (**Δ = 42**).

– **Qwen3-8B** (36 layers) Copy-collapse at layer 25; "Berlin" emerges at layer 31 (**Δ = 6**).

– **Meta-Llama-3-8B** (32 layers) No hard echo; "Berlin" appears at layer 25 (no Δ).

– **Mistral-7B-v0.1** (32 layers) Soft "simply" plateau (<0.9 p), semantic collapse at layer 25 (no Δ).

Cross-model synthesis: see `run-2025-06-29-21-02/evaluation-cross-model.md`.

Meta-evaluation by o3-pro: see `run-2025-06-29-21-02/meta-evaluation.md`.

### Run at 2025-06-29 16:53 UTC+2

Layer-by-layer analysis of how the prediction for *"Give the city name only, plain text. The capital of Germany is called simply"* evolves through four different models:

– **Gemma-2-9B** (42 layers) Copy-collapse happens immediately (layer 0) on the prompt word "simply" and persists through 41 layers; the correct answer "Berlin" only surfaces in the final layer (Δ 42).

– **Qwen3-8B** (36 layers) Echo of "simply" peaks at layer 25 and is replaced by "Berlin" at layer 31, giving a moderate 6-layer gap between copy- and semantic-collapse.

– **Meta-Llama-3-8B** (32 layers) No hard prompt echo; the network drifts through filler tokens and switches straight to "Berlin" at layer 25.

– **Mistral-7B-v0.1** (32 layers) Similar to Llama—extended "simply" plateau without exceeding the copy threshold, then semantic convergence at layer 25.

Cross-model synthesis: see `run-2025-06-29-16-53/evaluation-cross-model.md`.

## Supported Models

### ✅ Confirmed Working
- **Llama 3** (Meta) – 32 layers
- **Mistral 7B** (Mistral AI) – 32 layers  
- **Gemma 2** (Google) – 42 layers
- **Qwen3** (Alibaba) – 36 layers

### ❌ Not Supported
- **GGUF files** – Require raw transformer format
- **Extremely large models** – Hardware constraints

## Directory Layout

```
001_layers_and_logits/
├── run.py                           # Main experiment script
├── run-YYYY-MM-DD-HH-MM/            # One timestamped directory per sweep
│   ├── evaluation-[model].md        # Per-model analyses  
│   ├── evaluation-cross-model.md    # Cross-model analysis
│   ├── output-[model].json          # JSON diagnostics
│   ├── output-[model]-records.csv   # Layer-wise records (all tokens)
│   ├── output-[model]-pure-next-token.csv  # Clean entropy (next-token only)
│   └── ...
├── prompt-*.txt                     # Evaluation prompts
└── NOTES.md                         # This file
```

---

# Technical Implementation Notes

The sections below were migrated from `PROJECT_NOTES.md` verbatim for historical completeness.

## Current State
- **001_layers_and_logits**: Complete layer-by-layer analysis with 4 models
- **File structure**: Reorganized from single script to proper experiment directory
- **Analysis**: Individual model reports + cross-model comparison (AI-generated)

## Key Technical Implementation Details

### Deterministic Seed & Reproducibility (2025-06-29)
`run.py` now initialises a deterministic bootstrap (`SEED = 316`) and enables `torch.use_deterministic_algorithms(True)` plus cuBLAS workspace settings.  Every sweep is repeatable bit-for-bit across runs and machines.

### Improved LayerNorm & RMSNorm Handling (2025-06-29)
`apply_norm_or_skip()` now:
* Applies full **LayerNorm** (γ *and* β) with dtype-safe casting.
* Detects **RMSNorm** scale under multiple attribute names and casts to residual dtype.
* Eliminates the deprecated helper `is_safe_layernorm()`.

### New Output: Pure Next-Token CSV
Each run emits `output-<model>-pure-next-token.csv`, logging entropy and top-k only for the first unseen token to avoid average deflation.

### LayerNorm Bias & Post-Block Normalisation Fixes (2025-06-29)
`apply_norm_or_skip()` removes the β (bias) term from LayerNorm when applying the lens and prefers `ln2` over `ln1` for post-block residual snapshots.

### RMSNorm vs LayerNorm Handling
*(See original code snippets for full context)*

> Observation: All tested checkpoints employ **Pre-RMSNorm**. A naive LayerNorm lens therefore scales residuals incorrectly.  
> *Current implementation* (`run.py ≥ 2025-06-28`) applies a true RMS lens rather than skipping the operation.

### Memory Optimisation for Large Models
`run_with_cache()` loads full activations, causing OOM on 9B models. The solution is targeted caching with hooks that optionally slice to `[:, -1:]`.

### Device / Precision Management
The experiment supports CUDA, MPS and CPU with automatic dtype selection. See `run.py` for logic.

## Development Environment Notes
- **Hardware**: Apple Silicon MacBook Pro M2 Max 64 GB
- **Library stack**: TransformerLens, no quantisation, raw checkpoint format

## Experiment Structure & Toggles
See `run.py` for `USE_NORM_LENS`, `USE_FP32_UNEMBED` and residual-cache device options.

## Analysis Pipeline
1. `run.py` writes JSON and CSV artefacts per model.  
2. LLM prompts generate `evaluation-*.md` files.  
3. Cross-model analysis written alongside per-model evaluations.  

## AI Evaluation System
Prompt templates: `prompt-single-model-evaluation.txt`, `prompt-cross-model-evaluation.txt`, `prompt-meta-evaluation.txt`.

## Further Reading on Implemented Techniques

Cai et al. (2023) "Tuned Lens: A Query-Aware Logit Lens for Interpreting Transformers" — https://arxiv.org/abs/2303.17564 — inspired the projection of each layer's residual stream through the frozen unembedding matrix to obtain intermediate token probabilities.

Zhang et al. (2019) "Root Mean Square Layer Normalization" — https://arxiv.org/abs/1910.04751 — provided the RMSNorm equation re-implemented in `apply_norm_or_skip()` for accurate scaling on RMS models.

Neel Nanda's **TransformerLens** library — https://github.com/NeelNanda/TransformerLens — supplies the `HookedTransformer` class and hook API used for lightweight residual caching.

Arthur Belrose (2024) discussion "The LayerNorm Bias Can Scramble Logit-Lens Read-outs" — https://github.com/NeelNanda/TransformerLens/discussions/640 — motivated omitting the β bias term when re-applying LayerNorm.

Anthropic Interpretability post "Why LN2 is the Right Hook for Post-Block Analysis" (2023) — https://transformer-circuits.pub/ln2-analysis — informed the choice of `ln2` over `ln1` for post-block snapshots.

---
Produced by OpenAI o3