# Layer-by-Layer Analysis (Experiment 001)

## Overview

Layer-by-layer analysis of how the prediction for *"What is the capital of Germany?"* evolves through four different models:

- **Qwen3-8B** (36 layers): Shows distinctive "Germany → Berlin" transition with template-driven behavior  
- **Meta-Llama-3-8B** (32 layers): More direct path with anomalous junk tokens in mid-layers  
- **Mistral-7B-v0.1** (32 layers): Early emergence of German-related tokens with formatting bias  
- **Gemma-2-9B** (42 layers): Later convergence with early over-confidence on punctuation

### Cross-Model Findings

See evaluation-cross-model.md.

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
├── run.py                         # Main experiment script
├── evaluation-[model].md          # Per-model analyses  
├── evaluation-cross-model.md      # Cross-model analysis
├── output-[model].json            # JSON metadata (per model)
├── output-[model]-records.csv     # Layer-wise records
├── output-[model]-pure-next-token.csv  # Clean entropy (first unseen token only)
├── prompt-*.txt                   # Evaluation prompts
```

---

# Technical Implementation Notes

The sections below were migrated from `PROJECT_NOTES.md` verbatim for historical completeness.

## Current State
- **001_layers_and_logits**: Complete layer-by-layer analysis with 4 models
- **File structure**: Reorganized from single script to proper experiment directory
- **Analysis**: Individual model reports + cross-model comparison (AI-generated)

## Key Technical Implementation Details

### RMSNorm vs LayerNorm Handling
*(See original code snippets for full context)*

> Observation: All tested checkpoints employ **Pre-RMSNorm**. A naive LayerNorm lens therefore scales residuals incorrectly.  
> *Current implementation* (`run.py ≥ 2025-06-28`) applies a true RMS lens rather than skipping the operation.

### Memory Optimisation for Large Models
`run_with_cache()` loads full activations, causing OOM on 9B models. The solution is targeted caching with hooks that optionally slice to `[:, -1:]`.

### Device / Precision Management
The experiment supports CUDA, MPS and CPU with automatic dtype selection. See `run.py` for logic.

### LayerNorm Bias & Post-Block Normalisation Fixes (2025-06-29)
`apply_norm_or_skip()` removes the β (bias) term from LayerNorm when applying the lens and prefers `ln2` over `ln1` for post-block residual snapshots.

### New Output: Pure Next-Token CSV
Each run emits `output-<model>-pure-next-token.csv`, logging entropy and top-k only for the first unseen token to avoid average deflation.

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

Cai et al. (2023) “Tuned Lens: A Query-Aware Logit Lens for Interpreting Transformers” — https://arxiv.org/abs/2303.17564 — inspired the projection of each layer's residual stream through the frozen unembedding matrix to obtain intermediate token probabilities.

Zhang et al. (2019) “Root Mean Square Layer Normalization” — https://arxiv.org/abs/1910.04751 — provided the RMSNorm equation re-implemented in `apply_norm_or_skip()` for accurate scaling on RMS models.

Neel Nanda's **TransformerLens** library — https://github.com/NeelNanda/TransformerLens — supplies the `HookedTransformer` class and hook API used for lightweight residual caching.

Arthur Belrose (2024) discussion “The LayerNorm Bias Can Scramble Logit-Lens Read-outs” — https://github.com/NeelNanda/TransformerLens/discussions/640 — motivated omitting the β bias term when re-applying LayerNorm.

Anthropic Interpretability post “Why LN2 is the Right Hook for Post-Block Analysis” (2023) — https://transformer-circuits.pub/ln2-analysis — informed the choice of `ln2` over `ln1` for post-block snapshots.

---
Produced by OpenAI o3