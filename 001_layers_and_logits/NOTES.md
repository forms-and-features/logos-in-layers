# Layer-by-Layer Analysis (Experiment 001)

## Supported Models

### ✅ Confirmed Working
- **Llama 3** (Meta) — 8B, 70B
- **Mistral** (Mistral AI) — 7B; Small‑24B‑Base‑2501  
- **Gemma 2** (Google) — 9B, 27B
- **Qwen** (Alibaba) — Qwen3 8B/14B; Qwen2.5 72B
- **Yi** (01.AI) — 34B

### ❌ Not Supported
- **GGUF files** – Require raw transformer format
- **Extremely large models** – Hardware constraints

## Directory Layout

```
001_layers_and_logits/
├── run.py                           # Main experiment script
├── run-latest/                      # Results and evaluation of the latest run
|   ├── evaluation-meta.md           # Meta-evaluation of the latest run
│   ├── evaluation-cross-models.md   # Cross-model analysis
│   ├── evaluation-[model].md        # Per-model analyses  
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

## Updates and Key Technical Implementation Details

### Deterministic Seed & Reproducibility (2025-06-29)
`run.py` now initialises a deterministic bootstrap (`SEED = 316`) and enables `torch.use_deterministic_algorithms(True)` plus cuBLAS workspace settings.  Every sweep is repeatable bit-for-bit across runs and machines.

### Improved LayerNorm & RMSNorm Handling (2025-06-29)
`apply_norm_or_skip()` now:
* Applies full **LayerNorm** (γ *and* β) with dtype-safe casting; computes LN/RMS statistics in fp32, then casts back to the residual dtype.
* Detects **RMSNorm** scale under multiple attribute names and applies ε inside √; computes in fp32 then casts back.
* Eliminates the deprecated helper `is_safe_layernorm()`.

### New Output: Pure Next-Token CSV
Each run emits `output-<model>-pure-next-token.csv`, logging entropy and top-k only for the first unseen token to avoid average deflation.

### LayerNorm Bias & Post-Block Normalisation Fixes (2025-06-29)
`apply_norm_or_skip()` keeps the β (bias) term for LayerNorm, and prefers `ln2` over `ln1` for post-block residual snapshots (or the next block’s `ln1` for pre-norm architectures; `ln_final` at the last layer).

### RMSNorm vs LayerNorm Handling
*(See original code snippets for full context)*

> Observation: All tested checkpoints employ **Pre-RMSNorm**. A naive LayerNorm lens therefore scales residuals incorrectly.  
> *Current implementation* (`run.py ≥ 2025-06-28`) applies a true RMS lens rather than skipping the operation.

### Memory Optimisation for Large Models
`run_with_cache()` loads full activations, causing OOM on 9B models. The solution is targeted caching with hooks that optionally slice to `[:, -1:]`.

### Device / Precision Management
The experiment supports CUDA, MPS and CPU with automatic dtype selection. Device selection is dynamic per model: a conservative memory‑fit estimator chooses `cuda → mps → cpu` when possible; otherwise the model is skipped. On CPU, models ≤27B use fp32; ≥30B use bf16. When compute runs in bf16/fp16, the unembedding matrix is auto‑promoted to fp32 and logits are decoded in fp32; LN/RMS statistics are computed in fp32 and cast back. See `run.py` and `layers_core/device_policy.py` for logic.

## Sub‑word‑aware copy detection (2025-08-25)
Copy‑collapse is now detected at the token‑ID level via a contiguous subsequence match against the prompt using a rolling window `k=1`. Defaults tightened to `copy_threshold=0.95` and `copy_margin=0.10`; no entropy fallback inside the copy rule (entropy collapse is tracked separately). Trivial whitespace/punctuation echoes are ignored. Provenance fields (`copy_thresh`, `copy_window_k`, `copy_match_level`) are included in diagnostics.

## Per‑layer probability and KL metrics (2025-08-26)
Added the following per‑layer pure next‑token metrics (PROJECT_NOTES §1.3):

- CSV columns: `p_top1`, `p_top5` (cumulative), `p_answer`, `kl_to_final_bits`, `answer_rank`.
- JSON diagnostics: `first_kl_below_0.5`, `first_kl_below_1.0`, `first_rank_le_1`, `first_rank_le_5`, `first_rank_le_10`.
- KL details: KL(P_layer || P_final) in bits, computed via `layers_core.numerics.kl_bits` in fp32 with epsilon guards.
- Implementation detail: metric computation factored as `layers_core.metrics.compute_next_token_metrics` and used by `run.py`.

## Development Environment Notes
- **Hardware**: Apple Silicon MacBook Pro M2 Max 64 GB
- **Library stack**: TransformerLens, no quantisation, raw checkpoint format

## Experiment Structure & Toggles
See `run.py` for `USE_NORM_LENS` and residual-cache device options. The unembedding promotion to fp32 is now automatic when compute dtype is bf16/fp16; the manual `--fp32-unembed` flag remains available but is typically unnecessary.

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
Produced by OpenAI GPT-5, OpenAI o3
