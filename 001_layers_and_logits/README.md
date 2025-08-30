# 001 · Layers and Logits

Layer-by-layer logit-lens analysis for causal LLMs. Computes per-layer next-token distributions from the residual stream (with architecture-aware normalization), tracks entropy, copy-collapse, and where semantics emerge.

## What It Does

- Residual-lens with proper normalization: LN/RMS with ε inside sqrt, architecture-aware γ selection (pre/post‑norm).
- Metrics per layer:
  - Entropy in bits (from full softmax)
- Copy-collapse: ID-level contiguous subsequence of prompt token IDs (k=1) with threshold + margin (no entropy fallback)
  - Semantic collapse: top‑1 equals the ground-truth token (e.g., “Berlin”)
- Outputs: compact JSON metadata and two CSVs (all positions; pure next‑token only).

## Run

All models (default device selection: `auto` — prefers `cuda` → `mps` → `cpu` based on a conservative memory‑fit estimate):

```bash
cd 001_layers_and_logits
python run.py
```

Single model to a custom directory:

```bash
cd 001_layers_and_logits
python run.py --device cpu --out_dir ./some_dir mistralai/Mistral-7B-v0.1
```

## Outputs

Run creates or rotates `run-latest/` (previous → `run-YYYYMMDD-HHMM/`). For each model:

- `output-<model>.json` — compact metadata: diagnostics (including `L_copy`, `L_semantic`, summary thresholds `first_kl_below_0.5`, `first_kl_below_1.0`, `first_rank_le_1`, `first_rank_le_5`, `first_rank_le_10`), final prediction, model stats
  - Gold-token alignment block `gold_answer`: `{ string, pieces, first_id, answer_ids, variant }`
  - `diagnostics.gold_alignment` status (`ok`/`unresolved`); `is_answer` and `p_answer/answer_rank` are based on `gold_answer.first_id`
  - Negative control: `control_prompt` (context, control gold alignment) and `control_summary` `{ first_control_margin_pos, max_control_margin }`
- `output-<model>-records.csv` — per-layer/per-position top‑k with `rest_mass` (now includes leading `prompt_id`)
- `output-<model>-pure-next-token.csv` — per-layer pure next‑token top‑k with flags and metrics: `copy_collapse`, `entropy_collapse`, `is_answer`, `p_top1`, `p_top5`, `p_answer`, `kl_to_final_bits`, `answer_rank`, `cos_to_final` (PROJECT_NOTES §1.5), and `control_margin` for control rows; both CSVs include a leading `prompt_id` column (`pos` or `ctl`).

## Scripts

- CPU-only tests (no downloads): `scripts/run_cpu_tests.sh`
- KL self-test (network + HF auth may be required):
  - Default (Mistral‑7B on CPU): `scripts/self_test.sh`
  - Custom: `scripts/self_test.sh <MODEL_ID> <DEVICE>` (e.g., `mps`)

Self-test notes: `--self-test` validates scaling and prints results; it does not write JSON/CSV artifacts or rotate `run-latest/`.

## CLI Flags

- `--device {auto|cuda|mps|cpu}` — compute device (default `auto` picks best fit)
- `--out_dir PATH` — output directory (default: `run-latest/` rotation)
- `--fp32-unembed` — cast unembedding weights to fp32
- `--keep-residuals` — save residual tensors (`*.pt`) alongside CSVs
- `--copy-threshold FLOAT` — min P(top‑1) for copy collapse (default 0.95)
- `--copy-margin FLOAT` — require P(top‑1) − P(top‑2) > margin (default 0.10)
- `--self-test` — run KL sanity test; no artifacts are written

## layers_core (Internals)

- `norm_utils` — LN/RMS normalization, ε inside sqrt, architecture detection, γ selection
- `numerics` — entropy in bits; safe casting before unembed (`force_fp32_unembed`)
- `csv_io` — writers for records and pure next‑token CSVs (stable schemas with `rest_mass`)
- `collapse_rules` — copy‑collapse (ID‑subsequence with threshold + margin; no entropy fallback) and semantic match
- `device_policy` — dtype selection (CUDA fp16; Gemma bf16; MPS fp16; CPU fp32); unembed auto‑promotion rule
- `hooks` — attach/detach residual hooks (embeddings, pos, resid_post)
- `run_dir` — `run-latest/` rotation with timestamp
- `config` — `ExperimentConfig` dataclass passed through the orchestrator

## Supported Models and Dtypes

- Supported families: Llama‑3 (8B/70B), Mistral‑7B and Mistral‑Small‑24B‑Base‑2501, Gemma‑2‑9B/27B, Qwen‑3‑8B/14B, Qwen‑2.5‑72B, Yi‑34B (raw HF format; no GGUF).
- Dtype policy: CUDA fp16 (Gemma → bf16), MPS fp16, CPU fp32.

Device selection is dynamic per model. The runner estimates memory usage (weights + overhead + headroom) for each available device and auto-picks the best fit in order `cuda → mps → cpu`. Models that do not fit on any device are skipped with a clear log.

## Self-Test Details

- In `--self-test`, uses the model’s tokenizer if available and runs KL scaling validation when the interface exposes HF‑style heads; otherwise prints a note and continues.
- Standalone alternative: `python kl_sanity_test.py <MODEL_ID> [--device cpu|mps|cuda]`

## Prompts

Reference prompt sets live as `.txt` files in this folder (single model, cross‑model, meta). They inform evaluation reports and manual probes.

## Troubleshooting

- Hugging Face auth: `huggingface-cli login` for gated models (e.g., Llama family).
- Memory: prefer `--device mps` or `--device cpu` on low‑VRAM systems; disable `--keep-residuals` to reduce memory usage.
- Artifacts: `run-latest/` rotation writes a timestamp file and three outputs per model; self‑tests do not write artifacts.
