# Evaluation Report: Qwen/Qwen2.5-72B

*Run executed on: 2025-10-05 18:16:50*

## 1. Overview
Qwen/Qwen2.5-72B was probed on 2025-10-05 to trace when surface copy transitions into semantic prediction under a normalization-aware logit lens. The probe reports rank/KL/cosine/entropy trajectories layer-by-layer and audits lens artefacts (raw vs norm), normalization provenance/effects, and final-head calibration.

## 2. Method sanity-check
- Prompt & indexing: context ends with “called simply” (no trailing space):
  "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply"  [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:802].
  Gold-alignment confirms rows for `prompt_id=pos`, `prompt_variant=orig` exist [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8869].
- Normalizer provenance: `strategy = "next_ln1"` (pre-norm); L0 uses `blocks[0].ln1`, final uses `ln_final` [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7592, 8258].
- Per-layer normalizer effect: normalization spike flagged; early large ratios present before semantics, e.g. L2 `resid_norm_ratio=8.939`, `delta_resid_cos=0.577` [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7606].
- Unembed bias: `present=false`, `l2_norm=0.0`; cosines are bias-free [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:827].
- Environment & determinism: `device=cpu`, torch `2.8.0+cu128`, `deterministic_algorithms=true`, `seed=316` [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:915].
- Numeric health: `any_nan=false`, `any_inf=false`, `layers_flagged=[]` [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8811].
- Copy mask: size `6244`; sample of ignored strings: `! " # $ % & ' ( ) * + , - . / :` [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7229].
- Gold alignment: ok; pieces = ["ĠBerlin"], `gold_alignment_rate=1.0` [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8822, 8866].
- Repeatability (1.39): skipped in deterministic env [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8818].
- Norm trajectory: `shape="spike"`, slope `0.0642`, `r2=0.923`, `n_spikes=55` [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8960, 9198].
- Measurement guidance: `prefer_ranks=true`, `suppress_abs_probs=true`, preferred lens `"norm"`, `use_confirmed_semantics=false` [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9176].

## 3. Quantitative findings (layer‑by‑layer)
Positive prompt only (`prompt_id=pos`, `prompt_variant=orig`). Entropy in bits; top‑1 token name only (no probabilities).

| Layer | Entropy (bits) | Top‑1 token |
|---|---|---|
| L 0 | 17.214 | s [output-Qwen2.5-72B-pure-next-token.csv:1] |
| L 20 | 16.129 | 有期徒 [output-Qwen2.5-72B-pure-next-token.csv:22] |
| L 40 | 16.192 | iéndo [output-Qwen2.5-72B-pure-next-token.csv:98] |
| L 60 | 16.886 | ' [output-Qwen2.5-72B-pure-next-token.csv:118] |
| L 80 | 4.116 | **Berlin** [output-Qwen2.5-72B-pure-next-token.csv:138] |

- Semantic layer: L_semantic_norm = 80; confirmed semantics: none (`confirmed_source: "none"`) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8894].
- Control margin: `first_control_margin_pos=0`, `max_control_margin=0.2070` [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9201].
- Entropy drift: gap vs teacher entropy (bits) p25=12.03, p50=12.50, p75=12.77 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9324].
- Normalizer effect snapshots: L0 `(resid_norm_ratio=0.716, Δcos=0.475)`; L40 `(0.409, 0.747)`; L80 `(0.172, 0.929)` [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7600, 8207, 8247].

## 4. Qualitative findings

### 4.1. Copy vs semantics (Δ‑gap)
No copy‑reflex detected: strict copy milestones are null across τ∈{0.70,0.80,0.90,0.95}, and soft copy is also null; stability tag `"none"` [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7315]. Earliest strict copy at τ=0.70 and τ=0.95 are both null; `norm_only_flags[τ]` are null [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7323]. With no L_copy, Δ̂ is n.a.; semantic onset occurs at the final layer (fraction 1.0) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7360, 9218].

### 4.2. Lens sanity: Raw‑vs‑Norm
Lens artefact risk is high: legacy `0.593`, v2 `0.743` [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7580, 9210]. Symmetric metrics indicate large raw/norm divergence: `js_divergence_p50=0.105`, `l1_prob_diff_p50=0.615`; earliest layers with JS≤0.1 or L1≤0.5 do not occur (both `0`) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7578, 9214]. Top‑K overlap is modest (`jaccard_raw_norm_p50=0.316`) with no layer ≥0.5 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7579, 9216]. Prevalence: `pct_layers_kl_ge_1.0=0.321`; a norm‑only semantics layer exists at L=80 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7568, 9218]. Caution: prefer ranks and confirmed semantics; avoid absolute probabilities near semantics.

### 4.3. Tuned‑Lens analysis
Tuned lens is missing (`status="missing"`), so semantics are reported under the norm lens by default [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8914, 9171]. No rotation/temperature/positional/head audits available.

### 4.4. KL, ranks, cosine, entropy milestones
- KL thresholds (norm lens): `first_kl_below_1.0=80`, `first_kl_below_0.5=80` [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7256]. Final‑head calibration is excellent: `kl_to_final_bits≈0` and `top1_agree=true` [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8834].
- Ranks (preferred lens=norm): `first_rank_le_10=74`, `le_5=78`, `le_1=80` [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7260, 9192].
- Cosine milestones (norm): ge_0.2=0, ge_0.4=0, ge_0.6=53 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7353].
- Entropy: large positive drift vs teacher entropy throughout most of the depth (gap bits p50≈12.50), converging by the final layer [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9324].

### 4.5. Prism (shared‑decoder diagnostic)
Prism present and compatible [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:846]. KL drops meaningfully at p25/p50 (Δ≈3.16/2.83 bits) but slightly regresses at p75 (−0.54) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:839]. Rank milestones under Prism remain unset (no earlier ranks) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:829]. Verdict: Neutral — KL improvement without an earlier `first_rank_le_1`.

### 4.6. Ablation & stress tests
- Style ablation: `L_copy_orig=null`, `L_sem_orig=80`; `L_copy_nf=null`, `L_sem_nf=80`; `ΔL_sem=0` → no stylistic sensitivity observed [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9198].
- Control prompt: French control uses “France… called simply”; control summary `first_control_margin_pos=0`, `max_control_margin=0.207` [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9201].
- Important‑word trajectory (records CSV): “Berlin” rises in the last ~10 layers and is top‑1 at L=80 (pos=15) [001_layers_baseline/run-latest/output-Qwen2.5-72B-records.csv:4106]; it is already in the top predictions by L=74–79 [001_layers_baseline/run-latest/output-Qwen2.5-72B-records.csv:3911, 4071]. “Germany” appears in the context token positions across early layers (e.g., [001_layers_baseline/run-latest/output-Qwen2.5-72B-records.csv:15, 52]).

### 4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓
- LayerNorm bias removed ✓
- FP32 unembed promoted ✓
- Punctuation/markup anchoring noted ✓
- Copy‑reflex ✗
- Preferred lens honored ✓
- Confirmed semantics reported ✗ (none available)
- Dual‑lens artefact metrics (incl. v2, JS/Jaccard/L1) cited ✓
- Tuned‑lens audit done n.a. (missing)
- normalization_provenance present (ln_source @ L0/final) ✓
- per‑layer normalizer effect present (resid_norm_ratio, delta_resid_cos) ✓
- deterministic_algorithms true ✓
- numeric_health clean ✓
- copy_mask plausible ✓
- milestones.csv / evaluation_pack.citations used ✓

---
Produced by OpenAI GPT-5 
