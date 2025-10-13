# Evaluation Report: google/gemma-2-27b

*Run executed on: 2025-10-12 20:56:18*

## Overview
- Model: `google/gemma-2-27b`; layers analyzed: 46 (norm lens; raw lens sidecars). The probe traces copy-reflex vs semantic emergence using ranks/KL/cosine/entropy and audits lens artifacts and calibration.
- Run window: timestamp `timestamp-20251012-2056`; outputs in `001_layers_baseline/run-latest/`. Includes tuned‑lens, raw‑vs‑norm, prism, milestones, and micro‑suite sidecars.

## Method sanity-check
- Prompt & indexing
  - "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply"  (no trailing space)  `001_layers_baseline/run-latest/output-gemma-2-27b.json:819`
  - Rows with `prompt_id=pos`,`prompt_variant=orig` present, e.g., `Germany→Berlin,0,pos,orig,0,…`  `001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:1`
- Normalizer provenance
  - "strategy": "next_ln1"; arch="pre_norm"  `001_layers_baseline/run-latest/output-gemma-2-27b.json:5945`
  - First/last ln sources: "blocks[0].ln1" … "ln_final"  `001_layers_baseline/run-latest/output-gemma-2-27b.json:5950` / `001_layers_baseline/run-latest/output-gemma-2-27b.json:6364`
- Per‑layer normalizer effect
  - Early spike flag present: "normalization_spike": true; norm trajectory shape="spike"  `001_layers_baseline/run-latest/output-gemma-2-27b.json:842` / `001_layers_baseline/run-latest/output-gemma-2-27b.json:8478`
- Unembed bias
  - "unembed_bias": { "present": false, "l2_norm": 0.0 } — cosine metrics are bias‑free  `001_layers_baseline/run-latest/output-gemma-2-27b.json:835`
- Environment & determinism
  - "device": "cpu", "deterministic_algorithms": true, "seed": 316  `001_layers_baseline/run-latest/output-gemma-2-27b.json:8864`
- Numeric health
  - any_nan=false, any_inf=false, layers_flagged=[]  `001_layers_baseline/run-latest/output-gemma-2-27b.json:6658`
- Copy mask
  - size=4668; sample: "\n", "\n\n", … (whitespace runs)  `001_layers_baseline/run-latest/output-gemma-2-27b.json:5648`
- Gold alignment
  - { ok: true, variant: "with_space", pieces: ["▁Berlin"], first_id: 12514 }  `001_layers_baseline/run-latest/output-gemma-2-27b.json:6676`
- Repeatability (skipped)
  - { status: "skipped", reason: "deterministic_env" }  `001_layers_baseline/run-latest/output-gemma-2-27b.json:6664`
- Norm trajectory
  - { shape: "spike", slope≈0.088, r2≈0.846, n_spikes=16 }  `001_layers_baseline/run-latest/output-gemma-2-27b.json:8478`
- Measurement guidance
  - { prefer_ranks: true, suppress_abs_probs: true, preferred_lens_for_reporting: "norm", use_confirmed_semantics: true }  `001_layers_baseline/run-latest/output-gemma-2-27b.json:10716`
- Semantic margin
  - { delta_abs: 0.002, p_uniform: 3.9e‑06, margin_ok_at_L_semantic_norm: true }  `001_layers_baseline/run-latest/output-gemma-2-27b.json:10520`
- Micro‑suite
  - aggregates present: { n: 5, L_semantic_confirmed_median: 46, delta_hat_median: 1.0, n_missing: 0 }  `001_layers_baseline/run-latest/output-gemma-2-27b.json:10888`

## Quantitative findings (layer‑by‑layer)
- L 0 — entropy 0.00050 bits; top‑1 'simply' (copy of prompt word)  [row 1 in CSV]  `001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:1`
- L 3 — entropy 0.8857 bits; top‑1 'simply' (copy persists)  [row 4 in CSV]  `001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:4`
- L 4 — entropy 0.6183 bits; top‑1 'simply' (copy persists)  [row 5 in CSV]  `001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:5`
- L 10–40 — mixed non‑semantic tokens; rising cosine to final (see cos milestones below)
- **L 46 — entropy 0.1180 bits; top‑1 'Berlin' (answer_rank=1; confirmed)**  [row 48 in CSV]  `001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48`
- Control margin: first_control_margin_pos=0; max_control_margin reported (calibration caveat)  `001_layers_baseline/run-latest/output-gemma-2-27b.json:8916`
- Micro‑suite: median L_semantic_confirmed=46; median Δ̂=1.0; e.g., "Germany→Berlin" cites row_index=46  `001_layers_baseline/run-latest/output-gemma-2-27b.json:10650`
- Entropy drift: entropy_gap_bits p50≈4.68 bits across depths  `001_layers_baseline/run-latest/output-gemma-2-27b.json:10810`
- Normalizer snapshot: early resid_norm_ratio≈0.79, delta_resid_cos≈0.57 at L0 (spiky trajectory flagged)  `001_layers_baseline/run-latest/output-gemma-2-27b.json:5953`

## Qualitative findings

### Copy vs semantics (Δ‑gap)
Early layers show a clear copy‑reflex: strict copy triggers at layer 0 across τ∈{0.70,0.80,0.90,0.95} in this prompt (see copy flags in the CSV row), with soft k1 copy at layer 0 as well. Confirmed semantics arrives at the final layer. Depth fractions report Δ̂=1.0, indicating semantics only at the end. The copy‑thresholds stability tag is "mixed" and norm_only_flags are false across τ in summary. Quotes: "L_copy_strict": {"0.7": 0, "0.8": 0, "0.9": 0, "0.95": 0}  `001_layers_baseline/run-latest/output-gemma-2-27b.json:5719`; depth fractions { semantic_frac: 1.0, L_copy_strict_frac: 0.0 }  `001_layers_baseline/run-latest/output-gemma-2-27b.json:5760`.

### Lens sanity: Raw‑vs‑Norm
Raw vs norm shows high artifact risk: lens_artifact_score_v2=1.0 (tier=high) with large divergence between raw and norm (js_divergence_p50≈0.87; l1_prob_diff_p50≈1.89). Top‑K overlap is moderate (jaccard_p50=0.56; first_jaccard≥0.5 at layer 1). Most layers have KL(norm||raw) ≥1.0 (≈97.9%), and there is a norm‑only semantics layer at 46. Quotes: artifact block with risk_tier="high"  `001_layers_baseline/run-latest/output-gemma-2-27b.json:10756`. Given this, prefer rank milestones and confirmed semantics, and treat absolute probabilities as unreliable (measurement_guidance says suppress_abs_probs=true).

### Tuned‑Lens analysis
Tuned lens is marked calibration‑only and does not shift rank milestones (le_10/le_5/le_1 all 46 baseline vs tuned). KL attribution indicates most improvement comes from temperature, not rotation (ΔKL_temp_p50≈0.53; ΔKL_rot_p50≈−0.03; small interaction). Positional OOD gap is modest (≈0.084). Final‑layer head mismatch shows KL≈1.13 bits dropping to ≈0.55 after τ⋆ (≈3.01). Quotes: audit_summary rotation/temperature and head_mismatch  `001_layers_baseline/run-latest/output-gemma-2-27b.json:10553` and milestones deltas  `001_layers_baseline/run-latest/output-gemma-2-27b.json:10496`. Last‑layer consistency warns: kl_to_final_bits≈1.14; kl_after_temp_bits≈0.57; top1_agree=true  `001_layers_baseline/run-latest/output-gemma-2-27b.json:6679`.

### KL, ranks, cosine, entropy milestones
- KL: first_kl_le_{1.0,0.5} are null (baseline and tuned) at the granularity reported  `001_layers_baseline/run-latest/output-gemma-2-27b.json:10512`.
- Ranks: first_rank_le_{10,5,1} = 46 (preferred lens=norm; tuned unchanged)  `001_layers_baseline/run-latest/output-gemma-2-27b.json:10500`.
- Cosine: ge_{0.2}=1, ge_{0.4}=46, ge_{0.6}=46 (norm)  `001_layers_baseline/run-latest/output-gemma-2-27b.json:5751`.
- Entropy: rising entropy gap vs teacher (p50≈4.68 bits); late calibration aligns with confirmed semantics at L=46 while early layers reflect surface copying rather than meaning  `001_layers_baseline/run-latest/output-gemma-2-27b.json:10810`.
- Margin gate: margin_ok_at_L_semantic_norm=true  `001_layers_baseline/run-latest/output-gemma-2-27b.json:10520`.

### Prism (shared‑decoder diagnostic)
Prism is present/compatible. It reduces KL by ≈23.7 bits at p50 but does not advance rank milestones (all remain 46). Quotes: prism kl_bits_at_percentiles delta p50≈23.73; rank_milestones unchanged  `001_layers_baseline/run-latest/output-gemma-2-27b.json:888`.
Verdict: Neutral — strong KL drop (calibration) without earlier ranks.

### Ablation & stress tests
- Filler ablation: L_copy_orig=0 → L_copy_nf=3 (ΔL_copy=+3); L_sem unchanged at 46  `001_layers_baseline/run-latest/output-gemma-2-27b.json:8880`.
- Control prompt: first_control_margin_pos=0; first_control_strong_pos=46 (magnitude reported; interpret comparatively per guidance)  `001_layers_baseline/run-latest/output-gemma-2-27b.json:8916`.
- Important words: prompt token "Germany" trajectory cited (records CSV; pos=13 at L0)  `001_layers_baseline/run-latest/output-gemma-2-27b-records.csv:13`; predicted token trajectory shows early 'simply' (L0–L4) to 'Berlin' at L46 (pure CSV rows 1,4,5,48).

### Checklist
- RMS lens ✓  (RMSNorm detected; next_ln1 strategy)
- LayerNorm bias removed ✓  (present=false)
- FP32 unembed promoted ✓  ("mixed_precision_fix": "casting_to_fp32_before_unembed")
- Punctuation / markup anchoring noted ✓  (copy_mask whitespace sample)
- Copy‑reflex ✓  (strict/soft copy at L0)
- Preferred lens honored ✓  (norm; ranks reported)
- Confirmed semantics reported ✓  (L=46; tuned‑confirmed; margin gate ok)
- Dual‑lens artifact metrics (incl. v2, JS/Jaccard/L1) cited ✓
- Tuned‑lens audit (rotation/temp/positional/head) ✓
- normalization_provenance present (ln_source @ L0/final) ✓
- per‑layer normalizer effect present (resid_norm_ratio, delta_resid_cos) ✓
- deterministic_algorithms true (caution unnecessary) ✓
- numeric_health clean ✓
- copy_mask plausible ✓
- evaluation_pack.citations used for quotes ✓

---
Produced by OpenAI GPT-5
