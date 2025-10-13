# Evaluation Report: mistralai/Mistral-Small-24B-Base-2501

*Run executed on: 2025-10-11 21:50:12*

## 1. Overview

Mistral-Small-24B-Base-2501 (24B) evaluated on 2025-10-11 with a layer-by-layer logit lens pipeline. The probe measures copy vs semantics onset, KL-to-final and rank milestones, cosine/entropy trajectories, and raw-vs-norm lens diagnostics with tuned-lens/prism sidecars.

## 2. Method sanity-check

- Prompt & indexing: context prompt ends with “called simply” and no trailing space: "Give the city name only, plain text. The capital of Germany is called simply" (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:820). Positive rows exist for `prompt_id=pos`, `prompt_variant=orig` (e.g., Germany→Berlin L0 is present at 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:2).
- Normalizer provenance: `strategy` = "next_ln1" and pre-norm architecture. Early/last sources: `per_layer[0].ln_source = blocks[0].ln1` and `per_layer[40].ln_source = ln_final` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4177, 4686–4690, 4762–4766).
- Per-layer normalizer effect: normalization spike flag is set; early layers show large `resid_norm_ratio`/`delta_resid_cos` before semantics (e.g., L1 resid_norm_ratio ≈ 75.79, delta_resid_cos ≈ 0.73) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:842–851, 4236–4244).
- Unembed bias: `present=false`, `l2_norm=0.0`; cosines are bias‑free (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:829–838).
- Environment & determinism: `device=cpu`, `torch_version=2.8.0+cu128`, `deterministic_algorithms=true`, `seed=316` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:6891–6902).
- Numeric health: `any_nan=false`, `any_inf=false`, `layers_flagged=[]` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4827–4860).
- Copy mask: `size=2931` with plausible ignored tokens (control chars, whitespace, punctuation) sample (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:3858–3910).
- Gold alignment: `ok=true`, `variant=with_space`, `pieces=["ĠBerlin"]`; `gold_alignment_rate=1.0` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4811–4855, 6924–6935).
- Repeatability: skipped due to deterministic env (`status=skipped`, `reason=deterministic_env`) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4807–4810, 8671–8680).
- Norm trajectory: `shape="spike"`, `slope≈0.105`, `r2≈0.915`, `n_spikes=34` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:6509–6516, 8813–8820).
- Measurement guidance: `prefer_ranks=true`, `suppress_abs_probs=true`, `preferred_lens_for_reporting=tuned`, `use_confirmed_semantics=true` with reasons `["normalization_spike", "rank_only_near_uniform"]` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:8618–8631).
- Semantic margin: `delta_abs=0.002`, `p_uniform=7.629e-06`, `margin_ok_at_L_semantic_norm=false` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4114–4125, 8572–8586).
- Micro‑suite: aggregates present (`n=5`, `n_missing=0`); medians `L_semantic_norm=32`, `L_semantic_confirmed=33` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:6166–6200, 8721–8760).

## 3. Quantitative findings (layer‑by‑layer)

- L 0 — entropy 16.9985 bits; top‑1 token ‘Forbes’ (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:2).
- L 10 — entropy 16.8359 bits; top‑1 token ‘hétérogènes’ (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:12).
- L 20 — entropy 16.7424 bits; top‑1 token ‘luš’ (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:22).
- L 30 — entropy 16.7426 bits; top‑1 token ‘-на’ (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:32).
- L 33 — entropy 16.7740 bits; top‑1 token ‘Berlin’; answer_rank=1; cos_to_final≈0.107, cos_to_answer≈0.101; answer_logit_gap≈0.322; answer_vs_top1_gap≈0.851 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:35). This is the confirmed semantics layer.
- L 40 — entropy 3.1807 bits; top‑1 token ‘Berlin’; final‑head KL=0 (agreement) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:42; 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4839–4855).

Bold semantic layer: The run confirms L_semantic_confirmed = 33 and L_semantic_norm = 33 (from window; confirmed source=raw). Uniform‑margin gate fails at L_semantic_norm (`margin_ok=false`). Quote: "L_semantic_confirmed": 33; "confirmed_source": "raw" (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:6405–6411, 6467–6489).

- Control margin: `first_control_margin_pos=1`, `max_control_margin≈0.468` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:6948–6960).
- Micro‑suite: median L_semantic_confirmed = 33; Δ̂ median = null; e.g., Germany→Berlin row_index=33 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:8761–8812).
- Entropy drift: entropy_gap_bits percentiles p25≈13.58, p50≈13.59, p75≈13.66 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:3962–3968, 8860–8868).
- Normalizer effect snapshots: at L33, `resid_norm_ratio≈3.88`, `delta_resid_cos≈0.929` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4612–4620); final layer `resid_norm_ratio≈3.86`, `delta_resid_cos≈0.939` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4746–4766).

## 4. Qualitative findings

### 4.1. Copy vs semantics (Δ‑gap)

Copy‑reflex not detected: no `copy_collapse` and no early `copy_soft_k1@0.5` hits in layers 0–3; strict copy thresholds are null across τ∈{0.70,0.80,0.90,0.95} with stability="none" and no norm‑only flags (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:3968–4006). With no detected copy layer, Δ̂ is not defined (evaluation_pack depth_fractions.delta_hat = null) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:8652–8660).

### 4.2. Lens sanity: Raw‑vs‑Norm

Artifact risk is low: `lens_artifact_score_v2≈0.185` (tier=low). Robust metrics: `js_divergence_p50≈0.035`, `l1_prob_diff_p50≈0.347`, `first_js_le_0.1=0`, `first_l1_le_0.5=0`. Top‑K overlap: `jaccard_raw_norm_p50≈0.538`, `first_jaccard_raw_norm_ge_0.5=3`. Prevalence: `pct_layers_kl_ge_1.0≈0.024`, `n_norm_only_semantics_layers=0` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4058–4125, 4188–4220, 8648–8680). Interpretation: early semantics are unlikely to be lens‑induced; prefer rank milestones per measurement guidance.

### 4.3. Tuned‑Lens analysis

Preference: tuned lens is not calibration‑only (`tuned_is_calibration_only=false`) and is the preferred reporting lens per guidance (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:8888–8900, 8618–8631).

Attribution: ΔKL components at percentiles show rotation dominates temperature:
- `delta_kl_rot_p25/50/75 ≈ 1.62/1.76/1.87` bits, `delta_kl_temp_p25/50/75 ≈ −0.48/−0.33/−0.06`, interaction p50 ≈ 3.62 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:8679–8704).

Rank earliness: first rank‑1 occurs later under tuned (le_1: baseline=33, tuned=39; Δ=+6), with le_5/le_10 also later (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:8402–8444). Use confirmed semantics (L=33, source=raw) for onset.

Positional generalization: `pos_in_dist_le_0.92≈4.43`, `pos_ood_ge_0.96≈5.08`, `pos_ood_gap≈0.66` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:8706–8726).

Head mismatch: last‑layer agreement holds (`kl_bits_tuned_final=0`, `tau_star_modelcal=1.0`) and `diagnostics.last_layer_consistency` shows perfect calibration (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:8727–8740, 4839–4855).

### 4.4. KL, ranks, cosine, entropy milestones

KL: first KL≤1.0 and ≤0.5 both at L=40 under both lenses; final KL≈0 (healthy final‑head calibration) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4867–4880, 8429–8444).

Ranks: baseline (norm) milestones le_10=30, le_5=30, le_1=33; tuned le_10=34, le_5=35, le_1=39 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:8402–8444).

Cosine: norm‑lens thresholds reached at ge_0.2→L35, ge_0.4→L40, ge_0.6→L40 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4006–4013).

Entropy: strong drift from near‑uniform early layers; entropy_gap_bits percentiles p25/p50/p75 ≈ 13.58/13.59/13.66 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:3962–3968, 8860–8868). This aligns with early KL being large and ranks improving only after mid‑depth; final‑layer entropy is low (3.18 bits) with consistent top‑1.

Margin gate: at the rank‑1 milestone (L33), the uniform‑margin gate fails at L_semantic_norm (`margin_ok=false`) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4114–4125, 8572–8586).

### 4.5. Prism (shared‑decoder diagnostic)

Present/compatible; sampled layers [embed, 9, 19, 29]. KL deltas vs baseline are negative (KL increases under Prism): p50 baseline≈10.84 → Prism≈16.82 (Δ≈−5.98); rank milestones unchanged (null deltas) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:866–906). Verdict: Regressive.

### 4.6. Ablation & stress tests

Stylistic ablation (`no_filler`): `L_sem_orig=33`, `L_sem_nf=31`, ΔL_sem=−2 (5% of 40), indicating mild stylistic sensitivity (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:6902–6924).

Negative/control prompts: control_summary present with `first_control_margin_pos=1`; `max_control_margin≈0.468` indicates the France/Paris control is separated from the target (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:6948–6960).

Important‑word trajectory (records CSV):
- At L33, prompt token ‘Germany’ co‑activates ‘Berlin’ among top predictions for that position (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-records.csv:635).
- At L33, the next‑token position has top‑1 ‘Berlin’ (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-records.csv:638).

### 4.7. Checklist (✓/✗/n.a.)

- RMS lens ✓
- LayerNorm bias removed ✓
- FP32 unembed promoted ✓ (analysis path)
- Punctuation / markup anchoring noted ✓ (copy mask sample includes punctuation)
- Copy‑reflex ✗ (no early copy flags)
- Preferred lens honored ✓ (tuned; confirmed semantics used)
- Confirmed semantics reported ✓ (L=33, source=raw)
- Dual‑lens artefact metrics (incl. v2, JS/Jaccard/L1) cited ✓
- Tuned‑lens audit done (rotation/temp/positional/head) ✓
- normalization_provenance present (ln_source @ L0/final) ✓
- per‑layer normalizer effect present (resid_norm_ratio, delta_resid_cos) ✓
- deterministic_algorithms true ✓
- numeric_health clean ✓
- copy_mask plausible ✓
- milestones.csv or evaluation_pack.citations used for quotes ✓ (e.g., 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-milestones.csv:3–4)

---
Produced by OpenAI GPT-5 
