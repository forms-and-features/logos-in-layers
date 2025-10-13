# Evaluation Report: google/gemma-2-9b

*Run executed on: 2025-10-12 20:56:18*

## Overview
This evaluation reviews google/gemma-2-9b on a capitals fact probe run on 2025‑10‑12. The probe measures copy vs semantic emergence using rank/KL/cosine/entropy trajectories, and audits lens behaviour (raw vs norm, tuned‑lens, and prism) with structural normalization diagnostics.

## Method sanity‑check
- Prompt & indexing: context ends with “called simply” (no trailing space): "Give the city name only, plain text. The capital of Germany is called simply" (001_layers_baseline/run-latest/output-gemma-2-9b.json:4). Positive rows exist for `prompt_id=pos`, `prompt_variant=orig` (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2).
- Normalizer provenance: `arch="pre_norm"`, `strategy="next_ln1"`; early `ln_source` is `blocks[0].ln1` and final uses `ln_final` (001_layers_baseline/run-latest/output-gemma-2-9b.json:5932,6581–6596).
- Per‑layer normalizer effect: normalization trajectory flagged as spike; `norm_trajectory.shape="spike"`, `n_spikes=1` (001_layers_baseline/run-latest/output-gemma-2-9b.json:10698–10706). Measurement flags include `normalization_spike=true` (001_layers_baseline/run-latest/output-gemma-2-9b.json:852–858).
- Unembed bias: `present=false`, `l2_norm=0.0`; cosines are bias‑free (001_layers_baseline/run-latest/output-gemma-2-9b.json:834–846).
- Environment & determinism: `device="cpu"`, `torch=2.8.0+cu128`, `deterministic_algorithms=true`, `seed=316` (001_layers_baseline/run-latest/output-gemma-2-9b.json:8787–8801). Reproducibility OK.
- Numeric health: `any_nan=false`, `any_inf=false`, `layers_flagged=[]` (001_layers_baseline/run-latest/output-gemma-2-9b.json:6620–6630).
- Copy mask: size `4668`; sample shows newline runs (e.g., "\n\n…") (001_layers_baseline/run-latest/output-gemma-2-9b.json:5608–5660). Plausible for tokenizer.
- Gold alignment: `ok=true`, variant `with_space`, `gold_alignment_rate=1.0` (001_layers_baseline/run-latest/output-gemma-2-9b.json:6608–6660, 6560–6580).
- Repeatability (1.39): skipped under deterministic env; `{max_rank_dev,p95_rank_dev,top1_flip_rate}=null` (001_layers_baseline/run-latest/output-gemma-2-9b.json:10708–10718).
- Norm trajectory: `shape="spike"`, `slope≈0.052`, `r2≈0.987`, `n_spikes=1` (001_layers_baseline/run-latest/output-gemma-2-9b.json:10698–10706).
- Measurement guidance: `prefer_ranks=true`, `suppress_abs_probs=true`, `preferred_lens_for_reporting="norm"`, `use_confirmed_semantics=true` (001_layers_baseline/run-latest/output-gemma-2-9b.json:10640–10660).
- Semantic margin: `delta_abs=0.002`, `margin_ok_at_L_semantic_norm=true`, `L_semantic_confirmed_margin_ok_norm=42` (001_layers_baseline/run-latest/output-gemma-2-9b.json:5766–5773).
- Micro‑suite: present; aggregates show `n=5`, `n_missing=0` (001_layers_baseline/run-latest/output-gemma-2-9b.json:10832–10844).

## Quantitative findings (layer‑by‑layer)
- L 0 — entropy 1.672e‑05 bits, top‑1 ‘simply’ (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2)
- L 10 — entropy 0.2814 bits, top‑1 ‘simply’ (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:12)
- L 21 — entropy 1.8665 bits, top‑1 ‘simply’ (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:28)
- L 32 — entropy 0.0625 bits, top‑1 is punctuation (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:39)
- L 42 — entropy 0.3701 bits, top‑1 ‘Berlin’ (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:49)

Bold semantic layer: L 42 is the confirmed semantic layer (preferred lens: norm; confirmed by tuned), meeting the uniform‑margin gate.

- Control margin: `first_control_margin_pos=18`, `max_control_margin≈0.868` (001_layers_baseline/run-latest/output-gemma-2-9b.json:8888–8900).
- Micro‑suite: median L_semantic_confirmed = 42 and median Δ̂ = 1.0 across 5 facts; e.g., France→Paris reaches L=42 (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:148).
- Entropy drift: summary gaps p25/p50/p75 ≈ −2.89/−2.80/−1.63 bits (001_layers_baseline/run-latest/output-gemma-2-9b.json:10712–10720).
- Confidence margin: at L 42, `answer_logit_gap≈2.59` (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:49). Normalizer snapshot at L 42: `resid_norm_ratio≈0.220`, `delta_resid_cos≈0.745` (same row).

## Qualitative findings

### 4.1 Copy vs semantics (Δ‑gap)
Copy‑reflex ✓. Early layers show strict copy in the pure lens at L=0 (and by τ∈{0.70,0.95}), while semantics is only achieved at L=42. Evaluation milestones give `L_copy_strict=0`, `L_copy_soft(k=1)=0`, and `L_semantic_norm=42`, with Δ̂=1.0 (001_layers_baseline/run-latest/output-gemma-2-9b.json:10598–10620). Copy threshold stability is “mixed”; earliest strict copy at τ=0.70 and τ=0.95 is at L=0; `norm_only_flags[τ]=null` (001_layers_baseline/run-latest/output-gemma-2-9b.json:5710–5736).

### 4.2 Lens sanity: Raw‑vs‑Norm
Artifact risk is high: `lens_artifact_score=0.58`, `lens_artifact_score_v2=0.59`, tier=high (001_layers_baseline/run-latest/output-gemma-2-9b.json:5927–5934, 10680–10690). Symmetric metrics: `js_divergence_p50≈0.0063`, `l1_prob_diff_p50≈0.029`, with earliest `first_js_le_0.1=0`, `first_l1_le_0.5=0` (001_layers_baseline/run-latest/output-gemma-2-9b.json:5836–5840, 10680–10700). Top‑K overlap: `jaccard_raw_norm_p50≈0.639`, `first_jaccard_raw_norm_ge_0.5=3` (001_layers_baseline/run-latest/output-gemma-2-9b.json:5836–5840, 10680–10700). Prevalence: `pct_layers_kl_ge_1.0≈0.302`, `n_norm_only_semantics_layers=1`, `earliest_norm_only_semantic=42` (001_layers_baseline/run-latest/output-gemma-2-9b.json:5828–5840, 10680–10700). Caution: because risk tier is high and a norm‑only semantic layer appears at 42 in the window/full audits (001_layers_baseline/run-latest/output-gemma-2-9b.json:5785–5815), prefer rank milestones and confirmed semantics for onset claims.

### 4.3 Tuned‑Lens analysis
Preference: tuned lens is not calibration‑only (`tuned_is_calibration_only=false`), but guidance prefers reporting semantics under the norm lens (001_layers_baseline/run-latest/output-gemma-2-9b.json:10640–10660, 10736–10750). Attribution: ΔKL (tuned vs baseline decomposition) at percentiles shows `delta_kl_rot_p50≈0.00005`, `delta_kl_temp_p50≈−0.013`, small interaction `≈0.0026` (001_layers_baseline/run-latest/output-gemma-2-9b.json:10720–10738). Rank earliness: no earlier rank milestones; `first_rank_le_{10,5,1}=42` unchanged (001_layers_baseline/run-latest/output-gemma-2-9b.json:8865–8867). Positional generalization: `pos_ood_ge_0.96=0.0`, `pos_in_dist_le_0.92≈0.00028` (001_layers_baseline/run-latest/output-gemma-2-9b.json:10738–10758). Head mismatch: `tau_star_modelcal≈2.85` drops tuned final KL from ≈1.08 bits to ≈0.41 bits (001_layers_baseline/run-latest/output-gemma-2-9b.json:10758–10768). Last‑layer agreement: `top1_agree=true`, but `kl_to_final_bits≈1.01` and `warn_high_last_layer_kl=true` — focus on ranks (001_layers_baseline/run-latest/output-gemma-2-9b.json:6608–6630).

### 4.4 KL, ranks, cosine, entropy milestones
- KL: `first_kl_below_1.0=null`, `first_kl_below_0.5=null` under norm, and final KL is not ~0 (001_layers_baseline/run-latest/output-gemma-2-9b.json:5650–5651, 6608–6614). This reflects head calibration differences; use ranks/KL thresholds.
- Ranks (preferred lens: norm): `first_rank_le_{10,5,1}=42` (001_layers_baseline/run-latest/output-gemma-2-9b.json:5652–5654). Uniform‑margin gate passes at this layer (001_layers_baseline/run-latest/output-gemma-2-9b.json:5766–5773).
- Cosine: milestones under norm are ge_0.2 at L=1, ge_0.4 at L=42, ge_0.6 at L=42 (001_layers_baseline/run-latest/output-gemma-2-9b.json:5738–5756).
- Entropy: entropy gaps are negative throughout much of the depth (p50 ≈ −2.80 bits), with drift consistent with increasing calibration late (001_layers_baseline/run-latest/output-gemma-2-9b.json:10712–10720). This aligns with late KL/rank consolidation.

### 4.5 Prism (shared‑decoder diagnostic)
Prism is present/compatible (k=512; layers embed/9/20/30) (001_layers_baseline/run-latest/output-gemma-2-9b.json:858–874). KL deltas vs baseline are worse at p25/p50/p75 (≈ −11.63/−10.33/−24.42; baseline minus prism), and rank milestones do not improve (null deltas) (001_layers_baseline/run-latest/output-gemma-2-9b.json:874–906). Verdict: Regressive.

### 4.6 Ablation & stress tests
Ablation summary: `L_copy_orig=0`, `L_sem_orig=42`, `L_copy_nf=0`, `L_sem_nf=42`, so ΔL_sem=0 (robust to no‑filler) (001_layers_baseline/run-latest/output-gemma-2-9b.json:8810–8830). Control prompts: control summary shows early control margin at L=18 and strong control by L=42 (001_layers_baseline/run-latest/output-gemma-2-9b.json:8888–8900). Negative prompt “Berlin is the capital of” yields country tokens at top‑1; “Berlin” appears only among lower ranks (001_layers_baseline/run-latest/output-gemma-2-9b.json:12, 40–56). Important‑word trajectory: final layer distribution features ‘Berlin’ at top‑1 (records row) (001_layers_baseline/run-latest/output-gemma-2-9b-records.csv:893).

### 4.7 Checklist (✓/✗/n.a.)
- RMS lens ✓ (RMSNorm detected) (001_layers_baseline/run-latest/output-gemma-2-9b.json:812–815)
- LayerNorm bias removed ✓ (`layernorm_bias_fix=not_needed_rms_model`) (001_layers_baseline/run-latest/output-gemma-2-9b.json:814)
- FP32 unembed promoted ✓ (`mixed_precision_fix=casting_to_fp32_before_unembed`, `unembed_dtype=torch.float32`) (001_layers_baseline/run-latest/output-gemma-2-9b.json:811,817)
- Punctuation/markup anchoring noted ✓ (late layers show quotes/punctuation before answer) (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:39)
- Copy‑reflex ✓ (early strict copy) (001_layers_baseline/run-latest/output-gemma-2-9b.json:10598–10620)
- Preferred lens honored ✓ (norm) (001_layers_baseline/run-latest/output-gemma-2-9b.json:10640–10660)
- Confirmed semantics reported ✓ (L=42; source tuned) (001_layers_baseline/run-latest/output-gemma-2-9b.json:8297–8304)
- Dual‑lens artifact metrics (incl. v2, JS/Jaccard/L1) cited ✓ (001_layers_baseline/run-latest/output-gemma-2-9b.json:5828–5840, 10680–10700)
- Tuned‑lens audit (rotation/temp/positional/head) ✓ (001_layers_baseline/run-latest/output-gemma-2-9b.json:10720–10768)
- normalization_provenance present (ln_source @ L0/final) ✓ (001_layers_baseline/run-latest/output-gemma-2-9b.json:5932, 6581–6596)
- per‑layer normalizer effect present (resid_norm_ratio, delta_resid_cos) ✓ (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:49)
- deterministic_algorithms true ✓ (001_layers_baseline/run-latest/output-gemma-2-9b.json:8787–8801)
- numeric_health clean ✓ (001_layers_baseline/run-latest/output-gemma-2-9b.json:6620–6630)
- copy_mask plausible ✓ (001_layers_baseline/run-latest/output-gemma-2-9b.json:5608–5660)
- milestones.csv or evaluation_pack.citations used for quotes ✓ (001_layers_baseline/run-latest/output-gemma-2-9b.json:10598–10620)

---
Produced by OpenAI GPT-5
