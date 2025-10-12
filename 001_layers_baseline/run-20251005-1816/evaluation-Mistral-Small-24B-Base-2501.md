# Evaluation Report: mistralai/Mistral-Small-24B-Base-2501

*Run executed on: 2025-10-05 18:16:50*

## 1. Overview
This EVAL reviews Mistral-Small-24B-Base-2501 (40 layers) from the latest run snapshot. The probe measures copy-reflex vs. semantic onset and tracks KL-to-final, rank milestones, cosine, and entropy trajectories, with raw-vs-norm lens diagnostics and a tuned-lens audit.

## 2. Method sanity-check
- Prompt & indexing: context prompt ends with “called simply” (no trailing space): "Give the city name only, plain text. The capital of Germany is called simply" (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json). Positive rows exist: `prompt_id=pos, prompt_variant=orig` (e.g., row 2 in 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv).
- Normalizer provenance: `strategy = next_ln1`; per-layer sources show `L0 → blocks[0].ln1`, final uses `ln_final` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json).
- Per-layer normalizer effect: spike-shaped norm trajectory (`shape = "spike"`, `r2 = 0.9146`; 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json). Early layers have high `resid_norm_ratio` with large `delta_resid_cos` (e.g., L0: `43.83`, `0.49`; L1: `75.79`, `0.73`).
- Unembed bias: `present = False`, `l2_norm = 0.0`; cosine metrics are bias-free (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json).
- Environment & determinism: `device = cpu`, `torch = 2.8.0+cu128`, `deterministic_algorithms = True`, `seed = 316` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json). Reproducibility OK.
- Numeric health: `any_nan = False`, `any_inf = False`, `layers_flagged = []` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json).
- Copy mask: `size = 2931`; sample unavailable (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json). Size is plausible for BPE tokenization.
- Gold alignment: `gold_alignment.ok = True` with variant `with_space`; `gold_alignment_rate = 1.0` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json).
- Repeatability (v1.39): `status = skipped` due to `deterministic_env`; `max_rank_dev/p95_rank_dev/top1_flip_rate = None` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json).
- Norm trajectory: `shape = spike`, `slope = 0.105`, `r2 = 0.9146`, `n_spikes = 34` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json).
- Measurement guidance: `prefer_ranks = True`, `suppress_abs_probs = True`, `preferred_lens_for_reporting = tuned`, `use_confirmed_semantics = True` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json).

## 3. Quantitative findings (layer‑by‑layer)
From positive rows only (pos/orig) in pure next-token CSV:
- L 0 — entropy 16.9985 bits; top‑1 ‘Forbes’; answer_rank 21319; KL_to_final 10.52 bits; resid_norm_ratio 43.83; delta_resid_cos 0.49 (row 2 in 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv).
- L 10 — entropy 16.8359 bits; top‑1 ‘hétérogènes’; answer_rank 98780; KL_to_final 10.91 bits; resid_norm_ratio 16.96; delta_resid_cos 0.97 (row 12 in 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv).
- L 20 — entropy 16.7424 bits; top‑1 ‘luš’; answer_rank 90800; KL_to_final 10.84 bits; resid_norm_ratio 10.52; delta_resid_cos 0.98 (row 22 in 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv).
- L 33 — entropy 16.7740 bits; top‑1 ‘Berlin’; answer_rank 1; KL_to_final 10.52 bits; answer_logit_gap 0.3223; resid_norm_ratio 3.88; delta_resid_cos 0.93 (row 35 in 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv).
- L 40 — entropy 3.1807 bits; top‑1 ‘Berlin’; answer_rank 1; KL_to_final 0.00 bits; answer_logit_gap 0.7614; resid_norm_ratio 3.86; delta_resid_cos 0.94 (row 42 in 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv).

Semantic layer: L_semantic_norm = 33 and L_semantic_confirmed = 33 (source = raw) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json). Preferred lens: tuned; confirmed semantics enabled.

Control margins: `first_control_margin_pos = 1`, `max_control_margin = 0.468` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json).

Entropy drift: entropy gap percentiles `(p25, p50, p75) = (13.58, 13.59, 13.66)` bits (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json).

## 4. Qualitative findings

### 4.1. Copy vs semantics (Δ‑gap)
No copy‑reflex detected: `L_copy_strict = None`, `L_copy_soft[k] = None` for k∈{1,2,3} (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json). Depth fractions report `semantic_frac = 0.825`, `delta_hat = None` (no copy onset to compare). Threshold stability: `stability = "none"`; earliest strict copy at τ=0.70 and τ=0.95 are both `None` and `norm_only_flags[τ] = None` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json).

### 4.2. Lens sanity: Raw‑vs‑Norm
Artifact metrics are low: `lens_artifact_score = 0.1146`, `lens_artifact_score_v2 = 0.1847`, tier = low; symmetric metrics show `js_divergence_p50 = 0.0353`, `l1_prob_diff_p50 = 0.3473`, with `first_js_le_0.1 = 0`, `first_l1_le_0.5 = 0`; top‑K overlap `jaccard_raw_norm_p50 = 0.5385`, `first_jaccard_raw_norm_ge_0.5 = 3` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json). Prevalence: `pct_layers_kl_ge_1.0 = 0.0244`, `n_norm_only_semantics_layers = 0`, `earliest_norm_only_semantic = None` (low lens‑induced risk near semantics).

### 4.3. Tuned‑Lens analysis
Preference: `tuned_is_calibration_only = False`; `preferred_semantics_lens_hint = tuned` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json). Attribution: rotation reduces KL strongly (`ΔKL_rot` p25/p50/p75 = 1.58/1.67/1.83 bits), temperature alone slightly reduces KL (−0.43/−0.33/−0.12 bits), with positive interaction (`ΔKL_interaction_p50 = 3.50`) and overall tuned improvement summary `delta_kl_tuned_p50 = 4.96` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json). Positional generalization is healthy (`pos_ood_ge_0.96 = 4.96`, `pos_in_dist_le_0.92 = 4.48`, `pos_ood_gap = 0.48`). Head‑mismatch: `tau_star_modelcal = 1.0`; last‑layer agreement clean (`kl_bits_tuned_final = 0.0` → `after_tau_star = 0.0`), consistent with `last_layer_consistency.kl_to_final_bits = 0.0` and `top1_agree = True` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json). Rank‑earliness deltas not reported.

### 4.4. KL, ranks, cosine, entropy milestones
KL thresholds: `first_kl_below_1.0 = 40`, `first_kl_below_0.5 = 40`; final KL ≈ 0 with `warn_high_last_layer_kl = False` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json). Ranks (preferred lens = tuned; baseline values reported): `first_rank_le_10 = 30`, `first_rank_le_5 = 30`, `first_rank_le_1 = 33` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json). Cosine milestones from CSV: `cos_to_final ≥ 0.2` at L35 (0.225) [row 37], and `≥ 0.4`/`≥ 0.6` at L40 (~1.0) [row 42] (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv). Entropy falls from 16.9985 bits (L0, row 2) to 3.1807 bits (L40, row 42); drift vs teacher summarized by entropy‑gap percentiles `(13.58, 13.59, 13.66)` bits.

### 4.5. Prism (shared‑decoder diagnostic)
Prism present/compatible (layers: embed, 9, 19, 29). KL deltas are negative at sampled percentiles (baseline − prism): p25 −1.93, p50 −5.98, p75 −4.97 bits, indicating higher KL under Prism at median; rank milestones unchanged (prism milestones not reached) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json). Verdict: Regressive.

### 4.6. Ablation & stress tests
No‑filler ablation shifts semantics slightly earlier: `L_sem_orig = 33 → L_sem_nf = 31` (`ΔL_sem = −2`, ~5% of 40 layers), with no copy milestone in either condition (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json). Control prompts summary: `first_control_margin_pos = 1`, `max_control_margin = 0.468`. Important‑word trajectory (records CSV) shows ‘Berlin’ rising near the end of the prompt in late layers: “is … Berlin …” (layer 30) [rows 582–584], and “is … Berlin …” (layer 31) [row 600] (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-records.csv).

### 4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓ (RMSNorm detected)
- LayerNorm bias removed ✓ (not applicable in RMS; bias fix not needed)
- FP32 unembed promoted ✓ (unembed dtype = torch.float32)
- Punctuation / markup anchoring noted ✗
- Copy‑reflex ✗ (no strict/soft milestone)
- Preferred lens honored ✓ (tuned; confirmed semantics used)
- Confirmed semantics reported ✓ (L=33, source=raw)
- Dual‑lens artefact metrics cited ✓ (v2, JS/Jaccard/L1)
- Tuned‑lens audit done ✓ (rotation/temp/positional/head)
- normalization_provenance present ✓ (ln_source @ L0/final)
- per‑layer normalizer effect present ✓ (resid_norm_ratio, delta_resid_cos)
- deterministic_algorithms true ✓
- numeric_health clean ✓
- copy_mask plausible ✓ (size=2931)
- evaluation_pack.citations used ✓ (rows referenced)

---
Produced by OpenAI GPT-5 
