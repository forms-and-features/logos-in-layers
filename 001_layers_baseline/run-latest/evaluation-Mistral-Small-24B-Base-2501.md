## 1. Overview

This run probes mistralai/Mistral-Small-24B-Base-2501 with a normed logit-lens along the full 40-layer stack, targeting the first unseen token after “called simply” and tracking entropy, KL-to-final, rank, cosine, surface mass, and copy detectors. The gold answer is Berlin and the probe captures a late semantic collapse at L33 with clean last-layer head calibration and comprehensive sidecars (Prism and Tuned-Lens) for robustness.

## 2. Method sanity‑check

Diagnostics confirm the normalized lens is applied with RMSNorm, FP32 shadow unembed, and rotary positional encoding in layer 0: “use_norm_lens: true … unembed_dtype: torch.float32 … layer0_position_info: token_only_rotary_model” (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:807–816). The context prompt matches and ends with “called simply” (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4, 817). Last‑layer head calibration is excellent: “kl_to_final_bits: 0.0 … top1_agree: true … temp_est: 1.0 … kl_after_temp_bits: 0.0 … warn_high_last_layer_kl: false” (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1071–1089). Units: entropy and KL in bits.

Copy detectors and flags are present: “copy_flag_columns: [copy_strict@0.95, …, copy_soft_k{1,2,3}@0.5]” (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1593–1601). Configuration is explicit: “copy_soft_config: { threshold: 0.5, window_ks: [1,2,3], extra_thresholds: [] }” (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:880–888). Strict rule is ID‑level contiguous subsequence (k=1): “copy_match_level: id_subsequence … copy_thresh: 0.95 … copy_window_k: 1” (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:932–946). Gold alignment is “ok” (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1069). Negative control is present with its own gold and summary (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1610–1628). Ablation summary exists with both variants (orig/no_filler): “L_sem_orig: 33 … L_sem_nf: 31 … delta_L_sem: -2” (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1602–1609).

Summary thresholds (norm lens) are provided: “first_kl_below_0.5: 40, first_kl_below_1.0: 40, first_rank_le_1: 33, first_rank_le_5: 30, first_rank_le_10: 30” (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:939–943). Measurement guidance does not restrict probabilities: “prefer_ranks: false, suppress_abs_probs: false” (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:2108–2114).

Raw‑vs‑Norm window check: radius 4, center_layers [30,33,40], norm_only_semantics_layers [] and “max_kl_norm_vs_raw_bits_window: 5.9801” (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1041–1068). Lens sanity sampling reports “lens_artifact_risk: low; first_norm_only_semantic_layer: null; max_kl_norm_vs_raw_bits: 0.1793” (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1587–1591). Threshold sweep is present with stability “none” and no strict copy at τ ∈ {0.70,0.95} (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1165–1190 and 1171–1176).

Copy flag check in pure next-token CSV (pos, orig): no rows where copy_collapse=True or any copy_soft_k{1,2,3}@0.5=True in layers 0–3; strict and soft remain null throughout. First semantic layer is L33 (answer_rank=1 at row “pos,orig,33,…” with is_answer=True) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:35). Strict L_copy_strict is null; Δ is therefore undefined under strict and soft.

## 3. Quantitative findings

Per‑layer summary (pos, orig). Bold indicates L_semantic (first is_answer=True for Berlin).

| Layer | Entropy (bits) | Top‑1 |
|---:|---:|:--|
| 0 | 16.9985 | Forbes |
| 1 | 16.9745 | 随着时间的 |
| 2 | 16.9441 | 随着时间的 |
| 3 | 16.8120 | 随着时间的 |
| 4 | 16.8682 |  quelcon |
| 5 | 16.9027 | народ |
| 6 | 16.9087 | народ |
| 7 | 16.8978 | народ |
| 8 | 16.8955 |  quelcon |
| 9 | 16.8852 |  simply |
| 10 | 16.8359 |  hétérogènes |
| 11 | 16.8423 | 从那以后 |
| 12 | 16.8401 |  simply |
| 13 | 16.8709 |  simply |
| 14 | 16.8149 | стен |
| 15 | 16.8164 | luš |
| 16 | 16.8300 | luš |
| 17 | 16.7752 | luš |
| 18 | 16.7608 | luš |
| 19 | 16.7746 | luš |
| 20 | 16.7424 | luš |
| 21 | 16.7747 |  simply |
| 22 | 16.7644 |  simply |
| 23 | 16.7690 | -на |
| 24 | 16.7580 | -на |
| 25 | 16.7475 |  «** |
| 26 | 16.7692 |  «** |
| 27 | 16.7763 |  «** |
| 28 | 16.7407 |  «** |
| 29 | 16.7604 |  «** |
| 30 | 16.7426 | -на |
| 31 | 16.7931 | -на |
| 32 | 16.7888 | -на |
| **33** | **16.7740** | **Berlin** |
| 34 | 16.7613 | Berlin |
| 35 | 16.7339 | Berlin |
| 36 | 16.6994 | Berlin |
| 37 | 16.5133 | " """ |
| 38 | 15.8694 | " """ |
| 39 | 16.0050 | Berlin |
| 40 | 3.1807 | Berlin |

Bold semantic layer (first is_answer=True): L33 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:35).

Control margin (JSON control_summary): first_control_margin_pos = 1; max_control_margin = 0.4679627253462968 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1625–1627). Gold alignment is ok for both Berlin and Paris (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1069, 1610–1623).

Ablation (no‑filler): L_copy_orig = null; L_sem_orig = 33; L_copy_nf = null; L_sem_nf = 31; ΔL_copy = null; ΔL_sem = −2 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1602–1609).

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n/a (L_copy null). Soft ΔHk not available (all L_copy_soft[k] null).

Confidence milestones (pure CSV): p_top1 > 0.30 at L40; p_top1 > 0.60 not reached; final-layer p_top1 = 0.455 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:42).

Rank milestones (JSON): rank ≤ 10 at L30, rank ≤ 5 at L30, rank ≤ 1 at L33 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:939–943).

KL milestones (JSON): first_kl_below_1.0 at L40; first_kl_below_0.5 at L40; KL decreases to ≈0 at final, consistent with last‑layer agreement (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:939–943, 1071–1078).

Cosine milestones (JSON): cos_to_final ≥ 0.2 at L35; ≥ 0.4 at L40; ≥ 0.6 at L40; final cos_to_final ≈ 1.0 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1026–1031; pure CSV final row: 42).

Depth fractions (JSON): L_semantic_frac = 0.825; first_rank_le_5_frac = 0.75 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1033–1036).

Copy robustness (threshold sweep): stability = “none”; strict L_copy earliest is null at τ=0.70 and τ=0.95; norm_only_flags all null (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1165–1190).

Prism Sidecar Analysis. Presence: compatible=true (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:825–836). Early-depth stability: KL medians increase under Prism vs baseline: p50 baseline 10.84 vs Prism 16.82 (Δ = −5.98 bits) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:856–870). Rank milestones under Prism are null where baseline reaches le_1=33 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:839–849). Verdict: Regressive (higher KL, no earlier ranks).

Tuned‑Lens sidecar (present). ΔKL medians (baseline − tuned) at percentiles: p25=4.19, p50=4.59, p75=5.35 bits (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:2060–2072). Rank earliness degrades under tuned (baseline le_1=33 vs tuned le_1=39) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:2047–2058). Surface→meaning: L_surface_to_meaning_norm=40 with answer_mass=0.455, echo_mass=0.00424 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1009–1016). Geometry: L_geom_norm=31 (cos_to_answer≈0.095, cos_to_prompt_max≈0.074) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1017–1021). Coverage: L_topk_decay_norm=0 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1021). Norm temp snapshots (KL vs teacher at per-layer τ): {25%: 10.60, 50%: 10.69, 75%: 16.71} bits (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1090–1101). Entropy drift (tuned): at L1, entropy 10.56 vs teacher 3.18 (Δ≈+7.38 bits) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token-tuned.csv:1).

## 4. Qualitative patterns & anomalies

The negative control “Berlin is the capital of” yields a sharp country completion with Germany top‑1 and Berlin still present in the top‑10: “Germany 0.8021 … Berlin 0.00481” (semantic leakage: Berlin rank 7, p=0.00481) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:10–39). Across the main prompt, Berlin first enters the top‑3 around L30 (“Berlin, 0.00017”, rank 3), rises to rank 2 by L31–32, and becomes top‑1 at L33 with low absolute mass (0.00042) that grows through late layers to 0.455 at L40 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:32–42). Important-word evolution in the records CSV shows early distractors and multilingual artifacts; mid‑stack the top‑1s include script/markup tokens (“«**” at L25–29) and hyphenated substrings, before the answer stabilizes; quotation tokens dominate L37–38 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:25–39). Removing “simply” modestly advances semantics (ΔL_sem = −2), suggesting weak stylistic anchoring rather than strong filler‑driven semantics (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1602–1609).

Rest‑mass declines with depth: from ~0.9999 early to 0.181 at L40 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:42), consistent with concentration on a few tokens; no post‑semantic spikes indicative of precision loss are observed. Rotation vs amplification: cos_to_final rises by L35 (≥0.2) and approaches 1.0 at L40 while KL-to-final remains high until the end (first_kl≤1.0 at L40), indicating an “early direction, late calibration” signature that resolves at the final head (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1026–1031, 939–943, 1071–1078). Final‑head calibration is clean (warn_high_last_layer_kl=false; temp_est=1.0; kl_after_temp_bits=0.0) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1071–1089).

Lens sanity is strong: raw‑lens sampling reports lens_artifact_risk=low and no norm‑only semantics; the windowed raw‑vs‑norm check over layers 26–40 shows a large max KL(‖) window value (5.98 bits) but no norm‑only semantics layers surfaced (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1041–1068, 1531–1591). Temperature robustness (from test_prompts): at T=0.1, Berlin dominates (0.9995) and at T=2.0 it remains top‑1 at ~0.03 with entropy ≈14.36 bits (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:745–793). Important‑word trajectory summary: Berlin first appears top‑3 at L30, achieves top‑1 at L33, and punctuation tokens briefly disrupt top‑1 at L37–38 before the final head reasserts Berlin (rows 32–40 in the pure CSV).

Checklist
- RMS lens? ✓ (RMSNorm with norm lens; 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:807–816)
- LayerNorm bias removed? ✓ (“not_needed_rms_model”; 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:812)
- Entropy rise at unembed? n.a. (final head is the teacher; last-layer KL≈0)
- FP32 un‑embed promoted? ✓ (“unembed_dtype: torch.float32”; 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:809)
- Punctuation / markup anchoring? ✓ (top‑1 “" """” at L37–38; pure CSV rows 37–38)
- Copy‑reflex? ✗ (no strict or soft hits in L0–3; all null)
- Grammatical filler anchoring? ✗ (early top‑1s are non‑filler multilingual tokens)

## 5. Limitations & data quirks

- Rest_mass is top‑k coverage only; do not interpret as lens fidelity (KL/entropy quoted in bits). Here it falls to 0.181 at L40 as the distribution concentrates (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:42).
- KL is lens‑sensitive; however last‑layer KL≈0 and warn_high_last_layer_kl=false, so final calibration appears reliable (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1071–1089). Prefer within‑model rank milestones for cross‑model comparison.
- Raw‑vs‑norm lens differences were assessed in sampled mode; the windowed check shows a large max KL window value (5.98 bits) but no norm‑only semantics layers; treat early „semantics” cautiously if relying on absolute probabilities (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1041–1068, 1531–1591).
- Surface‑mass relies on tokenizer segmentation; cross‑model absolute masses are not comparable; trends reported within model only.

## 6. Model fingerprint

Mistral‑Small‑24B‑Base‑2501: collapse at L33; final entropy 3.18 bits; Berlin stabilizes late with clean final‑head calibration.

---
Produced by OpenAI GPT-5 
