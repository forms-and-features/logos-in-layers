# Evaluation Report: google/gemma-2-9b
## 1. Overview
Gemma‑2‑9B (google/gemma-2-9b), 9B parameters; run date 2025‑08‑30. The probe traces per‑layer next‑token predictions with a norm lens, reporting entropy/KL, copy vs. semantic collapse, calibration, and ablation/control diagnostics.

## 2. Method sanity‑check
JSON and CSV indicate rotary position handling and the intended RMSNorm lens are applied: "layer0_position_info": "token_only_rotary_model" and "use_norm_lens": true [L807–L816]. The context prompt ends with “called simply” and has no trailing space: "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply" [L804, L817]. Diagnostics include L_copy/L_copy_H/L_semantic/delta_layers and implementation flags: "L_copy": 0, "L_copy_H": 0, "L_semantic": 42, "delta_layers": 42; "use_norm_lens": true, "unembed_dtype": "torch.float32" [L812–L828]. Copy rule parameters are present and match the spec: "copy_thresh": 0.95, "copy_window_k": 1, "copy_match_level": "id_subsequence" [L829–L834]. Gold‑token alignment is ID‑based and ok: "gold_alignment": "ok" [L840–L841]. Negative control fields exist: "control_prompt" and "control_summary" [L1018–L1037]. Ablation exists and includes both variants: "ablation_summary" with L_copy/L_sem for orig and no_filler [L1010–L1016]. For the main table below, rows are filtered to prompt_id = pos, prompt_variant = orig.

Summary indices: first_kl_below_0.5 = null, first_kl_below_1.0 = null, first_rank_le_1 = 42, first_rank_le_5 = 42, first_rank_le_10 = 42 [L835–L839]. Units are bits (CSV columns entropy, kl_to_final_bits). Last‑layer head calibration shows KL not ≈ 0 at final: final CSV row has kl_to_final_bits = 1.0129 [row 49 in CSV], and diagnostics report last_layer_consistency: top1_agree = true, p_top1_lens = 0.9298 vs p_top1_model = 0.3943, temp_est = 2.6102, kl_after_temp_bits = 0.3499, warn_high_last_layer_kl = true [L842–L867]. For Gemma, this family‑specific behavior warrants relying on rank milestones over absolute probabilities.

Copy‑collapse flag check: first row with copy_collapse = True
  layer = 0 , p1 = 0.9999993 , p2 = 7.7e‑07  [row 2 in CSV]
  ✓ rule satisfied

Lens sanity (raw_lens_check): mode = sample; summary reports lens_artifact_risk = "high", max_kl_norm_vs_raw_bits = 12.9056, first_norm_only_semantic_layer = null [L948–L1009]. No norm‑only semantic layer is flagged; nonetheless, high artifact risk suggests caution and within‑model comparisons.

## 3. Quantitative findings
Per‑layer (pos, orig):
- L 0 — entropy 0.0000 bits, top‑1 ‘ simply’  [row 2]
- L 1 — entropy 0.0000 bits, top‑1 ‘ simply’  [row 3]
- L 2 — entropy 0.0000 bits, top‑1 ‘ simply’
- L 3 — entropy 0.0004 bits, top‑1 ‘ simply’
- L 4 — entropy 0.0021 bits, top‑1 ‘ simply’
- L 5 — entropy 0.0023 bits, top‑1 ‘ simply’
- L 6 — entropy 0.1279 bits, top‑1 ‘ simply’
- L 7 — entropy 0.0336 bits, top‑1 ‘ simply’
- L 8 — entropy 0.0984 bits, top‑1 ‘ simply’
- L 9 — entropy 0.1021 bits, top‑1 ‘ simply’
- L 10 — entropy 0.2814 bits, top‑1 ‘ simply’
- L 11 — entropy 0.3330 bits, top‑1 ‘ simply’
- L 12 — entropy 0.1093 bits, top‑1 ‘ simply’
- L 13 — entropy 0.1374 bits, top‑1 ‘ simply’
- L 14 — entropy 0.1658 bits, top‑1 ‘ simply’
- L 15 — entropy 0.7349 bits, top‑1 ‘ simply’
- L 16 — entropy 3.5683 bits, top‑1 ‘ simply’
- L 17 — entropy 3.0994 bits, top‑1 ‘ simply’
- L 18 — entropy 3.3367 bits, top‑1 ‘ simply’; rest_mass peaks here at 0.2563 [computed]
- L 19 — entropy 1.3823 bits, top‑1 ‘ simply’
- L 20 — entropy 3.1634 bits, top‑1 ‘ simply’
- L 21 — entropy 1.8665 bits, top‑1 ‘ simply’
- L 22 — entropy 2.1901 bits, top‑1 ‘ simply’
- L 23 — entropy 3.1811 bits, top‑1 ‘ simply’
- L 24 — entropy 1.1070 bits, top‑1 ‘ simply’
- L 25 — entropy 2.1189 bits, top‑1 ‘ the’  [row 32]
- L 26 — entropy 2.3713 bits, top‑1 ‘ the’
- L 27 — entropy 1.8425 bits, top‑1 ‘ the’
- L 28 — entropy 1.2267 bits, top‑1 ‘ "’
- L 29 — entropy 0.3160 bits, top‑1 ‘ "’
- L 30 — entropy 0.1341 bits, top‑1 ‘ "’
- L 31 — entropy 0.0461 bits, top‑1 ‘ "’
- L 32 — entropy 0.0625 bits, top‑1 ‘ "’
- L 33 — entropy 0.0427 bits, top‑1 ‘ "’
- L 34 — entropy 0.0900 bits, top‑1 ‘ "’
- L 35 — entropy 0.0234 bits, top‑1 ‘ "’
- L 36 — entropy 0.0741 bits, top‑1 ‘ "’
- L 37 — entropy 0.0825 bits, top‑1 ‘ "’
- L 38 — entropy 0.0335 bits, top‑1 ‘ "’
- L 39 — entropy 0.0469 bits, top‑1 ‘ "’
- L 40 — entropy 0.0362 bits, top‑1 ‘ "’
- L 41 — entropy 0.1767 bits, top‑1 ‘ "’
- L 42 — entropy 0.3701 bits, top‑1 ‘Berlin’  [row 49]

Bold semantic layer: L 42 (is_answer = True; gold_answer.string = “Berlin”).

Ablation (no‑filler): from JSON ablation_summary — L_copy_orig = 0, L_sem_orig = 42; L_copy_nf = 0, L_sem_nf = 42; ΔL_copy = 0, ΔL_sem = 0 [L1010–L1016]. Interpretation: removing “simply” neither delays nor advances copy or semantics here.

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = 0.0000 − 0.3701 = −0.3701.

Confidence milestones (pure CSV):
- p_top1 > 0.30 at layer 0; p_top1 > 0.60 at layer 0; final‑layer p_top1 = 0.9298 [row 49].

Rank milestones (diagnostics):
- rank ≤ 10 at layer 42; rank ≤ 5 at layer 42; rank ≤ 1 at layer 42 [L835–L839].

KL milestones (diagnostics/CSV):
- first_kl_below_1.0 = null; first_kl_below_0.5 = null [L835–L836]. KL is not ≈ 0 at final (1.0129 bits) [row 49], consistent with last_layer_consistency; treat final p_top1 as not directly comparable across families.

Cosine milestones (pure CSV):
- first cos_to_final ≥ 0.2 at layer 1; ≥ 0.4 at layer 42; ≥ 0.6 at layer 42; final cos_to_final = 0.9993 [row 49].

## 4. Qualitative patterns & anomalies
Early layers show a strong copy reflex: layers 0–3 have copy_collapse = True with near‑delta distributions on “simply” (e.g., “ simply”, 0.9999993) [row 2]. Mid‑stack, grammatical filler and punctuation dominate: “the” becomes top‑1 at L25–L27 (e.g., “ the”, 0.6023) [row 32], then a quotation mark token holds top‑1 through L28–L41. The answer only becomes top‑1 at the final layer (L42), aligning with a late semantic collapse consistent with tuned‑lens style observations of rotating into the final direction then calibrating late (see Tuned‑Lens 2303.08112 for similar phenomena).

Negative control (“Berlin is the capital of”): top‑5 are “ Germany” 0.8766, “ the” 0.0699, “ modern” 0.0077, “ a” 0.0053, “ ” 0.0034; Berlin still appears at rank 9 with p = 0.00187 — semantic leakage: Berlin rank 9 (p = 0.00187) [L10–L31].

Records context: Across layers for the next‑token position, “simply” dominates early (copy), then “the” and punctuation (“\"”) take over mid/late layers before the final pivot to “Berlin”. “Berlin” only enters top‑1 at L42; diagnostics show first_rank_le_5 = 42, matching this late consolidation [L835–L839]. Tokens semantically tied to the answer (“Germany”, quotes around names) appear frequently in late‑layer top‑5 (final top‑k includes “\"”, “ “ ”, and case variants of “Berlin”) [row 49].

Instruction ablation: ΔL_copy = 0 and ΔL_sem = 0 (no‑filler vs. orig) [L1010–L1016], suggesting the one‑word style cue “simply” is not driving the late semantic collapse timing here.

Rest‑mass sanity: Rest_mass falls to near‑zero by the end; the maximum occurs pre‑semantic at L18 (0.2563), then declines, ending at 1.1e‑05 [row 49]. No precision‑loss spikes after L_semantic.

Rotation vs amplification: cos_to_final rises early (≥0.2 by L1) while KL to final remains high until the end, then the answer rank jumps to 1 only at L42 — an “early direction, late calibration” signature. Final‑layer KL ≈ 1.01 bits and warn_high_last_layer_kl = true [L842–L867] indicate final‑head calibration issues; prefer rank‑based statements over absolute probabilities for cross‑model claims.

Head calibration (final layer): last_layer_consistency reports top1_agree = true, p_top1_lens = 0.9298 vs p_top1_model = 0.3943, temp_est = 2.6102, kl_after_temp_bits = 0.3499; cfg_transform is null [L842–L867]. These support the family‑known Gemma calibration pattern.

Lens sanity: raw_lens_check.summary: lens_artifact_risk = high; max_kl_norm_vs_raw_bits = 12.9056; first_norm_only_semantic_layer = null [L994–L1009]. A sampled check only; treat as a sanity screen and rely on within‑model trends.

Temperature robustness: At T = 0.1, Berlin rank 1 (p = 0.9809; entropy 0.137 bits) [L669–L696]. At T = 2.0, Berlin rank 1 (p = 0.0893; entropy 9.001 bits) [L736–L749]. Entropy rises sharply with temperature.

Important‑word trajectory — “Berlin” first enters top‑1 only at L42 and stabilises there; “Germany” and quotation marks are frequent competitors in late layers (e.g., final top‑k shows “ \"”, 0.2660) [row 49]. Early layers are dominated by copy (“simply”), then a mid‑stack shift to fillers (“the”) and punctuation.

Checklist:
✓ RMS lens? (RMSNorm; use_norm_lens = true) [L812–L817]
✓ LayerNorm bias removed? (not needed for RMS) [L810–L813]
✓ Entropy rise at unembed? (final 0.3701 > many mid‑layers)
✗ FP32 un‑embed promoted? (use_fp32_unembed = false; unembed_dtype = torch.float32) [L813–L815]
✓ Punctuation / markup anchoring? (quote token dominates L28–L41)
✓ Copy‑reflex? (layers 0–3 copy_collapse = True; e.g., row 2)
✗ Grammatical filler anchoring? (0–5 top‑1 not in {is,the,a,of})

## 5. Limitations & data quirks
Final‑layer KL ≈ 1.01 bits with warn_high_last_layer_kl = true indicates final‑head calibration; rely on rank milestones and qualitative KL trends rather than absolute probabilities. Raw‑vs‑norm lens differences are sampled only (raw_lens_check.mode = sample) with lens_artifact_risk = high; treat early‑layer interpretations cautiously. Rest_mass remains < 0.3 after L_semantic (1.1e‑05 at L42), reducing concern about norm‑lens mis‑scale.

## 6. Model fingerprint
Gemma‑2‑9B: collapse at L 42; final entropy 0.37 bits; “Berlin” only at final layer; mid‑stack punctuation/filler anchoring.

---
Produced by OpenAI GPT-5
*Run executed on: 2025-08-30 18:51:32*
