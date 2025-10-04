# Evaluation Report: google/gemma-2-9b
## 1. Overview
Gemma‑2‑9B (google/gemma-2-9b), 42 layers, evaluated 2025‑09‑30. The probe traces layer‑wise next‑token behavior under a norm lens, capturing early copy/filler dynamics and the final semantic collapse to the gold token.

Quotes: "num_layers": 42 [001_layers_baseline/run-latest/output-gemma-2-9b.json:1557]; "Experiment started: 2025-09-30 23:57:21" [001_layers_baseline/run-latest/timestamp-20250930-2357].

## 2. Method sanity‑check
Diagnostics confirm RMS norm lensing and head calibration checks. "use_norm_lens": true, "unembed_dtype": "torch.float32" [001_layers_baseline/run-latest/output-gemma-2-9b.json:807–809]. The context prompt is correct and ends with “called simply”: "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply" [001_layers_baseline/run-latest/output-gemma-2-9b.json:816].

Gold alignment is ID‑based and ok: "gold_alignment": "ok" [001_layers_baseline/run-latest/output-gemma-2-9b.json:1086]. Measurement guidance instructs rank‑first reporting due to calibration risk: "prefer_ranks": true, "suppress_abs_probs": true, reasons include "warn_high_last_layer_kl" and "high_lens_artifact_risk" [001_layers_baseline/run-latest/output-gemma-2-9b.json:2143–2149]. Preferred lens is "norm" with confirmed semantics enabled: "preferred_lens_for_reporting": "norm", "use_confirmed_semantics": true [001_layers_baseline/run-latest/output-gemma-2-9b.json:2152–2153].

Copy/collapse bookkeeping present: L_copy, L_copy_soft, L_semantic, delta_layers, and soft variants exist (e.g., "L_copy": 0, "L_semantic": 42, "delta_layers": 42; "L_copy_soft": {"1": 0, "2": null, "3": null}) [001_layers_baseline/run-latest/output-gemma-2-9b.json:988–1006]. Strict‑copy config and sweep are present with labels mirrored in the files: copy_flag_columns include "copy_strict@{0.95,0.7,0.8,0.9}", "copy_soft_k{1,2,3}@0.5" [001_layers_baseline/run-latest/output-gemma-2-9b.json:1626–1633]. Threshold sweep is tagged: "stability": "mixed" and earliest strict L_copy_strict = 0 at τ=0.70 and τ=0.95 [001_layers_baseline/run-latest/output-gemma-2-9b.json:997–1011].

Copy‑reflex: strict copy fires immediately at L0–L4 (top‑1 “ simply”, copy_collapse=True). Example early row: "pos,orig,0,… copy_collapse,True …" [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2]. Soft‑copy k1 also fires at L0 ("copy_soft_k1@0.5", True) in the same row.

Last‑layer head calibration: final KL is not ≈0; "kl_to_final_bits": 1.0129 with "top1_agree": true; "p_top1_lens": 0.9298 vs "p_top1_model": 0.3943; "temp_est": 2.6102; "kl_after_temp_bits": 0.3499 [001_layers_baseline/run-latest/output-gemma-2-9b.json:1078–1090]. Per guidance, prefer rank‑based statements over absolute probabilities for this family.

Raw‑vs‑Norm checks: windowed diagnostic sampled around {0,42} with radius 4 shows very large divergence at the last layer: "center_layers": [0, 42], "norm_only_semantics_layers": [42], "max_kl_norm_vs_raw_bits_window": 92.316, "mode": "window" [001_layers_baseline/run-latest/output-gemma-2-9b.json:1045–1066]. Full scan corroborates high artifact risk: pct_layers_kl_ge_1.0 = 0.302, earliest_norm_only_semantic = 42, max_kl_norm_vs_raw_bits = 92.316, lens_artifact_score.tier = "high" [001_layers_baseline/run-latest/output-gemma-2-9b.json:1068–1080]. Raw‑lens sample rows confirm mid‑stack divergences (e.g., layer 22 KL ≈ 12.91 bits) [001_layers_baseline/run-latest/output-gemma-2-9b.json:1588–1596]. Treat any pre‑final “early semantics” cautiously; use rank milestones.

Norm‑temperature snapshots are present: "kl_to_final_bits_norm_temp@25%": 23.09 (L10), @50%: 21.27 (L21), @75%: 5.79 (L32) [001_layers_baseline/run-latest/output-gemma-2-9b.json:1104–1112]. Tau per layer is available (tau_norm_per_layer …) [001_layers_baseline/run-latest/output-gemma-2-9b.json:880–912].

Prism sidecar is present and compatible (diagnostic only): "mode": "auto", "k": 512, "layers": ["embed", 9, 20, 30] [001_layers_baseline/run-latest/output-gemma-2-9b.json:832–847]. Prism metrics provide baseline vs sidecar KL snapshots [001_layers_baseline/run-latest/output-gemma-2-9b.json:856–878].

Summary indices (baseline, norm lens): first_kl_below_1.0 = null; first_kl_below_0.5 = null; first_rank_le_{10,5,1} = 42 [001_layers_baseline/run-latest/output-gemma-2-9b.json:1124–1132, 986–994]. Units for KL/entropy are bits throughout.

## 3. Quantitative findings
Table (pos, orig only). Bolded row marks the confirmed semantic layer (L_semantic_confirmed = 42, source=tuned) [001_layers_baseline/run-latest/output-gemma-2-9b.json:1452–1460].

| Layer | Entropy (bits) | Top‑1 token |
|---|---:|---|
| L 0 | 0.000 |  simply |
| L 1 | 0.000 |  simply |
| L 2 | 0.000 |  simply |
| L 3 | 0.000 |  simply |
| L 4 | 0.002 |  simply |
| L 5 | 0.002 |  simply |
| L 6 | 0.128 |  simply |
| L 7 | 0.034 |  simply |
| L 8 | 0.098 |  simply |
| L 9 | 0.102 |  simply |
| L 10 | 0.281 |  simply |
| L 11 | 0.333 |  simply |
| L 12 | 0.109 |  simply |
| L 13 | 0.137 |  simply |
| L 14 | 0.166 |  simply |
| L 15 | 0.735 |  simply |
| L 16 | 3.568 |  simply |
| L 17 | 3.099 |  simply |
| L 18 | 3.337 |  simply |
| L 19 | 1.382 |  simply |
| L 20 | 3.163 |  simply |
| L 21 | 1.866 |  simply |
| L 22 | 2.190 |  simply |
| L 23 | 3.181 |  simply |
| L 24 | 1.107 |  simply |
| L 25 | 2.119 |  the |
| L 26 | 2.371 |  the |
| L 27 | 1.842 |  the |
| L 28 | 1.227 |  " |
| L 29 | 0.316 |  " |
| L 30 | 0.134 |  " |
| L 31 | 0.046 |  " |
| L 32 | 0.063 |  " |
| L 33 | 0.043 |  " |
| L 34 | 0.090 |  " |
| L 35 | 0.023 |  " |
| L 36 | 0.074 |  " |
| L 37 | 0.083 |  " |
| L 38 | 0.033 |  " |
| L 39 | 0.047 |  " |
| L 40 | 0.036 |  " |
| L 41 | 0.177 |  " |
| **L 42** | **0.370** | ** Berlin** |

Control margin (JSON): first_control_margin_pos = 18; max_control_margin = 0.8677 [001_layers_baseline/run-latest/output-gemma-2-9b.json:1659–1660]. Gold alignment is ok; use rank‑based statements when comparing probabilities across families per guidance.

Ablation (no‑filler): L_copy_orig = 0, L_sem_orig = 42; L_copy_nf = 0, L_sem_nf = 42; ΔL_copy = 0; ΔL_sem = 0 [001_layers_baseline/run-latest/output-gemma-2-9b.json:1635–1644]. No stylistic‑cue shift observed.

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = 1.67e‑05 − 0.37007 = −0.37005 (copy is extremely low‑entropy) [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2,49]. Soft ΔH₁ = −0.37005 (k=1; L_copy_soft[1]=0) [001_layers_baseline/run-latest/output-gemma-2-9b.json:1002–1006].

Confidence milestones (generic top‑1, within‑model): p_top1 > 0.30 at L0; p_top1 > 0.60 at L0; final‑layer p_top1 = 0.9298 [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:49].

Rank milestones (norm lens, baseline): rank ≤ 10 at L42; rank ≤ 5 at L42; rank ≤ 1 at L42 [001_layers_baseline/run-latest/output-gemma-2-9b.json:986–994]. Confirmed semantics: L_semantic_confirmed = 42 (source=tuned) [001_layers_baseline/run-latest/output-gemma-2-9b.json:1456–1460].

KL milestones: first_kl_below_1.0 = null; first_kl_below_0.5 = null; final KL_to_final_bits = 1.0129 (not ≈0), see last_layer_consistency [001_layers_baseline/run-latest/output-gemma-2-9b.json:1078–1084]. Per guidance, treat absolute probabilities cautiously.

Cosine milestones (norm lens): ge_0.2 at L1; ge_0.4 at L42; ge_0.6 at L42; final cos_to_final = 0.9993 [001_layers_baseline/run-latest/output-gemma-2-9b.json:1032–1039; 001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:49]. Depth fractions: L_semantic_frac = 1.0 [001_layers_baseline/run-latest/output-gemma-2-9b.json:1043–1046].

Tuned‑lens attribution (ΔKL in bits at depth percentiles): at ≈25%: ΔKL_tuned = −0.280, ΔKL_temp = −8.158, ΔKL_rot = +7.878; at ≈50%: ΔKL_tuned = −10.518, ΔKL_temp = −6.097, ΔKL_rot = −4.420; at ≈75%: ΔKL_tuned = +0.199, ΔKL_temp = −3.932, ΔKL_rot = +4.131 [001_layers_baseline/run-latest/output-gemma-2-9b.json:2124–2136]. Gate prefer_tuned=false; milestones unchanged (rank deltas = 0) [001_layers_baseline/run-latest/output-gemma-2-9b.json:2000–2016, 2024–2038].

Copy robustness (sweep): "stability": "mixed"; earliest L_copy_strict = 0 at τ=0.70 and at τ=0.95; norm_only_flags[τ] are false for all shown τ [001_layers_baseline/run-latest/output-gemma-2-9b.json:997–1011]. Windowed raw‑vs‑norm sidecar confirms divergence locations [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token-rawlens-window.csv:18–20].

Prism sidecar analysis: present and compatible. Early/mid‑depth KL is substantially worse than baseline (e.g., p50: baseline 15.17 bits vs prism 25.51 bits; Δ ≈ −10.33 bits) [001_layers_baseline/run-latest/output-gemma-2-9b.json:869–878]. Rank milestones do not improve (nulls in prism summary) [001_layers_baseline/run-latest/output-gemma-2-9b.json:858–867]. Copy flags do not spuriously flip at early layers (no strict copy under Prism at L0) [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token-prism.csv:1]. Verdict: Regressive.

## 4. Qualitative patterns & anomalies
Filler/copy anchoring is immediate and strong: layers 0–14 are dominated by the grammatical filler “ simply” as top‑1 with near‑deterministic probabilities [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2–15]. Around L25–27, the generic article “ the” takes over, then punctuation/quote tokens dominate L28–41, before a sharp final‑layer collapse to the answer “ Berlin” at L42 [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:25–29,41,49]. This matches a surface→format→meaning trajectory.

Negative control (Berlin→country): the top‑5 for “Berlin is the capital of” is cleanly aligned—“ Germany” 0.8766, then “ the” 0.0699 [001_layers_baseline/run-latest/output-gemma-2-9b.json:10–18]. Semantic leakage is minimal but present in tail: “ Berlin” appears with p ≈ 0.0019 [001_layers_baseline/run-latest/output-gemma-2-9b.json:31–39].

Records evolution: early positions in the prompt are highly confident on instruction tokens (e.g., at layer 0, pos 1: top‑1 “Give” with p≈0.99997) [001_layers_baseline/run-latest/output-gemma-2-9b-records.csv:3]. This confidence persists across the instruction template before NEXT attains semantics only at the final layer, consistent with high surface anchoring.

Collapse‑layer stability: removing the filler (“no_filler”) does not shift collapse (L_sem_nf = 42; ΔL_sem = 0) [001_layers_baseline/run-latest/output-gemma-2-9b.json:1639–1644]. Surface mass and coverage milestones indicate late surface‑to‑meaning: L_topk_decay_norm = 28 with topk_prompt_mass@50 ≈ 0.047 [001_layers_baseline/run-latest/output-gemma-2-9b.json:1018–1024].

Rest‑mass sanity: rest_mass at L42 is ≈ 1.11e‑05 (top‑k covers nearly all mass) [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:49].

Rotation vs amplification: cosine rises to ≈1.0 only at L42 while KL to final remains high even after norm‑temperature (e.g., KL_temp ≈ 5.79 bits at 75% depth) [001_layers_baseline/run-latest/output-gemma-2-9b.json:1108–1112]. This is “early direction, late calibration”: direction aligns late with strong head‑calibration mismatch and final‑head KL ≈1.01 bits.

Head calibration (final layer): known Gemma signature; temp_est ≈ 2.61 with KL_after_temp_bits ≈ 0.35 and warn_high_last_layer_kl=true [001_layers_baseline/run-latest/output-gemma-2-9b.json:1084–1092]. Treat absolute probabilities cautiously; rely on ranks/KL thresholds and within‑model trends.

Lens sanity: raw_lens_check.summary = { lens_artifact_risk: "high", max_kl_norm_vs_raw_bits: 12.91, first_norm_only_semantic_layer: null } [001_layers_baseline/run-latest/output-gemma-2-9b.json:1600–1608]. Sample row shows large mid‑depth divergence (layer 22, KL ≈ 12.91 bits) [001_layers_baseline/run-latest/output-gemma-2-9b.json:1588–1596]. Window/full diagnostics flag norm‑only semantics at the last layer and very high window KL (≈92.32 bits) [001_layers_baseline/run-latest/output-gemma-2-9b.json:1059–1066, 1072–1080].

Temperature robustness (tuned lens teacher entropy): at L21, entropy ≈10.41 bits vs teacher ≈0.154 bits (drift ≈+10.26 bits); NEXT remains low‑rank only at the final layer under tuned as well [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token-tuned.csv:26].

Important‑word trajectory: “Berlin” first appears as the final‑layer top‑1 at L42 and is not present in early top‑5; instruction words (“simply”, “the”) dominate earlier layers [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2–15,25–27,49].

Checklist
– RMS lens? ✓ (RMSNorm; pre‑norm) [001_layers_baseline/run-latest/output-gemma-2-9b.json:810–812, 1562]
– LayerNorm bias removed? ✓ (not_needed_rms_model) [001_layers_baseline/run-latest/output-gemma-2-9b.json:812]
– Entropy rise at unembed? ✓ (late spikes/quotes then final collapse) [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:28–41,49]
– FP32 un‑embed promoted? ✓ ("mixed_precision_fix": casting_to_fp32_before_unembed; unembed_dtype fp32) [001_layers_baseline/run-latest/output-gemma-2-9b.json:809, 813–816]
– Punctuation / markup anchoring? ✓ (quote token dominates L28–41) [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:28–41]
– Copy‑reflex? ✓ (strict/soft at L0) [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2]
– Grammatical filler anchoring? ✓ (“is/the/of” family visible; “the” at L25–27) [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:25–27]
– Preferred lens honored? ✓ (norm) [001_layers_baseline/run-latest/output-gemma-2-9b.json:2152]
– Confirmed semantics reported? ✓ (L_semantic_confirmed=42; source=tuned) [001_layers_baseline/run-latest/output-gemma-2-9b.json:1456–1460]
– Full dual‑lens metrics cited? ✓ (pct_layers_kl_ge_1.0, n_norm_only_semantics_layers, earliest_norm_only_semantic, lens_artifact_score tier) [001_layers_baseline/run-latest/output-gemma-2-9b.json:1068–1080]
– Tuned‑lens attribution done? ✓ (ΔKL_tuned/ΔKL_temp/ΔKL_rot at ~25/50/75%) [001_layers_baseline/run-latest/output-gemma-2-9b.json:2124–2136]

## 5. Limitations & data quirks
Final‑head calibration is non‑negligible (KL≈1.01 bits; warn_high_last_layer_kl=true), so absolute probabilities are not directly comparable across families; prefer rank milestones and within‑model trends [001_layers_baseline/run-latest/output-gemma-2-9b.json:1078–1092, 2143–2151]. Raw‑vs‑norm divergence is high (window max ≈92.32 bits; lens_artifact_risk=high), and norm‑only semantics appear at the last layer under the window/full diagnostics—treat any “early semantics” cautiously and prefer confirmed semantics [001_layers_baseline/run-latest/output-gemma-2-9b.json:1045–1080]. Surface‑mass metrics depend on tokenizer; use within‑model comparisons only.

## 6. Model fingerprint
Gemma‑2‑9B: collapse at L 42; final entropy 0.37 bits; “Berlin” is rank 1 only at the final layer.

---
Produced by OpenAI GPT-5
*Run executed on: 2025-09-30 23:57:21*
