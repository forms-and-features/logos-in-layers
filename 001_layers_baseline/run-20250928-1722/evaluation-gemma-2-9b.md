# Evaluation Report: google/gemma-2-9b
**1. Overview**

google/gemma-2-9b (9B) probed on 2025‑09‑28 with a normalization (RMSNorm) lens, logging layer‑by‑layer NEXT‑token distributions, copy/filler collapse, and semantic emergence. The run captures strict early copy of the filler token “ simply” at L0 and a late semantic collapse to the gold token “Berlin” at L42.

**2. Method Sanity‑Check**

The run uses the intended norm lens with RMSNorm and rotary positions: “use_norm_lens: true … first_block_ln1_type: "RMSNorm" … layer0_position_info: "token_only_rotary_model"” [001_layers_baseline/run-latest/output-gemma-2-9b.json:808–817]. The context prompt ends with “called simply” and no trailing space: “context_prompt”: “Give the city name only, plain text. The capital of Germany is called simply” [001_layers_baseline/run-latest/output-gemma-2-9b.json:818].

Copy configuration and flags are present and consistent across JSON/CSV: “copy_thresh”: 0.95, “copy_window_k”: 1, “copy_match_level”: “id_subsequence” [001_layers_baseline/run-latest/output-gemma-2-9b.json:856–861]; soft‑copy config “threshold”: 0.5, “window_ks”: [1,2,3] [001_layers_baseline/run-latest/output-gemma-2-9b.json:840–847]; CSV flag columns mirror these labels: ["copy_strict@0.95","copy_soft_k1@0.5","copy_soft_k2@0.5","copy_soft_k3@0.5"] [001_layers_baseline/run-latest/output-gemma-2-9b.json:1383–1390]. Gold alignment is ok: “gold_alignment”: “ok” [001_layers_baseline/run-latest/output-gemma-2-9b.json:900–906]. Negative control prompt is present with summary: “The capital of France…” and control summary { first_control_margin_pos: 18, max_control_margin: 0.8677 } [001_layers_baseline/run-latest/output-gemma-2-9b.json:1398–1410; 1411–1414]. Ablation summary exists with both variants populated: { L_copy_orig: 0, L_sem_orig: 42, L_copy_nf: 0, L_sem_nf: 42, delta_L_copy: 0, delta_L_sem: 0 } [001_layers_baseline/run-latest/output-gemma-2-9b.json:1389–1396].

Strict copy‑collapse fires in early layers (ID‑level k=1, τ=0.95, δ=0.10): at L0 the top‑1 is “ simply” with p=0.9999993 and copy_collapse=True [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2]. Earliest soft‑copy k=1@τ=0.5 also fires at L0 [same row], while k2/k3 remain null in diagnostics. The strict semantic layer is L42 (is_answer=True), where the top‑1 is “ Berlin”, p=0.9298 [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:49].

Summary indices (bits, ranks): first_kl_below_0.5 = null; first_kl_below_1.0 = null; first_rank_le_1 = 42; first_rank_le_5 = 42; first_rank_le_10 = 42 [001_layers_baseline/run-latest/output-gemma-2-9b.json:900–906]. KL/entropy units are in bits; the final row KL is not ≈0, and last‑layer consistency is reported: “kl_to_final_bits: 1.0129 … top1_agree: true … p_top1_lens: 0.9298 vs p_top1_model: 0.3943 … temp_est: 2.610 … kl_after_temp_bits: 0.3499 … warn_high_last_layer_kl: true” [001_layers_baseline/run-latest/output-gemma-2-9b.json:962–981]. Treat absolute probabilities cautiously (known Gemma pattern).

Lens sanity (raw vs norm): mode=“sample”, lens_artifact_risk=“high”, max_kl_norm_vs_raw_bits=12.91, first_norm_only_semantic_layer=null [001_layers_baseline/run-latest/output-gemma-2-9b.json:1378–1382]. Example sample: “layer: 0 … top1_agree: true … answer_rank_norm: 1468 vs raw: 22” [raw_lens_check.samples[0]] [001_layers_baseline/run-latest/output-gemma-2-9b.json:1210–1220].

Copy‑collapse flag check: first firing row = layer 0, top‑1 “ simply”, p1=0.9999993; runner’s copy_collapse=True (✓ rule satisfied) [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2]. Soft‑copy: earliest k1@τ=0.5=True at L0; k2/k3 never fire in early layers.

For all quantitative results below, rows are filtered to prompt_id=pos and prompt_variant=orig (pure‑next‑token CSV). 

**3. Quantitative Findings**

Table — per layer: L i — entropy H bits, top‑1 ‘token’ (pos/orig). Bold marks L_semantic (is_answer=True).

| Layer | Entropy (bits) | Top‑1 token |
|---:|---:|:--|
| 0 | 0.000017 | simply |
| 1 | 0.000000 | simply |
| 2 | 0.000031 | simply |
| 3 | 0.000430 | simply |
| 4 | 0.002116 | simply |
| 5 | 0.002333 | simply |
| 6 | 0.127902 | simply |
| 7 | 0.033569 | simply |
| 8 | 0.098417 | simply |
| 9 | 0.102087 | simply |
| 10 | 0.281391 | simply |
| 11 | 0.333046 | simply |
| 12 | 0.109330 | simply |
| 13 | 0.137400 | simply |
| 14 | 0.165772 | simply |
| 15 | 0.734873 | simply |
| 16 | 3.568274 | simply |
| 17 | 3.099445 | simply |
| 18 | 3.336717 | simply |
| 19 | 1.382336 | simply |
| 20 | 3.163441 | simply |
| 21 | 1.866495 | simply |
| 22 | 2.190102 | simply |
| 23 | 3.181111 | simply |
| 24 | 1.107039 | simply |
| 25 | 2.118879 | the |
| 26 | 2.371327 | the |
| 27 | 1.842460 | the |
| 28 | 1.226664 | " |
| 29 | 0.315988 | " |
| 30 | 0.134063 | " |
| 31 | 0.046090 | " |
| 32 | 0.062538 | " |
| 33 | 0.042715 | " |
| 34 | 0.090030 | " |
| 35 | 0.023370 | " |
| 36 | 0.074091 | " |
| 37 | 0.082534 | " |
| 38 | 0.033455 | " |
| 39 | 0.046899 | " |
| 40 | 0.036154 | " |
| 41 | 0.176738 | " |
| **42** | **0.370067** | **Berlin** |

Control margin (JSON): first_control_margin_pos = 18; max_control_margin = 0.8677 [001_layers_baseline/run-latest/output-gemma-2-9b.json:1411–1414].

Ablation (no‑filler): L_copy_orig=0, L_sem_orig=42; L_copy_nf=0, L_sem_nf=42; ΔL_copy=0, ΔL_sem=0 [001_layers_baseline/run-latest/output-gemma-2-9b.json:1389–1396].

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = 0.0000167 − 0.370067 ≈ −0.37005. Soft ΔH₁ (k=1) = −0.37005; k=2,3 n.a. (no soft copy layer). Final‑layer p_top1 = 0.9298 [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:49].

Confidence milestones (generic top‑1): p_top1 > 0.30 at L0; p_top1 > 0.60 at L0; final p_top1 = 0.9298.

Rank milestones (diagnostics): rank ≤10 at L42; rank ≤5 at L42; rank ≤1 at L42 [001_layers_baseline/run-latest/output-gemma-2-9b.json:900–906].

KL milestones (diagnostics): first_kl_below_1.0 = null; first_kl_below_0.5 = null [001_layers_baseline/run-latest/output-gemma-2-9b.json:900–903]. KL decreases near the top but is not ≈0 at final (1.0129 bits); last‑layer consistency warns of calibration bias [001_layers_baseline/run-latest/output-gemma-2-9b.json:962–981].

Cosine milestones (within‑model): cos_to_final ≥0.2 at L1; ≥0.4 at L42; ≥0.6 at L42; final cos_to_final = 0.9993 [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:49].

Prism Sidecar Analysis. Prism artifacts are present and compatible [001_layers_baseline/run-latest/output-gemma-2-9b.json:826–838]. At sampled depths, KL(P_layer||P_final) is consistently much worse than baseline (bits; baseline → Prism): L0 14.39 → 28.48; L9 14.53 → 27.67; L20 18.98 → 27.15; L30 1.80 → 26.70; L42 1.01 → 28.73 [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:37,49; 001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token-prism.csv:1–9]. Prism never achieves rank ≤10/5/1 (n.a. at all depths). Cosine to final is also smaller in magnitude at early/mid depths (e.g., L0: −0.074 baseline vs −0.231 Prism). Copy flags do not spuriously flip under Prism (no early copy_collapse). Verdict: Regressive (KL increases and rank milestones degrade).

Tuned‑Lens Sidecar. ΔKL at depth percentiles (Δ = KL_norm − KL_tuned): ~−0.28 bits @25%, ~−10.52 bits @50%, ~+0.20 bits @75% (tuned worse at mid‑depths). Tuned rank milestones remain unchanged (first_rank≤{10,5,1} = 42). Entropy drift at mid‑depth (L21): baseline −1.07 bits vs tuned +7.48 bits (entropy − teacher_entropy_bits). Coverage improves earlier under tuned (L_topk_decay_norm=28 vs L_topk_decay_tuned=16) [001_layers_baseline/run-latest/output-gemma-2-9b.json:896–899]. Norm‑temp snapshots are present: KL_temp@{25,50,75}% = {23.09, 21.27, 5.79} bits [001_layers_baseline/run-latest/output-gemma-2-9b.json:982–995]. Overall tuned lens shows mid‑stack regression (cf. tuned_lens_regression=true; skip warning large) [001_layers_baseline/run-latest/output-gemma-2-9b.json:1210–1246].

**4. Qualitative Patterns & Anomalies**

Early layers exhibit a strong copy reflex on the grammatical adverb “ simply”: L0–L7 are dominated by “ simply” as generic top‑1 with p≈1.0 and copy_collapse=True [e.g., “…, simply, 0.9999993, … copy_collapse, True …” (layer 0) 001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2]. Mid‑stack transitions shift top‑1 to punctuation/quotes (“\"”, “ ” ”), indicating a surface‑form anchoring before semantic resolution. The semantic collapse is late and abrupt: at L42 the answer token “ Berlin” becomes top‑1 with p=0.9298 and is_answer=True [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:49].

Negative control (Berlin → country): “Berlin is the capital of” yields a clean top‑5 with “ Germany” top‑1 (p=0.8766) and “ Berlin” present but low (p=0.00187): “( Germany, 0.8766) … ( Berlin, 0.00187)” [001_layers_baseline/run-latest/output-gemma-2-9b.json:14–19,46–47]. Semantic leakage: Berlin rank 9 (p = 0.00187).

Important‑word trajectory (records). “Germany”, “capital”, and the instruction token “ simply” stay high‑confidence across prompt positions at L0 (e.g., “ Germany, 0.9999980” at pos 13; “ capital, 1.0” at pos 11; “ simply, 0.9999993” at NEXT) [001_layers_baseline/run-latest/output-gemma-2-9b-records.csv:13–16; 001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2]. The target “Berlin” does not appear in NEXT top‑5 until L42, at which point it dominates, consistent with a late semantic resolution.

Instructional phrasing. Removing “simply” (no‑filler) does not change collapse indices here (ΔL_copy=0; ΔL_sem=0) [001_layers_baseline/run-latest/output-gemma-2-9b.json:1389–1396]. In the test‑prompt family without “simply”, the model still directly outputs “ Berlin” with high probability (e.g., “Germany’s capital city is called” → p=0.6645) [001_layers_baseline/run-latest/output-gemma-2-9b.json:232–241], but layer‑level earliness cannot be inferred from test‑prompt samples alone.

Rest‑mass sanity. Top‑k rest_mass remains very low near and after semantic collapse; at L42 rest_mass ≈ 1.11e‑05 [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:49]. No spikes suggestive of numerical precision loss were observed.

Rotation vs amplification. Cosine to final rises early (≥0.2 by L1) while KL remains high until late, and the final‑row KL is not ≈0; together with the last‑layer consistency warning, this indicates “early direction, late calibration” and a final‑head calibration mismatch. Prefer rank milestones over absolute probabilities for cross‑family comparisons.

Head calibration (final layer). JSON flags warn_high_last_layer_kl=true with temp_est≈2.61 and KL_after_temp≈0.35 bits; top‑1 agrees but p_top1_lens (0.93) exceeds p_top1_model (0.39) [001_layers_baseline/run-latest/output-gemma-2-9b.json:962–981]. Treat reported probabilities as lens‑calibrated rather than directly comparable to the model head (Gemma family pattern).

Lens sanity. Raw‑vs‑norm sampling reports lens_artifact_risk=high and max_kl_norm_vs_raw≈12.91 bits [001_layers_baseline/run-latest/output-gemma-2-9b.json:1378–1382]. A sample shows rank divergences at L0 (answer_rank_norm=1468 vs raw=22) despite top‑1 agreement [001_layers_baseline/run-latest/output-gemma-2-9b.json:1210–1220]. Use within‑model trends and rank milestones.

Temperature robustness. At T=2.0, “ Berlin” remains top‑1 but softened (p=0.0893) and entropy rises to 9.00 bits [001_layers_baseline/run-latest/output-gemma-2-9b.json:1115–1142].

Checklist
- RMS lens?  ✓  (RMSNorm) [001_layers_baseline/run-latest/output-gemma-2-9b.json:811–812]
- LayerNorm bias removed?  ✓  (“not_needed_rms_model”) [001_layers_baseline/run-latest/output-gemma-2-9b.json:813]
- Entropy rise at unembed?  ✓  (teacher entropy 2.94 bits at final) [001_layers_baseline/run-latest/output-gemma-2-9b.json:1200–1219]
- FP32 un‑embed promoted?  ✓  (unembed_dtype=float32; casting_to_fp32_before_unembed) [001_layers_baseline/run-latest/output-gemma-2-9b.json:809–816]
- Punctuation / markup anchoring?  ✓  (quotes dominate L28–L41)
- Copy‑reflex?  ✓  (copy_collapse=True at L0–L3) [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2–5]
- Grammatical filler anchoring?  ✗ (top‑1 in L0–5 is “ simply”, not {is,the,a,of})

**5. Limitations & Data Quirks**

- Final‑row KL≈1.01 bits with warn_high_last_layer_kl=true indicates final‑head calibration mismatch; rely on ranks and KL thresholds, not absolute probabilities, for cross‑model claims [001_layers_baseline/run-latest/output-gemma-2-9b.json:962–981].
- Raw‑vs‑norm lens check is “sample” mode with lens_artifact_risk=high and max_kl_norm_vs_raw≈12.91 bits, so early “semantics” may be lens‑induced; prefer within‑model trends [001_layers_baseline/run-latest/output-gemma-2-9b.json:1378–1382].
- Surface‑mass metrics and rest_mass reflect tokenizer/top‑k coverage, not fidelity; use last_layer_consistency and raw‑lens checks for calibration sanity.

**6. Model Fingerprint**

Gemma‑2‑9B: copy at L0; semantic at L42; final lens entropy 0.37 bits; “Berlin” first appears as top‑1 only at L42.

---
Produced by OpenAI GPT-5 
*Run executed on: 2025-09-28 17:22:48*
