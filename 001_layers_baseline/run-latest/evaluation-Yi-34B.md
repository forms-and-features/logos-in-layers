# Evaluation Report: 01-ai/Yi-34B

**1. Overview**
- Model: 01-ai/Yi-34B (34B), 60 layers (pre-norm). Run date: 2025‑09‑23.
- Probe captures layerwise entropy, calibration (KL to final), top‑k, copy flags, cosine trajectory, and answer rank for the positive prompt; control and no‑filler variants are included.

**2. Method Sanity‑Check**
JSON confirms normalized lens with FP32 unembedding and rotary positional model handling: > "use_norm_lens": true; "unembed_dtype": "torch.float32" [output-Yi-34B.json:807,809]. The context prompt ends exactly with “called simply”: > "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply" [output-Yi-34B.json:817]. Layer‑0 position info indicates rotary (token‑only) embeddings: > "layer0_position_info": "token_only_rotary_model" [output-Yi-34B.json:816].

Copy/collapse diagnostics present with strict and soft configs; thresholds and labels align across JSON/CSV: > "copy_thresh": 0.95, "copy_window_k": 1, "copy_match_level": "id_subsequence" [output-Yi-34B.json:846‑848]; > "copy_soft_config": { "threshold": 0.5, "window_ks": [1,2,3], "extra_thresholds": [] } [output-Yi-34B.json:833‑841]; > "copy_flag_columns": ["copy_strict@0.95","copy_soft_k1@0.5","copy_soft_k2@0.5","copy_soft_k3@0.5"] [output-Yi-34B.json:1077‑1082].

Gold alignment is ID‑level and resolved: > "gold_alignment": "ok" [output-Yi-34B.json:898]; gold_answer first_id=19616 for "Berlin" [output-Yi-34B.json:1111‑1118]. Negative control is present with summary: > "first_control_margin_pos": 1, "max_control_margin": 0.5835719932238135 [output-Yi-34B.json:1107‑1108]. Ablation summary exists with both variants; L_sem unchanged: > "L_sem_orig": 44, "L_sem_nf": 44, "delta_L_sem": 0 [output-Yi-34B.json:1085‑1089].

KL/rank summary indices (bits): > "first_kl_below_0.5": 60; "first_kl_below_1.0": 60; "first_rank_le_1": 44; "first_rank_le_5": 44; "first_rank_le_10": 43 [output-Yi-34B.json:849‑853]. Final‑head calibration is excellent (KL≈0): > "kl_to_final_bits": 0.000278…, "top1_agree": true, "p_top1_lens": 0.5555 vs "p_top1_model": 0.5627 [output-Yi-34B.json:900‑905].

Lens sanity (raw vs norm) is sampled and flags risk: > "mode": "sample" [output-Yi-34B.json:1016] and > "summary": { "first_norm_only_semantic_layer": 46, "max_kl_norm_vs_raw_bits": 80.5715, "lens_artifact_risk": "high" } [output-Yi-34B.json:1072‑1075]. Treat any pre‑final early‑semantics cautiously and prefer rank milestones.

Copy‑collapse flags: no strict or soft copy fired in layers 0–3 or anywhere (pure CSV shows False throughout for copy_collapse and copy_soft_k1/k2/k3; e.g., final row shows all False) [output-Yi-34B-pure-next-token.csv:63].

Copy‑collapse first‑True check: none (no row with copy_collapse=True). Soft copy flags earliest: none (no row with copy_soft_k1@0.5=True).

Prism sidecar is available and compatible (k=512; layers [embed,14,29,44]) [output-Yi-34B.json:819‑831].

Units: KL and entropy are in bits (CSV columns: entropy; kl_to_final_bits).

**3. Quantitative Findings**
Main table (pos, orig). One layer per row: “L n — entropy X bits, top‑1 ‘token’”. Bold = first semantic layer (is_answer=True).

| Layer | Entropy (bits) | Top‑1 |
|---|---:|---|
| L 0 | 15.962 | ' Denote' |
| L 1 | 15.942 | '.' |
| L 2 | 15.932 | '.' |
| L 3 | 15.839 | 'MTY' |
| L 4 | 15.826 | 'MTY' |
| L 5 | 15.864 | 'MTY' |
| L 6 | 15.829 | 'MTQ' |
| L 7 | 15.862 | 'MTY' |
| L 8 | 15.873 | '其特征是' |
| L 9 | 15.836 | '审理终结' |
| L 10 | 15.797 | '~\\\\' |
| L 11 | 15.702 | '~\\\\' |
| L 12 | 15.499 | '州里' |
| L 13 | 15.552 | 'ODM' |
| L 14 | 15.719 | 'ODF' |
| L 15 | 15.704 | '其特征是' |
| L 16 | 15.716 | '其特征是' |
| L 17 | 15.739 | '其特征是' |
| L 18 | 15.728 | '其特征是' |
| L 19 | 15.740 | '其特征是' |
| L 20 | 15.731 | '其特征是' |
| L 21 | 15.694 | '其特征是' |
| L 22 | 15.588 | 'ODP' |
| L 23 | 15.600 | '_______' |
| L 24 | 15.565 | '_______' |
| L 25 | 15.534 | '_______' |
| L 26 | 15.513 | '_______' |
| L 27 | 15.509 | '_______' |
| L 28 | 15.509 | '_______' |
| L 29 | 15.509 | '_______' |
| L 30 | 15.508 | 'ODM' |
| L 31 | 15.470 | ' ODM' |
| L 32 | 15.493 | '_______' |
| L 33 | 15.507 | '_______' |
| L 34 | 15.500 | '_______' |
| L 35 | 15.513 | '_______' |
| L 36 | 15.530 | '_______' |
| L 37 | 15.486 | 'MDE' |
| L 38 | 15.486 | 'MDM' |
| L 39 | 15.504 | 'MDM' |
| L 40 | 15.528 | 'MDM' |
| L 41 | 15.519 | 'MDM' |
| L 42 | 15.535 | 'keV' |
| L 43 | 15.518 | ' """' |
| **L 44** | **15.327** | **' Berlin'** |
| L 45 | 15.293 | ' Berlin' |
| L 46 | 14.834 | ' Berlin' |
| L 47 | 14.731 | ' Berlin' |
| L 48 | 14.941 | ' Berlin' |
| L 49 | 14.696 | ' Berlin' |
| L 50 | 14.969 | ' Berlin' |
| L 51 | 14.539 | ' Berlin' |
| L 52 | 15.137 | ' Berlin' |
| L 53 | 14.870 | ' Berlin' |
| L 54 | 14.955 | ' Berlin' |
| L 55 | 14.932 | ' Berlin' |
| L 56 | 14.745 | ' Berlin' |
| L 57 | 14.748 | ' ' |
| L 58 | 13.457 | ' ' |
| L 59 | 7.191 | ' ' |
| L 60 | 2.981 | ' Berlin' |

Control margin (JSON): first_control_margin_pos = 1; max_control_margin = 0.5835719932238135 [output-Yi-34B.json:1107‑1108].

Ablation (no‑filler): L_copy_orig = null, L_sem_orig = 44; L_copy_nf = null, L_sem_nf = 44; ΔL_copy = null; ΔL_sem = 0 [output-Yi-34B.json:1084‑1089].

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n/a (L_copy is null). Soft ΔHk (k∈{1,2,3}) = n/a (all L_copy_soft[k] null) [output-Yi-34B.json:860‑868].

Confidence milestones (pure CSV): p_top1 > 0.30 at layer 60; p_top1 > 0.60 not reached; final p_top1 = 0.5555 [output-Yi-34B-pure-next-token.csv:63].

Rank milestones (JSON): rank ≤ 10 at layer 43; rank ≤ 5 at layer 44; rank ≤ 1 at layer 44 [output-Yi-34B.json:851‑853].

KL milestones (JSON/CSV): first_kl_below_1.0 at 60; first_kl_below_0.5 at 60 [output-Yi-34B.json:849‑850]. KL decreases with depth and ≈0 at final; final‑row kl_to_final_bits = 0.000278 [output-Yi-34B-pure-next-token.csv:63].

Cosine milestones (pure CSV): cos_to_final ≥ 0.2 at L 1; ≥ 0.4 at L 44; ≥ 0.6 at L 51; final cos_to_final = 1.0000 [output-Yi-34B-pure-next-token.csv:63].

Prism Sidecar Analysis
- Presence: compatible=true; artifacts present (k=512; layers [embed,14,29,44]) [output-Yi-34B.json:819‑831].
- Early-depth stability (KL vs final, bits): baseline vs Prism at depths {0,15,30,45} →
  - L0: 12.01 vs 12.13 (Prism higher) [output-Yi-34B-pure-next-token.csv:40; output-Yi-34B-pure-next-token-prism.csv:1].
  - L15: 13.12 vs 12.18 (Prism slightly lower) [output-Yi-34B-pure-next-token.csv:40‑45; output-Yi-34B-pure-next-token-prism.csv:2].
  - L30: 13.54 vs 12.17 (Prism lower) [output-Yi-34B-pure-next-token.csv:42; output-Yi-34B-pure-next-token-prism.csv:3].
  - L45: 11.16 vs 12.17 (Prism higher) [output-Yi-34B-pure-next-token.csv:47; output-Yi-34B-pure-next-token-prism.csv:4].
- Rank milestones: answer_rank fields in Prism pure CSV are unavailable (null); deltas vs baseline cannot be computed.
- Top‑1 agreement: at sampled depths, both baselines disagree with final (“ Berlin”) until late; Prism does not show earlier agreement (e.g., L45 Prism top‑1 'POSE', not ' Berlin') [output-Yi-34B-pure-next-token-prism.csv:4].
- Cosine drift: Prism cos_to_final at early/mid layers is lower or negative (e.g., L45: −0.0028 vs baseline 0.4958) [output-Yi-34B-pure-next-token-prism.csv:4; output-Yi-34B-pure-next-token.csv:47].
- Copy flags: no flips; copy_collapse and soft flags remain False in Prism CSV (no True rows found).
- Verdict: Regressive (no earlier rank‑1; KL not consistently reduced; cosine worse mid‑stack).

**4. Qualitative Patterns & Anomalies**
Negative control “Berlin is the capital of” shows clean behavior: > " Germany", 0.8398; " the", 0.0537; " which", 0.0288; " what", 0.0120; " Europe", 0.0060 [output-Yi-34B.json:14‑31]. No Berlin in top‑5 → no semantic leakage.

Important‑word trajectory (records CSV): Berlin enters the NEXT‑token top‑5 at L 44 and stabilizes thereafter (e.g., > " Berlin", 0.00846 at layer 44 [output-Yi-34B-records.csv:788]; remains top‑1 through mid‑50s [output-Yi-34B-records.csv:824‑852]). “Germany” appears in top‑5 at L 44 and 46 but not consistently thereafter (> " Germany", 0.00154 at L 44; > " 德国", 0.00291 at L 46 [output-Yi-34B-records.csv:788,824]). “capital” is in top‑5 around L 43–49 then fades (> " capital", 0.00232 at L 44; 0.00598 at L 47 [output-Yi-34B-records.csv:788,842]).

Collapse‑layer vs instruction: Removing “simply” does not shift the semantic layer (ΔL_sem=0) [output-Yi-34B.json:1085‑1089], suggesting low sensitivity to this stylistic cue for this prompt.

Rest‑mass sanity: Top‑k rest_mass is high near the semantic turn (0.981 at L 44) and falls by the final layer (0.175) [output-Yi-34B-pure-next-token.csv:46,63], consistent with sharper mass concentration as calibration improves. Note rest_mass is top‑k coverage, not a lens‑fidelity metric.

Rotation vs amplification: cos_to_final rises early (≥0.2 by L1; ≥0.4 by L44) while KL to final remains large until very late (first <1.0 only at L60). This indicates “early direction, late calibration”: the representation points roughly toward the final head direction mid‑stack, but the distribution is not yet calibrated; probabilities (p_top1) only cross 0.30 at the final layer [output-Yi-34B-pure-next-token.csv:63]. Final‑head calibration is good (KL≈0) [output-Yi-34B.json:900].

Head calibration (final layer): No warning; > "warn_high_last_layer_kl": false; > "temp_est": 1.0; > "kl_after_temp_bits": 0.000278… [output-Yi-34B.json:906‑917].

Lens sanity: Raw‑vs‑norm summary flags high artifact risk and a “norm‑only semantics” layer at 46 (first_norm_only_semantic_layer=46; max_kl_norm_vs_raw_bits=80.57) [output-Yi-34B.json:1072‑1075]. Treat pre‑final “early semantics” cautiously; rely on rank milestones.

Temperature robustness: At T=0.1, Berlin rank 1 (p=0.9999996; entropy≈0) [output-Yi-34B.json:669‑687]. At T=2.0, Berlin rank 1 with p≈0.0488; entropy ≈12.49 bits [output-Yi-34B.json:736‑764, 760‑780].

Checklist: RMS lens? ✓ (RMSNorm model, norm lens active) [output-Yi-34B.json:807,810‑813]. LayerNorm bias removed? n/a for RMS ("not_needed_rms_model") [output-Yi-34B.json:812]. Entropy rise at unembed? Final entropy 2.98 bits with strong top‑1; earlier layers high entropy (15.9 bits) [output-Yi-34B-pure-next-token.csv:1,63]. FP32 un‑embed promoted? ✓ (use_fp32_unembed true; unembed_dtype float32) [output-Yi-34B.json:808‑809]. Punctuation/markup anchoring? Present in early/mid layers (e.g., quotes/commas in top‑5 around L 43–49) [output-Yi-34B-records.csv:770,842,860]. Copy‑reflex? ✗ (no strict/soft flags in L0–3). Grammatical filler anchoring? Mild — common function words appear in mid‑layers but not dominating L0–5.

**5. Limitations & Data Quirks**
- High raw‑vs‑norm lens artifact risk (“high”; first_norm_only_semantic_layer=46; max_kl_norm_vs_raw_bits≈80.57) [output-Yi-34B.json:1072‑1075]; prefer rank thresholds over absolute probabilities for early layers.
- Final‑layer KL≈0; nonetheless, cross‑model probability comparisons should be avoided; use within‑model ranks and KL milestones.
- Prism pure CSV lacks answer_rank fields (null), preventing rank‑milestone deltas; Prism KL/cosine are not consistently better here and appear regressive mid‑stack.
- Rest_mass near L_semantic is large (0.981 at L44) due to limited top‑k coverage; do not treat as lens fidelity.
- Raw‑lens mode is “sample”, so raw vs norm differences are sampled sanity, not exhaustive.

**6. Model Fingerprint**
“Yi‑34B: semantic turn at L 44; final entropy 2.98 bits; ‘Berlin’ stabilizes mid‑40s; KL≈0 at final.”

---
Produced by OpenAI GPT-5 
