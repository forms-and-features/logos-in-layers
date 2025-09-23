**Overview**
- Model: google/gemma-2-9b. This probe runs a norm-lens, layer-by-layer next-token analysis and records entropy, rank/probability milestones, KL-to-final, and cosine-to-final per layer. Final head calibration is summarized and a Prism sidecar is compared where available.

**Method Sanity‑Check**
- Norm lens and positions: use_norm_lens = true; token-only rotary positions (no additive pos-embed at L0). Quote: "use_norm_lens": true (001_layers_baseline/run-latest/output-gemma-2-9b.json:807); "layer0_position_info": "token_only_rotary_model" (001_layers_baseline/run-latest/output-gemma-2-9b.json:817).
- Context prompt ends with “called simply”: "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply" (001_layers_baseline/run-latest/output-gemma-2-9b.json:817).
- Copy‑collapse rule present and firing early (τ=0.95, δ=0.10; ID‑subsequence, k=1). Quote: "copy_thresh": 0.95, "copy_window_k": 1, "copy_match_level": "id_subsequence" (001_layers_baseline/run-latest/output-gemma-2-9b.json:837–839). Pure CSV has copy_collapse = True in layers 0–3; e.g., L0: “…,copy_collapse,True,entropy_collapse,True,is_answer,False,…” (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2).
- Gold alignment OK, ID‑based: diagnostics.gold_alignment = "ok"; gold_answer.first_id = 12514 for "▁Berlin". Quote: "gold_alignment": "ok" (001_layers_baseline/run-latest/output-gemma-2-9b.json:845); "first_id": 12514 (001_layers_baseline/run-latest/output-gemma-2-9b.json:1168).
- Negative control present: control_prompt + control_summary. Quote: "control_prompt" … (001_layers_baseline/run-latest/output-gemma-2-9b.json:1033); "control_summary": { "first_control_margin_pos": 18, "max_control_margin": 0.8677… } (001_layers_baseline/run-latest/output-gemma-2-9b.json:1061–1064).
- Ablation present: ablation_summary and both prompt_variant rows exist. Quote: ablation_summary { "L_copy_orig": 0, "L_sem_orig": 42, "L_copy_nf": 0, "L_sem_nf": 42 } (001_layers_baseline/run-latest/output-gemma-2-9b.json:1091–1096); pure CSV contains pos,no_filler rows (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:50–54).
- Summary indices (bits): first_kl_below_0.5 = null; first_kl_below_1.0 = null; rank milestones at 42. Quote: "first_kl_below_0.5": null, "first_kl_below_1.0": null, "first_rank_le_1": 42, "first_rank_le_5": 42, "first_rank_le_10": 42 (001_layers_baseline/run-latest/output-gemma-2-9b.json:840–844).
- Last‑layer head calibration: final lens vs final head KL = 1.01 bits with warning. Quote: "kl_to_final_bits": 1.0129…, "top1_agree": true, "p_top1_lens": 0.9298 vs "p_top1_model": 0.3943, "temp_est": 2.6102, "kl_after_temp_bits": 0.3499, "warn_high_last_layer_kl": true (001_layers_baseline/run-latest/output-gemma-2-9b.json:846–864). CSV final row shows kl_to_final_bits ≈ 1.0129 at L42 (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:49).
- Lens sanity: raw-vs-norm check mode = sample; lens_artifact_risk = high; max_kl_norm_vs_raw_bits = 12.9056; first_norm_only_semantic_layer = null. Quote: "mode": "sample" (001_layers_baseline/run-latest/output-gemma-2-9b.json:962); "lens_artifact_risk": "high", "max_kl_norm_vs_raw_bits": 12.9055, "first_norm_only_semantic_layer": null (001_layers_baseline/run-latest/output-gemma-2-9b.json:1008–1016).
- Copy‑collapse flag check (first firing): layer = 0, top‑1 = “ simply”, p1 = 0.9999993; top‑2 = “simply”, p2 ≈ 7.7e‑07 — ✓ rule satisfied. Quote: L0 row with copy_collapse=True (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2).

**Quantitative Findings**
Main table (pos, orig). One row per layer: “L i — entropy X bits, top‑1 ‘token’”. Bold marks L_semantic.

| Layer | Entropy (bits) | Top‑1 token |
|---|---:|---|
| L 0 | 0.000017 | ‘ simply’ |
| L 1 | 0.000000 | ‘ simply’ |
| L 2 | 0.000031 | ‘ simply’ |
| L 3 | 0.000430 | ‘ simply’ |
| L 4 | 0.002116 | ‘ simply’ |
| L 5 | 0.002333 | ‘ simply’ |
| L 6 | 0.127902 | ‘ simply’ |
| L 7 | 0.033569 | ‘ simply’ |
| L 8 | 0.098417 | ‘ simply’ |
| L 9 | 0.102087 | ‘ simply’ |
| L 10 | 0.281391 | ‘ simply’ |
| L 11 | 0.333046 | ‘ simply’ |
| L 12 | 0.109330 | ‘ simply’ |
| L 13 | 0.137400 | ‘ simply’ |
| L 14 | 0.165772 | ‘ simply’ |
| L 15 | 0.734873 | ‘ simply’ |
| L 16 | 3.568274 | ‘ simply’ |
| L 17 | 3.099445 | ‘ simply’ |
| L 18 | 3.336717 | ‘ simply’ |
| L 19 | 1.382336 | ‘ simply’ |
| L 20 | 3.163441 | ‘ simply’ |
| L 21 | 1.866495 | ‘ simply’ |
| L 22 | 2.190102 | ‘ simply’ |
| L 23 | 3.181111 | ‘ simply’ |
| L 24 | 1.107039 | ‘ simply’ |
| L 25 | 2.118879 | ‘ the’ |
| L 26 | 2.371327 | ‘ the’ |
| L 27 | 1.842460 | ‘ the’ |
| L 28 | 1.226664 | ‘ "’ |
| L 29 | 0.315988 | ‘ "’ |
| L 30 | 0.134063 | ‘ "’ |
| L 31 | 0.046090 | ‘ "’ |
| L 32 | 0.062538 | ‘ "’ |
| L 33 | 0.042715 | ‘ "’ |
| L 34 | 0.090030 | ‘ "’ |
| L 35 | 0.023370 | ‘ "’ |
| L 36 | 0.074091 | ‘ "’ |
| L 37 | 0.082534 | ‘ "’ |
| L 38 | 0.033455 | ‘ "’ |
| L 39 | 0.046899 | ‘ "’ |
| L 40 | 0.036154 | ‘ "’ |
| L 41 | 0.176738 | ‘ "’ |
| **L 42** | 0.370067 | ‘ Berlin’ |

Notes on selected rows (CSV quotes):
- L42 (semantic): “ Berlin”, p_top1 = 0.9298; kl_to_final_bits = 1.0129; cos_to_final = 0.9993 (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:49).
- L31: top‑1 is a quote token, entropy ≈ 0.0461 bits; KL to final ≈ 1.8722 bits; cos_to_final ≈ 0.3811 (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:38).

Control margin (JSON): first_control_margin_pos = 18; max_control_margin = 0.8677 (001_layers_baseline/run-latest/output-gemma-2-9b.json:1061–1064).

Ablation (no‑filler): L_copy_orig = 0; L_sem_orig = 42; L_copy_nf = 0; L_sem_nf = 42; ΔL_copy = 0; ΔL_sem = 0 (001_layers_baseline/run-latest/output-gemma-2-9b.json:1091–1096).

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = 0.000017 − 0.370067 ≈ −0.37005.

Confidence milestones (pure CSV):
- p_top1 > 0.30 at layer 0; p_top1 > 0.60 at layer 0; final-layer p_top1 = 0.9298 (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2,49).

Rank milestones (diagnostics):
- rank ≤ 10 at layer 42; rank ≤ 5 at layer 42; rank ≤ 1 at layer 42 (001_layers_baseline/run-latest/output-gemma-2-9b.json:842–844).

KL milestones (diagnostics + CSV):
- first_kl_below_1.0 = null; first_kl_below_0.5 = null (001_layers_baseline/run-latest/output-gemma-2-9b.json:840–841). KL decreases late: L0 ≈ 14.39; L10 ≈ 14.94; L21 ≈ 15.17; L31 ≈ 1.87; final (L42) ≈ 1.01 bits (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2,12,28,38,45,49). Final KL ≠ 0 — treat final p_top1 as family‑specific; prefer ranks.

Cosine milestones (pure CSV):
- first cos_to_final ≥ 0.2 at L1; ≥ 0.4 at L42; ≥ 0.6 at L42; final cos_to_final = 0.9993 (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:3,49).

Prism Sidecar Analysis
- Presence: compatible = true; k = 512; layers [embed, 9, 20, 30] (001_layers_baseline/run-latest/output-gemma-2-9b.json:819–831).
- Early-depth stability (KL P_layer||P_final): baseline vs Prism
  - L0: 14.39 vs 28.48; L10: 14.94 vs 26.57; L21: 15.17 vs 25.51; L31: 1.87 vs 26.01; L42: 1.01 vs 28.73 (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2,12,28,38,49; 001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token-prism.csv:2,12,23,33,44).
- Prism rank milestones (pure CSV): no rank ≤10/5/1 achieved at any layer (answer_rank remains large; e.g., L42 answer_rank ≈ 97202) (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token-prism.csv:44).
- Top‑1 agreement to final: none at sampled depths; neither baseline nor Prism matches final head early.
- Cosine drift: baseline cos_to_final ≈ 0.381 at L31 vs Prism ≈ −0.012 at L31; Prism never approaches final (final cos ≈ 0.0867) (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:38; 001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token-prism.csv:33,44).
- Copy flags: baseline copy_collapse True at L0–L3; Prism copy_collapse False at early layers (e.g., L0 False) (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2–5; 001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token-prism.csv:2). Prism’s transform breaks token‑ID locality, so copy detection does not apply cleanly.
- Verdict: Regressive (KL increases ≥10–15 bits relative to baseline early/mid; no earlier rank milestones).

**Qualitative Patterns & Anomalies**
The model exhibits a strong copy/filler reflex: the NEXT token remains “ simply” top‑1 with p>0.95 through early/mid layers (e.g., L11 p≈0.953) before shifting to punctuation near L31, and only at L42 does the answer emerge, top‑1 “ Berlin” p≈0.930 (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:11,38,49). KL drops sharply only late (L31 ≈ 1.87 bits), while cosine drifts up earlier (L31 cos≈0.381), indicating “early direction, late calibration.” Final‑head calibration mismatch is pronounced for this family: lens p_top1 0.9298 vs model p_top1 0.3943; temp_est ≈ 2.61; kl_after_temp ≈ 0.35 bits; warn_high_last_layer_kl = true (001_layers_baseline/run-latest/output-gemma-2-9b.json:846–864). Raw‑vs‑norm check flags high lens‑artifact risk (max_kl_norm_vs_raw_bits ≈ 12.91; sample mode), so pre‑final “early semantics” should be treated cautiously (001_layers_baseline/run-latest/output-gemma-2-9b.json:1008–1016).

Negative control (“Berlin is the capital of”): top‑5 = [“ Germany” 0.8766, “ the” 0.0699, “ modern” 0.0077, “ a” 0.0053, “ ” 0.0034]. Berlin still appears at rank 9 with p≈0.00187 — semantic leakage (001_layers_baseline/run-latest/output-gemma-2-9b.json:7–26).

Important‑word trajectory (records CSV; IMPORTANT_WORDS = ["Germany", "Berlin", "capital", "Answer", "word", "simply"]): the grammatical cue “ simply” dominates NEXT early (L0–L15) and fades by L31 (punctuation rises) before the answer appears at L42 (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2–16,38,49). Within the context, important tokens like “ Germany” are consistently salient at their positions (e.g., L0 pos=13: top‑1 “ Germany”, p≈0.999998) (001_layers_baseline/run-latest/output-gemma-2-9b-records.csv:15). No evidence of close German city distractors in final top‑k beyond “ BERLIN/berlin” variants (001_layers_baseline/run-latest/output-gemma-2-9b.json:920–952).

Instruction sensitivity: removing “simply” (“no_filler”) does not change collapse indices: L_copy_nf = 0; L_sem_nf = 42 (ΔL_sem = 0), suggesting negligible stylistic‑cue impact in this setup (001_layers_baseline/run-latest/output-gemma-2-9b.json:1091–1096).

Rest‑mass sanity: rest_mass remains tiny at/after semantic collapse (e.g., L31 ≈ 4.22e‑05; L42 ≈ 1.11e‑05) — no precision loss indicated (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:38,49).

Rotation vs amplification: cos_to_final rises modestly by mid‑stack (≈0.38 at L31) while KL remains >1 bit until late; final‑head KL ≈1 bit plus a known Gemma calibration gap (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:38–49; 001_layers_baseline/run-latest/output-gemma-2-9b.json:846–864). Prefer rank‑based statements.

Head calibration (final layer): warn_high_last_layer_kl = true; temp_est ≈ 2.61; kl_after_temp_bits ≈ 0.35; cfg_transform null (001_layers_baseline/run-latest/output-gemma-2-9b.json:846–864). This matches known Gemma family behavior.

Lens sanity: raw_lens_check.mode = sample; lens_artifact_risk = high; max_kl_norm_vs_raw_bits ≈ 12.91 (001_layers_baseline/run-latest/output-gemma-2-9b.json:962,1008–1016). Treat early semantics cautiously; rely on ranks and within‑model trends.

Temperature robustness: at T=0.1, Berlin rank 1 (p≈0.9809; entropy ≈ 0.137 bits); at T=2.0, Berlin rank 1 (p≈0.0893; entropy ≈ 9.00 bits) (001_layers_baseline/run-latest/output-gemma-2-9b.json:668–696,736–764).

Checklist
- RMS lens? ✓ (first_block_ln1_type = "RMSNorm") (001_layers_baseline/run-latest/output-gemma-2-9b.json:809–813)
- LayerNorm bias removed? n.a. (RMS model) (001_layers_baseline/run-latest/output-gemma-2-9b.json:811–813)
- Entropy rise at unembed? ✓ (final head entropy ≈ 2.94 bits vs lens L42 ≈ 0.37) (001_layers_baseline/run-latest/output-gemma-2-9b.json:869; 001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:49)
- FP32 un‑embed promoted? ✓ (unembed_dtype = "torch.float32"; use_fp32_unembed = false) (001_layers_baseline/run-latest/output-gemma-2-9b.json:809,808)
- Punctuation / markup anchoring? ✓ (quote tokens dominate L31–L41) (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:38–41)
- Copy‑reflex? ✓ (copy_collapse True at L0–L3) (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2–5)
- Grammatical filler anchoring? ✗ for NEXT (top‑1 is “simply”, not {is,the,a,of}) (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2–16)

**Limitations & Data Quirks**
- Final‑head calibration: kl_to_final_bits ≈ 1.01 and warn_high_last_layer_kl = true; treat final p_top1 as family‑specific and rely on ranks for cross‑model claims (001_layers_baseline/run-latest/output-gemma-2-9b.json:846–864).
- Lens artifact risk: “high” in sampled raw‑vs‑norm check; max_kl_norm_vs_raw_bits ≈ 12.91. Treat pre‑final “early semantics” cautiously (001_layers_baseline/run-latest/output-gemma-2-9b.json:1008–1016).
- Prism sidecar appears regressive for this model (substantially higher KL; no earlier rank milestones), so baseline lens is preferred for interpretation (see KL comparisons above).

**Model Fingerprint**
“Gemma‑2‑9B: collapse at L 42; final entropy 2.94 bits; ‘Berlin’ appears only at the last layer.”

---
Produced by OpenAI GPT-5
