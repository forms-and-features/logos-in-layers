# Evaluation Report: Qwen/Qwen3-14B

**Overview**
Qwen3-14B (40 layers). Run artifacts dated 20250923 (see `timestamp-20250923-1651`). This probe captures layer-wise next-token distributions under a norm lens, entropy/KL in bits, copy/semantic collapse, cosine-to-final trajectory, soft/strict copy flags, and control margins.

**Method Sanity‑Check**
Diagnostics confirm RMS norm lens with rotary positions: "use_norm_lens": true and "layer0_position_info": "token_only_rotary_model" (001_layers_baseline/run-latest/output-Qwen3-14B.json:807,816). The context prompt ends exactly with “called simply”: "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply" (001_layers_baseline/run-latest/output-Qwen3-14B.json:4). Copy/semantic indices and settings are present: "L_copy": null, "L_copy_H": 32, "L_semantic": 36, "copy_thresh": 0.95, "copy_window_k": 1, "copy_match_level": "id_subsequence" (001_layers_baseline/run-latest/output-Qwen3-14B.json:842–848). Soft detectors: "copy_soft_config": {"threshold": 0.5, "window_ks": [1, 2, 3], "extra_thresholds": []} and flag labels "copy_flag_columns": ["copy_strict@0.95","copy_soft_k1@0.5","copy_soft_k2@0.5","copy_soft_k3@0.5"] (001_layers_baseline/run-latest/output-Qwen3-14B.json:833–841,1077–1081). Gold alignment is ID-based and OK: "gold_alignment": "ok" (001_layers_baseline/run-latest/output-Qwen3-14B.json:898). Negative control is present with summary: {"first_control_margin_pos": 0, "max_control_margin": 0.9741542933185612} (001_layers_baseline/run-latest/output-Qwen3-14B.json:1091–1106). Ablation exists and both variants appear in CSV (pos/orig and pos/no_filler), with summary {"L_copy_orig": null, "L_sem_orig": 36, "L_copy_nf": null, "L_sem_nf": 36, "delta_L_copy": null, "delta_L_sem": 0} (001_layers_baseline/run-latest/output-Qwen3-14B.json:1083–1089). Summary indices: {"first_kl_below_0.5": 40, "first_kl_below_1.0": 40, "first_rank_le_1": 36, "first_rank_le_5": 33, "first_rank_le_10": 32} (001_layers_baseline/run-latest/output-Qwen3-14B.json:849–853). Units are bits for KL/entropy (field names “_bits”). Last‑layer head calibration is perfect: {"kl_to_final_bits": 0.0, "top1_agree": true, "p_top1_lens": 0.34514, "p_top1_model": 0.34514, "temp_est": 1.0, "warn_high_last_layer_kl": false} (001_layers_baseline/run-latest/output-Qwen3-14B.json:899–910). Lens sanity (raw vs norm): mode "sample" with summary {"lens_artifact_risk": "high", "max_kl_norm_vs_raw_bits": 17.6735, "first_norm_only_semantic_layer": null} (001_layers_baseline/run-latest/output-Qwen3-14B.json:1015,1072–1075). Copy-collapse flags in layers 0–3 are all False (strict τ=0.95, k=1; soft τ_soft=0.5, k∈{1,2,3}); no strict or soft copy firing observed in any layer (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:2–5).

**Quantitative Findings**
Main table (pos, orig; pure next-token):
- L 0 – entropy 17.212854 bits, top-1 '梳'
- L 1 – entropy 17.212021 bits, top-1 '地处'
- L 2 – entropy 17.211170 bits, top-1 '是一部'
- L 3 – entropy 17.209875 bits, top-1 'tics'
- L 4 – entropy 17.208380 bits, top-1 'tics'
- L 5 – entropy 17.207327 bits, top-1 '-minded'
- L 6 – entropy 17.205141 bits, top-1 '过去的'
- L 7 – entropy 17.186316 bits, top-1 '�'
- L 8 – entropy 17.179604 bits, top-1 '-minded'
- L 9 – entropy 17.187605 bits, top-1 '-minded'
- L 10 – entropy 17.169565 bits, top-1 ' (?)'
- L 11 – entropy 17.151134 bits, top-1 '时代的'
- L 12 – entropy 17.165318 bits, top-1 'といって'
- L 13 – entropy 17.115282 bits, top-1 ' nav'
- L 14 – entropy 17.140715 bits, top-1 ' nav'
- L 15 – entropy 17.148745 bits, top-1 '唿'
- L 16 – entropy 17.134632 bits, top-1 '闯'
- L 17 – entropy 17.137224 bits, top-1 '唿'
- L 18 – entropy 17.100914 bits, top-1 '____'
- L 19 – entropy 17.075287 bits, top-1 '____'
- L 20 – entropy 16.932322 bits, top-1 '____'
- L 21 – entropy 16.985991 bits, top-1 '年夜'
- L 22 – entropy 16.954144 bits, top-1 '年夜'
- L 23 – entropy 16.839663 bits, top-1 '____'
- L 24 – entropy 16.760223 bits, top-1 '____'
- L 25 – entropy 16.757845 bits, top-1 '年夜'
- L 26 – entropy 16.668522 bits, top-1 '____'
- L 27 – entropy 16.031609 bits, top-1 '____'
- L 28 – entropy 15.234417 bits, top-1 '____'
- L 29 – entropy 14.186926 bits, top-1 '这个名字'
- L 30 – entropy 7.789196 bits, top-1 '这个名字'
- L 31 – entropy 5.161718 bits, top-1 '____'
- L 32 – entropy 0.815953 bits, top-1 '____'
- L 33 – entropy 0.481331 bits, top-1 '____'
- L 34 – entropy 0.594809 bits, top-1 '____'
- L 35 – entropy 0.667881 bits, top-1 '____'
- L 36 – entropy 0.312212 bits, top-1 ' Berlin' — semantic
- L 37 – entropy 0.905816 bits, top-1 ' ____'
- L 38 – entropy 1.212060 bits, top-1 ' ____'
- L 39 – entropy 0.952112 bits, top-1 ' Berlin'
- L 40 – entropy 3.583520 bits, top-1 ' Berlin'

Control margin (JSON): first_control_margin_pos = 0; max_control_margin = 0.9741542933185612 (001_layers_baseline/run-latest/output-Qwen3-14B.json:1106).

Ablation (no-filler): L_copy_orig = null, L_sem_orig = 36; L_copy_nf = null, L_sem_nf = 36; ΔL_copy = null, ΔL_sem = 0 (001_layers_baseline/run-latest/output-Qwen3-14B.json:1083–1089).

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (L_copy = null). Soft ΔHk (k∈{1,2,3}) = n.a. (all L_copy_soft[k] = null).

Confidence milestones (pure CSV): p_top1 > 0.30 at layer 31; p_top1 > 0.60 at layer 32; final-layer p_top1 = 0.345 (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:34–42).

Rank milestones (JSON): rank ≤ 10 at layer 32; rank ≤ 5 at layer 33; rank ≤ 1 at layer 36 (001_layers_baseline/run-latest/output-Qwen3-14B.json:851–853).

KL milestones (JSON/CSV): first_kl_below_1.0 at layer 40; first_kl_below_0.5 at layer 40; KL decreases with depth and is ≈ 0 at final (001_layers_baseline/run-latest/output-Qwen3-14B.json:849–850,899; 001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:42).

Cosine milestones (pure CSV): first cos_to_final ≥ 0.2 at layer 5; ≥ 0.4 at layer 29; ≥ 0.6 at layer 36; final cos_to_final = 0.99999 (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:6,30,38,42).

Prism Sidecar Analysis
- Presence: prism sidecar compatible and present (001_layers_baseline/run-latest/output-Qwen3-14B.json:819–831). Baseline vs Prism KL at sampled depths (bits): L0 12.96 → 13.17; L10 12.91 → 13.17; L20 12.98 → 13.23; L30 12.73 → 13.44; L36 1.66 → 15.20; L40 0.00 → 14.84 (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:2,10,22,32,38,42; 001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token-prism.csv:1,10,22,32,38,42). Clear regression (higher KL across depths).
- Rank milestones (Prism): no layers reach rank ≤10/≤5/≤1 (answer_rank never ≤10) (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token-prism.csv:1–42). Delta vs baseline: later/never across all thresholds.
- Top‑1 agreement: Prism never aligns early with the final top‑1 (' Berlin'); e.g., at L40, top‑1 is 'ixon' (p≈0.00273) vs baseline final ' Berlin' (p≈0.34514) (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token-prism.csv:42; 001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:42).
- Cosine drift: Prism cos_to_final remains low throughout (final ≈ 0.0657) vs baseline rising early (≥0.2 by L5, ≥0.6 by L36), indicating strong regression in directional alignment (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token-prism.csv:42; 001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:6,38,42).
- Copy flags: copy_collapse and soft-copy flags remain False under Prism; no spurious flips (scan of pos/orig rows).
- Verdict: Regressive (KL increases and rank milestones are later/never with poor cosine alignment).

**Qualitative Patterns & Anomalies**
Negative control (“Berlin is the capital of”): top‑5 are “ Germany” (0.632), “ which” (0.2468), “ the” (0.0737), “ what” (0.0094), “ a” (0.0048) — no leakage of “Berlin” (001_layers_baseline/run-latest/output-Qwen3-14B.json:13–31). For the guidance variant “Germany’s capital city is called simply”, the model already places “ Berlin” at 0.8729 (001_layers_baseline/run-latest/output-Qwen3-14B.json:61–63).

Records and important‑word trajectory (SCRIPT IMPORTANT_WORDS = ["Germany", "Berlin", "capital", "Answer", "word", "simply"]). Around the NEXT position (pos=15), ‘Berlin’ strengthens late: at L34 pos=14 (‘ called’) Berlin is top‑2 (p≈0.309) and appears at pos=13 (p≈0.0259) and pos=15 (p≈0.0156) (001_layers_baseline/run-latest/output-Qwen3-14B-records.csv:640–641,639). By L36, Berlin is top‑1 at all three context positions (pos=13/14/15) with high confidence: “ Berlin, 0.9851” / “ Berlin, 0.9933” / “ Berlin, 0.9530” (001_layers_baseline/run-latest/output-Qwen3-14B-records.csv:708–710). Final layer keeps Berlin dominant across positions (e.g., pos=15: “ Berlin, 0.3451” with substantial distractor punctuation) (001_layers_baseline/run-latest/output-Qwen3-14B-records.csv:875).

Collapse‑layer index vs ‘one‑word’ instruction: ablation shows no shift (L_sem_orig = 36; L_sem_nf = 36; ΔL_sem = 0), indicating semantics are robust to removing “simply” (001_layers_baseline/run-latest/output-Qwen3-14B.json:1083–1089).

Rest‑mass sanity: after L_semantic, the maximum rest_mass is 0.236 at L40 (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:42), consistent with concentrated top‑k mass but some spread at the final head.

Rotation vs amplification: cos_to_final rises early (≥0.2 by L5; ≥0.6 by L36) while KL to final remains high until late (≤1 bit only at L40). This is an “early direction, late calibration” pattern: direction aligns well before the final head calibrates probabilities (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:6,38,42; 001_layers_baseline/run-latest/output-Qwen3-14B.json:849–850).

Head calibration (final layer): perfect agreement with the model head — {"kl_to_final_bits": 0.0, "top1_agree": true, "temp_est": 1.0, "kl_after_temp_bits": 0.0, "warn_high_last_layer_kl": false} (001_layers_baseline/run-latest/output-Qwen3-14B.json:899–910). Absolute probabilities at final are trustworthy within this model.

Lens sanity: raw‑vs‑norm check reports {"lens_artifact_risk": "high", "max_kl_norm_vs_raw_bits": 17.6735, "first_norm_only_semantic_layer": null} in sample mode (001_layers_baseline/run-latest/output-Qwen3-14B.json:1015,1072–1075). Treat any pre‑final “early semantics” cautiously and prefer rank milestones and within‑model trajectories.

Temperature robustness: at T=0.1, Berlin rank 1 (p = 0.9742; entropy 0.173 bits); at T=2.0, Berlin top‑1 p = 0.0363 (entropy 13.161 bits) (001_layers_baseline/run-latest/output-Qwen3-14B.json:670–678,737–744).

Checklist
- RMS lens? ✓ (RMSNorm; norm lens enabled)
- LayerNorm bias removed? n.a. (RMS model; no LN bias)
- Entropy rise at unembed? ✗ (entropy drops toward final; 17.21 → 3.58 bits)
- FP32 un-embed promoted? ✓ (unembed_dtype "torch.float32"; mixed-precision cast) (001_layers_baseline/run-latest/output-Qwen3-14B.json:811–813)
- Punctuation / markup anchoring? ✓ (late-stack ‘____’, quotes, punctuation dominate many layers)
- Copy-reflex? ✗ (no strict/soft copy flags in L0–3)
- Grammatical filler anchoring? ✗ (early top‑1 not dominated by {“is”, “the”, “a”, “of”})

**Limitations & Data Quirks**
- Raw‑vs‑norm lens differences use sampled checks; lens_artifact_risk is "high" with max_kl_norm_vs_raw_bits = 17.6735 — interpret any pre‑final semantics cautiously (001_layers_baseline/run-latest/output-Qwen3-14B.json:1015,1072–1075).
- KL is lens‑sensitive; here final KL≈0 so final probabilities are calibrated, but cross‑model comparisons should still prefer rank milestones and qualitative KL trends.
- Strict/soft copy indices are null; ΔH and soft ΔHk are not computable and we rely on rank milestones and cosine milestones instead.

**Model Fingerprint**
“Qwen3‑14B: collapse at L 36; final entropy 3.58 bits; ‘Berlin’ consolidates across positions by L 36 with early direction (cos≥0.2 by L5) but late calibration (KL<1 only at L40).”

---
Produced by OpenAI GPT-5
*Run executed on: 2025-09-23 16:51:10*
