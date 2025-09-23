# Evaluation Report: Qwen/Qwen3-8B
**Overview**
- Model: Qwen/Qwen3-8B (8B params; 36 layers). The probe runs a layer-by-layer logit-lens sweep with normalization fixes, tracking entropy, KL to final, cosine to final logits, and answer rank (001_layers_baseline/run-latest/output-Qwen3-8B.json:955–960).
- Focus: Germany → Berlin one-token recall; outputs include per-layer CSVs (pure next-token and records) and diagnostics/ablation summaries.

**Method Sanity‑Check**
- Intended lens and positional encoding are active: “use_norm_lens”: true, RMSNorm model with alignment fixes, and rotary position info: “layer0_position_info”: “token_only_rotary_model” (001_layers_baseline/run-latest/output-Qwen3-8B.json:807–817). The context prompt ends exactly with “called simply” (001_layers_baseline/run-latest/output-Qwen3-8B.json:4).
- Copy rule configuration present with required fields: “copy_thresh”: 0.95, “copy_window_k”: 1, “copy_match_level”: “id_subsequence” (001_layers_baseline/run-latest/output-Qwen3-8B.json:837–839). Gold alignment is ok (001_layers_baseline/run-latest/output-Qwen3-8B.json:845).
- Summary indices available: first_kl_below_0.5 = 36; first_kl_below_1.0 = 36; first_rank_le_1 = 31; first_rank_le_5 = 29; first_rank_le_10 = 29 (001_layers_baseline/run-latest/output-Qwen3-8B.json:840–844). Units for KL/entropy are bits (CSV headers and fields use “entropy” and “kl_to_final_bits”).
- Last-layer head calibration is clean: JSON shows exact match (kl_to_final_bits = 0.0, top1_agree = true, temp_est = 1.0) (001_layers_baseline/run-latest/output-Qwen3-8B.json:846–854). Final CSV row also has kl_to_final_bits = 0.0 (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:38).
- Raw vs norm lens sampling: mode = “sample”; lens_artifact_risk = “high”; max_kl_norm_vs_raw_bits = 13.6049; no norm-only semantic layer flagged (001_layers_baseline/run-latest/output-Qwen3-8B.json:1018–1022). Prefer rank milestones over absolute probabilities for early layers.
- Negative control and ablation present: control_prompt and control_summary exist (001_layers_baseline/run-latest/output-Qwen3-8B.json:1032–1050). Ablation summary exists with L_copy_orig = null, L_sem_orig = 31, L_copy_nf = null, L_sem_nf = 31, delta_L_sem = 0 (001_layers_baseline/run-latest/output-Qwen3-8B.json:1024–1031). CSV has both prompt_variant = orig and no_filler for positive rows (e.g., 001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:39).
- Copy-collapse flag check (pos, orig): no row fires; layers 0–3 all show copy_collapse = False (e.g., 001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:2–5). ✓ rule did not fire spuriously.

**Quantitative Findings**
- Per-layer (pos, orig) summary — L i — entropy X bits, top‑1 ‘token’ (bold = semantic layer where is_answer = True):
  - L 0 — entropy 17.213 bits, top‑1 ‘CLICK’
  - L 1 — entropy 17.211 bits, top‑1 ‘apr’
  - L 2 — entropy 17.211 bits, top‑1 ‘财经’
  - L 3 — entropy 17.208 bits, top‑1 ‘-looking’
  - L 4 — entropy 17.206 bits, top‑1 ‘院子’
  - L 5 — entropy 17.204 bits, top‑1 ‘ (?)’
  - L 6 — entropy 17.196 bits, top‑1 ‘ly’
  - L 7 — entropy 17.146 bits, top‑1 ‘ (?)’
  - L 8 — entropy 17.132 bits, top‑1 ‘ (?)’
  - L 9 — entropy 17.119 bits, top‑1 ‘ (?)’
  - L 10 — entropy 17.020 bits, top‑1 ‘ (?)’
  - L 11 — entropy 17.128 bits, top‑1 ‘ifiable’
  - L 12 — entropy 17.117 bits, top‑1 ‘ifiable’
  - L 13 — entropy 17.126 bits, top‑1 ‘ifiable’
  - L 14 — entropy 17.053 bits, top‑1 ‘"’
  - L 15 — entropy 17.036 bits, top‑1 ‘"’
  - L 16 — entropy 16.913 bits, top‑1 ‘-’
  - L 17 — entropy 16.972 bits, top‑1 ‘-’
  - L 18 — entropy 16.911 bits, top‑1 ‘-’
  - L 19 — entropy 16.629 bits, top‑1 ‘ly’
  - L 20 — entropy 16.696 bits, top‑1 ‘_’
  - L 21 — entropy 16.408 bits, top‑1 ‘_’
  - L 22 — entropy 15.219 bits, top‑1 ‘ ______’
  - L 23 — entropy 15.220 bits, top‑1 ‘____’
  - L 24 — entropy 10.893 bits, top‑1 ‘____’
  - L 25 — entropy 13.454 bits, top‑1 ‘____’
  - L 26 — entropy 5.558 bits, top‑1 ‘____’
  - L 27 — entropy 4.344 bits, top‑1 ‘____’
  - L 28 — entropy 4.786 bits, top‑1 ‘____’
  - L 29 — entropy 1.778 bits, top‑1 ‘-minded’
  - L 30 — entropy 2.203 bits, top‑1 ‘ Germany’
  - L 31 — entropy 0.454 bits, top‑1 ‘ Berlin’
  - L 32 — entropy 1.037 bits, top‑1 ‘ German’
  - L 33 — entropy 0.988 bits, top‑1 ‘ Berlin’
  - L 34 — entropy 0.669 bits, top‑1 ‘ Berlin’
  - L 35 — entropy 2.494 bits, top‑1 ‘ Berlin’
  - L 36 — entropy 3.123 bits, top‑1 ‘ Berlin’
  (source: 001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:2–38)
- Semantic snap: L_semantic = 31 (is_answer = True) (001_layers_baseline/run-latest/output-Qwen3-8B.json:835; 001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:33). Gold answer: “Berlin” (001_layers_baseline/run-latest/output-Qwen3-8B.json:1051–1060).
- Control margin (ctl): first_control_margin_pos = 1; max_control_margin = 0.9999977350 (001_layers_baseline/run-latest/output-Qwen3-8B.json:1047–1050).
- Ablation (no‑filler): L_copy_orig = null; L_sem_orig = 31; L_copy_nf = null; L_sem_nf = 31; ΔL_copy = null; ΔL_sem = 0 (001_layers_baseline/run-latest/output-Qwen3-8B.json:1024–1031). Interpretation: removing “simply” does not shift semantic collapse.
- ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (L_copy = null).
- Confidence milestones (generic top‑1, not necessarily “Berlin”): p_top1 > 0.30 at layer 29; p_top1 > 0.60 at layer 29; final-layer p_top1 = 0.4334 (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:29,38).
- Rank milestones (diagnostics): rank ≤ 10 at layer 29; rank ≤ 5 at layer 29; rank ≤ 1 at layer 31 (001_layers_baseline/run-latest/output-Qwen3-8B.json:842–844).
- KL milestones (diagnostics): first_kl_below_1.0 at layer 36; first_kl_below_0.5 at layer 36 (001_layers_baseline/run-latest/output-Qwen3-8B.json:840–841). KL decreases with depth and is ≈ 0 at final (CSV and JSON agree).
- Cosine milestones (within-model): first cos_to_final ≥ 0.2 at L 36; ≥ 0.4 at L 36; ≥ 0.6 at L 36; final cos_to_final = 1.0000 (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:38).

Prism Sidecar Analysis
- Presence: prism_summary.compatible = true and present = true (001_layers_baseline/run-latest/output-Qwen3-8B.json:819–825). Proceeding with sidecar comparison.
- Early-depth stability (KL to final, bits):
  - Baseline L {0,9,18,27,36}: KL = {12.7879, 12.6138, 12.4124, 6.1433, 0.0}; Prism: {12.9411, 12.9775, 13.0004, 13.1762, 15.1765} (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:2,11,20,29,38; 001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token-prism.csv:2,11,20,29,38).
  - KL increases under Prism and remains high at final, indicating miscalibration relative to baseline final.
- Rank milestones (Prism): first_rank_le_{10,5,1} not reached (answer_rank stays >10, including final layer answer_rank = 51431) (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token-prism.csv:38).
- Top‑1 agreement: No early layers where Prism’s top‑1 matches the final answer; baseline also disagrees until late (see above rows).
- Cosine drift: Prism cos_to_final at sampled depths {−0.096, −0.086, −0.117, −0.195, 0.037} is lower than baseline and never stabilizes (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token-prism.csv:2,11,20,29,38).
- Copy flags: copy_collapse remains False at layers 0–3 under both decoders (e.g., 001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token-prism.csv:2–5).
- Verdict: Regressive (higher KL throughout; no improvement in rank milestones).

**Qualitative Patterns & Anomalies**
The final head is well-calibrated: JSON last_layer_consistency shows kl_to_final_bits = 0.0 with top1_agree = true and temp_est = 1.0 (001_layers_baseline/run-latest/output-Qwen3-8B.json:846–854). Raw‑vs‑norm lens sampling indicates high lens_artifact_risk (max_kl_norm_vs_raw_bits = 13.60), so pre‑final “early semantics” should be treated cautiously and rank milestones preferred (001_layers_baseline/run-latest/output-Qwen3-8B.json:1018–1022). Negative control “Berlin is the capital of” yields top‑5: “ Germany, 0.7286; which, 0.2207; the, 0.0237; what, 0.0114; __, 0.0023” (001_layers_baseline/run-latest/output-Qwen3-8B.json:433–451). Semantic leakage: Berlin appears rank ~9 in the test prompt top‑k: “ Berlin, 0.00046” (001_layers_baseline/run-latest/output-Qwen3-8B.json:46–48).

Records evolution (important words). The script tags important words ["Germany", "Berlin", "capital", "Answer", "word", "simply"] (001_layers_baseline/run.py:334). In the pure next‑token CSV, “Berlin” first becomes top‑1 at L 31 and remains top‑1 at multiple late layers: “(layer 31, token = ‘ Berlin’, p = 0.936)” [row 33 in CSV]; “(layer 34, token = ‘ Berlin’, p = 0.837)” [row 36]; final “(layer 36, token = ‘ Berlin’, p = 0.433)” [row 38] (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:33,36,38). Prior layers show generic/punctuation or filler tokens dominating the top‑1 (rows 2–30).

Rest‑mass sanity. Rest_mass falls after the semantic snap; the maximum after L_semantic is 0.1754 at layer 36, suggesting adequate top‑k coverage and no precision loss spike (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:38).

Rotation vs amplification. KL_to_final decreases steadily and reaches ≈ 0 only at the final layer, while cos_to_final only reaches ≥ 0.2/0.4/0.6 at L36, indicating relatively late alignment of direction and calibration; once semantics lock in at L31, answer_rank improves to 1 while calibration continues to evolve (001_layers_baseline/run-latest/output-Qwen3-8B.json:840–844; 001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:33,38).

Temperature robustness (from test prompts). “Germany’s capital city is called simply” shows strong answer confidence at inference time: “ Berlin, 0.772” (001_layers_baseline/run-latest/output-Qwen3-8B.json:61–63). Variants with slightly different wording also rank Berlin highly (001_layers_baseline/run-latest/output-Qwen3-8B.json:108–110, 155–157).

Checklist
- RMS lens? ✓ (RMSNorm types; use_norm_lens = true) (001_layers_baseline/run-latest/output-Qwen3-8B.json:807–813).
- LayerNorm bias removed? ✓ not needed (RMS) (001_layers_baseline/run-latest/output-Qwen3-8B.json:812).
- Entropy rise at unembed? ✓ final entropy 3.123 bits (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:38).
- FP32 un-embed promoted? ✓ unembed_dtype torch.float32; mixed_precision_fix set (001_layers_baseline/run-latest/output-Qwen3-8B.json:809,815–816).
- Punctuation / markup anchoring? ✓ early layers dominated by punctuation/fillers (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:14–23).
- Copy-reflex? ✗ (no copy_collapse True in layers 0–3) (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:2–5).
- Grammatical filler anchoring? ✓ early layers top‑1 among {“-”, “_”, quotes} (rows 14–23 in pure CSV).

**Limitations & Data Quirks**
- Raw‑vs‑norm lens: sample mode with high lens_artifact_risk; max_kl_norm_vs_raw_bits = 13.60 (001_layers_baseline/run-latest/output-Qwen3-8B.json:1018–1022). Treat pre‑final semantics cautiously and prefer rank thresholds.
- Prism sidecar appears misaligned to the baseline final head: KL rises and answer_rank never reaches ≤ 10, even at final (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token-prism.csv:38). Use baseline metrics for conclusions.
- Rest_mass remains ≤ 0.18 after L_semantic; no evidence of top‑k truncation errors, but rest_mass is top‑k coverage only (not a lens fidelity metric).

**Model Fingerprint**
Qwen3‑8B: semantic collapse at L 31; final entropy 3.123 bits; “Berlin” stabilizes as top‑1 in late layers.

---
Produced by OpenAI GPT-5
*Run executed on: 2025-09-21 17:13:26*
