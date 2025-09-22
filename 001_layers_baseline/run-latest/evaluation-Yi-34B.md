# Evaluation Report: 01-ai/Yi-34B

*Run executed on: 2025-09-21 17:13:26*
**Overview**
- Model: 01-ai/Yi-34B (Yi-34B). The probe captures layer-by-layer next-token distributions, entropy (bits), early copy/filler behavior, and where the gold answer emerges as top-1 under a calibrated norm lens. Negative control and a stylistic ablation are included; final-head calibration is checked.

**Method Sanity-Check**
- Context and lens: diagnostics confirm RMSNorm lens with FP32 unembed and rotary positions: “first_block_ln1_type: RMSNorm” and “layer0_position_info: token_only_rotary_model” (001_layers_baseline/run-latest/output-Yi-34B.json:810,816). The context prompt ends exactly with “called simply” (001_layers_baseline/run-latest/output-Yi-34B.json:4).
- Implementation flags present: “use_norm_lens: true”, “use_fp32_unembed: true”, “unembed_dtype: torch.float32” (001_layers_baseline/run-latest/output-Yi-34B.json:807–809).
- Copy rule present and configured: “copy_thresh: 0.95”, “copy_window_k: 1”, “copy_match_level: id_subsequence” (001_layers_baseline/run-latest/output-Yi-34B.json:837–839). Gold alignment: “ok” (001_layers_baseline/run-latest/output-Yi-34B.json:845). Negative control objects exist: control_prompt and control_summary (001_layers_baseline/run-latest/output-Yi-34B.json:1032–1050). Ablation summary exists (001_layers_baseline/run-latest/output-Yi-34B.json:1024–1031).
- Summary indices: first_kl_below_0.5 = 60; first_kl_below_1.0 = 60; first_rank_le_1 = 44; first_rank_le_5 = 44; first_rank_le_10 = 43 (001_layers_baseline/run-latest/output-Yi-34B.json:840–844). Units for KL/entropy are bits (column names and values in CSV; e.g., final kl_to_final_bits ≈ 0 in 001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:63).
- Last-layer head calibration: diagnostics.last_layer_consistency present and well-aligned (kl_to_final_bits = 0.000278 bits; top1_agree = true; temp_est = 1.0) (001_layers_baseline/run-latest/output-Yi-34B.json:846–865). Final CSV row corroborates near‑zero KL and p_top1 ~0.556 (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:63).
- Lens sanity (raw vs norm): mode = sample; summary shows lens_artifact_risk = “high”, max_kl_norm_vs_raw_bits = 80.57, and first_norm_only_semantic_layer = 46 (001_layers_baseline/run-latest/output-Yi-34B.json:1019–1021). Treat any pre‑final “early semantics” with caution and prefer rank milestones.
- Copy-collapse flag check (layers 0–3): no rows with copy_collapse = True; ✓ rule did not fire spuriously. First semantic row has is_answer = True at layer 44 (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:46).

**Quantitative Findings**
- Main table (pos, orig). One row per layer: L i — entropy X bits, top‑1 ‘token’. Bold marks the semantic layer (first is_answer = True).
  - L 0 — entropy 15.9623 bits, top‑1 ‘ Denote’
  - L 1 — entropy 15.9418 bits, top‑1 ‘.’
  - L 2 — entropy 15.9320 bits, top‑1 ‘.’
  - L 3 — entropy 15.8391 bits, top‑1 ‘MTY’
  - L 4 — entropy 15.8259 bits, top‑1 ‘MTY’
  - L 5 — entropy 15.8635 bits, top‑1 ‘MTY’
  - L 6 — entropy 15.8295 bits, top‑1 ‘MTQ’
  - L 7 — entropy 15.8623 bits, top‑1 ‘MTY’
  - L 8 — entropy 15.8726 bits, top‑1 ‘其特征是’
  - L 9 — entropy 15.8364 bits, top‑1 ‘审理终结’
  - L 10 — entropy 15.7970 bits, top‑1 ‘~\\’
  - L 11 — entropy 15.7015 bits, top‑1 ‘~\\’
  - L 12 — entropy 15.7740 bits, top‑1 ‘~\\’
  - L 13 — entropy 15.7837 bits, top‑1 ‘其特征是’
  - L 14 — entropy 15.7394 bits, top‑1 ‘其特征是’
  - L 15 — entropy 15.7531 bits, top‑1 ‘其特征是’
  - L 16 — entropy 15.7136 bits, top‑1 ‘其特征是’
  - L 17 — entropy 15.7137 bits, top‑1 ‘其特征是’
  - L 18 — entropy 15.7164 bits, top‑1 ‘其特征是’
  - L 19 — entropy 15.6961 bits, top‑1 ‘ncase’
  - L 20 — entropy 15.6040 bits, top‑1 ‘ncase’
  - L 21 — entropy 15.6094 bits, top‑1 ‘ODM’
  - L 22 — entropy 15.6202 bits, top‑1 ‘ODM’
  - L 23 — entropy 15.6019 bits, top‑1 ‘ODM’
  - L 24 — entropy 15.5478 bits, top‑1 ‘ODM’
  - L 25 — entropy 15.5670 bits, top‑1 ‘ODM’
  - L 26 — entropy 15.5855 bits, top‑1 ‘ODM’
  - L 27 — entropy 15.2274 bits, top‑1 ‘ODM’
  - L 28 — entropy 15.4318 bits, top‑1 ‘MTU’
  - L 29 — entropy 15.4668 bits, top‑1 ‘ODM’
  - L 30 — entropy 15.5507 bits, top‑1 ‘ODM’
  - L 31 — entropy 15.5312 bits, top‑1 ‘ 版的’
  - L 32 — entropy 15.4545 bits, top‑1 ‘MDM’
  - L 33 — entropy 15.4551 bits, top‑1 ‘XFF’
  - L 34 — entropy 15.4775 bits, top‑1 ‘XFF’
  - L 35 — entropy 15.4711 bits, top‑1 ‘Mpc’
  - L 36 — entropy 15.4330 bits, top‑1 ‘MDM’
  - L 37 — entropy 15.4536 bits, top‑1 ‘MDM’
  - L 38 — entropy 15.4855 bits, top‑1 ‘MDM’
  - L 39 — entropy 15.5043 bits, top‑1 ‘MDM’
  - L 40 — entropy 15.5278 bits, top‑1 ‘MDM’
  - L 41 — entropy 15.5192 bits, top‑1 ‘MDM’
  - L 42 — entropy 15.5349 bits, top‑1 ‘keV’
  - L 43 — entropy 15.5179 bits, top‑1 ‘ "’
  - **L 44 — entropy 15.3273 bits, top‑1 ‘Berlin’** (is_answer = True; 001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:46)
  - L 45 — entropy 15.2932 bits, top‑1 ‘Berlin’
  - L 46 — entropy 14.8338 bits, top‑1 ‘Berlin’
  - L 47 — entropy 14.7312 bits, top‑1 ‘Berlin’
  - L 48 — entropy 14.9413 bits, top‑1 ‘Berlin’
  - L 49 — entropy 14.6958 bits, top‑1 ‘Berlin’
  - L 50 — entropy 14.9692 bits, top‑1 ‘Berlin’
  - L 51 — entropy 14.5389 bits, top‑1 ‘Berlin’
  - L 52 — entropy 15.1373 bits, top‑1 ‘Berlin’
  - L 53 — entropy 14.8697 bits, top‑1 ‘Berlin’
  - L 54 — entropy 14.9553 bits, top‑1 ‘Berlin’
  - L 55 — entropy 14.9323 bits, top‑1 ‘Berlin’
  - L 56 — entropy 14.7454 bits, top‑1 ‘Berlin’
  - L 57 — entropy 14.7484 bits, top‑1 ‘ ’
  - L 58 — entropy 13.4571 bits, top‑1 ‘ ’
  - L 59 — entropy 7.1911 bits, top‑1 ‘ ’
  - L 60 — entropy 2.9812 bits, top‑1 ‘Berlin’ (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:63)

- Control margin (control_summary): first_control_margin_pos = 1; max_control_margin = 0.58357 (001_layers_baseline/run-latest/output-Yi-34B.json:1047–1050).

- Ablation (no‑filler). From ablation_summary: L_copy_orig = null; L_sem_orig = 44; L_copy_nf = null; L_sem_nf = 44; ΔL_copy = null; ΔL_sem = 0 (001_layers_baseline/run-latest/output-Yi-34B.json:1024–1031). Interpretation: no stylistic‑cue sensitivity detected for semantics; copy reflex not observed.

- ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (no copy collapse).

- Confidence milestones (pure CSV): p_top1 > 0.30 at layer 60; p_top1 > 0.60 not reached; final-layer p_top1 = 0.5555 (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:63).

- Rank milestones (diagnostics): rank ≤ 10 at layer 43, ≤ 5 at 44, ≤ 1 at 44 (001_layers_baseline/run-latest/output-Yi-34B.json:842–844).

- KL milestones (diagnostics): first_kl_below_1.0 at 60; first_kl_below_0.5 at 60 (001_layers_baseline/run-latest/output-Yi-34B.json:840–841). KL decreases with depth and is ≈ 0 in the final row, consistent with good last‑layer head calibration (see last_layer_consistency above).

- Cosine milestones (pure CSV): first cos_to_final ≥ 0.2 at layer 1; ≥ 0.4 at 44; ≥ 0.6 at 51; final cos_to_final = 1.0000 (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:2,46,51,63).

- Prism Sidecar Analysis (compatible = true; 001_layers_baseline/run-latest/output-Yi-34B.json:819–831)
  - Early-depth stability (KL to final, bits): baseline vs prism at selected depths
    - L0: 12.008 vs 12.132 (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:2; 001_layers_baseline/run-latest/output-Yi-34B-pure-next-token-prism.csv:2)
    - L15: 13.119 vs 12.181 (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:17; 001_layers_baseline/run-latest/output-Yi-34B-pure-next-token-prism.csv:17)
    - L30: 13.538 vs 12.175 (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:32; 001_layers_baseline/run-latest/output-Yi-34B-pure-next-token-prism.csv:32)
    - L45: 11.162 vs 12.172 (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:47; 001_layers_baseline/run-latest/output-Yi-34B-pure-next-token-prism.csv:47)
  - Rank milestones (prism pure CSV): no layer reaches rank ≤ {10,5,1}; all null (sidecar CSV scan).
  - Top‑1 agreement at sampled depths: differs at L0 (“ Denote” vs “xit”) and L45 (“ Berlin” vs “POSE”).
  - Cosine drift: baseline cos higher and turns positive earlier; prism cos stays near/under 0 through mid‑stack (e.g., L45: 0.496 vs −0.0028; same lines as above).
  - Copy flags: no spurious flips; copy_collapse remains False.
  - Verdict: Regressive (earlier KL drops at L15/L30, but worse at L45 and no rank‑milestone gains; sidecar not helpful here).

**Qualitative Patterns & Anomalies**
The gold token ‘Berlin’ emerges at L44 as top‑1 with low confidence (p ≈ 0.0085; 001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:46) and is steadily amplified to p ≈ 0.5555 at the final layer with cos_to_final ≈ 1.0 and near‑zero KL to final (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:63). This is a classic “rotation then amplification” trajectory: direction begins to align by mid‑late layers (cos ≥ 0.4 at L44), while calibration arrives late as KL compresses near the end. Given raw‑vs‑norm lens divergence (lens_artifact_risk = “high”; first_norm_only_semantic_layer = 46), pre‑final semantics should be read via ranks rather than raw probabilities (001_layers_baseline/run-latest/output-Yi-34B.json:1019–1021).

Negative control: “Berlin is the capital of” produces “ Germany” as top‑1 (0.84), with no “Berlin” in the top‑5 — no leakage (001_layers_baseline/run-latest/output-Yi-34B.json:10–16). For the parallel test prompts, “Germany’s capital city is called simply” yields “ Berlin” at 0.469 (strong semantic cue) (001_layers_baseline/run-latest/output-Yi-34B.json:57–63).

Important-word trajectory: “Berlin” first enters any top‑5 at L44 and persists across the late stack; “capital” appears from L43 through L52; “Germany” appears at L44 and L46 (presence scan of pure CSV). Example: “… (‘Berlin’, 0.0085)” (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:46) → “… (‘Berlin’, 0.5555)” (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:63). Rest_mass falls much later; max after L_semantic is ≈ 0.981 at L44, indicating very low top‑k coverage near the emergence layer (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:46).

Head calibration: last‑layer consistency is excellent (kl_to_final_bits ≈ 0.00028; top1_agree = true; temp_est = 1.0) (001_layers_baseline/run-latest/output-Yi-34B.json:846–865). No final‑head calibration warning is raised.

Lens sanity: raw_lens_check reports lens_artifact_risk = “high”, first_norm_only_semantic_layer = 46, max_kl_norm_vs_raw_bits ≈ 80.57 (001_layers_baseline/run-latest/output-Yi-34B.json:1019–1021). Accordingly, early semantics may be lens‑induced; within‑model ranks are preferred over absolute probabilities for claims before the final layer.

Temperature robustness: at T = 0.1, Berlin rank 1 (p = 0.9999996; entropy ≈ 7e‑06); at T = 2.0, Berlin rank 1 (p ≈ 0.0488; entropy ≈ 12.49) (001_layers_baseline/run-latest/output-Yi-34B.json:669–676,736–743).

Stylistic ablation: removing “simply” does not change the semantic layer (ΔL_sem = 0), indicating limited sensitivity to this guidance style on this prompt (001_layers_baseline/run-latest/output-Yi-34B.json:1024–1031).

Checklist
- RMS lens? ✓ (RMSNorm; use_norm_lens = true)
- LayerNorm bias removed? ✓ (“not_needed_rms_model”)
- Entropy rise at unembed? n.a.
- FP32 un-embed promoted? ✓ (“use_fp32_unembed: true”)
- Punctuation / markup anchoring? ✓ (early layers dominated by punctuation/symbols)
- Copy-reflex? ✗ (no copy_collapse = True in layers 0–3)
- Grammatical filler anchoring? ✗ (early top‑1s are symbols/garbage tokens, not {is, the, a, of})

**Limitations & Data Quirks**
- raw_lens_check indicates high lens_artifact_risk and a norm‑only semantic layer at 46; treat pre‑final “early semantics” cautiously and prefer rank milestones. KL is lens‑sensitive; non‑zero KL in earlier layers reflects calibration differences that shrink only near the head.
- Rest_mass remains high after L_semantic (max ≈ 0.981 at L44), reflecting low top‑k coverage near semantic emergence and advising caution when interpreting early probabilities; rest_mass is top‑k coverage only, not lens fidelity.
- raw_lens_check mode is “sample”; treat it as a sampled sanity check rather than exhaustive.

**Model Fingerprint**
- Yi‑34B: collapse at L 44; final entropy 2.98 bits; ‘Berlin’ is rank 1 late and amplified to p ≈ 0.56.

---
Produced by OpenAI GPT-5
