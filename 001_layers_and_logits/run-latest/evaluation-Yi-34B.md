**1. Overview**
- Model: 01-ai/Yi-34B (34B), pre-norm RMSNorm; run on 2025-08-30 20:43 CEST.
- Probe captures layer-wise entropy, answer rank/probability, KL-to-final, and cosine alignment under a norm lens; final head calibration is near-perfect and gold alignment is ID-level.

**2. Method Sanity‑Check**
Diagnostics confirm the intended RMSNorm lens and rotary positions: “use_norm_lens”: true [L807] and “layer0_position_info”: "token_only_rotary_model" [L816]. The context prompt ends exactly with “called simply” with no trailing space: "Give the city name only, plain text. The capital of Germany is called simply" [L817]. Copy detection configuration is present and matches the spec: "copy_thresh": 0.95, "copy_window_k": 1, "copy_match_level": "id_subsequence" [L823–L825]. Implementation flags are recorded (unembed dtype, LN types, bias handling): "unembed_dtype": "torch.float32" [L809], "first_block_ln1_type": "RMSNorm", "final_ln_type": "RMSNorm" [L810–L811], and "layernorm_bias_fix": "not_needed_rms_model" [L812]. Summary indices exist and are consistent: "first_rank_le_1": 44, "first_rank_le_5": 44, "first_rank_le_10": 43; "first_kl_below_0.5": 60, "first_kl_below_1.0": 60 [L826–L830]. Gold‑token alignment is ID‑based and OK: "gold_alignment": "ok" [L831]; the gold answer is Berlin with first_id 19616 [L1038–L1046]. Last‑layer head consistency exists and KL-to-final is ≈0 in both CSV and JSON: kl_to_final_bits = 0.000278… [row 63 in pure CSV] and "kl_to_final_bits": 0.0002783696, "top1_agree": true, "p_top1_lens": 0.5555 vs "p_top1_model": 0.5627, "temp_est": 1.0, "warn_high_last_layer_kl": false [L832–L851]. No copy-collapse fired in early layers: copy_collapse = False for layers 0–3 in the pure CSV (e.g., rows 2–5). Lens sanity sampling flags high norm-only artifact risk: "mode": "sample" [L949] and summary "first_norm_only_semantic_layer": 46, "max_kl_norm_vs_raw_bits": 80.57, "lens_artifact_risk": "high" [L1004–L1008].

Copy-collapse flag check: No row with copy_collapse = True. ✓ rule not triggered (no spurious fires observed).

**3. Quantitative Findings**
- L 0 – entropy 15.962 bits, top-1 ' Denote'
- L 1 – entropy 15.942 bits, top-1 '.'
- L 2 – entropy 15.932 bits, top-1 '.'
- L 3 – entropy 15.839 bits, top-1 'MTY'
- L 4 – entropy 15.826 bits, top-1 'MTY'
- L 5 – entropy 15.864 bits, top-1 'MTY'
- L 6 – entropy 15.829 bits, top-1 'MTQ'
- L 7 – entropy 15.862 bits, top-1 'MTY'
- L 8 – entropy 15.873 bits, top-1 '其特征是'
- L 9 – entropy 15.836 bits, top-1 '审理终结'
- L 10 – entropy 15.797 bits, top-1 '~\\\\'
- L 11 – entropy 15.702 bits, top-1 '~\\\\'
- L 12 – entropy 15.774 bits, top-1 '~\\\\'
- L 13 – entropy 15.784 bits, top-1 '其特征是'
- L 14 – entropy 15.739 bits, top-1 '其特征是'
- L 15 – entropy 15.753 bits, top-1 '其特征是'
- L 16 – entropy 15.714 bits, top-1 '其特征是'
- L 17 – entropy 15.714 bits, top-1 '其特征是'
- L 18 – entropy 15.716 bits, top-1 '其特征是'
- L 19 – entropy 15.696 bits, top-1 'ncase'
- L 20 – entropy 15.604 bits, top-1 'ncase'
- L 21 – entropy 15.609 bits, top-1 'ODM'
- L 22 – entropy 15.620 bits, top-1 'ODM'
- L 23 – entropy 15.602 bits, top-1 'ODM'
- L 24 – entropy 15.548 bits, top-1 'ODM'
- L 25 – entropy 15.567 bits, top-1 'ODM'
- L 26 – entropy 15.585 bits, top-1 'ODM'
- L 27 – entropy 15.227 bits, top-1 'ODM'
- L 28 – entropy 15.432 bits, top-1 'MTU'
- L 29 – entropy 15.467 bits, top-1 'ODM'
- L 30 – entropy 15.551 bits, top-1 'ODM'
- L 31 – entropy 15.531 bits, top-1 ' 版的'
- L 32 – entropy 15.455 bits, top-1 'MDM'
- L 33 – entropy 15.455 bits, top-1 'XFF'
- L 34 – entropy 15.477 bits, top-1 'XFF'
- L 35 – entropy 15.471 bits, top-1 'Mpc'
- L 36 – entropy 15.433 bits, top-1 'MDM'
- L 37 – entropy 15.454 bits, top-1 'MDM'
- L 38 – entropy 15.486 bits, top-1 'MDM'
- L 39 – entropy 15.504 bits, top-1 'MDM'
- L 40 – entropy 15.528 bits, top-1 'MDM'
- L 41 – entropy 15.519 bits, top-1 'MDM'
- L 42 – entropy 15.535 bits, top-1 'keV'
- L 43 – entropy 15.518 bits, top-1 ' "'
- **L 44 – entropy 15.327 bits, top-1 ' Berlin'**
- L 45 – entropy 15.293 bits, top-1 ' Berlin'
- L 46 – entropy 14.834 bits, top-1 ' Berlin'
- L 47 – entropy 14.731 bits, top-1 ' Berlin'
- L 48 – entropy 14.941 bits, top-1 ' Berlin'
- L 49 – entropy 14.696 bits, top-1 ' Berlin'
- L 50 – entropy 14.969 bits, top-1 ' Berlin'
- L 51 – entropy 14.539 bits, top-1 ' Berlin'
- L 52 – entropy 15.137 bits, top-1 ' Berlin'
- L 53 – entropy 14.870 bits, top-1 ' Berlin'
- L 54 – entropy 14.955 bits, top-1 ' Berlin'
- L 55 – entropy 14.932 bits, top-1 ' Berlin'
- L 56 – entropy 14.745 bits, top-1 ' Berlin'
- L 57 – entropy 14.748 bits, top-1 ' '
- L 58 – entropy 13.457 bits, top-1 ' '
- L 59 – entropy 7.191 bits, top-1 ' '
- L 60 – entropy 2.981 bits, top-1 ' Berlin'

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (no copy collapse; L_copy = null [L819]).

Confidence milestones (pure CSV):
- p_top1 > 0.30 at layer 60; p_top1 > 0.60 at n.a.; final-layer p_top1 = 0.5555 ("p_top1_lens": 0.5555) [row 63 in pure CSV; L835–L838].

Rank milestones (diagnostics):
- rank ≤ 10 at layer 43, rank ≤ 5 at 44, rank ≤ 1 at 44 [L828–L830].

KL milestones (diagnostics + trend):
- first_kl_below_1.0 at layer 60; first_kl_below_0.5 at 60 [L826–L827]. KL decreases with depth: e.g., 12.0076 bits at L0 [row 2 in pure CSV] to ≈0 at final (0.000278 bits) [row 63 in pure CSV; L833].

Cosine milestones (pure CSV):
- first cos_to_final ≥ 0.2 at layer 1 (cos = 0.3396) [row 3 in pure CSV]; ≥ 0.4 at layer 44 (cos = 0.4324) [row 46]; ≥ 0.6 at layer 51 (cos = 0.6143) [row 53]; final cos_to_final = ~1.0 [row 63].

**4. Qualitative Patterns & Anomalies**
Negative control is clean and does not leak the city: “Berlin is the capital of” → top‑5: “ Germany” (0.84), “ the” (0.054), “ which” (0.029), “ what” (0.012), “ Europe” (0.006) [L14–L31]. Semantic leakage not observed (Berlin absent).

Important‑word trajectory (records CSV) shows the expected answer emerging around mid/late stack alongside semantically related tokens. At L43 for the last context token (“simply”), ‘Berlin’ already appears in the top‑20 (0.00053) [L770]. At L44, ‘Berlin’ becomes top‑1 for the next token position (is_answer = True) with p = 0.00846 [row 46 in pure CSV], and related tokens (capital, Germany, Munich, Frankfurt) appear among the top‑k for nearby positions (e.g., L44 pos=14 “is”: ‘Berlin’ 0.01049; ‘capital’ 0.00365) [L786–L788]. Through L46–L51, Berlin’s probability rises (0.0345 → 0.0597) and cosine alignment strengthens (0.528 → 0.614) [rows 48, 53].

Rotation vs amplification: The direction to the final head is found early (cos_to_final ≈ 0.43 at L44) while KL-to-final is still large (11.41 bits) and p_answer is small (0.0085), indicating early direction with late calibration; by L51, cos ≥ 0.6 and p_answer ≈ 0.060 with KL dropping to 9.79 bits [row 46; row 53]. Final-head calibration is excellent (KL_to_final ≈ 0; top1_agree = true) [row 63; L832–L851].

Lens sanity: Dual‑lens sampling flags “norm‑only semantics” at L46 with high artifact risk: "first_norm_only_semantic_layer": 46; "max_kl_norm_vs_raw_bits": 80.57; "lens_artifact_risk": "high" [L1004–L1008]. Treat early semantics cautiously; prefer rank milestones and within‑model comparisons. A sample shows raw vs norm discrepancies at multiple depths (e.g., L31: p_top1_raw = 0.2397 vs p_top1_norm = 0.00087) [L978–L986].

Temperature robustness: At T = 0.1, Berlin rank 1 with p ≈ 0.9999996 and near‑zero entropy [L670–L687]; at T = 2.0, Berlin remains rank 1 with p ≈ 0.0488 and entropy ≈ 12.49 bits [L737–L747].

One‑word instruction ablation: Removing “simply” does not shift the collapse index; L_semantic remains 44 and L_copy remains null ("L_sem_orig": 44; "L_sem_nf": 44; "L_copy_orig": null; "L_copy_nf": null) [L1010–L1016].

Rest‑mass sanity: Rest_mass falls with depth and is largest immediately after L_semantic (0.9807 at L44) before collapsing to 0.1753 by the final layer [row 46; row 63]. This supports that precision accumulates late rather than being present early.

Checklist
- RMS lens?: ✓ (RMSNorm + use_norm_lens true) [L807, L810–L811]
- LayerNorm bias removed?: ✓ (not needed for RMS models) [L812]
- Entropy rise at unembed?: ✗ (entropy drops to 2.98 bits at final) [row 63]
- FP32 un‑embed promoted?: ✓ ("use_fp32_unembed": true; "unembed_dtype": "torch.float32") [L808–L809]
- Punctuation / markup anchoring?: ✓ early layers dominated by punctuation/markup (e.g., L1 top‑1 ‘.’) [row 3]
- Copy‑reflex?: ✗ (no copy_collapse in layers 0–3; all False) [rows 2–5]
- Grammatical filler anchoring?: ✗ (no early top‑1 in {is, the, a, of}) [rows 2–8]

**5. Limitations & Data Quirks**
- Raw‑vs‑norm differences are significant (lens_artifact_risk = high; first_norm_only_semantic_layer = 46) [L1004–L1008]; treat early semantics as lens‑dependent and use rank milestones for cross‑model claims.
- No copy collapse detected; ΔH relative to L_copy is n.a. (diagnostics L_copy = null) [L819].
- Final KL is near‑zero; absolute probabilities appear well calibrated here, but calibration is lens‑sensitive generally; prefer rank milestones for robustness.
- Raw‑lens check is “sample” mode [L949]; findings are sampled sanity, not exhaustive.

**6. Model Fingerprint**
“Yi‑34B: collapse at L 44; final entropy 2.98 bits; ‘Berlin’ becomes top‑1 mid‑stack and calibrates smoothly.”

---
Produced by OpenAI GPT-5

