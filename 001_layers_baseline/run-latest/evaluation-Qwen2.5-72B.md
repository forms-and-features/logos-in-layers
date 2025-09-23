# Evaluation Report: Qwen/Qwen2.5-72B

*Run executed on: 2025-09-23 16:51:10*
**1. Overview**
- Model: Qwen/Qwen2.5-72B (80 layers; pre_norm architecture) [output-Qwen2.5-72B.json:1007–1014]. The probe performs a layer-by-layer logit‑lens sweep using a normalization‑aware lens with FP32 unembedding.
- Captures entropy, top‑k, answer rank/probability, KL to final, cosine to final, copy/entropy collapse flags, control margins, and ablation (no‑filler) behavior. Diagnostics confirm Prism sidecar availability for calibration comparisons [output-Qwen2.5-72B.json:819–832].

**2. Method Sanity‑Check**
The run uses the norm lens with FP32 unembedding and RMSNorm at both ln1 and ln_final; positional info indicates a token‑only rotary model. Diagnostics and CSVs align on implementation flags and copy detectors.
- Lens and precision flags: “use_norm_lens: true; use_fp32_unembed: true; unembed_dtype: torch.float32; first_block_ln1_type: RMSNorm; final_ln_type: RMSNorm; layer0_position_info: token_only_rotary_model” [output-Qwen2.5-72B.json:807–816].
- Context ends with “called simply”: “context_prompt: ‘Give the city name only, plain text. The capital of Germany is called simply’” [output-Qwen2.5-72B.json:817].
- Copy detector configuration present and mirrored in CSV/JSON: “copy_flag_columns = [‘copy_strict@0.95’, ‘copy_soft_k1@0.5’, ‘copy_soft_k2@0.5’, ‘copy_soft_k3@0.5’]” [output-Qwen2.5-72B.json:1077–1081]; “copy_soft_config: threshold 0.5; window_ks [1,2,3]; extra_thresholds []” [output-Qwen2.5-72B.json:833–841].
- Strict copy rule setup: “copy_thresh: 0.95; copy_window_k: 1; copy_match_level: id_subsequence” [output-Qwen2.5-72B.json:846–848].
- Summary indices: “first_kl_below_0.5: 80; first_kl_below_1.0: 80; first_rank_le_1: 80; first_rank_le_5: 78; first_rank_le_10: 74” [output-Qwen2.5-72B.json:849–853]. Units for KL/entropy are bits (by construction; see `bits_entropy_from_logits` in SCRIPT and CSV columns).
- Last‑layer head calibration: “kl_to_final_bits: 0.000109…; top1_agree: true; p_top1_lens 0.3395 vs p_top1_model 0.3383; temp_est: 1.0; warn_high_last_layer_kl: false” [output-Qwen2.5-72B.json:899–917]. CSV final row also has kl_to_final_bits ≈ 0 (“…, 0.00010909087118927091, 1, 1.0000004768371582,”) [output-Qwen2.5-72B-pure-next-token.csv:138].
- Lens sanity (raw vs norm): “summary: lens_artifact_risk: ‘high’; max_kl_norm_vs_raw_bits: 19.9098949045; first_norm_only_semantic_layer: null” [output-Qwen2.5-72B.json:1071–1075]. Sample shows large raw‑vs‑norm mismatch at layer 61: “kl_norm_vs_raw_bits: 19.9098949 … p_top1_raw: 0.7918 vs p_top1_norm: 0.000434” [output-Qwen2.5-72B.json:1059–1064]. Treat pre‑final “early semantics” with caution; prefer rank milestones.
- Gold alignment is ID‑based and OK: “gold_alignment: ok” [output-Qwen2.5-72B.json:898]. “gold_answer.first_id: 19846 (‘ĠBerlin’)” [output-Qwen2.5-72B.json:1111–1119].
- Negative control present with summary: control prompt text and gold alignment [output-Qwen2.5-72B.json:1091–1105]; control margins: “first_control_margin_pos: 0; max_control_margin: 0.2070” [output-Qwen2.5-72B.json:1106–1109].
- Ablation summary exists (no‑filler): “L_copy_orig: null; L_sem_orig: 80; L_copy_nf: null; L_sem_nf: 80; delta_L_sem: 0” [output-Qwen2.5-72B.json:1083–1090]. Positive rows appear for both variants in pure CSV (e.g., ‘pos,no_filler,…’ [output-Qwen2.5-72B-pure-next-token.csv:139]). Main table below filters to `prompt_id = pos`, `prompt_variant = orig`.
- Early copy checks (layers 0–3): no strict or soft flags fire; all `copy_collapse`, `copy_strict@0.95`, and `copy_soft_k1@0.5` are False [output-Qwen2.5-72B-pure-next-token.csv:2–5].

Copy‑collapse flag check: none fired (strict τ=0.95, δ=0.10; soft τ_soft=0.5 at k∈{1,2,3}). ✓ rule satisfied (no spurious early copy).
Soft copy flags: no `copy_soft_k1@0.5` fired at any layer (earliest: n/a).

**3. Quantitative Findings**
- Per‑layer (pos/orig, pure next token):
  - L 0 – entropy 17.2142 bits, top‑1 ‘s’ [output-Qwen2.5-72B-pure-next-token.csv:2]
  - L 1 – entropy 17.2142 bits, top‑1 ‘下一篇’ [output-Qwen2.5-72B-pure-next-token.csv:3]
  - L 2 – entropy 17.1425 bits, top‑1 ‘ولوج’ [output-Qwen2.5-72B-pure-next-token.csv:4]
  - L 3 – entropy 17.0631 bits, top‑1 ‘شدد’ [output-Qwen2.5-72B-pure-next-token.csv:5]
  - L 4 – entropy 17.0891 bits, top‑1 ‘.myapplication’ [output-Qwen2.5-72B-pure-next-token.csv:6]
  - L 5 – entropy 17.0072 bits, top‑1 ‘ستحق’ [output-Qwen2.5-72B-pure-next-token.csv:7]
  - L 6 – entropy 17.0315 bits, top‑1 ‘.myapplication’ [output-Qwen2.5-72B-pure-next-token.csv:8]
  - L 7 – entropy 16.9372 bits, top‑1 ‘.myapplication’ [output-Qwen2.5-72B-pure-next-token.csv:9]
  - L 8 – entropy 16.7980 bits, top‑1 ‘.myapplication’ [output-Qwen2.5-72B-pure-next-token.csv:10]
  - L 9 – entropy 16.1203 bits, top‑1 ‘ستحق’ [output-Qwen2.5-72B-pure-next-token.csv:11]
  - L 10 – entropy 16.5008 bits, top‑1 ‘.myapplication’ [output-Qwen2.5-72B-pure-next-token.csv:12]
  - L 11 – entropy 16.7180 bits, top‑1 ‘.myapplication’ [output-Qwen2.5-72B-pure-next-token.csv:13]
  - L 12 – entropy 16.7781 bits, top‑1 ‘かもしれ’ [output-Qwen2.5-72B-pure-next-token.csv:14]
  - L 13 – entropy 16.6314 bits, top‑1 ‘かもしれ’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 14 – entropy 16.3595 bits, top‑1 ‘かもしれ’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 15 – entropy 16.5170 bits, top‑1 ‘のではない’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 16 – entropy 16.4908 bits, top‑1 ‘iéndo’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 17 – entropy 16.2127 bits, top‑1 ‘iéndo’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 18 – entropy 16.1093 bits, top‑1 ‘有期徒’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 19 – entropy 15.7574 bits, top‑1 ‘有期徒’ [output-Qwen2.5-72B-pure-next-token.csv:21]
  - L 20 – entropy 16.1290 bits, top‑1 ‘有期徒’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 21 – entropy 16.1558 bits, top‑1 ‘有期徒’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 22 – entropy 15.9799 bits, top‑1 ‘有期徒’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 23 – entropy 16.4015 bits, top‑1 ‘.myapplication’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 24 – entropy 15.9989 bits, top‑1 ‘iéndo’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 25 – entropy 15.3506 bits, top‑1 ‘hế’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 26 – entropy 15.9435 bits, top‑1 ‘iéndo’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 27 – entropy 15.7559 bits, top‑1 ‘iéndo’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 28 – entropy 15.7500 bits, top‑1 ‘.myapplication’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 29 – entropy 15.8849 bits, top‑1 ‘.myapplication’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 30 – entropy 16.1225 bits, top‑1 ‘.myapplication’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 31 – entropy 16.1700 bits, top‑1 ‘.myapplication’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 32 – entropy 16.1708 bits, top‑1 ‘.myapplication’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 33 – entropy 16.4191 bits, top‑1 ‘hế’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 34 – entropy 16.2001 bits, top‑1 ‘iéndo’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 35 – entropy 16.4550 bits, top‑1 ‘hế’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 36 – entropy 16.4078 bits, top‑1 ‘iéndo’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 37 – entropy 16.2100 bits, top‑1 ‘iéndo’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 38 – entropy 16.4904 bits, top‑1 ‘hế’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 39 – entropy 16.4177 bits, top‑1 ‘iéndo’ [output-Qwen2.5-72B-pure-next-token.csv:97]
  - L 40 – entropy 16.1916 bits, top‑1 ‘iéndo’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 41 – entropy 16.4652 bits, top‑1 ‘hế’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 42 – entropy 16.5948 bits, top‑1 ‘hế’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 43 – entropy 16.4974 bits, top‑1 ‘hế’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 44 – entropy 16.6553 bits, top‑1 ‘続きを読む’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 45 – entropy 16.8770 bits, top‑1 ‘国际在线’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 46 – entropy 17.0023 bits, top‑1 ‘国际在线’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 47 – entropy 17.0133 bits, top‑1 ‘主义思想’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 48 – entropy 17.0217 bits, top‑1 ‘主义思想’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 49 – entropy 17.0217 bits, top‑1 ‘ reuseIdentifier’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 50 – entropy 16.9678 bits, top‑1 ‘uckets’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 51 – entropy 16.9723 bits, top‑1 ‘ "’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 52 – entropy 17.0086 bits, top‑1 ‘"’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 53 – entropy 16.9266 bits, top‑1 ‘"’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 54 – entropy 16.9081 bits, top‑1 ‘"’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 55 – entropy 16.9423 bits, top‑1 ‘"’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 56 – entropy 16.9382 bits, top‑1 ‘"’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 57 – entropy 16.8408 bits, top‑1 ‘ "’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 58 – entropy 16.9148 bits, top‑1 ‘ "’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 59 – entropy 16.9201 bits, top‑1 ‘ "’ [output-Qwen2.5-72B-pure-next-token.csv:117]
  - L 60 – entropy 16.8861 bits, top‑1 ‘ '’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 61 – entropy 16.9030 bits, top‑1 ‘ '’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 62 – entropy 16.8336 bits, top‑1 ‘ "’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 63 – entropy 16.8908 bits, top‑1 ‘ "’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 64 – entropy 16.8947 bits, top‑1 ‘ "’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 65 – entropy 16.8689 bits, top‑1 ‘ "’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 66 – entropy 16.8994 bits, top‑1 ‘ "’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 67 – entropy 16.8932 bits, top‑1 ‘ "’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 68 – entropy 16.7786 bits, top‑1 ‘ "’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 69 – entropy 16.8758 bits, top‑1 ‘ "’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 70 – entropy 16.7866 bits, top‑1 ‘ "’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 71 – entropy 16.5046 bits, top‑1 ‘ "’ [output-Qwen2.5-72B-pure-next-token.csv]
  - L 72 – entropy 16.6499 bits, top‑1 ‘ "’ [output-Qwen2.5-72B-pure-next-token.csv:130]
  - L 73 – entropy 15.7867 bits, top‑1 ‘ "’ [output-Qwen2.5-72B-pure-next-token.csv:131]
  - L 74 – entropy 16.0809 bits, top‑1 ‘ "’ [output-Qwen2.5-72B-pure-next-token.csv:132]
  - L 75 – entropy 13.3499 bits, top‑1 ‘ "’ [output-Qwen2.5-72B-pure-next-token.csv:133]
  - L 76 – entropy 14.7428 bits, top‑1 ‘ "’ [output-Qwen2.5-72B-pure-next-token.csv:134]
  - L 77 – entropy 10.8478 bits, top‑1 ‘ "’ [output-Qwen2.5-72B-pure-next-token.csv:135]
  - L 78 – entropy 15.3978 bits, top‑1 ‘ "’ [output-Qwen2.5-72B-pure-next-token.csv:136]
  - L 79 – entropy 16.6656 bits, top‑1 ‘ "’ [output-Qwen2.5-72B-pure-next-token.csv:137]
  - **L 80 – entropy 4.1158 bits, top‑1 ‘Berlin’** [output-Qwen2.5-72B-pure-next-token.csv:138]

Control margin (JSON): first_control_margin_pos = 0; max_control_margin = 0.2070 [output-Qwen2.5-72B.json:1106–1109].

Ablation (no‑filler): L_copy_orig = null; L_sem_orig = 80; L_copy_nf = null; L_sem_nf = 80; ΔL_copy = null; ΔL_sem = 0 [output-Qwen2.5-72B.json:1083–1090]. Interpretation: stylistic removal (“simply”) does not shift semantic collapse.

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n/a (L_copy = null). Soft ΔHk (k ∈ {1,2,3}) = n/a (no soft copy)

Confidence milestones (pure CSV):
- p_top1 > 0.30 at layer 77; p_top1 > 0.60 not reached; final p_top1 = 0.3395 [output-Qwen2.5-72B-pure-next-token.csv:135,138].

Rank milestones (diagnostics):
- rank ≤ 10 at L 74; rank ≤ 5 at L 78; rank ≤ 1 at L 80 [output-Qwen2.5-72B.json:851–853].

KL milestones (diagnostics):
- first_kl_below_1.0 at L 80; first_kl_below_0.5 at L 80 [output-Qwen2.5-72B.json:849–850]. KL decreases toward ≈ 0 at final; CSV final shows 0.000109 bits [output-Qwen2.5-72B-pure-next-token.csv:138].

Cosine milestones (pure CSV):
- first cos_to_final ≥ 0.2 at L 0; ≥ 0.4 at L 0; ≥ 0.6 at L 53; final cos_to_final = 1.0000 [output-Qwen2.5-72B-pure-next-token.csv:2,53,138].

Prism Sidecar Analysis
- Presence: compatible = true (k=512; layers [embed,19,39,59]) [output-Qwen2.5-72B.json:819–832].
- Early-depth stability (KL to final, bits):
  - L≈0: baseline 9.391 vs Prism 9.494 [output-Qwen2.5-72B-pure-next-token.csv:2; output-Qwen2.5-72B-pure-next-token-prism.csv:2–4].
  - L≈19: baseline 13.205 vs Prism 9.404 [output-Qwen2.5-72B-pure-next-token.csv:21; output-Qwen2.5-72B-pure-next-token-prism.csv].
  - L≈39: baseline 11.909 vs Prism 9.593 [output-Qwen2.5-72B-pure-next-token.csv:97; output-Qwen2.5-72B-pure-next-token-prism.csv].
  - L≈59: baseline 9.116 vs Prism 9.597 [output-Qwen2.5-72B-pure-next-token.csv:117; output-Qwen2.5-72B-pure-next-token-prism.csv].
- Rank milestones (Prism pure CSV): first_rank_le_{10,5,1} not reached at sampled depths (no earlier improvement vs baseline).
- Top‑1 agreement at sampled depths: both disagree with final (‘Berlin’) at all samples; e.g., baseline L59 top‑1 ‘ "’ vs Prism ‘贱’ [output-Qwen2.5-72B-pure-next-token.csv:117; output-Qwen2.5-72B-pure-next-token-prism.csv].
- Cosine drift: baseline cos ~0.498 at L59 vs Prism ~−0.101 (earlier stabilization not observed) [same lines].
- Copy flags: no flips; no copy flags fire under Prism (scan shows none).
- Verdict: Neutral — Prism reduces KL by >0.5 bits at some mid‑depths (19/39) but does not improve rank milestones or early top‑1 agreement.

**4. Qualitative Patterns & Anomalies**
The model exhibits a late semantic snap: ‘Berlin’ first appears in the top‑k only at L 78 (p≈0.0033) and reaches top‑1 at L 80 (p≈0.3395), consistent with rank milestones (≤5 at 78; 1 at 80). For example: “…, Berlin, 0.00334 …” (layer 78) [output-Qwen2.5-72B-pure-next-token.csv:136] and “Berlin, 0.33946” (layer 80) [output-Qwen2.5-72B-pure-next-token.csv:138]. Mid‑stack behavior is dominated by punctuation/quotes with ‘capital’ intermittently present (e.g., L 72–76 include ‘capital’ among top‑k) [output-Qwen2.5-72B-pure-next-token.csv:130–134]. This suggests late amplification of the correct token after a long punctuation‑anchored regime.

Negative control shows no semantic leakage toward ‘Berlin’: for “Berlin is the capital of”, top‑5 are “ Germany (0.7695), the (0.0864), which (0.0491), a (0.0125), what (0.0075)” [output-Qwen2.5-72B.json:14–31].

Records and important‑word trajectory (SCRIPT IMPORTANT_WORDS = [“Germany”, “Berlin”, “capital”, “Answer”, “word”, “simply”]):
- ‘capital’ becomes prominent in late‑mid layers (L 72–76), while ‘Berlin’ enters the top‑k only at L 78 and stabilizes as top‑1 at L 80 [output-Qwen2.5-72B-pure-next-token.csv:130–138]. Semantically related city tokens emerge in test prompts (e.g., “Frankfurt”, “Munich”) [output-Qwen2.5-72B.json:585–593], indicating the model retains a city prior before committing to the correct city token.

Instruction variations and ablation: Removing the stylistic filler (“simply”) does not shift semantics (ΔL_sem = 0) [output-Qwen2.5-72B.json:1083–1090]. Copy detection remains null in both variants, supporting that the collapse is semantic rather than copy‑reflex.

Rest‑mass sanity: Rest_mass falls into the final regime; max after L_semantic (≥80) is ≈0.298 [output-Qwen2.5-72B-pure-next-token.csv:138].

Rotation vs amplification: Cosine aligns early (cos ≥0.6 by L 53) while KL remains high until the end (first <1.0 at L 80), an “early direction, late calibration” pattern. Compare cos_to_final rises vs late KL drop and the sharp increase in p_answer near the end: “p_answer 0.00481” (L 77) [output-Qwen2.5-72B-pure-next-token.csv:135], “p_answer 0.33946” (L 80) [output-Qwen2.5-72B-pure-next-token.csv:138].

Head calibration (final layer): last_layer_consistency shows near‑zero final KL with top‑1 agreement and temp_est = 1.0 (no softcap/scale transforms applied) [output-Qwen2.5-72B.json:899–917]. No final‑head calibration warning.

Lens sanity: Raw‑vs‑norm sampling flags “lens_artifact_risk: high; max_kl_norm_vs_raw_bits: 19.9099; first_norm_only_semantic_layer: null” [output-Qwen2.5-72B.json:1071–1075]. A sample at L 61 shows strong raw/norm mismatch (norm p_top1 ≪ raw p_top1), underscoring that early “semantics” should be read via ranks and within‑model comparisons rather than absolute probabilities [output-Qwen2.5-72B.json:1059–1065].

Temperature robustness: At T = 0.1, Berlin rank 1 (p = 0.9526; entropy 0.2754 bits) [output-Qwen2.5-72B.json:670–676]. At T = 2.0, Berlin remains rank 1 but with p = 0.0162 and entropy rises to 15.0128 bits [output-Qwen2.5-72B.json:736–744].

Important‑word trajectory snapshot: “Berlin first appears in any top‑k at layer 78 (p ≈ 0.00334)” [output-Qwen2.5-72B-pure-next-token.csv:136]; “stabilises as top‑1 by layer 80 (p ≈ 0.3395)” [output-Qwen2.5-72B-pure-next-token.csv:138]; “capital repeatedly present in L 72–76 top‑k” [output-Qwen2.5-72B-pure-next-token.csv:130–134].

Checklist:
- RMS lens? ✓ [output-Qwen2.5-72B.json:810–811]
- LayerNorm bias removed? ✓ (not needed for RMS) [output-Qwen2.5-72B.json:812]
- Entropy rise at unembed? ✗ (final entropy 4.12 bits is low) [output-Qwen2.5-72B.json:921–927; output-Qwen2.5-72B-pure-next-token.csv:138]
- FP32 un‑embed promoted? ✓ [output-Qwen2.5-72B.json:808–809]
- Punctuation / markup anchoring? ✓ (quotes/punctuation top‑1 across L 51–79) [output-Qwen2.5-72B-pure-next-token.csv:51–79]
- Copy‑reflex? ✗ (no flags; layers 0–3 all False) [output-Qwen2.5-72B-pure-next-token.csv:2–5]
- Grammatical filler anchoring? ✗ (early top‑1 not {is, the, a, of}) [output-Qwen2.5-72B-pure-next-token.csv:2–6]

**5. Limitations & Data Quirks**
- Rest_mass ≈ 0.298 at final; not a fidelity metric, but suggests concentrated mass on top tokens; rely on KL/rank for calibration [output-Qwen2.5-72B-pure-next-token.csv:138].
- KL is lens‑sensitive; raw‑vs‑norm sampling indicates high artifact risk (max_kl_norm_vs_raw_bits ~ 19.91). Prefer rank milestones for pre‑final claims [output-Qwen2.5-72B.json:1071–1075].
- Raw‑lens checks are “sample” mode, so interpret as sampled sanity rather than exhaustive [output-Qwen2.5-72B.json:1015–1070].
- Final‑head calibration is good here (final KL ≈ 0); however, cross‑family comparisons should still prefer rank milestones and within‑model trends.

**6. Model Fingerprint**
“Qwen2.5‑72B: collapse at L 80; final entropy 4.12 bits; ‘Berlin’ emerges at L 78 and stabilizes at L 80; punctuation‑anchored mid‑stack.”

---
Produced by OpenAI GPT-5

