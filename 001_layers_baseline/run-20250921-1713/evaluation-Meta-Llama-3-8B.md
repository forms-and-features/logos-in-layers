# Evaluation Report: meta-llama/Meta-Llama-3-8B
**1. Overview**
Meta‑Llama‑3‑8B (32 layers), run 2025‑09‑21 (timestamp‑20250921‑1713). The probe logs layer‑by‑layer pure next‑token predictions with a normalization (RMS) lens, plus calibration diagnostics, ablation (no‑filler), a negative control, and a Prism sidecar for comparison.

**2. Method Sanity‑Check**
Diagnostics confirm the normalized residual lens and positional encoding handling: use_norm_lens = true and unembed_dtype = torch.float32; layer‑0 reports token‑only rotary positional info (pre‑norm RMS).
> "use_norm_lens": true; "unembed_dtype": "torch.float32" (and layer0_position_info = "token_only_rotary_model")  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:807,809,816

Context prompt ends with “called simply” (no trailing space): 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:4. Copy‑detection thresholds and rule are present: copy_thresh = 0.95, copy_window_k = 1, copy_match_level = "id_subsequence" 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:837,838,839. Gold alignment is ok 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:845. Negative control and summary are present 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1033,1047–1049. Ablation summary exists (L_sem_orig = 25; L_sem_nf = 25) 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1024–1031.

Summary indices (bits, ranks): first_kl_below_0.5 = 32; first_kl_below_1.0 = 32; first_rank_le_1 = 25; first_rank_le_5 = 25; first_rank_le_10 = 24 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:840–844. Last‑layer head calibration is clean: final kl_to_final_bits ≈ 0 in CSV (0.0 at L32) 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:36 and diagnostics.last_layer_consistency present with warn_high_last_layer_kl = false 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:846–864.

Lens sanity (raw vs norm): mode = sample; first_norm_only_semantic_layer = 25; max_kl_norm_vs_raw_bits = 0.0713; lens_artifact_risk = high 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1019–1021. Treat any pre‑final “early semantics” cautiously; prefer rank milestones over absolute probabilities.

Copy‑collapse flag check (pos/orig): no row from L0–L3 has copy_collapse = True (all False at layers 0–3) 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:2–5. ✓ rule did not fire spuriously.

**3. Quantitative Findings**
Layers table (pos, orig; entropy in bits; top‑1 token):

| Layer | Entropy | Top‑1 |
|---|---|---|
| L 0 | 16.96 | 'itzer' |
| L 1 | 16.94 | 'mente' |
| L 2 | 16.88 | 'mente' |
| L 3 | 16.89 | 'tones' |
| L 4 | 16.90 | 'interp' |
| L 5 | 16.87 | '�' |
| L 6 | 16.88 | 'tons' |
| L 7 | 16.88 | 'Exited' |
| L 8 | 16.86 | 'надлеж' |
| L 9 | 16.87 | 'biased' |
| L 10 | 16.85 | 'tons' |
| L 11 | 16.85 | 'tons' |
| L 12 | 16.88 | 'LEGAL' |
| L 13 | 16.84 | 'macros' |
| L 14 | 16.84 | 'tons' |
| L 15 | 16.85 | ' simply' |
| L 16 | 16.85 | ' simply' |
| L 17 | 16.85 | ' simply' |
| L 18 | 16.84 | ' simply' |
| L 19 | 16.84 | ' '' |
| L 20 | 16.83 | ' '' |
| L 21 | 16.83 | ' '' |
| L 22 | 16.83 | 'tons' |
| L 23 | 16.83 | 'tons' |
| L 24 | 16.83 | ' capital' |
| **L 25** | 16.81 | ' Berlin' |
| L 26 | 16.83 | ' Berlin' |
| L 27 | 16.82 | ' Berlin' |
| L 28 | 16.82 | ' Berlin' |
| L 29 | 16.80 | ' Berlin' |
| L 30 | 16.79 | ' Berlin' |
| L 31 | 16.84 | ':' |
| L 32 | 2.96 | ' Berlin' |

Gold answer string is “Berlin” 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1052–1060. L_semantic = 25 (first is_answer = True) 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:833–835,842–844; see also CSV row with “ Berlin”, p_answer = 0.0001323 at L25 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:29.

Control margin (JSON control_summary): first_control_margin_pos = 0; max_control_margin = 0.5186 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1048–1049.

Ablation (no‑filler): L_copy_orig = null; L_sem_orig = 25; L_copy_nf = null; L_sem_nf = 25; ΔL_copy = null; ΔL_sem = 0 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1066–1073. No shift under ablation.

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (L_copy = null; rely on rank milestones).

Confidence milestones (pure CSV):
- p_top1 > 0.30 at layer 32; p_top1 > 0.60 not reached; final‑layer p_top1 = 0.5202 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:36.

Rank milestones (diagnostics): rank ≤ 10 at L24, ≤ 5 at L25, ≤ 1 at L25 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:842–844.

KL milestones (diagnostics): first_kl_below_1.0 at L32; first_kl_below_0.5 at L32 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:840–841. KL decreases with depth and is ≈ 0 at final (CSV L32 kl_to_final_bits = 0.0) 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:36.

Cosine milestones (pure CSV): first cos_to_final ≥ 0.2 at L20 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:24; ≥ 0.4 at L30 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:34; ≥ 0.6 at L32 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:36; final cos_to_final = ~1.0 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:36.

Prism Sidecar Analysis
- Presence: prism_summary.compatible = true; k = 512; layers = [embed, 7, 15, 23] 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:820–832.
- Early‑depth stability (KL to final, bits): baseline vs Prism — L0: 11.57 vs 12.28; L7: 11.64 vs 18.16; L15: 11.56 vs 18.67; L23: 11.39 vs 20.07 (higher is worse) 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:2,9,19,27; 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token-prism.csv:2,9,17,25.
- Rank milestones (Prism pure CSV): first_rank_le_{10,5,1} not achieved (no layers with answer_rank ≤ 10) 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token-prism.csv:26–34.
- Top‑1 agreement: no notable Prism→final agreements where baseline disagreed at sampled depths; Prism top‑1s are unrelated tokens (e.g., “ Gro” at L32) 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token-prism.csv:34.
- Cosine drift: Prism cos_to_final is small/negative mid‑stack (e.g., L23 cos ≈ −0.11) while baseline is positive and rising (≈ 0.24 at L23) 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token-prism.csv:23; 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:33.
- Copy flags: copy_collapse remains False in both baselines and Prism (no flips) 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:2–5,29–36; 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token-prism.csv:26–34.
- Verdict: Regressive (KL increases markedly and rank milestones do not improve).

**4. Qualitative Patterns & Anomalies**
Negative control shows clean behavior: for “Berlin is the capital of”, the top‑5 are [“ Germany”, 0.896], [“ the”, 0.052], [“ and”, 0.008], [“ germany”, 0.00336], [“ modern”, 0.00297]; “ Berlin” is rank 6 (p ≈ 0.00289) — semantic leakage: Berlin rank 6 (p = 0.0029) 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:10–35.

Important‑word trajectory (pos/orig): “simply” appears as a generic top‑k item by L20 while the answer remains low‑rank (cos_to_final ≈ 0.21; KL ≈ 11.44 bits) 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:24. “capital” becomes top‑1 at L24 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:28. “Berlin” first enters any top‑5 at L25 and stabilizes through final 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:29,34–36. “Germany” remains in top‑5 through L27 before dropping 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:29–31.

Rotation vs amplification: cosine direction begins to align by L20 (≥ 0.2) while KL to final remains high (~11.44 bits), implying early direction, late calibration 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:24. Cosine crosses 0.4 only by L30 and 0.6 only at final, while rank flips to 1 at L25 and final KL → 0; this pattern is consistent with gradual rotation plus late amplification/normalization alignment.

Head calibration (final layer): last_layer_consistency shows kl_to_final_bits = 0.0, top1_agree = true, temp_est = 1.0; no warning flag 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:846–864.

Lens sanity: raw_lens_check indicates lens_artifact_risk = high and first_norm_only_semantic_layer = 25 (norm‑only semantics); max_kl_norm_vs_raw_bits ≈ 0.0713 bits 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1019–1021. Treat any pre‑final “early semantics” as potentially lens‑induced; prefer rank milestones.

Temperature robustness: at T = 0.1, “Berlin” rank 1 (p ≈ 0.999965; entropy ≈ 0.00057 bits) 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:670–686. At T = 2.0, “Berlin” remains top‑1 but with p ≈ 0.0366 and entropy ≈ 13.87 bits 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:736–758.

Instructional phrasing: test prompts without the “one‑word” instruction still strongly favor “ Berlin” (e.g., “Germany’s capital city is called” → p ≈ 0.752 top‑1) 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:245–251. In the main run, ablation (no‑filler) does not shift L_semantic (both 25) 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1024–1031.

Rest‑mass sanity: rest_mass stays ~0.9995 through pre‑final and drops to 0.1625 at final; max after L_semantic = 0.99955 at L25 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:29,36. This reflects top‑k coverage, not lens fidelity.

Checklist
✓ RMS lens (RMSNorm; pre‑norm) 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:811–816
✓ LayerNorm bias removed? n.a. (not needed on RMS) 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:813–816
✗ Entropy rise at unembed (final entropy drops to 2.96 bits) 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:36
✓ FP32 un‑embed promoted (analysis unembed dtype = torch.float32) 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:809
✓ Punctuation / markup anchoring noted (e.g., L31 top‑1 ':') 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:35
✗ Copy‑reflex (no copy_collapse in L0–L3) 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:2–5
✗ Grammatical filler anchoring (no {is,the,a,of} as top‑1 in L0–5) 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:2–7

**5. Limitations & Data Quirks**
- Rest_mass remains > 0.99 through pre‑final layers; treat only as top‑k coverage, not fidelity. Final KL ≈ 0 and last‑layer consistency is clean, but KL remains lens‑sensitive; rely on rank milestones for cross‑model claims 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:846–864,1019–1021.
- raw_lens_check ran in sample mode; risk = high with a norm‑only semantic layer flagged at L25, so pre‑final semantics may be lens‑induced 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1019–1021.
- Copy metrics (L_copy, L_copy_H) are null in this run; ΔH relative to L_copy cannot be computed 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:833–839,1066–1073.

**6. Model Fingerprint**
“Llama‑3‑8B: collapse at L 25; final entropy 2.96 bits; ‘Berlin’ stabilizes from L 25; control margin peaks at 0.519.”

---
Produced by OpenAI GPT-5
*Run executed on: 2025-09-21 17:13:26*
