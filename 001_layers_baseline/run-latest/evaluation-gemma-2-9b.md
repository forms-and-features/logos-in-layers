# Evaluation Report: google/gemma-2-9b

*Run executed on: 2025-09-29 23:35:16*
**Overview**

Google Gemma-2 9B (diagnostics.model = "google/gemma-2-9b") is evaluated on a single-token retrieval probe with a norm lens and RMSNorm handling. The run captures copy-reflex at L0 and semantic collapse only at the final layer, alongside KL-to-final, cosine-to-final, and control-prompt diagnostics.

**Method Sanity-Check**

Diagnostics confirm the norm lens and RMSNorm-aware alignment were used: "use_norm_lens": true and "unembed_dtype": "torch.float32" (001_layers_baseline/run-latest/output-gemma-2-9b.json:807,809). The context prompt ends with “called simply” and no trailing space: "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply" (001_layers_baseline/run-latest/output-gemma-2-9b.json:817). Copy detector config is present and strict thresholds are set: "copy_thresh": 0.95, "copy_window_k": 1, "copy_match_level": "id_subsequence" (001_layers_baseline/run-latest/output-gemma-2-9b.json:938,939,940). Gold alignment is OK: "gold_alignment": "ok" (001_layers_baseline/run-latest/output-gemma-2-9b.json:1067). Ablation summary is present for orig vs no_filler (Δ=0): {"L_copy_orig": 0, "L_sem_orig": 42, "L_copy_nf": 0, "L_sem_nf": 42} (001_layers_baseline/run-latest/output-gemma-2-9b.json:1602–1607). Negative control exists with summary: {"first_control_margin_pos": 18, "max_control_margin": 0.8677…} (001_layers_baseline/run-latest/output-gemma-2-9b.json:1624–1627).

Copy flags exist in both JSON and CSVs: "copy_flag_columns": ["copy_strict@0.95", …, "copy_soft_k1@0.5", …] (001_layers_baseline/run-latest/output-gemma-2-9b.json:1592–1600). Strict copy collapses at L0 (see Section 4); soft k1 also fires at L0. Summary indices are present: "first_kl_below_0.5": null, "first_kl_below_1.0": null, "first_rank_le_1": 42, "first_rank_le_5": 42, "first_rank_le_10": 42 (001_layers_baseline/run-latest/output-gemma-2-9b.json:941–945). Units for entropy/KL are bits in the CSV schema and diagnostics.

Last-layer head calibration: final KL is not ≈ 0; "kl_to_final_bits": 1.0129, "warn_high_last_layer_kl": true, with temperature estimate "temp_est": 2.6101 and "kl_after_temp_bits": 0.3499 (001_layers_baseline/run-latest/output-gemma-2-9b.json:1069–1086). Measurement guidance instructs rank-first reporting: {"prefer_ranks": true, "suppress_abs_probs": true, reasons include "warn_high_last_layer_kl", "norm_only_semantics_window", "high_lens_artifact_risk"} (001_layers_baseline/run-latest/output-gemma-2-9b.json:2088–2096).

Raw-vs-Norm window: radius 4 around layers [0,42], norm-only semantics flagged at layer 42, max window KL(norm||raw) = 92.316 bits (mode="window") (001_layers_baseline/run-latest/output-gemma-2-9b.json:1043–1066). Lens sanity (sample mode): lens_artifact_risk = "high"; max_kl_norm_vs_raw_bits = 12.906; first_norm_only_semantic_layer = null (001_layers_baseline/run-latest/output-gemma-2-9b.json:1530–1590). Copy-threshold sweep present: earliest L_copy_strict at τ=0.70 and τ=0.95 is 0; stability="mixed"; norm_only_flags[τ] all false (001_layers_baseline/run-latest/output-gemma-2-9b.json:990–1003,997–1014).

Copy-collapse flag check (pure CSV, pos/orig): first copy_collapse=True at layer 0 with top‑1 token " simply", p_top1=0.9999993; rule satisfied. Window sidecar shows the ID: top1_token_id=6574 at L0 (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv row for L0; 001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token-rawlens-window.csv:0–4). Soft copy k1@0.5 also first fires at L0 (pure CSV; see Section 3 milestones).

Prism sidecar present and compatible (k=512), but metrics show regression vs baseline (details in Section 3): diagnostics.prism_summary is present (001_layers_baseline/run-latest/output-gemma-2-9b.json:825–846).

"Context prompt ends with 'called simply'" check: "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply" (001_layers_baseline/run-latest/output-gemma-2-9b.json:817).

**Quantitative Findings**

Table (pos, orig). L_semantic is bolded (first is_answer=True). Gold answer string is "Berlin" (001_layers_baseline/run-latest/output-gemma-2-9b.json:2098–2108).

- L 0 – entropy 0.000017 bits, top‑1 ' simply'
- L 1 – entropy 0.000000 bits, top‑1 ' simply'
- L 2 – entropy 0.000031 bits, top‑1 ' simply'
- L 3 – entropy 0.000430 bits, top‑1 ' simply'
- L 4 – entropy 0.002116 bits, top‑1 ' simply'
- L 5 – entropy 0.004654 bits, top‑1 ' simply'
- L 6 – entropy 0.008124 bits, top‑1 ' simply'
- L 7 – entropy 0.012680 bits, top‑1 ' simply'
- L 8 – entropy 0.017640 bits, top‑1 ' simply'
- L 9 – entropy 0.025114 bits, top‑1 ' simply'
- L 10 – entropy 0.281391 bits, top‑1 ' simply'
- L 11 – entropy 0.111426 bits, top‑1 ' simply'
- L 12 – entropy 0.158366 bits, top‑1 ' simply'
- L 13 – entropy 0.211222 bits, top‑1 ' simply'
- L 14 – entropy 0.100893 bits, top‑1 ' simply'
- L 15 – entropy 0.161432 bits, top‑1 ' simply'
- L 16 – entropy 0.110002 bits, top‑1 ' simply'
- L 17 – entropy 0.071990 bits, top‑1 ' simply'
- L 18 – entropy 0.129170 bits, top‑1 ' simply'
- L 19 – entropy 0.093043 bits, top‑1 ' simply'
- L 20 – entropy 0.051503 bits, top‑1 ' simply'
- L 21 – entropy 1.866495 bits, top‑1 ' simply'
- L 22 – entropy 0.017224 bits, top‑1 ' simply'
- L 23 – entropy 0.018517 bits, top‑1 ' simply'
- L 24 – entropy 0.036987 bits, top‑1 ' simply'
- L 25 – entropy 0.048131 bits, top‑1 ' simply'
- L 26 – entropy 0.071675 bits, top‑1 ' simply'
- L 27 – entropy 0.094487 bits, top‑1 ' simply'
- L 28 – entropy 0.133719 bits, top‑1 ' simply'
- L 29 – entropy 0.177052 bits, top‑1 ' simply'
- L 30 – entropy 0.221196 bits, top‑1 ' simply'
- L 31 – entropy 0.046090 bits, top‑1 ' simply'
- L 32 – entropy 0.096579 bits, top‑1 ' simply'
- L 33 – entropy 0.146153 bits, top‑1 ' simply'
- L 34 – entropy 0.199044 bits, top‑1 ' simply'
- L 35 – entropy 0.248623 bits, top‑1 ' simply'
- L 36 – entropy 0.296041 bits, top‑1 ' simply'
- L 37 – entropy 0.329285 bits, top‑1 ' simply'
- L 38 – entropy 0.348654 bits, top‑1 ' simply'
- L 39 – entropy 0.357990 bits, top‑1 ' simply'
- L 40 – entropy 0.364694 bits, top‑1 ' simply'
- L 41 – entropy 0.369215 bits, top‑1 ' simply'
- L 42 – entropy 0.370067 bits, top‑1 ' Berlin'  —  is_answer=True

Control margin (JSON control_summary): first_control_margin_pos = 18; max_control_margin = 0.8677 (001_layers_baseline/run-latest/output-gemma-2-9b.json:1624–1627).

Ablation (no‑filler): L_copy_orig = 0, L_sem_orig = 42; L_copy_nf = 0, L_sem_nf = 42; ΔL_copy = 0, ΔL_sem = 0 (001_layers_baseline/run-latest/output-gemma-2-9b.json:1602–1607). Interpretation: removing “simply” did not shift copy or semantics.

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = 0.0000167 − 0.370067 ≈ −0.37005. Soft ΔH₁ (k=1) = −0.37005; k=2,3 n.a. (no soft layer reported).

Confidence milestones (pure CSV, within‑model only): p_top1 > 0.30 at layer 0; p_top1 > 0.60 at layer 0; final-layer p_top1 = 0.9298 (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv: L0, L42). Rank milestones (diagnostics): rank ≤ 10 at 42; rank ≤ 5 at 42; rank ≤ 1 at 42 (001_layers_baseline/run-latest/output-gemma-2-9b.json:943–945). KL milestones (diagnostics): first_kl_below_1.0 = null; first_kl_below_0.5 = null; final KL ≈ 1.013 bits (001_layers_baseline/run-latest/output-gemma-2-9b.json:941–943,1069). Cosine milestones (JSON): cos_to_final ≥ 0.2 at L1; ≥ 0.4 at L42; ≥ 0.6 at L42; final cos_to_final = 0.9993 (001_layers_baseline/run-latest/output-gemma-2-9b.json:1028–1033; 001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv: L42).

Depth fractions: L_semantic_frac = 1.0 (001_layers_baseline/run-latest/output-gemma-2-9b.json:1035–1041).

Copy robustness (threshold sweep): stability = "mixed"; earliest strict copy at τ=0.70 and τ=0.95 both L0; norm_only_flags[τ] all false (001_layers_baseline/run-latest/output-gemma-2-9b.json:990–1003,997–1014).

Prism Sidecar Analysis: compatible=true, but regressive. KL(P_layer||P_final) increases substantially at early/mid depths vs baseline (e.g., L0: 28.48 vs 14.39; L21: 25.51 vs 15.17; L42: 28.73 vs 1.01) (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token-prism.csv, 001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv). Rank milestones under Prism never reach ≤10/5/1 (diagnostics.prism_summary.metrics.rank_milestones.prism all null) (001_layers_baseline/run-latest/output-gemma-2-9b.json:867–908). Verdict: Regressive.

Tuned‑Lens sidecar: ΔKL medians at percentiles Δ = KL_norm − KL_tuned: p25 = −0.280, p50 = −10.518, p75 = +0.199 (001_layers_baseline/run-latest/output-gemma-2-9b.json:1710–1750). Rank milestones unchanged (≤10/5/1 all 42) (001_layers_baseline/run-latest/output-gemma-2-9b.json:1751–1770). Entropy drift at mid-depths is negative: at L21, entropy − teacher_entropy_bits ≈ −1.07 bits (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv: L21).

**Qualitative Patterns & Anomalies**

Negative control shows correct mapping “Berlin → Germany” with small leakage of the cue token: top‑5 for “Berlin is the capital of” includes “ Germany” (0.8766) and “ Berlin” in rank 9 (p=0.00187), indicating semantic leakage of the prompt entity (001_layers_baseline/run-latest/output-gemma-2-9b.json:10–53). Records at the final layer show strong answer preference at the last three positions: at L42, pos=16 (“ simply”), top‑1 is “ Berlin” (p=0.9298), with punctuation alternatives trailing (001_layers_baseline/run-latest/output-gemma-2-9b-records.csv:893).

Across layers, NEXT is dominated by copy of the filler token “ simply” through almost the entire stack; “Berlin” first becomes top‑1 only at the final layer (is_answer=True at L42; pure CSV). Important-word trajectory: the tokens immediately preceding NEXT (“ is”, “ called”, “ simply”) keep their surface forms as top‑1 across layers; by L42 they align to the semantic answer, e.g., at pos=14 (“ is”) the top‑1 is “ Berlin” (0.99999) and at pos=15 (“ called”) “ Berlin” (0.99984) (001_layers_baseline/run-latest/output-gemma-2-9b-records.csv:891–893). Semantically related variants (e.g., “ BERLIN”, “ berlin”) appear with low mass behind the canonical answer at L42 (001_layers_baseline/run-latest/output-gemma-2-9b-records.csv:893).

Collapse-layer index does not shift when the “one-word” instruction is ablated: ablations show L_sem unchanged at 42 and L_copy unchanged at 0 (001_layers_baseline/run-latest/output-gemma-2-9b.json:1602–1607). Rest_mass remains small after semantics and does not spike: max after L_semantic ≈ 1.11e‑05 (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv: L42).

Rotation vs amplification: cosine-to-final rises early (≥0.2 by L1) while KL to final stays high until the end; semantics (rank‑1) is achieved only at L42 with final KL ≈ 1.01 bits. This indicates early direction, late calibration under the norm lens. Final‑head calibration issue is flagged: {"temp_est": 2.610…, "kl_after_temp_bits": 0.3499, "warn_high_last_layer_kl": true} (001_layers_baseline/run-latest/output-gemma-2-9b.json:1069–1086). Lens sanity: raw_lens_check.summary shows lens_artifact_risk = "high" and max_kl_norm_vs_raw_bits = 12.906 (001_layers_baseline/run-latest/output-gemma-2-9b.json:1530–1590). Given measurement_guidance (prefer_ranks, suppress_abs_probs), we rely on rank milestones for collapse claims.

Temperature robustness (teacher comparison): entropy drift (entropy − teacher_entropy_bits) is negative at mid‑depth (e.g., L21: −1.07 bits), indicating sharper-than-teacher distributions mid‑stack (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv: L21). Copy‑reflex is pronounced: strict copy at L0–L3 with “ simply” (pure CSV rows L0–L3); soft k1 also fires at L0. Grammatical filler anchoring is present: top‑1 in early layers is a filler (“ simply”), and across the prompt many high‑probability tokens are function words or punctuation (records CSV early rows).

Checklist:
✓ RMS lens? (RMSNorm reported) (001_layers_baseline/run-latest/output-gemma-2-9b.json:810–813)
✓ LayerNorm bias removed? (not_needed_rms_model) (001_layers_baseline/run-latest/output-gemma-2-9b.json:812)
✓ Entropy rise at unembed? (final entropy 2.94 bits in final head; lens entropy at NEXT 0.37 bits at L42) (001_layers_baseline/run-latest/output-gemma-2-9b.json:1784–1800; pure CSV L42)
✓ FP32 unembed promoted? "unembed_dtype": "torch.float32" (001_layers_baseline/run-latest/output-gemma-2-9b.json:809)
✓ Punctuation / markup anchoring? (quotes, punctuation in top‑k near final) (001_layers_baseline/run-latest/output-gemma-2-9b-records.csv:893)
✓ Copy‑reflex? (strict and soft at L0–L3) (pure CSV L0–L3)
✓ Grammatical filler anchoring? (top‑1 ' simply' in L0–L5) (pure CSV L0–L5)

**Limitations & Data Quirks**

Final‑layer KL under the norm lens is ≈1.01 bits and flagged as high; family‑level head calibration is known for Gemma—use rank milestones for cross‑model claims (001_layers_baseline/run-latest/output-gemma-2-9b.json:1069–1086,2088–2096). Raw‑vs‑norm lens differences are substantial near the end (window max ≈92.316 bits) and lens_artifact_risk is "high" in sample mode; treat early “semantics” cautiously and prefer rank‑based milestones (001_layers_baseline/run-latest/output-gemma-2-9b.json:1043–1066,1530–1590). Surface‑mass and top‑k rest_mass depend on tokenizer coverage; within‑model trends are reliable, but avoid cross‑family probability comparisons per measurement guidance.

**Model Fingerprint**

Gemma‑2‑9B: strict copy at L0; semantic collapse at L42; final KL≈1.01 bits; final cos_to_final≈0.999.

---
Produced by OpenAI GPT-5 
