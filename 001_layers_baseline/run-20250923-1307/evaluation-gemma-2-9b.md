# Evaluation Report: google/gemma-2-9b

1. Overview
google/gemma-2-9b (9B) probed on 2025-09-23. The run traces layer-wise next-token behavior under a norm lens, capturing copy-collapse on the filler, semantic emergence of “Berlin,” calibration to the final head, and Prism sidecar comparison.

2. Method sanity-check
The diagnostics confirm the norm lens and rotary/token-only positional handling are in effect: “use_norm_lens: true” [001_layers_baseline/run-latest/output-gemma-2-9b.json:807] and “layer0_position_info: "token_only_rotary_model"” [001_layers_baseline/run-latest/output-gemma-2-9b.json:816]. The context prompt ends with “called simply” with no trailing space [001_layers_baseline/run-latest/output-gemma-2-9b.json:817].
Copy detection config appears in diagnostics: “copy_thresh: 0.95, copy_window_k: 1, copy_match_level: "id_subsequence"” [001_layers_baseline/run-latest/output-gemma-2-9b.json:846], with soft config “threshold: 0.5, window_ks: [1,2,3]” [001_layers_baseline/run-latest/output-gemma-2-9b.json:833]. Flags mirror labels via “copy_flag_columns”: ["copy_strict@0.95","copy_soft_k1@0.5","copy_soft_k2@0.5","copy_soft_k3@0.5"] [001_layers_baseline/run-latest/output-gemma-2-9b.json:1077]. Gold alignment is ok [001_layers_baseline/run-latest/output-gemma-2-9b.json:898]; the gold token block shows ID-level alignment for “Berlin”, first_id=12514 [001_layers_baseline/run-latest/output-gemma-2-9b.json:1115]. Negative control and summary present [001_layers_baseline/run-latest/output-gemma-2-9b.json:1091], with control margins reported [001_layers_baseline/run-latest/output-gemma-2-9b.json:1107]. Ablation summary exists [001_layers_baseline/run-latest/output-gemma-2-9b.json:1083]; both orig and no_filler rows appear in CSV (e.g., pos,no_filler,… [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:50]).
Summary indices: first_kl_below_0.5 = null; first_kl_below_1.0 = null; first_rank_le_1 = 42; first_rank_le_5 = 42; first_rank_le_10 = 42 [001_layers_baseline/run-latest/output-gemma-2-9b.json:849]. KL/entropy are reported in bits (see “kl_to_final_bits”) [001_layers_baseline/run-latest/output-gemma-2-9b.json:900]. Last-layer head calibration record exists and is not ≈0: kl_to_final_bits = 1.0129 bits with top1_agree = true; p_top1_lens = 0.9298 vs p_top1_model = 0.3943; temp_est = 2.61; kl_after_temp_bits = 0.3499; warn_high_last_layer_kl = true [001_layers_baseline/run-latest/output-gemma-2-9b.json:899]. Lens sanity (raw vs norm): mode=sample; lens_artifact_risk = "high"; max_kl_norm_vs_raw_bits = 12.9056; first_norm_only_semantic_layer = null [001_layers_baseline/run-latest/output-gemma-2-9b.json:1071].
Copy-collapse flags: the pure CSV marks copy_collapse=True from L0 with strict rule satisfied, e.g., layer=0: top-1 “ simply” p=0.9999993; top-2 “simply” p=7.73e-07 [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2] — ✓ rule satisfied. Earliest soft flag is also k1 at L0; k2,k3 remain false/null (see columns copy_soft_k{1,2,3}@0.5) [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2].

3. Quantitative findings
- L 0 — entropy 0.000017 bits, top-1 'simply'
- L 1 — entropy 0.000000 bits, top-1 'simply'
- L 2 — entropy 0.000031 bits, top-1 'simply'
- L 3 — entropy 0.000430 bits, top-1 'simply'
- L 4 — entropy 0.002116 bits, top-1 'simply'
- L 5 — entropy 0.002333 bits, top-1 'simply'
- L 6 — entropy 0.127902 bits, top-1 'simply'
- L 7 — entropy 0.033569 bits, top-1 'simply'
- L 8 — entropy 0.098417 bits, top-1 'simply'
- L 9 — entropy 0.102087 bits, top-1 'simply'
- L 10 — entropy 0.281391 bits, top-1 'simply'
- L 11 — entropy 0.333046 bits, top-1 'simply'
- L 12 — entropy 0.109330 bits, top-1 'simply'
- L 13 — entropy 0.137400 bits, top-1 'simply'
- L 14 — entropy 0.165772 bits, top-1 'simply'
- L 15 — entropy 0.734873 bits, top-1 'simply'
- L 16 — entropy 3.568274 bits, top-1 'simply'
- L 17 — entropy 3.099445 bits, top-1 'simply'
- L 18 — entropy 3.336717 bits, top-1 'simply'
- L 19 — entropy 1.382336 bits, top-1 'simply'
- L 20 — entropy 3.163441 bits, top-1 'simply'
- L 21 — entropy 1.866495 bits, top-1 'simply'
- L 22 — entropy 2.190102 bits, top-1 'simply'
- L 23 — entropy 3.181111 bits, top-1 'simply'
- L 24 — entropy 1.107039 bits, top-1 'simply'
- L 25 — entropy 2.118879 bits, top-1 'the'
- L 26 — entropy 2.371327 bits, top-1 'the'
- L 27 — entropy 1.842460 bits, top-1 'the'
- L 28 — entropy 1.226664 bits, top-1 '"'
- L 29 — entropy 0.315988 bits, top-1 '"'
- L 30 — entropy 0.134063 bits, top-1 '"'
- L 31 — entropy 0.046090 bits, top-1 '"'
- L 32 — entropy 0.062538 bits, top-1 '"'
- L 33 — entropy 0.042715 bits, top-1 '"'
- L 34 — entropy 0.090030 bits, top-1 '"'
- L 35 — entropy 0.023370 bits, top-1 '"'
- L 36 — entropy 0.074091 bits, top-1 '"'
- L 37 — entropy 0.082534 bits, top-1 '"'
- L 38 — entropy 0.033455 bits, top-1 '"'
- L 39 — entropy 0.046899 bits, top-1 '"'
- L 40 — entropy 0.036154 bits, top-1 '"'
- L 41 — entropy 0.176738 bits, top-1 '"'
- L 42 — entropy 0.370067 bits, top-1 'Berlin'

Control margin (JSON): first_control_margin_pos = 18; max_control_margin = 0.8677 [001_layers_baseline/run-latest/output-gemma-2-9b.json:1107].

Ablation (no‑filler): L_copy_orig = 0, L_sem_orig = 42; L_copy_nf = 0, L_sem_nf = 42; ΔL_copy = 0; ΔL_sem = 0 [001_layers_baseline/run-latest/output-gemma-2-9b.json:1083]. Interpretation: no stylistic-cue sensitivity.

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = 0.000017 − 0.370067 = −0.370050.
Soft ΔH₁ (bits) = same as strict (k=1 at L0); k=2,3 not present (null) [001_layers_baseline/run-latest/output-gemma-2-9b.json:860].
Confidence milestones (pure CSV): p_top1 > 0.30 at L0; p_top1 > 0.60 at L0; final-layer p_top1 = 0.9298 [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:49].
Rank milestones (JSON): rank ≤ 10 at L42; rank ≤ 5 at L42; rank ≤ 1 at L42 [001_layers_baseline/run-latest/output-gemma-2-9b.json:851].
KL milestones (JSON): first_kl_below_1.0 = null; first_kl_below_0.5 = null [001_layers_baseline/run-latest/output-gemma-2-9b.json:849]. KL decreases late but is not ≈0 at final; final-head calibration flagged (see Section 2).
Cosine milestones (pure CSV): first cos_to_final ≥ 0.2 at L1; ≥ 0.4 at L42; ≥ 0.6 at L42; final cos_to_final = 0.9993 [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:49].

Prism Sidecar Analysis
- Presence: compatible=true [001_layers_baseline/run-latest/output-gemma-2-9b.json:823].
- Early-depth stability (KL to final, bits): baseline vs Prism at L0: 14.39 vs 28.48; L10: 14.94 vs 26.57; L21: 15.17 vs 25.51; L31: 1.87 vs 26.01 [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2,12,23,33; 001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token-prism.csv:2,12,21,31].
- Rank milestones: baseline first_rank_le_{10,5,1}=42; Prism: none reached (answer_rank remains >10 throughout) [001_layers_baseline/run-latest/output-gemma-2-9b.json:851; 001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token-prism.csv:44].
- Top‑1 agreement: at sampled depths L∈{0,10,21,31}, neither baseline nor Prism agrees with final top‑1 “Berlin” [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:49; 001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token-prism.csv:44].
- Cosine drift: baseline cos_to_final rises from −0.074→0.381 by L31, Prism stays near 0 or negative (e.g., L0 −0.231; L10 −0.130; L21 0.021; L31 −0.012) [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token-prism.csv:2].
- Copy flags: baseline strict copy at L0 (“simply”); Prism fires none (likely due to sidecar distribution reshaping) [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2; 001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token-prism.csv:2].
- Verdict: Regressive — Prism greatly increases KL at early/mid layers and never attains answer-rank milestones.

4. Qualitative patterns & anomalies
Copy-reflex: strong early copy on the filler “simply” (L0–L5, p≈1.0), e.g., “ simply, 0.9999993 … simply, 7.73e−07” (layer 0) [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2]. This matches the strict ID-contiguous rule (τ=0.95, δ=0.10) configured in diagnostics [001_layers_baseline/run-latest/output-gemma-2-9b.json:846].
Rotation vs amplification: direction aligns early while calibration lags. Cosine-to-final crosses 0.2 by L1 and stays modest until late, while KL-to-final remains ≫1 bit until near the top; final KL is still ~1.01 bits [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:49; 001_layers_baseline/run-latest/output-gemma-2-9b.json:900]. This is “early direction, late calibration,” consistent with tuned-lens style analyses (cf. 2303.08112) but here with Gemma’s known head calibration issue.
Final‑head calibration: last-layer consistency shows top‑1 agreement yet large calibration gap: p_top1_lens 0.9298 vs p_top1_model 0.3943; temp_est ≈ 2.61; kl_after_temp_bits ≈ 0.35; warn_high_last_layer_kl = true [001_layers_baseline/run-latest/output-gemma-2-9b.json:899]. Prefer rank milestones over absolute p at final for cross‑family comparisons.
Lens sanity: raw‑vs‑norm lens check (sampled) flags lens_artifact_risk = high with max_kl_norm_vs_raw_bits ≈ 12.91; no norm‑only semantic layer was detected [001_layers_baseline/run-latest/output-gemma-2-9b.json:1071]. Treat any “early semantics” claims cautiously and favor rank thresholds.
Negative control: “Berlin is the capital of” yields top‑5 “ Germany 0.8766; the 0.0699; modern 0.0077; a 0.0053; ␠ 0.0034 … Berlin appears rank 9 (p=0.00187)” [001_layers_baseline/run-latest/output-gemma-2-9b.json:12]. → semantic leakage: Berlin rank 9 (p = 0.00187).
Important-word trajectory: “Germany,” “capital,” and “simply” are stable copies with near‑delta distributions in records at L0 (e.g., “ Germany, 0.999998” [001_layers_baseline/run-latest/output-gemma-2-9b-records.csv:15]; “ capital, 1.0” [001_layers_baseline/run-latest/output-gemma-2-9b-records.csv:13]; “ simply, 0.9999993” [001_layers_baseline/run-latest/output-gemma-2-9b-records.csv:18]). The answer “Berlin” only surfaces at the final layer as top‑1 in the pure next‑token view [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:49]. Semantically related distractors (e.g., “Bonn” 0.0152; “Frankfurt” 0.00325) appear in a test variant [001_layers_baseline/run-latest/output-gemma-2-9b.json:586].
One‑word instruction ablation: removing “simply” does not shift collapse; ΔL_copy = 0; ΔL_sem = 0 [001_layers_baseline/run-latest/output-gemma-2-9b.json:1088].
Rest‑mass sanity: rest_mass remains tiny at and after semantics (max after L_semantic ≈ 1.11e‑05), indicating top‑20 coverage is excellent [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:49].
Temperature robustness: T=0.1 → “Berlin” rank 1 (p≈0.981; entropy ≈0.137 bits) [001_layers_baseline/run-latest/output-gemma-2-9b.json:670]; T=2.0 → “Berlin” rank 1 (p≈0.089; entropy ≈9.00 bits) [001_layers_baseline/run-latest/output-gemma-2-9b.json:737].

Checklist
- RMS lens? ✓ [001_layers_baseline/run-latest/output-gemma-2-9b.json:810]
- LayerNorm bias removed? ✓ (RMS; not needed) [001_layers_baseline/run-latest/output-gemma-2-9b.json:812]
- Entropy rise at unembed? ✓ (e.g., 0.1767→0.3701 at L41→L42) [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:48]
- FP32 un-embed promoted? ✗ (use_fp32_unembed=false) [001_layers_baseline/run-latest/output-gemma-2-9b.json:808]
- Punctuation / markup anchoring? ✓ (late layers dominated by quotes/commas) [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:38]
- Copy-reflex? ✓ (copy_collapse True in L0–3) [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2]
- Grammatical filler anchoring? ✓ (top‑1 in L0–5 = “simply”; articles later “the”) [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:25]

5. Limitations & data quirks
Final KL-to-final ≈ 1.01 bits (warn_high_last_layer_kl = true) indicates final‑head calibration mismatch; rely on rank milestones and within‑model trends for confidence [001_layers_baseline/run-latest/output-gemma-2-9b.json:899]. Raw‑vs‑norm lens check ran in sample mode and flags high artifact risk; treat early semantics cautiously and prefer rank thresholds [001_layers_baseline/run-latest/output-gemma-2-9b.json:1071]. Rest_mass is low after L_semantic (~1e‑5), so coverage is good; no evidence of precision loss from top‑k truncation at the point of semantics [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:49].

6. Model fingerprint (one sentence)
Gemma‑2‑9B: collapse at L 42; final entropy 0.37 bits; “Berlin” only appears at the final layer; Prism regresses (KL high, no rank milestones).

---
Produced by OpenAI GPT-5
