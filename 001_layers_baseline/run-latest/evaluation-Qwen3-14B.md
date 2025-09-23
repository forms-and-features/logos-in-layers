**1. Overview**
- Model: Qwen/Qwen3-14B (40 layers; pre_norm). The probe traces next-token distributions across layers with a norm-lens and records entropy, rank, KL-to-final, cosine, and copy/collapse flags. [001_layers_baseline/run-latest/output-Qwen3-14B.json:1008]
- Target prompt is the one-word capital question; gold answer is Berlin. [001_layers_baseline/run-latest/output-Qwen3-14B.json:4], [001_layers_baseline/run-latest/output-Qwen3-14B.json:1111]

**2. Method Sanity‑Check**
- Prompt ends with “called simply” (no trailing space): “Give the city name only, plain text. The capital of Germany is called simply”. [001_layers_baseline/run-latest/output-Qwen3-14B.json:4]
- Norm-lens enabled; FP32 unembed dtype; RMSNorm stack; fixes applied as expected. [001_layers_baseline/run-latest/output-Qwen3-14B.json:807], [001_layers_baseline/run-latest/output-Qwen3-14B.json:809], [001_layers_baseline/run-latest/output-Qwen3-14B.json:810]
- Copy/semantic diagnostics present with strict and soft configs: L_copy, L_copy_H, L_semantic, delta_layers, L_copy_soft, delta_layers_soft; strict τ=0.95, k=1, id_subsequence; soft τ_soft=0.5, window_ks=[1,2,3], no extra thresholds. [001_layers_baseline/run-latest/output-Qwen3-14B.json:842], [001_layers_baseline/run-latest/output-Qwen3-14B.json:843], [001_layers_baseline/run-latest/output-Qwen3-14B.json:844], [001_layers_baseline/run-latest/output-Qwen3-14B.json:845], [001_layers_baseline/run-latest/output-Qwen3-14B.json:860], [001_layers_baseline/run-latest/output-Qwen3-14B.json:865], [001_layers_baseline/run-latest/output-Qwen3-14B.json:846], [001_layers_baseline/run-latest/output-Qwen3-14B.json:848], [001_layers_baseline/run-latest/output-Qwen3-14B.json:833]
- Copy flag columns mirrored in CSV: copy_strict@0.95 and copy_soft_k{1,2,3}@0.5. [001_layers_baseline/run-latest/output-Qwen3-14B.json:1077]
- Gold‑token alignment is ID‑based and ok. [001_layers_baseline/run-latest/output-Qwen3-14B.json:898], [001_layers_baseline/run-latest/output-Qwen3-14B.json:1115]
- Negative control is present with summary margins. [001_layers_baseline/run-latest/output-Qwen3-14B.json:1106]
- Ablation summary exists with both variants; L_sem unchanged (orig=36, no_filler=36). [001_layers_baseline/run-latest/output-Qwen3-14B.json:1084], [001_layers_baseline/run-latest/output-Qwen3-14B.json:1087]
- Summary indices (bits): first_kl_below_0.5=40; first_kl_below_1.0=40; rank milestones first_rank_le_10=32, le_5=33, le_1=36. [001_layers_baseline/run-latest/output-Qwen3-14B.json:849], [001_layers_baseline/run-latest/output-Qwen3-14B.json:850], [001_layers_baseline/run-latest/output-Qwen3-14B.json:853], [001_layers_baseline/run-latest/output-Qwen3-14B.json:852], [001_layers_baseline/run-latest/output-Qwen3-14B.json:851]
- Final-layer head calibration: CSV final row has kl_to_final_bits=0.0 and answer_rank=1; diagnostics confirm perfect agreement and temp_est=1.0. [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:42], [001_layers_baseline/run-latest/output-Qwen3-14B.json:900], [001_layers_baseline/run-latest/output-Qwen3-14B.json:906]
- Lens sanity: mode=sample; lens_artifact_risk=high; max_kl_norm_vs_raw_bits=17.67; no norm‑only semantics layer flagged. Treat pre‑final “early semantics” cautiously and prefer rank milestones. [001_layers_baseline/run-latest/output-Qwen3-14B.json:1016], [001_layers_baseline/run-latest/output-Qwen3-14B.json:1074], [001_layers_baseline/run-latest/output-Qwen3-14B.json:1073], [001_layers_baseline/run-latest/output-Qwen3-14B.json:1072]
- Copy‑collapse flags (strict τ=0.95, δ=0.10): none fire in layers 0–3; earliest soft k1@0.5 also absent. ✓ rule satisfied (no spurious early fires). [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:2], [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:3], [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:4], [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:5]

First copy flag row (none in layers 0–3); example collapse event is entropy-only at L=32 (entropy_collapse=True, not copy): “…, entropy 0.8159, … entropy_collapse=True …”. [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:34]

Soft copy flags: no copy_soft_k1@0.5 events across the stack. [001_layers_baseline/run-latest/output-Qwen3-14B.json:860]

**3. Quantitative Findings**
- Positive prompt only (prompt_id=pos, prompt_variant=orig). Table shows per-layer entropy (bits) and generic top‑1 token (not necessarily the answer before semantics):

L 0 – entropy 17.213 bits, top-1 '梳'
L 1 – entropy 17.212 bits, top-1 '地处'
L 2 – entropy 17.211 bits, top-1 '是一部'
L 3 – entropy 17.210 bits, top-1 'tics'
L 4 – entropy 17.208 bits, top-1 'tics'
L 5 – entropy 17.207 bits, top-1 '-minded'
L 6 – entropy 17.205 bits, top-1 '过去的'
L 7 – entropy 17.186 bits, top-1 '�'
L 8 – entropy 17.180 bits, top-1 '-minded'
L 9 – entropy 17.188 bits, top-1 '-minded'
L 10 – entropy 17.170 bits, top-1 ' (?)'
L 11 – entropy 17.151 bits, top-1 '时代的'
L 12 – entropy 17.165 bits, top-1 'といって'
L 13 – entropy 17.115 bits, top-1 ' nav'
L 14 – entropy 17.141 bits, top-1 ' nav'
L 15 – entropy 17.149 bits, top-1 '唿'
L 16 – entropy 17.135 bits, top-1 '闯'
L 17 – entropy 17.137 bits, top-1 '唿'
L 18 – entropy 17.101 bits, top-1 '____'
L 19 – entropy 17.075 bits, top-1 '____'
L 20 – entropy 16.932 bits, top-1 '____'
L 21 – entropy 16.986 bits, top-1 '年夜'
L 22 – entropy 16.954 bits, top-1 '年夜'
L 23 – entropy 16.840 bits, top-1 '____'
L 24 – entropy 16.760 bits, top-1 '____'
L 25 – entropy 16.758 bits, top-1 '年夜'
L 26 – entropy 16.669 bits, top-1 '____'
L 27 – entropy 16.032 bits, top-1 '____'
L 28 – entropy 15.234 bits, top-1 '____'
L 29 – entropy 14.187 bits, top-1 '这个名字'
L 30 – entropy 7.789 bits, top-1 '这个名字'
L 31 – entropy 5.162 bits, top-1 '____'
L 32 – entropy 0.816 bits, top-1 '____'
L 33 – entropy 0.481 bits, top-1 '____'
L 34 – entropy 0.595 bits, top-1 '____'
L 35 – entropy 0.668 bits, top-1 '____'
**L 36 – entropy 0.312 bits, top-1 ' Berlin'**
L 37 – entropy 0.906 bits, top-1 ' ____'
L 38 – entropy 1.212 bits, top-1 ' ____'
L 39 – entropy 0.952 bits, top-1 ' Berlin'
L 40 – entropy 3.584 bits, top-1 ' Berlin'

- Control margin (Paris vs Berlin): first_control_margin_pos=0; max_control_margin=0.974. [001_layers_baseline/run-latest/output-Qwen3-14B.json:1107], [001_layers_baseline/run-latest/output-Qwen3-14B.json:1108]

- Ablation (no_filler): L_copy_orig=null, L_sem_orig=36; L_copy_nf=null, L_sem_nf=36; ΔL_copy=null, ΔL_sem=0. Large shifts absent. [001_layers_baseline/run-latest/output-Qwen3-14B.json:1084], [001_layers_baseline/run-latest/output-Qwen3-14B.json:1085], [001_layers_baseline/run-latest/output-Qwen3-14B.json:1086], [001_layers_baseline/run-latest/output-Qwen3-14B.json:1087], [001_layers_baseline/run-latest/output-Qwen3-14B.json:1089]

- ΔH (bits) and Soft ΔHk: not computed (L_copy and L_copy_soft[k] are null). [001_layers_baseline/run-latest/output-Qwen3-14B.json:842], [001_layers_baseline/run-latest/output-Qwen3-14B.json:860]

- Confidence milestones (pure CSV): p_top1 > 0.30 at L=31; p_top1 > 0.60 at L=32; final-layer p_top1=0.345. [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:33], [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:34], [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:42]

- Rank milestones (diagnostics): rank ≤10 at L=32; ≤5 at L=33; ≤1 at L=36. [001_layers_baseline/run-latest/output-Qwen3-14B.json:853], [001_layers_baseline/run-latest/output-Qwen3-14B.json:852], [001_layers_baseline/run-latest/output-Qwen3-14B.json:851]

- KL milestones (bits): first_kl_below_1.0 at L=40; first_kl_below_0.5 at L=40; final row KL≈0 confirmed by both CSV and diagnostics. [001_layers_baseline/run-latest/output-Qwen3-14B.json:850], [001_layers_baseline/run-latest/output-Qwen3-14B.json:849], [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:42], [001_layers_baseline/run-latest/output-Qwen3-14B.json:900]

- Cosine milestones (pure CSV): cos_to_final ≥0.2 at L=5; ≥0.4 at L=29; ≥0.6 at L=36; final cos=0.99999. [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:7], [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:32], [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:38], [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:42]

Prism Sidecar Analysis
- Availability: present and compatible (k=512; layers=embed, 9, 19, 29). [001_layers_baseline/run-latest/output-Qwen3-14B.json:822], [001_layers_baseline/run-latest/output-Qwen3-14B.json:825]
- Early-depth stability (KL bits, baseline→Prism): L0 12.96→13.17; L10 12.91→13.17; L20 12.98→13.23; L30 12.73→13.44 (no early KL drop). [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:2], [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token-prism.csv:2], [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:12], [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token-prism.csv:12], [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:22], [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token-prism.csv:20], [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:32], [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token-prism.csv:30]
- Rank milestones (Prism): no rank≤{10,5,1} within sampled layers. [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token-prism.csv:1]
- Top‑1 agreement at sampled depths: no qualitative earlier alignment vs baseline; Prism does not advance first_rank_le_1. [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token-prism.csv:2]
- Cosine drift: Prism cos at early/mid layers remains modest (e.g., L30 cos≈0.058); no earlier stabilization. [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token-prism.csv:30]
- Copy flags: no spurious flips under Prism (no early copy flags in either). [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token-prism.csv:2]
- Verdict: Neutral to regressive (KL increases ~0.2–0.7 bits at early/mid depths; milestones unchanged).

**4. Qualitative Patterns & Anomalies**
The negative control “Berlin is the capital of” places “ Germany” as top‑1 with p≈0.63, followed by generic function words, indicating no leakage of “Berlin”. > “ Germany, 0.6320 …; which, 0.2468 …; the, 0.0737 …” [001_layers_baseline/run-latest/output-Qwen3-14B.json:14]. Records show “Berlin” emerging in the tail by L33 at the target position with p≈0.0025, rising at L34 (p≈0.0156), dipping at L35 (p≈0.0099), then becoming dominant at L36 (p≈0.953). > “… ‘Berlin’, 0.00253 …” [001_layers_baseline/run-latest/output-Qwen3-14B-records.csv:625]; > “… ‘Berlin’, 0.01557 …” [001_layers_baseline/run-latest/output-Qwen3-14B-records.csv:641]; > “… ‘Berlin’, 0.00987 …” [001_layers_baseline/run-latest/output-Qwen3-14B-records.csv:675]; > “… ‘Berlin’, 0.95302 …” [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:38]. This aligns with rank milestones and cosine increases (cos≥0.6 at L36) while KL remains >1 bit until final, i.e., early direction, late calibration. [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:38]

Rest‑mass sanity: after semantics, maximum rest_mass is ≈0.236, staying well below 0.3. [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:42]

Temperature and style sensitivity: The test prompts show strong answer concentration when phrased with “called simply” (e.g., p(“ Berlin”)=0.8729), but punctuation and scaffolding variants modulate probabilities substantially, consistent with stylistic anchoring rather than copy. > “… ‘ Berlin’, 0.87287 …” [001_layers_baseline/run-latest/output-Qwen3-14B.json:61].

Rotation vs amplification: p_answer rises from negligible at L31 (answer_rank≈1054; p_answer≈2.6e‑05) to top‑1 by L36, while cos_to_final crosses 0.6 at L36 and KL to final drops only near the end (first_kl_below_1.0 at L40). This pattern indicates direction alignment precedes probability calibration (“early direction, late calibration”). [001_layers_baseline/run-latest/output-Qwen3-14B.json:1065], [001_layers_baseline/run-latest/output-Qwen3-14B.json:849]

Head calibration (final layer): final‑head KL=0.0 with temp_est=1.0; no warning flag. [001_layers_baseline/run-latest/output-Qwen3-14B.json:900], [001_layers_baseline/run-latest/output-Qwen3-14B.json:906], [001_layers_baseline/run-latest/output-Qwen3-14B.json:917]

Lens sanity: raw‑vs‑norm sample shows high lens_artifact_risk (max KL≈17.67 bits) but no norm‑only semantics layer; therefore treat any pre‑final “early semantics” cautiously and rely on rank milestones. [001_layers_baseline/run-latest/output-Qwen3-14B.json:1073], [001_layers_baseline/run-latest/output-Qwen3-14B.json:1072]

Important‑word trajectory: In the position of interest, control tokens and placeholders dominate mid‑stack (e.g., “这个名字”, “____”), with “Berlin” entering top‑5 by L33 and stabilizing by L36; tokens semantically close (berlin/柏林/Deutschland) co‑appear with low probabilities before consolidation. > “… top‑5 includes ‘Berlin’, 0.00253 …” [001_layers_baseline/run-latest/output-Qwen3-14B-records.csv:625]; > “… ‘柏林’ present alongside ‘Berlin’ …” [001_layers_baseline/run-latest/output-Qwen3-14B-records.csv:641].

One‑word instruction effect: Ablation (no_filler) leaves L_sem unchanged (ΔL_sem=0), suggesting semantics are not delayed/advanced by removing “simply”. [001_layers_baseline/run-latest/output-Qwen3-14B.json:1089]

Checklist
- RMS lens? ✓ [001_layers_baseline/run-latest/output-Qwen3-14B.json:810]
- LayerNorm bias removed? ✓ (RMS model: not_needed_rms_model). [001_layers_baseline/run-latest/output-Qwen3-14B.json:812]
- Entropy rise at unembed? Final entropy 3.584 bits with correct calibration. [001_layers_baseline/run-latest/output-Qwen3-14B.json:922]
- FP32 un-embed promoted? ✓ (casting to fp32 before unembed; dtype float32). [001_layers_baseline/run-latest/output-Qwen3-14B.json:815], [001_layers_baseline/run-latest/output-Qwen3-14B.json:809]
- Punctuation / markup anchoring? Present mid‑stack (e.g., underscores/quotes in top‑k). [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:33]
- Copy‑reflex? ✗ (no early strict/soft copy flags at L0–3). [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:2]
- Grammatical filler anchoring? Present (top‑1 in L0–5 predominantly fillers/punctuation). [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:2]

**5. Limitations & Data Quirks**
- Lens‑sensitivity: raw‑vs‑norm check flags high risk (max_kl_norm_vs_raw_bits≈17.67; mode=sample), so pre‑final “early semantics” may be lens‑affected; rely on rank/cosine milestones rather than absolute probabilities before final. [001_layers_baseline/run-latest/output-Qwen3-14B.json:1074], [001_layers_baseline/run-latest/output-Qwen3-14B.json:1016]
- Final‑head calibration is good (kl_to_final_bits=0), but KL in late layers remains >0 until layer 40; treat within‑model trends qualitatively and avoid cross‑family probability comparisons. [001_layers_baseline/run-latest/output-Qwen3-14B.json:900], [001_layers_baseline/run-latest/output-Qwen3-14B.json:849]
- Rest_mass stays <0.3 after L_semantic (max≈0.236), so no top‑k coverage red flag. [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:42]

**6. Model Fingerprint**
“Qwen3‑14B: semantics at L 36; entropy 0.31 bits there; final p_top1(‘Berlin’)=0.345; early KL high (≤L39) with cosine ≥0.6 by L36.”

---
Produced by OpenAI GPT-5
