**Overview**
- Meta‑Llama‑3‑8B, 32 layers; layer‑by‑layer next‑token probe with norm lens. The run captures entropy, rank/ID‑level semantics, KL to final head, cosine to final direction, copy vs semantic collapse, and control margins.
- Positive prompt ends with “called simply”; gold answer is Berlin. Outputs include detailed pure next‑token CSV and compact JSON with diagnostics, ablation, and calibration checks.

**Method Sanity‑Check**
The JSON confirms the RMS norm‑lens and rotary positional encoding context: “use_norm_lens": true” and “layer0_position_info": "token_only_rotary_model” [L807, L816]; the prompt ends exactly with “called simply” (no trailing space) [L817]. Last‑layer head calibration is consistent: “kl_to_final_bits: 0.0; top1_agree: true; p_top1_lens = p_top1_model = 0.520…” [L833–L838].
- Diagnostics contain L_copy/L_copy_H/L_semantic/delta_layers and copy‑rule provenance: “copy_thresh: 0.95, copy_window_k: 1, copy_match_level: "id_subsequence"” [L819–L826]. Gold alignment is “ok” [L831]. Negative control present: “control_prompt … France …” [L1019] and control summary with margins [L1034–L1035]. Ablation summary exists with both variants (orig/no_filler) [L1010–L1016]; positive rows appear for both in the pure CSV (pos,orig and pos,no_filler).
- Summary indices (bits and ranks) in diagnostics: first_kl_below_0.5 = 32; first_kl_below_1.0 = 32; first_rank_le_1 = 25; first_rank_le_5 = 25; first_rank_le_10 = 24 [L826–L830]. Pure CSV final KL≈0 at the last layer [row 36 in CSV].
- Lens sanity (raw vs norm): mode = sample; summary flags “first_norm_only_semantic_layer: 25; max_kl_norm_vs_raw_bits: 0.0713; lens_artifact_risk: high” [L1005–L1007]. This indicates norm‑only semantics at L25; treat early semantics cautiously.
- Copy‑collapse flag check: no rows with copy_collapse=True in layers 0–3 (CSV, pos/orig). ✓ rule did not spuriously fire; no copy reflex.

**Quantitative Findings**
Use pos/orig pure next‑token rows only. Entropy is in bits.
- L 0 – entropy 16.957 bits, top‑1 `itzer`
- L 1 – entropy 16.942 bits, top‑1 `mente`
- L 2 – entropy 16.876 bits, top‑1 `mente`
- L 3 – entropy 16.894 bits, top‑1 `tones`
- L 4 – entropy 16.899 bits, top‑1 `interp`
- L 5 – entropy 16.873 bits, top‑1 `�`
- L 6 – entropy 16.880 bits, top‑1 `tons`
- L 7 – entropy 16.881 bits, top‑1 `Exited`
- L 8 – entropy 16.862 bits, top‑1 `надлеж`
- L 9 – entropy 16.867 bits, top‑1 `biased`
- L 10 – entropy 16.851 bits, top‑1 `tons`
- L 11 – entropy 16.854 bits, top‑1 `tons`
- L 12 – entropy 16.877 bits, top‑1 `LEGAL`
- L 13 – entropy 16.843 bits, top‑1 `macros`
- L 14 – entropy 16.835 bits, top‑1 `tons`
- L 15 – entropy 16.847 bits, top‑1 ` simply`
- L 16 – entropy 16.847 bits, top‑1 ` simply`
- L 17 – entropy 16.848 bits, top‑1 ` simply`
- L 18 – entropy 16.839 bits, top‑1 ` simply`
- L 19 – entropy 16.840 bits, top‑1 ` ''`
- L 20 – entropy 16.830 bits, top‑1 ` ''`
- L 21 – entropy 16.834 bits, top‑1 ` ''`
- L 22 – entropy 16.826 bits, top‑1 `tons`
- L 23 – entropy 16.828 bits, top‑1 `tons`
- L 24 – entropy 16.830 bits, top‑1 ` capital`
- **L 25 – entropy 16.814 bits, top‑1 ` Berlin` (L_semantic)**
- L 26 – entropy 16.828 bits, top‑1 ` Berlin`
- L 27 – entropy 16.819 bits, top‑1 ` Berlin`
- L 28 – entropy 16.819 bits, top‑1 ` Berlin`
- L 29 – entropy 16.799 bits, top‑1 ` Berlin`
- L 30 – entropy 16.795 bits, top‑1 ` Berlin`
- L 31 – entropy 16.838 bits, top‑1 `:`
- L 32 – entropy 2.961 bits, top‑1 ` Berlin`

Ablation (no‑filler): L_copy_orig = null, L_sem_orig = 25; L_copy_nf = null, L_sem_nf = 25; ΔL_copy = null, ΔL_sem = 0 [L1010–L1016]. Interpretation: no stylistic‑cue sensitivity for this item.
Control margin (JSON): first_control_margin_pos = 0; max_control_margin = 0.5186 [L1034–L1035].

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (no copy collapse).
Confidence milestones (pure CSV): p_top1 > 0.30 at layer 32; p_top1 > 0.60 n.a.; final‑layer p_top1 = 0.520.
Rank milestones (diagnostics): rank ≤ 10 at layer 24; rank ≤ 5 at layer 25; rank ≤ 1 at layer 25 [L828–L830].
KL milestones (diagnostics): first_kl_below_1.0 at layer 32; first_kl_below_0.5 at layer 32 [L826–L827]. KL is ≈0 at final; calibration matches last head [L833–L840].
Cosine milestones (pure CSV): first cos_to_final ≥ 0.2 at layer 20; ≥ 0.4 at layer 30; ≥ 0.6 at layer 32; final cos_to_final = 1.000.

**Qualitative Patterns & Anomalies**
The model exhibits late calibration: cos_to_final rises by L20 while KL_to_final remains >1 until the last layer; only at L32 do KL→0 and probabilities concentrate (p_top1 ≈ 0.52) [row 36 in CSV], indicating “early direction, late calibration” (cf. Tuned‑Lens 2303.08112). Norm‑only semantics are flagged at L25 (norm sees Berlin rank 1; raw sees rank 3) with low norm‑vs‑raw KL (≈0.054), and the lens‑artifact risk is “high” [L991–L1001, L1004–L1007]; so treat pre‑final semantics comparatively, not absolutely.

Negative control: “Berlin is the capital of” top‑5: “ Germany” 0.8955; “ the” 0.0525; “ and” 0.00755; “ germany” 0.00336; “ modern” 0.00297 [L13–L31]. Berlin still appears (rank 6, p = 0.00289) → semantic leakage [L33–L35].

Important‑word trajectory (records CSV): by L24–L26, the tokens following “is/called/simply” promote Berlin into the local top‑k. Examples: at L25 for “is”, top‑1 “ Berlin” (0.000445) [L516]; for “called”, top‑1 “ Berlin” (0.000305) [L517]; and at the next‑token position after “simply”, top‑1 “ Berlin” (0.000132) [L518]. This stabilizes across L26–L30 with minor punctuation competition [L535]. Semantically close distractors (“capital”, “Deutschland”, city names) persist in the top‑k through late layers.

One‑word instruction ablation: removing “simply” does not shift collapse; L_sem_nf = 25 (ΔL_sem = 0) [L1010–L1016].

Rest‑mass sanity: Rest_mass falls only at the last layer; max after L_semantic ≈ 0.9996 (e.g., at L25–L31), consistent with very flat pre‑final distributions and precise final calibration.

Rotation vs amplification: from L20→L32, cos_to_final increases (≥0.2 at L20; ≥0.4 at L30) while answer rank improves to 1 by L25, but KL to final remains ≥1 until L32; this pattern indicates representation direction stabilizes earlier than the head’s probabilistic calibration. Final‑head calibration is clean: temp_est=1.0; kl_after_temp_bits=0.0; no scale/softcap transforms [L833–L849].

Lens sanity: raw‑vs‑norm summary shows first_norm_only_semantic_layer at 25, max KL_norm_vs_raw ≈ 0.071, risk “high” [L1005–L1007]. This is a lens‑induced early semantics caution; prefer rank‑based milestones and within‑model comparisons before the final layer.

Temperature robustness: at T=0.1, Berlin rank 1, p≈0.999965, entropy≈0.00057; at T=2.0, Berlin p≈0.0366 with broadening entropy≈13.87 [T‑block].

Checklist: ✓/✗/n.a.
- RMS lens: ✓ (RMSNorm; use_norm_lens=true) [L807, L810–L816]
- LayerNorm bias removed: n.a. (RMS model; “not_needed_rms_model”) [L812]
- Entropy rise at unembed: ✗ (entropy drops from ~16.8 to ~3.0 at L32) [row 36 in CSV]
- FP32 un‑embed promoted: n.a. (unembed_dtype is torch.float32; use_fp32_unembed=false) [L808–L809]
- Punctuation/markup anchoring: present late (quotes/colon appear in top‑k at L29–L31) [row 35 in CSV]
- Copy‑reflex: ✗ (no copy_collapse True in L0–L3; CSV pos/orig)
- Grammatical filler anchoring: ✗ (no {is,the,a,of} as top‑1 in L0–L5; CSV)

**Limitations & Data Quirks**
- Rest_mass > 0.3 persists until the last layer, indicating flat pre‑final distributions; rely on rank milestones for early‑layer comparisons. Non‑zero pre‑final KL is lens‑sensitive; final KL≈0 indicates good head calibration here.
- Raw‑vs‑norm is in sample mode; findings are a sampled sanity check, not exhaustive. Norm‑only semantics at L25 suggest some early‑layer “semantics” may be lens‑induced.

**Model Fingerprint**
“Llama‑3‑8B: semantics at L 25; final entropy ≈ 3.0 bits; ‘Berlin’ becomes decisive only at the final head.”

---
Produced by OpenAI GPT-5 

