**Overview**
- Qwen/Qwen2.5-72B (80 layers; pre-norm per diagnostics) is probed with a norm-lens logit lens to track entropy, calibration, rank, and cosine-to-final across layers. The prompt targets a single-token answer: “Berlin”.
- The probe captures collapse dynamics, copy behavior, KL-to-final, and last-layer head consistency; outputs include per-layer pure-next-token metrics and records-driven context.

**Method Sanity-Check**
- Positional encodings and lens: “use_norm_lens”: true [L807]; “layer0_position_info”: “token_only_rotary_model” [L816]. The context_prompt ends exactly with “called simply” (no trailing space): “Give the city name only, plain text. The capital of Germany is called simply” [L817].
- Copy-collapse rule present in diagnostics (ID-level contiguous subsequence, k=1), with thresholds and units: “copy_thresh”: 0.95 [L823], “copy_window_k”: 1 [L824], “copy_match_level”: “id_subsequence” [L825]. No entropy fallback; whitespace/punctuation top‑1 tokens are ignored by the rule per implementation notes.
- Key diagnostics present: “L_copy”: null, “L_copy_H”: null, “L_semantic”: 80, “delta_layers”: null [L819–L822]. Gold alignment is ID‑based and succeeded: “gold_alignment”: “ok” [L831]. Negative control present: “control_prompt” [L1018–L1031] and “control_summary” [L1033–L1036]. Ablation summary exists with both variants populated (pos/orig and pos/no_filler) [L1010–L1017]. For the main table below, rows are filtered to prompt_id = pos and prompt_variant = orig.
- Summary indices (bits and ranks) from diagnostics: first_kl_below_0.5 = 80, first_kl_below_1.0 = 80, first_rank_le_1 = 80, first_rank_le_5 = 78, first_rank_le_10 = 74 [L826–L830]. KL/entropy units are in bits (see “kl_to_final_bits” label [L833]).
- Last-layer head calibration: CSV final row has kl_to_final_bits ≈ 0 (0.000109) with top‑1 agreement; diagnostics confirm consistency: top1_agree: true, p_top1_lens: 0.3395 vs p_top1_model: 0.3383, temp_est: 1.0 [L833–L840]. No warning: warn_high_last_layer_kl = false.
- Copy-collapse flag check: no row with copy_collapse = True in layers 0–3 (e.g., L0–L3 all False in pure CSV). Therefore no early copy‑reflex flagged.
- Lens sanity (raw vs norm): mode = “sample”; summary: first_norm_only_semantic_layer = null, max_kl_norm_vs_raw_bits = 19.9099, lens_artifact_risk = “high” [L1004–L1008]. Caution: early semantics could be lens‑induced in some settings; prefer rank‑based statements and within‑model comparisons.

**Quantitative Findings**
Gold answer: “Berlin” (ID-level alignment) [L1037–L1046].

| Layer | Entropy (bits) | Top-1 token |
|---:|---:|:---|
| L 0 | 17.214 | 's' |
| L 1 | 17.214 | '下一篇' |
| L 2 | 17.143 | 'ولوج' |
| L 3 | 17.063 | 'شدد' |
| L 4 | 17.089 | '.myapplication' |
| L 5 | 17.007 | 'ستحق' |
| L 6 | 17.031 | '.myapplication' |
| L 7 | 16.937 | '.myapplication' |
| L 8 | 16.798 | '.myapplication' |
| L 9 | 16.120 | 'ستحق' |
| L 10 | 16.501 | '.myapplication' |
| L 11 | 16.718 | '.myapplication' |
| L 12 | 16.778 | 'かもしれ' |
| L 13 | 16.631 | 'かもしれ' |
| L 14 | 16.359 | 'かもしれ' |
| L 15 | 16.517 | 'のではない' |
| L 16 | 16.491 | 'iñdo' |
| L 17 | 16.213 | 'iñdo' |
| L 18 | 16.109 | '有期徒' |
| L 19 | 15.757 | '有期徒' |
| L 20 | 16.129 | '有期徒' |
| L 21 | 16.156 | '有期徒' |
| L 22 | 15.980 | '有期徒' |
| L 23 | 16.401 | '.myapplication' |
| L 24 | 15.999 | 'iñdo' |
| L 25 | 15.351 | 'hế' |
| L 26 | 15.944 | 'iñdo' |
| L 27 | 15.756 | 'iñdo' |
| L 28 | 15.750 | '.myapplication' |
| L 29 | 15.885 | '.myapplication' |
| L 30 | 16.123 | '.myapplication' |
| L 31 | 16.170 | '.myapplication' |
| L 32 | 16.171 | '.myapplication' |
| L 33 | 16.419 | 'hế' |
| L 34 | 16.200 | 'iñdo' |
| L 35 | 16.455 | 'hế' |
| L 36 | 16.408 | 'iñdo' |
| L 37 | 16.210 | 'iñdo' |
| L 38 | 16.490 | 'hế' |
| L 39 | 16.418 | 'iñdo' |
| L 40 | 16.192 | 'iñdo' |
| L 41 | 16.465 | 'hế' |
| L 42 | 16.595 | 'hế' |
| L 43 | 16.497 | 'hế' |
| L 44 | 16.655 | '続きを読む' |
| L 45 | 16.877 | '国际在线' |
| L 46 | 17.002 | '国际在线' |
| L 47 | 17.013 | '主义思想' |
| L 48 | 17.022 | '主义思想' |
| L 49 | 17.022 | ' reuseIdentifier' |
| L 50 | 16.968 | 'uckets' |
| L 51 | 16.972 | ' "' |
| L 52 | 17.009 | '"' |
| L 53 | 16.927 | '"' |
| L 54 | 16.908 | '"' |
| L 55 | 16.942 | '"' |
| L 56 | 16.938 | '"' |
| L 57 | 16.841 | ' "' |
| L 58 | 16.915 | ' "' |
| L 59 | 16.920 | ' "' |
| L 60 | 16.886 | ' \' ' |
| L 61 | 16.903 | ' \' ' |
| L 62 | 16.834 | ' "' |
| L 63 | 16.891 | ' "' |
| L 64 | 16.895 | ' "' |
| L 65 | 16.869 | ' "' |
| L 66 | 16.899 | ' "' |
| L 67 | 16.893 | ' "' |
| L 68 | 16.779 | ' "' |
| L 69 | 16.876 | ' "' |
| L 70 | 16.787 | ' "' |
| L 71 | 16.505 | ' "' |
| L 72 | 16.650 | ' "' |
| L 73 | 15.787 | ' "' |
| L 74 | 16.081 | ' "' |
| L 75 | 13.350 | ' "' |
| L 76 | 14.743 | ' "' |
| L 77 | 10.848 | ' "' |
| L 78 | 15.398 | ' "' |
| L 79 | 16.666 | ' "' |
| **L 80** | **4.116** | **' Berlin'** |

Ablation (no‑filler) [L1010–L1017]:
- L_copy_orig = null, L_sem_orig = 80; L_copy_nf = null, L_sem_nf = 80; ΔL_copy = null, ΔL_sem = 0. Interpretation: no measurable shift in copy or semantic collapse from removing the “simply” filler; rely on rank milestones when copy metrics are null.

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (L_copy = null; L_semantic entropy = 4.116).

Confidence milestones (pure CSV):
- p_top1 > 0.30 at layer 77; p_top1 > 0.60: none; final-layer p_top1 = 0.339.

Rank milestones (diagnostics):
- rank ≤ 10 at layer 74; rank ≤ 5 at layer 78; rank ≤ 1 at layer 80 [L828–L830].

KL milestones (diagnostics):
- first_kl_below_1.0 at layer 80; first_kl_below_0.5 at layer 80 [L826–L827]. KL decreases with depth and is ≈ 0 at final (CSV and diagnostics agree), so final p_top1 is directly interpretable here.

Cosine milestones (pure CSV):
- first cos_to_final ≥ 0.2 at layer 0; ≥ 0.4 at layer 0; ≥ 0.6 at layer 53; final cos_to_final = 1.00.

**Qualitative Patterns & Anomalies**
The negative control “Berlin is the capital of” places “ Germany” overwhelmingly top‑1 (0.7695), followed by function words and wh‑words; Berlin does not appear in the top‑5: “ Germany (0.7695), the (0.0864), which (0.0491), a (0.0125), what (0.0075)” [L14–L20, L22–L31]. No semantic leakage is observed.

Records show “Berlin” emerging late and growing in prominence across multiple positions. It first appears in top‑5 lists around L72–74 (e.g., pos 13: “… Berlin, 0.00296 …” [L3909]; pos 15: “… Berlin, 0.00095 …” [L3911]). By L76–77 it strengthens (“… Berlin, 0.00723 …” [L3966]; “… Berlin, 0.01918 …” [L3995]). At the final layer, Berlin is decisively top‑1 across positions (e.g., L80 pos 13: 0.8035; pos 14: 0.5730; pos 15: 0.3395) [L4100–L4106]. This trajectory is consistent with late semantic consolidation.

The main prompt shows heavy punctuation/markup anchoring mid‑stack: from ~L51 to ~L79, the top‑1 token is almost always a quote character, while “Berlin” gradually rises in the top‑k but only attains top‑1 at L80 (see table; also “… ' "' …” at multiple layers). This pattern indicates early directional alignment without calibrated magnitude: cosine with final logits is ≥ 0.4 from L0 and ≥ 0.6 by L53, while KL to final remains high until late, i.e., “early direction, late calibration.” Diagnostics corroborate this: first_kl_below_1.0 at L80 and rank milestones improving only in the last quartile [L826–L830].

Head calibration at the final layer is sound: last‑layer KL ≈ 0 (0.000109), top‑1 agrees, and p_top1_lens ≈ p_top1_model (0.3395 vs 0.3383), with temp_est = 1.0 [L833–L840]. Thus final probabilities are trustworthy for this model/family; no CFG transform adjustments were needed.

Lens sanity: the raw‑vs‑norm check is sample‑based and flags lens_artifact_risk = “high”, with max_kl_norm_vs_raw_bits ≈ 19.91 and no “norm‑only semantics” layer [L949–L1008]. Caution is warranted when attributing early semantics to the lens; use rank milestones and within‑model trends rather than absolute probabilities in early layers.

Stylistic ablation: removing “simply” produces ΔL_sem = 0 and ΔL_copy = null [L1010–L1017], suggesting minimal sensitivity to this filler style for this prompt; semantics are anchored by the core factual content rather than instruction phrasing.

Important‑word trajectory: “Berlin” enters top‑5 around L72–74 and stabilizes by L80 [L3909–L3911, L4100–L4106]. Related words (“capital”, “Germany”) are frequent in early layers’ broader distributions but do not dominate top‑1; punctuation and quoting tokens occupy many top‑1 slots mid‑stack (e.g., “ "” at L51–L79 in the pure CSV), indicating formatting anchors before semantic resolution.

Rest‑mass sanity: Rest_mass falls steadily from ~1.00 at L0 to ~0.298 at L80 (pure CSV), with no spikes suggesting precision loss; max after L_semantic = 0.298.

Checklist
- RMS lens: ✓ (RMSNorm; pre‑norm) [L810–L816].
- LayerNorm bias removed: ✓ (“not_needed_rms_model”) [L812].
- Entropy rise at unembed: ✗ (final entropy 4.116 bits is much lower than mid‑stack; table).
- FP32 un‑embed promoted: ✓ (“use_fp32_unembed”: true) [L808–L809].
- Punctuation / markup anchoring: ✓ (quotes dominate top‑1 across many middle layers; table).
- Copy‑reflex: ✗ (no copy_collapse=True in L0–3; pure CSV).
- Grammatical filler anchoring: ✗ (no {is, the, a, of} as top‑1 in L0–5; pure CSV).

**Limitations & Data Quirks**
- L_copy is null (both variants), so ΔH relative to L_copy is unavailable; rely on rank/KL milestones when comparing copy vs semantics.
- KL is lens‑sensitive. Here final‑head calibration is good, but raw_lens_check flags lens_artifact_risk = “high” (sample mode); treat early‑layer probability statements cautiously and prefer rank‑based, within‑model trends.
- Raw‑vs‑norm check is sample‑based (mode = “sample”), so these findings are a sampled sanity rather than exhaustive.

**Model Fingerprint**
- Qwen‑2.5‑72B: collapse at L 80; final entropy 4.12 bits; “Berlin” appears in top‑5 ~L72 and is rank‑1 only at the final layer.

---
Produced by OpenAI GPT-5

