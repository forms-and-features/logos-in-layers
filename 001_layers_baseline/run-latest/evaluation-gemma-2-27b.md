**1. Overview**
google/gemma-2-27b (27B) was probed with a norm-lens, recording per-layer next-token predictions, entropy, KL to final, and cosine-to-final direction. Run date: 2025-08-30 18:51:32 (timestamp-20250830-1851).
The probe targets the first unseen token after “The capital of Germany is called simply”, using ID-level gold alignment for “Berlin”.

**2. Method Sanity-Check**
Diagnostics confirm norm-lens use (RMSNorm) and positional encoding handling: “use_norm_lens: true; unembed_dtype: torch.float32; layer0_position_info: token_only_rotary_model” [JSON L807–L817]. The context_prompt ends with “called simply” and no trailing space: “Give the city name only, plain text. The capital of Germany is called simply” [JSON L817]. Copy-collapse rule is the contiguous ID-level subsequence with τ=0.95, k=1, margin δ=0.10 and no entropy fallback: “copy_thresh: 0.95; copy_window_k: 1; copy_match_level: id_subsequence” [JSON L823–L826]. Gold alignment is ID-based and OK: “gold_alignment: "ok"” [JSON L831]. Negative control is present: `control_prompt` and `control_summary` [JSON L1018–L1036]. Ablation block exists with both variants in CSV: “L_copy_orig: 0; L_sem_orig: 46; L_copy_nf: 3; L_sem_nf: 46; delta_L_copy: 3; delta_L_sem: 0” [JSON L1011–L1016]; positive rows appear for `prompt_variant=orig` and `no_filler` in the pure CSV (e.g., lines 2 and 49). Summary indices: first_kl_below_0.5 = null, first_kl_below_1.0 = null, first_rank_le_1 = 46, first_rank_le_5 = 46, first_rank_le_10 = 46 [JSON L826–L830]. Units: KL and entropy are in bits (field name `kl_to_final_bits`, CSV `entropy`). Final-head calibration: last-layer KL≈1.135 bits, not ~0, with a `last_layer_consistency` snapshot present [JSON L833–L851]. Lens sanity (raw vs norm): mode=sample; “max_kl_norm_vs_raw_bits: 80.100…; lens_artifact_risk: "high"; first_norm_only_semantic_layer: null” [JSON L949–L950, L1005–L1007].
Copy-collapse flag check (pos, orig): first `copy_collapse=True` at layer 0 with top-1 “ simply” (p=0.99998) vs “ merely” (p≈7.5e-06) [CSV row 2] — ✓ rule satisfied.

**3. Quantitative Findings**
Gold answer: “Berlin”. Rows filtered to prompt_id=pos, prompt_variant=orig from pure-next-token CSV.
- L 0 – entropy 0.000 bits, top-1 ' simply'
- L 1 – entropy 8.758 bits, top-1 ''
- L 2 – entropy 8.764 bits, top-1 ''
- L 3 – entropy 0.886 bits, top-1 ' simply'
- L 4 – entropy 0.618 bits, top-1 ' simply'
- L 5 – entropy 8.520 bits, top-1 '๲'
- L 6 – entropy 8.553 bits, top-1 ''
- L 7 – entropy 8.547 bits, top-1 ''
- L 8 – entropy 8.529 bits, top-1 ''
- L 9 – entropy 8.524 bits, top-1 '𝆣'
- L 10 – entropy 8.345 bits, top-1 ' dieſem'
- L 11 – entropy 8.493 bits, top-1 '𝆣'
- L 12 – entropy 8.324 bits, top-1 ''
- L 13 – entropy 8.222 bits, top-1 ''
- L 14 – entropy 7.877 bits, top-1 ''
- L 15 – entropy 7.792 bits, top-1 ''
- L 16 – entropy 7.975 bits, top-1 ' dieſem'
- L 17 – entropy 7.786 bits, top-1 ' dieſem'
- L 18 – entropy 7.300 bits, top-1 'ſicht'
- L 19 – entropy 7.528 bits, top-1 ' dieſem'
- L 20 – entropy 6.210 bits, top-1 'ſicht'
- L 21 – entropy 6.456 bits, top-1 'ſicht'
- L 22 – entropy 6.378 bits, top-1 ' dieſem'
- L 23 – entropy 7.010 bits, top-1 ' dieſem'
- L 24 – entropy 6.497 bits, top-1 ' dieſem'
- L 25 – entropy 6.995 bits, top-1 ' dieſem'
- L 26 – entropy 6.220 bits, top-1 ' dieſem'
- L 27 – entropy 6.701 bits, top-1 ' dieſem'
- L 28 – entropy 7.140 bits, top-1 ' dieſem'
- L 29 – entropy 7.574 bits, top-1 ' dieſem'
- L 30 – entropy 7.330 bits, top-1 ' dieſem'
- L 31 – entropy 7.565 bits, top-1 ' dieſem'
- L 32 – entropy 8.874 bits, top-1 ' zuſammen'
- L 33 – entropy 6.945 bits, top-1 ' dieſem'
- L 34 – entropy 7.738 bits, top-1 ' dieſem'
- L 35 – entropy 7.651 bits, top-1 ' dieſem'
- L 36 – entropy 7.658 bits, top-1 ' dieſem'
- L 37 – entropy 7.572 bits, top-1 ' dieſem'
- L 38 – entropy 7.554 bits, top-1 ' パンチラ'
- L 39 – entropy 7.232 bits, top-1 ' dieſem'
- L 40 – entropy 8.711 bits, top-1 ' 展板'
- L 41 – entropy 7.082 bits, top-1 ' dieſem'
- L 42 – entropy 7.057 bits, top-1 ' dieſem'
- L 43 – entropy 7.089 bits, top-1 ' dieſem'
- L 44 – entropy 7.568 bits, top-1 ' dieſem'
- L 45 – entropy 7.141 bits, top-1 ' Geſch'
- L 46 – entropy 0.118 bits, top-1 ' Berlin'

Semantic collapse (first is_answer=True at ID-level): L 46 [CSV row 48].
Ablation (no-filler): L_copy_orig = 0, L_sem_orig = 46; L_copy_nf = 3, L_sem_nf = 46; ΔL_copy = 3, ΔL_sem = 0 [JSON L1011–L1016]. Interpretation: removing “simply” delays copy-reflex by 3 layers but does not shift semantic collapse (stylistic-cue removal affects early copying, not semantics).

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = 0.0005 − 0.1180 = −0.118.
Confidence milestones: p_top1 > 0.30 at L 0; p_top1 > 0.60 at L 0; final-layer p_top1 = 0.984.
Rank milestones (diagnostics): rank ≤ 10 at L 46; rank ≤ 5 at L 46; rank ≤ 1 at L 46 [JSON L828–L830].
KL milestones (diagnostics): first_kl_below_1.0 = n/a; first_kl_below_0.5 = n/a [JSON L826–L827]. KL decreases only at the end and is not ≈ 0 at final (1.135 bits), indicating final‑head calibration; `last_layer_consistency` provided [JSON L833–L851].
Cosine milestones: first cos_to_final ≥ 0.2 at L 1; ≥ 0.4 at L 46; ≥ 0.6 at L 46; final cos_to_final = 0.9994.

**4. Qualitative Patterns & Anomalies**
The model exhibits a strong copy-reflex at layer 0 (top-1 copy “ simply”, p=0.99998) with high margin over alternatives [CSV row 2], while semantic collapse occurs only at the final layer: “ Berlin, 0.984” with is_answer=True and answer_rank=1 [CSV row 48]. Negative control behaves correctly: for “Berlin is the capital of”, the top‑5 are “ Germany (0.868), the (0.065), and (0.0065), a (0.0062), Europe (0.0056)” — no leakage of “Berlin” [JSON L10–L16, L18–L30].
Records CSV shows the expected late emergence of the answer around the final block for key slots: at L46 the next token after “ is”/“ called”/“ simply” is “ Berlin” with probabilities near 1.0 (e.g., “… (‘Berlin’, 0.999998)” [L804]; “… (‘Berlin’, 0.999868)” [L805]; “… (‘Berlin’, 0.9841)” [L806]). Earlier layers for these positions do not include “Berlin” in top-5, indicating genuinely late semantics rather than early leakage.
Rest-mass sanity: rest_mass falls to 1.99e−07 by L46 (minimum overall), consistent with concentrated mass on top‑k at collapse; no spikes observed after L_semantic.
Rotation vs amplification: cos_to_final grows early to ~0.33 by L1 and remains moderate, but only jumps to ≥0.4 at the final layer while KL stays extremely high until the end (e.g., KL 41–43 bits mid‑stack; final 1.135 bits). This is an “early direction, late calibration” pattern: the representation points toward the final direction early, but probabilities calibrate only at the last layer.
Head calibration (final layer): Gemma‑family signature is present. `last_layer_consistency` reports “kl_to_final_bits: 1.1352; top1_agree: true; p_top1_lens: 0.9841 vs p_top1_model: 0.4226; temp_est: 2.61; kl_after_temp_bits: 0.5665” [JSON L833–L841, L850]. Treat absolute probabilities cautiously; prefer rank‑based statements within/between prompts.
Lens sanity: raw‑vs‑norm check reports mode="sample", “lens_artifact_risk: high; max_kl_norm_vs_raw_bits: 80.10; first_norm_only_semantic_layer: null” [JSON L949–L950, L1005–L1007]. Caution: early semantics may be lens‑induced in some families, but here the first ID‑level rank‑1 is at the final layer.
Temperature robustness: Not explored (empty temperature_exploration). Final‑head snapshot shows “ Berlin, 0.423” at the model head [JSON L858–L860], consistent with the calibration gap above.
Important-word trajectory: “Berlin” first enters any top‑5 only at L46 for positions preceding the answer slot (e.g., “… ‘ is’ → Berlin 0.999998; ‘ called’ → Berlin 0.999868; ‘ simply’ → Berlin 0.9841)” [records.csv L804–L806]. Earlier layers emphasize orthographic/rare tokens and non‑Germanic artifacts (e.g., L38 top‑1 “パンチラ”; L40 “展板”), then progressively converge toward the final direction and finally the answer token.
Stylistic ablation: Removing “simply” delays copy collapse (ΔL_copy=+3) but leaves semantic collapse unchanged (ΔL_sem=0) [JSON L1011–L1016], suggesting the adverb provides a stylistic anchor for early copying rather than affecting factual recall.

Checklist:
- RMS lens?: Yes — RMSNorm detected [JSON L810–L815].
- LayerNorm bias removed?: n.a. (RMS model: “not_needed_rms_model”) [JSON L812].
- Entropy rise at unembed?: n.a. (not explicitly instrumented; rely on CSV entropies).
- FP32 un-embed promoted?: No — `use_fp32_unembed: false`; unembed_dtype=torch.float32 via promotion logic [JSON L808–L809].
- Punctuation / markup anchoring?: Some mid‑stack odd tokens (e.g., L38 “パンチラ”, L40 “展板”) before convergence.
- Copy-reflex?: ✓ (layer 0 copy_collapse=True) [CSV row 2].
- Grammatical filler anchoring?: Not evident (layers 0–5 top‑1 not in {“is”, “the”, “a”, “of”}).

Quotes
> “use_norm_lens… unembed_dtype… layer0_position_info: token_only_rotary_model” [JSON L807–L817]
> “L_copy: 0… L_semantic: 46… copy_thresh: 0.95… copy_window_k: 1… copy_match_level: id_subsequence” [JSON L819–L826]
> “last_layer_consistency… kl_to_final_bits: 1.1352… p_top1_lens: 0.9841… p_top1_model: 0.4226… temp_est: 2.61” [JSON L833–L841, L850]
> “raw_lens_check… mode: sample… max_kl_norm_vs_raw_bits: 80.100… lens_artifact_risk: high” [JSON L949–L950, L1006–L1007]
> “Berlin is the capital of … top‑5: ‘ Germany’, 0.868; ‘ the’, 0.065; ‘ and’, 0.0065 …” [JSON L10–L16, L18–L24]
> “pos,orig,46… ‘ Berlin’, 0.984… is_answer=True… answer_rank=1” [CSV row 48]
> “pos,orig,0… ‘ simply’, 0.99998… ‘ merely’, 7.5e−06… copy_collapse=True” [CSV row 2]
> “records L46: … (‘Berlin’, 0.999998)… (‘Berlin’, 0.999868)… (‘Berlin’, 0.9841)” [records.csv L804–L806]

**5. Limitations & Data Quirks**
- Final‑head calibration: last‑layer KL≈1.135 bits with `warn_high_last_layer_kl=true`; treat final probabilities as family‑specific; prefer rank milestones and within‑model trends [JSON L833–L851].
- KL is lens‑sensitive; raw‑vs‑norm “lens_artifact_risk: high” and sampled mode only; treat KL thresholds qualitatively [JSON L949–L950, L1006–L1007].
- Rest_mass is low at collapse (≈2e−07) and high mid‑stack; no spike after L_semantic was observed in CSV, but raw‑vs‑norm sampling is not exhaustive.
- Temperature exploration absent; robustness across T not assessed.

**6. Model Fingerprint**
Gemma‑2‑27B: semantic collapse at L 46; final entropy 0.118 bits; Berlin emerges only at the last layer with strong copy‑reflex at L0.

---
Produced by OpenAI GPT-5
