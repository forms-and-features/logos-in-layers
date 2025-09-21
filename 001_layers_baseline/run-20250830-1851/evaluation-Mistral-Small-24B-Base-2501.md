# Evaluation Report: mistralai/Mistral-Small-24B-Base-2501
1. Overview
Mistral‑Small‑24B‑Base‑2501 (24B), run 2025‑08‑30. The probe traces layer‑by‑layer entropy, rank, and calibration using a norm‑lens with ID‑level gold alignment to locate copy vs. semantic collapse on the “Germany → Berlin” fact.

2. Method sanity‑check
Diagnostics confirm norm‑lens and rotary position info, with RMSNorm and fp32 unembed: “use_norm_lens: true” [L807], “layer0_position_info: token_only_rotary_model” [L816], “unembed_dtype: torch.float32” [L809]. Context prompt ends exactly with “called simply” and no trailing space: “context_prompt: … is called simply” [L817]. Pure CSV provides collapse flags and per‑layer calibration (copy_collapse, entropy_collapse, is_answer, p_top1, p_top5, p_answer, answer_rank, kl_to_final_bits, cos_to_final). No copy reflex in layers 0–3 (copy_collapse = False throughout). Copy rule and implementation flags present: “copy_thresh: 0.95, copy_window_k: 1, copy_match_level: id_subsequence” [L823–L825]; whitespace/punct top‑1 ignored and no entropy fallback per SCRIPT. Gold alignment is ID‑based and OK: “gold_alignment: ok” [L831]. Negative control blocks exist: control_prompt and control_summary [L1018–L1036]. Ablation summary present with both prompt_variant = orig and no_filler rows in CSV; main table below filters to prompt_id = pos, prompt_variant = orig. Summary indices: first_kl_below_0.5 = 40, first_kl_below_1.0 = 40, first_rank_le_1 = 33, first_rank_le_5 = 30, first_rank_le_10 = 30 [L826–L830]. Units for KL/entropy are bits (CSV column “kl_to_final_bits”; “entropy” values). Last‑layer head calibration is consistent: final CSV kl_to_final_bits ≈ 0 and diagnostics.last_layer_consistency present with “kl_to_final_bits: 0.0 … top1_agree: true … temp_est: 1.0 … kl_after_temp_bits: 0.0 … warn_high_last_layer_kl: false” [L832–L840].
Copy‑collapse flag check: no row with copy_collapse = True found in layers 0–3 → ✓ rule did not fire spuriously.
Lens sanity (raw vs. raw‑free): mode = sample [L949]; summary: lens_artifact_risk = low, max_kl_norm_vs_raw_bits = 0.1793, first_norm_only_semantic_layer = null [L1005–L1007]. No “norm‑only semantics” flagged.

3. Quantitative findings
Layer table (pos, orig). “L X — entropy Y bits, top‑1 ‘token’”. Bold marks L_semantic (first is_answer = True; gold_answer.string = “Berlin”).

- L 0 — entropy 16.9985 bits, top‑1 ‘Forbes’
- L 1 — entropy 16.9745 bits, top‑1 ‘随着时间的’
- L 2 — entropy 16.9441 bits, top‑1 ‘随着时间的’
- L 3 — entropy 16.8120 bits, top‑1 ‘随着时间的’
- L 4 — entropy 16.8682 bits, top‑1 ‘ quelcon’
- L 5 — entropy 16.9027 bits, top‑1 ‘народ’
- L 6 — entropy 16.9087 bits, top‑1 ‘народ’
- L 7 — entropy 16.8978 bits, top‑1 ‘народ’
- L 8 — entropy 16.8955 bits, top‑1 ‘ quelcon’
- L 9 — entropy 16.8852 bits, top‑1 ‘ simply’
- L 10 — entropy 16.8359 bits, top‑1 ‘ hétérogènes’
- L 11 — entropy 16.8423 bits, top‑1 ‘从那以后’
- L 12 — entropy 16.8401 bits, top‑1 ‘ simply’
- L 13 — entropy 16.8709 bits, top‑1 ‘ simply’
- L 14 — entropy 16.8149 bits, top‑1 ‘стен’
- L 15 — entropy 16.8164 bits, top‑1 ‘luš’
- L 16 — entropy 16.8300 bits, top‑1 ‘luš’
- L 17 — entropy 16.7752 bits, top‑1 ‘luš’
- L 18 — entropy 16.7608 bits, top‑1 ‘luš’
- L 19 — entropy 16.7746 bits, top‑1 ‘luš’
- L 20 — entropy 16.7424 bits, top‑1 ‘luš’
- L 21 — entropy 16.7747 bits, top‑1 ‘ simply’
- L 22 — entropy 16.7644 bits, top‑1 ‘ simply’
- L 23 — entropy 16.7690 bits, top‑1 ‘-на’
- L 24 — entropy 16.7580 bits, top‑1 ‘-на’
- L 25 — entropy 16.7475 bits, top‑1 ‘ «**’
- L 26 — entropy 16.7692 bits, top‑1 ‘ «**’
- L 27 — entropy 16.7763 bits, top‑1 ‘ «**’
- L 28 — entropy 16.7407 bits, top‑1 ‘ «**’
- L 29 — entropy 16.7604 bits, top‑1 ‘ «**’
- L 30 — entropy 16.7426 bits, top‑1 ‘-на’
- L 31 — entropy 16.7931 bits, top‑1 ‘-на’
- L 32 — entropy 16.7888 bits, top‑1 ‘-на’
- L 33 — entropy 16.7740 bits, top‑1 ‘Berlin’
- L 34 — entropy 16.7613 bits, top‑1 ‘Berlin’
- L 35 — entropy 16.7339 bits, top‑1 ‘Berlin’
- L 36 — entropy 16.6994 bits, top‑1 ‘Berlin’
- L 37 — entropy 16.5133 bits, top‑1 ‘"’
- L 38 — entropy 15.8694 bits, top‑1 ‘"’
- L 39 — entropy 16.0050 bits, top‑1 ‘Berlin’
- L 40 — entropy 3.1807 bits, top‑1 ‘Berlin’

Ablation (no‑filler): L_copy_orig = null, L_sem_orig = 33; L_copy_nf = null, L_sem_nf = 31; ΔL_copy = null; ΔL_sem = −2 [L1011–L1016]. Interpretation: removing “simply” slightly advances semantics (−2 layers; ~5% of 40), consistent with reduced filler anchoring.

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (no copy‑collapse observed).
Confidence milestones (pure CSV): p_top1 > 0.30 at layer 40; p_top1 > 0.60 at layer —; final‑layer p_top1 = 0.455.
Rank milestones (diagnostics): rank ≤ 10 at layer 30; rank ≤ 5 at layer 30; rank ≤ 1 at layer 33 [L828–L830].
KL milestones (diagnostics): first_kl_below_1.0 at layer 40; first_kl_below_0.5 at layer 40 [L826–L827]. KL decreases late and is ≈ 0 at final (see last‑layer consistency [L832–L840]).
Cosine milestones (pure CSV): first cos_to_final ≥ 0.2 at layer 35; ≥ 0.4 at layer 40; ≥ 0.6 at layer 40; final cos_to_final = 1.00.

4. Qualitative patterns & anomalies
The model shows a late semantic collapse: “Berlin” first enters top‑5 at L30 (rank 3; p = 0.00017) and becomes top‑1 by L33 (p ≈ 4.25e‑4), then steadily strengthens toward the head, consistent with logit‑lens literature (e.g., Tuned‑Lens 2303.08112). Examples: “…, Berlin, 0.000169…” [row 32 in pure CSV]; “(layer 33, token = ‘Berlin’, p = 0.000425, rank 1)” [row 35 in pure CSV]. Subword variants appear as semantics consolidate (“ber”, “-Ber”, “Бер”) before the final head, e.g., “…, ber, 0.000593; -Ber, 0.000468; Бер, 0.000369” [row 37 in pure CSV].

Negative control (“Berlin is the capital of”): top‑5 = Germany 0.802, which 0.067, the 0.0448, _ 0.0124, what 0.0109 [L12–L31]. Semantic leakage: Berlin rank 7 (p = 0.00481) [L37–L40]. Other test prompts are confident and well‑calibrated for Berlin (e.g., “Germany has its capital at” → Berlin 0.716 [L107–L122]).

Records CSV and “important words”: Early layers’ top‑1 for the NEXT position are dominated by punctuation/fillers and multilingual fragments (“ «**”, “-на”, “simply”), with no copy reflex in layers 0–3. “Berlin” appears in the top‑k only from L30 onward, rises to rank 2 by L31–L32, and becomes rank 1 by L33 [rows 32–36 in pure CSV], tracking a rotation‑then‑amplification pattern. Semantically related subpieces (“ber”, “-Ber”) co‑activate before clean resolution [row 37 in pure CSV].

Instructional phrasing: Removing the filler (“no_filler”) advances L_sem by 2 layers [L1011–L1016], suggesting mild stylistic anchoring to the “simply” cue rather than core semantics. We do not have layerwise collapse indices for the alternate test prompts (final‑head only), so we refrain from cross‑prompt collapse comparisons.

Rest‑mass sanity: Rest_mass falls sharply only near the head; maximum after L_semantic is 0.9988 (at L33), then declines toward 0.181 at the final layer (consistent with concentration of mass into top‑k).

Rotation vs. amplification: cos_to_final crosses 0.2 only at L35 and reaches 1.0 at the head, while KL to final remains >1 until the very end (first_kl_below_1.0 = 40), indicating “early direction, late calibration”. p_answer and answer_rank improve monotonically late in the stack as cos rises.

Head calibration: last‑layer consistency is clean (kl_to_final_bits = 0.0; top1_agree = true; temp_est = 1.0; warn_high_last_layer_kl = false) [L832–L840]; treat final probabilities as calibrated within this family.

Lens sanity: raw_lens_check summary indicates low artifact risk and no norm‑only semantics: “mode: sample … max_kl_norm_vs_raw_bits: 0.1793 … lens_artifact_risk: low” [L949, L1006–L1007]. A representative sample shows small raw‑vs‑norm divergence mid‑stack: “(layer 21, kl_norm_vs_raw_bits = 0.1793; top1_agree = true)” [L978–L983].

Temperature robustness: At T = 0.1, Berlin rank 1 (p = 0.9995; entropy 0.0061 bits) [L670–L676, L670–L671]; at T = 2.0, Berlin rank 1 (p = 0.0300; entropy 14.36 bits) [L737–L742]. Entropy increases as expected with higher temperature.

Important‑word trajectory: Berlin first enters any top‑5 at L30, stabilises at rank‑1 by L33, remains prominent through the final head; filler tokens (“simply”, quotes, dashes) dominate earlier layers and drop out after L36 [rows 30–36, 39–42 in pure CSV].

Stylistic ablation: ΔL_sem = −2 suggests the “simply” filler delays semantics slightly; removing it advances collapse, consistent with guidance‑style anchoring rather than core knowledge.

Checklist: 
– RMS lens? ✓ (RMSNorm; use_norm_lens true) [L807, L810–L811]  
– LayerNorm bias removed? ✓ (“layernorm_bias_fix: not_needed_rms_model”) [L812]  
– Entropy rise at unembed? ✗ (final entropy decreases sharply)  
– FP32 un‑embed promoted? ✓ (“unembed_dtype: torch.float32”; “casting_to_fp32_before_unembed”) [L809, L815]  
– Punctuation / markup anchoring? ✓ (e.g., “ «**”, quotes) [rows 25–29, 37–38 in pure CSV]  
– Copy‑reflex? ✗ (no copy_collapse True in layers 0–3)  
– Grammatical filler anchoring? ✗ (top‑1 in layers 0–5 not in {is, the, a, of})

5. Limitations & data quirks
Rest_mass remains >0.3 after L_semantic (0.9988 at L33), so top‑20 mass under‑represents the tail early in semantic formation; interpret top‑k probabilities qualitatively until late‑stack. KL is lens‑sensitive across layers; here final KL = 0 confirms head alignment, but rely on rank milestones for cross‑model claims. Raw‑vs‑norm lens sanity is “sample” mode, so treat those checks as sampled rather than exhaustive.

6. Model fingerprint
Mistral‑Small‑24B‑Base‑2501: collapse at L33; final entropy 3.18 bits; “Berlin” first appears rank‑3 at L30 and stabilises by L33.

---
Produced by OpenAI GPT-5
*Run executed on: 2025-08-30 18:51:32*
