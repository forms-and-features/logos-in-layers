**Overview**
- Qwen3‑8B (pre‑norm), 36+1 lens layers; run on 2025‑08‑30. Probe captures per‑layer entropy (bits), ID‑level answer rank/probability, KL to final head, cosine to final direction, copy/filler collapse flags, and control margins.
- Positive prompt ends with “called simply”; gold answer is Berlin. Outputs include pure next‑token CSV plus JSON diagnostics, ablation, raw‑vs‑norm lens sanity, and calibration checks.

**Method Sanity‑Check**
JSON confirms RMS norm‑lens and rotary position handling and the exact prompt ending: “use_norm_lens": true … “layer0_position_info": "token_only_rotary_model” [L807, L816]; context_prompt ends with “called simply” (no trailing space) [L817]. Last‑layer head calibration matches: “kl_to_final_bits: 0.0; top1_agree: true; p_top1_lens=p_top1_model=0.4334” [L833–L838].
- Diagnostics include collapse and rule fields: L_copy=null, L_copy_H=31, L_semantic=31, delta_layers=null [L819–L822]; copy_thresh=0.95, copy_window_k=1, copy_match_level="id_subsequence" [L823–L825]. Gold alignment is ok [L831]. KL/entropy units are bits (CSV column `kl_to_final_bits`, `entropy`).
- Summary indices (diagnostics): first_kl_below_0.5=36; first_kl_below_1.0=36; first_rank_le_1=31; first_rank_le_5=29; first_rank_le_10=29 [L826–L830]. CSV final row has kl_to_final_bits=0.0 at layer 36 [row 38 in CSV].
- Copy‑collapse check: no `copy_collapse=True` in layers 0–3 for pos/orig (CSV). ✓ rule did not spuriously fire; no early copy reflex.
- Raw‑vs‑norm lens sanity: mode=sample; summary shows lens_artifact_risk="high", max_kl_norm_vs_raw_bits=13.6049, first_norm_only_semantic_layer=null [L1005–L1009]. Caution that apparent early semantics may be lens‑induced.

Copy‑collapse flag check: none found (no row with `copy_collapse=True`).

**Quantitative Findings**
Use pos/orig pure next‑token rows only (entropy in bits).
- L 0 – entropy 17.213 bits, top‑1 `CLICK`
- L 1 – entropy 17.211 bits, top‑1 `apr`
- L 2 – entropy 17.211 bits, top‑1 `财经`
- L 3 – entropy 17.208 bits, top‑1 `-looking`
- L 4 – entropy 17.206 bits, top‑1 `院子`
- L 5 – entropy 17.204 bits, top‑1 ` (?)`
- L 6 – entropy 17.196 bits, top‑1 `ly`
- L 7 – entropy 17.146 bits, top‑1 ` (?)`
- L 8 – entropy 17.132 bits, top‑1 ` (?)`
- L 9 – entropy 17.119 bits, top‑1 ` (?)`
- L 10 – entropy 17.020 bits, top‑1 ` (?)`
- L 11 – entropy 17.128 bits, top‑1 `ifiable`
- L 12 – entropy 17.117 bits, top‑1 `ifiable`
- L 13 – entropy 17.126 bits, top‑1 `ifiable`
- L 14 – entropy 17.053 bits, top‑1 `"`
- L 15 – entropy 17.036 bits, top‑1 `"`
- L 16 – entropy 16.913 bits, top‑1 `-`
- L 17 – entropy 16.972 bits, top‑1 `-`
- L 18 – entropy 16.911 bits, top‑1 `-`
- L 19 – entropy 16.629 bits, top‑1 `ly`
- L 20 – entropy 16.696 bits, top‑1 `_`
- L 21 – entropy 16.408 bits, top‑1 `_`
- L 22 – entropy 15.219 bits, top‑1 ` ______`
- L 23 – entropy 15.220 bits, top‑1 `____`
- L 24 – entropy 10.893 bits, top‑1 `____`
- L 25 – entropy 13.454 bits, top‑1 `____`
- L 26 – entropy 5.558 bits, top‑1 `____`
- L 27 – entropy 4.344 bits, top‑1 `____`
- L 28 – entropy 4.786 bits, top‑1 `____`
- L 29 – entropy 1.778 bits, top‑1 `-minded`
- L 30 – entropy 2.203 bits, top‑1 ` Germany`
- **L 31 – entropy 0.454 bits, top‑1 ` Berlin`**
- L 32 – entropy 1.037 bits, top‑1 ` German`
- L 33 – entropy 0.988 bits, top‑1 ` Berlin`
- L 34 – entropy 0.669 bits, top‑1 ` Berlin`
- L 35 – entropy 2.494 bits, top‑1 ` Berlin`
- L 36 – entropy 3.123 bits, top‑1 ` Berlin`

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (no copy collapse; L_semantic=31 [L821]).
Confidence milestones (pure CSV): p_top1 > 0.30 at layer 29; p_top1 > 0.60 at layer 29; final‑layer p_top1 = 0.4334 [row 31 and row 38 in CSV].
Rank milestones (diagnostics): rank ≤ 10 at 29; rank ≤ 5 at 29; rank ≤ 1 at 31 [L828–L830].
KL milestones (diagnostics): first_kl_below_1.0 at 36; first_kl_below_0.5 at 36 [L826–L827]. KL decreases toward final and is 0.0 at layer 36 [L833–L840; row 38 in CSV].
Cosine milestones (pure CSV): first cos_to_final ≥ 0.2 at layer 36; ≥ 0.4 at 36; ≥ 0.6 at 36; final cos_to_final = 1.000.

**Qualitative Patterns & Anomalies**
The answer is decisively injected very late: Berlin first enters the top‑5 at L29 and becomes top‑1 at L31 (“ Berlin”, p=0.936) [row 33 in CSV]. Cosine alignment to the final direction remains low/negative until the last layer, then jumps to ≈1.0 at L36; together with KL_to_final→0 only at the final row, this indicates “late direction, late calibration” rather than early rotation.

Negative control “Berlin is the capital of” is well‑behaved: top‑5 are “ Germany” 0.7286, “ which” 0.2207, “ the” 0.0237, “ what” 0.0114, “ __” 0.00230 [L14–L32]. Berlin still appears lower in the list: “ Berlin”, 0.000459 → semantic leakage: Berlin rank 9 (p = 0.00046) [L46–L48].

Records CSV shows context‑token‑conditioned predictions aligning with the compositional structure by the final layer: at L36, “ called” → top‑1 “ Berlin” (0.671) [row 592], “ simply” → “ Berlin” (0.433) [row 593], “ Germany” → “ is” (0.892) [row 590], “ capital” → “ of” (0.850) [row 588]. In the pure next‑token view, “ Berlin” enters the top‑5 at L29 and stabilizes as top‑1 by L31; “ Germany” stays in the top‑5 from L30 through L36 (pos/orig).

One‑word instruction ablation: ablation_summary reports L_sem_orig=31 and L_sem_nf=31 (ΔL_sem=0) [L1011–L1016]; the collapse layer does not shift when removing “simply”.

Rest‑mass sanity: Rest_mass falls toward the end; max after L_semantic ≈ 0.175 (< 0.3), suggesting no precision loss from the norm lens.

Rotation vs amplification: p_answer climbs sharply at L31 (0.936; rank 1) while KL_to_final is still ≈1.06 bits, then the final head sets KL→0 at L36; cos_to_final only locks in at L36. This is consistent with late consolidation in both direction and calibration for this item.

Head calibration: diagnostics show clean final‑head alignment (kl_to_final_bits=0.0; top1_agree=true; temp_est=1.0; kl_after_temp_bits=0.0; no transforms) [L833–L851].

Lens sanity: raw‑vs‑norm summary flags risk “high” with a large norm‑vs‑raw discrepancy at deeper layers (e.g., L28: p_top1_norm=0.298 vs p_top1_raw=0.994; top1_agree=false) and max_kl_norm_vs_raw_bits=13.6049 [L991–L1001, L1005–L1007]. Prefer rank‑based within‑model statements for pre‑final layers.

Temperature robustness (JSON): at T=0.1, Berlin rank 1 (p=0.99915; entropy≈0.0099 bits) [L670–L676, L671]; at T=2.0, Berlin rank 1 but much flatter (p=0.0419; entropy≈13.40 bits) [L737–L743, L738].

Important‑word trajectory: “Berlin” first appears in any top‑5 at L29 and stabilizes by L31 (pure CSV). By L36, context‑token predictions align with the grammatical frame: “Germany … is … the capital … of … called … simply … Berlin” (records CSV examples above).

Checklist (✓/✗/n.a.)
- RMS lens: ✓ (RMSNorm; use_norm_lens=true) [L807, L810–L816]
- LayerNorm bias removed: n.a. (RMS model; “not_needed_rms_model”) [L812]
- Entropy rise at unembed: ✗ (entropy drops to ≈3.12 bits at final; row 38 in CSV)
- FP32 un‑embed promoted: n.a. (use_fp32_unembed=false; unembed_dtype=torch.float32) [L808–L809]
- Punctuation/markup anchoring: present in mid‑layers (e.g., quotes/underscores dominate around L14–L28; CSV)
- Copy‑reflex: ✗ (no copy_collapse=True in L0–L3; CSV)
- Grammatical filler anchoring: ✗ (no {is,the,a,of} as top‑1 in L0–L5; CSV)

**Limitations & Data Quirks**
- Final‑row calibration is clean here, but KL is lens‑sensitive pre‑final; rely on rank milestones before the last layer. Raw‑vs‑norm mode was “sample”, so raw lens sanity is sampled rather than exhaustive. Lens_artifact_risk is “high”; treat any apparent early semantics cautiously.
- Rest_mass remains < 0.3 after L_semantic (max ≈ 0.175), suggesting no norm‑lens precision loss on this item.

**Model Fingerprint**
“Qwen3‑8B: semantics at L 31; final entropy ≈ 3.1 bits; ‘Berlin’ only becomes stable near the last head.”

---
Produced by OpenAI GPT-5
