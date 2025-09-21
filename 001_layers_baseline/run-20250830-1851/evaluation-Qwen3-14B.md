# Evaluation Report: Qwen/Qwen3-14B
**1. Overview**
Qwen3-14B (14B params), evaluated on 2025-08-30, with a logit-lens pass over all layers. The probe tracks copy/filler collapse vs. semantic collapse and per-layer calibration (entropy, KL to final head, cosine alignment) for the prompt “Give the city name only, plain text. The capital of Germany is called simply”.

**2. Method sanity-check**
Diagnostics confirm rotary positions and RMSNorm lens with fp32 unembed: “layer0_position_info: token_only_rotary_model” and “unembed_dtype: torch.float32” [JSON diagnostics]. Context prompt ends exactly with “called simply” (no trailing space): “Give the city name only, plain text. The capital of Germany is called simply” [JSON prompt.context_prompt]. Gold alignment is ID-based and ok: first_id=19846 (‘ĠBerlin’), variant=with_space; diagnostics.gold_alignment = ok. Copy detection parameters present: copy_thresh=0.95, copy_window_k=1, copy_match_level=id_subsequence (no entropy fallback; whitespace/punct ignored) [JSON diagnostics]. Required indices present: first_kl_below_0.5=40; first_kl_below_1.0=40; first_rank_le_1=36; first_rank_le_5=33; first_rank_le_10=32. Last-layer head calibration is consistent: CSV final kl_to_final_bits ≈ 0.0 and diagnostics.last_layer_consistency shows top1_agree=True, p_top1_lens≈p_top1_model=0.345, temp_est=1.0, warn_high_last_layer_kl=False. Pure CSV has no copy_collapse=True in layers 0–3 at the next-token position → no early copy-reflex flagged. Copy-collapse flag check: no rows with copy_collapse=True; ✓ rule did not fire spuriously. Raw-vs-norm lens sanity: mode=sample; lens_artifact_risk=high; max_kl_norm_vs_raw_bits≈17.67; first_norm_only_semantic_layer=None → caution on potential lens-induced early semantics, though none flagged as norm-only. No‑filler ablation (“simply” removed) leaves L_sem unchanged at 36 (ablation_summary: L_sem_orig=36; L_sem_nf=36; Δ=0).

Quotes (locatable):
- “copy_thresh: 0.95 … copy_window_k: 1 … copy_match_level: id_subsequence” [JSON diagnostics]
- “kl_to_final_bits: 0.0 … p_top1_lens: 0.3451403 … warn_high_last_layer_kl: False” [JSON diagnostics.last_layer_consistency]

**3. Quantitative findings**
Layer-by-layer (pure next-token at the final prompt position):

| Layer | Entropy (bits), top‑1 ‘token’ |
|---|---|
| L 0 | 17.213, top‑1 ‘梳’ |
| L 1 | 17.212, top‑1 ‘地处’ |
| L 2 | 17.211, top‑1 ‘是一部’ |
| L 3 | 17.210, top‑1 ‘tics’ |
| L 4 | 17.208, top‑1 ‘tics’ |
| L 5 | 17.207, top‑1 ‘-minded’ |
| L 6 | 17.205, top‑1 ‘过去的’ |
| L 7 | 17.186, top‑1 ‘�’ |
| L 8 | 17.180, top‑1 ‘-minded’ |
| L 9 | 17.188, top‑1 ‘-minded’ |
| L 10 | 17.170, top‑1 ‘ (?)’ |
| L 11 | 17.151, top‑1 ‘时代的’ |
| L 12 | 17.165, top‑1 ‘といって’ |
| L 13 | 17.115, top‑1 ‘ nav’ |
| L 14 | 17.141, top‑1 ‘ nav’ |
| L 15 | 17.149, top‑1 ‘唿’ |
| L 16 | 17.135, top‑1 ‘闯’ |
| L 17 | 17.137, top‑1 ‘唿’ |
| L 18 | 17.101, top‑1 ‘____’ |
| L 19 | 17.075, top‑1 ‘____’ |
| L 20 | 16.932, top‑1 ‘____’ |
| L 21 | 16.986, top‑1 ‘年夜’ |
| L 22 | 16.954, top‑1 ‘年夜’ |
| L 23 | 16.840, top‑1 ‘____’ |
| L 24 | 16.760, top‑1 ‘____’ |
| L 25 | 16.758, top‑1 ‘年夜’ |
| L 26 | 16.669, top‑1 ‘____’ |
| L 27 | 16.032, top‑1 ‘____’ |
| L 28 | 15.234, top‑1 ‘____’ |
| L 29 | 14.187, top‑1 ‘这个名字’ |
| L 30 | 7.789, top‑1 ‘这个名字’ |
| L 31 | 5.162, top‑1 ‘____’ |
| L 32 | 0.816, top‑1 ‘____’ |
| L 33 | 0.481, top‑1 ‘____’ |
| L 34 | 0.595, top‑1 ‘____’ |
| L 35 | 0.668, top‑1 ‘____’ |
| **L 36** | 0.312, top‑1 ‘ Berlin’ |
| L 37 | 0.906, top‑1 ‘ ____’ |
| L 38 | 1.212, top‑1 ‘ ____’ |
| L 39 | 0.952, top‑1 ‘ Berlin’ |
| L 40 | 3.584, top‑1 ‘ Berlin’ |

Notes: L_semantic is L 36 (first is_answer=True; p_answer=0.953). CSV final-row kl_to_final_bits=0.0, confirming last-layer head consistency [row: layer 40 in pure CSV].

Confidence milestones (pure CSV):
- p_top1 > 0.30 at layer 31; p_top1 > 0.60 at layer 32; final-layer p_top1 = 0.345.

Rank milestones (diagnostics):
- rank ≤ 10 at layer 32; rank ≤ 5 at layer 33; rank ≤ 1 at layer 36.

KL milestones (diagnostics + CSV):
- first_kl_below_1.0 at layer 40; first_kl_below_0.5 at layer 40. KL decreases overall and is ≈ 0 at the final layer (CSV kl_to_final_bits=0.0); last_layer_consistency present (temp_est=1.0; top1_agree=True).

Cosine milestones (pure CSV):
- first cos_to_final ≥ 0.2 at layer 5; ≥ 0.4 at layer 29; ≥ 0.6 at layer 36; final cos_to_final ≈ 0.99999.

ΔH (bits) = n.a. (no copy collapse layer)

**4. Qualitative patterns & anomalies**
Semantic collapse is late and sharp: the answer ‘Berlin’ becomes top‑1 at L36 (p≈0.953), briefly diffuses (L37–38), then returns (L39) before final head calibration spreads mass (L40 p≈0.345; KL→0). This suggests an emergent direction captured by the unembedding near L36 with subsequent rotation/calibration dynamics, consistent with the “Tuned-Lens” view (arXiv:2303.08112). Negative control shows no leakage: “Berlin is the capital of” → top‑5: ‘Germany’ (0.632), ‘which’ (0.247), ‘the’ (0.074), ‘what’ (0.009), ‘a’ (0.0048) [test_prompts#0] — Berlin is absent. Records CSV shows the trajectory around the decision: at L30, top‑5 remain non‑English/markup; by L32, underscores dominate; at L36, Berlin appears as both ‘ Berlin’ and ‘Berlin’ and stabilizes by L39; examples: “L36 top‑5: (‘ Berlin’, 0.953), (‘____’, 0.041), (‘ ____’, 0.0023), (‘Berlin’, 0.0020), (‘柏林’, 0.00071)” [records CSV, layer 36]. Across test prompts, the “called simply” phrasing strongly elicits ‘Berlin’ (e.g., “Germany’s capital city is called simply” → top‑1 ‘ Berlin’ 0.873; entropy 0.927 [test_prompts#1]). When the instruction is altered or omitted, responses vary (e.g., “Give the city name only … Germany’s capital city is called” → top‑1 ‘ what’ 0.533 [test_prompts#10]); layer shift of collapse cannot be assessed from final-only test prompts.

Rest-mass sanity: Rest_mass is tiny at L36 (≈0.00088) but rises to 0.236 at L40, indicating a heavier tail after final-head calibration rather than precision loss. Rotation vs amplification: cos_to_final rises early (≥0.2 by L5) while KL to final stays high until late; strong direction emerges by L36 (cos≥0.6) with late calibration to match the final head — “early direction, late calibration.” Final-head calibration: diagnostics show temp_est=1.0 and warn_high_last_layer_kl=False; treat final probabilities as comparable within-model; cross-family comparisons should prefer rank milestones when final KL isn’t ≈0 (not the case here). Lens sanity: raw_lens_check.summary → lens_artifact_risk=high; max_kl_norm_vs_raw_bits≈17.67; first_norm_only_semantic_layer=None. Sample: “{layer: 31, kl_norm_vs_raw_bits: 17.67, p_top1_norm: 0.520, p_top1_raw: 0.682, answer_rank_norm: 1054, answer_rank_raw: 114}” [raw_lens_check.sample]. Caution that some early semantics may be lens‑induced; rely on within‑model trends and rank milestones.

Important‑word trajectory: In records CSV at the decision position, ‘Berlin’ first enters any top‑5 at L33 and stabilizes by L36–39; ‘Germany’ and ‘capital’ do not appear in the top‑5 near collapse. Example snapshots: “L30 top‑5: (‘这个名字’, 0.263), (‘____’, 0.195), (‘ ____’, 0.0236), (‘这个词’, 0.0176), (‘这座城市’, 0.0171)” and “L39 top‑5: (‘ Berlin’, 0.812), (‘ BER’, 0.126), (‘ Ber’, 0.050), (‘ ____’, 0.0032), (‘____’, 0.00218)” [records CSV, layers 30 and 39].

Checklist:
- RMS lens? ✓ (RMSNorm; use_norm_lens=True)
- LayerNorm bias removed? ✓ (“layernorm_bias_fix: not_needed_rms_model”)
- Entropy rise at unembed? ✗ (final entropy rises due to calibration spread, not a spike artifact)
- FP32 un-embed promoted? ✓ (use_fp32_unembed=True; unembed_dtype=torch.float32)
- Punctuation / markup anchoring? ✓ (underscore/markup tokens dominate mid‑stack)
- Copy-reflex? ✗ (no copy_collapse=True in layers 0–3)
- Grammatical filler anchoring? ✗ (layers 0–5 top‑1 are non‑fillers)

**5. Limitations & data quirks**
- Rest_mass increases to 0.236 at L40 (post‑collapse), indicating heavy‑tail mass beyond top‑20 — acceptable but note for precision claims. KL is lens‑sensitive; here final KL≈0 so probabilities are comparable, but raw‑vs‑norm differences are large at some layers (max KL≈17.67 bits), and raw_lens_check mode=sample; treat lens findings as sampled sanity, not exhaustive. No copy collapse detected (L_copy=None), so ΔH relative to copy cannot be computed.

**6. Model fingerprint**
Qwen3‑14B: collapse at L 36; final entropy 3.58 bits; ‘Berlin’ emerges sharply then diffuses under final head.

---
Produced by OpenAI GPT-5
*Run executed on: 2025-08-30 18:51:32*
