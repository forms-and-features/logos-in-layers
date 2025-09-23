**Overview**
- Model: google/gemma-2-27b (47 layers; pre_norm; RMSNorm). Run timestamp: 2025-09-21 19:45 local (file mtime of JSON: 001_layers_baseline/run-latest/output-gemma-2-27b.json).
- Probe captures layer-wise entropy, top-1 trajectories, ID-level answer alignment, KL-to-final, cosine-to-final, copy collapse, and ablation (no-filler) effects on the Germany→Berlin task. Final-layer lens shows Berlin as top-1 with high confidence while last-layer head calibration diverges (KL ≈ 1.14 bits).

**Method Sanity-Check**
The prompt and norm lens configuration are as intended. The context prompt ends with “called simply” (no trailing space):
> "Give the city name only, plain text. The capital of Germany is called simply"  [001_layers_baseline/run-latest/output-gemma-2-27b.json:4]
Norm lens is enabled with FP32 un-embed dtype: 
> "use_norm_lens": true, "unembed_dtype": "torch.float32"  [001_layers_baseline/run-latest/output-gemma-2-27b.json:807–809]
Positional encoding info is recorded: 
> "layer0_position_info": "token_only_rotary_model"  [001_layers_baseline/run-latest/output-gemma-2-27b.json:816]
Copy rule parameters are present and use ID-level subsequence matching (k=1, τ=0.95, δ=0.10):
> "copy_thresh": 0.95, "copy_window_k": 1, "copy_match_level": "id_subsequence"  [001_layers_baseline/run-latest/output-gemma-2-27b.json:837–839]
Summary indices are present (bits):
> "first_kl_below_0.5": null, "first_kl_below_1.0": null, "first_rank_le_1": 46, "first_rank_le_5": 46, "first_rank_le_10": 46  [001_layers_baseline/run-latest/output-gemma-2-27b.json:840–844]
Gold alignment uses ID-level tokens and is OK:
> "gold_alignment": "ok"  [001_layers_baseline/run-latest/output-gemma-2-27b.json:845]; gold token: "▁Berlin" (first_id 12514)  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1052–1059]
Negative control is present:
> "control_prompt" … France→Paris  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1032–1046]; "control_summary": {"first_control_margin_pos": 0, "max_control_margin": 0.9910899400710897}  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1047–1050]
Ablation exists and both variants appear in CSVs (`prompt_variant = orig` and `no_filler`):
> "ablation_summary": {"L_copy_orig": 0, "L_sem_orig": 46, "L_copy_nf": 3, "L_sem_nf": 46, "delta_L_copy": 3, "delta_L_sem": 0}  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1024–1030]
Lens sanity check indicates high artifact risk (sample mode):
> summary: {"first_norm_only_semantic_layer": null, "max_kl_norm_vs_raw_bits": 80.10008036401692, "lens_artifact_risk": "high"}  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1018–1022]
Last-layer head calibration diverges (expected for Gemma):
> last_layer_consistency: {"kl_to_final_bits": 1.1352, "top1_agree": true, "p_top1_lens": 0.9841, "p_top1_model": 0.4226, "temp_est": 2.6102, "kl_after_temp_bits": 0.5665, "warn_high_last_layer_kl": true}  [001_layers_baseline/run-latest/output-gemma-2-27b.json:846–865]
Copy‑collapse flag check (pure CSV; pos/orig): first `copy_collapse = True` is at layer 0 with a copied token from the prompt: 
> (layer 0, top‑1 = ‘simply’, p = 0.99998; top‑2 = ‘merely’, p = 7.5e‑06)  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2]  ✓ rule satisfied

**Quantitative Findings**
Main table (pos, orig). Each row: L i – entropy X bits, top‑1 ‘token’. Bold marks the first semantic layer (is_answer=True). Units: bits.

- L 0 – entropy 0.000 bits, top-1 'simply'
- L 1 – entropy 8.758 bits, top-1 ''
- L 2 – entropy 8.764 bits, top-1 ''
- L 3 – entropy 0.886 bits, top-1 'simply'
- L 4 – entropy 0.618 bits, top-1 'simply'
- L 5 – entropy 8.520 bits, top-1 '๲'
- L 6 – entropy 8.553 bits, top-1 ''
- L 7 – entropy 8.547 bits, top-1 ''
- L 8 – entropy 8.529 bits, top-1 ''
- L 9 – entropy 8.524 bits, top-1 '𝆣'
- L 10 – entropy 8.345 bits, top-1 'dieſem'
- L 11 – entropy 8.493 bits, top-1 '𝆣'
- L 12 – entropy 8.324 bits, top-1 ''
- L 13 – entropy 8.222 bits, top-1 ''
- L 14 – entropy 7.877 bits, top-1 ''
- L 15 – entropy 7.792 bits, top-1 ''
- L 16 – entropy 7.975 bits, top-1 'dieſem'
- L 17 – entropy 7.786 bits, top-1 'dieſem'
- L 18 – entropy 7.300 bits, top-1 'ſicht'
- L 19 – entropy 7.528 bits, top-1 'dieſem'
- L 20 – entropy 6.210 bits, top-1 'ſicht'
- L 21 – entropy 6.456 bits, top-1 'ſicht'
- L 22 – entropy 6.378 bits, top-1 'dieſem'
- L 23 – entropy 7.010 bits, top-1 'dieſem'
- L 24 – entropy 6.497 bits, top-1 'dieſem'
- L 25 – entropy 6.995 bits, top-1 'dieſem'
- L 26 – entropy 6.220 bits, top-1 'dieſem'
- L 27 – entropy 6.701 bits, top-1 'dieſem'
- L 28 – entropy 7.140 bits, top-1 'dieſem'
- L 29 – entropy 7.574 bits, top-1 'dieſem'
- L 30 – entropy 7.330 bits, top-1 'dieſem'
- L 31 – entropy 7.565 bits, top-1 'dieſem'
- L 32 – entropy 8.874 bits, top-1 'zuſammen'
- L 33 – entropy 6.945 bits, top-1 'dieſem'
- L 34 – entropy 7.738 bits, top-1 'dieſem'
- L 35 – entropy 7.651 bits, top-1 'dieſem'
- L 36 – entropy 7.658 bits, top-1 'dieſem'
- L 37 – entropy 7.572 bits, top-1 'dieſem'
- L 38 – entropy 7.554 bits, top-1 'パンチラ'
- L 39 – entropy 7.232 bits, top-1 'dieſem'
- L 40 – entropy 8.711 bits, top-1 '展板'
- L 41 – entropy 7.082 bits, top-1 'dieſem'
- L 42 – entropy 7.057 bits, top-1 'dieſem'
- L 43 – entropy 7.089 bits, top-1 'dieſem'
- L 44 – entropy 7.568 bits, top-1 'dieſem'
- L 45 – entropy 7.141 bits, top-1 'Geſch'
- **L 46 – entropy 0.118 bits, top-1 'Berlin'**

Control margin (JSON): first_control_margin_pos = 0; max_control_margin = 0.9910899400710897  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1047–1050]

Ablation (no‑filler): L_copy_orig = 0, L_sem_orig = 46; L_copy_nf = 3, L_sem_nf = 46; ΔL_copy = 3, ΔL_sem = 0  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1024–1030]

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = 0.00050 − 0.11805 = −0.11755

Confidence milestones (pure CSV, pos/orig):
- p_top1 > 0.30 at layer 0; p_top1 > 0.60 at layer 0; final-layer p_top1 = 0.9841  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2,48]

Rank milestones (diagnostics):
- rank ≤ 10 at layer 46; rank ≤ 5 at layer 46; rank ≤ 1 at layer 46  [001_layers_baseline/run-latest/output-gemma-2-27b.json:842–844]

KL milestones (diagnostics + CSV):
- first_kl_below_1.0: null; first_kl_below_0.5: null  [001_layers_baseline/run-latest/output-gemma-2-27b.json:840–841]; final-layer kl_to_final_bits = 1.1352 (not ≈ 0), consistent with last-layer calibration divergence  [001_layers_baseline/run-latest/output-gemma-2-27b.json:846–865]. KL generally decreases sharply only at the final layer but remains >1 bit at final.

Cosine milestones (pure CSV, pos/orig):
- first cos_to_final ≥ 0.2 at layer 1; ≥ 0.4 at layer 46; ≥ 0.6 at layer 46; final cos_to_final = 0.99939  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48]

Prism Sidecar Analysis
- Presence: prism sidecar present and compatible (mode "auto", k=512)  [001_layers_baseline/run-latest/output-gemma-2-27b.json:819–825].
- Early-depth stability (KL at layers 0/⌊n/4⌋/⌊n/2⌋/⌊3n/4⌋/final): baseline ≈ {16.85, 41.85, 43.15, 42.51, 1.14} vs Prism ≈ {19.43, 19.43, 19.42, 19.43, 20.17} bits  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv, 001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token-prism.csv]. Prism reduces mid‑stack KL substantially but does not approach final.
- Rank milestones (Prism): first_rank_le_{10,5,1} = none observed (answer never reaches top‑10). Baseline achieved all at layer 46.
- Top‑1 agreement: at sampled depths, Prism top‑1 tokens do not agree with the final (‘Berlin’), including at the final layer (e.g., layer 46 Prism top‑1 ‘furiously’, p≈1.7e−4)  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token-prism.csv:48].
- Cosine drift: Prism cos_to_final is negative at early/mid layers (e.g., L0 ≈ −0.089, L23 ≈ −0.095) and remains negative at final (≈ −0.070), indicating no alignment with the final direction.
- Copy flags: baseline fires copy at L0 (✓), Prism does not (all False at L0–3), consistent with Prism removing the early copy prior.
- Verdict: Regressive — despite lower mid‑stack KL, Prism fails to recover rank milestones or top‑1 agreement and remains far from the final distribution.

**Qualitative Patterns & Anomalies**
Early layers show a strong copy reflex: layer 0 top‑1 copies the trailing word ‘simply’ with p≈0.99998  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2]. Mid‑stack is dominated by orthographic/punctuation and historical‑orthography artifacts (e.g., ‘dieſem’, ‘ſicht’, non‑Latin tokens), while cosine to final creeps up slowly and only snaps into alignment at the last layer (cos ≈ 0.999 at L46). KL remains very high throughout the stack and drops only at the end, but not to ≈0; this is a classic “early direction, late calibration” pattern under a norm lens with a mis‑calibrated final head.

Negative control (“Berlin is the capital of”): top‑5 are " Germany" (0.8676), " the" (0.0650), " and" (0.0065), " a" (0.0062), " Europe" (0.0056); Berlin does not appear  [001_layers_baseline/run-latest/output-gemma-2-27b.json:10–31]. No semantic leakage.

Important‑word trajectory (records CSV; IMPORTANT_WORDS = ["Germany", "Berlin", "capital", "Answer", "word", "simply"]): ‘Germany’ is consistently salient around its token (e.g., layer 3 at pos=13: top‑1 ‘Germany’, p≈0.579)  [001_layers_baseline/run-latest/output-gemma-2-27b-records.csv:66]. The answer ‘Berlin’ enters decisively only at the end and saturates across positions (e.g., L46 pos=14: ‘Berlin’, p≈0.999998; pos=15: p≈0.999868; NEXT pos=16: p≈0.9841)  [001_layers_baseline/run-latest/output-gemma-2-27b-records.csv:804–806]. The grammatical cue ‘simply’ is top‑1 in early layers at the NEXT position (L0/L3/L4), then vanishes as semantics consolidate  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2–6]. Closest semantic distractors (e.g., ‘Munich’, ‘Bonn’) appear only as minor mass at final (e.g., ‘Munich’ p≈0.0058 in final prediction)  [001_layers_baseline/run-latest/output-gemma-2-27b.json:904–909].

Collapse‑layer instruction sensitivity: Removing “simply” delays copy by +3 layers (L_copy: 0→3) but leaves semantics unchanged (L_sem: 46→46), indicating stylistic anchoring affects the copy reflex but not answer formation  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1024–1030].

Rest‑mass sanity: Rest_mass is tiny at the semantic layer (≈2.0e−07 at L46), consistent with concentrated probability mass and no precision loss  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48].

Rotation vs amplification: cos_to_final rises late (≥0.4 only at L46) while KL stays high until the end; p_answer and rank jump abruptly only at L46 (rank 1, p_answer≈0.984). This is “early direction, late calibration”; given final KL ≈1.14 bits and warn_high_last_layer_kl = true, prefer rank milestones over absolute probabilities for cross‑family comparisons  [001_layers_baseline/run-latest/output-gemma-2-27b.json:846–865].

Head calibration (final layer): temp_est ≈ 2.61 reduces KL but leaves it at ≈0.57 bits (kl_after_temp_bits) and `warn_high_last_layer_kl = true`  [001_layers_baseline/run-latest/output-gemma-2-27b.json:853–854,864]. Treat final-layer probabilities as family‑specific; rely on rank milestones and within‑model trends.

Lens sanity: raw‑vs‑norm sample indicates high artifact risk with max_kl_norm_vs_raw_bits ≈ 80.10 and no “norm‑only semantics” layer; early “semantics” should be treated cautiously and rank‑based milestones preferred  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1018–1022].

Temperature robustness: At T = 0.1, Berlin rank 1 (p ≈ 0.9898); at T = 2.0, Berlin remains rank 1 (p ≈ 0.0492). Entropy rises from ≈0.082 bits to ≈12.63 bits as T increases  [001_layers_baseline/run-latest/output-gemma-2-27b.json:670–676,737–745,738–739].

Checklist
- RMS lens? ✓ (RMSNorm; pre_norm)  [001_layers_baseline/run-latest/output-gemma-2-27b.json:810–816,960–961]
- LayerNorm bias removed? ✓ ("not_needed_rms_model")  [001_layers_baseline/run-latest/output-gemma-2-27b.json:812]
- Entropy rise at unembed? ✓ (final prediction entropy ≈ 2.886 bits vs L46 ≈ 0.118)  [001_layers_baseline/run-latest/output-gemma-2-27b.json:869; 001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48]
- FP32 un‑embed promoted? ✗ (`use_fp32_unembed`: false), but `unembed_dtype` is torch.float32  [001_layers_baseline/run-latest/output-gemma-2-27b.json:808–809,815]
- Punctuation / markup anchoring? ✓ (early top‑1 tokens like ‘๲’, ‘’, ‘’)  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:7–9]
- Copy‑reflex? ✓ (copy_collapse = True at L0)  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2]
- Grammatical filler anchoring? ✗ (early top‑1 not in {“is”, “the”, “a”, “of”})

**Limitations & Data Quirks**
- Final KL-to-final ≈ 1.135 bits with `warn_high_last_layer_kl = true`; treat final probabilities as calibration‑specific and prefer rank milestones for cross‑model claims  [001_layers_baseline/run-latest/output-gemma-2-27b.json:846–865].
- `raw_lens_check` ran in sample mode and flags high lens‑artifact risk (max_kl_norm_vs_raw_bits ≈ 80.10); early “semantics” may be lens‑induced  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1018–1022].
- Rest_mass is not a fidelity metric; its near‑zero final value only reflects top‑k coverage. KL/entropy reported are in bits.
- Prism sidecar appears misaligned with final logits (negative cosine, no rank milestones); treat its outputs as exploratory diagnostics only.

**Model Fingerprint**
“Gemma‑2‑27B: collapse at L 46; final entropy (lens) 0.118 bits; ‘Berlin’ becomes top‑1 only at the last layer; strong copy reflex at L0.”

---
Produced by OpenAI GPT-5

