**Overview**

- Model: google/gemma-2-27b (27B). The probe measures layer-by-layer next-token distributions with a norm lens, tracking copy-collapse, entropy, rank and calibration milestones, and cosine drift to the final direction.
- Summary: Strict copy occurs immediately at L 0 on the filler token; the gold answer ‘Berlin’ only becomes top‑1 at the final layer L 46. Final-layer lens vs head shows family-typical miscalibration (KL ≈ 1.14 bits), so probability calibration should be treated cautiously across families.

**Method Sanity‑Check**

- Norm lens and positional encoding confirmed: “use_norm_lens”: true (001_layers_baseline/run-latest/output-gemma-2-27b.json:807) and “layer0_position_info”: “token_only_rotary_model” (001_layers_baseline/run-latest/output-gemma-2-27b.json:816). Context prompt ends with “called simply” (001_layers_baseline/run-latest/output-gemma-2-27b.json:4).
- Copy detectors and indices present: “L_copy”: 0 (001_layers_baseline/run-latest/output-gemma-2-27b.json:842); “L_semantic”: 46 (001_layers_baseline/run-latest/output-gemma-2-27b.json:844); “delta_layers”: 46 (001_layers_baseline/run-latest/output-gemma-2-27b.json:845). Strict settings: “copy_thresh”: 0.95 (001_layers_baseline/run-latest/output-gemma-2-27b.json:846), “copy_window_k”: 1 (001_layers_baseline/run-latest/output-gemma-2-27b.json:847), “copy_match_level”: “id_subsequence” (001_layers_baseline/run-latest/output-gemma-2-27b.json:848). Soft config: threshold 0.5 (001_layers_baseline/run-latest/output-gemma-2-27b.json:834), window_ks [1,2,3] (001_layers_baseline/run-latest/output-gemma-2-27b.json:835), extra_thresholds [] (001_layers_baseline/run-latest/output-gemma-2-27b.json:840). Soft deltas present (001_layers_baseline/run-latest/output-gemma-2-27b.json:865).
- Flag columns mirrored in CSV/JSON: “copy_strict@0.95” (001_layers_baseline/run-latest/output-gemma-2-27b.json:1078), “copy_soft_k1@0.5” (001_layers_baseline/run-latest/output-gemma-2-27b.json:1079), “copy_soft_k2@0.5” (001_layers_baseline/run-latest/output-gemma-2-27b.json:1080), “copy_soft_k3@0.5” (001_layers_baseline/run-latest/output-gemma-2-27b.json:1081).
- Gold alignment: ok (001_layers_baseline/run-latest/output-gemma-2-27b.json:898). Gold answer block: “Berlin”, first_id 12514 (001_layers_baseline/run-latest/output-gemma-2-27b.json:1115).
- Negative control present: control prompt (France→Paris) with gold alignment ok (001_layers_baseline/run-latest/output-gemma-2-27b.json:1092). Control summary: first_control_margin_pos = 0 (001_layers_baseline/run-latest/output-gemma-2-27b.json:1107); max_control_margin = 0.9910899400710897 (001_layers_baseline/run-latest/output-gemma-2-27b.json:1108).
- Ablation present: L_copy_orig = 0 (001_layers_baseline/run-latest/output-gemma-2-27b.json:1084); L_sem_orig = 46 (001_layers_baseline/run-latest/output-gemma-2-27b.json:1085); L_copy_nf = 3 (001_layers_baseline/run-latest/output-gemma-2-27b.json:1086); L_sem_nf = 46 (001_layers_baseline/run-latest/output-gemma-2-27b.json:1087); delta_L_copy = 3 (001_layers_baseline/run-latest/output-gemma-2-27b.json:1088); delta_L_sem = 0 (001_layers_baseline/run-latest/output-gemma-2-27b.json:1089). Positive rows exist for both prompt_variant=orig and no_filler (e.g., 001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:49).
- Summary indices: first_rank_le_1 = 46 (001_layers_baseline/run-latest/output-gemma-2-27b.json:851); first_rank_le_5 = 46 (001_layers_baseline/run-latest/output-gemma-2-27b.json:852); first_rank_le_10 = 46 (001_layers_baseline/run-latest/output-gemma-2-27b.json:853); first_kl_below_1.0 = null (001_layers_baseline/run-latest/output-gemma-2-27b.json:850); first_kl_below_0.5 = null (001_layers_baseline/run-latest/output-gemma-2-27b.json:849).
- Last‑layer head calibration exists and is not ≈0: kl_to_final_bits = 1.1352 (001_layers_baseline/run-latest/output-gemma-2-27b.json:900); top1_agree = true (001_layers_baseline/run-latest/output-gemma-2-27b.json:901); p_top1_lens = 0.9841 (001_layers_baseline/run-latest/output-gemma-2-27b.json:902) vs p_top1_model = 0.4226 (001_layers_baseline/run-latest/output-gemma-2-27b.json:903); temp_est = 2.61 (001_layers_baseline/run-latest/output-gemma-2-27b.json:906); kl_after_temp_bits = 0.5665 (001_layers_baseline/run-latest/output-gemma-2-27b.json:907); warn_high_last_layer_kl = true (001_layers_baseline/run-latest/output-gemma-2-27b.json:917). Prefer rank milestones over absolute probabilities for cross‑family claims.
- Lens sanity (raw vs norm): mode=sample (001_layers_baseline/run-latest/output-gemma-2-27b.json:1015); summary.lens_artifact_risk = “high” (001_layers_baseline/run-latest/output-gemma-2-27b.json:1074); max_kl_norm_vs_raw_bits = 80.10 (001_layers_baseline/run-latest/output-gemma-2-27b.json:1073); first_norm_only_semantic_layer = null (001_layers_baseline/run-latest/output-gemma-2-27b.json:1072). Treat any “early semantics” cautiously and prefer rank thresholds.
- Copy‑collapse flag check: first row with copy_collapse=True is layer 0 with top‑1 ‘ simply’ p=0.99998; next ‘ merely’ p≈7.5e‑06 (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2). ✓ rule satisfied. Soft copy k1@0.5 also fires at L 0; k2/k3 remain null in diagnostics.

**Quantitative Findings**

- One‑layer‑per‑row (prompt_id=pos, prompt_variant=orig; entropy in bits, top‑1 token):
  - L 0 – entropy 0.000 bits, top-1 ' simply'  [row 2 in CSV]
  - L 1 – entropy 8.758 bits, top-1 ''  [row 3 in CSV]
  - L 2 – entropy 8.764 bits, top-1 ''  [row 4 in CSV]
  - L 3 – entropy 0.886 bits, top-1 ' simply'  [row 5 in CSV]
  - L 4 – entropy 0.618 bits, top-1 ' simply'  [row 6 in CSV]
  - L 5 – entropy 8.520 bits, top-1 '๲'  [row 7 in CSV]
  - L 6 – entropy 8.553 bits, top-1 ''  [row 8 in CSV]
  - L 7 – entropy 8.547 bits, top-1 ''  [row 9 in CSV]
  - L 8 – entropy 8.529 bits, top-1 ''  [row 10 in CSV]
  - L 9 – entropy 8.524 bits, top-1 '𝆣'  [row 11 in CSV]
  - L 10 – entropy 8.345 bits, top-1 ' dieſem'  [row 12 in CSV]
  - L 11 – entropy 8.493 bits, top-1 '𝆣'  [row 13 in CSV]
  - L 12 – entropy 8.324 bits, top-1 ''  [row 14 in CSV]
  - L 13 – entropy 8.222 bits, top-1 ''  [row 15 in CSV]
  - L 14 – entropy 7.877 bits, top-1 ''  [row 16 in CSV]
  - L 15 – entropy 7.792 bits, top-1 ''  [row 17 in CSV]
  - L 16 – entropy 7.975 bits, top-1 ' dieſem'  [row 18 in CSV]
  - L 17 – entropy 7.786 bits, top-1 ' dieſem'  [row 19 in CSV]
  - L 18 – entropy 7.300 bits, top-1 'ſicht'  [row 20 in CSV]
  - L 19 – entropy 7.528 bits, top-1 ' dieſem'  [row 21 in CSV]
  - L 20 – entropy 6.210 bits, top-1 'ſicht'  [row 22 in CSV]
  - L 21 – entropy 6.456 bits, top-1 'ſicht'  [row 23 in CSV]
  - L 22 – entropy 6.378 bits, top-1 ' dieſem'  [row 24 in CSV]
  - L 23 – entropy 7.010 bits, top-1 ' dieſem'  [row 25 in CSV]
  - L 24 – entropy 6.497 bits, top-1 ' dieſem'  [row 26 in CSV]
  - L 25 – entropy 6.995 bits, top-1 ' dieſem'  [row 27 in CSV]
  - L 26 – entropy 6.220 bits, top-1 ' dieſem'  [row 28 in CSV]
  - L 27 – entropy 6.701 bits, top-1 ' dieſem'  [row 29 in CSV]
  - L 28 – entropy 7.140 bits, top-1 ' dieſem'  [row 30 in CSV]
  - L 29 – entropy 7.574 bits, top-1 ' dieſem'  [row 31 in CSV]
  - L 30 – entropy 7.330 bits, top-1 ' dieſem'  [row 32 in CSV]
  - L 31 – entropy 7.565 bits, top-1 ' dieſem'  [row 33 in CSV]
  - L 32 – entropy 8.874 bits, top-1 ' zuſammen'  [row 34 in CSV]
  - L 33 – entropy 6.945 bits, top-1 ' dieſem'  [row 35 in CSV]
  - L 34 – entropy 7.738 bits, top-1 ' dieſem'  [row 36 in CSV]
  - L 35 – entropy 7.651 bits, top-1 ' dieſem'  [row 37 in CSV]
  - L 36 – entropy 7.658 bits, top-1 ' dieſem'  [row 38 in CSV]
  - L 37 – entropy 7.572 bits, top-1 ' dieſem'  [row 39 in CSV]
  - L 38 – entropy 7.554 bits, top-1 ' パンチラ'  [row 40 in CSV]
  - L 39 – entropy 7.232 bits, top-1 ' dieſem'  [row 41 in CSV]
  - L 40 – entropy 8.711 bits, top-1 ' 展板'  [row 42 in CSV]
  - L 41 – entropy 7.082 bits, top-1 ' dieſem'  [row 43 in CSV]
  - L 42 – entropy 7.057 bits, top-1 ' dieſem'  [row 44 in CSV]
  - L 43 – entropy 7.089 bits, top-1 ' dieſem'  [row 45 in CSV]
  - L 44 – entropy 7.568 bits, top-1 ' dieſem'  [row 46 in CSV]
  - L 45 – entropy 7.141 bits, top-1 ' Geſch'  [row 47 in CSV]
  - **L 46 – entropy 0.118 bits, top-1 ' Berlin'  [row 48 in CSV]**

- Control margin (control_summary): first_control_margin_pos = 0 (001_layers_baseline/run-latest/output-gemma-2-27b.json:1107); max_control_margin = 0.9910899400710897 (001_layers_baseline/run-latest/output-gemma-2-27b.json:1108).
- Ablation (no‑filler): L_copy_orig = 0; L_sem_orig = 46; L_copy_nf = 3; L_sem_nf = 46; ΔL_copy = 3; ΔL_sem = 0 (see 001_layers_baseline/run-latest/output-gemma-2-27b.json:1084,1085,1086,1087,1088,1089). Interpretation: removing “simply” delays copy collapse (later L=3 vs L=0) but does not shift semantics (still L=46).
- ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = 0.00050 − 0.11805 = −0.11755 [rows 2, 48 in CSV]. Soft ΔH₁ (k=1) = −0.11755; k=2,3 = n/a (no soft copy layer).
- Confidence milestones (pure CSV): p_top1 > 0.30 at layer 0; p_top1 > 0.60 at layer 0; final-layer p_top1 = 0.9841 [row 48 in CSV].
- Rank milestones (diagnostics): rank ≤ 10 at layer 46; rank ≤ 5 at 46; rank ≤ 1 at 46 (001_layers_baseline/run-latest/output-gemma-2-27b.json:851, 852, 853).
- KL milestones (diagnostics): first_kl_below_1.0 = null (001_layers_baseline/run-latest/output-gemma-2-27b.json:850); first_kl_below_0.5 = null (001_layers_baseline/run-latest/output-gemma-2-27b.json:849). Final KL_to_final_bits = 1.1352 [row 48 in CSV], not ≈ 0; see last_layer_consistency for head calibration.
- Cosine milestones (pure CSV): first cos_to_final ≥ 0.2 at L 1; ≥ 0.4 at L 46; ≥ 0.6 at L 46; final cos_to_final = 0.99939 [rows 3, 48 in CSV].

Prism Sidecar Analysis
- Presence: Prism artifacts present and compatible (mode=auto) (001_layers_baseline/run-latest/output-gemma-2-27b.json:820), “compatible”: true (001_layers_baseline/run-latest/output-gemma-2-27b.json:823), k=512 (001_layers_baseline/run-latest/output-gemma-2-27b.json:824), layers include 10,22,33 (001_layers_baseline/run-latest/output-gemma-2-27b.json:827).
- Early-depth stability (KL layer→final): baseline vs Prism at L=0/11/23/34/46:
  - L0: 16.85 vs 19.43 bits; L11: 41.85 vs 19.43; L23: 43.15 vs 19.42; L34: 42.51 vs 19.43; L46: 1.14 vs 20.17 bits [baseline/prism CSV sampled]. Prism does not approach final; KL remains ≈19–20 bits.
- Rank milestones (Prism): first_rank_le_{10,5,1} = none (no layer achieves ≤10; prism CSV).
- Top‑1 agreement: no improvements at sampled depths; Prism top‑1s are unrelated, baseline reaches answer only at L 46.
- Cosine drift: baseline cos_to_final rises to 0.999 at L 46; Prism cos remains near 0 and negative at sampled depths, including L 46 (≈ −0.070).
- Copy flags: baseline copy_collapse=True at L 0 (row 2 in baseline CSV) but False under Prism; flips are expected since Prism rotates/scales the representation for calibration and may disrupt local ID‑level copy signatures.
- Verdict: Regressive (KL does not drop toward final; no earlier/lower rank milestones; degraded agreement/cosine).

**Qualitative Patterns & Anomalies**

The negative control “Berlin is the capital of” returns Germany strongly: “ Germany”, 0.8676 (001_layers_baseline/run-latest/output-gemma-2-27b.json:14); “ the”, 0.0650 (001_layers_baseline/run-latest/output-gemma-2-27b.json:18). No Berlin appears in the top‑10 here, as expected. For variants without “simply,” e.g. “Germany’s capital city is called,” Berlin is already top‑1 with p=0.598 (001_layers_baseline/run-latest/output-gemma-2-27b.json:249); “Give the country name only, plain text. Berlin is the capital of” also yields “ Germany”, 0.449 (001_layers_baseline/run-latest/output-gemma-2-27b.json:437), confirming the control direction works.

Important‑word trajectory: “simply” dominates the earliest layers at the answer position and triggers strict copy at L 0: “ simply”, 0.99998; “ merely”, ~7.5e‑06 (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2). Across the stack the top‑1 drifts among non‑semantic glyphs/orthographic tokens (e.g., “ dieſem”, “ſicht”, “ パンチラ”, “ 展板”; rows 12, 20, 40, 42 in CSV) until the final layer aligns on “ Berlin” with lens p=0.984 (row 48). In records.csv, the final layer shows “ Berlin” saturating multiple preceding positions as well, e.g., at pos=14 (“ is”), “ Berlin”, 0.999998 (001_layers_baseline/run-latest/output-gemma-2-27b-records.csv:804), signaling a sharp late collapse to the correct continuation.

Copy‑reflex is present: copy_collapse=True at L 0 on the filler token (row 2 in pure CSV). This matches the strict rule (τ=0.95, δ=0.10) and soft k1@0.5 firing at the same depth; k2/k3 do not fire in diagnostics.

Rest‑mass sanity: coverage is diffuse mid‑stack but collapses by the final layer (rest_mass ≈ 1.99e‑07 at L 46; 001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48). No spike post‑semantic, suggesting acceptable numerical precision.

Rotation vs amplification: cos_to_final rises early only modestly and reaches ≈1.0 at L 46, while KL_to_final_bits remains high throughout and never falls below 1 bit even at the end (row 48). This is an “early direction, late calibration” pattern typical when the final head is miscalibrated relative to the norm lens direction; rank improves only at L 46.

Head calibration (final layer): warn_high_last_layer_kl=true with temp_est ≈ 2.61 and kl_after_temp_bits ≈ 0.567 (001_layers_baseline/run-latest/output-gemma-2-27b.json:906–907,917). Treat final‑row probabilities as family‑specific; prefer rank milestones for comparisons.

Lens sanity: raw_lens_check summary indicates lens_artifact_risk=“high” (001_layers_baseline/run-latest/output-gemma-2-27b.json:1074) with max_kl_norm_vs_raw_bits≈80.10 (001_layers_baseline/run-latest/output-gemma-2-27b.json:1073) and no norm‑only semantics layer (001_layers_baseline/run-latest/output-gemma-2-27b.json:1072). Early “semantics” should be treated cautiously and within‑model only; rank thresholds are more reliable than absolute probabilities.

Temperature robustness: At T=0.1, Berlin rank 1 (p=0.9898; entropy≈0.082) (001_layers_baseline/run-latest/output-gemma-2-27b.json:670); at T=2.0, Berlin rank 1 with p≈0.049 (entropy≈12.63) (001_layers_baseline/run-latest/output-gemma-2-27b.json:737). Entropy rises strongly with temperature as expected.

Checklist:
- RMS lens? ✓ (first_block_ln1_type=RMSNorm (001_layers_baseline/run-latest/output-gemma-2-27b.json:810); final_ln_type=RMSNorm (001_layers_baseline/run-latest/output-gemma-2-27b.json:811))
- LayerNorm bias removed? ✓ (not_needed_rms_model) (001_layers_baseline/run-latest/output-gemma-2-27b.json:812)
- Entropy rise at unembed? ✓ (final entropy 2.886 bits; 001_layers_baseline/run-latest/output-gemma-2-27b.json:922)
- FP32 un-embed promoted? ✓ (“unembed_dtype”: “torch.float32”; mixed_precision_fix: casting_to_fp32_before_unembed) (001_layers_baseline/run-latest/output-gemma-2-27b.json:809,815)
- Punctuation / markup anchoring? ✓ (mid‑stack non‑semantic glyphs dominate; e.g., rows 38–42 in CSV)
- Copy‑reflex? ✓ (copy_collapse=True in layers 0–3: row 2 in CSV)
- Grammatical filler anchoring? ✓ (top‑1 ∈ {is,the,a,of} early; “ simply” dominates L 0–4)

**Limitations & Data Quirks**

- Final KL_to_final_bits ≈ 1.14 bits (not ≈0); Gemma family shows head calibration issues. Prefer rank milestones for cross‑model claims.
- raw_lens_check mode=sample with lens_artifact_risk=“high” and max_kl_norm_vs_raw_bits≈80 bits; treat raw vs norm differences as sanity, not exhaustive.
- Rest_mass is top‑k coverage only; it falls sharply by L 46 but should not be used as a lens‑fidelity metric.
- Prism sidecar degrades alignment (KL≈19–20 bits across depths; no rank milestones); do not interpret Prism probabilities as calibrated for this model.

**Model Fingerprint**

Gemma‑2‑27B: copy at L 0; semantics at L 46; final lens p_top1≈0.98 with last‑layer KL≈1.14 bits; cosine snaps to ≈1 only at the final layer.

---
Produced by OpenAI GPT-5 
