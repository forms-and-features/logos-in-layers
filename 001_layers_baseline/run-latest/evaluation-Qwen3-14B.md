# Evaluation Report: Qwen/Qwen3-14B

*Run executed on: 2025-10-04 18:04:23*
**1. Overview**

Qwen3-14B (40 layers) evaluated on 2025-10-04 (run-latest timestamp-20251004-1804). Layer-by-layer logit-lens probe tracks copy cues, entropy, rank milestones, KL to final, cosine geometry, and calibrated sidecars (Prism, Tuned-Lens) across the stack. The gold answer is ID-aligned to “Berlin”.

**2. Method Sanity-Check**

The run uses the intended norm lens with RMSNorm and pre-norm γ selection; LayerNorm bias is absent and an FP32 shadow unembed is used. The context prompt ends with “called simply” (no trailing space):
- “context_prompt": "Give the city name only, plain text. The capital of Germany is called simply”  [001_layers_baseline/run-latest/output-Qwen3-14B.json:76]
- “use_norm_lens": true, "use_fp32_unembed": false, "unembed_dtype": "torch.float32"”  [001_layers_baseline/run-latest/output-Qwen3-14B.json:66-69]

Normalizer provenance confirms pre-norm strategy and expected sources at the ends of the stack:
- “strategy": "next_ln1" … per_layer[0].ln_source = "blocks[0].ln1"; per_layer[last].ln_source = "ln_final"”  [001_layers_baseline/run-latest/output-Qwen3-14B.json:7226-7260,7481-7560]

Per-layer normalizer effect is well-behaved near and before semantics (no extreme spikes):
- “layer 0 … resid_norm_ratio: 0.4964, delta_resid_cos: 0.9896”  [001_layers_baseline/run-latest/output-Qwen3-14B.json:7206-7218]
- “layer 36 … resid_norm_ratio: 0.2344, delta_resid_cos: 0.7329”  [001_layers_baseline/run-latest/output-Qwen3-14B.json:7517-7533]

Unembedding bias is absent (cosine metrics are bias-free):
- “unembed_bias": { "present": false, "l2_norm": 0.0, "max_abs": 0.0 }”  [001_layers_baseline/run-latest/output-Qwen3-14B.json:826-834]

Environment and determinism are set; seed fixed:
- “torch_version": "2.8.0+cu128", "device": "cpu", "deterministic_algorithms": true, "seed": 316”  [001_layers_baseline/run-latest/output-Qwen3-14B.json:8373-8396]

Numeric health is clean:
- “any_nan": false, "any_inf": false, … "layers_flagged": []”  [001_layers_baseline/run-latest/output-Qwen3-14B.json:7808-7816]

Gold alignment is ID-level and ok; the gold answer pieces are provided:
- “gold_alignment": "ok"”  [001_layers_baseline/run-latest/output-Qwen3-14B.json:7822]
- “gold_answer": { "string": "Berlin", "pieces": ["ĠBerlin"], "first_id": 19846 }”  [001_layers_baseline/run-latest/output-Qwen3-14B.json:8913-8931]

Copy mask is present and plausible (punctuation/control heavy; size = 6112):
- “copy_mask": { … "size": 6112 }”  [001_layers_baseline/run-latest/output-Qwen3-14B.json:7078-7088]

Copy/collapse detectors and thresholds are recorded (strict τ, sweep, soft config); implementation flags present:
- “copy_thresh": 0.95, "copy_window_k": 1, "copy_match_level": "id_subsequence" … "tau_list": [0.7,0.8,0.9,0.95] … "stability": "none"”  [001_layers_baseline/run-latest/output-Qwen3-14B.json:7068-7096,7136-7180]
- “copy_flag_columns": ["copy_strict@0.95","copy_strict@0.7","copy_strict@0.8","copy_strict@0.9","copy_soft_k1@0.5","copy_soft_k2@0.5","copy_soft_k3@0.5"]”  [001_layers_baseline/run-latest/output-Qwen3-14B.json:8396-8413]

Last-layer head calibration is excellent (KL≈0 after optional temperature):
- “last_layer_consistency": { "kl_to_final_bits": 0.0, "top1_agree": true, "p_top1_lens": 0.34514, "p_top1_model": 0.34514, "temp_est": 1.0, "kl_after_temp_bits": 0.0 }”  [001_layers_baseline/run-latest/output-Qwen3-14B.json:7816-7832]

Measurement guidance recommends rank-led reporting and Tuned-Lens as the preferred lens; confirmed semantics are enabled:
- “measurement_guidance": { "prefer_ranks": true, "suppress_abs_probs": true, … "preferred_lens_for_reporting": "tuned", "use_confirmed_semantics": true }”  [001_layers_baseline/run-latest/output-Qwen3-14B.json:8913-8926]

Raw-vs-Norm windowed check (window mode) and full scan indicate high artifact risk; no norm-only semantics layers were found in the window, but overall lens-artefact score is high:
- “raw_lens_window": { "radius": 4, "center_layers": [33,36,40], … "norm_only_semantics_layers": [], "max_kl_norm_vs_raw_bits_window": 98.5781, "mode": "window" }”  [001_layers_baseline/run-latest/output-Qwen3-14B.json:7181-7214]
- “raw_lens_full": { "pct_layers_kl_ge_1.0": 0.7561, "pct_layers_kl_ge_0.5": 0.8293, "n_norm_only_semantics_layers": 0, "earliest_norm_only_semantic": null, "max_kl_norm_vs_raw_bits": 98.5781, "score": { "lens_artifact_score": 0.5537, "tier": "high" } }”  [001_layers_baseline/run-latest/output-Qwen3-14B.json:7214-7226]
- Sampled raw-vs-norm deltas show large discrepancies at mid-depth (artifact risk high): “layer 31 … kl_norm_vs_raw_bits: 17.6735 … top1_agree: false”  [001_layers_baseline/run-latest/output-Qwen3-14B.json:8330-8351]

Strict copy (k=1, τ=0.95, δ=0.10) and soft copy (τ_soft=0.5, window_ks={1,2,3}) both remain null across layers; no early copy-reflex flags in L0–L3 in the pure CSV. Earliest layers (0–3) show copy_collapse=False and soft flags=False in all rows (no hits)  [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:1-4].

Control prompt and summary present; negative control is configured and gold-aligned:
- “control_prompt … The capital of France is called simply … gold_alignment: ok”  [001_layers_baseline/run-latest/output-Qwen3-14B.json:8398-8413]
- “control_summary": { "first_control_margin_pos": 0, "max_control_margin": 0.9741542933185612 }”  [001_layers_baseline/run-latest/output-Qwen3-14B.json:8419-8423]

KL milestones and units confirmed (bits):
- “first_kl_below_1.0": 40, "first_kl_below_0.5": 40, "first_rank_le_1": 36, "first_rank_le_5": 33, "first_rank_le_10": 32”  [001_layers_baseline/run-latest/output-Qwen3-14B.json:7083-7087]

Prism sidecar is present/compatible; sampled layers at [embed,9,19,29], but metrics indicate regressive KL vs baseline (see Section 3):
- “prism_summary": { "present": true, "compatible": true, "k": 512, "layers": ["embed",9,19,29] … }”  [001_layers_baseline/run-latest/output-Qwen3-14B.json:834-855]

Tuned-Lens sidecar is present with translator temperatures enabled and attribution prepared; reporting prefers tuned:
- “tuned_lens.attribution … prefer_tuned": true”  [001_layers_baseline/run-latest/output-Qwen3-14B.json:8830-8915]

Raw-lens sanity (sample mode) explicitly cautions high lens-artifact risk; prefer rank milestones and confirmed semantics:
- “raw_lens_check.summary": { "first_norm_only_semantic_layer": null, "max_kl_norm_vs_raw_bits": 17.6735, "lens_artifact_risk": "high" }”  [001_layers_baseline/run-latest/output-Qwen3-14B.json:8352-8367]

Copy threshold sweep present; earliest strict layers at τ∈{0.70,0.95} remain null; stability: “none”  [001_layers_baseline/run-latest/output-Qwen3-14B.json:7136-7180].

Lens selection and milestones (baseline):
- “L_surface_to_meaning_norm": 36; “L_geom_norm": 35; “cos_milestones.norm": {ge_0.2: 5, ge_0.4: 29, ge_0.6: 36}”  [001_layers_baseline/run-latest/output-Qwen3-14B.json:7163-7176]
- “depth_fractions": { "L_semantic_frac": 0.9, "first_rank_le_5_frac": 0.825 }”  [001_layers_baseline/run-latest/output-Qwen3-14B.json:7181-7188]

Last-layer CSV consistency check (final row):
- “layer 40 … p_top1 = 0.34514, … kl_to_final_bits = 0.0, answer_rank = 1”  [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:42]

**3. Quantitative Findings**

Table: per-layer NEXT prediction (prompt_id=pos, prompt_variant=orig). Bold indicates confirmed semantic collapse (L_semantic_confirmed=36; confirmed_source=raw).

| L | Entropy (bits) | Top-1 token |
|---|-----------------|-------------|
| 0 | 17.212854 | "梳" |
| 1 | 17.212021 | "地处" |
| 2 | 17.211170 | "是一部" |
| 3 | 17.209875 | "tics" |
| 4 | 17.208380 | "tics" |
| 5 | 17.207327 | "-minded" |
| 6 | 17.205141 | "过去的" |
| 7 | 17.186316 | "�" |
| 8 | 17.179604 | "-minded" |
| 9 | 17.187605 | "-minded" |
| 10 | 17.169565 | " (?)" |
| 11 | 17.151134 | "时代的" |
| 12 | 17.165318 | "といって" |
| 13 | 17.115282 | " nav" |
| 14 | 17.140715 | " nav" |
| 15 | 17.148745 | "唿" |
| 16 | 17.134632 | "闯" |
| 17 | 17.137224 | "唿" |
| 18 | 17.100914 | "____" |
| 19 | 17.075287 | "____" |
| 20 | 16.932322 | "____" |
| 21 | 16.985991 | "年夜" |
| 22 | 16.954144 | "年夜" |
| 23 | 16.839663 | "____" |
| 24 | 16.760223 | "____" |
| 25 | 16.757845 | "年夜" |
| 26 | 16.668522 | "____" |
| 27 | 16.031609 | "____" |
| 28 | 15.234417 | "____" |
| 29 | 14.186926 | "这个名字" |
| 30 | 7.789196 | "这个名字" |
| 31 | 5.161718 | "____" |
| 32 | 0.815953 | "____" |
| 33 | 0.481331 | "____" |
| 34 | 0.594809 | "____" |
| 35 | 0.667881 | "____" |
| **36** | 0.312212 | " Berlin" |
| 37 | 0.905816 | " ____" |
| 38 | 1.212060 | " ____" |
| 39 | 0.952112 | " Berlin" |
| 40 | 3.583520 | " Berlin" |

Control margin: first_control_margin_pos = 0; max_control_margin = 0.9741542933185612  [001_layers_baseline/run-latest/output-Qwen3-14B.json:8419-8423]. Gold alignment is OK; absolute-probability statements are de-emphasized per measurement guidance.

Ablation (no‑filler): L_copy_orig = null, L_sem_orig = 36; L_copy_nf = null, L_sem_nf = 36; ΔL_copy = null, ΔL_sem = 0  [001_layers_baseline/run-latest/output-Qwen3-14B.json:8398-8406].

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (no strict or soft copy layer). Soft ΔHk (k∈{1,2,3}) = n.a. (no soft copy layer).

Confidence milestones (pure CSV): p_top1 > 0.30 at layer 31; p_top1 > 0.60 at layer 32; final-layer p_top1 = 0.3451  [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:42].

Rank milestones (baseline diagnostics): rank ≤ 10 at L32, ≤ 5 at L33, ≤ 1 at L36  [001_layers_baseline/run-latest/output-Qwen3-14B.json:7085-7087].

KL milestones (baseline): first_kl_below_1.0 at L40; first_kl_below_0.5 at L40; KL decreases toward final and is 0.0 at the last layer (well-calibrated head)  [001_layers_baseline/run-latest/output-Qwen3-14B.json:7083-7084, 7816-7832].

Cosine milestones (baseline): cos_to_final ≥ 0.2 at L5, ≥ 0.4 at L29, ≥ 0.6 at L36; final cos_to_final = 0.99999  [001_layers_baseline/run-latest/output-Qwen3-14B.json:7169-7176; 001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:42].

Surface→meaning and coverage (baseline): L_surface_to_meaning_norm = 36 with answer_mass_at_L ≈ 0.9530 and echo_mass ≈ 4.39e-06; L_topk_decay_norm = 0 with topk_prompt_mass_at_L = 0.0  [001_layers_baseline/run-latest/output-Qwen3-14B.json:7163-7169].

Tuned-Lens attribution and milestones (preferred for reporting):
- Rank earliness (tuned − norm): Δ first_rank_le_1 = +3 (39 vs 36); Δ le_5 = +1; Δ le_10 = +1  [001_layers_baseline/run-latest/output-Qwen3-14B.json:8830-8870].
- KL at depth percentiles: ΔKL_tuned ≈ {p25: 4.68, p50: 4.49, p75: 3.90} bits; temperature-only accounts for {−0.077, −0.226, +0.241} bits; rotation gains ΔKL_rot ≈ {4.76, 4.72, 3.66} bits  [001_layers_baseline/run-latest/output-Qwen3-14B.json:8870-8910].
- Tuned surface/geom: L_surface_to_meaning_tuned = 36; L_geom_tuned = 35; L_topk_decay_tuned = 1  [001_layers_baseline/run-latest/output-Qwen3-14B.json:7936-7960].

Prism sidecar analysis: present/compatible but regressive vs baseline.
- Early-depth KL: Prism KL is higher at p25/p50/p75 by ≈{0.26, 0.25, 0.71} bits (negative deltas)  [001_layers_baseline/run-latest/output-Qwen3-14B.json:822-855].
- Rank milestones: prism first_rank_le_{10,5,1} = null (no improvement)  [001_layers_baseline/run-latest/output-Qwen3-14B.json:822-855].
- Copy flags: none flip under Prism (no strict/soft hits in sidecar CSV).
- Verdict: Regressive (higher KL, no earlier rank milestones).

Raw‑vs‑Norm (window/full): sample mode summary shows lens_artifact_risk="high" and large norm-vs-raw KL discrepancies at mid-depth (e.g., L31); full scan tier=high  [001_layers_baseline/run-latest/output-Qwen3-14B.json:8311-8367, 7214-7226]. Prefer rank milestones and confirmed semantics.

**4. Qualitative Patterns & Anomalies**

At the NEXT position, semantics consolidate abruptly late in the stack. “Berlin” becomes rank‑1 at L36 with p≈0.953 (“ Berlin”, 0.9530)  [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:38], while the final layer remains calibrated with lower p_top1≈0.345 and KL_to_final=0 (“ Berlin”, 0.3451; KL=0.0)  [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:42]. Cosine-to-final rises early (≥0.2 by L5; ≥0.4 by L29) while KL to final remains high until late, indicating “early direction, late calibration” typical of norm-lens probes.

Negative control (probe of the model’s general semantics) shows clean behavior for “Berlin is the capital of”: top‑5 is “ Germany, 0.6320; which, 0.2468; the, 0.0737; what, 0.00943; a, 0.00475”  [001_layers_baseline/run-latest/output-Qwen3-14B.json:10-30]. No semantic leakage of “Berlin” into the country slot is expected here; the distribution is coherent.

Important-word trajectory (records CSV; IMPORTANT_WORDS = ["Germany", "Berlin", "capital", "Answer", "word", "simply"]). Around NEXT (pos=15, token “ simply”), ‘Berlin’ enters weakly by L32 (“ Berlin, 5.83e‑04”)  [001_layers_baseline/run-latest/output-Qwen3-14B-records.csv:609], rises through L33–L35, and becomes decisively rank‑1 by L36 (“ Berlin, 0.9530”)  [001_layers_baseline/run-latest/output-Qwen3-14B-records.csv:710]. Adjacent context positions collapse concurrently: at L36 pos=14 (“ called”), ‘Berlin’ rank‑1 p≈0.9933  [001_layers_baseline/run-latest/output-Qwen3-14B-records.csv:709]; at L36 pos=13 (“ is”), ‘Berlin’ rank‑1 p≈0.9851  [001_layers_baseline/run-latest/output-Qwen3-14B-records.csv:708]. This indicates a localized late-stage consolidation around the answer across a small neighborhood of positions.

One‑word instruction ablation (removing “simply”) does not shift semantics: L_sem_nf = 36 equals L_sem_orig = 36 (ΔL_sem = 0)  [001_layers_baseline/run-latest/output-Qwen3-14B.json:8398-8406].

Rest-mass sanity: top‑k rest_mass is extremely low at semantic collapse (L36 rest_mass ≈ 0.00088) and increases at the final layer (0.2364), reflecting reduced top‑k coverage rather than lens fidelity  [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:38,42].

Rotation vs amplification: p_answer and rank improve sharply near L36 while KL_to_final remains >1 bit at L36 (≈1.66 bits) and only vanishes at L40; cosine milestones rise earlier (≥0.2 by L5), indicating an early direction that is calibrated late  [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:38,42; 001_layers_baseline/run-latest/output-Qwen3-14B.json:7169-7176].

Head calibration (final layer): last-layer KL≈0 and temp_est≈1.0 (no softcap/scale applied)  [001_layers_baseline/run-latest/output-Qwen3-14B.json:7816-7832]. Treat probabilities comparatively within model per measurement guidance  [001_layers_baseline/run-latest/output-Qwen3-14B.json:8913-8926].

Lens sanity: raw-lens sample shows high artifact risk; first_norm_only_semantic_layer=null but mid-depth discrepancies are large (e.g., L31 kl_norm_vs_raw_bits ≈ 17.67)  [001_layers_baseline/run-latest/output-Qwen3-14B.json:8330-8367]. The full scan tier is “high”  [001_layers_baseline/run-latest/output-Qwen3-14B.json:7214-7226]. Prefer rank-based milestones and confirmed semantics.

Temperature robustness: at T=0.1, ‘Berlin’ rank‑1 p≈0.974, entropy≈0.173 bits; at T=2.0, ‘Berlin’ rank≈1 with p≈0.036, entropy≈13.161 bits  [001_layers_baseline/run-latest/output-Qwen3-14B.json:667-742].

Checklist
- RMS lens? ✓
- LayerNorm bias removed? ✓ (bias-free cosine)  [001_layers_baseline/run-latest/output-Qwen3-14B.json:826-834]
- Entropy rise at unembed? ✓ (final entropy reported per NEXT row)
- FP32 un-embed promoted? ✓ (unembed_dtype=torch.float32)  [001_layers_baseline/run-latest/output-Qwen3-14B.json:66-69]
- Punctuation / markup anchoring? ✓ (early layers show fillers/punctuation in top‑1; see table)
- Copy-reflex? ✗ (no strict/soft hits in L0–3)
- Grammatical filler anchoring? ✓ (fillers dominate L0–5 top‑1)
- Preferred lens honored in milestones? ✓ (tuned preferred; attribution reported)
- Confirmed semantics reported when available? ✓ (L_semantic_confirmed=36; source=raw)  [001_layers_baseline/run-latest/output-Qwen3-14B.json:8076-8215]
- Full dual‑lens metrics cited? ✓ (pct_layers_kl_ge_1.0, n_norm_only_semantics_layers, earliest_norm_only_semantic, lens_artifact_score tier)  [001_layers_baseline/run-latest/output-Qwen3-14B.json:7214-7226]
- Tuned‑lens attribution done? ✓ (ΔKL_tuned, ΔKL_temp, ΔKL_rot at ≈25/50/75%)  [001_layers_baseline/run-latest/output-Qwen3-14B.json:8870-8910]
- normalization_provenance present? ✓ (ln_source verified at layer 0 and final)  [001_layers_baseline/run-latest/output-Qwen3-14B.json:7226-7260,7481-7560]
- per-layer normalizer effect present? ✓ (resid_norm_ratio, delta_resid_cos)  [001_layers_baseline/run-latest/output-Qwen3-14B.json:7206-7218,7517-7533]
- unembed bias audited? ✓ (present=false)  [001_layers_baseline/run-latest/output-Qwen3-14B.json:826-834]
- deterministic_algorithms = true? ✓  [001_layers_baseline/run-latest/output-Qwen3-14B.json:8373-8396]
- numeric_health clean? ✓ (no NaN/Inf; none flagged)  [001_layers_baseline/run-latest/output-Qwen3-14B.json:7808-7816]
- copy_mask present & plausible? ✓ (size=6112)  [001_layers_baseline/run-latest/output-Qwen3-14B.json:7078-7090]
- layer_map present? ✓ (present in diagnostics)  [001_layers_baseline/run-latest/output-Qwen3-14B.json:7561-7796]

**5. Limitations & Data Quirks**

- Raw‑vs‑norm differences are substantial (lens_artifact_risk=tier “high” and large mid-depth KL deltas), so early “semantics” must be treated cautiously. Prefer rank milestones and the confirmed semantic layer (L=36). Calibration comparisons across model families should rely on ranks/KL thresholds rather than absolute probabilities. Rest_mass is top‑k coverage, not a fidelity metric; its rise after L36 (to 0.236 at L40) reflects narrower top‑k lists rather than loss of fidelity  [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:42].
- Prism is diagnostic-only and regressive here (higher KL, no earlier rank milestones), so it should not override baseline conclusions  [001_layers_baseline/run-latest/output-Qwen3-14B.json:822-855].
- Measurement guidance explicitly flags high lens-artifact risk and normalization spike; follow prefer_ranks=true and suppress_abs_probs=true  [001_layers_baseline/run-latest/output-Qwen3-14B.json:8913-8926].

**6. Model Fingerprint**

Qwen3‑14B: collapse at L 36 (confirmed, raw); final-layer KL=0.0; early cosine direction (≥0.2 by L5) with late calibration; p_top1 crosses 0.60 by L32.

---
Produced by OpenAI GPT-5 
