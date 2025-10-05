# Evaluation Report: Qwen/Qwen2.5-72B

## 1. Overview
Qwen/Qwen2.5-72B (80 layers; pre-norm; RMSNorm) was probed layer-by-layer with a norm lens to track ranks, KL-to-final, cosine geometry, and surface-vs-meaning mass. The run focuses on a single factual prompt (“Germany → Berlin”) with stylistic ablation and a France→Paris control; Prism sidecar was available for diagnostic calibration.

Quotes: “model: Qwen/Qwen2.5-72B” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:805]; “num_layers: 80; architecture: pre_norm” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8690–8698].

## 2. Method sanity‑check
The intended norm lens and positional regime are in effect. Diagnostics confirm use of the norm lens, FP32 unembed, and RMSNorm: “use_norm_lens: true; use_fp32_unembed: true; unembed_dtype: torch.float32; first_block_ln1_type: RMSNorm; final_ln_type: RMSNorm” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:807–811]. Layer‑0 is token‑only rotary: “layer0_position_info: token_only_rotary_model” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:816]. The context prompt matches and ends with “called simply”: “context_prompt: … is called simply” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:4].

Normalizer provenance matches pre‑norm γ selection: “normalization_provenance.strategy: next_ln1” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7396]; per‑layer ln_source starts at “blocks[0].ln1” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7400] and uses “ln_final” at L80 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8040]. Residual metrics are present; early spikes are flagged (“flags.normalization_spike: true” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:832]).

Copy detectors and labels are configured as required. Strict: “copy_thresh: 0.95; copy_window_k: 1; copy_match_level: id_subsequence” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7252–7254]. Soft defaults: “copy_soft_config.threshold: 0.5; window_ks: [1,2,3]” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:900–921]. CSV/JSON flag columns mirror these: “copy_flag_columns: ["copy_strict@0.95", …, "copy_soft_k1@0.5", …]” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8775–8780]. Gold alignment is OK: “gold_alignment: ok” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8543].

Environment and determinism are recorded: “deterministic_algorithms: true; device: cpu; dtype_compute: torch.bfloat16; seed: 316” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8768–8776]. Unembed bias is absent: “unembed_bias.present: false; l2_norm: 0.0” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:826–833]. Numeric health is clean: “any_nan: false; any_inf: false; layers_flagged: []” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8537–8545].

Last‑layer head calibration is good: “kl_to_final_bits: 0.000109…; top1_agree: true; p_top1_lens: 0.3395; p_top1_model: 0.3383; temp_est: 1.0; warn_high_last_layer_kl: false” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8548–8564]. Units are bits; see “teacher_entropy_bits” and “kl_to_final_bits” columns in CSV [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:61–63].

Raw‑vs‑Norm sanity (window): “center_layers: [78,80]; radius: 4; norm_only_semantics_layers: [80]; max_kl_norm_vs_raw_bits_window: 83.3150” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7379–7391]. Raw‑lens (sample) summary: “lens_artifact_risk: high; max_kl_norm_vs_raw_bits: 19.9099; first_norm_only_semantic_layer: null” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8699, 8756–8758]. Raw‑lens (full) summary: “pct_layers_kl_ge_1.0: 0.321; n_norm_only_semantics_layers: 1; earliest_norm_only_semantic: 80; max_kl_norm_vs_raw_bits: 83.3150; tier: high” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7387, 8705–8714]. Caution: early semantics under the norm lens may be lens‑induced; prefer rank milestones.

Measurement guidance: “prefer_ranks: true; suppress_abs_probs: true; preferred_lens_for_reporting: "norm"; use_confirmed_semantics: false; reasons: ["norm_only_semantics_window","high_lens_artifact_risk",…]” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8816–8827].

Copy thresholds sweep is present but strict copy is null at all τ: “L_copy_strict {0.7,0.8,0.9,0.95} = null; stability: none” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:904–916, 924–928]. No strict or soft copy flags fire in layers 0–3 in the pure CSV (scanned; no True for copy_collapse or copy_soft_k1@0.5).

Control prompt block is present; control_summary is null: “control_prompt: … France … gold_alignment: ok” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8793–8805]; “control_summary: null” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:873]. Ablation is present: “L_sem_orig: 80; L_sem_nf: 80; delta_L_sem: 0” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8784–8790].

Prism sidecar is available and compatible: “prism_summary.present: true; compatible: true; k: 512; layers: [embed,19,39,59]” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:836–851]. Prism metrics show early KL reductions: “delta p25: 3.162 bits; p50: 2.834; p75: −0.543” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:865–885]; rank milestones under Prism are unavailable (null) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:856–864].

## 3. Quantitative findings
Table (pos, orig). Layer-wise entropy (bits) and generic top‑1 token. Bold marks the semantic layer under the norm lens (L_semantic_norm = 80).

| Layer | Entropy (bits) | Top-1 token |
|---:|---:|:---|
| L 0 | 17.214 | "s" |
| L 1 | 17.214 | "下一篇" |
| L 2 | 17.143 | "ولوج" |
| L 3 | 17.063 | "شدد" |
| L 4 | 17.089 | ".myapplication" |
| L 5 | 17.007 | "ستحق" |
| L 6 | 17.031 | ".myapplication" |
| L 7 | 16.937 | ".myapplication" |
| L 8 | 16.798 | ".myapplication" |
| L 9 | 16.120 | "ستحق" |
| L 10 | 16.501 | ".myapplication" |
| L 11 | 16.718 | ".myapplication" |
| L 12 | 16.778 | "かもしれ" |
| L 13 | 16.631 | "かもしれ" |
| L 14 | 16.359 | "かもしれ" |
| L 15 | 16.517 | "のではない" |
| L 16 | 16.491 | "iéndo" |
| L 17 | 16.213 | "iéndo" |
| L 18 | 16.109 | "有期徒" |
| L 19 | 15.757 | "有期徒" |
| L 20 | 16.129 | "有期徒" |
| L 21 | 16.156 | "有期徒" |
| L 22 | 15.980 | "有期徒" |
| L 23 | 16.401 | ".myapplication" |
| L 24 | 15.999 | "iéndo" |
| L 25 | 15.351 | "hế" |
| L 26 | 15.944 | "iéndo" |
| L 27 | 15.756 | "iéndo" |
| L 28 | 15.750 | ".myapplication" |
| L 29 | 15.885 | ".myapplication" |
| L 30 | 16.123 | ".myapplication" |
| L 31 | 16.170 | ".myapplication" |
| L 32 | 16.171 | ".myapplication" |
| L 33 | 16.419 | "hế" |
| L 34 | 16.200 | "iéndo" |
| L 35 | 16.455 | "hế" |
| L 36 | 16.408 | "iéndo" |
| L 37 | 16.210 | "iéndo" |
| L 38 | 16.490 | "hế" |
| L 39 | 16.418 | "iéndo" |
| L 40 | 16.192 | "iéndo" |
| L 41 | 16.465 | "hế" |
| L 42 | 16.595 | "hế" |
| L 43 | 16.497 | "hế" |
| L 44 | 16.655 | "続きを読む" |
| L 45 | 16.877 | "国际在线" |
| L 46 | 17.002 | "国际在线" |
| L 47 | 17.013 | "主义思想" |
| L 48 | 17.022 | "主义思想" |
| L 49 | 17.022 | " reuseIdentifier" |
| L 50 | 16.968 | "uckets" |
| L 51 | 16.972 | " "" |
| L 52 | 17.009 | """ |
| L 53 | 16.927 | """ |
| L 54 | 16.908 | """ |
| L 55 | 16.942 | """ |
| L 56 | 16.938 | """ |
| L 57 | 16.841 | " "" |
| L 58 | 16.915 | " "" |
| L 59 | 16.920 | " "" |
| L 60 | 16.886 | " '" |
| L 61 | 16.903 | " '" |
| L 62 | 16.834 | " "" |
| L 63 | 16.891 | " "" |
| L 64 | 16.895 | " "" |
| L 65 | 16.869 | " "" |
| L 66 | 16.899 | " "" |
| L 67 | 16.893 | " "" |
| L 68 | 16.779 | " "" |
| L 69 | 16.876 | " "" |
| L 70 | 16.787 | " "" |
| L 71 | 16.505 | " "" |
| L 72 | 16.650 | " "" |
| L 73 | 15.787 | " "" |
| L 74 | 16.081 | " "" |
| L 75 | 13.350 | " "" |
| L 76 | 14.743 | " "" |
| L 77 | 10.848 | " "" |
| L 78 | 15.398 | " "" |
| L 79 | 16.666 | " "" |
| **L 80** | 4.116 | " Berlin" |

Control margin (JSON) unavailable: “control_summary: null” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:873].

Ablation (no‑filler): L_copy_orig = null; L_sem_orig = 80; L_copy_nf = null; L_sem_nf = 80; ΔL_copy = null; ΔL_sem = 0 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8784–8790].

Additional milestones and diagnostics (norm lens unless noted):
- Rank milestones (diagnostics): first_rank≤10 at L74; ≤5 at L78; ≤1 at L80 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7259, 7258, 7257].
- KL milestones (bits): first_kl_below_1.0 at L80; first_kl_below_0.5 at L80 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7256, 7255]. Final-layer KL≈0: “kl_to_final_bits: 0.000109…” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8548].
- Cosine milestones: ge_0.2 at L0; ge_0.4 at L0; ge_0.6 at L53 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7346–7352]. Final cosine: cos_to_final ≈ 1.00 at L80 [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:65].
- Surface→meaning and masses: L_surface_to_meaning_norm = 80; answer_mass_at_L ≈ 0.3395; echo_mass_at_L ≈ 0.0446 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7335; 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:69–71].
- Coverage/decay: L_topk_decay_norm = 0 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7341].
- Norm temperature calibration snapshots (KL vs teacher at τ): @25%≈9.844, @50%≈9.814, @75%≈7.634 bits [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8564–8575].
- Rest‑mass sanity: final rest_mass ≈ 0.298 (L80) [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:47].

Copy robustness (threshold sweep): stability = “none”; earliest strict copy at τ=0.70 and τ=0.95 are null [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:904–928].

Prism sidecar: Early‑depth KL deltas indicate lower KL at p25/p50 by ≈3.16/2.83 bits, with a slight regression at p75 (−0.54). Rank milestones under Prism are null, so we cannot assert earlier rank collapse; verdict: Neutral to mildly Helpful at early/mid depths [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:856–885].

## 4. Qualitative patterns & anomalies
Negative control (prompt: “Berlin is the capital of”): Top‑5 shows strong “ Germany” (rank‑1) and function words, no “Berlin” leakage: “ Germany, 0.7695; the, 0.0864; which, 0.0491; a, 0.0125; what, 0.0075” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:14–31].

Important‑word trajectory (records CSV; IMPORTANT_WORDS=[“Germany”, “Berlin”, “capital”, “Answer”, “word”, “simply”] [001_layers_baseline/run.py:391]). Around NEXT (pos=15, token “ simply”), ‘Berlin’ enters the top‑20 late and strengthens: at L74 pos=15 “ Berlin, 0.000953” [001_layers_baseline/run-latest/output-Qwen2.5-72B-records.csv:3911]; L76 pos=15 “ Berlin, 0.001379” [001_layers_baseline/run-latest/output-Qwen2.5-72B-records.csv:3966]; L77 pos=15 “ Berlin, 0.004809” [001_layers_baseline/run-latest/output-Qwen2.5-72B-records.csv:3999]; final L80 pos=15: “ Berlin, 0.3395 (top‑1)” [001_layers_baseline/run-latest/output-Qwen2.5-72B-records.csv:4106]. Adjacent positions show earlier salience: L74 pos=13 “ Berlin, 0.002963 (top‑2)” [001_layers_baseline/run-latest/output-Qwen2.5-72B-records.csv:3909]; L74 pos=14 “ Berlin, 0.001571 (top‑4)” [001_layers_baseline/run-latest/output-Qwen2.5-72B-records.csv:3910]. Late‑stack predictions are dominated by punctuation/quotes until collapse (e.g., layers 74–79) [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token-rawlens-window.csv:1–15].

Collapse layer stability without the “one‑word” adverb: ablation shows L_sem_nf = 80, ΔL_sem = 0 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8788–8790].

Rest‑mass sanity: rest_mass falls by the final layer; after L_semantic the maximum observed is ≈0.298 at L80 [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:47].

Rotation vs amplification: cos_to_final rises early (≥0.4 by L0; ≥0.6 by L53), while KL to final remains high until late, indicating “early direction, late calibration” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7346–7352; 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:65]. Final‑layer head calibration is excellent (KL≈0; top‑1 agree) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8548–8564].

Lens sanity: Raw‑vs‑Norm shows high artifact risk (sample: “lens_artifact_risk: high; max_kl_norm_vs_raw_bits: 19.9099” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8756–8758]; full: “max_kl_norm_vs_raw_bits: 83.3150; earliest_norm_only_semantic: 80; tier: high” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7387, 8705–8714]). Treat any pre‑final “early semantics” cautiously; rely on rank milestones. Copy‑threshold sweep shows no strict copy at τ∈{0.70,0.95}; soft detectors also do not fire in early layers (pure CSV). Copy mask is large and plausible for tokenizer punctuation: “copy_mask … size≈6.2k; sample ["!","\"","#","$","%"]” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:981; jq sample].

Temperature robustness (final distribution exploration): at T=0.1, ‘Berlin’ is rank‑1 (entropy≈0.275 bits) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:670–674, 668–679]; at T=2.0, ‘Berlin’ remains in top‑k with much higher entropy≈15.013 bits [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:737–741, 736–751].

Checklist:
✓ RMS lens?
✓ LayerNorm bias removed? (RMS model; not needed) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:812]
✓ Entropy rise at unembed? (teacher_entropy_bits present) [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:61]
✓ FP32 un‑embed promoted? [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:808–809, 815]
✓ Punctuation / markup anchoring? (late layers 74–79) [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token-rawlens-window.csv:1–15]
✗ Copy‑reflex? (no strict or soft hits in L0–3; pure CSV)
✓ Grammatical filler anchoring? (quotes/commas dominate L74–79) [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token-rawlens-window.csv:1–15]
✓ Preferred lens honored in milestones (norm per guidance) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8816–8827]
✗ Confirmed semantics reported (no confirmed source; L_semantic_confirmed = null) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8588–8592]
✓ Full dual‑lens metrics cited (pct_layers_kl_ge_1.0, n_norm_only_semantics_layers, earliest_norm_only_semantic, tier) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8699–8714]
n.a. Tuned‑lens attribution (missing) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8576–8582]
✓ normalization_provenance present (ln_source at layer 0 and final) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7400, 8040]
✓ per‑layer normalizer effect metrics present (resid_norm_ratio, delta_resid_cos) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7398–8045]
✓ unembed bias audited (bias‑free cosine guaranteed) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:826–833]
✓ deterministic_algorithms = true [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8768]
✓ numeric_health clean (no NaN/Inf; no flagged early layers) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8537–8545]
✓ copy_mask present and plausible [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:981]
✓ layer_map present (indexing audit) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8529–8533]

## 5. Limitations & data quirks
- Measurement guidance requests rank‑first and suppresses absolute probabilities due to norm‑only semantics window and high lens artifact risk [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8816–8827]. Treat cross‑model probability comparisons with caution.
- Raw‑vs‑Norm differences are substantial (max KL≈83.3 bits full; risk tier: high), and norm‑only semantics at L80 are observed. Prefer rank milestones and within‑model trends; treat early “semantics” as lens‑induced when not corroborated [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7387, 8705–8714].
- Control summary fields are null for this run (no control_margin time‑series in JSON) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:873].
- Surface‑mass is tokenizer‑dependent; compare within model runs only. Rest_mass ≈0.30 at final indicates residual top‑k truncation mass and is not a fidelity metric [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:47].

## 6. Model fingerprint
Qwen‑2.5‑72B: collapse at L 80; final KL≈0.0001 bits; cosine≥0.6 by L 53; ‘Berlin’ stabilizes top‑1 only at the final layer.

---
Produced by OpenAI GPT-5
