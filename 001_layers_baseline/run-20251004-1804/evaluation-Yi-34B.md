**Overview**
- Model: `01-ai/Yi-34B` (34B). Probe targets layerwise next-token behavior on the positive prompt ending with “called simply”.
- Run artifacts include baseline norm lens, Tuned Lens, Prism, raw-vs-norm checks, ablation and a France→Paris control.

**Method Sanity-Check**
The probe applies the intended normalization lens and uses rotary position encodings at layer 0, and the context prompt matches the required suffix without a trailing space. Quotes:
- “use_norm_lens: true” [001_layers_baseline/run-latest/output-Yi-34B.json:807]
- “layer0_position_info: "token_only_rotary_model"” [001_layers_baseline/run-latest/output-Yi-34B.json:816]
- “context_prompt: "Give the city name only, plain text. The capital of Germany is called simply"” [001_layers_baseline/run-latest/output-Yi-34B.json:4]

Gold-alignment is ID-level and OK: “gold_alignment: "ok"” [001_layers_baseline/run-latest/output-Yi-34B.json:3525]. Last-layer head calibration is excellent: “kl_to_final_bits: 0.000278…; top1_agree: true; temp_est: 1.0” [001_layers_baseline/run-latest/output-Yi-34B.json:3526–3534]. KL/entropy units are in bits (e.g., columns `kl_to_final_bits`, `teacher_entropy_bits` in CSV; see final-row KL ≈ 0 on [row 63 in CSV]).

Copy detectors present but do not fire (strict or soft). Configuration and thresholds are recorded and mirrored in flags columns:
- “copy_thresh: 0.95; copy_window_k: 1; copy_match_level: "id_subsequence"” [001_layers_baseline/run-latest/output-Yi-34B.json:2501–2503]
- “copy_soft_config: { "threshold": 0.5, "window_ks": [1,2,3] }” [001_layers_baseline/run-latest/output-Yi-34B.json:889–896]
- “copy_flag_columns: [ "copy_strict@0.95", …, "copy_soft_k3@0.5" ]” [001_layers_baseline/run-latest/output-Yi-34B.json:4098–4106]

Normalization provenance is explicit and architecture-aware: “strategy: "next_ln1"” [001_layers_baseline/run-latest/output-Yi-34B.json:2658], “ln_source: "blocks[0].ln1"” at L0 [001_layers_baseline/run-latest/output-Yi-34B.json:2661–2666], and “ln_source: "ln_final"” at the unembed [001_layers_baseline/run-latest/output-Yi-34B.json:3141–3146]. Per-layer normalizer metrics are present (resid_norm_ratio, delta_resid_cos), with an explicit flag: “normalization_spike: true” [001_layers_baseline/run-latest/output-Yi-34B.json:82–84]. Unembedding bias is absent, ensuring bias-free geometry: “present: false; l2_norm: 0.0” [001_layers_baseline/run-latest/output-Yi-34B.json:77–80].

Environment and determinism: “device: "cpu"; deterministic_algorithms: true; seed: 316” [001_layers_baseline/run-latest/output-Yi-34B.json:4089–4096]. Numeric health is clean: “any_nan: false; any_inf: false; layers_flagged: []” [001_layers_baseline/run-latest/output-Yi-34B.json:3518–3525]. Copy-mask size and sample are plausible for tokenizer filtering: “size: 1513” [001_layers_baseline/run-latest/output-Yi-34B.json:2495].

Summary indices exist and are consistent (bits, ranks): “first_kl_below_0.5: 60; first_kl_below_1.0: 60; first_rank_le_1: 44; first_rank_le_5: 44; first_rank_le_10: 43” [001_layers_baseline/run-latest/output-Yi-34B.json:2504–2508].

Raw-vs-Norm window sanity shows high artifact risk and norm-only semantics near the purported collapse layer: “center_layers: [44, 60]; radius: 4; norm_only_semantics_layers: [44,45,46,47,48,56,60]; max_kl_norm_vs_raw_bits_window: 90.4657” [001_layers_baseline/run-latest/output-Yi-34B.json:2610–2630]. The sampled check concurs: “first_norm_only_semantic_layer: 46; lens_artifact_risk: "high"” [001_layers_baseline/run-latest/output-Yi-34B.json:4078–4090]. Full check metrics: “pct_layers_kl_ge_1.0: 0.656; n_norm_only_semantics_layers: 14; earliest_norm_only_semantic: 44; max_kl_norm_vs_raw_bits: 90.4657; tier: "high"” [001_layers_baseline/run-latest/output-Yi-34B.json:2644–2660]. Measurement guidance instructs rank-first reporting and tuned preference: “prefer_ranks: true; suppress_abs_probs: true; preferred_lens_for_reporting: "tuned"; use_confirmed_semantics: true” [001_layers_baseline/run-latest/output-Yi-34B.json:4644–4656].

Gold answer alignment is ID-based and OK; diagnostics and ablation are present: “L_copy: null; L_semantic: 44” [001_layers_baseline/run-latest/output-Yi-34B.json:2497–2499]; “ablation_summary … L_sem_orig: 44; L_sem_nf: 44; delta_L_sem: 0” [001_layers_baseline/run-latest/output-Yi-34B.json:4108–4114]. Control present with France→Paris and positive margins: “first_control_margin_pos: 1; max_control_margin: 0.58357” [001_layers_baseline/run-latest/output-Yi-34B.json:4131–4134].

Prism sidecar present/compatible: “compatible: true; k: 512; layers: [embed, 14, 29, 44]” [001_layers_baseline/run-latest/output-Yi-34B.json:85–93]. Tuned‑Lens is loaded; attribution prefers tuned: “prefer_tuned: true” [001_layers_baseline/run-latest/output-Yi-34B.json:4608–4660].

Copy‑reflex: No early strict copy flags (layers 0–3) and no soft k1@τ_soft hits in L0–L3 in the pure CSV; earliest `is_answer=true` appears later (L44) [row 46 in CSV].

**3. Quantitative Findings**
Table built from `prompt_id = pos`, `prompt_variant = orig` (pure-next-token CSV). Bold marks the semantic layer under the baseline norm lens (L=44); measurement guidance prefers tuned values for summary bullets, but the table remains baseline.

| Layer | Entropy (bits) | Top-1 |
|---|---:|---|
| 0 | 15.962 |  Denote |
| 1 | 15.942 | . |
| 2 | 15.932 | . |
| 3 | 15.839 | MTY |
| 4 | 15.826 | MTY |
| 5 | 15.864 | MTY |
| 6 | 15.829 | MTQ |
| 7 | 15.862 | MTY |
| 8 | 15.873 | 其特征是 |
| 9 | 15.836 | 审理终结 |
| 10 | 15.797 | ~\\\\ |
| 11 | 15.702 | ~\\\\ |
| 12 | 15.774 | ~\\\\ |
| 13 | 15.784 | 其特征是 |
| 14 | 15.739 | 其特征是 |
| 15 | 15.753 | 其特征是 |
| 16 | 15.714 | 其特征是 |
| 17 | 15.714 | 其特征是 |
| 18 | 15.716 | 其特征是 |
| 19 | 15.696 | ncase |
| 20 | 15.604 | ncase |
| 21 | 15.609 | ODM |
| 22 | 15.620 | ODM |
| 23 | 15.602 | ODM |
| 24 | 15.548 | ODM |
| 25 | 15.567 | ODM |
| 26 | 15.585 | ODM |
| 27 | 15.227 | ODM |
| 28 | 15.432 | MTU |
| 29 | 15.467 | ODM |
| 30 | 15.551 | ODM |
| 31 | 15.531 |  版的 |
| 32 | 15.455 | MDM |
| 33 | 15.455 | XFF |
| 34 | 15.477 | XFF |
| 35 | 15.471 | Mpc |
| 36 | 15.433 | MDM |
| 37 | 15.454 | MDM |
| 38 | 15.486 | MDM |
| 39 | 15.504 | MDM |
| 40 | 15.528 | MDM |
| 41 | 15.519 | MDM |
| 42 | 15.535 | keV |
| 43 | 15.518 |  " |
| **44** | 15.327 | ** Berlin** |
| 45 | 15.293 |  Berlin |
| 46 | 14.834 |  Berlin |
| 47 | 14.731 |  Berlin |
| 48 | 14.941 |  Berlin |
| 49 | 14.696 |  Berlin |
| 50 | 14.969 |  Berlin |
| 51 | 14.539 |  Berlin |
| 52 | 15.137 |  Berlin |
| 53 | 14.870 |  Berlin |
| 54 | 14.955 |  Berlin |
| 55 | 14.932 |  Berlin |
| 56 | 14.745 |  Berlin |
| 57 | 14.748 |   |
| 58 | 13.457 |   |
| 59 | 7.191 |   |
| 60 | 2.981 |  Berlin |

Control margin (JSON): “first_control_margin_pos: 1; max_control_margin: 0.583572” [001_layers_baseline/run-latest/output-Yi-34B.json:4131–4134].

Ablation (no-filler): “L_copy_orig: null; L_sem_orig: 44; L_copy_nf: null; L_sem_nf: 44; delta_L_sem: 0” [001_layers_baseline/run-latest/output-Yi-34B.json:4108–4114]. Interpretation: stylistic filler removal does not shift semantic collapse.

Below-table milestones and deltas
- ΔH (bits) = entropy(L_copy) − entropy(L_semantic): n.a. (L_copy_strict = null) [001_layers_baseline/run-latest/output-Yi-34B.json:2497–2504].
- Soft ΔHk (k∈{1,2,3}): n.a. (all L_copy_soft[k] = null) [001_layers_baseline/run-latest/output-Yi-34B.json:2515–2523].
- Confidence milestones (CSV): p_top1 > 0.30 at layer 60; p_top1 > 0.60 not reached; final p_top1 = 0.5555 [row 63 in CSV]. Note: measurement_guidance requests rank-first reporting.
- Rank milestones — preferred lens tuned (JSON): tuned {≤10: 44, ≤5: 44, ≤1: 46}; baseline {≤10: 43, ≤5: 44, ≤1: 44} [001_layers_baseline/run-latest/output-Yi-34B.json:4557–4590 and 2504–2508].
- KL milestones (JSON): first_kl_below_1.0: 60; first_kl_below_0.5: 60; final KL ≈ 0 [001_layers_baseline/run-latest/output-Yi-34B.json:2504–2505, 3526–3534].
- Cosine milestones — norm (JSON): ge_0.2: 1; ge_0.4: 44; ge_0.6: 51 [001_layers_baseline/run-latest/output-Yi-34B.json:2612–2620]. Final cosine (CSV): cos_to_final ≈ 1.0 at L60 [row 63 in CSV].
- Depth fractions (JSON): L_semantic_frac = 0.733 [001_layers_baseline/run-latest/output-Yi-34B.json:2604].
- Copy robustness (JSON): copy_thresholds.stability = "none"; earliest L_copy_strict at τ∈{0.70,0.95} = null; norm_only_flags also null [001_layers_baseline/run-latest/output-Yi-34B.json:2557–2582].
 - Tuned‑Lens attribution (JSON, percentiles): ΔKL_tuned/ΔKL_temp/ΔKL_rot ≈
   • p25: 5.88 / 0.91 / 4.97; • p50: 6.10 / 1.28 / 4.82; • p75: 8.39 / 4.39 / 3.998 [001_layers_baseline/run-latest/output-Yi-34B.json:4584–4608]. Rank earliness shifts: tuned first_rank_le_1 at L46 vs L44 baseline [001_layers_baseline/run-latest/output-Yi-34B.json:4563–4590, 2506].
 - Entropy drift (CSV): at mid‑depth L45, entropy ≈ 15.293 bits vs teacher_entropy_bits ≈ 2.941 (drift ≈ +12.35 bits) [row 47 in CSV].

Prism Sidecar Analysis
- Presence/compatibility: “compatible: true; k: 512; layers: [embed,14,29,44]” [001_layers_baseline/run-latest/output-Yi-34B.json:85–93].
- Early-depth stability (KL, JSON percentiles): baseline p25/p50/p75 = 13.12/13.54/11.16 bits vs Prism 12.18/12.17/12.17; Δ = +0.94/+1.36/−1.01 bits [001_layers_baseline/run-latest/output-Yi-34B.json:855–874].
- Rank milestones (Prism pure CSV): not achieved within probed depths (first_rank_le_{10,5,1} = null) — consistent with JSON summary [001_layers_baseline/run-latest/output-Yi-34B.json:834–853].
- Copy flags: no spurious flips (no `copy_collapse=True` in Prism CSV).
- Verdict: Neutral/ambiguous — modest KL gains at shallow/mid depths but no earlier rank-1.

**4. Qualitative Patterns & Anomalies**
Negative control shows no leakage: for “Berlin is the capital of”, top-5 are Germany-heavy and punctuation, without “Berlin”: “ Germany, 0.8398; the, 0.0537; which, 0.0288; what, 0.0120; Europe, 0.0060” [001_layers_baseline/run-latest/output-Yi-34B.json:10–31]. By contrast, stylistic phrasings with “called”/“named” often put “ Berlin” top-1 in final predictions (test_prompts).

Important-word trajectory (records CSV): “Berlin” first appears with small mass before collapse and then strengthens. Example pre-collapse: “Berlin, 0.000527…” at L43,pos=16 [001_layers_baseline/run-latest/output-Yi-34B-records.csv:770]. Collapse begins at L44 with `is_answer=True` and rank‑1 under the norm lens: “ Berlin, p=0.00846; answer_rank=1” [row 46 in CSV]. It then amplifies steadily through L49–L56 (e.g., L49 p≈0.0470 [row 51 in CSV]) before saturating at the final head (L60 p=0.5555; KL≈0) [row 63 in CSV].

Instructional brevity sensitivity: removing the filler (“no_filler”) leaves L_sem unchanged (44 vs 44; ΔL_sem=0) [001_layers_baseline/run-latest/output-Yi-34B.json:4108–4114], indicating collapse timing is not anchored on that stylistic token here.

Rest‑mass sanity (CSV): rest_mass remains high after L_semantic (e.g., 0.9807 at L44 [row 46 in CSV]) and falls late to 0.1753 at L60 [row 63 in CSV]. This reflects limited top‑k coverage early; per guidance, treat as coverage only, not lens fidelity.

Rotation vs amplification: cosine direction aligns early (ge_0.2 at L1; ge_0.4 at L44) while KL stays high until late (first_kl_below_1.0 at L60) [001_layers_baseline/run-latest/output-Yi-34B.json:2612–2620, 2504–2505]. This is an “early direction, late calibration” pattern — consistent with logit‑lens expectations that rotational alignment precedes calibration to the final head.

Final‑head calibration: “warn_high_last_layer_kl: false; temp_est: 1.0; kl_after_temp_bits ≈ 0.000278” [001_layers_baseline/run-latest/output-Yi-34B.json:3526–3536]. Treat absolute probabilities comparatively within‑model per measurement_guidance.

Lens sanity: raw‑vs‑norm checks flag norm‑only semantics starting near L44 with very large KL gaps to raw (≈80–90 bits) [001_layers_baseline/run-latest/output-Yi-34B.json:4022–4090, 2610–2630]. Risk tier is “high” with 14 norm‑only semantic layers and max KL≈90.47 bits [001_layers_baseline/run-latest/output-Yi-34B.json:2644–2660]. Prefer rank milestones and confirmed semantics; indeed, “confirmed_semantics … L_semantic_confirmed: 44 (source: tuned)” [001_layers_baseline/run-latest/output-Yi-34B.json:3905–3913].

Temperature robustness (JSON): at T=0.1, “Berlin” rank‑1 with p≈0.9999996; entropy ≈ 0 [001_layers_baseline/run-latest/output-Yi-34B.json:11–19]. At T=2.0, “Berlin” remains rank‑1 but diffuse (p≈0.0488; entropy ≈12.49 bits) [001_layers_baseline/run-latest/output-Yi-34B.json:77–84].

Important‑word evolution: “called/simply” layers carry filler/punctuation and generic tokens; “Berlin” enters the top‑lists by L43 and stabilizes rank‑1 at L44–L46 with rising cosine to final; “Germany/capital” remain frequent distractors in mid‑layers (records CSV around L44–L49: capital, Germany co‑present [001_layers_baseline/run-latest/output-Yi-34B-records.csv:786–878]).

Checklist
- RMS lens? ✓ (RMSNorm; strategy next_ln1) [001_layers_baseline/run-latest/output-Yi-34B.json:2658–2666]
- LayerNorm bias removed? ✓ (bias‑free unembed) [001_layers_baseline/run-latest/output-Yi-34B.json:77–80]
- Entropy rise at unembed? ✗ (final entropy is low, 2.98 bits) [row 63 in CSV]
- FP32 un‑embed promoted? ✓ (“use_fp32_unembed: true; unembed_dtype: torch.float32”) [001_layers_baseline/run-latest/output-Yi-34B.json:58–60]
- Punctuation / markup anchoring? ✓ (early layers dominated by punctuation/markers; see rows up to L14)
- Copy‑reflex? ✗ (no strict or soft copy flags in L0–L3)
- Grammatical filler anchoring? ✓ (frequent tokens {is, the, of, a} and quotes around L40s–L50s; e.g., records at L44–L47)
- Preferred lens honored in milestones? ✓ (tuned rank milestones foregrounded)
- Confirmed semantics reported? ✓ (“L_semantic_confirmed: 44; source: tuned”) [001_layers_baseline/run-latest/output-Yi-34B.json:3909–3913]
- Full dual‑lens metrics cited? ✓ (raw_lens_full pct≥1.0, n_norm_only_semantics, earliest, max KL, tier) [001_layers_baseline/run-latest/output-Yi-34B.json:2644–2660]
- Tuned‑lens attribution done? ✓ (ΔKL_tuned, ΔKL_temp, ΔKL_rot at ~25/50/75%) [001_layers_baseline/run-latest/output-Yi-34B.json:4584–4608]
- normalization_provenance present? ✓ (ln_source verified at L0 and final) [001_layers_baseline/run-latest/output-Yi-34B.json:2661–2666, 3141–3146]
- Per‑layer normalizer effect metrics present? ✓ (resid_norm_ratio, delta_resid_cos) [001_layers_baseline/run-latest/output-Yi-34B.json:2665–3146]
- Unembed bias audited? ✓ (bias‑free cosine guaranteed) [001_layers_baseline/run-latest/output-Yi-34B.json:77–80]
- deterministic_algorithms = true? ✓ [001_layers_baseline/run-latest/output-Yi-34B.json:4093]
- Numeric health clean? ✓ (no NaN/Inf) [001_layers_baseline/run-latest/output-Yi-34B.json:3518–3525]
- Copy mask present and plausible? ✓ (“size: 1513”) [001_layers_baseline/run-latest/output-Yi-34B.json:2495]
- layer_map present for indexing? ✓ (norm vs post‑block map enumerated) [001_layers_baseline/run-latest/output-Yi-34B.json:3145–3515]

**5. Limitations & Data Quirks**
- High lens‑artifact risk (raw‑vs‑norm): norm‑only semantics at/around L44 with extremely large KL gaps to raw (≈80–90 bits). Prefer rank milestones and confirmed semantics; avoid cross‑model probability comparisons. [001_layers_baseline/run-latest/output-Yi-34B.json:4022–4090, 2610–2660]
- Measurement guidance explicitly requests rank‑first reporting and tuned preference; absolute probabilities suppressed for cross‑family statements [001_layers_baseline/run-latest/output-Yi-34B.json:4644–4656].
- Rest_mass reflects top‑k coverage only and is high post‑collapse until late (e.g., 0.9807 at L44), so treat as coverage not fidelity (CSV rows 46–60). 
- KL is lens‑sensitive; final KL≈0 confirms last‑head alignment, but early KL trends should be interpreted qualitatively within‑model.

**6. Model Fingerprint**
Yi‑34B: collapse at L 44 (confirmed by tuned); final entropy 2.98 bits; direction aligns early (cos≥0.4 by L 44) and calibrates late (KL<1 by L 60).

---
Produced by OpenAI GPT-5
