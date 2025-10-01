**Overview**
- Model: 01-ai/Yi-34B (60 transformer blocks; pre-norm). The probe runs a norm-lens logit‑lens sweep over the NEXT token of a short fact prompt and records per‑layer entropy, ranks, KL to final, surface masses, and geometry. Semantic collapse (gold token Berlin) occurs mid‑late stack with high final head alignment.

**Method Sanity‑Check**
- Context and lensing are as intended. The prompt ends with “called simply” (no trailing space): “Give the city name only, plain text. The capital of Germany is called simply” [001_layers_baseline/run-latest/output-Yi-34B.json:817]. Norm lens is active with FP32 unembed and RMS norms: “use_norm_lens: true … unembed_dtype: torch.float32 … first_block_ln1_type: RMSNorm … layer0_position_info: token_only_rotary_model” [001_layers_baseline/run-latest/output-Yi-34B.json:807–816,809,810,816].
- Gold alignment is OK and ID‑level: gold ‘Berlin’ has first_id=19616 and variant=with_space [001_layers_baseline/run-latest/output-Yi-34B.json:2144–2152].
- Copy detector configuration present: “copy_thresh: 0.95, copy_window_k: 1, copy_match_level: id_subsequence … copy_soft_threshold: 0.5, copy_soft_window_ks: [1,2,3]” [001_layers_baseline/run-latest/output-Yi-34B.json:956–969,964–969]; strict τ sweep list and stability recorded with all null L_copy_strict and stability="none" [001_layers_baseline/run-latest/output-Yi-34B.json:1009–1034]. The CSV/JSON expose matching copy flag columns “copy_strict@{0.95,0.7,0.8,0.9}” and “copy_soft_k{1,2,3}@0.5” [001_layers_baseline/run-latest/output-Yi-34B.json:1620–1628].
- Summary indices present: first_rank_le_10=43, le_5=44, le_1=44; first_kl_below_1.0=60, first_kl_below_0.5=60; L_semantic=44; L_copy=null [001_layers_baseline/run-latest/output-Yi-34B.json:952–963,959–963]. Units for KL/entropy are bits throughout (teacher_entropy_bits present in CSV; see Section 3).
- Last‑layer head calibration is excellent: kl_to_final_bits=0.000278; top‑1 agrees; p_top1_lens≈0.556 vs p_top1_model≈0.563; temp_est=1.0; warn_high_last_layer_kl=false [001_layers_baseline/run-latest/output-Yi-34B.json:1096–1115]. Final‑row CSV KL≈0 corroborates.
- Measurement guidance advises rank‑led reporting due to norm‑only semantics risk: “prefer_ranks: true … reasons: [norm_only_semantics_window, high_lens_artifact_risk]” [001_layers_baseline/run-latest/output-Yi-34B.json:2134–2141].
- Raw‑vs‑Norm window: center_layers=[44,60], radius=4, layers_checked list provided; norm‑only semantics layers include {44,45,46,47,48,56,60}; max_kl_norm_vs_raw_bits_window=90.47 [001_layers_baseline/run-latest/output-Yi-34B.json:1061–1094]. Lens sanity (sampled): first_norm_only_semantic_layer=46; lens_artifact_risk="high" [001_layers_baseline/run-latest/output-Yi-34B.json:1558–1618].
- Threshold sweep sanity: copy_thresholds present with stability="none" and earliest L_copy_strict(τ=0.70)=null, L_copy_strict(τ=0.95)=null [001_layers_baseline/run-latest/output-Yi-34B.json:1009–1020,1033–1034].
- Copy‑collapse flags in pure CSV: no row has copy_collapse=True in L0–L3; soft flags copy_soft_k1@0.5 never fire in early layers (or at all). Rule holds; no spurious strict copy found.

**Quantitative Findings**
| Layer | Entropy (bits) | Top‑1 token |
|---:|---:|:---|
| L 0 | 15.962 | 'Denote' |
| L 1 | 15.942 | '.' |
| L 2 | 15.932 | '.' |
| L 3 | 15.839 | 'MTY' |
| L 4 | 15.826 | 'MTY' |
| L 5 | 15.864 | 'MTY' |
| L 6 | 15.829 | 'MTQ' |
| L 7 | 15.862 | 'MTY' |
| L 8 | 15.873 | '其特征是' |
| L 9 | 15.836 | '审理终结' |
| L 10 | 15.797 | '~\\\\' |
| L 11 | 15.702 | '~\\\\' |
| L 12 | 15.774 | '~\\\\' |
| L 13 | 15.784 | '其特征是' |
| L 14 | 15.739 | '其特征是' |
| L 15 | 15.753 | '其特征是' |
| L 16 | 15.714 | '其特征是' |
| L 17 | 15.714 | '其特征是' |
| L 18 | 15.716 | '其特征是' |
| L 19 | 15.696 | 'ncase' |
| L 20 | 15.604 | 'ncase' |
| L 21 | 15.609 | 'ODM' |
| L 22 | 15.620 | 'ODM' |
| L 23 | 15.602 | 'ODM' |
| L 24 | 15.548 | 'ODM' |
| L 25 | 15.567 | 'ODM' |
| L 26 | 15.585 | 'ODM' |
| L 27 | 15.227 | 'ODM' |
| L 28 | 15.432 | 'MTU' |
| L 29 | 15.467 | 'ODM' |
| L 30 | 15.551 | 'ODM' |
| L 31 | 15.531 | '版的' |
| L 32 | 15.455 | 'MDM' |
| L 33 | 15.455 | 'XFF' |
| L 34 | 15.477 | 'XFF' |
| L 35 | 15.471 | 'Mpc' |
| L 36 | 15.433 | 'MDM' |
| L 37 | 15.454 | 'MDM' |
| L 38 | 15.486 | 'MDM' |
| L 39 | 15.504 | 'MDM' |
| L 40 | 15.528 | 'MDM' |
| L 41 | 15.519 | 'MDM' |
| L 42 | 15.535 | 'keV' |
| L 43 | 15.518 | '"' |
| **L 44** | 15.327 | 'Berlin' |
| L 45 | 15.293 | 'Berlin' |
| L 46 | 14.834 | 'Berlin' |
| L 47 | 14.731 | 'Berlin' |
| L 48 | 14.941 | 'Berlin' |
| L 49 | 14.696 | 'Berlin' |
| L 50 | 14.969 | 'Berlin' |
| L 51 | 14.539 | 'Berlin' |
| L 52 | 15.137 | 'Berlin' |
| L 53 | 14.870 | 'Berlin' |
| L 54 | 14.955 | 'Berlin' |
| L 55 | 14.932 | 'Berlin' |
| L 56 | 14.745 | 'Berlin' |
| L 57 | 14.748 | '' |
| L 58 | 13.457 | '' |
| L 59 | 7.191 | '' |
| L 60 | 2.981 | 'Berlin' |

- Control margin (France→Paris control): first_control_margin_pos=1; max_control_margin=0.5836 [001_layers_baseline/run-latest/output-Yi-34B.json:1652–1655]. Gold alignment for both gold tokens is OK [001_layers_baseline/run-latest/output-Yi-34B.json:1650].
- Ablation (no‑filler): L_copy_orig=null, L_sem_orig=44; L_copy_nf=null, L_sem_nf=44; ΔL_copy=null, ΔL_sem=0 [001_layers_baseline/run-latest/output-Yi-34B.json:1629–1636].
- ΔH (bits) = entropy(L_copy) − entropy(L_semantic): not computed (no strict or soft copy layer detected).
- Soft ΔH_k (k∈{1,2,3}): not computed (L_copy_soft[k]=null for all k) [001_layers_baseline/run-latest/output-Yi-34B.json:970–978].
- Confidence milestones (pure CSV): p_top1>0.30 at L60; p_top1>0.60 not reached; final‑layer p_top1=0.556 (norm lens; matches model head within ~0.7%) [001_layers_baseline/run-latest/output-Yi-34B.json:1096–1104].
- Rank milestones (diagnostics): rank≤10 at L43; rank≤5 at L44; rank≤1 at L44 [001_layers_baseline/run-latest/output-Yi-34B.json:961–963].
- KL milestones (diagnostics): first_kl_below_1.0 at L60; first_kl_below_0.5 at L60; KL decreases with depth and is ≈0 at final (see last_layer_consistency) [001_layers_baseline/run-latest/output-Yi-34B.json:959–960,1096–1104].
- Cosine milestones (JSON): cos_to_final ≥0.2 at L1; ≥0.4 at L44; ≥0.6 at L51; final cos_to_final≈1.0 [001_layers_baseline/run-latest/output-Yi-34B.json:1046–1051; 001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:60].
- Copy robustness: copy_thresholds.stability="none"; earliest strict copy L(τ=0.70)=null; L(τ=0.95)=null; no norm‑only flags (all null) [001_layers_baseline/run-latest/output-Yi-34B.json:1009–1034].

Prism Sidecar Analysis
- Presence: compatible=true; artifacts loaded [001_layers_baseline/run-latest/output-Yi-34B.json:825–833].
- Early‑depth stability (KL vs final, percentiles): baseline p25=13.12, p50=13.54, p75=11.16 vs prism p25=12.18, p50=12.17, p75=12.17; deltas +0.94, +1.36, −1.01 bits [001_layers_baseline/run-latest/output-Yi-34B.json:856–871]. Sampled layers show mixed effects (e.g., L15/L30 improve; L45 degrades).
- Rank milestones (prism pure CSV): first_rank_le_{10,5,1} = null (no attainment) [001_layers_baseline/run-latest/output-Yi-34B.json:845–877].
- Top‑1 agreement/cosine: prism cos_to_final remains near 0 at mid‑late depths where baseline aligns (e.g., at L45 baseline cos≈0.496 vs prism≈−0.003; see CSVs).
- Copy flags: no strict/soft copy flags flip under Prism at early layers.
- Verdict: Regressive (earlier KL at some depths but fails to reach semantic rank milestones and degrades mid‑late KL).

Tuned‑Lens (side‑by‑side highlights)
- ΔKL medians at depth percentiles (baseline − tuned): Δp25=5.88 bits, Δp50=6.10 bits, Δp75=8.39 bits (tuned lower KL) [001_layers_baseline/run-latest/output-Yi-34B.json:2109–2124].
- Rank earliness: baseline first_rank_le_1=44 vs tuned=46 (later), le_5: 44 vs 44, le_10: 43 vs 44 [001_layers_baseline/run-latest/output-Yi-34B.json:2094–2106].
- Entropy drift (mid‑depth): entropy − teacher_entropy_bits ≈ 12.61 bits at L30; shrinks to ≈0.04 bits at L60 (baseline) [001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:30,60].
- Surface→meaning (norm lens): L_surface_to_meaning_norm=51 with answer_mass_at_L≈0.060 and echo_mass_at_L≈0.006 [001_layers_baseline/run-latest/output-Yi-34B.json:1035–1043]. Geometry: L_geom_norm=46 with cos_to_answer≈0.125 and cos_to_prompt_max≈0.096 [001_layers_baseline/run-latest/output-Yi-34B.json:1038–1040].

**Qualitative Patterns & Anomalies**
- Negative control: “Berlin is the capital of” yields top‑5 next tokens: “ Germany, 0.8398; the, 0.0537; which, 0.0288; what, 0.0120; Europe, 0.0060” — no ‘Berlin’ (semantic leakage not observed) [001_layers_baseline/run-latest/output-Yi-34B.json:12–31].
- Important‑word trajectory (records CSV; IMPORTANT_WORDS = ["Germany", "Berlin", "capital", "Answer", "word", "simply"] [001_layers_baseline/run.py:350]). Around the last context token (“ simply”, pos=16), ‘Berlin’ becomes top‑1 at L44 (“… ‘ Berlin’, 0.00846”) [001_layers_baseline/run-latest/output-Yi-34B-records.csv:788]; at adjacent positions it is already salient: L44 pos=14 (‘ is’): “ Berlin, 0.01049” [001_layers_baseline/run-latest/output-Yi-34B-records.csv:786]; L44 pos=15 (‘ called’): “ Berlin, 0.00826” [001_layers_baseline/run-latest/output-Yi-34B-records.csv:787]. This suggests late consolidation around the answer token across a small neighborhood.
- Collapse‑layer stability under ablation: removing “simply” does not shift L_semantic (44→44; ΔL_sem=0) and no copy layer emerges (ΔL_copy=null) [001_layers_baseline/run-latest/output-Yi-34B.json:1629–1636]. This points to minimal stylistic anchoring for this probe.
- Rest‑mass sanity: top‑k rest_mass remains high early and spikes after L_semantic (e.g., ≈0.978 at L45), reflecting top‑k coverage limits rather than lens fidelity; we treat KL/rank for calibration [001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:45].
- Rotation vs amplification: KL to final declines with depth and is ≈0 at L60 while p_answer and answer_rank improve sharply at L44. Cosine aligns early (≥0.2 at L1) but major calibration occurs much later (KL still ≳13 bits mid‑stack), a classic “early direction, late calibration” signature consistent with norm‑lens views.
- Final‑layer head calibration: last_layer_consistency shows tiny KL (0.00028 bits), top‑1 agreement, and temp_est=1.0; no family‑level softcaps/scales detected [001_layers_baseline/run-latest/output-Yi-34B.json:1096–1115].
- Lens sanity: raw‑lens check flags “lens_artifact_risk: high” and “first_norm_only_semantic_layer: 46”; the windowed raw‑vs‑norm CSV shows huge KL gaps near collapse (e.g., KL_norm_vs_raw≈80–90 bits at L46–60) [001_layers_baseline/run-latest/output-Yi-34B.json:1558–1618,1061–1094]. Accordingly, we treat any pre‑final “early semantics” cautiously and rely on rank milestones.
- Temperature robustness: At T=0.1, Berlin rank=1 with p≈0.9999996 and entropy≈7e‑6 bits; at T=2.0, Berlin remains top‑1 with p≈0.0488 and entropy≈12.49 bits [001_layers_baseline/run-latest/output-Yi-34B.json:736–801].

Checklist
- RMS lens? ✓ (RMSNorm; norm lens on) [001_layers_baseline/run-latest/output-Yi-34B.json:807–816].
- LayerNorm bias removed? n.a. (RMS model; not needed) [001_layers_baseline/run-latest/output-Yi-34B.json:812].
- Entropy rise at unembed? ✓ (final entropy≈2.98 bits vs mid‑stack ≈15.6 bits; teacher≈2.94 bits) [001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:15,30,60].
- FP32 un‑embed promoted? ✓ (use_fp32_unembed=true; unembed_dtype=torch.float32) [001_layers_baseline/run-latest/output-Yi-34B.json:808–809].
- Punctuation / markup anchoring? ✓ (early layers dominated by punctuation/fillers; e.g., L1 top‑1 '.')
- Copy‑reflex? ✗ (no strict or soft copy flags in L0–L3; none at all).
- Grammatical filler anchoring? ✓ (top‑1 in L0–5 includes punctuation/fillers like '.', quotes).

**Limitations & Data Quirks**
- Raw‑vs‑norm lens differences are substantial around collapse (max KL_norm_vs_raw≈90.47 bits in the window; sample mode), so early “semantics” can be lens‑induced; we prefer rank milestones and within‑model trends [001_layers_baseline/run-latest/output-Yi-34B.json:1061–1094,1558–1618].
- Rest_mass after L_semantic remains high (>0.3), reflecting top‑k coverage, not fidelity; avoid interpreting it as a lens‑quality metric.
- KL is lens‑sensitive; despite the excellent final‑head alignment here, cross‑model probability comparisons should be avoided. Use the provided milestones for within‑model claims.

**Model Fingerprint**
- Yi‑34B: collapse at L 44; final entropy ≈2.98 bits; ‘Berlin’ appears rank 1 mid‑late, Prism regressive on ranks, tuned lens lowers KL but does not improve rank earliness.

---
Produced by OpenAI GPT-5 

