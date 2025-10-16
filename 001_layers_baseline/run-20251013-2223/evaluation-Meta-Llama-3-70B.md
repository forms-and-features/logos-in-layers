# Evaluation Report: meta-llama/Meta-Llama-3-70B

*Run executed on: 2025-10-13 22:23:35*

## EVAL

**Overview**
- Meta-Llama-3-70B evaluated on 2025-10-13; probe analyzes layer-wise copy vs semantic emergence using norm lens with rank/KL/cosine/entropy trajectories and lens diagnostics. Model: `meta-llama/Meta-Llama-3-70B` (diagnostics.model) 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:807.
- Focus: semantic onset, copy suppression, raw-vs-norm artefact risk, gate stability, last-layer calibration, and entropy drift; confirmed-semantics window preferred per guidance 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9630.

**Method sanity-check**
- Prompt & indexing: context prompt ends with “called simply” and matches positive prompt rows. Quote: "Give the city name only, plain text. The capital of Germany is called simply" 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:811. Positive row example exists: L40, `prompt_id=pos`, `prompt_variant=orig` 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:42.
- Normalizer provenance: `arch=pre_norm`, `strategy=next_ln1`; per-layer uses `blocks[0].ln1` at L0 and `ln_final` at final. Quotes: strategy and L0 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7426,7440; final layer source 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8210.
- Per-layer normalizer effect: early resid_norm_ratio/delta_resid_cos spikes present and flagged. Example spikes: L1 resid_norm_ratio 11.54, delta_resid_cos 0.689 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7447–7455; flag: `normalization_spike: true` 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:834.
- Unembed bias: absent, bias-free cosines. Quote: `"present": false, "l2_norm": 0.0` 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:826–836.
- Environment & determinism: deterministic with fixed seed. Quote: `deterministic_algorithms: true`, `seed: 316` 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9454–9464.
- Numeric health: clean. Quote: `any_nan: false, any_inf: false, layers_flagged: []` 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8649–8656.
- Copy mask: present and plausible; includes punctuation/control tokens; size reported. Quotes: `ignored_token_str_sample: ["!","\"","#","$",...]`, `size: 6022` 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7009–7020,7041.
- Gold alignment: ok, single-piece `ĠBerlin`. Quote: `"ok": true, "pieces": ["ĠBerlin"], "variant": "with_space"` 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8685–8710.
- Repeatability (1.39): skipped due to deterministic env. Quote: `"status": "skipped", "reason": "deterministic_env"` 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8656–8660.
- Norm trajectory: spike-shaped with high fit. Quote: `shape: "spike", r2: 0.9367, n_spikes: 15` 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9062–9068.
- Measurement guidance: prefer ranks; suppress absolute probs; use norm lens and confirmed semantics. Quotes: `"prefer_ranks": true, "suppress_abs_probs": true` and `preferred_lens_for_reporting: "norm", use_confirmed_semantics: true` 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9630–9648.
- Semantic margin: uniform-margin gate fails at L_semantic_norm. Quote: `delta_abs: 0.002, p_uniform: 7.8e-06, margin_ok_at_L_semantic_norm: false` 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9514–9522.
- Gate-stability: calibration-sensitive; both gates pass fraction = 0.0 at L=40 and min=0.0. Quote: `both_gates_pass_frac: 0.0`, `min_both_gates_pass_frac: 0.0` 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7189–7198.
- Position-window: not executed; no grid or rank1 frac. Quote: `grid: [], rank1_frac: null` 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9055–9062.
- Micro-suite: present; n=5; no missing facts; confirmed medians. Quote: `L_semantic_confirmed_median: 40, n_missing: 0` 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9035–9048.

**Quantitative findings (layer-by-layer)**

| Layer | Entropy (bits) | Top-1 token | Answer rank | KL→final (bits, norm-temp) | Cos→final | Cos→answer | Notes |
|---|---:|---|---:|---:|---:|---:|---|
| 0 | 16.968 | winding | 115765 | 10.5015 | 0.00429 | -0.01275 | pre-block; copy flags false [row 2 in CSV] 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:2 |
| 10 | 16.952 | inds | 34376 | 10.5080 | -0.01758 | 0.00496 | early noise; no copy reflex [row 12] 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:12 |
| 20 | 16.946 | nut | 6991 | 10.6859 | -0.02581 | 0.01320 | pre-semantics [row 22] 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:22 |
| 38 | 16.934 | simply | 3 | 10.6136 | 0.08367 | 0.05319 | approach window [row 40] 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:40 |
| **40** | 16.937 | Berlin | 1 | 10.6251 | 0.09681 | 0.05370 | semantic onset (confirmed; weak margins) [row 42] 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:42 |
| 43 | 16.941 | Berlin | 1 | 10.4509 | 0.06187 | 0.05330 | run-of-two stability target (L_semantic_run2=43) [row 45] 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:45 |
| 60 | 16.923 | Berlin | 1 | 4.3499 | 0.07423 | 0.10079 | cos crossover observed [row 62] 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:62 |
| 80 | 2.589 | Berlin | 1 | 0.00073 | 0.99999 | 0.09851 | final calibration agrees; KL≈0 [row 82] 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:82 |

- Semantic layer: bolded below per gates and confirmation: L_semantic_norm=40 and L_semantic_confirmed=40 (source=raw) 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8947–8956,9660–9667. Uniform and top-2 gates fail at L=40; treat as weak rank‑1 (see 4.4).
- Control summary: `first_control_margin_pos=0`, `max_control_margin=0.5168`, `first_control_strong_pos=80` 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9501–9516.
- Micro-suite: median L_semantic_confirmed = 40; IQR for L_semantic_norm = [49, 60]; n_missing=0. Example fact citation: Germany→Berlin at L=40 [row 40] 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9035–9048.
- Entropy drift: large positive gaps vs teacher at percentiles: p25=14.33, p50=14.34, p75=14.35 bits 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9669–9699.
- Normalizer effect snapshots: resid_norm_ratio/delta_resid_cos at L=40: 1.2305 and 0.9774 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7794–7803; answer_logit_gap at L=40 ≈0.031 (weak), strengthens by L=60 ≈0.743 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:42,62.

**Qualitative findings**

4.1. Copy vs semantics (Δ-gap)
- Copy-reflex ✗. No early copy collapse or soft-copy k1 within layers 0–3; strict and soft detectors return null across τ∈{0.70,0.95} and k∈{1,2,3}. Quotes: all null in `copy_thresholds.L_copy_strict` and `L_copy_soft` 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7111–7133,7069–7083. Stability tag: `stability: "none"` 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7131–7140.
- Δ̂ undefined (no copy onset). Depth fractions report semantic_frac=0.5 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7147–7165. Earliest strict copy at τ=0.70 and τ=0.95: null; `norm_only_flags[τ]`: null 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7119–7133.

4.2. Lens sanity: Raw‑vs‑Norm
- Artefact scores: lens_artifact_score=0.3323; lens_artifact_score_v2=0.3439; tier=medium 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7418–7426,9668–9676. Raw-lens check summary risk: low 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9430–9436.
- Symmetric metrics: js_divergence_p50=0.00245; l1_prob_diff_p50=0.0918; first_js_le_0.1=0; first_l1_le_0.5=0 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9668–9676.
- Top‑K overlap: jaccard_raw_norm_p50=0.515; first_jaccard_raw_norm_ge_0.5=11 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7252–7260.
- Prevalence: pct_layers_kl_ge_1.0=0.0123; n_norm_only_semantics_layers=2; earliest_norm_only_semantic=79; window max KL≈1.244 bits 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7189–7239.
- Lens-consistency at semantic targets: L=38 jaccard@10=0.667, spearman_top50≈0.575; L=40 jaccard@10=0.818, spearman_top50≈0.658 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8660–8670.
- Caution: medium artefact tier and late norm‑only semantics (L79–L80) are far from the candidate semantic layer (L40), reducing risk of lens‑induced early semantics. Early normalization spikes (below) motivate rank/KL preference over absolute probabilities.

4.3. Tuned‑Lens analysis
- Status: missing (diagnostic‑only path present, no tuned lens data). Quote: `"tuned_lens": { "status": "missing" }` 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8940–8946,9508–9513. Prefer norm lens for reporting.

4.4. KL, ranks, cosine, entropy milestones
- KL: first_kl_below_1.0 = 80; final KL≈0.00073 bits and top‑1 agree → good final‑head calibration 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7043–7052,8713–8723.
- Ranks (preferred lens): first_rank_le_10 = 38; first_rank_le_5 = 38; first_rank_le_1 = 40 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7046–7055.
- Cosine: ge_0.2/ge_0.4/ge_0.6 all at L=80 under norm 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7140–7147.
- Entropy: monotonic decrease into final; large entropy gaps vs teacher at p25/p50/p75 ≈14.33/14.34/14.35 bits 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9669–9699. Entropy trends align with KL/rank improvements (late calibration).
- Margin gates: at L_semantic_norm=40, `margin_ok_at_L_semantic_norm=false` and no top‑2 pass; treat as weak rank‑1. Quote: summary.semantic_margin and semantic_gate 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9514–9522,9522–9531.
- Stability: L_semantic_run2=43 (advisory stable window), but `gate_stability_small_scale` min both‑gates pass frac=0.0 → calibration‑sensitive; avoid absolute‑p claims 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7165–7198.
- Position‑window: not assessed (no grid); generalization beyond the measured next‑token position should be cautious 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9055–9062.

4.5. Prism (shared‑decoder diagnostic)
- Presence/compatibility: present and compatible; sampled at layers [embed,19,39,59] 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:844–858.
- KL deltas: prism shows higher KL by ≈1 bit at p50 (baseline p50=10.42 vs prism p50=11.42; delta≈−1.00) 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:884–904.
- Rank milestones: prism rank milestones are null (no improvement) 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:864–878.
- Verdict: Regressive — KL increases and no earlier ranks.

4.6. Ablation & stress tests
- No‑filler ablation: `L_sem_orig=40` → `L_sem_nf=42` (ΔL_sem=+2, <10% of 80 layers) 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9467–9476.
- Negative/control prompts: control summary present; `first_control_strong_pos=80`, max control margin reported 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9501–9516. Test prompt “Berlin is the capital of” has country continuation in top‑k (qualitatively aligns with control intent) 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7–44.
- Important‑word trajectory: “Berlin” appears in top‑k before semantic onset (e.g., L31 and L33 include “ Berlin” within top‑k set) 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-records.csv:680,719.

4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓ 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:809–817
- LayerNorm bias removed ✓ 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:820–833
- FP32 unembed promoted ✓ 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:816–823
- Punctuation / markup anchoring noted ✓ 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7009–7020
- Copy‑reflex ✗ 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7069–7083,7111–7133
- Preferred lens honored ✓ 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9630–9648
- Confirmed semantics reported ✓ 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8947–8956,9660–9667
- Dual‑lens artefact metrics (incl. v2, JS/Jaccard/L1) cited ✓ 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9668–9676,7252–7260
- Tuned‑lens audit done (rotation/temp/positional/head) n.a. (missing) 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8940–8946
- normalization_provenance present (ln_source @ L0/final) ✓ 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7440,8210
- per‑layer normalizer effect present (resid_norm_ratio, delta_resid_cos) ✓ 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7426–7480
- deterministic_algorithms true ✓ 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9454–9464
- numeric_health clean ✓ 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8649–8656
- copy_mask plausible ✓ 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7009–7041
- milestones.csv or evaluation_pack.citations used for quotes ✓ 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9707
- gate_stability_small_scale reported ✓ 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7165–7198
- position_window stability reported ✓ (no data) 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9055–9062

---
Produced by OpenAI GPT-5
