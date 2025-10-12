# Evaluation Report: 01-ai/Yi-34B

**Overview**
- Model: 01-ai/Yi-34B; run window timestamp present as `timestamp-20251011-2150` and final prediction recorded in the latest pack (see `evaluation_pack.model = "Yi-34B"`). The probe measures copy vs. semantic onset via rank milestones and KL thresholds, tracks cosine/entropy trajectories, and audits lens health (raw vs. norm), plus tuned-lens and Prism diagnostics.
- Baseline prompt targets the next token of “Give the city name only, plain text. The capital of Germany is called simply” and evaluates layer-by-layer predictions and calibration.

**Method Sanity‑Check**
- Prompt & indexing: context ends exactly with “called simply” with no trailing space (`001_layers_baseline/run-latest/output-Yi-34B.json:4`). Positive/`orig` rows are present (gold alignment lists `prompt_id="pos"`, `prompt_variant="orig"`) (`001_layers_baseline/run-latest/output-Yi-34B.json:3819`).
- Normalizer provenance: `arch = "pre_norm"`, `strategy = "next_ln1"`; early layer `ln_source = blocks[0].ln1`, final uses `ln_final` (`001_layers_baseline/run-latest/output-Yi-34B.json:2824`, `001_layers_baseline/run-latest/output-Yi-34B.json:3362`).
- Per‑layer normalizer effect: normalization spike flagged; L0 shows `resid_norm_ratio ≈ 6.14`, `delta_resid_cos ≈ 0.57` while L44 is far lower (`resid_norm_ratio ≈ 0.20`) (`001_layers_baseline/run-latest/output-Yi-34B.json:2833`, `001_layers_baseline/run-latest/output-Yi-34B.json:3234`).
- Unembed bias: `present=false`, `l2_norm=0.0`; cosines are bias‑free (`001_layers_baseline/run-latest/output-Yi-34B.json:828`).
- Environment & determinism: `device="cpu"`, `torch=2.8.0+cu128`, `deterministic_algorithms=true`, `seed=316` (`001_layers_baseline/run-latest/output-Yi-34B.json:5839`).
- Numeric health: `any_nan=false`, `any_inf=false`, `layers_flagged=[]` (`001_layers_baseline/run-latest/output-Yi-34B.json:3747`).
- Copy mask: `ignored_token_ids` list present (e.g., `97, 98, 99, 100, …`) consistent with punctuation/ASCII spans (`001_layers_baseline/run-latest/output-Yi-34B.json:968`).
- Gold alignment: `ok: true`, `variant: "with_space"`, `pieces: ["▁Berlin"]` (`001_layers_baseline/run-latest/output-Yi-34B.json:3758`).
- Repeatability (1.39): skipped in deterministic env (`status: "skipped"`, `reason: "deterministic_env"`) (`001_layers_baseline/run-latest/output-Yi-34B.json:3754`).
- Norm trajectory: `shape="spike"`, `slope≈0.074`, `r2≈0.926`, `n_spikes=4` (`001_layers_baseline/run-latest/output-Yi-34B.json:5457`).
- Measurement guidance: `prefer_ranks=true`, `suppress_abs_probs=true`; preferred lens=`tuned`, `use_confirmed_semantics=true` (`001_layers_baseline/run-latest/output-Yi-34B.json:7586`).
- Semantic margin: `delta_abs=0.002`, `p_uniform=1.5625e-05`, `margin_ok_at_L_semantic_norm=true`, `L_semantic_confirmed_margin_ok_norm=44` (`001_layers_baseline/run-latest/output-Yi-34B.json:2629`).
- Micro‑suite: aggregates exist with `n=5`, `n_missing=0`, medians show `L_semantic_confirmed_median=44` (`001_layers_baseline/run-latest/output-Yi-34B.json:7494`).

**Quantitative Findings (Layer‑by‑Layer)**
- Short trajectory (positive, `orig`):
  - L 0 — entropy 15.96 bits; top‑1 token “Denote” (`001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:2`).
  - L 10 — entropy 15.80 bits; top‑1 token “~\\\\” (`001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:12`).
  - L 20 — entropy 15.60 bits; top‑1 token “ncase” (`001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:22`).
  - L 30 — entropy 15.55 bits; top‑1 token “ODM” (`001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:32`).
  - L 44 — entropy 15.33 bits; top‑1 token “Berlin”; answer_rank=1 — this is the confirmed semantic layer. (bolded below) (`001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:46`).
  - L 60 — entropy 2.98 bits; top‑1 token “Berlin”; answer_rank=1 (`001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:63`).
- Semantic layer: L **44** (confirmed; preferred lens=tuned; `evaluation_pack.milestones.L_semantic_confirmed.layer=44`) (`001_layers_baseline/run-latest/output-Yi-34B.json:7609`). Margin gate OK at L_semantic_norm (`L_semantic_confirmed_margin_ok_norm=44`).
- Control margin: first control margin at prompt position 1; max control margin reported (`control_summary.first_control_margin_pos=1`, `max_control_margin≈0.584`) (`001_layers_baseline/run-latest/output-Yi-34B.json:5898`).
- Micro‑suite: median confirmed semantics at L=44 (n=5, no missing). Example fact citation: “Germany→Berlin” at row 44 (`evaluation_pack.micro_suite.citations.fact_rows`) (`001_layers_baseline/run-latest/output-Yi-34B.json:7695`).
- Entropy drift: median entropy gap (bits) p25/p50/p75 ≈ 12.29 / 12.59 / 12.78 (relative to teacher entropy) (`001_layers_baseline/run-latest/output-Yi-34B.json:7708`).

**Qualitative Findings**

4.1. Copy vs semantics (Δ‑gap)
- Copy‑reflex: ✗. No strict or soft copy milestone detected (all `L_copy_* = null`) and early layers show `copy_collapse=False`, `copy_soft_k{1,2,3}@0.5=False` (e.g., L0–L3) (`001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:2`).
- With `L_copy_strict` null, Δ̂ is not defined (and `evaluation_pack.milestones.depth_fractions.delta_hat=null`) (`001_layers_baseline/run-latest/output-Yi-34B.json:7617`).
- Threshold stability: `copy_thresholds.stability="none"`; earliest strict copy at τ=0.70 and τ=0.95: null; `norm_only_flags[τ] = null` (`001_layers_baseline/run-latest/output-Yi-34B.json:2569`).

4.2. Lens sanity: Raw‑vs‑Norm
- Artifact risk: `lens_artifact_score_v2=0.943` (tier=high). JS/L1 medians: `js_divergence_p50≈0.369`, `l1_prob_diff_p50≈1.089`; `first_js_le_0.1 = 0`, `first_l1_le_0.5 = 0` (`001_layers_baseline/run-latest/output-Yi-34B.json:2820`, `001_layers_baseline/run-latest/output-Yi-34B.json:2682`).
- Top‑K overlap: `jaccard_raw_norm_p50≈0.111`, `first_jaccard_raw_norm_ge_0.5=1` (`001_layers_baseline/run-latest/output-Yi-34B.json:2685`).
- Prevalence: `pct_layers_kl_ge_1.0≈0.656`, `n_norm_only_semantics_layers=14`, `earliest_norm_only_semantic=44` (`001_layers_baseline/run-latest/output-Yi-34B.json:2665`). Caution: early semantics near L44 may be lens‑induced; rely on rank milestones and confirmed semantics.

4.3. Tuned‑Lens analysis
- Preference: `tuned_is_calibration_only=false`; preferred lens for reporting is “tuned” (`measurement_guidance.preferred_lens_for_reporting="tuned"`) (`001_layers_baseline/run-latest/output-Yi-34B.json:7586`, `001_layers_baseline/run-latest/output-Yi-34B.json:7488`).
- Attribution (ΔKL, bits): rotation dominates with positive interaction — `ΔKL_rot_p25/p50/p75 ≈ 3.16/3.50/3.69`, `ΔKL_temp_p25/p50/p75 ≈ −1.03/−0.66/0.61`, `ΔKL_interaction_p50 ≈ 3.33` (`001_layers_baseline/run-latest/output-Yi-34B.json:7660`).
- Rank earliness: tuned vs. baseline `first_rank_le_{10,5,1}` deltas are {+1, 0, +2} (later or unchanged) (`001_layers_baseline/run-latest/output-Yi-34B.json:7420`).
- Positional generalization: `pos_in_dist_le_0.92 ≈ 5.59`, `pos_ood_ge_0.96 ≈ 6.45`, gap ≈ 0.85 bits (`001_layers_baseline/run-latest/output-Yi-34B.json:7675`).
- Head mismatch: `tau_star_modelcal=1.0`; last‑layer tuned KL ≈ 0.0013 bits, unchanged after τ* (`001_layers_baseline/run-latest/output-Yi-34B.json:7688`).
- Final‑head agreement: `top1_agree=true`, `kl_after_temp_bits≈0.00028`, `warn_high_last_layer_kl=false` (`001_layers_baseline/run-latest/output-Yi-34B.json:3770`).

4.4. KL, ranks, cosine, entropy milestones
- KL: `first_kl_below_1.0 = 60` (baseline and tuned). Final KL ≈ 0 (well‑calibrated head) (`001_layers_baseline/run-latest/output-Yi-34B.json:7488`, `001_layers_baseline/run-latest/output-Yi-34B.json:3770`).
- Ranks: preferred lens (tuned) `first_rank_le_{10,5,1} = {44,44,46}` (baseline: {43,44,44}) (`001_layers_baseline/run-latest/output-Yi-34B.json:7406`). Note: margin gate at `L_semantic_norm` is satisfied.
- Cosine: norm‑lens milestones `ge_0.2 = 1`, `ge_0.4 = 44`, `ge_0.6 = 51` (`001_layers_baseline/run-latest/output-Yi-34B.json:2607`).
- Entropy: consistent large positive gaps vs. teacher entropy (p25/p50/p75 ≈ 12.29/12.59/12.78 bits), with monotonic drop late as ranks settle (`001_layers_baseline/run-latest/output-Yi-34B.json:7708`).

4.5. Prism (shared‑decoder diagnostic)
- Presence/compatibility: Prism present and compatible; layers checked: embed, 14, 29, 44 (`001_layers_baseline/run-latest/output-Yi-34B.json:837`).
- KL deltas: median KL drops by ≈1.36 bits (`p50`), but rank milestones unchanged/NA under Prism (`001_layers_baseline/run-latest/output-Yi-34B.json:920`).
- Verdict: Neutral — modest KL improvement without earlier rank milestones.

4.6. Ablation & stress tests
- Ablation summary: `L_sem_orig=44`, `L_sem_nf=44`, `ΔL_sem=0` — no stylistic sensitivity across “orig” vs “no_filler” (`001_layers_baseline/run-latest/output-Yi-34B.json:5862`).
- Negative/control prompt: “Berlin is the capital of …” returns country continuation with low entropy; qualitative top‑1 aligns with control (`001_layers_baseline/run-latest/output-Yi-34B.json:12`).
- Important‑word trajectory: around L44–L49, “Berlin” rises to top‑1 across the final token positions (e.g., `pos=16` at L44/L46/L47 shows “Berlin” as top‑1) (`001_layers_baseline/run-latest/output-Yi-34B-records.csv:788`, `001_layers_baseline/run-latest/output-Yi-34B-records.csv:824`, `001_layers_baseline/run-latest/output-Yi-34B-records.csv:842`).

4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓
- LayerNorm bias removed ✓
- FP32 unembed promoted ✓
- Punctuation / markup anchoring noted ✓
- Copy‑reflex ✗
- Preferred lens honored ✓
- Confirmed semantics reported ✓
- Dual‑lens artefact metrics (incl. v2, JS/Jaccard/L1) cited ✓
- Tuned‑lens audit (rotation/temp/positional/head) ✓
- normalization_provenance present (ln_source @ L0/final) ✓
- per‑layer normalizer effect present (resid_norm_ratio, delta_resid_cos) ✓
- deterministic_algorithms true ✓
- numeric_health clean ✓
- copy_mask plausible ✓
- milestones.csv or evaluation_pack.citations used for quotes ✓

---
Produced by OpenAI GPT-5 

*Run executed on: 2025-10-11 21:50:12*
