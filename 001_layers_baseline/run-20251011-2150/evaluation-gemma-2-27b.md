# Evaluation Report: google/gemma-2-27b

## EVAL

**Overview**
- Model: `google/gemma-2-27b` with 46 layers; run artifacts present for norm, raw, tuned, and prism lenses (001_layers_baseline/run-latest/output-gemma-2-27b.json:806,8717).
- Probe measures copy vs. semantic emergence, KL-to-final and rank milestones, cosine and entropy trajectories, and dual‑lens artefacts; tuned/prism audits included (001_layers_baseline/run-latest/output-gemma-2-27b.json:5653,5750,5751,5752,6664,840).

**Method sanity-check**
- Prompt & indexing: Positive context ends with “called simply” (no trailing space): "Give the city name only, plain text. The capital of Germany is called simply" (001_layers_baseline/run-latest/output-gemma-2-27b.json:4). Positive rows exist: `Germany→Berlin,0,pos,orig,0,…` [row 2 in CSV] (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:1).
- Normalizer provenance: `arch: "pre_norm"`, `strategy: "next_ln1"` (001_layers_baseline/run-latest/output-gemma-2-27b.json:5927). `per_layer[0].ln_source = blocks[0].ln1` and final `ln_source = ln_final` (001_layers_baseline/run-latest/output-gemma-2-27b.json:5933,6347).
- Per‑layer normalizer effect: `resid_norm_ratio`/`delta_resid_cos` present per layer; quality flag shows spike: `"normalization_spike": true` (001_layers_baseline/run-latest/output-gemma-2-27b.json:851). Example L0: `resid_norm_ratio: 0.786…`, `delta_resid_cos: 0.572…` (001_layers_baseline/run-latest/output-gemma-2-27b.json:5936–5941). No candidate‑layer overlap flagged in numeric health (see below).
- Unembed bias: `"present": false, "l2_norm": 0.0` (001_layers_baseline/run-latest/output-gemma-2-27b.json:831,832).
- Environment & determinism: `device: "cpu"`, `dtype_compute: "torch.float32"`, `deterministic_algorithms: true`, `seed: 316` (001_layers_baseline/run-latest/output-gemma-2-27b.json:8733–8741). Deterministic; repeatability bench skipped accordingly.
- Numeric health: `any_nan: false`, `any_inf: false`, `layers_flagged: []` (001_layers_baseline/run-latest/output-gemma-2-27b.json:6640–6647).
- Copy mask: `ignored_token_ids` listed; sample shows punctuation/formatting IDs (001_layers_baseline/run-latest/output-gemma-2-27b.json:953,980). No explicit `size` field present.
- Gold alignment: `ok: true`, variant `with_space`, IDs/pieces align to "▁Berlin" (001_layers_baseline/run-latest/output-gemma-2-27b.json:6669–6671,6679–6681).
- Repeatability: skipped due to deterministic env (`status: "skipped"`, `reason: "deterministic_env"`) (001_layers_baseline/run-latest/output-gemma-2-27b.json:6649,10518).
- Norm trajectory: `shape: "spike"`, `slope: 0.0878`, `r2: 0.8458`, `n_spikes: 16` (001_layers_baseline/run-latest/output-gemma-2-27b.json:8350–8358).
- Measurement guidance: `prefer_ranks: true`, `suppress_abs_probs: true`, `preferred_lens_for_reporting: "norm"`, `use_confirmed_semantics: true` (001_layers_baseline/run-latest/output-gemma-2-27b.json:10465–10506). Reasons include `warn_high_last_layer_kl`, `norm_only_semantics_window`, `high_lens_artifact_risk`.
- Semantic margin: `delta_abs: 0.002`, `p_uniform: 3.90625e-06`, `margin_ok_at_L_semantic_norm: true` (001_layers_baseline/run-latest/output-gemma-2-27b.json:10366–10373).
- Micro‑suite: aggregates present with `n: 5`, `n_missing: 0`; medians show `L_semantic_norm_median = 46`, `L_semantic_confirmed_median = 46` (001_layers_baseline/run-latest/output-gemma-2-27b.json:10373–10410).

**Quantitative findings (layer‑by‑layer)**
- Preferred lens: norm; confirmed semantics used per guidance (001_layers_baseline/run-latest/output-gemma-2-27b.json:10465–10506).
- Short table (pos/orig only; entropy quoted, top‑1 token text):
  - L 0 — entropy 0.0005 bits, top‑1 ‘ simply’ [row 2 in CSV] (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:1).
  - L 3 — entropy 0.8857 bits, top‑1 ‘ simply’ [row 5 in CSV] (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:1).
  - L 23 — entropy 7.0104 bits, top‑1 ‘ dieſem’ [row 25 in CSV] (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:25).
  - L 46 — entropy 0.1180 bits, top‑1 ‘ Berlin’ [row 48 in CSV] (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48).
- Semantic layer: Bold confirmed semantics at L 46 (source tuned; confirmed semantics enabled) (001_layers_baseline/run-latest/output-gemma-2-27b.json:10506,10576–10583).
- Control margin: `first_control_margin_pos: 0`, `max_control_margin: 0.9910899400710897` (001_layers_baseline/run-latest/output-gemma-2-27b.json:8789–8800).
- Micro‑suite summary: median `L_semantic_confirmed = 46`, median Δ̂ = 1.0; e.g., "Germany→Berlin" cites `row_index: 46` (001_layers_baseline/run-latest/output-gemma-2-27b.json:10380–10388,10399–10407).
- Entropy drift: gaps vs teacher at p25/p50/p75 are 4.117/4.680/5.388 bits (001_layers_baseline/run-latest/output-gemma-2-27b.json:5706–5710).
- Normalizer effect snapshots: L 0 `resid_norm_ratio: 0.786`, `delta_resid_cos: 0.572` (001_layers_baseline/run-latest/output-gemma-2-27b.json:5936–5941); L 46 `resid_norm_ratio: 0.0527`, `delta_resid_cos: 0.6617` (001_layers_baseline/run-latest/output-gemma-2-27b.json:6347–6354).

**Qualitative findings**

4.1. Copy vs semantics (Δ‑gap)
Early layers exhibit a copy‑reflex: layer 0 is marked `copy_collapse=True` and `copy_soft_k1@0.5=True` [row 2 in CSV], with the top‑1 ‘ simply’ echoing the prompt suffix (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:1). Strict copy is recorded at L=0 across τ∈{0.70,0.95} (`L_copy_strict: {0.7: 0, 0.95: 0}`) with `stability: "mixed"` and no norm‑only flags (001_layers_baseline/run-latest/output-gemma-2-27b.json:5710–5727). Confirmed semantics occurs at the final layer, L 46 (001_layers_baseline/run-latest/output-gemma-2-27b.json:10576–10583). The depth fraction Δ̂ is 1.0 (copy at 0, semantics at 46 over 46 layers) (001_layers_baseline/run-latest/output-gemma-2-27b.json:10498–10506). This pattern is consistent with a strong surface echo followed by late semantic resolution.

4.2. Lens sanity: Raw‑vs‑Norm
Dual‑lens audits show high artefact risk: `lens_artifact_score_v2: 1.0`, `tier: "high"` (001_layers_baseline/run-latest/output-gemma-2-27b.json:5923–5931). Symmetric metrics show large divergence: `js_divergence_p50: 0.8653`, `l1_prob_diff_p50: 1.8929`, with `first_js_le_0.1: 0`, `first_l1_le_0.5: 0` (001_layers_baseline/run-latest/output-gemma-2-27b.json:5813–5816,10507–10510). Top‑K overlap median is moderate (`jaccard_raw_norm_p50: 0.5625`), with `first_jaccard_raw_norm_ge_0.5: 1` (001_layers_baseline/run-latest/output-gemma-2-27b.json:5828–5831,10510–10518). Prevalence is high: `pct_layers_kl_ge_1.0: 0.9787`, `n_norm_only_semantics_layers: 1`, `earliest_norm_only_semantic: 46` (001_layers_baseline/run-latest/output-gemma-2-27b.json:5796–5813). Caution: early semantics near the candidate layer could be lens‑induced; we therefore follow guidance to prefer ranks and confirmed semantics.

4.3. Tuned‑Lens analysis
The tuned lens is marked calibration‑only: `tuned_is_calibration_only: true` with `preferred_semantics_lens_hint: "tuned_for_calibration_only"` (001_layers_baseline/run-latest/output-gemma-2-27b.json:10528–10535). Attribution shows small rotation effects and substantial temperature effects on KL: `delta_kl_rot_p50: -0.029`, `delta_kl_temp_p50: 0.528`, `delta_kl_interaction_p50: 0.028` (001_layers_baseline/run-latest/output-gemma-2-27b.json:10528–10535). Positional generalization is modest (`pos_in_dist_le_0.92: 0.435`, `pos_ood_ge_0.96: 0.519`, `pos_ood_gap: 0.0836`) (001_layers_baseline/run-latest/output-gemma-2-27b.json:10535–10546). Head mismatch calibration improves final KL under temperature scaling (tuned final KL and after `tau_star` reported) (001_layers_baseline/run-latest/output-gemma-2-27b.json:10546–10553). Last‑layer consistency warns of calibration issues: `warn_high_last_layer_kl: true` and `kl_to_final_bits: 1.135…` (001_layers_baseline/run-latest/output-gemma-2-27b.json:6663–6672). Baseline semantics are therefore reported under the norm lens, with tuned used for calibration confirmation only.

4.4. KL, ranks, cosine, entropy milestones
KL milestones under the preferred lens do not cross {1.0, 0.5} bits early (`first_kl_below_1.0: null`, `first_kl_below_0.5: null`) (001_layers_baseline/run-latest/output-gemma-2-27b.json:5651–5652). Ranks reach top‑1 only at L 46 (`first_rank_le_1: 46`) (001_layers_baseline/run-latest/output-gemma-2-27b.json:5653). Cosine milestones (norm lens) reach `ge_0.2` at L 1, and `ge_0.4`/`ge_0.6` at L 46 (001_layers_baseline/run-latest/output-gemma-2-27b.json:5750–5752). Entropy declines late; gap vs teacher is large across depth percentiles (001_layers_baseline/run-latest/output-gemma-2-27b.json:5706–5710). The uniform‑margin gate passes at L 46 (`margin_ok_at_L_semantic_norm: true`) (001_layers_baseline/run-latest/output-gemma-2-27b.json:10366–10373).

4.5. Prism (shared‑decoder diagnostic)
Prism is present and compatible (001_layers_baseline/run-latest/output-gemma-2-27b.json:840–848). KL drops substantially at sampled depths: baseline p50≈43.15 → prism p50≈19.42 (Δ≈23.73 bits), with no change in rank milestones (all null deltas) (001_layers_baseline/run-latest/output-gemma-2-27b.json:864–888). Verdict: Helpful — large early KL reductions with unchanged rank earliness.

4.6. Ablation & stress tests
No‑filler ablation shifts copy but not semantics: `L_copy_orig: 0`, `L_copy_nf: 3` (ΔL_copy=3); `L_sem_orig: 46`, `L_sem_nf: 46` (ΔL_sem=0) (001_layers_baseline/run-latest/output-gemma-2-27b.json:8755–8765). Control prompt is "The capital of France is called simply" with gold "Paris" (001_layers_baseline/run-latest/output-gemma-2-27b.json:8765–8781). Control margins summarize as above (001_layers_baseline/run-latest/output-gemma-2-27b.json:8789–8800). Important‑word trajectory: in the positive context, layer‑0 token ‘Germany’ is top‑1 on its position (001_layers_baseline/run-latest/output-gemma-2-27b-records.csv:15); the predicted next token evolves from ‘ simply’ early to ‘ Berlin’ only at L 46 (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2,48).

4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓ (RMSNorm; norm lens in use) (001_layers_baseline/run-latest/output-gemma-2-27b.json:806–813).
- LayerNorm bias removed ✓ (`layernorm_bias_fix: not_needed_rms_model`) (001_layers_baseline/run-latest/output-gemma-2-27b.json:813–816).
- FP32 unembed promoted ✓ (`unembed_dtype: torch.float32`) (001_layers_baseline/run-latest/output-gemma-2-27b.json:811–816).
- Punctuation / markup anchoring noted ✓ (copy mask shows punctuation IDs) (001_layers_baseline/run-latest/output-gemma-2-27b.json:953–1040).
- Copy‑reflex ✓ (copy collapse at L 0; soft k1 hit) (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2).
- Preferred lens honored ✓ (`preferred_lens_for_reporting: "norm"`) (001_layers_baseline/run-latest/output-gemma-2-27b.json:10465–10472).
- Confirmed semantics reported ✓ (`use_confirmed_semantics: true`, source tuned at L 46) (001_layers_baseline/run-latest/output-gemma-2-27b.json:10472–10506,10576–10583).
- Dual‑lens artefact metrics (incl. v2, JS/Jaccard/L1) cited ✓ (001_layers_baseline/run-latest/output-gemma-2-27b.json:5796–5835,10506–10518).
- Tuned‑lens audit done (rotation/temp/positional/head) ✓ (001_layers_baseline/run-latest/output-gemma-2-27b.json:10528–10553).
- normalization_provenance present (ln_source @ L0/final) ✓ (001_layers_baseline/run-latest/output-gemma-2-27b.json:5927,5933,6347).
- per‑layer normalizer effect present (resid_norm_ratio, delta_resid_cos) ✓ (001_layers_baseline/run-latest/output-gemma-2-27b.json:5936–5965,6320–6354).
- deterministic_algorithms true ✓ (001_layers_baseline/run-latest/output-gemma-2-27b.json:8733–8741).
- numeric_health clean ✓ (001_layers_baseline/run-latest/output-gemma-2-27b.json:6640–6647).
- copy_mask plausible ✓ (001_layers_baseline/run-latest/output-gemma-2-27b.json:953–1040).
- milestones.csv used for quotes ✓ (001_layers_baseline/run-latest/output-gemma-2-27b-milestones.csv:1).

Limitations and data quirks
- Final‑head calibration is imperfect (`warn_high_last_layer_kl: true`; non‑zero final KL), so probability comparisons are suppressed; ranks/KL thresholds are preferred (001_layers_baseline/run-latest/output-gemma-2-27b.json:6663–6672,10465–10472).
- Raw‑vs‑norm artefact tier is high with one norm‑only semantic layer at 46; early semantics could be lens‑induced without confirmation (001_layers_baseline/run-latest/output-gemma-2-27b.json:5796–5813,5923–5931).
- Normalization spike flagged in diagnostics; we did not observe numeric‑health layer failures (001_layers_baseline/run-latest/output-gemma-2-27b.json:851,6640–6647).

---
Produced by OpenAI GPT-5

*Run executed on: 2025-10-11 21:50:12*
