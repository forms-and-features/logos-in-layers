# Evaluation Report: Qwen/Qwen2.5-72B

*Run executed on: 2025-10-13 22:23:35*

## EVAL

**1. Overview**

This evaluation covers Qwen2.5-72B using the layer-by-layer logit lens probe to contrast copy reflexes versus semantic emergence, and to track KL-to-final, rank progressions, cosine geometry, and entropy trajectories with lens diagnostics. Evidence is drawn from `001_layers_baseline/run-latest/output-Qwen2.5-72B.json` and companion CSVs for pure lens, raw-vs-norm, and Prism.

**2. Method sanity-check**

- Prompt & indexing: Context prompt ends with “called simply” with no trailing space: "Give the city name only, plain text. The capital of Germany is called simply"  (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:299). Positive baseline rows exist: "Germany→Berlin,0,pos,orig,0,15,…"  [row 2 in 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv].
- Normalizer provenance: "arch": "pre_norm"; first layer ln_source "blocks[0].ln1" and final "ln_final" (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9264 and 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7256–7260). Example per-layer entry shows expected fields: "layer": 2, "ln_source": "blocks[2].ln1", "resid_norm_ratio": 8.939… (001_layers_baseline/run-latest/output-Qwen2.5-72B.json: per-layer dump).
- Per-layer normalizer effect: Early spikes are present (e.g., L2 resid_norm_ratio ≈ 8.94; delta_resid_cos ≈ 0.58) (001_layers_baseline/run-latest/output-Qwen2.5-72B.json: per-layer; see also measurement reasons "normalization_spike").
- Unembed bias: "present": false; "l2_norm": 0.0 (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:834–837). Cosines are thus bias‑free.
- Environment & determinism: "device": "cpu", "dtype_compute": "torch.bfloat16", "deterministic_algorithms": true, "seed": 316 (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9650–9658). Reproducibility is adequate.
- Numeric health: "any_nan": false, "any_inf": false, "layers_flagged": [] (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8859–8866).
- Copy mask: "size": 6244 (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7248–7260). Plausible for a modern tokenizer.
- Gold alignment: "ok": true, "pieces": ["ĠBerlin"] (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8904–8922). "gold_alignment_rate": 1.0 (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8953, 9890).
- Repeatability (1.39): "status": "skipped", "reason": "deterministic_env" (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8862–8868).
- Norm trajectory: "shape": "spike", slope ≈ 0.064, r2 ≈ 0.923, n_spikes=55 (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9264–9270).
- Measurement guidance: "prefer_ranks": true, "suppress_abs_probs": true, "preferred_lens_for_reporting": "norm", "use_confirmed_semantics": false (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9831–9850). Reasons include "norm_only_semantics_window", "high_lens_artifact_risk", "low_lens_consistency_at_semantic".
- Semantic margin: "delta_abs": 0.002, "p_uniform": 6.576e-06, "margin_ok_at_L_semantic_norm": true (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7378–7390).
- Gate‑stability (small rescalings): L_semantic_norm=80: uniform_margin_pass_frac=1.0, top2_gap_pass_frac=0.0, both_gates_pass_frac=0.0; min_both_gates_pass_frac=0.0 (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7402–7424). Calibration‑sensitive.
- Position‑window: grid=[], n_positions=0, rank1_frac=null (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9257–9263). Not measured; avoid generalizing beyond the measured next‑token position.
- Micro‑suite: Aggregates present with n=5, n_missing=3; L_semantic_norm_median=80 (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9808–9828). Example fact citation: Germany→Berlin has row_index=80 (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9757–9770).

**3. Quantitative findings (layer‑by‑layer)**

- L 0 — entropy 17.214 bits; top‑1 's' [row 2 in 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv].
- L 74 — entropy 16.081 bits; top‑1 '"""'; answer_rank=9 [row 132 in 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv].
- L 78 — entropy 15.398 bits; top‑1 '"""'; answer_rank=5 [row 136 in 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv].
- L 79 — entropy 16.666 bits; answer_rank=2 [row 137 in 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv].
- L 80 — entropy 4.116 bits; top‑1 'Berlin'; answer_rank=1; KL_to_final≈0.00011 bits [row 138 in 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv].
- Control margins: first_control_margin_pos=0; max_control_margin≈0.207 (001_layers_baseline/run-latest/output-Qwen2.5-72B.json: control_summary).
- Micro‑suite: median L_semantic_norm=80; delta_hat_median=null; example fact citation row_index=80 (Germany→Berlin) (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9757–9770, 9808–9828).
- Entropy drift (aggregates): entropy_gap_bits_p25≈12.03, p50≈12.50, p75≈12.77 (001_layers_baseline/run-latest/output-Qwen2.5-72B.json: evaluation_pack.entropy).

Bold semantic layer: L_semantic_norm=80. Uniform‑margin gate passes, but Top‑2 gate is absent/unstable and gate‑stability fails under small rescalings; report as weak semantic onset.

**4. Qualitative findings**

4.1. Copy vs semantics (Δ‑gap)

Copy‑reflex is not detected early: strict copy milestones are null and soft‑k hits are absent in early layers (evaluation_pack.milestones shows L_copy_strict: null; L_copy_soft: {k: null, layer: null}). Thus semantics emerge only at the last layer (L_semantic_norm=80). Δ̂ is not reported (delta_hat=null). Threshold stability not provided (summary.copy_thresholds=null). Earliest strict copy at τ=0.70 and τ=0.95 are not present (001_layers_baseline/run-latest/output-Qwen2.5-72B.json: evaluation_pack.milestones).

4.2. Lens sanity: Raw‑vs‑Norm

Artifact metrics indicate high risk: lens_artifact_score_v2≈0.7426, tier=high (001_layers_baseline/run-latest/output-Qwen2.5-72B-artifact-audit.csv:_summary). Symmetric robustness is weak: js_divergence_p50≈0.1052; l1_prob_diff_p50≈0.6151; first_js_le_0.1=None; first_l1_le_0.5=None (001_layers_baseline/run-latest/output-Qwen2.5-72B.json: diagnostics.raw_lens_full). Top‑K overlap is low: jaccard_raw_norm_p50≈0.316; first_jaccard_raw_norm_ge_0.5=None (same). Prevalence: pct_layers_kl_ge_1.0≈0.321; a norm‑only semantic layer appears at 80 (diagnostics.raw_lens_window.norm_only_semantics_layers=[80]). Lens consistency at targets is low (e.g., at L=80: jaccard@10=0.25, jaccard@50≈0.299, spearman_top50≈0.370) (001_layers_baseline/run-latest/output-Qwen2.5-72B.json: diagnostics.lens_consistency). Caution: early semantics may be lens‑induced; prefer rank/KL milestones and treat absolute probabilities as suppressed.

4.3. Tuned‑Lens analysis

Not present for this run (tuned_lens.audit_summary: null; tuned variants CSV absent). Prefer norm lens for semantics per measurement guidance (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9831–9850).

4.4. KL, ranks, cosine, entropy milestones

KL thresholds: first_kl_below_1.0 and 0.5 occur at L80 (KL_to_final≈0.00011 bits) [row 138 in 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv]. Ranks (norm lens): first_rank_le_10 at L74 [row 132], first_rank_le_5 at L78 [row 136], first_rank_le_1 at L80 [row 138]. Cosine milestones not reported (summary.cos_milestones=null). Entropy decreases sharply late; aggregated entropy gaps are large (evaluation_pack.entropy p50≈12.50), aligning with late calibration rather than early certainty. Margin gate: uniform‑margin passes at L80; Top‑2 margin is absent/unstable; stability: no run‑of‑two; gate‑stability min_both_gates_pass_frac=0.0 and position‑window is unmeasured, so treat the onset as calibration‑sensitive and position‑specific.

4.5. Prism (shared‑decoder diagnostic)

Present (output-Qwen2.5-72B-pure-next-token-prism.csv). At L80, Prism does not recover the gold token ('Berlin'); answer_rank is 106311 and KL is large (20.68 bits) [row 120 in 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token-prism.csv], whereas the norm lens aligns with the final head (KL≈0.00011 bits; answer_rank=1) [row 138 in pure CSV]. Verdict: Regressive (higher KL and later/no rank‑1).

4.6. Ablation & stress tests

No‑filler ablation leaves semantics unchanged: "L_sem_orig": 80, "L_sem_nf": 80, "delta_L_sem": 0 (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9669–9696). Control summary: first_control_margin_pos=0; max_control_margin≈0.207 (001_layers_baseline/run-latest/output-Qwen2.5-72B.json: control_summary). Important‑word trajectory: by L74, 'Berlin' already appears among top candidates at the final position (records CSV shows 'Berlin' among top items at pos=15) [row 3911 in 001_layers_baseline/run-latest/output-Qwen2.5-72B-records.csv].

4.7. Checklist (✓/✗/n.a.)

- RMS lens ✓ (eps_inside_sqrt=true; pre_norm; correct γ selection)
- LayerNorm bias removed ✓ (unembed_bias.present=false)
- FP32 unembed promoted ✓/n.a. (final‑head agreement OK; probabilities suppressed per guidance)
- Punctuation/markup anchoring noted ✓ (early top‑Ks dominated by quotes/punctuation; e.g., L74 [row 132])
- Copy‑reflex ✗ (no strict/soft copy milestones; evaluation_pack.milestones)
- Preferred lens honored ✓ (measurement_guidance.preferred_lens_for_reporting = "norm")
- Confirmed semantics reported ✗ (summary.L_semantic_confirmed: null)
- Dual‑lens artefact metrics cited ✓ (artifact‑audit summary; raw‑vs‑norm JS/L1/Jaccard)
- Tuned‑lens audit done ✗ (not present)
- normalization_provenance present ✓ (ln_source @ L0 and final)
- per‑layer normalizer effect present ✓ (resid_norm_ratio, delta_resid_cos reported)
- deterministic_algorithms true ✓ (env shows deterministic_algorithms=true)
- numeric_health clean ✓ (any_nan=false, any_inf=false)
- copy_mask plausible ✓ (size=6244)
- milestones.csv or evaluation_pack.citations used ✓ (milestones and micro_suite row_index cited)
- gate_stability_small_scale reported ✓ (min_both_gates_pass_frac=0.0; per‑target fractions quoted)
- position_window stability reported ✓ (grid=[], rank1_frac=null)

---
Produced by OpenAI GPT-5
