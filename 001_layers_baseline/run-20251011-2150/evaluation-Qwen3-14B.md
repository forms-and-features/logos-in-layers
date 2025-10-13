# Evaluation Report: Qwen/Qwen3-14B

*Run executed on: 2025-10-11 21:50:12*
## EVAL

### Overview
Qwen3-14B (40 layers), run on 2025-10-11 (timestamp file). The probe measures copy vs. semantics onset, KL-to-final and rank milestones, cosine and entropy trajectories, and dual‑lens (raw vs. norm) diagnostics, plus tuned‑lens attribution/calibration.

### Method sanity‑check
- Prompt & indexing: context ends with “called simply” with no trailing space; pos/orig rows present, e.g., L0 and L36 for Germany→Berlin [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:2] and [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:38].
- Normalizer provenance: strategy=next_ln1; ln_source L0=blocks[0].ln1, final=ln_final (output-Qwen3-14B.json → diagnostics.normalization_provenance).
- Per‑layer normalizer effect: norm trajectory shape=spike (r2≈0.905), flagged in guidance (“normalization_spike”) (output-Qwen3-14B.json → diagnostics.norm_trajectory, measurement_guidance.reasons).
- Unembed bias: present=False; l2_norm=0.0; cosines are bias‑free (output-Qwen3-14B.json → diagnostics.unembed_bias).
- Environment & determinism: device=cpu, torch=2.8.0, deterministic_algorithms=True, seed=316 (output-Qwen3-14B.json → provenance.env). Repeatability self‑test skipped due to deterministic_env (diagnostics.repeatability).
- Numeric health: any_nan=False, any_inf=False, layers_flagged=[] (output-Qwen3-14B.json → diagnostics.numeric_health).
- Copy mask: size=6112; sample=[]; plausible for tokenizer granularity (output-Qwen3-14B.json → diagnostics.copy_mask).
- Gold alignment: ok; pieces=["ĠBerlin"], variant=with_space (output-Qwen3-14B.json → diagnostics.gold_alignment).
- Repeatability (1.39): status=skipped; reason=deterministic_env (output-Qwen3-14B.json → diagnostics.repeatability).
- Norm trajectory: shape=spike; slope≈0.111; r2≈0.905; n_spikes=14 (output-Qwen3-14B.json → diagnostics.norm_trajectory).
- Measurement guidance: prefer_ranks=True; suppress_abs_probs=True; preferred_lens_for_reporting=tuned; use_confirmed_semantics=True (output-Qwen3-14B.json → measurement_guidance).
- Semantic margin: δ_abs=0.002, p_uniform≈6.58e−6; margin_ok_at_L_semantic_norm=True; L_semantic_confirmed_margin_ok_norm=36 (output-Qwen3-14B.json → summary.semantic_margin).
- Micro‑suite: aggregates present; n=5, n_missing=0; median L_semantic_confirmed=36 (output-Qwen3-14B.json → evaluation_pack.micro_suite.aggregates).

### Quantitative findings (layer‑by‑layer)
- L 0 — entropy 17.213 bits, top‑1 ‘梳’ [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:2]
- L 5 — entropy 17.207 bits, top‑1 ‘-minded’ [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:7]
- L 10 — entropy 17.170 bits, top‑1 ‘ (?)’ [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:12]
- L 20 — entropy 16.932 bits, top‑1 ‘____’ [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:22]
- L 30 — entropy 7.789 bits, top‑1 ‘这个名字’ [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:32]
- L 36 — entropy 0.312 bits, top‑1 ‘Berlin’ (answer) [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:38]
- L 40 — entropy 3.584 bits, top‑1 ‘Berlin’ [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:42]

Bold semantic layer: L 36 (confirmed; margin gate OK). Control margin: first_control_margin_pos=0; max_control_margin≈0.974 (output-Qwen3-14B.json → control_summary).

Micro‑suite: median L_semantic_confirmed=36 (n=5, n_missing=0). For concreteness, France→Paris hits at L36 [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:120].

Entropy drift (Germany→Berlin, pos/orig): L0 17.213 vs. teacher 3.584 (higher), L30 7.789 vs. 3.584 (higher), L36 0.312 vs. 3.584 (lower), L40 3.584 vs. 3.584 (aligned) [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:2,32,38,42]. Percentiles: entropy_gap_bits p25≈4.21, p50≈13.40, p75≈13.59 (output-Qwen3-14B.json → diagnostics.entropy_gap_bits_percentiles).

Representative normalizer effect at L36 (pos/orig): resid_norm_ratio≈0.234; delta_resid_cos≈0.733 [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:38]. Confidence margin snapshot at L36: answer_logit_gap≈3.14 (answer is top‑1) [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:38].

### Qualitative findings

#### 4.1 Copy vs semantics (Δ‑gap)
No early copy‑reflex: layers 0–3 show copy_collapse=False and copy_soft_k1@0.5=False (e.g., L0–L3 rows) [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:2]. Strict copy thresholds τ∈{0.70,0.95}: earliest strict copy = None at both; stability=“none” (output-Qwen3-14B.json → diagnostics.copy_thresholds). With L_copy_strict missing and no soft hit, Δ̂ is undefined; evaluation_pack.milestones.depth_fractions.delta_hat=None (output-Qwen3-14B.json → evaluation_pack.milestones).

#### 4.2 Lens sanity: Raw‑vs‑Norm
Artifact risk is high: lens_artifact_score_v2≈0.704, tier=high; js_divergence_p50≈0.513; l1_prob_diff_p50≈1.432; first_js_le_0.1=0; first_l1_le_0.5=0; top‑K overlap jaccard_p50≈0.25; first_jaccard_ge_0.5=0 (output-Qwen3-14B.json → diagnostics.raw_lens_full.score/topk_overlap and evaluation_pack.artifact). Prevalence: pct_layers_kl_ge_1.0≈0.756; no norm‑only semantic layers; earliest_norm_only_semantic=None (output-Qwen3-14B.json → diagnostics.raw_lens_full). Windowed check near semantics reports max KL≈98.6 bits (radius=4 around layers 33/36/40) (output-Qwen3-14B.json → diagnostics.raw_lens_window). Caution: early “semantics” could be lens‑induced; prefer ranks and confirmed semantics.

#### 4.3 Tuned‑Lens analysis
Preference: tuned_is_calibration_only=False; preferred_semantics_lens_hint=tuned (output-Qwen3-14B.json → tuned_lens.audit_summary). Attribution: ΔKL rotation (p25≈1.73, p50≈1.97, p75≈2.06), ΔKL temperature (p25≈−0.11, p50≈≈0.00, p75≈+0.03), interaction p50≈3.22; tuned ΔKL p50≈5.01 (output-Qwen3-14B.json → tuned_lens.audit_summary.rotation_vs_temperature). Rank earliness (tuned vs norm, summary): first_rank_le_10 33 (tuned) vs 32 (norm); le_5 34 vs 33; le_1 39 vs 36 (later with tuned) (output-Qwen3-14B.json → tuned_lens.summary.metrics.rank_milestones). Positional generalization: pos_in_dist_le_0.92≈5.51; pos_ood_ge_0.96≈5.01; gap≈−0.49 (output-Qwen3-14B.json → tuned_lens.audit_summary.positional). Head mismatch: tau_star_modelcal=1.0; kl_bits_tuned_final→0.0 after τ⋆; last‑layer consistency: kl_to_final_bits=0.0, top1_agree=True (output-Qwen3-14B.json → tuned_lens.audit_summary.head_mismatch; diagnostics.last_layer_consistency).

#### 4.4 KL, ranks, cosine, entropy milestones
KL: first_kl_below_1.0=40; first_kl_below_0.5=40; final KL≈0 (output-Qwen3-14B.json → diagnostics.first_kl_below_* and diagnostics.last_layer_consistency). Ranks (preferred=tuned; baseline in parentheses): le_10=33 (32), le_5=34 (33), le_1=39 (36) (output-Qwen3-14B.json → tuned_lens.summary.metrics.rank_milestones). Cosine milestones (norm lens): ge_0.2@L5, ge_0.4@L29, ge_0.6@L36 (output-Qwen3-14B.json → diagnostics.cos_milestones). Entropy: large early gap vs teacher, sharp drop by L36, then calibrated to teacher at L40 [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:2,32,38,42]. Margin gate: passes at L36 (summary.semantic_margin.margin_ok_at_L_semantic_norm=True).

#### 4.5 Prism (shared‑decoder diagnostic)
Present and compatible (k=512; layers [embed,9,19,29]) (output-Qwen3-14B.json → diagnostics.prism_summary). KL deltas at sampled depths: prism KL is higher by ~0.25–0.71 bits at p50/p75; rank milestones unchanged (prism: none earlier) (output-Qwen3-14B.json → diagnostics.prism_summary.metrics). Verdict: Regressive.

#### 4.6 Ablation & stress tests
No‑filler ablation: L_sem_orig=36, L_sem_nf=36 → ΔL_sem=0 (output-Qwen3-14B.json → ablation_summary). Control prompts: first_control_margin_pos=0; max_control_margin≈0.974 (output-Qwen3-14B.json → control_summary). Test prompt “Berlin is the capital of” shows ‘Germany’ as the next‑token top candidate (output-Qwen3-14B.json → test_prompts[0]). Important‑word trajectory: ‘Berlin’ is top‑1 by L36 and remains so at L40 [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:38,42].

#### 4.7 Checklist (✓/✗/n.a.)
✓ RMS lens
✓ LayerNorm bias removed
✗ FP32 unembed promoted (use_fp32_unembed=False)
✗ Punctuation/markup anchoring noted
✗ Copy‑reflex (absent)
✓ Preferred lens honored (tuned for reporting; confirmed semantics)
✓ Confirmed semantics reported
✓ Dual‑lens artefact metrics (v2, JS/Jaccard/L1) cited
✓ Tuned‑lens audit (rotation/temp/positional/head) cited
✓ normalization_provenance present (ln_source @ L0/final)
✓ per‑layer normalizer effect (resid_norm_ratio, delta_resid_cos)
✓ deterministic_algorithms true
✓ numeric_health clean
✓ copy_mask plausible
✓ CSV row citations used

---
Produced by OpenAI GPT-5

