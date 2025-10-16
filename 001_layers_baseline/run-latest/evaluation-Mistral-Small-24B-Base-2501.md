# Evaluation Report: mistralai/Mistral-Small-24B-Base-2501

*Run executed on: 2025-10-16 07:26:19*

## EVAL

**Overview**
- Mistral-Small-24B-Base-2501 (≈24B), run timestamp 2025-10-16 07:26 (see `001_layers_baseline/run-latest/timestamp-20251016-0726`). The probe traces copy vs. semantic emergence using logit-lens across layers, tracking KL-to-final, rank milestones, cosine geometry, entropy, and lens diagnostics.
- Both norm and raw lenses are audited; tuned-lens is loaded for calibration/rotation analysis; final-head calibration is checked via last-layer consistency.

**Method Sanity‑Check**
- Prompt & indexing: "Give the city name only, plain text. The capital of Germany is called simply" (no trailing space) in `diagnostics.context_prompt`; positive/original rows present (e.g., L33 with top‑1 Berlin) in `output-Mistral-Small-24B-Base-2501-pure-next-token.csv:35`.
- Normalizer provenance: arch "pre_norm"; strategy `{ primary: "next_ln1", ablation: "post_ln2_vs_next_ln1@targets" }`; `per_layer[0].ln_source = blocks[0].ln1`, final uses `ln_final` (see `diagnostics.normalization_provenance`).
- Per‑layer normalizer effect: early spikes flagged; `norm_trajectory.shape = "spike"` and `flags.normalization_spike = true`. Representative early values: `resid_norm_ratio=43.83`, `delta_resid_cos=0.49` at L0.
- Unembed bias: `present=false; l2_norm=0.0` (cosines are bias‑free).
- Environment & determinism: `device=cpu`, `torch=2.8.0+cu128`, `deterministic_algorithms=true`, `seed=316` (see `provenance.env`).
- Last‑layer consistency: `kl_to_final_bits=0.0`, `top1_agree=true`, `temp_est=1.0`, `warn_high_last_layer_kl=false` (final‑head well calibrated).
- Repeatability (forward‑of‑two): `enabled=true`, `mode="skipped_deterministic"`, `gate.repeatability_forward_pass=null` (skipped due to deterministic env); treat forward‑of‑two gate as unassessed.
- Decoding‑point ablation (pre‑norm): `gate.decoding_point_consistent=false`; at `L_semantic_norm` (33): `rank1_agree=true`, `jaccard@10=0.333`, `jaccard@50=0.429`. At `first_rank_le_5` (30): `rank1_agree=false`, `jaccard@10=0.333` (see `diagnostics.decoding_point`).
- Numeric health: `any_nan=false`, `any_inf=false`, `layers_flagged=[]`.
- Copy mask: ignored_token_ids sample includes control/punctuation; `size=2931`; sample: `"\t", "\n", "\u001c", " ", "!", "\"", "#"` (see `diagnostics.copy_mask`).
- Gold alignment: `gold_alignment_rate=1.0`, variant `with_space`; Germany→Berlin and control France→Paris aligned (see `gold_alignment_prompts`).
- Repeatability (decode micro‑check §1.39): `max_rank_dev=0.0`, `p95_rank_dev=0.0`, `top1_flip_rate=0.0` (clean per‑layer repeatability).
- Norm trajectory: `shape="spike"`, `slope≈0.105`, `r2≈0.915`, `n_spikes=34`.
- Measurement guidance: `prefer_ranks=true`, `suppress_abs_probs=true`; reasons include `decoding_point_sensitive`, `low_lens_consistency_at_semantic`, `pos_window_low_stability`; `preferred_lens_for_reporting="tuned"`; `use_confirmed_semantics=true`.
- Semantic margin: `{ delta_abs=0.002, p_uniform≈7.6e-06, margin_ok_at_L_semantic_norm=false }`; no confirmed‑margin override.
- Gate‑stability (small rescalings): `min_both_gates_pass_frac=0.0`; at L_semantic_norm(33): both uniform and top‑2 pass fractions `0.0` → calibration‑sensitive.
- Position‑window: grid `[0.2,0.4,0.6,0.8,0.92,0.98]`; `rank1_frac=0.0` at `L_semantic_norm=33` → position‑fragile.
- Raw‑vs‑Norm artifact overview: risk tier `low`; `lens_artifact_score_v2=0.1847`, `js_divergence_p50=0.0353`, `l1_prob_diff_p50=0.3473`, `first_js_le_0.1=0`, `first_l1_le_0.5=0`, `jaccard_raw_norm_p50=0.5385`, `first_jaccard_raw_norm_ge_0.5=3`.
- Micro‑suite: `aggregates.n=5`; `L_semantic_confirmed_median=33`, `L_semantic_norm_median=32` (IQR 31–33), `n_missing=0`.

**Quantitative Findings (layer‑by‑layer)**
- L0 — entropy 16.9985 bits, top‑1 'Forbes' (non‑answer) [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:2].
- L30 — entropy 16.7426 bits, top‑1 '‑на' (non‑answer); answer rank 3 [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:32].
- L31 — entropy 16.7931 bits, top‑1 '‑на' (non‑answer); answer rank 2 [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:33].
- L32 — entropy 16.7888 bits, top‑1 '‑на' (non‑answer); answer rank 2 [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:34].
- L33 — entropy 16.7740 bits, top‑1 'Berlin' (answer rank 1) [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:35].

Semantic layer: prefer confirmed semantics under guidance. L_semantic_confirmed = 33 (source=raw), margin gate at L_semantic_norm fails; arch is pre_norm and decoding_point gate fails, so report as bold but decoding‑point sensitive.

- Control summary: `first_control_margin_pos=1`, `max_control_margin=0.468`, `first_control_strong_pos=31` (see `control_summary`).
- Micro‑suite: median L_semantic_confirmed=33; median L_semantic_norm=32 (IQR 31–33). Example fact: France→Paris hits rank‑1 at L33 [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:117].
- Entropy drift: entropy gap vs teacher large across depths; percentiles `{p25≈13.58, p50≈13.59, p75≈13.66}` (summary.entropy). Representative at L33: `entropy_bits=16.7740` vs `teacher_entropy_bits=3.1807` [pure CSV row 35].
- Confidence snapshot at L33: `answer_logit_gap≈0.322` and `answer_vs_top1_gap≈0.107` (use logit margins; absolute probabilities suppressed) [pure CSV row 35]. Early normalizer effect at L0: `resid_norm_ratio≈43.83`, `delta_resid_cos≈0.49` (diagnostics.normalization_provenance).

**Qualitative Findings**

4.1. Copy vs semantics (Δ‑gap)
No early copy reflex: strict copy never fires at τ∈{0.70,0.95} and soft windows k∈{1,2,3} absent (`summary.copy_thresholds.L_copy_strict = null` across τ; `stability="none"`). Earliest strict copy at τ=0.70 and τ=0.95: null; no `norm_only_flags` set. Using L_semantic_norm=33 and no L_copy, Δ̂ ≈ 33/40 = 0.825 (also `evaluation_pack.milestones.depth_fractions.semantic_frac=0.825`).

4.2. Lens sanity: Raw‑vs‑Norm
Artifact risk is low: `lens_artifact_score_v2=0.1847 (tier=low)` with symmetric metrics `js_divergence_p50=0.0353`, `l1_prob_diff_p50=0.3473`, `first_js_le_0.1=0`, `first_l1_le_0.5=0`. Top‑K overlap robust in median: `jaccard_raw_norm_p50=0.5385`, with `first_jaccard_raw_norm_ge_0.5=3`. No norm‑only semantic layers near candidate semantics (`n_norm_only_semantics_layers=0`). At the semantic target, raw vs norm lens‑consistency is moderate: at L33, `jaccard@10≈0.538`, `jaccard@50≈0.515`, `spearman_top50≈0.494` (`diagnostics.lens_consistency`). Given decoding‑point sensitivity and margin failures, prefer rank/KL milestones and confirmed semantics over absolute probabilities.

4.3. Tuned‑Lens analysis
`tuned_is_calibration_only=false`; guidance prefers tuned for reporting (`preferred_lens_for_reporting="tuned"`, `preferred_semantics_lens_hint="tuned"`). Rotation dominates temperature in KL deltas: `ΔKL_rot_p50≈1.764`, `ΔKL_temp_p50≈−0.328`, with positive interaction `≈3.620` and net `ΔKL_tuned_p50≈5.083` (tuned audit). Positional generalization: `pos_in_dist_le_0.92≈4.43` vs `pos_ood_ge_0.96≈5.08` (gap ≈0.656). Head mismatch clean: `kl_bits_tuned_final=0.0` and `tau_star_modelcal=1.0`. Last‑layer consistency shows perfect agreement (`kl_to_final_bits=0.0`, `top1_agree=true`). We therefore use tuned for qualitative timing, but retain norm/confirmed layer 33 for headline semantics under guidance.

4.4. KL, ranks, cosine, entropy milestones
KL to final crosses 1.0 and 0.5 bits only at the end (`first_kl_below_1.0=40`, `first_kl_below_0.5=40`); final KL≈0 (last‑layer consistency). Ranks (norm): `first_rank_le_10=30`, `first_rank_le_5=30`, `first_rank_le_1=33` (margin gate at L33 fails; treat as weak rank‑1). Cosine to final (norm) milestones are late: `ge_0.2=35`, `ge_0.4=40`, `ge_0.6=40`. Entropy monotonically decreases with large positive drift vs teacher (`entropy_gap_bits_p50≈13.59`), with semantic rank improvements preceding calibration of KL late. Stability reminders: (a) margin gate at L_semantic_norm fails, (b) no run‑of‑two strong layer; (c) decoding‑point gate fails in pre‑norm; (d) gate‑stability under small rescalings 0.0 and position‑window rank1_frac=0.0 → calibration‑sensitive and position‑fragile.

4.5. Prism (shared‑decoder diagnostic)
Prism present and compatible. At sampled depths, prism increases KL (δ at p50 ≈ −5.98 bits relative to baseline norm, i.e., worse), with no earlier `first_rank_le_1` (baseline 33 vs prism null). Verdict: Regressive for this prompt/config.

4.6. Ablation & stress tests
No‑filler ablation shifts semantics earlier by 2 layers: `L_sem_orig=33`, `L_sem_nf=31` (`ΔL_sem = −2`), under 10% of depth (n_layers=40) → mild stylistic sensitivity. Control prompt (France→Paris) shows strong positive margins (`first_control_margin_pos=1`, `first_control_strong_pos=31`). Important‑word trajectory (records CSV): context token 'Germany' tracked across depth, e.g., L0 and L1 show diffuse non‑semantic distributions [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-records.csv:15, 34].

4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓
- LayerNorm bias removed ✓
- FP32 unembed promoted ✓
- Punctuation / markup anchoring noted ✓
- Copy‑reflex ✗
- Preferred lens honored ✓ (tuned for reporting; norm used for confirmed layer)
- Confirmed semantics reported ✓ (L33, decoding‑point sensitive)
- Dual‑lens artefact metrics cited ✓ (incl. v2, JS/Jaccard/L1)
- Tuned‑lens audit done ✓ (rotation/temp/positional/head)
- normalization_provenance present ✓ (ln_source @ L0/final)
- per‑layer normalizer effect present ✓ (resid_norm_ratio, delta_resid_cos)
- deterministic_algorithms true ✓
- numeric_health clean ✓
- copy_mask plausible ✓ (size=2931; punctuation/whitespace examples)
- milestones.csv or evaluation_pack.citations used ✓
- gate_stability_small_scale reported ✓ (min both‑gates pass frac=0.0)
- position_window stability reported ✓ (rank1_frac=0.0)

—
Produced by OpenAI GPT-5
