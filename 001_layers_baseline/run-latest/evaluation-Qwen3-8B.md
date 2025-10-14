# Evaluation Report: Qwen/Qwen3-8B

*Run executed on: 2025-10-13 22:23:35*

## EVAL

**Overview**
- Qwen3-8B (36 layers; pre-norm; d_model 4096) evaluated on 2025-10-13 across a capital‑city prompt battery. The probe measures copy vs. semantic emergence using rank/KL, cosine, entropy trajectories, and lens diagnostics (norm vs. raw, tuned, prism).
- We emphasize rank/KL milestones and lens consistency; probabilities are suppressed per guidance due to high artefact risk and family calibration.

**Method sanity-check**
- Prompt & indexing: "Give the city name only, plain text. The capital of Germany is called simply" (no trailing space) [001_layers_baseline/run-latest/output-Qwen3-8B.json]. Positive rows `prompt_id=pos, prompt_variant=orig` are present (e.g., layer 0 and 31) [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:2, 33].
- Normalizer provenance: `strategy: next_ln1`; `L0 ln_source: blocks[0].ln1`; `L_last ln_source: ln_final` [001_layers_baseline/run-latest/output-Qwen3-8B.json].
- Per-layer normalizer effect: norm trajectory `shape: spike`, `n_spikes: 15`, `slope: 0.1288`, `r2: 0.9215` [001_layers_baseline/run-latest/output-Qwen3-8B.json]. Spikes are flagged; see caution under lens artefacts.
- Unembed bias: `present: False`, `l2_norm: 0.0` (cosines are bias‑free) [001_layers_baseline/run-latest/output-Qwen3-8B.json].
- Environment & determinism: `device: cpu`, `torch 2.8.0+cu128`, `deterministic_algorithms: True`, `seed: 316` [001_layers_baseline/run-latest/output-Qwen3-8B.json].
- Numeric health: `any_nan: False`, `any_inf: False`, `layers_flagged: []` [001_layers_baseline/run-latest/output-Qwen3-8B.json].
- Copy mask: `size: 6112`, sample `['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':']` — plausible for tokenizer punctuation/markers [001_layers_baseline/run-latest/output-Qwen3-8B.json].
- Gold alignment: `{ok: True, pieces: ['ĠBerlin'], variant: 'with_space'}` [001_layers_baseline/run-latest/output-Qwen3-8B.json].
- Repeatability (1.39): `{status: 'skipped', reason: 'deterministic_env'}` [001_layers_baseline/run-latest/output-Qwen3-8B.json].
- Norm trajectory: `shape: spike`, `r2: 0.9215`, `n_spikes: 15` [001_layers_baseline/run-latest/output-Qwen3-8B.json].
- Measurement guidance: `{prefer_ranks: True, suppress_abs_probs: True, preferred_lens_for_reporting: 'tuned', use_confirmed_semantics: True}` with reasons including `high_lens_artifact_risk` and `pos_window_low_stability` [001_layers_baseline/run-latest/output-Qwen3-8B.json].
- Semantic margin: `{delta_abs: 0.002, p_uniform: 6.58e-06, margin_ok_at_L_semantic_norm: True, L_semantic_margin_ok_norm: 31}` [001_layers_baseline/run-latest/output-Qwen3-8B.json].
- Gate‑stability (small rescalings): `min_both_gates_pass_frac: 1.0`; at target `layer 31, both_gates_pass_frac: 1.0` [001_layers_baseline/run-latest/output-Qwen3-8B.json].
- Position‑window: `{L_semantic_norm: 31, grid: [0.2,0.4,0.6,0.8,0.92,0.98], rank1_frac: 0.0}` → position‑fragile [001_layers_baseline/run-latest/output-Qwen3-8B.json].
- Micro‑suite: aggregates present with `n=5`, `L_semantic_confirmed_median=31`, `delta_hat_median=0.0556`, `n_missing=0` [001_layers_baseline/run-latest/output-Qwen3-8B.json].

**Quantitative findings (layer‑by‑layer)**
- Control margins: `first_control_margin_pos=1`, `first_control_strong_pos=30`, `max_control_margin≈1.0` [001_layers_baseline/run-latest/output-Qwen3-8B.json].
- Positive (pos/orig) next‑token snapshots for Germany→Berlin:
  - L 0 — entropy 17.213 bits, top‑1 'CLICK' (answer rank 14864) [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:2].
  - L 30 — entropy 2.203 bits, top‑1 'Germany' (answer rank 2) [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:32].
  - L 31 — entropy 0.454 bits, top‑1 'Berlin'; answer_logit_gap 3.08; resid_norm_ratio 0.252; Δresid_cos 0.756 [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:33].
  - L 33 — entropy 0.988 bits, top‑1 'Berlin' (stable run‑of‑two onset) [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:35].
  - L 35 — entropy 2.494 bits, top‑1 'Berlin' (final) [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:37].
- Micro‑suite (medians): `L_semantic_confirmed_median=31`, `Δ̂_median=0.0556` across 5 facts; example citation Germany→Berlin at L=31 [001_layers_baseline/run-latest/output-Qwen3-8B-milestones.csv:3] and pure CSV row 33.
- Entropy drift: large early gap (p50 ≈ 13.79 bits) with remaining gap at p25 ≈ 1.66 bits [001_layers_baseline/run-latest/output-Qwen3-8B.json]. Representative: L0 (17.213 vs teacher 3.123), L31 (0.454 vs teacher 3.123) [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:2, 33].

**Qualitative findings**

4.1. Copy vs semantics (Δ‑gap)
- Copy‑reflex ×: no `copy_collapse` or `copy_soft_k1@0.5` in layers 0–3 for the positive prompt [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:2–5]. With no strict copy milestones (`L_copy_strict: None`; `L_copy_soft: None`) [001_layers_baseline/run-latest/output-Qwen3-8B-milestones.csv:3–4], semantics emerges sharply at L31.
- Δ̂ not reported (no copy onset). Stability tag: `summary.copy_thresholds: None`; strict copy at τ=0.70 and τ=0.95 are absent in pos/orig rows (no hits found in CSV scan).

4.2. Lens sanity: Raw‑vs‑Norm
- Artefact scores: `lens_artifact_score_v2=0.704`, tier `high` [001_layers_baseline/run-latest/output-Qwen3-8B.json]. Window check: `max_kl_norm_vs_raw_bits_window=38.10` over layers 25–36; no norm‑only semantics layers [001_layers_baseline/run-latest/output-Qwen3-8B.json].
- Symmetric/robust metrics: `first_js_le_0.1=0`, `first_l1_le_0.5=0`; p50 JS/L1 available via evaluation pack: `js_divergence_p50=0.358`, `l1_prob_diff_p50=1.134` [001_layers_baseline/run-latest/output-Qwen3-8B.json].
- Top‑K overlap: `jaccard_raw_norm_p50=0.282`, `first_jaccard_raw_norm_ge_0.5=0` [001_layers_baseline/run-latest/output-Qwen3-8B.json]. Prevalence: `pct_layers_kl_ge_1.0=0.7568`, `n_norm_only_semantics_layers=0` [001_layers_baseline/run-latest/output-Qwen3-8B.json].
- Lens‑consistency at semantic target is low: norm vs raw at L31 `jaccard@10≈0.176`, `jaccard@50≈0.370`, `spearman_top50≈0.341` [001_layers_baseline/run-latest/output-Qwen3-8B.json]. Given the high tier and low consistency, early semantics may be view‑dependent; we rely on rank/KL and confirmed semantics rather than absolute probabilities.

4.3. Tuned‑Lens analysis
- Preference: `preferred_semantics_lens_hint='tuned'`, `tuned_is_calibration_only=False`; we therefore use tuned for reporting where applicable [001_layers_baseline/run-latest/output-Qwen3-8B.json].
- Attribution (rotation vs temperature): ΔKL_rot p25/p50/p75 ≈ 0.81/0.92/1.01 bits; ΔKL_temp p25/p50/p75 ≈ 0.00/0.03/0.05; interaction p50 ≈ 2.82 [001_layers_baseline/run-latest/output-Qwen3-8B.json].
- Positional generalization: `pos_ood_ge_0.96≈3.77`, `pos_in_dist_le_0.92≈4.999`, `pos_ood_gap≈-1.23` [001_layers_baseline/run-latest/output-Qwen3-8B.json].
- Head mismatch: `tau_star_modelcal=1.0`, final KL stays `0.0 → 0.0` after τ* [001_layers_baseline/run-latest/output-Qwen3-8B.json].
- Last‑layer agreement: `top1_agree: True`, `kl_to_final_bits=0.0`, `warn_high_last_layer_kl=False` [001_layers_baseline/run-latest/output-Qwen3-8B.json]. No evidence that tuned vs model head disagree at the end.

4.4. KL, ranks, cosine, entropy milestones
- KL: final KL ≈ 0 per last‑layer consistency; mid‑depth KL under norm lens near semantics is modest (`kl_to_final_bits≈1.06` at L31) [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:33].
- Ranks (preferred lens: tuned): confirmed semantics at L31 (raw source) and L31 under norm; stable run‑of‑two at L33 [001_layers_baseline/run-latest/output-Qwen3-8B-milestones.csv:3–4].
- Cosine: `cos_to_answer` rises around L31 (≈0.107) with `cos_to_final` still negative (≈−0.252) [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:33].
- Entropy: strong early‑to‑mid drop (evaluation p50 gap ≈ 13.79 bits) with an uptick at the final layer (2.49 bits at L35) [001_layers_baseline/run-latest/output-Qwen3-8B.json; 001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:37]. Margin gate: uniform‑margin passes at L31; top‑2 gate is unspecified; we treat L31 as tentatively rank‑1 and prefer the run‑of‑two stability at L33. Position‑window rank‑1 fraction is 0.0 at L31 → position‑fragile.

4.5. Prism (shared‑decoder diagnostic)
- Present. At L31, prism is regressive vs norm: `kl_to_final_bits≈14.78` and top‑1 differs (rank 27292) [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token-prism.csv:33] versus norm `kl≈1.06` and rank‑1 answer [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:33]. Verdict: Regressive.

4.6. Ablation & stress tests
- No‑filler ablation: `L_sem_orig=31`, `L_sem_nf=31`, `ΔL_sem=0` → stylistically robust [001_layers_baseline/run-latest/output-Qwen3-8B.json].
- Negative/control prompts: strong control margins (`first_control_strong_pos=30`, `max_control_margin≈1.0`) [001_layers_baseline/run-latest/output-Qwen3-8B.json].
- Important‑word trajectory: token ' Germany' at position 12 shows self‑evidence near semantics (top‑1 under its context at L31) [001_layers_baseline/run-latest/output-Qwen3-8B-records.csv:607]. The next‑token answer crystallizes at L31 ('Berlin') [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:33].

4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓
- LayerNorm bias removed n.a.
- FP32 unembed promoted n.a.
- Punctuation / markup anchoring noted ✓
- Copy‑reflex ✗
- Preferred lens honored ✓ (tuned for reporting; ranks emphasized)
- Confirmed semantics reported ✓ (L31; raw)
- Dual‑lens artefact metrics (incl. v2, JS/Jaccard/L1) cited ✓
- Tuned‑lens audit done (rotation/temp/positional/head) ✓
- normalization_provenance present (ln_source @ L0/final) ✓
- per‑layer normalizer effect present (resid_norm_ratio, delta_resid_cos) ✓
- deterministic_algorithms true ✓
- numeric_health clean ✓
- copy_mask plausible ✓
- milestones.csv or evaluation_pack.citations used for quotes ✓
- gate_stability_small_scale reported ✓
- position_window stability reported ✓

---
Produced by OpenAI GPT-5 
