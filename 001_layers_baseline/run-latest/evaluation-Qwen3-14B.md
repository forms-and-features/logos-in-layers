# Evaluation Report: Qwen/Qwen3-14B

## Overview
Qwen/Qwen3-14B evaluated on capital-fact prompt; run timestamp recorded as `timestamp-20251012-2056`. The probe measures copy vs. semantic onset across layers using ranks/KL/cosine/entropy, with raw‑vs‑norm lens diagnostics and tuned‑lens audits.

## Method sanity‑check
- Prompt & indexing: context ends with “called simply” with no trailing space; positive/original rows present in pure CSV (e.g., `layer=36` is semantic; `output-Qwen3-14B-pure-next-token.csv:38`).
- Normalizer provenance: `strategy="next_ln1"`; L0 ln source `blocks[0].ln1`; final `ln_final` (eps inside sqrt; γ used) (`output-Qwen3-14B.json: .diagnostics.normalization_provenance`). L0: `ln_source="blocks[0].ln1"`; last: `ln_source="ln_final"`.
- Per‑layer normalizer effect: flagged spike trajectory (`diagnostics.flags.normalization_spike=true`; `diagnostics.norm_trajectory.shape="spike"`, `r2=0.9046`, `n_spikes=14`). Representative deltas: L36 `resid_norm_ratio=0.2344`, `delta_resid_cos=0.7329` (`output-Qwen3-14B-pure-next-token.csv:38`).
- Unembed bias: `present=false`, `l2_norm=0.0`; cosines are bias‑free (`output-Qwen3-14B.json: .diagnostics.unembed_bias`).
- Environment & determinism: `device="cpu"`, torch `2.8.0+cu128`, `deterministic_algorithms=true`, `seed=316` (`output-Qwen3-14B.json: .provenance.env`).
- Numeric health: `any_nan=false`, `any_inf=false`, `layers_flagged=[]` (`output-Qwen3-14B.json: .diagnostics.numeric_health`).
- Copy mask: size `6112`, sample IDs `[0,1,2,3,4,5,6,7,8,9]` (tokenizer punctuation/control plausible) (`output-Qwen3-14B.json: .diagnostics.copy_mask`).
- Gold alignment: `{ok: true, variant: with_space, first_id: 19846, pieces: ["ĠBerlin"]}` (`output-Qwen3-14B.json: .diagnostics.gold_alignment`).
- Repeatability: skipped in deterministic env (`output-Qwen3-14B.json: .diagnostics.repeatability = {status:"skipped"}`).
- Norm trajectory: `shape="spike"`, `slope=0.1107`, `r2=0.9046` (`output-Qwen3-14B.json: .diagnostics.norm_trajectory`).
- Measurement guidance: `{prefer_ranks:true, suppress_abs_probs:true, preferred_lens_for_reporting:"tuned", use_confirmed_semantics:true}` with reasons `[high_lens_artifact_risk, high_lens_artifact_score, normalization_spike]` (`output-Qwen3-14B.json: .measurement_guidance`).
- Semantic margin: `{delta_abs:0.002, p_uniform:6.58e-06, margin_ok_at_L_semantic_norm:true}` (`output-Qwen3-14B.json: .summary.semantic_margin`).
- Micro‑suite: aggregates present; `n=5`, `n_missing=0`; medians: `L_semantic_confirmed_median=36`, `L_semantic_norm_median=36` (`output-Qwen3-14B.json: .evaluation_pack.micro_suite.aggregates`).

## Quantitative findings (layer‑by‑layer)

| Layer | Entropy (bits) | Top‑1 token | Answer rank | Citation |
|---|---|---|---:|---|
| 0 | 17.2129 | ‘梳’ | 100736 | `output-Qwen3-14B-pure-next-token.csv:2` |
| 10 | 17.1696 | ‘ (?)’ | 105853 | `output-Qwen3-14B-pure-next-token.csv:12` |
| 20 | 16.9323 | ‘____’ | 60487 | `output-Qwen3-14B-pure-next-token.csv:22` |
| 30 | 7.7892 | ‘这个名字’ | 8818 | `output-Qwen3-14B-pure-next-token.csv:32` |
| 33 | 0.4813 | ‘____’ | 4 | `output-Qwen3-14B-pure-next-token.csv:35` |
| **36** | 0.3122 | ‘ Berlin’ | 1 | `output-Qwen3-14B-pure-next-token.csv:38` |
| 40 | 3.5835 | ‘ Berlin’ | 1 | `output-Qwen3-14B-pure-next-token.csv:42` |

- Semantic layer: Bolded L36 is the confirmed layer (`evaluation_pack.milestones.L_semantic_confirmed.layer=36`, source=raw; `output-Qwen3-14B-milestones.csv:3–4`). Measurement guidance requests confirmed semantics and tuned lens for reporting.
- Control margin: `first_control_margin_pos=0`, `first_control_strong_pos=36`, `max_control_margin=0.97415` (`output-Qwen3-14B.json: .control_summary`).
- Micro‑suite: median `L_semantic_confirmed=36`, median Δ̂ is null; example fact row: Germany→Berlin cited at `output-Qwen3-14B-pure-next-token.csv:38` (`output-Qwen3-14B.json: .evaluation_pack.micro_suite.aggregates` and `.citations.fact_rows`).
- Entropy drift: entropy‑gap medians vs teacher: p25=4.206, p50=13.402, p75=13.586 bits (`output-Qwen3-14B.json: .evaluation_pack.entropy`).
- Confidence margins (logit): at L33, `answer_vs_top1_gap=-5.907` (answer below top‑1) (`output-Qwen3-14B-pure-next-token.csv:35`); at L36, `answer_logit_gap=3.143` (answer is top‑1) (`output-Qwen3-14B-pure-next-token.csv:38`).
- Normalizer snapshots: L36 `resid_norm_ratio=0.2344`, `delta_resid_cos=0.7329`; L40 `resid_norm_ratio=0.0735`, `delta_resid_cos=0.5886` (`output-Qwen3-14B-pure-next-token.csv:38,42`).

## Qualitative findings

### 4.1 Copy vs semantics (Δ‑gap)
No early copy reflex detected: for layers 0–3, `copy_collapse=False` and `copy_soft_k1@0.5=False` (`output-Qwen3-14B-pure-next-token.csv:2–5`). `L_copy_strict` is null at τ∈{0.70,0.95} and `L_copy_soft[k]` is null (`output-Qwen3-14B.json: .evaluation_pack.milestones` and tuned summaries show `L_copy_strict` null with `stability:"none"`). With no copy milestone, Δ̂ is undefined (`depth_fractions.delta_hat=null`). Stability tag unavailable in baseline summary.

### 4.2 Lens sanity: Raw‑vs‑Norm
Artifact risk is high: `lens_artifact_score_v2=0.7037` (tier=high), legacy score `0.5537` (`output-Qwen3-14B.json: .diagnostics.raw_lens_full.score`). Symmetric metrics: `js_divergence_p50=0.5130`, `l1_prob_diff_p50=1.4323`, with no early convergence (`first_js_le_0.1=0`, `first_l1_le_0.5=0`). Top‑K overlap is low: `jaccard_raw_norm_p50=0.25`, `first_jaccard_raw_norm_ge_0.5=0` (`output-Qwen3-14B.json: .diagnostics.raw_lens_full`). Prevalence: `pct_layers_kl_ge_1.0=0.7561`, `n_norm_only_semantics_layers=0`, `earliest_norm_only_semantic=null`. Caution: early semantics could be lens‑induced; prefer rank milestones and confirmed semantics per `measurement_guidance`.

### 4.3 Tuned‑Lens analysis
Preference: `tuned_is_calibration_only=false`; `preferred_semantics_lens_hint="tuned"` (`output-Qwen3-14B.json: .tuned_lens.audit_summary`). Attribution: KL deltas at percentiles show strong rotation effect (`ΔKL_rot p25/p50/p75 ≈ 1.73/1.97/2.06` bits), negligible temperature (`≈ −0.11/−0.00/+0.03`), with interaction `≈ 3.22` bits; total tuned ΔKL (p50) `≈ 5.01` bits (`output-Qwen3-14B.json: .tuned_lens.audit_summary.rotation_vs_temperature`). Positional generalization: `pos_in_dist_le_0.92=5.51`, `pos_ood_ge_0.96=5.01`, gap `−0.49` bits (OOD slightly worse) (`output-Qwen3-14B.json: .tuned_lens.audit_summary.positional`). Head mismatch: `tau_star_modelcal=1.0`; final‑layer KL `0.0` before/after τ⋆ (`output-Qwen3-14B.json: .tuned_lens.audit_summary.head_mismatch`). Rank earliness: tuned `first_rank_le_1=39` vs baseline norm `36` in one summary (later by +3L) (`output-Qwen3-14B.json: tuned summaries around 8210–8320); variability across runs shows 34–39. Last‑layer consistency is clean for baseline (`kl_to_final_bits=0.0`, `top1_agree=true`; `output-Qwen3-14B.json: .diagnostics.last_layer_consistency`).

### 4.4 KL, ranks, cosine, entropy milestones
- KL: `first_kl_below_1.0=40`, `first_kl_below_0.5=40` (final KL≈0 per last‑layer check) (`output-Qwen3-14B.json: raw_lens_check/milestones and .diagnostics.last_layer_consistency`).
- Ranks (preferred lens tuned; baseline in parentheses): `first_rank_le_10=33 (32)`, `le_5=34 (33)`, `le_1=39 (36)` (tuned later) (`output-Qwen3-14B.json: tuned summaries and 7093–7096`). Margin gate: uniform‑margin gate passes at L36 (`margin_ok_at_L_semantic_norm=true`).
- Cosine: baseline cos_to_final crosses >0.4 by L33 (`0.447`; `output-Qwen3-14B-pure-next-token.csv:35`) and >0.6 by L36 (`0.610`; `:38`). Tuned cos milestones (one summary): `ge_0.6=29`, indicating earlier alignment under tuned rotation (`output-Qwen3-14B.json: tuned summaries 8320–8460`).
- Entropy: non‑monotonic—drops sharply to near‑deterministic around L33–36 (e.g., 0.481→0.312 bits; `output-Qwen3-14B-pure-next-token.csv:35,38`), then rises to teacher‑like at L40 (`3.5835` bits; `:42`). Median entropy‑gap vs teacher is large (p50≈13.40 bits) early, consistent with calibration occurring late (`output-Qwen3-14B.json: .evaluation_pack.entropy`).

### 4.5 Prism (shared‑decoder diagnostic)
Present and compatible (`diagnostics.prism_summary.present=true`). KL deltas at probed depths are negative (prism increases KL): p25 `−0.257`, p50 `−0.250`, p75 `−0.709` bits; no rank‑milestone improvement (`delta.le_* = null`) (`output-Qwen3-14B.json: .diagnostics.prism_summary.metrics`). Verdict: Regressive.

### 4.6 Ablation & stress tests
No‑filler ablation: `L_sem_orig=36`, `L_sem_nf=36` (ΔL_sem=0; insensitive to filler) (`output-Qwen3-14B.json: .ablation_summary`). Control: `first_control_strong_pos=36`; test prompt “Berlin is the capital of …” shows the expected country token among top candidates (`output-Qwen3-14B.json: .test_prompts[0]`). Important‑word trajectory: control margin strengthens by L36 (`output-Qwen3-14B.json: .control_summary`).

### 4.7 Checklist (✓/✗/n.a.)
- RMS lens ✓ (`RMSNorm`; eps inside sqrt; γ used) (`output-Qwen3-14B.json: .diagnostics.*`)
- LayerNorm bias removed ✓ (`layernorm_bias_fix:"not_needed_rms_model"`)
- FP32 unembed promoted ✓ (`unembed_dtype:"torch.float32"`; `mixed_precision_fix` present)
- Punctuation / markup anchoring noted ✓ (underscore/quote tokens dominate mid‑layers; e.g., `top1='____'` at L20/L33; `output-Qwen3-14B-pure-next-token.csv:22,35`)
- Copy‑reflex ✗ (no early `copy_*` flags in L0–3; `output-Qwen3-14B-pure-next-token.csv:2–5`)
- Preferred lens honored ✓ (tuned for rank reporting; confirmed semantics used)
- Confirmed semantics reported ✓ (`L_semantic_confirmed=36`; `output-Qwen3-14B-milestones.csv:3–4`)
- Dual‑lens artefact metrics (incl. v2/JS/Jaccard/L1) cited ✓ (`output-Qwen3-14B.json: .diagnostics.raw_lens_full`)
- Tuned‑lens audit (rotation/temp/positional/head) ✓ (`output-Qwen3-14B.json: .tuned_lens.audit_summary`)
- normalization_provenance present ✓ (`ln_source@L0/final`; `output-Qwen3-14B.json: .diagnostics.normalization_provenance`)
- per‑layer normalizer effect present ✓ (`resid_norm_ratio`, `delta_resid_cos`; `output-Qwen3-14B-pure-next-token.csv:38`)
- deterministic_algorithms true ✓ (`output-Qwen3-14B.json: .provenance.env`)
- numeric_health clean ✓ (`output-Qwen3-14B.json: .diagnostics.numeric_health`)
- copy_mask plausible ✓ (`size=6112`; `output-Qwen3-14B.json: .diagnostics.copy_mask`)
- milestones.csv or evaluation_pack.citations used ✓ (`output-Qwen3-14B-milestones.csv:3–4`; `output-Qwen3-14B.json: .evaluation_pack.citations`)

---
Produced by OpenAI GPT-5 
*Run executed on: 2025-10-12 20:56:18*
