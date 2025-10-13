# Evaluation Report: meta-llama/Meta-Llama-3-8B

# EVAL

**Overview**
- Meta-Llama-3-8B run on 2025-10-12 20:56; outputs located under `001_layers_baseline/run-latest` (see `timestamp-20251012-2056`). The probe evaluates copy vs. semantic onset across layers using norm/raw/tuned lenses with KL, ranks, cosine, and entropy trajectories.
- Diagnostics include last-layer consistency, normalization provenance and effects, artifact risk (raw vs norm), repeatability, gold alignment, and measurement guidance (`001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json`).

**Method Sanity‑Check**
- Prompt & indexing: context prompt ends with “called simply” and no trailing space: "Give the city name only, plain text. The capital of Germany is called simply" (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:4). Positive rows exist with `prompt_id=pos`, `prompt_variant=orig` (e.g., layer 0 row 2 in `001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:2`).
- Normalizer provenance: `arch=pre_norm`, `strategy=next_ln1` (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7264–7266). Sources: L0 `blocks[0].ln1` (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7269) and final `ln_final` at layer 32 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7550–7552).
- Per‑layer normalizer effect: early spike flagged and visible in `resid_norm_ratio`/`delta_resid_cos` prior to semantics, e.g., L0 `resid_norm_ratio=18.19`, `delta_resid_cos=0.53` (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7272–7274); flag `normalization_spike=true` (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:842).
- Unembed bias: `present=false`, `l2_norm=0.0` — cosines are bias‑free (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:834–838).
- Environment & determinism: device `cpu`, `dtype_compute=torch.float32`, `deterministic_algorithms=true`, `seed=316` (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:9969–9977).
- Numeric health: `any_nan=false`, `any_inf=false`, `layers_flagged=[]` (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7770–7776). No overlap with candidate layers.
- Copy mask: `ignored_token_ids` list present (sample: 0,1,2,3,4,5…) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:942–951). Size not explicitly provided.
- Gold alignment: `ok`, `variant=with_space`, pieces `["ĠBerlin"]` (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7779–7787).
- Repeatability: status `skipped` due to deterministic env (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7773–7777). No `{max_rank_dev, p95_rank_dev, top1_flip_rate}` reported.
- Norm trajectory: `shape="spike"`, `slope=0.113`, `r2=0.953`, `n_spikes=18` (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11872–11876).
- Measurement guidance: `prefer_ranks=true`, `suppress_abs_probs=true`, reasons include `norm_only_semantics_window`, `high_lens_artifact_risk`, `normalization_spike`, `rank_only_near_uniform`; preferred lens for reporting `tuned`; `use_confirmed_semantics=true` (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11811–11824).
- Semantic margin: `delta_abs=0.002`, `p_uniform≈7.8e-06`; `margin_ok_at_L_semantic_norm=false`; `L_semantic_margin_ok_norm=32` (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11702–11708).
- Micro‑suite: aggregates present (`n=5`, `n_missing=0`); medians `L_semantic_confirmed=25`, `L_semantic_margin_ok_norm=32` (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11700–11709, 11798–11801).

**Quantitative Findings (Layer‑by‑Layer)**
- L 0 — entropy 16.96 bits; top‑1 token ‘itzer’ (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:2).
- L 10 — entropy 16.85 bits; top‑1 token ‘tons’ (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:14).
- L 20 — entropy 16.83 bits; top‑1 token "'" (apostrophe) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:24).
- L 25 — entropy 16.81 bits; top‑1 token ‘Berlin’ (confirmed) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:29).
- L 32 — entropy 2.96 bits; top‑1 token ‘Berlin’ (final head agrees) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:36).

- Semantic layer: bolded L 25 (confirmed semantics; preferred lens for reporting is tuned, but confirmation source here is raw). Note uniform‑margin gate fails at `L_semantic_norm` (weak); margin passes only at L 32 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11820–11824, 11702–11708).
- Control margin: `first_control_margin_pos=0`, `max_control_margin=0.5186` (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:10025–10030).
- Micro‑suite: median `L_semantic_confirmed=25`; Δ̂ median not reported. Example fact citation: Germany→Berlin at row index 25 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11920–11928).
- Entropy drift: `entropy_gap_bits_p25/p50/p75 ≈ 13.87/13.88/13.91` bits (relative to teacher entropy 2.961) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11878–11882, 7179).
- Confidence margins/normalizer effect: at L 25, `answer_logit_gap≈0.4075` (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:29); resid norm ratio drops from 18.19 (L0) to 1.50 (L25) to 2.25 (final) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7272–7274, 7548–7554).

**Qualitative Findings**

4.1. Copy vs semantics (Δ‑gap)
- No strict or soft copy milestone detected across thresholds; `L_copy_strict` is null for τ∈{0.70,0.80,0.90,0.95} and `copy_soft k∈{1,2,3}` are null; stability=`none` (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7054–7078). Early layers 0–3 show no `copy_collapse` or `copy_soft_k1@0.5` (rows 2–5 in `001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv`). With no copy milestone, Δ̂ is n.a.; semantics occurs at ~78% depth (`L_semantic_frac=0.781`) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7099–7101).

4.2. Lens sanity: Raw‑vs‑Norm
- Artifact risk is medium: `lens_artifact_score_v2=0.4590` (tier=medium) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7257–7261). Symmetric metrics: `js_divergence_p50=0.0168`, `l1_prob_diff_p50=0.2403`, with no early convergence (`first_js_le_0.1=0`, `first_l1_le_0.5=0`) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7167–7178). Top‑K overlap: `jaccard_raw_norm_p50=0.408`, `first_jaccard_raw_norm_ge_0.5=3` (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7180–7188).
- Prevalence: `pct_layers_kl_ge_1.0=0.03`, `n_norm_only_semantics_layers=5`, `earliest_norm_only_semantic=25` (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7160–7165). Caution: early semantics near L 25 may be partially lens‑induced; prefer rank milestones and confirmed semantics.

4.3. Tuned‑Lens analysis
- Preference: not calibration‑only (`tuned_is_calibration_only=false`); guidance prefers tuned lens for reporting (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11672–11678, 11821–11824).
- Attribution: ΔKL decomposition shows rotation dominates: `delta_kl_rot_p25/p50/p75 ≈ 2.08/2.56/2.86`; temperature effects small (p50 ≈ −0.04); interaction p50 ≈ 2.34 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11649–11661).
- Rank earliness: tuned ranks are later (first_rank_le_{10,5,1}=32 vs. baseline 24/25/25) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11518–11530).
- Positional generalization: `pos_in_dist_le_0.92=5.64`, `pos_ood_ge_0.96=4.87`, `pos_ood_gap≈−0.77` (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11659–11670).
- Head mismatch: clean final‑layer agreement after model‑cal (τ*): `kl_bits_tuned_final=0.0`, `tau_star_modelcal=1.0` (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11672–11675). Baseline last‑layer consistency also clean: `kl_to_final_bits=0.0`, `top1_agree=true` (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7789–7807).

4.4. KL, ranks, cosine, entropy milestones
- KL: `first_kl_le_1.0=32` both baseline and tuned; final KL≈0 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11540–11547, 7790–7797). Do not infer final‑row probability regressions.
- Ranks: preferred lens (tuned) reaches rank‑1 only at L 32; baseline reaches rank‑1 at L 25 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11512–11520). Margin gate fails at `L_semantic_norm` (weak), passes at L 32 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11702–11708).
- Cosine: norm‑lens milestones at ge_0.2=20, ge_0.4=30, ge_0.6=32 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7092–7096).
- Entropy: strong downward drift from ~16.9 bits early to ~3.0 bits at L 32; summary gaps p25/p50/p75 ≈ 13.87/13.88/13.91 bits relative to teacher entropy 2.961 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11878–11882, 7179).

4.5. Prism (shared‑decoder diagnostic)
- Present and compatible; sampled layers at `embed, 7, 15, 23` (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:844–855). KL deltas are negative (KL increases vs baseline): p25 −5.37, p50 −8.29, p75 −9.78; no earlier rank milestones (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:858–882). Verdict: Regressive.

4.6. Ablation & stress tests
- Ablation summary: `L_sem_orig=25`, `L_sem_nf=25`; `ΔL_sem=0` (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:9989–9998). No stylistic sensitivity between `orig` and `no_filler`.
- Controls: `first_control_margin_pos=0`, strong by L 25; `max_control_margin=0.5186`, `max_control_top2_logit_gap≈0.912` (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:10025–10032).
- Important‑word trajectory: near‑answer tokens appear before confirmation. Example: layer 24, position 14 token “is” has top‑1 ‘Berlin’ among next‑token candidates (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-records.csv:499). Test prompt check: "Berlin is the capital of" → top‑1 “ Germany” (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:12–18).

4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓
- LayerNorm bias removed ✓
- FP32 unembed promoted ✓
- Punctuation / markup anchoring noted ✓
- Copy‑reflex ✗
- Preferred lens honored ✓
- Confirmed semantics reported ✓
- Dual‑lens artefact metrics (incl. `lens_artifact_score_v2`, JS/Jaccard/L1) cited ✓
- Tuned‑lens audit done (rotation/temp/positional/head) ✓
- normalization_provenance present (ln_source @ L0/final) ✓
- per‑layer normalizer effect present (resid_norm_ratio, delta_resid_cos) ✓
- deterministic_algorithms true (or caution noted) ✓
- numeric_health clean ✓
- copy_mask plausible ✓
- milestones.csv or evaluation_pack.citations used for quotes ✓

---
Produced by OpenAI GPT-5 
*Run executed on: 2025-10-12 20:56:18*
