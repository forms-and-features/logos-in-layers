# Evaluation Report: Qwen/Qwen2.5-72B

*Run executed on: 2025-10-11 21:50:12*
**EVAL**

**Overview**
- Model: Qwen2.5-72B; run timestamp: 2025-10-11 21:50:12 (see `001_layers_baseline/run-latest/timestamp-20251011-2150`). The probe tracks copy-reflex vs semantic emergence via ranks/KL/cosine/entropy across layers, with raw-vs-norm lens diagnostics and artifact checks.

**Method Sanity‑Check**
- Prompt & indexing: context ends with “called simply” (no trailing space): "Give the city name only, plain text. The capital of Germany is called simply"  (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:4). Positive rows exist: `prompt_id=pos,prompt_variant=orig` (001_layers_baseline/run-latest/output-Qwen2.5-72B-records.csv:2).
- Normalizer provenance: `arch=pre_norm`, `strategy=next_ln1` (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7602–7604). L0 `ln_source=blocks[0].ln1`; final uses `ln_final` (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7607,8327).
- Per-layer normalizer effect: early spikes are flagged (`flags.normalization_spike=true`) and large `resid_norm_ratio/delta_resid_cos` appear pre‑final (e.g., L2 `resid_norm_ratio=8.939...`, `delta_resid_cos=0.5767`), while final has `resid_norm_ratio≈0.172`, `delta_resid_cos≈0.929` (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7626–7627,8328–8329).
- Unembed bias: `present=false`, `l2_norm=0.0` — cosine metrics are bias‑free (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:831–833).
- Environment & determinism: `device=cpu`, `dtype_compute=torch.bfloat16`, `deterministic_algorithms=true`, `seed=316` (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9567–9576). Reproducibility looks good.
- Numeric health: `any_nan=false`, `any_inf=false`, `layers_flagged=[]` (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8825–8829).
- Copy mask: size `6244`, sample of ignored strings includes punctuation like !, ", #, $ (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7252–7252,7234). Plausible for tokenizer filtering.
- Gold alignment: `{ok=true, variant=with_space, pieces=[ĠBerlin]}`; `gold_alignment_rate=1.0` (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8835–8843,8879).
- Repeatability: skipped due to deterministic env (`status=skipped`); no `{max_rank_dev,p95_rank_dev,top1_flip_rate}` reported (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8831–8832,9779–9783).
- Norm trajectory: `shape=spike`, `slope=0.064`, `r2=0.923`, `n_spikes=55` (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9183–9190).
- Measurement guidance: `prefer_ranks=true`, `suppress_abs_probs=true`, `preferred_lens_for_reporting=norm`, `use_confirmed_semantics=false` (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9729–9741).
- Semantic margin: `delta_abs=0.002`, `p_uniform≈6.6e-06`, `margin_ok_at_L_semantic_norm=true` (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7374–7379).
- Micro‑suite: present; `n=5` with `n_missing=3`; medians `L_semantic_norm_median=80` and `L_semantic_margin_ok_norm_median=80` (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9168–9176).

**Quantitative Findings (Layer‑by‑Layer)**
- Preferred lens: norm (per guidance). Confirmed semantics unavailable; report `L_semantic_norm` and gate status.
- Last‑layer calibration: `kl_to_final_bits≈1.09e-4`, `top1_agree=true`, `warn_high_last_layer_kl=false` (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8848–8865).
- Control margin: `first_control_margin_pos=0`, `max_control_margin=0.207` (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9622–9624).

| Layer | Entropy (bits) | Top‑1 token | Notes |
| --- | --- | --- | --- |
| 0 | 17.2143 | 's' | Early high entropy; assorted non‑answer tokens (001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:2). |
| 10 | 16.5008 | '.myapplication' | No copy/semantics flags; KL still high vs final (001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:12). |
| 53 | 16.9266 | '""""' | Cosine to final crosses ≥0.6 by mid‑depth (see cos milestones) (001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:111). |
| 80 | 4.1158 | ' Berlin' | Bold semantic layer — L_semantic_norm=80, gate OK (001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:138; milestones row shows semantic at 80: 001_layers_baseline/run-latest/output-Qwen2.5-72B-milestones.csv:3). |

- Micro‑suite (medians): `L_semantic_norm_median=80`, `Δ̂` median unavailable; fact citation example `Germany→Berlin` uses row_index=80 (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9109–9117).
- Entropy drift: teacher entropy `4.1356` and median gap `entropy_gap_bits_p50=12.496` (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7421,9796–9798).
- Normalizer snapshots: L0 `resid_norm_ratio≈0.716`, `delta_resid_cos≈0.475`; final `resid_norm_ratio≈0.172`, `delta_resid_cos≈0.929` (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7607–7609,8328–8329).

**Qualitative Findings**

4.1. Copy vs semantics (Δ‑gap)
No strict or soft copy milestone is detected; early layers 0–3 have `copy_collapse=False` and `copy_soft_k1@0.5=False` (e.g., L0–3 flags are all False: 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:2–5). `summary.copy_thresholds` shows all strict thresholds τ∈{0.70,0.80,0.90,0.95} are null and `stability=none` (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7327–7345). With no `L_copy_strict` or `L_copy_soft` layer, Δ̂ is undefined (evaluation_pack.milestones shows `delta_hat=null`) and semantics occurs at the final layer (`L_semantic_norm=80`), indicating no early copy reflex in this setup.

4.2. Lens sanity: Raw‑vs‑Norm
Artifact risk is high: `lens_artifact_score_v2=0.7426` (tier=high), with `js_divergence_p50=0.1052`, `l1_prob_diff_p50=0.6151`, and no early convergence (`first_js_le_0.1=0`, `first_l1_le_0.5=0`) (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7596–7599,9755–9771). Top‑K overlap is modest (`jaccard_raw_norm_p50=0.3158`; no layer reaches 0.5) (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7422–7425,9755–9763). Prevalence: `pct_layers_kl_ge_1.0=0.321`, `n_norm_only_semantics_layers=1`, `earliest_norm_only_semantic=80` (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7403–7407,9759–9766). Caution: early semantics could be lens‑induced; rely on rank milestones and the margin gate rather than absolute probabilities.

4.3. Tuned‑Lens analysis
Tuned lens is missing for this model (`tuned_lens.status=missing`), so norm lens remains the reporting lens; no rotation/temperature or positional/head audits to report (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9069–9074).

4.4. KL, ranks, cosine, entropy milestones
KL thresholds (norm): `first_kl_below_1.0=80`, `first_kl_below_0.5=80`, with near‑zero final KL and no calibration warning (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7261–7265,8848–8866). Ranks (norm): `first_rank_le_10=74`, `first_rank_le_5=78`, `first_rank_le_1=80` (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7263–7265). Cosine milestones (norm): `ge_0.6` reached by layer 53 (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7358–7365). Entropy steadily drops toward the teacher (`entropy_gap_bits_p50=12.496`) while late layers provide calibration (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9796–9798). Margin gate passes at L=80, so the rank‑1 milestone is not near‑uniform (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7374–7379).

4.5. Prism (shared‑decoder diagnostic)
Prism present and compatible; rank milestones not reported for prism (null), but KL shows early reductions: ΔKL at sampled depths ≈ +3.16 (p25), +2.83 (p50), −0.54 (p75) bits vs norm (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:840–922). Verdict: Neutral — KL improves early but no confirmed shift in rank milestones.

4.6. Ablation & stress tests
No‑filler ablation: `L_sem_orig=80`, `L_sem_nf=80`, `ΔL_sem=0` (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9588–9600). Control summary present: `first_control_margin_pos=0`, `max_control_margin=0.207` (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9622–9624). Test prompt check: "Berlin is the capital of" appears in the side prompts and yields the expected country token at top‑1 (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:12). Important‑word trajectory: the answer ‘ Berlin’ first becomes top‑1 at the final layer (001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:138), while early layers emit assorted non‑answer tokens (e.g., L0 's' at 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:2).

4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓
- LayerNorm bias removed ✓
- FP32 unembed promoted ✓
- Punctuation / markup anchoring noted ✓
- Copy‑reflex ✗ (not observed)
- Preferred lens honored ✓ (norm)
- Confirmed semantics reported ✗ (not available; margin‑gated norm used)
- Dual‑lens artefact metrics (incl. v2/JS/Jaccard/L1) cited ✓
- Tuned‑lens audit done (rotation/temp/positional/head) ✗ (missing)
- normalization_provenance present (ln_source @ L0/final) ✓
- per‑layer normalizer effect present (resid_norm_ratio, delta_resid_cos) ✓
- deterministic_algorithms true ✓
- numeric_health clean ✓
- copy_mask plausible ✓
- milestones.csv or evaluation_pack.citations used ✓

---
Produced by OpenAI GPT-5

