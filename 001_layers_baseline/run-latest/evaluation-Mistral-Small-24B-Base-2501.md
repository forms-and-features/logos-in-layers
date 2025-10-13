# Evaluation Report: mistralai/Mistral-Small-24B-Base-2501

*Run executed on: 2025-10-12 20:56:18*
## EVAL

**1. Overview**
- Model: `Mistral-Small-24B-Base-2501` (40 layers); run timestamp `timestamp-20251012-2056` exists in `001_layers_baseline/run-latest`.
- The probe tracks copy vs. semantic onset with rank/KL/cosine/entropy trajectories and dual‑lens diagnostics (raw vs. normalized; tuned‑lens audits). See `001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json`.

**2. Method sanity-check**
- Prompt & indexing: context ends with “called simply” with no trailing space: "Give the city name only, plain text. The capital of Germany is called simply" (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4). Positive/original rows present in pure CSV (e.g., layer 0 row 2: `prompt_id=pos`, `prompt_variant=orig`) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:2).
- Normalizer provenance: `arch=pre_norm` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4195), `strategy=next_ln1` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4196); L0 `ln_source=blocks[0].ln1` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4200); final uses `ln_final` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4560).
- Per‑layer normalizer effect: early `resid_norm_ratio`/`delta_resid_cos` recorded; no norm‑only semantic layers near candidate semantics (`n_norm_only_semantics_layers=0`) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4078). Example L3 `resid_norm_ratio=62.67` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4230).
- Unembed bias: `present=false`, `l2_norm=0.0` — cosine metrics are bias‑free (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:835).
- Environment & determinism: `device=cpu`, `torch=2.8.0+cu128`, `deterministic_algorithms=true`, `seed=316` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:7025; 7027). Repeatability skipped due to deterministic env (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4825).
- Numeric health: `any_nan=false`, `any_inf=false`, `layers_flagged=[]` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4818).
- Copy mask: `ignored_token_ids` list present; sample starts `[1009, 1010, 1011, …]` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:952). Token‑class mix (punctuation/markup/space‑like) looks plausible for tokenizer.
- Gold alignment: `ok=true`, `variant=with_space` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4829); `pieces=["ĠBerlin"]` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4840).
- Repeatability (1.39): status "skipped" (`deterministic_env`) — metrics not populated (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4825). Flag: n/a.
- Norm trajectory: `shape="spike"`, `slope≈0.105`, `r2≈0.915`, `n_spikes=34` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:8929).
- Measurement guidance: `{prefer_ranks=true, suppress_abs_probs=true, preferred_lens_for_reporting="tuned", preferred_semantics_lens_hint="tuned", use_confirmed_semantics=true}`; reasons `["normalization_spike","rank_only_near_uniform"]` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:8870; 8877).
- Semantic margin: `{delta_abs=0.002, p_uniform=7.63e-06, L_semantic_margin_ok_norm=35, margin_ok_at_L_semantic_norm=false}` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4031). Gate fails at L_semantic_norm → treat as weak.
- Micro‑suite: aggregates `{n=5, L_semantic_confirmed_median=33, n_missing=0}` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:8782; 8784); fact citations provided (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:9042).

**3. Quantitative findings (layer‑by‑layer)**
- Table (positive/original prompt only; preferred lens for ranks = tuned; semantics bolded by confirmed layer):
  - L 0 — entropy 16.9985 bits; top‑1 ‘Forbes’; answer rank 21319 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:2)
  - L 30 — entropy 16.7426 bits; top‑1 ‘-на’; answer rank 3 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:32)
  - L 31 — entropy 16.7931 bits; top‑1 ‘-на’; answer rank 2 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:33)
  - L 33 — entropy 16.7740 bits; top‑1 ‘Berlin’; answer rank 1 — bold semantic layer (confirmed) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:35)
  - L 39 — entropy 16.0050 bits; top‑1 ‘Berlin’; answer rank 1 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:41)
- Control margin: `first_control_margin_pos=1`, `max_control_margin=0.4679627` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:7076).
- Micro‑suite: median `L_semantic_confirmed=33` across 5 facts; example citation `Germany→Berlin` row 33 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:8784; 9042).
- Entropy drift: teacher entropy `3.1807` bits; entropy gap percentiles p25/p50/p75 ≈ 13.58/13.59/13.66 bits (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4094; 8936).

**4. Qualitative findings**

4.1. Copy vs semantics (Δ‑gap)
The run shows no early copy‑reflex: strict copy at τ∈{0.70,0.80,0.90,0.95} is null; soft copy k∈{1,2,3} at τ_soft=0.5 also null near layers 0–3 (summary.copy_thresholds shows `L_copy_strict=null` with stability "none", 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:3971; pure CSV L0–L3 have `copy_collapse=False` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:2)). With `L_semantic_confirmed=33` and no detected copy layer, Δ̂ cannot be computed (`delta_hat=null`). Stability tag: "none"; earliest strict copy at τ=0.70 and τ=0.95: both null; no `norm_only_flags` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:3990).

4.2. Lens sanity: Raw‑vs‑Norm
Artifact metrics indicate low risk: `lens_artifact_score_v2=0.1847`, tier "low" (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4190). Robustness metrics: `js_divergence_p50=0.0353`, `l1_prob_diff_p50=0.3473`, with `first_js_le_0.1=0` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4092) and `first_l1_le_0.5=0` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4093). Top‑K overlap is moderate (`jaccard_raw_norm_p50=0.5385`; first ≥0.5 by layer 3) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4097). Prevalence is low (`pct_layers_kl_ge_1.0=0.024`, `n_norm_only_semantics_layers=0`) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4076; 4078). Net: prefer rank milestones; early semantics are unlikely to be lens‑induced here.

4.3. Tuned‑Lens analysis
Preference is tuned for reporting (`preferred_semantics_lens_hint="tuned"`, `tuned_is_calibration_only=false`) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:8736; 8878). Attribution: rotation reduces KL substantially (ΔKL_rot p50 ≈ 1.76 bits) while temperature alone slightly increases KL (ΔKL_temp p50 ≈ −0.33 bits); interaction term sizeable (ΔKL_interaction_p50 ≈ 3.62 bits) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:8942; 8945; 8947). Rank earliness shifts later under tuned: `first_rank_le_10=34`, `le_5=35`, `le_1=39` vs. baseline `le_10=30`, `le_5=30`, `le_1=33` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:8666). Positional generalization is stable (`pos_in_dist_le_0.92=4.43`, `pos_ood_ge_0.96=5.08`, gap ≈ 0.66) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:8959). Head mismatch clean: `kl_bits_tuned_final=0.0`, `tau_star_modelcal=1.0` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:8964). Last‑layer agreement holds: `kl_to_final_bits=0.0`, `top1_agree=true` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4841).

4.4. KL, ranks, cosine, entropy milestones
KL crosses below 1.0 bit only at the final layer for both lenses (`first_kl_le_1.0=40`) and does not indicate earlier head calibration issues (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:8699). Preferred ranks (tuned) reach thresholds late (`le_10=34`, `le_5=35`, `le_1=39`; baseline parenthetical: `30/30/33`) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:8666). Cosine to final under norm lens passes `ge_0.2` at L35 and reaches `ge_0.4/0.6` by L40 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4011). Entropy decreases modestly late; drift vs. teacher entropy remains large (entropy gap p50 ≈ 13.59 bits) even as ranks improve (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:8936). Margin gate reminder: at `L_semantic_norm=33`, the uniform‑margin gate fails (`margin_ok_at_L_semantic_norm=false`) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4033).

4.5. Prism (shared‑decoder diagnostic)
Prism is present/compatible (`k=512`) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:849). Relative to baseline norm, KL delta p50 is ≈ −5.98 bits (prism worse) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:888). Verdict: Regressive — KL increases and no earlier rank milestone (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:892).

4.6. Ablation & stress tests
No‑filler ablation shifts semantics earlier by 2 layers (`L_sem_orig=33`, `L_sem_nf=31`, `ΔL_sem=−2`) — below 10% of depth (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:7041). Control prompt summary exists (`first_control_margin_pos=1`, `max_control_margin=0.468`) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:7076). The negative test “Berlin is the capital of” is included (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:12); correct country appears in top‑k for that prompt (avoid abs‑p per guidance).

4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓ (epsilon inside sqrt; pre‑norm/next_ln1) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4195)
- LayerNorm bias removed ✓ (`unembed_bias.present=false`) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:835)
- FP32 unembed promoted ✓ (`unembed_dtype=torch.float32`) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:811)
- Punctuation / markup anchoring noted ✓ (early top‑1 includes quotes/punct) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:9)
- Copy‑reflex ✗ (no strict/soft copy detected) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:3971; 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:2)
- Preferred lens honored ✓ (tuned for ranks; baseline parenthetical) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:8877; 8666)
- Confirmed semantics reported ✓ (`L_semantic_confirmed=33`) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:8894)
- Dual‑lens artefact metrics cited ✓ (lens_artifact_score_v2, JS/L1/Jaccard) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4190; 4084; 4097)
- Tuned‑lens audit done ✓ (rotation/temp/positional/head) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:8942)
- normalization_provenance present ✓ (ln_source @ L0/final) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4200)
- per‑layer normalizer effect present ✓ (`resid_norm_ratio`, `delta_resid_cos`) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4230)
- deterministic_algorithms true ✓ (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:7025)
- numeric_health clean ✓ (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4818)
- copy_mask plausible ✓ (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:952)
- milestones.csv / citations used ✓ (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:8735; 9042)

---
Produced by OpenAI GPT-5
