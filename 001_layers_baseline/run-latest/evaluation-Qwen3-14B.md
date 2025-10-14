# Evaluation Report: Qwen/Qwen3-14B

*Run executed on: 2025-10-13 22:23:35*
**EVAL**

**Overview**
- Qwen/Qwen3-14B (40 layers) evaluated 2025-10-14; probe measures copy vs. semantics onset, with KL/rank/cosine/entropy trajectories and dual‑lens diagnostics (norm, tuned) plus raw‑vs‑norm artefact checks. See `001_layers_baseline/run-latest/output-Qwen3-14B.json:12238` and `001_layers_baseline/run-latest/output-Qwen3-14B.json:12132`.
- Context prompt ends with “called simply” and the semantic target is the first unseen token after that context; rows exist for `prompt_id=pos`, `prompt_variant=orig` (e.g., `001_layers_baseline/run-latest/output-Qwen3-14B-records.csv:65`).

**Method Sanity‑Check**
- Prompt & indexing: “Give the city name only, plain text. The capital of Germany is called simply” (`001_layers_baseline/run-latest/output-Qwen3-14B.json:812`); positive row at the next‑token position exists (`001_layers_baseline/run-latest/output-Qwen3-14B-records.csv:65`).
- Normalizer provenance: pre‑norm with strategy “next_ln1”; L0 `ln_source=blocks[0].ln1`, final `ln_source=ln_final` (epsilon inside sqrt; γ used) (`001_layers_baseline/run-latest/output-Qwen3-14B.json:7389`, `001_layers_baseline/run-latest/output-Qwen3-14B.json:7779`).
- Per‑layer normalizer effect: early layers show no abnormal spikes before semantics; e.g., L1 `resid_norm_ratio≈0.0835`, `delta_resid_cos≈0.663` and smooth growth through mid‑layers (`001_layers_baseline/run-latest/output-Qwen3-14B.json:7400`).
- Unembed bias: absent (`present=false`, `l2_norm=0.0`) and cosines are bias‑free (`001_layers_baseline/run-latest/output-Qwen3-14B.json:872`).
- Environment & determinism: `device=cpu`, `torch_version=2.8.0+cu128`, `deterministic_algorithms=true`, `seed=316` (`001_layers_baseline/run-latest/output-Qwen3-14B.json:10306`).
- Numeric health: `any_nan=false`, `any_inf=false`, `layers_flagged=[]` (`001_layers_baseline/run-latest/output-Qwen3-14B.json:8012`).
- Copy mask: size `6112`; sample includes punctuation and markup tokens appropriate for Qwen3 tokenizer (`001_layers_baseline/run-latest/output-Qwen3-14B.json:7068`).
- Gold alignment: aligned to “ĠBerlin” with `gold_alignment_rate=1.0` (`001_layers_baseline/run-latest/output-Qwen3-14B.json:8097`, `001_layers_baseline/run-latest/output-Qwen3-14B.json:8146`).
- Repeatability (1.39): skipped due to deterministic environment (`status=skipped`) (`001_layers_baseline/run-latest/output-Qwen3-14B.json:8019`).
- Norm trajectory: shape “spike” with positive slope; `r2≈0.90` (`001_layers_baseline/run-latest/output-Qwen3-14B.json:12184`).
- Measurement guidance: `prefer_ranks=true`, `suppress_abs_probs=true`, `preferred_lens_for_reporting=tuned`, `use_confirmed_semantics=true` (reasons include “high_lens_artifact_risk”) (`001_layers_baseline/run-latest/output-Qwen3-14B.json:12152`).
- Semantic margin: `delta_abs=0.002`, `p_uniform≈6.6e‑6`, `margin_ok_at_L_semantic_norm=true`; confirmed margin OK at L=36 (`001_layers_baseline/run-latest/output-Qwen3-14B.json:12048`).
- Gate‑stability (small‑scale): at `L_semantic_norm=36`, both gates pass 1.00 of small rescalings; `min_both_gates_pass_frac=1.0` → not calibration‑sensitive (`001_layers_baseline/run-latest/output-Qwen3-14B.json:7228`).
- Position‑window: `rank1_frac=0.0` across six positions at `L_semantic_norm=36` → position‑fragile (`001_layers_baseline/run-latest/output-Qwen3-14B.json:12082`).
- Micro‑suite: aggregates present (`n=5`), `L_semantic_confirmed_median=36`, no missing facts (`n_missing=0`) (`001_layers_baseline/run-latest/output-Qwen3-14B.json:12067`).

**Quantitative Findings (Layer‑by‑Layer)**
- L 0 — entropy 17.21 bits, top‑1 token non‑semantic; `fact_key=Germany→Berlin`, `pos=15` (`001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:2`).
- L 5 — entropy 17.21 bits, top‑1 non‑semantic (`001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:7`).
- L 10 — entropy high, non‑semantic top‑1 (representative mid‑depth; see KL@50% at L20 in diagnostics) (`001_layers_baseline/run-latest/output-Qwen3-14B.json:8136`).
- L 29 — entropy 14.19 bits, surface tokens dominate; `topk_jaccard_raw_norm@50≈0.41` around this region (`001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:31`, `001_layers_baseline/run-latest/output-Qwen3-14B.json:7316`).
- L 33 — low entropy window with formatting tokens leading; answer appears in top‑k but not rank‑1 (`001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:35`).
- L 36 — entropy 0.31 bits, top‑1 ‘Berlin’; margin gate passes; confirmed semantics layer. Bolded: L 36 (`001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:38`, `001_layers_baseline/run-latest/output-Qwen3-14B.json:12048`).
- Controls: `first_control_margin_pos=0`, `max_control_margin≈0.97`, `first_control_strong_pos=36` (`001_layers_baseline/run-latest/output-Qwen3-14B.json:10363`).
- Micro‑suite: median `L_semantic_confirmed=36` (IQR 35–36); example fact citation: Germany→Berlin row 36 (`001_layers_baseline/run-latest/output-Qwen3-14B.json:12072`, `001_layers_baseline/run-latest/output-Qwen3-14B.json:12108`).
- Entropy drift: median entropy gap vs teacher large (p50≈13.40 bits), consistent with early over‑entropy then late calibration (`001_layers_baseline/run-latest/output-Qwen3-14B.json:12236`).
- Normalizer snapshots: early `resid_norm_ratio` small and rising; at L36 `resid_norm_ratio≈0.234`, `delta_resid_cos≈0.733` (`001_layers_baseline/run-latest/output-Qwen3-14B.json:7720`).

**Qualitative Findings**

4.1. Copy vs semantics (Δ‑gap)
Copy‑reflex ✗. No early layers 0–3 show `copy_collapse=True` or soft‑copy flags in the pure next‑token view (`001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:2`–`5`). Strict copy is null at τ∈{0.70,0.95}; stability “none” and no norm‑only flags (`001_layers_baseline/run-latest/output-Qwen3-14B.json:7148`). Reported Δ̂ uses semantic fraction: `semantic_frac=0.9` → Δ̂≈0.9 since copy onset is null (`001_layers_baseline/run-latest/output-Qwen3-14B.json:7200`). Summary copy thresholds stable: none observed across τ (`001_layers_baseline/run-latest/output-Qwen3-14B.json:7148`).

4.2. Lens sanity: Raw‑vs‑Norm
Artefact risk high: `lens_artifact_score_v2=0.7037` (tier=high). Symmetric metrics: `js_divergence_p50=0.513`, `l1_prob_diff_p50=1.432`, with no early safe layers (`first_js_le_0.1=0`, `first_l1_le_0.5=0`). Top‑K overlap is low (`jaccard_raw_norm_p50=0.25`, `first_jaccard_raw_norm_ge_0.5=0`). Prevalence: `pct_layers_kl_ge_1.0≈0.756`, no norm‑only semantic layers near candidates (`001_layers_baseline/run-latest/output-Qwen3-14B.json:12172`). At the semantic target, lens‑consistency is modest (norm vs raw at L36 `jaccard@10≈0.33`, `spearman_top50≈0.34`) and improves norm vs tuned (p50 `spearman_top50≈0.61`) (`001_layers_baseline/run-latest/output-Qwen3-14B.json:8040`). Given high artefact tier, prefer rank/KL milestones and confirmed semantics over absolute probabilities per guidance (`001_layers_baseline/run-latest/output-Qwen3-14B.json:12152`).

4.3. Tuned‑Lens analysis
Preference: tuned lens is not calibration‑only (`tuned_is_calibration_only=false`) and is the preferred semantics lens (`preferred_semantics_lens_hint=tuned`) (`001_layers_baseline/run-latest/output-Qwen3-14B.json:11898`, `001_layers_baseline/run-latest/output-Qwen3-14B.json:11880`). Attribution (ΔKL, bits): rotation dominates temperature (e.g., `ΔKL_rot_p50≈1.97`, `ΔKL_temp_p50≈−0.00`, interaction `≈3.22`; tuned total `≈5.01`) (`001_layers_baseline/run-latest/output-Qwen3-14B.json:11864`). Rank earliness (tuned vs norm): baseline `first_rank_le_1=36`; tuned summaries vary by prompt form with onsets {34–39} and no clear systematic earlier onset across all variants (`001_layers_baseline/run-latest/output-Qwen3-14B.json:7095`, `001_layers_baseline/run-latest/output-Qwen3-14B.json:10384`). Positional generalization: in tuned audit, OOD vs in‑dist gap `pos_ood_gap≈−0.49` (weaker OOD) across grid {0.2…0.98} (`001_layers_baseline/run-latest/output-Qwen3-14B.json:11874`). Head mismatch: tuned final‑layer KL `0.0` and `tau_star_modelcal=1.0` indicate strong last‑layer agreement (`001_layers_baseline/run-latest/output-Qwen3-14B.json:11874`). Last‑layer consistency: `top1_agree=true`, `kl_to_final_bits=0.0` (`001_layers_baseline/run-latest/output-Qwen3-14B.json:8109`).

4.4. KL, ranks, cosine, entropy milestones
KL: `first_kl_below_1.0=40`, `first_kl_below_0.5=40` (late calibration; final KL≈0), consistent with strong final‑head calibration (`001_layers_baseline/run-latest/output-Qwen3-14B.json:7088`). Ranks (preferred lens tuned; baseline in parentheses): tuned `first_rank_le_10∈{32–33}`, `le_5∈{32–36}`, `le_1∈{34–39}` vs norm `(32,33,36)` depending on variant (`001_layers_baseline/run-latest/output-Qwen3-14B.json:10384`, `001_layers_baseline/run-latest/output-Qwen3-14B.json:7095`). Cosine: norm milestones `ge_0.2=5`, `ge_0.4=29`, `ge_0.6=36` (`001_layers_baseline/run-latest/output-Qwen3-14B.json:7190`). Entropy: gaps grow through depth (p50≈13.40 bits), then collapse near semantics and final calibration (`001_layers_baseline/run-latest/output-Qwen3-14B.json:12236`). Margin gate: uniform‑margin passes at L36; top‑2 gap is not invoked (null), so treat rank‑1 as strong by uniform but not by top‑2. Stability: bolded semantics is single‑layer onset; `L_semantic_run2=39` exists but differs from confirmed layer; position‑fragile (`rank1_frac=0.0`) and artefact tier high ⇒ emphasize ranks/KL over probabilities (`001_layers_baseline/run-latest/output-Qwen3-14B.json:7208`, `001_layers_baseline/run-latest/output-Qwen3-14B.json:12082`).

4.5. Prism (shared‑decoder diagnostic)
Present and compatible; deltas indicate mild regressions in KL (e.g., p50 increases by ≈0.25 bits; p75 by ≈0.71) and no earlier rank milestones; verdict: Regressive (`001_layers_baseline/run-latest/output-Qwen3-14B.json:844`).

4.6. Ablation & stress tests
No‑filler ablation: `L_sem_orig=36`, `L_sem_nf=36`, `ΔL_sem=0` → no stylistic sensitivity at 10%‑of‑depth granularity (`001_layers_baseline/run-latest/output-Qwen3-14B.json:10329`). Negative/control prompt exists; control summary shows immediate control margin evidence and strong control near semantics (`first_control_margin_pos=0`, `first_control_strong_pos=36`) (`001_layers_baseline/run-latest/output-Qwen3-14B.json:10363`).

4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓
- LayerNorm bias removed ✓
- FP32 unembed promoted ✓
- Punctuation / markup anchoring noted ✓
- Copy‑reflex ✗
- Preferred lens honored ✓
- Confirmed semantics reported ✓
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
