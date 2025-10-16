# Evaluation Report: mistralai/Mistral-Small-24B-Base-2501

*Run executed on: 2025-10-13 22:23:35*

## EVAL

**Overview**
- Model: mistralai/Mistral-Small-24B-Base-2501 (40 layers). Run timestamp indicates 2025-10-13 22:23 (see `001_layers_baseline/run-latest/timestamp-20251013-2223`).
- Probe quantifies copy vs. semantics via layerwise logit-lens (norm and tuned), tracking KL-to-final, rank milestones, cosine geometry, and entropy, with dual-lens artifact diagnostics and position-window stability.

**Method Sanity‑Check**
- Prompt & indexing: context ends with “called simply” and no trailing space: "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply" (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4). Positive rows with `prompt_id=pos`, `prompt_variant=orig` exist, e.g. L0 and L33: rows 2 and 35 in `output-Mistral-Small-24B-Base-2501-pure-next-token.csv`.
- Normalizer provenance: "strategy": "next_ln1" (pre‑norm) with early source `blocks[0].ln1` and last `blocks[39].ln1` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4215, 4219, 4570).
- Per‑layer normalizer effect: early spikes present and flagged — e.g., L0 `resid_norm_ratio`: 43.8333, `delta_resid_cos`: 0.4922 (…json:4222–4224); global flag `"normalization_spike": true` (…json:844).
- Unembed bias: absent; "present": false, "l2_norm": 0.0 (…json:834–836). Cosines are bias‑free by construction.
- Environment & determinism: device=cpu, torch=2.8.0+cu128, dtype=torch.float32, `deterministic_algorithms: true`, seed=316 (…json:7133–7141). Reproducibility OK.
- Numeric health: `any_nan=false`, `any_inf=false`, `layers_flagged=[]` (…json:4836–4842).
- Copy mask: size=2931; sample includes control/whitespace/punctuation like "\t", "\n", " ", "!", '"', "#" (…json:3888–3912).
- Gold alignment: `ok=true`, variant `with_space`, first_id=10874 (ĠBerlin) (…json:4921–4931). `gold_alignment_rate=1.0` (…json:9054).
- Repeatability (1.39): skipped due to deterministic env (`status: "skipped"`) (…json:9047–9051).
- Norm trajectory: shape="spike", slope=0.105, r2=0.915, n_spikes=34 (…json:9056–9063).
- Measurement guidance: `prefer_ranks=true`, `suppress_abs_probs=true`, preferred lens=`tuned`, `use_confirmed_semantics=true` (…json:8995–9009).
- Semantic margin: `delta_abs=0.002`, `p_uniform=7.63e-06`, `L_semantic_margin_ok_norm=35`, `margin_ok_at_L_semantic_norm=false` (…json:8872–8878).
- Gate‑stability (small rescalings): at `L_semantic_norm=33`, both gates pass frac=0.0; `min_both_gates_pass_frac=0.0` → calibration‑sensitive (…json:4057–4064).
- Position‑window: grid=[0.2,0.4,0.6,0.8,0.92,0.98], `rank1_frac=0.0` at L_sem=33 → position‑fragile (…json:8915–8918).
- Micro‑suite: aggregates present; n=5, n_missing=0; medians: `L_semantic_confirmed_median=33`, `L_semantic_norm_median=32` (…json:8979–8993).

**Quantitative Findings (Layer‑by‑Layer)**
- L 0 — entropy 16.999 bits, top‑1 ‘Forbes’, answer_rank=21319 [row 2 in `output-Mistral-Small-24B-Base-2501-pure-next-token.csv`].
- L 30 — entropy 16.743 bits, top‑1 ‘-на’, answer_rank=3, cos_to_final=0.099 [row 32 in pure CSV].
- L 31 — entropy 16.774–16.793 bits at prompt tokens; ‘Berlin’ begins to appear as top‑1 around the suffix (records) [rows 600–602 in `output-Mistral-Small-24B-Base-2501-records.csv`].
- L 40 — entropy 3.181 bits, top‑1 ‘Berlin’, answer_rank=1 [row 42 in pure CSV].

Bolded semantic layer: L 33 — entropy 16.774 bits, top‑1 ‘Berlin’, answer_rank=1; answer_logit_gap≈0.322 [row 35 in pure CSV]. This is the confirmed semantics layer (`L_semantic_confirmed=33`, source=raw; Δ_window=2) (…json:6629–6633, 9023–9027).

- Control margins: `first_control_margin_pos=1`, `max_control_margin=0.468`, `first_control_strong_pos=31` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:7188–7192).
- Micro‑suite: median `L_semantic_confirmed=33`, median Δ̂ not defined (null). Example fact citation: Japan→Tokyo at row_index 278 (…json:8921–8931).
- Entropy drift: median `(entropy_bits − teacher_entropy_bits)` ≈ 13.595 bits (evaluation pack p50) (…json:9063–9066).

**Qualitative Findings**

4.1. Copy vs semantics (Δ‑gap)
Copy‑reflex is absent early: `L_copy_strict=null` and no `copy_soft` hits; earliest strict copy at τ∈{0.70,0.95} is null (stability=“none”) (…json:3973–4011). Consequently Δ̂ is undefined; semantics onset fraction ≈0.825 (`L_semantic_norm=33` over 40 layers) (…json:4018–4023). Copy thresholds show no `norm_only` flags across τ (…json:3996–4007). Stability tag: none.

4.2. Lens sanity: Raw‑vs‑Norm
Artifact risk is low: `lens_artifact_score_v2=0.1847` (legacy score 0.115), with `js_divergence_p50=0.0353`, `l1_prob_diff_p50=0.3473`, `first_js_le_0.1=0`, `first_l1_le_0.5=0`, `jaccard_raw_norm_p50=0.538`, `first_jaccard_raw_norm_ge_0.5=3` (…json:9034–9045). Prevalence: `pct_layers_kl_ge_1.0=0.024`, `n_norm_only_semantics_layers=0`, `earliest_norm_only_semantic=null` (…json:9042–9045). At the semantic target, norm vs raw shows moderate agreement (J@10=0.538, Spearman_top50=0.494 at L33) but norm vs tuned is low (J@10≈0.053, Spearman_top50≈0.224 at L33) (…json:4868–4894). Given measurement guidance (`suppress_abs_probs=true`) and low cross‑lens alignment at the target, I rely on rank/KL milestones and confirmed semantics for onset.

4.3. Tuned‑Lens analysis
Tuned lens is not “calibration‑only” and is preferred for reporting (guidance), but confirmed semantics come from raw (source=raw). Attribution indicates the tuned improvement is dominated by a rotation component: ΔKL_rot at p50≈1.764 bits; ΔKL_temp slightly negative (≈−0.328); interaction adds ≈3.62 bits; total ΔKL_tuned_p50≈5.08 (…json:8820–8827). Rank earliness shifts later under tuned (le_10: 34, le_5: 35, le_1: 39 vs baseline 30/30/33) (…json:8777–8787). Positional audit shows OOD gap ≈0.656 (…json:8838–8840). Head mismatch is clean (`tau_star_modelcal=1.0`, final KL 0.0 after τ*) (…json:8842–8845). Last‑layer consistency is perfect (KL=0.0, top‑1 agree) (…json:4933–4956).

4.4. KL, ranks, cosine, entropy milestones
KL: `first_kl_below_1.0=40`, `first_kl_below_0.5=40`; final KL≈0 (…json:3914–3916, 4933–4936). Ranks (preferred lens=tuned): le_10=34, le_5=35, le_1=39 (baseline: 30,30,33) (…json:8777–8787). Cosine (norm lens): ge_0.2→L35, ge_0.4→L40, ge_0.6→L40 (…json:4012–4018). Entropy decreases late (median drift p50≈13.595 bits) (…json:9063–9066). Uniform‑margin gate fails at `L_semantic_norm` (`margin_ok_at_L_semantic_norm=false`) and top‑2 gate is null (…json:8875–8886); treat the single‑layer onset as weak unless confirmed. Stability advisories: gate‑stability min both‑gates pass frac=0.0 (calibration‑sensitive) and position‑window rank1_frac=0.0 (position‑fragile) (…json:4057–4064, 8915–8918).

4.5. Prism (shared‑decoder diagnostic)
Prism is present/compatible (k=512 at layers embed/9/19/29) (…json:846–858). It increases KL substantially (ΔKL p50≈−5.98 bits relative to baseline; negative means worse) and provides no earlier rank milestones (null) (…json:876–897). Verdict: Regressive.

4.6. Ablation & stress tests
No‑filler ablation yields `L_sem_nf=31` vs `L_sem_orig=33` (ΔL_sem=−2, ≈5% of depth) — mild stylistic sensitivity (…json:7200–7202, 6200–6220; ablation summary earlier shows −2). Control prompt summary confirms robust control margins (`first_control_margin_pos=1`, `first_control_strong_pos=31`) (…json:7188–7192). Records show “Berlin” emerging across the suffix at L31–32 (…records.csv:600–602, 618–619).

**Checklist (✓/✗/n.a.)**
- RMS lens ✓
- LayerNorm bias removed ✓
- FP32 unembed promoted ✓
- Punctuation / markup anchoring noted ✓
- Copy‑reflex ✗ (absent)
- Preferred lens honored ✓ (tuned for reporting; confirmed semantics from raw)
- Confirmed semantics reported ✓ (L=33, source=raw)
- Dual‑lens artefact metrics cited ✓ (incl. v2, JS/Jaccard/L1)
- Tuned‑lens audit done ✓ (rotation/temp/positional/head)
- normalization_provenance present ✓ (ln_source @ L0/L39)
- per‑layer normalizer effect present ✓ (resid_norm_ratio, delta_resid_cos)
- deterministic_algorithms true ✓
- numeric_health clean ✓
- copy_mask plausible ✓
- milestones.csv/evaluation_pack citations used ✓
- gate_stability_small_scale reported ✓
- position_window stability reported ✓

---
Produced by OpenAI GPT-5
