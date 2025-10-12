# Evaluation Report: mistralai/Mistral-7B-v0.1

*Run executed on: 2025-10-11 21:50:12*
**Overview**
- Model: Mistral-7B-v0.1; run-latest artifacts present (e.g., `001_layers_baseline/run-latest/timestamp-20251011-2150`). The probe measures copy reflex vs. emergent semantics using rank/KL/cosine/entropy trajectories with norm/rawlens diagnostics and tuned‑lens audits.

**Method sanity‑check**
- Prompt & indexing: Context prompt ends with “called simply” and no trailing space: "Give the city name only, plain text. The capital of Germany is called simply" 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:10. Positive rows exist for `prompt_id=pos`, `prompt_variant=orig` (e.g., Germany→Berlin L=0 and L=25) 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:2, 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:27.
- Normalizer provenance: `arch: pre_norm`, `strategy: next_ln1` 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2400. Sources: L0 `blocks[0].ln1` 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2405; final `ln_final` 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2732.
- Per‑layer normalizer effect: Early spikes are flagged (`normalization_spike: true`) 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:829; large `resid_norm_ratio` and `delta_resid_cos` at L≤3, e.g., L2 `resid_norm_ratio≈258.96`, `delta_resid_cos≈0.85` 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2413.
- Unembed bias: `present: false`, `l2_norm: 0.0` 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:822. Cosines are bias‑free by construction.
- Environment & determinism: `device: cpu`, `torch_version: 2.8.0+cu128`, `deterministic_algorithms: true`, `seed: 316` 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:5003.
- Numeric health: `any_nan: false`, `any_inf: false`, `layers_flagged: []` 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2904.
- Copy mask: Ignored token id list present (sample: 12, 13, 14, 15, 16, 31…) 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:943. Size not explicitly reported.
- Gold alignment: `gold_alignment: ok` with `gold_alignment_rate: 1.0` 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2979, 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2951.
- Repeatability: Skipped in deterministic env (`status: skipped`, `reason: deterministic_env`) 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2913; `{max_rank_dev, p95_rank_dev, top1_flip_rate} = null` 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:6859.
- Norm trajectory: `shape: spike`, `slope≈0.148`, `r2≈0.963`, `n_spikes: 26` 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:6766.
- Measurement guidance: `prefer_ranks: true`, `suppress_abs_probs: true`, `preferred_lens_for_reporting: tuned`, `use_confirmed_semantics: true` 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:6717.
- Semantic margin: `delta_abs: 0.002`, `p_uniform: 3.125e-05`, `margin_ok_at_L_semantic_norm: true`; `L_semantic_confirmed_margin_ok_norm: 25` 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:6618.
- Micro‑suite: Aggregates present with `n=5`, `n_missing=1`; `L_semantic_confirmed_median=25` 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:6705.

**Quantitative findings (layer‑by‑layer)**
- L 0 — entropy 14.96 bits; top‑1 token “dabei” [row 2 in CSV] 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:2.
- L 11 — entropy 14.74 bits; top‑1 token “[…]” [row 13 in CSV] 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:13.
- L 25 — entropy 13.60 bits; top‑1 token “Berlin”; answer_rank=1; KL_to_final_bits_norm_temp≈4.82 [row 27 in CSV] 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:27.
- L 32 — entropy 3.61 bits; top‑1 token “Berlin”; KL_to_final_bits≈0.0 [row 34 in CSV] 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:34.

- Semantic layer: Bold confirmed layer is L 25 (preferred lens tuned; confirmation source=raw) 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:6750. Margin gate holds at L 25 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:6618.
- Control margin: `first_control_margin_pos: 2`, `max_control_margin≈0.654` 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:5055.
- Micro‑suite: Median L_semantic_confirmed = 25; median Δ̂ not defined (no copy milestone). Example fact row: Germany→Berlin uses row_index=25 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:6870.
- Entropy drift: Median entropy gap vs teacher is large (p50≈10.99 bits) 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:6786, narrowing to ≈0 at the final layer (KL≈0) 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2925.

**Qualitative findings**

4.1. Copy vs semantics (Δ‑gap)
No strict or soft copy collapse is detected; earliest strict copy at τ∈{0.70,0.95} is null and `stability: none` 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2216. Early layers 0–3 in pure CSV do not flag copy (all copy flags False) 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:2–5. With no copy milestone, Δ̂ is not defined (evaluation_pack depth_fractions lacks delta_hat) 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:6755.

4.2. Lens sanity: Raw‑vs‑Norm
Artifact risk is high: `lens_artifact_score_v2≈0.670` (tier=high) 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2369. Symmetric metrics: `js_divergence_p50≈0.074`, `l1_prob_diff_p50≈0.505`; earliest `js≤0.1` at L0 and `l1≤0.5` at L0 by definition of the summary counters 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2364. Top‑K overlap: `jaccard_raw_norm_p50≈0.408`, first ≥0.5 at L 19 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2366. Prevalence: `pct_layers_kl_ge_1.0≈0.242`, `n_norm_only_semantics_layers=1`, earliest norm‑only semantic at L 32 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2367. Caution: prefer rank milestones and confirmed semantics near L 25 due to high artifact tier.

4.3. Tuned‑Lens analysis
Preference: Not calibration‑only (`tuned_is_calibration_only=false`); `preferred_semantics_lens_hint: tuned` 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:6818. Attribution: ΔKL at percentiles shows sizeable rotation and interaction terms (`ΔKL_rot_p50≈2.21`, `ΔKL_temp_p50≈−0.26`, `ΔKL_interaction_p50≈2.13`; tuned total p50≈4.14) 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:6788. Rank earliness: tuned reaches `first_rank_le_1=32` vs baseline 25 (later by 7 layers); `first_kl_le_1.0`: tuned 26 vs baseline 32 (earlier by 6) 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:6529. Positional generalization: `pos_ood_ge_0.96≈4.14`, `pos_in_dist_le_0.92≈4.91`, gap≈−0.77 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:6801. Head mismatch: `tau_star_modelcal=1.0`, `kl_bits_tuned_final=0.0` and remains 0.0 after τ* 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:6811. Last‑layer consistency holds (`kl_to_final_bits=0.0`, `top1_agree=true`) 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2924.

4.4. KL, ranks, cosine, entropy milestones
KL: `first_kl_le_1.0` baseline at L 32; tuned at L 26 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:6529. Final KL≈0 with top‑1 agreement 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2925. Ranks (preferred lens=tuned): `le_10=25`, `le_5=25`, `le_1=32` (baseline: 23/25/25) 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:6531. Cosine milestones (norm lens): ge≥0.2 at L 11, ≥0.4 at L 25, ≥0.6 at L 26 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2248. Entropy: non‑monotonic early with spikes (norm trajectory “spike”), then sharp drop by the final layer to the teacher entropy (3.611 bits) 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:6766, 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:34. Margin gate: holds at L 25 (reporting of rank‑1 is valid under uniform‑margin gate) 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:6618.

4.5. Prism (shared‑decoder diagnostic)
Present and compatible 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:832. KL deltas are negative (Prism larger KL): p50 baseline≈10.33 vs prism≈27.87 (Δ≈−17.54); no earlier ranks 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:863. Verdict: Regressive.

4.6. Ablation & stress tests
No‑filler ablation: `L_sem_orig=25`, `L_sem_nf=24` (ΔL_sem=−1, <10% of 32 layers) 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:5038. Control prompt present (“Berlin is the capital of …”) with top‑1 “Germany” 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:12. Important‑word trajectory: at L 25 the top‑2 candidate includes “Germany” while “Berlin” is top‑1 [row 27 in CSV] 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:27; by L 32, “Berlin” remains top‑1 [row 34] 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:34.

4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓
- LayerNorm bias removed ✓
- FP32 unembed promoted ✓
- Punctuation / markup anchoring noted ✓
- Copy‑reflex ✗ (no early copy flags)
- Preferred lens honored ✓ (tuned for ranks; norm for context)
- Confirmed semantics reported ✓
- Dual‑lens artifact metrics (incl. v2/JS/Jaccard/L1) cited ✓
- Tuned‑lens audit done (rotation/temp/positional/head) ✓
- normalization_provenance present (ln_source @ L0/final) ✓
- per‑layer normalizer effect present (resid_norm_ratio, delta_resid_cos) ✓
- deterministic_algorithms true ✓
- numeric_health clean ✓
- copy_mask plausible ✓ (IDs listed; size n.a.)
- milestones.csv or evaluation_pack.citations used for quotes ✓

---
Produced by OpenAI GPT-5
