# Evaluation Report: mistralai/Mistral-7B-v0.1

*Run executed on: 2025-10-16 07:26:19*

## EVAL

**1. Overview**
- Model: `mistralai/Mistral-7B-v0.1` (7B), run artifacts dated 2025-10-16. The probe measures copy vs. semantic emergence using rank/KL thresholds, cosine and entropy trajectories, and lens diagnostics (norm/raw/tuned; prism diagnostic).
- It reports layer milestones, artifact risk (JS/L1/Jaccard), normalization provenance/effects, repeatability, decoding-point ablation, gate stability, and a small micro‑suite.

**2. Method sanity‑check**
- Prompt & indexing: context ends with “called simply” with no trailing space: "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply" (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json). Positive rows exist, e.g., `pos,orig,L=0/5/11/23/25/32` [rows 2, 7, 13, 25, 27, 34 in 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv].
- Normalizer provenance: "arch": "pre_norm"; strategy primary "next_ln1"; ablation "post_ln2_vs_next_ln1@targets" (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2433–2444). At L0: "ln_source": "blocks[0].ln1"; final: "ln_source": "ln_final" (…:2449–2458, 2768–2780).
- Per‑layer normalizer effect: early spikes are flagged; large `resid_norm_ratio` and `delta_resid_cos` before semantics, e.g., L0 `resid_norm_ratio=115.17`, `delta_resid_cos=0.3076`; L2 `resid_norm_ratio=258.96`, `delta_resid_cos=0.8539` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2449–2468). "flags.normalization_spike": true (…:835–846).
- Unembed bias: "present": false, "l2_norm": 0.0 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:826–838). Cosines are bias‑free by construction.
- Environment & determinism: device "cpu", torch "2.8.0", `deterministic_algorithms`: true, `seed`: 316 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:5328–5340). Repeatability driver also reports `deterministic_algorithms: true` (…:2942–2956).
- Repeatability (forward‑of‑two): mode "skipped_deterministic"; pass1.layer=25; pass2.layer=null; delta_layers=null; `topk_jaccard_at_primary_layer=null`; `gate.repeatability_forward_pass=null` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3105–3130). Caution: gate not executed, so single‑run onsets may be unstable.
- Decoding‑point ablation (pre‑norm): `decoding_point_consistent=true`. At target `L_semantic_norm` L=25: `rank1_agree=true`, `jaccard@10=0.538`, `jaccard@50=0.613`, `spearman_top50=0.734` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2954–3046, 3000–3036). No decoding‑point sensitivity.
- Numeric health: `any_nan=false`, `any_inf=false`, `layers_flagged=[]` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2939–2950).
- Copy mask: `size: 1179` with punctuation/control tokens sampled (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2128–2144). Plausible for tokenizer hygiene.
- Gold alignment: `ok: true`, `variant: with_space`, `first_id: 8430`, `pieces: ["▁Berlin"]` and `gold_alignment_rate=1.0` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3058–3086, 3162–3186).
- Repeatability (decode micro‑check §1.39) and forward‑of‑two (§1.54): `max_rank_dev=0.0`, `p95_rank_dev=0.0`, `top1_flip_rate=0.0` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2946–2954). Deterministic run but forward‑of‑two gate was skipped.
- Norm trajectory: `shape: "spike"`, `slope≈0.148`, `r2≈0.963` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:7238–7250).
- Measurement guidance: `prefer_ranks=true`, `suppress_abs_probs=true`; reasons include `norm_only_semantics_window`, `high_lens_artifact_risk`, `normalization_spike`, `pos_window_low_stability`, `scale_sensitive_semantic_gate`; preferred lens = `tuned`; `use_confirmed_semantics=true` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:7184–7227).
- Semantic margin: `delta_abs=0.002`, `p_uniform=3.125e-05`, `margin_ok_at_L_semantic_norm=true` and `L_semantic_confirmed_margin_ok_norm=25` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:7064–7072, 2211–2230).
- Gate‑stability (small rescalings): `min_both_gates_pass_frac=0.0`; at `L_semantic_norm` L=25: `uniform_margin_pass_frac=1.0`, `top2_gap_pass_frac=0.0`, `both_gates_pass_frac=0.0` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2276–2295, 2228–2242). Calibration‑sensitive.
- Position‑window: grid `[0.2,0.4,0.6,0.8,0.92,0.98]`, `rank1_frac=0.0` at L_sem (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:7095–7112). Position‑fragile.
- Micro‑suite: `evaluation_pack.micro_suite.aggregates` present with `n=5`, `n_missing=1`. Median `L_semantic_confirmed=25` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:7310–7420, 7295–7310).

**3. Quantitative findings (layer‑by‑layer)**
- L 0 — entropy 14.96 bits, top‑1 'dabei' [row 2 in 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv]
- L 5 — entropy 14.83 bits, top‑1 '…' [row 7 in 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv]
- L 11 — entropy 14.74 bits, top‑1 '…' [row 13 in 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv]
- L 23 — entropy 14.40 bits, top‑1 'simply' [row 25 in 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv]
- L 25 — entropy 13.60 bits, top‑1 'Berlin'; answer_rank=1 [row 27 in 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv]
- L 32 — entropy 3.61 bits, top‑1 'Berlin'; final‑head aligned [row 34 in 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv]

Bolded semantic layer: L_semantic_norm = 25. Uniform margin passes but Top‑2 gate is not satisfied and gate‑stability is low; treat as weak rank‑1 and calibration‑sensitive. Decoding‑point gate passed (pre‑norm), so no decoding‑point sensitivity.

Controls: `first_control_margin_pos=2`, `max_control_margin≈0.654`, `first_control_strong_pos=24` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:5394–5412).

Micro‑suite: median `L_semantic_confirmed=25` (IQR for `L_semantic_norm` = [24, 27]); one fact missing semantics; example fact citation `Germany→Berlin` at row 25 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:7068–7112, 7310–7420).

Entropy drift: teacher entropy ≈3.61 bits, gaps at percentiles p25/p50/p75 ≈ 10.60/10.99/11.19 bits (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2216–2224, 2336–2344). Residual norm/cos spikes occur well before L_semantic and are flagged.

Normalizer snapshots: large `resid_norm_ratio` and `delta_resid_cos` early (e.g., L2 `258.96`, `0.8539`) trending down by late layers (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2449–2468, 2688–2708).

**4. Qualitative findings**

4.1. Copy vs semantics (Δ‑gap)
No strict or soft copy reflex is detected in early layers: all `L_copy_strict@{0.70,0.95}` and `L_copy_soft[k]` are null, stability="none" (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2211–2230). Therefore Δ̂ cannot be computed (no copy onset; `depth_fractions.delta_hat=null`, 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:7217–7226). Copy‑thresholds stability tag: none; earliest strict copy (τ=0.70/0.95): null; no norm‑only flags.

4.2. Lens sanity: Raw‑vs‑Norm
Artifact risk is high: `lens_artifact_score_v2=0.6695` (tier=high), with `js_divergence_p50≈0.074`, `l1_prob_diff_p50≈0.505`, `first_js_le_0.1=0`, `first_l1_le_0.5=0` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2330–2360, 7226–7231). Top‑K overlap is moderate: `jaccard_raw_norm_p50≈0.408`, `first_jaccard_raw_norm_ge_0.5=19` (…:2350–2360, 7230–7236). Prevalence: `pct_layers_kl_ge_1.0≈0.242`, `n_norm_only_semantics_layers=1`, `earliest_norm_only_semantic=32` near the final layer (…:2330–2350). Lens‑consistency at semantic targets is moderate (`jaccard@10≈0.429`, `spearman_top50≈0.504` at L=25; 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3028–3050). Given the high tier and a norm‑only semantics layer late, prefer rank milestones and confirmed semantics; avoid absolute‑p claims.

4.3. Tuned‑Lens analysis
Preference: `tuned_is_calibration_only=false`; guidance prefers tuned for reporting (`preferred_semantics_lens_hint: tuned`) (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:7268–7298, 7184–7196). Attribution: ΔKL percentiles show sizable tuned gains with rotation dominating: `delta_kl_rot_p50≈2.21`, `delta_kl_temp_p50≈−0.26`, `delta_kl_interaction_p50≈2.13`, overall `delta_kl_tuned_p50≈4.14` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:7236–7266, 6996–7056). Rank earliness: tuned leaves `first_rank_le_5` unchanged (25) but delays `le_1` to 32 vs 25 baseline (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:6948–6996). Positional generalization: `pos_in_dist_le_0.92≈4.91` vs `pos_ood_ge_0.96≈4.14`, small negative gap (…:7019–7053). Head mismatch: last‑layer KL=0 both before and after `τ*` (…:7050–7060). Last‑layer consistency: `kl_to_final_bits=0.0`, `top1_agree=true` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3125–3160).

4.4. KL, ranks, cosine, entropy milestones
KL: `first_kl_below_1.0=32`, `first_kl_below_0.5=32`; final KL≈0 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2143–2152, 5352–5412, 3125–3160). Ranks (preferred lens per guidance = tuned; baseline in parentheses): `first_rank_le_10=25 (23)`, `le_5=25 (25)`, `le_1=32 (25)` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:6948–6996). Cosine milestones (norm lens): `ge_0.2=11`, `ge_0.4=25`, `ge_0.6=26` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2249–2261). Entropy decreases steadily; gaps vs teacher entropy are large across the stack (p25/p50/p75 ~ 10.6/11.0/11.2 bits; 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2196–2224). Margin gate: uniform gate passes at L=25 but Top‑2 gap gate does not; label the onset as weak. Stability: bolded onset is not a run‑of‑two strong onset; coupled with small‑scale gate‑stability=0.0 and `position_window.rank1_frac=0.0`, treat timing as calibration‑sensitive and position‑fragile.

4.5. Prism (shared‑decoder diagnostic)
Present and compatible. KL deltas are strongly negative (KL increases under prism): baseline p50≈10.33 bits vs prism p50≈27.87 bits, with rank milestones not improving (`le_1`: null) (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:880–916). Verdict: Regressive.

4.6. Ablation & stress tests
No‑filler ablation: `L_sem_orig=25` → `L_sem_nf=24` (`ΔL_sem=-1`, <10% of 32 layers; low stylistic sensitivity) (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:5352–5360).
Control prompt: `first_control_margin_pos=2`, `max_control_margin≈0.654`, `first_control_strong_pos=24` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:5394–5412). Negative/control probe "Berlin is the capital of" has Germany in top‑1 at shallow depths (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1–40).
Important‑word trajectory (records CSV): around L=22–25, 'Berlin' and 'Germany' rise into the shortlist: e.g., L=23,pos=15 includes Berlin in top‑ranks [row 408 in 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-records.csv]; L=24,pos=14/15 top‑1 becomes Berlin [rows 424–425]; L=25,pos=14 top‑1 Berlin and Germany/Capital nearby [row 441].

4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓
- LayerNorm bias removed ✓
- FP32 unembed promoted ✓
- Punctuation / markup anchoring noted ✓
- Copy‑reflex ✗ (none detected)
- Preferred lens honored ✓ (tuned for reporting)
- Confirmed semantics reported ✓ (L=25 under raw; used for robustness)
- Dual‑lens artefact metrics cited (incl. v2, JS/Jaccard/L1) ✓
- Tuned‑lens audit (rotation/temp/positional/head) ✓
- normalization_provenance present (ln_source @ L0/final) ✓
- per‑layer normalizer effect (resid_norm_ratio, delta_resid_cos) ✓
- deterministic_algorithms true ✓
- numeric_health clean ✓
- copy_mask plausible ✓
- milestones.csv or evaluation_pack.citations used for quotes ✓
- gate_stability_small_scale reported ✓
- position_window stability reported ✓

---
Produced by OpenAI GPT-5

