# Evaluation Report: meta-llama/Meta-Llama-3-8B

*Run executed on: 2025-10-13 22:23:35*

## EVAL

**1. Overview**
Meta-Llama-3-8B was probed with a layer-by-layer logit-lens pipeline to measure copy vs. semantics onset, KL-to-final calibration, ranks, cosine geometry, and entropy trajectories, alongside lens diagnostics and tuned‑lens audits. The run targeted the fact “Germany→Berlin” with a constrained prompt ending in “called simply”, and includes raw‑vs‑norm and positional stability diagnostics (see 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11937).

**2. Method sanity-check**
- Prompt & indexing: context prompt ends with “called simply” and matches the baseline positive setup: "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply"  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:805. Positive rows exist for `prompt_id=pos, prompt_variant=orig` (e.g., layer 0 row)  001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:2.
- Normalizer provenance: "arch": "pre_norm", "strategy": "next_ln1"; at L0 "ln_source": "blocks[0].ln1", final uses "ln_final"  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7282 and 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7560.
- Per-layer normalizer effect: resid_norm_ratio and delta_resid_cos trend down without a pre‑semantic explosion; overall trajectory flagged as "shape": "spike", r2≈0.95  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:9698.
- Unembed bias: "present": false, "l2_norm": 0.0; cosines are bias‑free by construction  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:826.
- Environment & determinism: device "cpu", torch "2.8.0+cu128", "deterministic_algorithms": true, "seed": 316  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:10082 and 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:10089.
- Numeric health: any_nan=false, any_inf=false, layers_flagged=[]  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7785.
- Copy mask: ignored token strings sample includes punctuation ["!","\"","#","$","%",…], with size=6022  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:6968 and 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:6988.
- Gold alignment: ok=true, variant="with_space", pieces=["ĠBerlin"], answer_ids=[20437]  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7869.
- Repeatability (1.39): skipped due to deterministic_env; no {max_rank_dev, p95_rank_dev, top1_flip_rate} available  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7792.
- Norm trajectory: "shape": "spike", slope≈0.113, r2≈0.953, n_spikes=18  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:9698.
- Measurement guidance: {"prefer_ranks": true, "suppress_abs_probs": true, "preferred_lens_for_reporting": "tuned", "use_confirmed_semantics": true} with reasons including norm‑only semantics and low lens consistency  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11937.
- Semantic margin: delta_abs=0.002 (logit units); margin_ok_at_L_semantic_norm=false  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7108.
- Gate‑stability (small rescalings): at L_semantic_norm both_gates_pass_frac=0.0; min_both_gates_pass_frac=0.0 (calibration‑sensitive)  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7132.
- Position‑window: grid=[0.2,0.4,0.6,0.8,0.92,0.98], rank1_frac=0.0 at L_semantic_norm (position‑fragile)  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11848.
- Micro‑suite: aggregates present; n=5, n_missing=0; median L_semantic_confirmed=25  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:12030 and 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11863.

**3. Quantitative findings (layer‑by‑layer)**
- L 0 — entropy 16.9568 bits, top‑1 "itzer"  001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:2.
- L 10 — entropy 16.8506 bits, top‑1 "tons"  001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:14.
- L 20 — entropy 16.8304 bits, top‑1 "'"  001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:24.
- L 25 — entropy 16.8142 bits, top‑1 "Berlin"; answer_rank=1  001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:29.
- L 32 — entropy 2.9610 bits, top‑1 "Berlin" (final)  001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:36.

Control check: first_control_margin_pos=0, first_control_strong_pos=25; max_control_top2_logit_gap reported  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:10137.

Micro‑suite: median L_semantic_confirmed=25 and median Δ̂ unavailable; concrete citation for baseline fact at row 25  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:12030.

Entropy drift: entropy_gap_bits p50≈13.88 bits across layers (evaluation_pack.entropy)  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11996. Representative logit‑gap and normalizer snapshots at L25 show answer_logit_gap and resid_norm_ratio available for context  001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:29.

Bold semantic layer: L 25 — prefer confirmed/run‑of‑two over single‑layer onset (L_semantic_confirmed=25, L_semantic_run2=25)  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:9583 and 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7121.

**4. Qualitative findings**

4.1. Copy vs semantics (Δ‑gap)
Copy‑reflex ✗. No strict or soft copy hits were found in early layers (0–3) in the pure CSV for the positive prompt; all copy flags are False in these rows  001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:2. The milestones report L_copy_strict=null at τ∈{0.70,0.95} and no soft k∈{1,2,3} onsets; stability="none"  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7056. Δ̂ depth fraction is not provided (null) in the pack  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11937.

4.2. Lens sanity: Raw‑vs‑Norm
Lens artifact score v2≈0.459 (tier=medium); legacy≈0.418  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7278 and 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-artifact-audit.csv:1. Symmetric metrics show js_divergence_p50≈0.0168, l1_prob_diff_p50≈0.240; earliest thresholds first_js_le_0.1=0, first_l1_le_0.5=0  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7196. Top‑K overlap median jaccard@50≈0.408, with first_jaccard_raw_norm_ge_0.5 at layer 3  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7199. Norm‑only semantics are present in the candidate window (layers [25,27–30]) with max_kl_norm_vs_raw≈5.26 bits  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7149. Lens‑consistency at semantic targets is low (jaccard@10≈0.25; spearman_top50≈0.18), advising caution about early view‑dependent semantics  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7805.

4.3. Tuned‑Lens analysis
Preferred for reporting: tuned lens; it is not calibration‑only and the audit confirms a head‑consistent final layer (kl_bits_tuned_final=0 after τ⋆)  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:12030. Attribution: ΔKL_rot at p50≈2.56 bits; ΔKL_temp near zero at p50; interaction≈2.34; tuned total p50≈4.87 bits  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11996. Positional generalization: pos_in_dist_le_0.92≈5.64, pos_ood_ge_0.96≈4.87, gap≈−0.77 (weaker OOD)  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:12030. Ranks: baseline norm first_rank_le_1 at 25, tuned summaries vary by run (often 25–32)  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:6999 and 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:10551. Last‑layer consistency: top1_agree=true; kl_to_final_bits=0.0; temp_est=1.0  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7882.

4.4. KL, ranks, cosine, entropy milestones
KL: first_kl_below_1.0=32 and first_kl_below_0.5=32 (late calibration; norms settle only at the head)  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:6997. Ranks (preferred lens per guidance: tuned; baseline in parentheses): first_rank_le_10=(24), first_rank_le_5=(25), first_rank_le_1=(25)  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:6999. Cosine milestones (norm): ge_0.2→20, ge_0.4→30, ge_0.6→32  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7094. Entropy decreases steeply late; median entropy gap≈13.88 bits  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11996. Margin gates: uniform‑margin fails at L_semantic_norm (treat as weak rank‑1); no top‑2 confirmation at that layer  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7108. Stability: a run‑of‑two onset exists (L_semantic_run2=25), but gate‑stability small‑scale is 0.0 and position‑window rank‑1 fraction is 0.0; headline claims are calibration‑sensitive and position‑fragile  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7132 and 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11848.

4.5. Prism (shared‑decoder diagnostic)
Prism is present/compatible. At sampled depths, KL deltas are negative (higher KL) relative to baseline: p50 delta≈−8.29 bits; rank milestones show no improvement (prism first_rank_le_1=null in the summary)  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:805 and 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:825. Verdict: Regressive.

4.6. Ablation & stress tests
Style ablation (“no_filler”): L_sem_orig=25, L_sem_nf=25 (ΔL_sem=0)  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:10103. Negative/control prompt (France→Paris) present; control summary reports first_control_margin_pos=0 and first_control_strong_pos=25  001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:10137. Important‑word trajectory: around L=25 the positive stream’s next‑token ranks for “Berlin” become 1 across positions; e.g., "L 25, pos 16 — top‑1 Berlin"  001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:29 and supporting context in records  001_layers_baseline/run-latest/output-Meta-Llama-3-8B-records.csv:516.

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
- milestones.csv used for quotes ✓
- gate_stability_small_scale reported ✓
- position_window stability reported ✓

---
Produced by OpenAI GPT-5
