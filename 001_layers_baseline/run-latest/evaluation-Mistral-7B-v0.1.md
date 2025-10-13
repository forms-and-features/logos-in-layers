# Evaluation Report: mistralai/Mistral-7B-v0.1

*Run executed on: 2025-10-12 20:56:18*
**Overview**
Mistral-7B-v0.1 (32 layers). Run-latest artifacts indicate a CPU, deterministic run (seed 316) focused on layerwise copy vs. semantics using norm/rawlens, with KL, rank, cosine, and entropy trajectories plus tuned/prism lens diagnostics (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json).

**Method Sanity‑Check**
- Prompt & indexing: context ends with “called simply” and no trailing space: "Give the city name only, plain text. The capital of Germany is called simply" (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json). Positive/original rows present, e.g., layer 0 and 25: 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:2 and :27.
- Normalizer provenance: strategy "next_ln1" with ln sources blocks[0].ln1 at L0 and ln_final at last (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json).
- Per‑layer normalizer effect: norm trajectory shape "spike" with many spikes (n_spikes=26, r2=0.963, slope≈0.148), so treat early spikes as flagged (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json).
- Unembed bias: present=False; l2_norm=0.0 (cosines are bias‑free) (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json).
- Environment & determinism: device=cpu, torch=2.8.0; deterministic_algorithms=True; seed=316 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json).
- Numeric health: any_nan=False; any_inf=False; layers_flagged=[] (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json).
- Copy mask: size=1179; sample includes punctuation and control chars (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json).
- Gold alignment: ok=True; variant=with_space; first_id=8430; pieces=["▁Berlin"] (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json). gold_alignment_rate=1.0.
- Repeatability (1.39): status=skipped (deterministic_env) (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json).
- Norm trajectory: shape="spike"; slope≈0.148; r2≈0.963; n_spikes=26 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json).
- Measurement guidance: prefer_ranks=True; suppress_abs_probs=True; reasons=[norm_only_semantics_window, high_lens_artifact_risk, high_lens_artifact_score, normalization_spike]; preferred_lens_for_reporting=tuned; use_confirmed_semantics=True (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json).
- Semantic margin: delta_abs=0.002; p_uniform=3.125e‑05; margin_ok_at_L_semantic_norm=True; L_semantic_confirmed_margin_ok_norm=25 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json).
- Micro‑suite: evaluation_pack.micro_suite.aggregates present with n=5, n_missing=1; L_semantic_confirmed_median=25 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json).

**Quantitative Findings (Layer‑by‑Layer)**
- L 0 — entropy 14.96 bits, top‑1 ‘dabei’ (pos/orig) (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:2)
- L 11 — entropy 14.74 bits, top‑1 ‘[…]’ (punctuation surrogate), copy flags false (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:13)
- **L 25 — entropy 13.60 bits, top‑1 ‘Berlin’**; answer_rank=1; semantic margin ok (pos/orig) (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:27)
- L 26 — entropy 13.54 bits, top‑1 ‘Berlin’ (pos/orig) (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:28)
- L 31 — entropy 10.99 bits, top‑1 quotes; ‘Berlin’ rank=3 (pos/orig) (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:33)
- Control margin: first_control_margin_pos=2; max_control_margin=0.6539 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json).
- Micro‑suite: median L_semantic_confirmed=25; median Δ̂ unavailable; example fact citation L=25 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:27).
- Entropy drift: median entropy gap vs teacher ≈10.99 bits (p50), with p25≈10.60, p75≈11.19 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json).
- Confidence margins: at L=25, answer_logit_gap≈0.90; answer_vs_top1_gap≈0.015 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:27).

**Qualitative Findings**

4.1. Copy vs semantics (Δ‑gap)
No early copy reflex detected: neither strict nor soft copy flags fire in layers 0–3 (copy_collapse=False; copy_soft_k1@0.5=False) across early rows (see L0/L11 examples above). With L_copy_strict and L_copy_soft absent (null in thresholds), Δ̂ is undefined; semantics emerges at ~78% depth (L=25/32). Stability tag for copy thresholds is "none" (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json).

4.2. Lens sanity: Raw‑vs‑Norm
Artifact risk is high: lens_artifact_score_v2=0.6695; tier=high (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json). Symmetric/robust metrics: js_divergence_p50=0.0741; l1_prob_diff_p50=0.5050; first_js_le_0.1=0; first_l1_le_0.5=0 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json). Top‑K overlap: jaccard_raw_norm_p50=0.408; first_jaccard_raw_norm_ge_0.5=19. Prevalence: pct_layers_kl_ge_1.0≈0.242; n_norm_only_semantics_layers=1; earliest_norm_only_semantic=32 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json). Given the high tier and a norm‑only semantic near the end, prefer rank milestones and confirmed semantics when interpreting early signals.

4.3. Tuned‑Lens analysis
Tuned lens is not calibration‑only and is the preferred semantics lens (preferred_semantics_lens_hint=tuned; tuned_is_calibration_only=False) (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json). Rotation vs temperature: ΔKL_rot p25/p50/p75 ≈ 1.99/2.21/3.14 bits; ΔKL_temp p25/p50/p75 ≈ −0.41/−0.26/2.32; interaction p50≈2.13 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json). Positional generalization: pos_ood_ge_0.96≈4.14; pos_in_dist_le_0.92≈4.91; pos_ood_gap≈−0.77 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json). Head mismatch: tau_star_modelcal=1.0; kl_bits_tuned_final=0.0 → after_tau_star=0.0; last‑layer agreement holds (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json). Last‑layer consistency confirms top‑1 agreement and near‑zero KL (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json).

4.4. KL, ranks, cosine, entropy milestones
KL: first_kl_below_1.0=32 and first_kl_below_0.5=32; final KL≈0 with good last‑layer calibration (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json). Ranks (preferred lens=tuned): first_rank_le_10=23, first_rank_le_5=25, first_rank_le_1=25 (baseline values identical) (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json). Cosine milestones (norm lens): ge_0.2 at L11, ge_0.4 at L25, ge_0.6 at L26 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json). Entropy declines steadily (p50 entropy gap≈10.99 bits), with the biggest qualitative shifts coinciding with rank‑5/1 milestones around L=25; ensure margin gate holds (it does at L=25).

4.5. Prism (shared‑decoder diagnostic)
Present and compatible (layers: embed,7,15,23). KL deltas vs baseline are strongly negative (e.g., p50 ΔKL≈−17.54), indicating higher KL under prism and no earlier rank milestones (no Δ on le_{10,5,1}) (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json). Verdict: Regressive.

4.6. Ablation & stress tests
No‑filler vs original: L_sem_nf=24 vs L_sem_orig=25 (ΔL_sem=−1), well under 10% of depth → limited stylistic sensitivity (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json). Control prompts present with first_control_margin_pos=2 and strong control by L24; ‘Paris’ dominates the control continuation as expected (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json). Important‑word trajectory examples: around semantics, ‘Berlin’ becomes top‑1 at L25 and remains high through L27–28 (e.g., rows 27–29), before punctuation rises at L29–31 (rows 31–33) (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:27,28,29,31,33).

4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓
- LayerNorm bias removed ✓
- FP32 unembed promoted ✓
- Punctuation / markup anchoring noted ✓
- Copy‑reflex ✗ (none ≤ L3)
- Preferred lens honored ✓ (tuned)
- Confirmed semantics reported ✓ (L=25)
- Dual‑lens artefact metrics (incl. v2, JS/Jaccard/L1) cited ✓
- Tuned‑lens audit done (rotation/temp/positional/head) ✓
- normalization_provenance present (ln_source @ L0/final) ✓
- per‑layer normalizer effect present (resid_norm_ratio, delta_resid_cos) ✓
- deterministic_algorithms true ✓
- numeric_health clean ✓
- copy_mask plausible ✓
- milestones.csv or evaluation_pack.citations used for quotes ✓

---
Produced by OpenAI GPT-5
