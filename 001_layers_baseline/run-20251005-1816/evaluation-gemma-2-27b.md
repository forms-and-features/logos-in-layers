# Evaluation Report: google/gemma-2-27b

*Run executed on: 2025-10-05 18:16:50*

## EVAL

**Overview**
The probe evaluates google/gemma-2-27b (46 layers) on 2025-10-05, analyzing copy-reflex vs. semantic emergence using rank/KL thresholds, cosine and entropy trajectories, and dual-lens diagnostics. Raw-vs-Norm and tuned‑lens audits are included to assess lens artefact risk and calibration.

**Method sanity‑check**
- Prompt & indexing: context ends with “called simply” and no trailing space: "context_prompt": "... is called simply" (001_layers_baseline/run-latest/output-gemma-2-27b.json:812). Positive rows present, e.g., "pos,orig,0,..." [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2].
- Normalizer provenance: "arch": "pre_norm", "strategy": "next_ln1"; L0 ln_source "blocks[0].ln1"; final uses "ln_final" (001_layers_baseline/run-latest/output-gemma-2-27b.json:5913, 5920–5923).
- Per‑layer normalizer effect: early spikes flagged via "flags.normalization_spike": true (001_layers_baseline/run-latest/output-gemma-2-27b.json:846); L0 snapshot shows resid_norm_ratio 0.7865, delta_resid_cos 0.5722 [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2].
- Unembed bias: "unembed_bias": { "present": false, "l2_norm": 0.0 } (001_layers_baseline/run-latest/output-gemma-2-27b.json:826–836). Cosines are bias‑free.
- Environment & determinism: device "cpu", torch "2.8.0+cu128", "deterministic_algorithms": true, seed 316 (001_layers_baseline/run-latest/output-gemma-2-27b.json:7285–7293). Reproducibility OK.
- Numeric health: any_nan=false, any_inf=false, layers_flagged=[] (001_layers_baseline/run-latest/output-gemma-2-27b.json:6626–6633).
- Copy mask: size 4668, sample contains whitespace runs ("\n" repeats) (001_layers_baseline/run-latest/output-gemma-2-27b.json:5618–5642, 5643–5646).
- Gold alignment: "ok": true, first_id=12514, pieces=["▁Berlin"], variant="with_space" (001_layers_baseline/run-latest/output-gemma-2-27b.json:6637–6650).
- Repeatability (1.39): skipped due to deterministic_env; max_rank_dev/p95_rank_dev/top1_flip_rate = null (001_layers_baseline/run-latest/output-gemma-2-27b.json:7979–7987).
- Norm trajectory: shape="spike", slope=0.0878, r2=0.846, n_spikes=16 (001_layers_baseline/run-latest/output-gemma-2-27b.json:7119–7126, 7989–7993).
- Measurement guidance: prefer_ranks=true, suppress_abs_probs=true, preferred_lens_for_reporting="norm", use_confirmed_semantics=true; reasons include "warn_high_last_layer_kl", "norm_only_semantics_window", "high_lens_artifact_risk" (001_layers_baseline/run-latest/output-gemma-2-27b.json:7926–7940).

**Quantitative findings (layer‑by‑layer)**
- L 0 — entropy 0.00050 bits, top‑1 " simply"; copy flags set (strict@0.95=True) [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2].
- L 3 — entropy 0.88567 bits, top‑1 " simply" (rank semantic not yet achieved) [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:5].
- L 23 — entropy 7.01041 bits, top‑1 " dieſem"; mid‑depth KL to final 41.60 bits [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:25].
- L 46 — entropy 0.11805 bits, top‑1 "Berlin"; answer_rank=1; kl_to_final_bits=1.1352 [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48].
- Semantic layer: bolded per guidance — L_semantic_norm=46; confirmed at 46 by tuned (source="tuned") (001_layers_baseline/run-latest/output-gemma-2-27b.json:7088–7101).
- Control margin: first_control_margin_pos=0; max_control_margin=0.99109 (001_layers_baseline/run-latest/output-gemma-2-27b.json:7341–7343).
- Entropy drift: teacher_entropy_bits=2.8856; gaps p25=4.1170, p50=4.6796, p75=5.3879 (001_layers_baseline/run-latest/output-gemma-2-27b.json:5782–5810, 7998–8004).
- Normalizer snapshot: L0 resid_norm_ratio=0.7865, delta_resid_cos=0.5722; L3 resid_norm_ratio=0.0348, delta_resid_cos=0.4799 [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2,5].

**Qualitative findings**

4.1. Copy vs semantics (Δ‑gap)
Copy‑reflex ✓. Layer 0 registers copy collapse: "copy_collapse, copy_strict@0.95=True" [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2]. Earliest strict copy is L=0 at τ=0.70 and τ=0.95 ("L_copy_strict": {"0.7": 0, "0.95": 0}; stability="mixed") (001_layers_baseline/run-latest/output-gemma-2-27b.json:5705–5726). With L_semantic_norm=46, Δ̂ = 1.0 of depth (evaluation_pack.depth_fractions.delta_hat=1.0) (001_layers_baseline/run-latest/output-gemma-2-27b.json:7959–7967). This is an extreme surface‑to‑meaning gap: all semantics confirmed only at the final layer (confirmed_source="tuned") (001_layers_baseline/run-latest/output-gemma-2-27b.json:7098–7101).

4.2. Lens sanity: Raw‑vs‑Norm
Lens artefact risk is high: legacy score 0.987 and lens_artifact_score_v2=1.0, tier="high" (001_layers_baseline/run-latest/output-gemma-2-27b.json:5909, 5917–5920; 7967–7971). Symmetric metrics indicate large divergence between raw and norm: js_divergence_p50=0.8653, l1_prob_diff_p50=1.8929; no layer with JS≤0.1/L1≤0.5 (first_js_le_0.1=0; first_l1_le_0.5=0) (001_layers_baseline/run-latest/output-gemma-2-27b.json:5799–5800, 7968–7971). Top‑K overlap is moderate: jaccard_raw_norm_p50=0.5625; first_jaccard_raw_norm_ge_0.5=1 (001_layers_baseline/run-latest/output-gemma-2-27b.json:5802–5826). Prevalence: pct_layers_kl_ge_1.0=0.9787; one norm‑only semantics layer at 46 ("earliest_norm_only_semantic": 46) (001_layers_baseline/run-latest/output-gemma-2-27b.json:5782–5798). Given the high tier and a norm‑only hit at the candidate semantics layer, prefer rank milestones and confirmed semantics when interpreting early signals.

4.3. Tuned‑Lens analysis
Tuned lens is marked calibration‑only: "tuned_is_calibration_only": true; prefer norm lens for semantics (001_layers_baseline/run-latest/output-gemma-2-27b.json:789–806, 7926–7940). Attribution shows temperature dominates KL changes: ΔKL_temp p25/p50/p75 = 0.168/0.475/0.651; rotation modestly reduces KL: ΔKL_rot p25/p50/p75 = −0.057/−0.030/0.041 (001_layers_baseline/run-latest/output-gemma-2-27b.json:7940–7976). Head mismatch improves with model‑calibrated τ*: kl_bits_tuned_final 1.1352 → 0.5131; tau_star_modelcal=2.85 (001_layers_baseline/run-latest/output-gemma-2-27b.json:789–806, 7994–8004). Rank earliness is unchanged: first_rank_le_{10,5,1}=46 (baseline and tuned) (001_layers_baseline/run-latest/output-gemma-2-27b.json:5648–5650, 6856–6859). Last‑layer agreement check: kl_to_final_bits=1.1352, top1_agree=true, kl_after_temp_bits=0.5665, warn_high_last_layer_kl=true (001_layers_baseline/run-latest/output-gemma-2-27b.json:6666–6676).

4.4. KL, ranks, cosine, entropy milestones
- KL: Norm lens never reaches KL≤1.0 (first_kl_below_1.0=null) (001_layers_baseline/run-latest/output-gemma-2-27b.json:5646–5647), while tuned attains KL≤1.0 at L=46 (001_layers_baseline/run-latest/output-gemma-2-27b.json:6856–6857). Final‑head calibration is imperfect per last-layer KL.
- Ranks: first_rank_le_10/5/1 = 46 under preferred lens (norm); tuned unchanged (001_layers_baseline/run-latest/output-gemma-2-27b.json:5648–5650, 6857–6859).
- Cosine: ge_0.2@L=1, ge_0.4@L=46, ge_0.6@L=46 (norm) (001_layers_baseline/run-latest/output-gemma-2-27b.json:5743–5747).
- Entropy: teacher entropy 2.8856 bits, with positive entropy gaps p25/p50/p75 ≈ 4.12/4.68/5.39 (001_layers_baseline/run-latest/output-gemma-2-27b.json:5782–5810, 7998–8004). Entropy drops sharply only at the final layer where rank resolves (row L=46) [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48].

4.5. Prism (shared‑decoder diagnostic)
Prism present/compatible (k=512; layers=[embed,10,22,33]) (001_layers_baseline/run-latest/output-gemma-2-27b.json:820–846). KL drops substantially at sampled depths (p50: 43.15 → 19.42; Δ≈23.73 bits) without earlier rank milestones (prism rank milestones null) (001_layers_baseline/run-latest/output-gemma-2-27b.json:846–888). Verdict: Neutral — strong KL reductions but no demonstrated earlier rank ≤1.

4.6. Ablation & stress tests
Ablations: L_copy_orig=0, L_sem_orig=46; L_copy_nf=3, L_sem_nf=46; ΔL_copy=3, ΔL_sem=0 (001_layers_baseline/run-latest/output-gemma-2-27b.json:7310–7331). Control prompts present; summary shows first_control_margin_pos=0 and max_control_margin=0.9911 (001_layers_baseline/run-latest/output-gemma-2-27b.json:7341–7343). Important‑word trajectory: “Germany” appears as early top‑1 at shallow layers (e.g., L=3: token="Germany" top‑1) [001_layers_baseline/run-latest/output-gemma-2-27b-records.csv:66; 83], while “Berlin” only resolves at L=46 [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48].

4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓ — first/final LN are RMSNorm (001_layers_baseline/run-latest/output-gemma-2-27b.json:812–816)
- LayerNorm bias removed ✓ — "layernorm_bias_fix": "not_needed_rms_model" (001_layers_baseline/run-latest/output-gemma-2-27b.json:812–816)
- FP32 unembed promoted ✓ — "mixed_precision_fix": "casting to fp32 before unembed"; unembed_dtype="torch.float32" (001_layers_baseline/run-latest/output-gemma-2-27b.json:812–816)
- Punctuation/markup anchoring noted ✓ — copy‑mask ignores whitespace; sample shows newline runs (001_layers_baseline/run-latest/output-gemma-2-27b.json:5618–5646)
- Copy‑reflex ✓ — early copy flags at L0 [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2]
- Preferred lens honored ✓ — reporting under norm; tuned used for confirmation (001_layers_baseline/run-latest/output-gemma-2-27b.json:7926–7940)
- Confirmed semantics reported ✓ — L_semantic_confirmed=46, source=tuned (001_layers_baseline/run-latest/output-gemma-2-27b.json:7098–7101)
- Dual‑lens artefact metrics cited ✓ — v2 score, JS/Jaccard/L1 (001_layers_baseline/run-latest/output-gemma-2-27b.json:5782–5826, 5909–5920)
- Tuned‑lens audit done ✓ — rotation/temp/positional/head (001_layers_baseline/run-latest/output-gemma-2-27b.json:789–806, 7926–7976)
- normalization_provenance present ✓ — ln_source @ L0 and final (001_layers_baseline/run-latest/output-gemma-2-27b.json:5913, 5920–5923)
- per‑layer normalizer effect present ✓ — resid_norm_ratio, delta_resid_cos [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2,5]
- deterministic_algorithms true ✓ (001_layers_baseline/run-latest/output-gemma-2-27b.json:7285–7293)
- numeric_health clean ✓ (001_layers_baseline/run-latest/output-gemma-2-27b.json:6626–6633)
- copy_mask plausible ✓ (001_layers_baseline/run-latest/output-gemma-2-27b.json:5618–5646)
- milestones.csv used ✓ — L_copy/L_semantic rows (001_layers_baseline/run-latest/output-gemma-2-27b-milestones.csv:1–4)

—
Produced by OpenAI GPT-5
