# Evaluation Report: 01-ai/Yi-34B

*Run executed on: 2025-10-16 07:26:19*
**EVAL**

**Overview**
- Model: `Yi-34B` (01-ai/Yi-34B); pre‑norm RMS stack with norm‑lens analysis and tuned‑lens audit. The probe measures copy vs semantics on a next‑token fact, tracking KL to final head, rank milestones, cosine alignment, and entropy, with raw‑vs‑norm lens diagnostics and positional/tuned audits.
- Snapshot: run‑latest outputs (norm lens primary; tuned lens preferred for semantics per guidance). Final‑head calibration is strong: "kl_to_final_bits": 0.0002783696239154555; "top1_agree": true (001_layers_baseline/run-latest/output-Yi-34B.json:3972–3990).

**Method sanity‑check**
- Prompt & indexing: Context ends with "called simply" and no trailing space (001_layers_baseline/run-latest/output-Yi-34B.json:810–819). Positive baseline rows exist (`prompt_id=pos`, `prompt_variant=orig`), e.g., L0 and L44 (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:2,46).
- Normalizer provenance: arch "pre_norm"; strategy {"primary":"next_ln1","ablation":"post_ln2_vs_next_ln1@targets"} (001_layers_baseline/run-latest/output-Yi-34B.json:2860–2868). L0 ln_source="blocks[0].ln1"; final ln_source="ln_final" (001_layers_baseline/run-latest/output-Yi-34B.json:2871–2880, 3537–3545).
- Per‑layer normalizer effect: Early spike flagged ("normalization_spike": true) with large L0 ratios ("resid_norm_ratio": 6.1399; "delta_resid_cos": 0.5717) (001_layers_baseline/run-latest/output-Yi-34B.json:822–825, 2871–2880). No abnormalities flagged at candidate semantics ("layers_flagged": []) (001_layers_baseline/run-latest/output-Yi-34B.json:3786–3801).
- Unembed bias: "present": false; "l2_norm": 0.0 (cosines are bias‑free) (001_layers_baseline/run-latest/output-Yi-34B.json:818–825).
- Environment & determinism: deterministic with SEED=316; device=cpu; dtype_compute=torch.bfloat16 (provenance.env; deterministic_algorithms=true) (001_layers_baseline/run-latest/output-Yi-34B.json:6168–6183).
- Repeatability (forward‑of‑two): Skipped due to deterministic env; pass1.layer=44; pass2.layer=null; delta_layers=null; gate.repeatability_forward_pass=null (001_layers_baseline/run-latest/output-Yi-34B.json:3948–3968).
- Decoding‑point ablation (pre‑norm): gate.decoding_point_consistent=true; at target L=44, rank1_agree=true, jaccard@10=0.6667 (001_layers_baseline/run-latest/output-Yi-34B.json:3801–3866).
- Numeric health: any_nan=false; any_inf=false; layers_flagged=[] (001_layers_baseline/run-latest/output-Yi-34B.json:3786–3801).
- Copy mask: size=1513; sample includes punctuation/markup tokens (",", ".", "```", "<!--", "-->") (001_layers_baseline/run-latest/output-Yi-34B.json:2488–2510).
- Gold alignment: ok=true; variant=with_space; first_id=19616 ("▁Berlin"); gold_alignment_rate=1.0 (001_layers_baseline/run-latest/output-Yi-34B.json:3995–4016, 4009–4015).
- Repeatability (decode micro‑check §1.39) and forward‑of‑two (§1.54): max_rank_dev=0.0; p95_rank_dev=0.0; top1_flip_rate=0.0 (001_layers_baseline/run-latest/output-Yi-34B.json:3786–3796, 8108–8116).
- Norm trajectory: shape="spike"; slope=0.0742; r2=0.926; n_spikes=4 (001_layers_baseline/run-latest/output-Yi-34B.json:5788–5796, 8118–8124).
- Measurement guidance: prefer_ranks=true; suppress_abs_probs=true; preferred_lens_for_reporting="tuned"; use_confirmed_semantics=true; reasons include "norm_only_semantics_window" and "high_lens_artifact_risk" (001_layers_baseline/run-latest/output-Yi-34B.json:8055–8070).
- Semantic margin: {delta_abs=0.002, p_uniform=1.5625e-05, margin_ok_at_L_semantic_norm=true}; L_semantic_margin_ok_norm=44 (001_layers_baseline/run-latest/output-Yi-34B.json:2619–2637).
- Gate‑stability (small rescalings): min_both_gates_pass_frac=1.0; per‑target (L=44) both_gates_pass_frac=1.0 (001_layers_baseline/run-latest/output-Yi-34B.json:2650–2667).
- Position‑window: grid=[0.2,0.4,0.6,0.8,0.92,0.98]; L_semantic_norm=44; rank1_frac=0.0 (position‑fragile) (001_layers_baseline/run-latest/output-Yi-34B.json:5774–5786).
- Micro‑suite: evaluation_pack.micro_suite.aggregates present; n=5, n_missing=0; median L_semantic_confirmed=44 (001_layers_baseline/run-latest/output-Yi-34B.json:8208–8231).

**Quantitative findings (layer‑by‑layer)**
- L 0 — entropy 15.9623 bits; top‑1 'Denote' … (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:2)
- L 15 — entropy 15.7531 bits; top‑1 '其特征是' … (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:17)
- L 30 — entropy 15.5507 bits; top‑1 'ODM' … (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:32)
- L 44 — top‑1 becomes the gold token 'Berlin'; answer_rank=1; margin gate passes (uniform) (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:46; 001_layers_baseline/run-latest/output-Yi-34B.json:2619–2637)
- L 60 — entropy 2.9812 bits; top‑1 'Berlin'; final KL≈0 (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:63; 001_layers_baseline/run-latest/output-Yi-34B.json:3972–3990)

Bolded semantic layer: L_semantic_confirmed = L 44 (confirmed under tuned lens; uniform‑margin OK) (001_layers_baseline/run-latest/output-Yi-34B.json:8066–8070, 8184–8200).

Controls: first_control_margin_pos=1; max_control_margin=0.5836; first_control_strong_pos=42 (001_layers_baseline/run-latest/output-Yi-34B.json:6228–6236).

Micro‑suite: median L_semantic_confirmed=44 across n=5 facts; example citation: "France→Paris" at row 163 (001_layers_baseline/run-latest/output-Yi-34B.json:8208–8231).

Entropy drift: entropy_gap_bits p25/50/75 = 12.29 / 12.59 / 12.78 bits vs teacher (001_layers_baseline/run-latest/output-Yi-34B.json:2568–2576).

Normalizer snapshots: L0 resid_norm_ratio=6.1399, delta_resid_cos=0.5717; at L44 resid_norm_ratio=0.2027, delta_resid_cos=0.9265 (001_layers_baseline/run-latest/output-Yi-34B.json:2871–2880, 3233–3245).

**Qualitative findings**

4.1. Copy vs semantics (Δ‑gap)
No strict or soft copy onset is detected (all τ strict null; soft k∈{1,2,3} null) (001_layers_baseline/run-latest/output-Yi-34B.json:2568–2602). Earliest strong semantics appear at L 44, well after early layers, consistent with the absence of a copy‑reflex in layers 0–3 (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:2–5). Stability tag: copy thresholds stability="none" (001_layers_baseline/run-latest/output-Yi-34B.json:2594–2602).

4.2. Lens sanity: Raw‑vs‑Norm
Lens artifact risk tier is high: lens_artifact_score_v2=0.9434; p50 JS=0.3687; p50 L1=1.0891; first_js_le_0.1=0; first_l1_le_0.5=0 (001_layers_baseline/run-latest/output-Yi-34B.json:2667–2730). Top‑K overlap is low: jaccard_raw_norm_p50=0.1111; first_jaccard_raw_norm_ge_0.5=1 (001_layers_baseline/run-latest/output-Yi-34B.json:2721–2730). Prevalence: pct_layers_kl_ge_1.0=0.6557; n_norm_only_semantics_layers=14; earliest_norm_only_semantic=44 (001_layers_baseline/run-latest/output-Yi-34B.json:2701–2721). At the semantic target, lens‑consistency is modest: jaccard@10=0.0526; jaccard@50=0.1905; spearman_top50=0.4614 at L=44 (001_layers_baseline/run-latest/output-Yi-34B.json:3868–3930). Given measurement_guidance.suppress_abs_probs=true and high artefact tier, early semantics should be interpreted via ranks/KL, not absolute probabilities (001_layers_baseline/run-latest/output-Yi-34B.json:8055–8070).

4.3. Tuned‑Lens analysis
Preferred for semantics: preferred_lens_for_reporting="tuned"; tuned_is_calibration_only=false (001_layers_baseline/run-latest/output-Yi-34B.json:8055–8070, 8154–8159). Rotation contributes the bulk of KL drop relative to temperature: ΔKL_rot p25/p50/p75 = 3.16 / 3.50 / 3.69 bits; ΔKL_temp p25/p50/p75 = −1.03 / −0.66 / 0.61; interaction p50 ≈ 3.33 (001_layers_baseline/run-latest/output-Yi-34B.json:8124–8142). Positional generalization shows an OOD gap of ≈0.85 bits (pos_ood_gap) (001_layers_baseline/run-latest/output-Yi-34B.json:8142–8154). Head mismatch is minimal: tau_star_modelcal=1.0; final tuned KL≈0.0013 bits, unchanged after tau* (001_layers_baseline/run-latest/output-Yi-34B.json:8148–8154). Last‑layer agreement is excellent (see last_layer_consistency above).

4.4. KL, ranks, cosine, entropy milestones
- KL: first_kl_below_1.0 = 60; first_kl_below_0.5 = 60 (tuned summaries) (001_layers_baseline/run-latest/output-Yi-34B.json:6368–6386). Final KL≈0 confirms good calibration (001_layers_baseline/run-latest/output-Yi-34B.json:3972–3990).
- Ranks: tuned first_rank_le_{10,5,1} = {44,44,44}; baseline norm the same at L=44 (001_layers_baseline/run-latest/output-Yi-34B.json:6368–6386, 6218–6236). Margin gate: uniform gate passes at L=44 (001_layers_baseline/run-latest/output-Yi-34B.json:2619–2637).
- Cosine: norm‑lens cos milestones ge_0.2=1, ge_0.4=44, ge_0.6=51 (001_layers_baseline/run-latest/output-Yi-34B.json:2612–2619).
- Entropy: monotone decrease late; gap vs teacher large across depth (entropy_gap_bits p50≈12.59) aligning with late KL compression (001_layers_baseline/run-latest/output-Yi-34B.json:2568–2576, 3972–3990).
- Stability: run‑of‑two semantic onset exists (L_semantic_run2=44) but full forward‑of‑two gate skipped; treat single‑run onset as nominally stable but unverified (001_layers_baseline/run-latest/output-Yi-34B.json:2639–2647, 3948–3968). Position‑window rank1_frac=0.0, so semantics are position‑fragile (001_layers_baseline/run-latest/output-Yi-34B.json:5774–5786).

4.5. Prism
Present and compatible (001_layers_baseline/run-latest/output-Yi-34B.json:830–842). KL deltas (baseline − prism): p25≈0.94, p50≈1.36, p75≈−1.01 bits; rank milestones unchanged (no earlier first_rank_le_1) (001_layers_baseline/run-latest/output-Yi-34B.json:856–892). Verdict: Neutral — some early KL drop but no earlier rank‑1 milestone and mixed later KL behavior.

4.6. Ablation & stress tests
Style ablation (no_filler): L_sem_orig=44; L_sem_nf=44; ΔL_sem=0 (001_layers_baseline/run-latest/output-Yi-34B.json:6204–6220). Negative control prompt ("Berlin is the capital of"): top‑1 is " Germany"; no evidence of gold leakage (001_layers_baseline/run-latest/output-Yi-34B.json:1–34). Important‑word trajectory: the gold token first emerges at L44 and is amplified toward the end ("Berlin" rank=1 at L44 and L60) (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:46,63).

4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓
- LayerNorm bias removed ✓
- FP32 unembed promoted ✓
- Punctuation / markup anchoring noted ✓
- Copy‑reflex ✗
- Preferred lens honored ✓
- Confirmed semantics reported ✓
- Dual‑lens artefact metrics (incl. lens_artifact_score_v2, JS/Jaccard/L1) cited ✓
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

