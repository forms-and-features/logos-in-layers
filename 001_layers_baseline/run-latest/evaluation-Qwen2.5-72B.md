# Evaluation Report: Qwen/Qwen2.5-72B

## 1. Overview
Qwen/Qwen2.5-72B was probed on 2025-10-16 for next-token prediction across 80 layers. The probe tracks copy vs. semantic collapse with rank/KL milestones, cosine and entropy trajectories, and dual-lens (raw vs norm) diagnostics including decoding-point ablation and artifact risk.

## 2. Method sanity-check
- Prompt & indexing: "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply" (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:819). Positive/original rows present, e.g., "Germany→Berlin,0,pos,orig,0,15,⟨NEXT⟩,17.2142…" (001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:2).
- Normalizer provenance: "arch": "pre_norm"; "strategy": { "primary": "next_ln1", "ablation": "post_ln2_vs_next_ln1@targets" } (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7638–7641). Per-layer ln source: L0 "blocks[0].ln1" (…:7646–7654) and final "ln_final" at layer 80 (…:8365–8373).
- Per-layer normalizer effect: early spikes present and flagged: e.g., layer 2 has "resid_norm_ratio": 8.939…, "delta_resid_cos": 0.5767 (…:7652–7660). "normalization_spike": true (…:839–846).
- Unembed bias: "present": false; "l2_norm": 0.0 (…:828–836). Cosines reported are bias-free by design.
- Environment & determinism: device "cpu"; torch "2.8.0+cu128"; "deterministic_algorithms": true; seed 316 (…:9735–9751). Reproducibility OK.
- Repeatability (forward-of-two): enabled but "mode": "skipped_deterministic"; milestones primary=L_semantic_norm, pass1.layer=80, pass2.layer=null, delta_layers=null; topk_jaccard_at_primary_layer=null; gate.repeatability_forward_pass=null (…:8985–9002). Treat as not assessed.
- Decoding-point ablation (pre-norm): gate.decoding_point_consistent=false (…:8936). At L_semantic_norm (layer 80): rank1_agree=true, jaccard@10=0.4286 (…:8899–8911). At first_rank_le_5 (layer 78): rank1_agree=false, jaccard@10=0.3333 (…:8918–8930). Mark early semantics as decoding-point sensitive.
- Numeric health: any_nan=false, any_inf=false, layers_flagged=[] (…:8863–8868).
- Copy mask: size=6244; sample tokens ["!", '"', "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", ":"] (…:7237–7259). Plausible for tokenizer punctuation/control.
- Gold alignment: ok=true, variant="with_space", answer_ids includes Berlin id=19846 (…:9019–9038).
- Repeatability (decode micro-check §1.39): layers_checked=81, max_rank_dev=0.0, p95_rank_dev=0.0, top1_flip_rate=0.0 (…:8870–8876). Stable under deterministic replay.
- Norm trajectory: shape="spike", slope=0.0642, r2=0.9226, n_spikes=55 (…:9969–9983).
- Measurement guidance: prefer_ranks=true; suppress_abs_probs=true; preferred_lens_for_reporting="norm"; use_confirmed_semantics=false; reasons include "high_lens_artifact_risk", "decoding_point_sensitive" (…:9920–9937). Follow ranks/KL, avoid absolute p.
- Semantic margin: { delta_abs=0.002, p_uniform=6.576e-06, L_semantic_margin_ok_norm=80, margin_ok_at_L_semantic_norm=true } (…:9804–9812). Uniform-margin gate passes at L_semantic_norm.
- Gate-stability (small rescalings): for L_semantic_norm: uniform_margin_pass_frac=1.0, top2_gap_pass_frac=0.0, both_gates_pass_frac=0.0; min_both_gates_pass_frac=0.0 (…:7400–7415). Calibration-sensitive under rescaling.
- Position-window: grid=[], L_semantic_norm=80, rank1_frac=null, n_positions=0 (…:9346–9352). Position stability not measured; avoid generalization across positions.
- Micro-suite: present. Aggregates: n=5, n_missing=3, L_semantic_norm_median=80, L_semantic_confirmed_median=null (…:9990–10008, 10099–10110). Cited rows include Germany→Berlin=80 and Japan→Tokyo=566.

## 3. Quantitative findings (layer-by-layer)
- L 0 — entropy 17.214 bits; top‑1 's'; answer_rank=67377; cos_to_final=0.588 (001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:2).
- L 40 — entropy 16.192 bits; top‑1 'iéndo'; answer_rank=106367; cos_to_final=−0.710 (…-pure-next-token.csv:98).
- L 74 — entropy 16.081 bits; top‑1 ' " '; answer_rank=9; cos_to_final=0.739 (…-pure-next-token.csv:132).
- L 78 — entropy 15.398 bits; top‑1 ' " '; answer_rank=5; cos_to_final=0.755 (…-pure-next-token.csv:136).
- L 80 — entropy 4.116 bits; top‑1 ' Berlin'; answer_rank=1; cos_to_final≈1.000 (…-pure-next-token.csv:138).

Bolded semantic layer: L 80 — L_semantic_norm (uniform‑margin OK; decoding‑point sensitive; calibration‑sensitive under small rescalings). First‑rank milestones (norm lens): first_rank_le_10=74, first_rank_le_5=78, first_rank_le_1=80; first_kl_below_1.0=80; first_kl_below_0.5=80 (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7248–7267).

Controls: first_control_margin_pos=0; max_control_margin=0.2070; first_control_strong_pos=null (…/output-Qwen2.5-72B.json:9799–9819).

Micro‑suite: median L_semantic_norm=80 across n=5 facts (n_missing=3). Example citation: "Japan→Tokyo … layer 80 … top‑1 ' Tokyo' … answer_rank=1" (001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:1115).

Entropy drift: entropy_gap_bits_p50=12.496 bits (evaluation_pack.entropy) with large early‑layer gap decreasing sharply only at L≈80 (…/output-Qwen2.5-72B.json:9990–9998).

Confidence margins and normalizer snapshot: at L 80, answer_logit_gap is recorded; resid_norm_ratio=0.1723, delta_resid_cos=0.929 (…-pure-next-token.csv:138; …/output-Qwen2.5-72B.json:8365–8373).

## 4. Qualitative findings

### 4.1. Copy vs semantics (Δ‑gap)
No early copy reflex observed: L_copy_strict is null at τ∈{0.70,0.80,0.90,0.95} and L_copy_soft[k] is null for k∈{1,2,3} (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7277–7338). Δ̂ not provided (evaluation_pack.milestones.depth_fractions.delta_hat=null; …:9969–9980). Copy‑thresholds stability: "stability": "none"; norm_only_flags are null (…:7325–7348). Earliest strict copy at τ=0.70 and τ=0.95: null (…:7332–7338). The copy‑to‑semantics gap thus hinges entirely on semantic onset at the end of the stack.

### 4.2. Lens sanity: Raw‑vs‑Norm
Artifact risk is high: lens_artifact_score_v2=0.7426 (tier=high); js_divergence_p50=0.105, l1_prob_diff_p50=0.615; first_js_le_0.1=0; first_l1_le_0.5=0; jaccard_raw_norm_p50=0.316; first_jaccard_raw_norm_ge_0.5=null; pct_layers_kl_ge_1.0=0.321; n_norm_only_semantics_layers=1; earliest_norm_only_semantic=80 (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9960–9971; 7417–7470). Caution: semantics appears only under the norm lens at the final layer; treat early "semantics" as potentially lens‑induced and prefer rank/KL milestones.

Lens‑consistency at targets is low: at L_semantic_norm=80, jaccard@10=0.25, jaccard@50=0.299, spearman_top50=0.370 (…:8950–9020). This supports using rank/KL over absolute probabilities.

### 4.3. Tuned‑Lens analysis
Tuned lens is missing for this run (status="missing", path points to Qwen2.5‑72B tuned set; …:9232–9240, 9799–9804). Per measurement_guidance, prefer the norm lens for semantics; tuned‑lens attribution/rotation/temperature audits are not applicable.

### 4.4. KL, ranks, cosine, entropy milestones
- KL: first_kl_below_1.0=80 and first_kl_below_0.5=80 (…:7266–7267). Final‑head calibration is clean: last_layer_consistency.kl_to_final_bits≈0.00011; top1_agree=true (…:9005–9023).
- Ranks: first_rank_le_10=74, first_rank_le_5=78, first_rank_le_1=80 under the preferred norm lens (…:7248–7267). Uniform‑margin gate passes at L=80; Top‑2 gap gate not established; mark result as calibration‑sensitive (gate_stability_small_scale both_gates_pass_frac=0.0; …:7400–7415) and decoding‑point sensitive (…:8878–8936).
- Cosine: cos milestones (norm) reach ge_0.6 at layer 53 (…:7345–7355). cos_to_final at L=80 ≈1.0 (…-pure-next-token.csv:138).
- Entropy: non‑monotonic with large early entropy gap (entropy_gap_bits_p50=12.496) that collapses only at the end (…:9969–9998), aligning with rank/KL completion at L≈80.

### 4.5. Prism (shared‑decoder diagnostic)
Prism present/compatible (k=512; layers [embed,19,39,59]) with early KL reductions: ΔKL_p25≈3.16 bits, ΔKL_p50≈2.83 bits, ΔKL_p75≈−0.54 bits (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:852–890). Rank milestones under Prism were not established (null), and first_kl_le_1.0 did not improve (baseline=80, prism=null) (…:888–906). Verdict: Neutral (KL drop without demonstrated earlier rank‑1).

### 4.6. Ablation & stress tests
- No‑filler ablation: L_sem_orig=80, L_sem_nf=80; ΔL_sem=0 (…:9772–9781). No stylistic sensitivity.
- Negative/control: control_summary shows first_control_margin_pos=0 and max_control_margin=0.207 (…:9799–9819).
- Important‑word trajectory: in records, 'Berlin' enters the top‑k by mid‑70s and rises to rank‑1 by L=80, e.g., layer 74 includes 'Berlin' among candidates (001_layers_baseline/run-latest/output-Qwen2.5-72B-records.csv:3911) and becomes top‑1 at layer 80 (…:4106).

### 4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓
- LayerNorm bias removed ✓ (RMS model; bias not present)
- FP32 unembed promoted ✓ (use_fp32_unembed=true)
- Punctuation / markup anchoring noted ✓ (with_space variant; quotes/commas visible in top‑k)
- Copy‑reflex ✗ (no strict/soft copy milestone)
- Preferred lens honored ✓ (norm)
- Confirmed semantics reported ✗ (confirmed source=none)
- Dual‑lens artefact metrics cited ✓ (incl. score_v2, JS/Jaccard/L1)
- Tuned‑lens audit done n.a. (tuned missing)
- normalization_provenance present ✓ (ln_source @ L0/final)
- per‑layer normalizer effect present ✓ (resid_norm_ratio, delta_resid_cos)
- deterministic_algorithms true ✓
- numeric_health clean ✓
- copy_mask plausible ✓
- milestones.csv or evaluation_pack.citations used for quotes ✓
- gate_stability_small_scale reported ✓
- position_window stability reported ✓ (not measured here)

---
Produced by OpenAI GPT-5

*Run executed on: 2025-10-16 07:26:19*
