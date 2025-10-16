# Evaluation Report: meta-llama/Meta-Llama-3-8B

*Run executed on: 2025-10-16 07:26:19*

1. **Overview**
Meta-Llama-3-8B was probed on 2025-10-16 to analyze when copy behavior yields to semantic prediction. The run tracks layer-wise ranks, KL-to-final, cosine geometry, and entropy, with diagnostics for normalization provenance, lens artefacts (raw vs norm), tuned-lens audits, and stability gates.

2. **Method sanity-check**
- Prompt & indexing: "Give the city name only, plain text. The capital of Germany is called simply" (no trailing space) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:12]. Positive rows exist: `prompt_id=pos`, `prompt_variant=orig` (e.g., layer 0 and 25) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:2, 29].
- Normalizer provenance: `arch="pre_norm"`; `strategy.primary="next_ln1"`, `strategy.ablation="post_ln2_vs_next_ln1@targets"` [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7225]. First/last sources: L0 `ln_source="blocks[0].ln1"` [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7238] and final `ln_source="ln_final"` [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7596].
- Per-layer normalizer effect: early `resid_norm_ratio`/`delta_resid_cos` spikes are flagged by `flags.normalization_spike=true` [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:838], with L0 `resid_norm_ratio=18.19`, `delta_resid_cos=0.53` [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7242].
- Unembed bias: `present=false`, `l2_norm=0.0` (cosine metrics are bias-free) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:832].
- Environment & determinism: `device="cpu"`, torch `2.8.0+cu128`, `deterministic_algorithms=true`, `seed=316` [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:10177].
- Repeatability (forward-of-two): mode `skipped_deterministic`; `pass1.layer=25`, `pass2.layer=null`, `delta_layers=null`, `topk_jaccard_at_primary_layer=null`, `gate.repeatability_forward_pass=null` [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7954]. Caution: gate not evaluated due to deterministic skip.
- Decoding-point ablation (pre-norm): `gate.decoding_point_consistent=true`; at target `L_semantic_norm` (layer 25): `rank1_agree=true`, `jaccard@10=0.538` [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7803, 7832].
- Numeric health: `any_nan=false`, `any_inf=false`, `layers_flagged=[]` [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7788].
- Copy mask: `size=6022`; sample includes punctuation tokens (e.g., `(`, `)`, `*`, `+`, `,`, `-`, `.`, `/`, `:`) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:6983, 6978]. Plausible for tokenizer.
- Gold alignment: `ok=true`, `variant="with_space"`, `first_id=20437` for "ĠBerlin" [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7942]. `gold_alignment_rate=1.0` [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:8011].
- Repeatability (decode micro‑check §1.39) and forward‑of‑two (§1.54): `max_rank_dev=0.0`, `p95_rank_dev=0.0`, `top1_flip_rate=0.0` [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7795].
- Norm trajectory: `shape="spike"`, `slope=0.113`, `r2=0.953`, `n_spikes=18` [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:12098].
- Measurement guidance: `prefer_ranks=true`, `suppress_abs_probs=true`; reasons include `norm_only_semantics_window`, `high_lens_artifact_risk`, `normalization_spike`, `rank_only_near_uniform`, `low_lens_consistency_at_semantic`, `pos_window_low_stability`, `scale_sensitive_semantic_gate`; `preferred_lens_for_reporting="tuned"`, `use_confirmed_semantics=true` [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:12029].
- Semantic margin: `delta_abs=0.002`, `p_uniform≈0`, `margin_ok_at_L_semantic_norm=false`, `L_semantic_margin_ok_norm=32` [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7113].
- Gate‑stability (small rescalings): `min_both_gates_pass_frac=0.0` and at target (L25) both passes 0.0 → calibration‑sensitive [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7132].
- Position‑window: grid `[0.2, 0.4, 0.6, 0.8, 0.92, 0.98]`, `rank1_frac=0.0` at L_sem → position‑fragile [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11940].
- Micro‑suite: aggregates present (`n=5`, `L_semantic_confirmed_median=25`, `n_missing=0`); example facts include Germany→Berlin (row 25) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:12198].

3. **Quantitative findings (layer‑by‑layer)**
- L 0 — entropy 16.96 bits, top‑1 ‘itzer’ [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:2].
- L 24 — entropy 16.83 bits, top‑1 ‘capital’ [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:28].
- L 25 — entropy 16.81 bits, top‑1 ‘Berlin’, answer_rank=1 [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:29].
- L 32 — entropy 2.961 bits, top‑1 ‘Berlin’, answer_rank=1 [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:36].

Bolded semantic layer: L 25 — L_semantic_confirmed (confirmed under raw; guidance prefers tuned lens for reporting) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-milestones.csv:4; 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:12029, 12141].

Controls: `first_control_margin_pos=0`, `max_control_margin=0.5186`, `first_control_strong_pos=25` [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:10229].

Micro‑suite: median `L_semantic_confirmed=25` (n=5). Example citation: Germany→Berlin row 25 [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:12198].

Entropy drift: entropy gap percentiles vs teacher at p25/p50/p75 are 13.87/13.88/13.91 bits [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7050].

Confidence snapshot: at L 25, `answer_logit_gap=0.4075` [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:29]. Normalizer snapshots: L0 `resid_norm_ratio=18.19`, L25 `1.50`, final `2.25` [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7242, 7559, 7599].

4. **Qualitative findings**

4.1. **Copy vs semantics (Δ‑gap)**
No early copy‑reflex: layers 0–3 have `copy_collapse=False` and `copy_soft_k1@0.5=False` (e.g., L0 row) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:2]. `L_copy_strict` is null for τ∈{0.70,0.95} and `stability="none"` [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7060]. Δ̂ is not provided (`depth_fractions.delta_hat=null`) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:12063]. Earliest strict copy at τ=0.70 and τ=0.95: null; `norm_only_flags` null [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7067].

4.2. **Lens sanity: Raw‑vs‑Norm**
Artifact metrics: `lens_artifact_score_v2=0.459` (tier=medium) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7279]. Symmetric metrics: `js_divergence_p50=0.0168`, `l1_prob_diff_p50=0.2403`, `first_js_le_0.1=0`, `first_l1_le_0.5=0` [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7197]. Top‑K overlap: `jaccard_raw_norm_p50=0.408`, `first_jaccard_raw_norm_ge_0.5=3` [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7205]. Prevalence: `n_norm_only_semantics_layers=5`, `earliest_norm_only_semantic=25` [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7187]. Lens‑consistency at semantic target is low: at L25 `jaccard@10=0.25`, `spearman_top50=0.181` [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7868]. Caution: early semantics near L25 may be view‑dependent; rely on ranks/KL and confirmed semantics.

4.3. **Tuned‑Lens analysis**
Preference: `tuned_is_calibration_only=false`; guidance hints `preferred_semantics_lens_hint="tuned"` [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:12133, 12029]. Attribution: ΔKL (tuned vs components) p25/p50/p75 shows rotation dominates (e.g., `delta_kl_rot_p50=2.56`, `delta_kl_temp_p50≈-0.04`, `delta_kl_interaction_p50=2.34`) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:12070]. Rank earliness: tuned is later or unchanged for the baseline fact (`first_rank_le_{10,5,1}=32` vs norm `24/25/25`) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:10250, 6999]. Positional generalization: `pos_ood_ge_0.96=4.87`, `pos_in_dist_le_0.92=5.64`, `pos_ood_gap=-0.77` (audit present) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:12086]. Head mismatch: `tau_star_modelcal=1.0`, `kl_bits_tuned_final=0.0` → after τ* remains `0.0` [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:12130]. Final‑head agreement is clean: `top1_agree=true`, `kl_to_final_bits=0.0`, `warn_high_last_layer_kl=false` [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7974].

4.4. **KL, ranks, cosine, entropy milestones**
KL: `first_kl_below_1.0=32` [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:6998]; final KL≈0 (last‑layer consistency) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7974]. Ranks (norm lens baseline): `first_rank_le_10=24`, `first_rank_le_5=25`, `first_rank_le_1=25` [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7001]. Cosine milestones (norm): ge_0.2=20, ge_0.4=30, ge_0.6=32 [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7094]. Entropy gap grows early and compresses late (p25/p50/p75 ≈ 13.87/13.88/13.91 bits) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7050]. Margin gate reminder: `margin_ok_at_L_semantic_norm=false` and small‑scale gate pass‑fraction=0.0 at L25; treat single‑layer onsets as calibration‑sensitive and lean on confirmed semantics [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7117, 7132].

4.5. **Prism (shared‑decoder diagnostic)**
Present/compatible (`k=512`). KL deltas at sampled depths are negative vs baseline (e.g., p50 delta ≈ −8.29 bits), with no earlier rank milestones (`le_{10,5,1}=null`) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:835–868, 872–897]. Verdict: Regressive (KL increases and no earlier ranks).

4.6. **Ablation & stress tests**
Ablation summary: `L_copy_orig=null`, `L_sem_orig=25`, `L_copy_nf=null`, `L_sem_nf=25`, `ΔL_sem=0` [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:10200]. Control prompt: "Berlin is the capital of" → top‑1 ‘ Germany’ in the decode sample [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:18]. Important‑word trajectory: ‘Germany’ surfaces at multiple prompt positions across depths (e.g., rows 24, 41, 76 in records.csv), while ‘Berlin’ becomes top‑1 at the next‑token position by L25 [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-records.csv:24, 41, 76; 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:29].

4.7. **Checklist (✓/✗/n.a.)**
- RMS lens ✓ [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:816].
- LayerNorm bias removed ✓ (`layernorm_bias_fix=not_needed_rms_model`) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:814].
- FP32 unembed promoted ✓ (`mixed_precision_fix=casting_to_fp32_before_unembed`) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:818].
- Punctuation / markup anchoring noted ✓ (copy mask includes punctuation) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:6978].
- Copy‑reflex ✗ (none at layers 0–3) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:2–5].
- Preferred lens honored ✓ (reporting references tuned preference; confirmed semantics cited) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:12029, 12141].
- Confirmed semantics reported ✓ (L25, source=raw) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:9672].
- Dual‑lens artefact metrics cited ✓ (`lens_artifact_score_v2`, JS/Jaccard/L1) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7197, 7205, 7279].
- Tuned‑lens audit done ✓ (rotation/temp/positional/head) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:12070].
- normalization_provenance present ✓ (ln_source @ L0/final) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7238, 7596].
- per‑layer normalizer effect present ✓ (`resid_norm_ratio`, `delta_resid_cos`) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7238].
- deterministic_algorithms true ✓ [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:10186].
- numeric_health clean ✓ [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7788].
- copy_mask plausible ✓ [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:6978].
- milestones.csv used for quotes ✓ [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-milestones.csv:3–4].
- gate_stability_small_scale reported ✓ [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7132].
- position_window stability reported ✓ [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11940].

---
Produced by OpenAI GPT-5
