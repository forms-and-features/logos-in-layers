# Evaluation Report: 01-ai/Yi-34B

*Run executed on: 2025-10-13 22:23:35*

**Overview**
This evaluation reviews Yi‑34B (60 layers) run on 2025‑10‑13. The probe measures copy vs. semantics onset and tracks KL to final head, rank milestones, cosine alignment, and entropy trajectories, with dual‑lens (raw vs. norm) diagnostics and tuned‑lens calibration.

**Method sanity‑check**
- Prompt & indexing: "Give the city name only, plain text. The capital of Germany is called simply" [001_layers_baseline/run-latest/output-Yi-34B.json:808]. Positive rows exist for `prompt_id=pos, prompt_variant=orig` [001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:2].
- Normalizer provenance: `strategy = "next_ln1"` [001_layers_baseline/run-latest/output-Yi-34B.json:2862]; first layer `ln_source = blocks[0].ln1` [001_layers_baseline/run-latest/output-Yi-34B.json:2865-2867]; final unembed uses `ln_final` [001_layers_baseline/run-latest/output-Yi-34B.json:3777-3781].
- Per‑layer normalizer effect: early spike flagged (`flags.normalization_spike=true`) with spike‑shaped norm trajectory [001_layers_baseline/run-latest/output-Yi-34B.json:840,8026-8031]; e.g., L0 `resid_norm_ratio=6.1399`, `delta_resid_cos=0.5717` [001_layers_baseline/run-latest/output-Yi-34B.json:2869-2871].
- Unembed bias: `present=false`, `l2_norm=0.0` (cosines are bias‑free) [001_layers_baseline/run-latest/output-Yi-34B.json:834-840].
- Environment & determinism: `device=cpu`, `torch=2.8.0+cu128`, `deterministic_algorithms=true`, `seed=316` [001_layers_baseline/run-latest/output-Yi-34B.json:6080-6089].
- Numeric health: `any_nan=false`, `any_inf=false`, `layers_flagged=[]` [001_layers_baseline/run-latest/output-Yi-34B.json:3783-3789].
- Copy mask: `ignored_token_ids=[97,98,99,100,…]` (punct/markers plausible for tokenizer) [001_layers_baseline/run-latest/output-Yi-34B.json:972-980].
- Gold alignment: `ok=true`, `variant="with_space"`, ids match "▁Berlin" [001_layers_baseline/run-latest/output-Yi-34B.json:3868-3878]; `gold_alignment_rate=1.0` [001_layers_baseline/run-latest/output-Yi-34B.json:3917].
- Repeatability: skipped due to deterministic env (`status=skipped`) [001_layers_baseline/run-latest/output-Yi-34B.json:3790-3793,8016-8021].
- Norm trajectory: `shape="spike"`, `slope=0.074`, `r2=0.926`, `n_spikes=4` [001_layers_baseline/run-latest/output-Yi-34B.json:8026-8031].
- Measurement guidance: `prefer_ranks=true`, `suppress_abs_probs=true`, `preferred_lens_for_reporting="tuned"`, `use_confirmed_semantics=true` [001_layers_baseline/run-latest/output-Yi-34B.json:7963-7979].
- Semantic margin: `delta_abs=0.002`, `p_uniform=1.5625e-05`, `margin_ok_at_L_semantic_norm=true` [001_layers_baseline/run-latest/output-Yi-34B.json:7840-7846].
- Gate‑stability (small rescalings): `min_both_gates_pass_frac=1.0`; at L_semantic_norm=44, both gates pass fraction = 1.0 [001_layers_baseline/run-latest/output-Yi-34B.json:2658-2666].
- Position‑window: grid [0.2,0.4,0.6,0.8,0.92,0.98] with `rank1_frac=0.0` at L_semantic_norm=44 (position‑fragile) [001_layers_baseline/run-latest/output-Yi-34B.json:7874-7886].
- Micro‑suite: aggregates present (`n=5`, `n_missing=0`); medians `L_semantic_norm_median=44`, `L_semantic_confirmed_median=44` [001_layers_baseline/run-latest/output-Yi-34B.json:7948-7961,7951-7953].

**Quantitative findings (layer‑by‑layer)**
- L 0 — entropy 15.9623 bits, top‑1 'Denote' [001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:2].
- L 1 — entropy 15.9418 bits, top‑1 '.' [001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:3].
- L 2 — entropy 15.9320 bits, top‑1 '.' [001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:4].
- L 3 — entropy 15.8391 bits, top‑1 'MTY' [001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:5].
- L 44 — entropy 15.3273 bits, top‑1 'Berlin' [001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:46].
- L 60 — entropy 2.9812 bits, top‑1 'Berlin' [001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:63].

Bold semantic layer: L_semantic_confirmed = 44 (preferred lens tuned) [001_layers_baseline/run-latest/output-Yi-34B.json:7991-7995; 001_layers_baseline/run-latest/output-Yi-34B-milestones.csv:4].

Control margins: `first_control_margin_pos=1`, `max_control_margin=0.5836`, `first_control_strong_pos=42` [001_layers_baseline/run-latest/output-Yi-34B.json:6135-6141].

Micro‑suite (medians): `L_semantic_confirmed_median=44`, IQR for `L_semantic_norm` = [42,44]; example fact 'France→Paris' cites row 163 [001_layers_baseline/run-latest/output-Yi-34B.json:7951-7958,7910-7912].

Entropy drift: large teacher‑gap medians `p50=12.5865` bits (p25=12.2860, p75=12.7751) [001_layers_baseline/run-latest/output-Yi-34B.json:8032-8035].

**Qualitative findings**

4.1. Copy vs semantics (Δ‑gap)
No copy reflex was detected under strict or soft criteria: `L_copy_strict` at τ∈{0.70,0.80,0.90,0.95} are null and soft k∈{1,2,3} are null, stability "none" [001_layers_baseline/run-latest/output-Yi-34B.json:2574-2600]. The semantic onset occurs at layer 44/60 (fraction ≈0.733) [001_layers_baseline/run-latest/output-Yi-34B.json:2619-2626,7991-7995]. Copy‑thresholds stability: `stability="none"`; earliest strict copy at τ=0.70 and τ=0.95 are null; `norm_only_flags[τ]` are null [001_layers_baseline/run-latest/output-Yi-34B.json:2581-2599].

4.2. Lens sanity: Raw‑vs‑Norm
Dual‑lens artefact risk is high: `lens_artifact_score_v2=0.9434` (tier=high) [001_layers_baseline/run-latest/output-Yi-34B.json:2854-2858,8002-8015]. Symmetric metrics show large divergences: `js_divergence_p50=0.3687`, `l1_prob_diff_p50=1.0891`, with no early convergence (`first_js_le_0.1=0`, `first_l1_le_0.5=0`) [001_layers_baseline/run-latest/output-Yi-34B.json:8003-8008]. Top‑K overlap is low (`jaccard_raw_norm_p50=0.1111`; first ≥0.5 at layer 1) [001_layers_baseline/run-latest/output-Yi-34B.json:8009-8010]. Prevalence: `pct_layers_kl_ge_1.0=0.6557`, `n_norm_only_semantics_layers=14`, `earliest_norm_only_semantic=44` [001_layers_baseline/run-latest/output-Yi-34B.json:8011-8014,2698-2706,2701-2706]. At the semantic target, lens‑consistency is weak across views: norm vs raw `jaccard@10=0.0526`, `jaccard@50=0.1905`, `spearman_top50=0.4614` [001_layers_baseline/run-latest/output-Yi-34B.json:3816-3832]. Caution: early semantics may be lens‑induced; prefer rank milestones and confirmed semantics.

4.3. Tuned‑Lens analysis
Preference: tuned lens is not calibration‑only and is preferred for semantics (`tuned_is_calibration_only=false`, `preferred_semantics_lens_hint="tuned"`) [001_layers_baseline/run-latest/output-Yi-34B.json:7815-7816,8066-8068]. Attribution: KL reductions attribute mainly to rotation (ΔKL_rot_p50=3.50 bits) with smaller temperature effects (ΔKL_temp_p50≈−0.66), interaction ≈3.33 [001_layers_baseline/run-latest/output-Yi-34B.json:8038-8046,7818-7834]. Rank earliness: tuned reaches `first_rank_le_1=44` with `first_rank_le_{5,10}=44` [001_layers_baseline/run-latest/output-Yi-34B.json:4246-4255]. Positional generalization: `pos_ood_ge_0.96=6.4468`, `pos_in_dist_le_0.92=5.5936`, `pos_ood_gap=0.8532` [001_layers_baseline/run-latest/output-Yi-34B.json:8048-8059,7806-7808]. Head mismatch is negligible (`tau_star_modelcal=1.0`; `kl_bits_tuned_final` unchanged after τ⋆) [001_layers_baseline/run-latest/output-Yi-34B.json:8061-8066,7810-7814]. Last‑layer agreement is excellent: `top1_agree=true`, `kl_to_final_bits≈0` [001_layers_baseline/run-latest/output-Yi-34B.json:3880-3890]. Baseline (norm) is reported alongside, but tuned is used for semantic onset and ranks per guidance.

4.4. KL, ranks, cosine, entropy milestones
KL: baseline `first_kl_below_1.0=60`, `first_kl_below_0.5=60` [001_layers_baseline/run-latest/output-Yi-34B.json:2515-2516]. Final KL is ≈0 (last‑layer consistency) [001_layers_baseline/run-latest/output-Yi-34B.json:3880-3890]. Ranks (preferred lens tuned): `first_rank_le_{10,5,1}=44` (baseline parenthetical via prism shows no change) [001_layers_baseline/run-latest/output-Yi-34B.json:4246-4255,7810-7837]. Cosine milestones (norm): `ge_0.2 → L1`, `ge_0.4 → L44`, `ge_0.6 → L51` [001_layers_baseline/run-latest/output-Yi-34B.json:2612-2617]. Entropy: large teacher‑gap median (p50≈12.59 bits) and non‑monotonic with a spike‑shaped norm trajectory [001_layers_baseline/run-latest/output-Yi-34B.json:8032-8035,8026-8031]. Uniform‑margin gate passes at L=44; Top‑2 gate is not reported; stability (run‑of‑two) holds (`L_semantic_run2=44`) [001_layers_baseline/run-latest/output-Yi-34B.json:7840-7857,2645-2647]. Position‑window stability is low (`rank1_frac=0.0`), so claims are position‑fragile [001_layers_baseline/run-latest/output-Yi-34B.json:7874-7886].

4.5. Prism
Prism is present and compatible; KL drops across depth (p50 Δ≈1.36 bits), but rank milestones are unchanged (`le_1`/`le_5`/`le_10` remain null shifts) and `first_kl_le_1.0` unchanged at 60 [001_layers_baseline/run-latest/output-Yi-34B.json:7819-7837,7809-7817]. Verdict: Neutral (KL improves without earlier ranks).

4.6. Ablation & stress tests
No‑filler ablation leaves semantics unchanged: `L_sem_orig=44`, `L_sem_nf=44` (`ΔL_sem=0`) [001_layers_baseline/run-latest/output-Yi-34B.json:6101-6108]. Control summary shows early positive control margin (`first_control_margin_pos=1`) with strong control by L42 [001_layers_baseline/run-latest/output-Yi-34B.json:6135-6141]. Negative/control prompt "Berlin is the capital of" produces ' Germany' as top‑1 (and no 'Berlin' in top‑10) [001_layers_baseline/run-latest/output-Yi-34B.json:12-38]. In the positive row at L44, 'Germany' appears within top‑5 alongside 'Berlin' [001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:46].

4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓
- LayerNorm bias removed ✓
- FP32 unembed promoted ✓
- Punctuation / markup anchoring noted ✗
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
