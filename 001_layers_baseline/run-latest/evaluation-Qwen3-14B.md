# Evaluation Report: Qwen/Qwen3-14B

*Run executed on: 2025-10-16 07:26:19*
1. **Overview**
- Qwen3-14B; run timestamp in folder: `timestamp-20251016-0726` (16 Oct). Probe measures copy vs semantic onset, KL/rank/cosine/entropy trajectories, normalization provenance, raw-vs-norm lens artefacts, tuned-lens audit, and stability gates. Source: `001_layers_baseline/run-latest/output-Qwen3-14B.json`.

2. **Method sanity‑check**
- Prompt & indexing: "Give the city name only, plain text. The capital of Germany is called simply" [001_layers_baseline/run-latest/output-Qwen3-14B.json:6]. Positive rows exist: `Germany→Berlin,0,pos,orig,0,...` [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:2].
- Normalizer provenance: `arch="pre_norm"`, `strategy.primary="next_ln1"` [001_layers_baseline/run-latest/output-Qwen3-14B.json:7390]. First layer ln_source `blocks[0].ln1`; final stack entry `ln_final` [001_layers_baseline/run-latest/output-Qwen3-14B.json:7749] and [001_layers_baseline/run-latest/output-Qwen3-14B.json:8008].
- Per‑layer normalizer effect: early `resid_norm_ratio=0.496` and `delta_resid_cos=0.990` at L0 [001_layers_baseline/run-latest/output-Qwen3-14B.json:7390]; normalization spike flagged `true` [001_layers_baseline/run-latest/output-Qwen3-14B.json:839].
- Unembed bias: `present=false`, `l2_norm=0.0` (cosines bias‑free) [001_layers_baseline/run-latest/output-Qwen3-14B.json:834].
- Environment & determinism: `device="cpu"`, torch 2.8, `deterministic_algorithms=true`, `seed=316` [001_layers_baseline/run-latest/output-Qwen3-14B.json:10396].
- Repeatability (forward‑of‑two): mode `skipped_deterministic`; pass1.layer=36, pass2.layer=null, `gate.repeatability_forward_pass=null` [001_layers_baseline/run-latest/output-Qwen3-14B.json:8178]. Caution: forward‑of‑two not executed.
- Decoding‑point ablation (pre‑norm): `gate.decoding_point_consistent=false` [001_layers_baseline/run-latest/output-Qwen3-14B.json:8109]. At target `L_semantic_norm` L=36: `rank1_agree=true`, `jaccard@10=0.25` [001_layers_baseline/run-latest/output-Qwen3-14B.json:8050].
- Numeric health: `any_nan=false`, `any_inf=false`, `layers_flagged=[]` [001_layers_baseline/run-latest/output-Qwen3-14B.json:8016].
- Copy mask: `size=6112`, sample ignored strings `["!","\"","#","$","%","&","'","(",")","*","+",",","-",".","/",":"]` [001_layers_baseline/run-latest/output-Qwen3-14B.json:7068].
- Gold alignment: `ok=true`, pieces `["ĠBerlin"]` [001_layers_baseline/run-latest/output-Qwen3-14B.json:8166]. `gold_alignment_rate=1.0` [001_layers_baseline/run-latest/output-Qwen3-14B.json:8235].
- Repeatability (decode micro‑check §1.39): `max_rank_dev=0.0`, `p95_rank_dev=0.0`, `top1_flip_rate=0.0` [001_layers_baseline/run-latest/output-Qwen3-14B.json:8023].
- Norm trajectory: `shape="spike"`, `slope=0.1107`, `r2=0.9046` [001_layers_baseline/run-latest/output-Qwen3-14B.json:9999].
- Measurement guidance: `prefer_ranks=true`, `suppress_abs_probs=true`, `preferred_lens_for_reporting="tuned"`, `use_confirmed_semantics=true` [001_layers_baseline/run-latest/output-Qwen3-14B.json:12260].
- Semantic margin: `delta_abs=0.002`, `p_uniform=6.58e-06`, `margin_ok_at_L_semantic_norm=true` [001_layers_baseline/run-latest/output-Qwen3-14B.json:12137].
- Gate‑stability: `min_both_gates_pass_frac=1.0` at `L_semantic_norm=36` (both gates=1.0) [001_layers_baseline/run-latest/output-Qwen3-14B.json:7229].
- Position‑window: grid `[0.2,0.4,0.6,0.8,0.92,0.98]`, `rank1_frac=0.0` at L_sem (position‑fragile) [001_layers_baseline/run-latest/output-Qwen3-14B.json:9999].
- Micro‑suite: present; `n=5`, `L_semantic_confirmed_median=36`, `n_missing=0` [001_layers_baseline/run-latest/output-Qwen3-14B.json:12380] and [001_layers_baseline/run-latest/output-Qwen3-14B.json:12429].

3. **Quantitative findings (layer‑by‑layer)**
- L 0 — entropy 17.213 bits, top‑1 '梳' [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:2].
- L 32 — entropy 0.816 bits, top‑1 '____' (answer rank ≤10) [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:34].
- L 33 — entropy 0.481 bits, top‑1 '____' (answer rank 4) [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:35].
- L 36 — entropy 0.312 bits, top‑1 ' Berlin' (is_answer=True) [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:38].
- Bolded semantic layer: L 36 — L_semantic_confirmed (margin_ok), decoding‑point sensitive (pre‑norm gate=false) [001_layers_baseline/run-latest/output-Qwen3-14B.json:12288] and [001_layers_baseline/run-latest/output-Qwen3-14B.json:8109].
- Control: `first_control_margin_pos=0`, `max_control_margin=0.974`, `first_control_strong_pos=36` [001_layers_baseline/run-latest/output-Qwen3-14B.json:10452].
- Micro‑suite: median `L_semantic_confirmed=36` (example fact row index 36 for Germany→Berlin) [001_layers_baseline/run-latest/output-Qwen3-14B.json:12380].
- Entropy drift: `entropy_gap_bits_p50=13.40` bits (vs teacher) [001_layers_baseline/run-latest/output-Qwen3-14B.json:12328].
- Confidence margins at L 36: `answer_logit_gap≈3.14`, `resid_norm_ratio≈0.234`, `delta_resid_cos≈0.733` [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:38].

4. **Qualitative findings**

4.1. Copy vs semantics (Δ‑gap)
No strict copy onset is detected (all τ∈{0.70,0.80,0.90,0.95} null) [001_layers_baseline/run-latest/output-Qwen3-14B.json:7153]. Early layers 0–3 do not show `copy_collapse` or `copy_soft_k1@0.5` hits [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:2]. With no copy onset, Δ̂ is not defined (`delta_hat=null`) [001_layers_baseline/run-latest/output-Qwen3-14B.json:12292]. Stability tag: `copy_thresholds.stability="none"` [001_layers_baseline/run-latest/output-Qwen3-14B.json:7153].

4.2. Lens sanity: Raw‑vs‑Norm
Artifact risk is high: `lens_artifact_score_v2=0.7037` (tier=high) with `js_divergence_p50=0.513`, `l1_prob_diff_p50=1.432`, `first_js_le_0.1=0`, `first_l1_le_0.5=0`, `jaccard_raw_norm_p50=0.25`, `first_jaccard_raw_norm_ge_0.5=0` [001_layers_baseline/run-latest/output-Qwen3-14B.json:12276]. Prevalence: `pct_layers_kl_ge_1.0=0.756`, `n_norm_only_semantics_layers=0` [001_layers_baseline/run-latest/output-Qwen3-14B.json:12276]. At the semantic target, norm vs raw lens consistency is modest (`jaccard@10=0.333`, `spearman_top50≈0.342`) [001_layers_baseline/run-latest/output-Qwen3-14B.json:8116]. Caution: early semantics should be treated as view‑dependent; prefer rank/KL milestones and confirmed semantics.

4.3. Tuned‑Lens analysis
Tuned lens is not “calibration‑only” (`tuned_is_calibration_only=false`) and is preferred for semantics (`preferred_semantics_lens_hint="tuned"`) [001_layers_baseline/run-latest/output-Qwen3-14B.json:12028]. Attribution (ΔKL, tuned vs baseline): rotation dominates with `delta_kl_rot_p50≈1.97` and large interaction `≈3.22`; temperature effect near zero [001_layers_baseline/run-latest/output-Qwen3-14B.json:12028]. Rank earliness shifts later under tuned (`first_rank_le_1: 39` vs baseline 36; `le_5: 34` vs 33; `le_10: 33` vs 32) [001_layers_baseline/run-latest/output-Qwen3-14B.json:12028]. Positional generalization: `pos_ood_ge_0.96≈5.01`, `pos_in_dist_le_0.92≈5.51`, gap −0.49 [001_layers_baseline/run-latest/output-Qwen3-14B.json:12028]. Head calibration matches (`kl_bits_tuned_final=0.0`, `tau_star_modelcal=1.0`) and last‑layer agreement holds (`kl_to_final_bits=0.0`, `top1_agree=true`) [001_layers_baseline/run-latest/output-Qwen3-14B.json:12028] and [001_layers_baseline/run-latest/output-Qwen3-14B.json:8199].

4.4. KL, ranks, cosine, entropy milestones
KL: `first_kl_below_1.0=40` and `first_kl_below_0.5=40`; final KL≈0 (last‑layer consistent) [001_layers_baseline/run-latest/output-Qwen3-14B.json:12028] and [001_layers_baseline/run-latest/output-Qwen3-14B.json:8199]. Ranks (preferred lens=tuned): `first_rank_le_10=33` (baseline 32), `le_5=34` (baseline 33), `le_1=39` (baseline 36) [001_layers_baseline/run-latest/output-Qwen3-14B.json:12028]. Cosine (norm lens): `ge_0.2=5`, `ge_0.4=29`, `ge_0.6=36` [001_layers_baseline/run-latest/output-Qwen3-14B.json:7191]. Entropy drift is large (`p50≈13.40` bits) [001_layers_baseline/run-latest/output-Qwen3-14B.json:12328]. Margin gate: uniform passes at L 36 (`margin_ok_at_L_semantic_norm=true`) [001_layers_baseline/run-latest/output-Qwen3-14B.json:12137]; top‑2 gate not established (null). Stability: run‑of‑two check skipped; treat single‑run onset as potentially unstable; gate‑stability small‑scale passes (1.0) but position window is fragile (`rank1_frac=0.0`).

4.5. Prism (shared‑decoder diagnostic)
Present/compatible [001_layers_baseline/run-latest/output-Qwen3-14B.json:845]. KL deltas indicate higher KL under prism at p50 (`baseline=12.978`, `prism=13.228`, Δ≈+0.25 bits) and no earlier rank milestones [001_layers_baseline/run-latest/output-Qwen3-14B.json:845]. Verdict: Regressive.

4.6. Ablation & stress tests
No‑filler ablation: `L_sem_orig=36`, `L_sem_nf=36` (ΔL_sem=0) [001_layers_baseline/run-latest/output-Qwen3-14B.json:10418]. Control prompt check: test prompt "Berlin is the capital of" shows top‑1 ' Germany' [001_layers_baseline/run-latest/output-Qwen3-14B.json:12]. Important‑word trajectory: 'Berlin' appears in top‑k by L32–L34 (e.g., presence at L32 [001_layers_baseline/run-latest/output-Qwen3-14B-records.csv:609]; L33 [001_layers_baseline/run-latest/output-Qwen3-14B-records.csv:625]; strong by L34 [001_layers_baseline/run-latest/output-Qwen3-14B-records.csv:640]).

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
- milestones.csv or evaluation_pack.citations used ✓
- gate_stability_small_scale reported ✓
- position_window stability reported ✓

---
Produced by OpenAI GPT-5
