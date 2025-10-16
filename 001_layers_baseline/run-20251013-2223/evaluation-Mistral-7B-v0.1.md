# Evaluation Report: mistralai/Mistral-7B-v0.1

*Run executed on: 2025-10-13 22:23:35*

**Overview**
- Model: mistralai/Mistral-7B-v0.1 (32 layers). Run date: 2025-10-13 22:23:35 (`timestamp-20251013-2223`).
- Probe measures copy-reflex vs semantic onset via ranks/KL/cosine/entropy across layers, under norm lens with lens diagnostics (raw-vs-norm, artifact v2), plus tuned/prism audits and stability checks.

**Method sanity-check**
- Prompt & indexing: context ends with “called simply” (no trailing space): "Give the city name only, plain text. The capital of Germany is called simply" (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4). Positive rows with `prompt_id=pos`, `prompt_variant=orig` present (e.g., row 2 in 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv).
- Normalizer provenance: arch="pre_norm", strategy="next_ln1" (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2434–2436); ln source L0="blocks[0].ln1" (2439) and final uses "ln_final" (2727).
- Per-layer normalizer effect: normalization spike flagged ("normalization_spike": true, 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:839–843). Norm trajectory shape="spike" with high fit (slope=0.1476, r2=0.9628, n_spikes=26; 7155–7159). Early resid_norm_ratio large (e.g., L0=115.17, L2=258.96; 2442–2463).
- Unembed bias: present=false, l2_norm=0.0 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:834–838). Cosines are bias-free.
- Environment & determinism: device=cpu, torch=2.8.0+cu128, deterministic_algorithms=true, seed=316 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:5236–5247). Reproducible.
- Numeric health: any_nan=false, any_inf=false, layers_flagged=[] (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2936–2942).
- Copy mask: size=1179; sample includes whitespace/punctuation (e.g., "\t", "\n", "", "!", '"', "#", "$", "%", "&") (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2125–2143). Plausible for tokenizer filtering.
- Gold alignment: ok, variant=with_space, pieces=["▁Berlin"], answer_ids=[8430] (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3021–3031).
- Repeatability: skipped due to deterministic_env (max_rank_dev=null, p95_rank_dev=null, top1_flip_rate=null; 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:7145–7149). Note: not a high-variance run; just unmeasured.
- Norm trajectory: shape="spike" (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:7155–7159).
- Measurement guidance: prefer_ranks=true; suppress_abs_probs=true; reasons include "norm_only_semantics_window", "high_lens_artifact_risk", "normalization_spike", "pos_window_low_stability", "scale_sensitive_semantic_gate"; preferred_lens_for_reporting="tuned"; use_confirmed_semantics=true (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:7092–7107).
- Semantic margin: delta_abs=0.002, p_uniform=3.125e-05; margin_ok_at_L_semantic_norm=true; L_semantic_confirmed_margin_ok_norm=25 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:6969–6976).
- Gate-stability (small rescalings): at L_semantic_norm=25, uniform_margin_pass_frac=1.0 but top2_gap_pass_frac=0.0; both_gates_pass_frac=0.0; min_both_gates_pass_frac=0.0 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2295–2303). Calibration-sensitive.
- Position-window: grid=[0.2,0.4,0.6,0.8,0.92,0.98], L_semantic_norm=25, rank1_frac=0.0 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:7003–7015). Position-fragile.
- Micro-suite: aggregates present (n=5, L_semantic_confirmed_median=25, L_semantic_norm_median=24, n_missing=1; 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:6988–7002, 7076–7088). Example fact citation row_index: Germany→Berlin row_index=25 (7021–7029).

**Quantitative findings (layer-by-layer)**
- L 0 — entropy 14.961 bits, top‑1 ‘dabei’ [row 2 in 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv].
- L 8 — entropy 14.821 bits, top‑1 ‘[…]’ [row 10 in 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv].
- L 22 — entropy 14.387 bits, top‑1 ‘“’ (quotes cluster) [row 24 in 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv].
- L 24 — entropy 14.212 bits, top‑1 ‘simply’ [row 26 in 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv].
- L 25 — entropy 13.599 bits, top‑1 ‘Berlin’, answer_rank=1; cos_to_final=0.4245, cos_to_answer=0.1536; answer_logit_gap≈0.0146 [row 27 in 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv].
- L 32 — entropy 3.611 bits, top‑1 ‘Berlin’, answer_rank=1; final-head KL=0.0 (consistency) [row 34 in 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv; 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3033–3057].

Control margins: first_control_margin_pos=2; max_control_margin=0.6539; first_control_strong_pos=24 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:5292–5297).

Micro‑suite: median L_semantic_confirmed=25 (median L_semantic_norm=24); Δ̂ median not available; example fact Germany→Berlin (row_index=25) (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:6988–7002, 7018–7030).

Entropy drift: entropy_gap_bits p25/p50/p75 = 10.60 / 10.99 / 11.19 bits (evaluation aggregates; 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:7161–7165). Teacher entropy≈3.611 bits (2349), aligning with final layer entropy (4863–4864).

Normalizer snapshots: resid_norm_ratio near L_semantic (e.g., L25 resid_norm_ratio=9.97; delta_resid_cos=0.902; 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2667–2733).

**Qualitative findings**

4.1. Copy vs semantics (Δ‑gap)
No strict or soft copy onset detected in this fact (all `L_copy_strict@{0.70,0.95}` and `L_copy_soft[k]` are null; summary copy_thresholds stability="none"; 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2211–2237). Earliest soft copy across other facts occurs at k1=26 for one fact (micro-suite; 5818–5873), but not for the baseline fact. With L_semantic_norm=25 and no L_copy variant, Δ̂ is not defined for this fact; evaluation_pack depth fractions report semantic_frac≈0.781 and no copy_frac (7115–7129, 2256–2266). Copy‑reflex: not observed in layers 0–3 for this baseline (rows 2–5 in 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv). Copy thresholds: L_copy_strict@0.70=null, @0.95=null; norm_only_flags are null at all τ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:5377–5396, 5508–5526, 5639–5658). Stability tag: "none".

4.2. Lens sanity: Raw‑vs‑Norm
Artifact scores are high: lens_artifact_score=0.545 and lens_artifact_score_v2=0.670 (tier="high"), with js_divergence_p50=0.074, l1_prob_diff_p50=0.505; first_js_le_0.1 at layer 0; first_l1_le_0.5 at layer 0 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2430–2348). Top‑K overlap median jaccard@50=0.408; first_jaccard_raw_norm_ge_0.5 at layer 19 (2350–2356). Prevalence: pct_layers_kl_ge_1.0=0.242; n_norm_only_semantics_layers=1; earliest_norm_only_semantic=32 (2331–2335). Caution: early semantics near candidates may be lens‑induced; prefer rank milestones and confirmed semantics. Lens consistency at semantic target is moderate (norm vs raw at L=25: jaccard@10=0.429, jaccard@50=0.471, spearman_top50≈0.504; 2963–2967). Under norm vs tuned at L=25: jaccard@50≈0.266, spearman_top50≈0.375 (2996–3016), indicating view‑dependence.

4.3. Tuned‑Lens analysis
Guidance prefers tuned lens and confirmed semantics (preferred_lens_for_reporting="tuned"; tuned_is_calibration_only=false; 7092–7107, 6944–6946). Attribution: ΔKL_rot dominates over temperature at p25/p50 (ΔKL_rot_p25=1.989, ΔKL_rot_p50=2.212; ΔKL_temp_p25=−0.406, ΔKL_temp_p50=−0.263; interaction_p50=2.134; 7167–7176). Rank earliness is limited: first_rank_le_{10,5,1} (tuned) = {25,25,32} vs baseline norm {22,24,25} (5303–5316, 5442–5446), suggesting tuned shifts rank‑1 later to 32 while leaving le_5 similar. Positional generalization: pos_ood_ge_0.96=4.144, pos_in_dist_le_0.92=4.911, gap=−0.767 (7166–7189), indicating weaker OOD position behavior. Head mismatch: kl_bits_tuned_final=0.0; tau_star_modelcal=1.0 (7190–7194). Last‑layer consistency is perfect (kl_to_final_bits=0, top1_agree=true; 3033–3057).

4.4. KL, ranks, cosine, entropy milestones
- KL thresholds: first_kl_below_1.0=27 (norm) and 26 (tuned summary #1), first_kl_below_0.5=32 in all views (5442–5444, 5311–5313). Final KL≈0 (3033–3057).
- Ranks: first_rank_le_{10,5,1} (norm) = {22,24,25}; tuned = {25,25,32} (5444–5446; 5313–5315). Margin gate: uniform‑margin passes at L=25 (6969–6976); top‑2 gap gate fails at L=25 and is scale‑sensitive (both_gates_pass_frac=0.0; 2295–2303).
- Cosine milestones: norm ge_{0.2,0.4,0.6} at layers {11,25,26} (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2249–2254). Tuned shows ge_{0.2,0.4,0.6} at layer 1 across summaries (5408–5413, 5539–5544, 5670–5675).
- Entropy: strong early decrease (entropy_gap_bits p50≈10.99; 7161–7165); final entropy matches teacher (4863–4864). Entropy decreases precede KL convergence; semantics consolidate around L=25.
- Stability notes: Bold semantics layer below uses confirmed semantics; however, gate‑stability is poor (min both‑gates pass frac=0.0) and position‑window rank‑1 fraction=0.0, so treat semantic onset as calibration‑sensitive and position‑fragile.

4.5. Prism (shared‑decoder diagnostic)
Prism present and compatible (k=512 at layers embed/7/15/23; 844–857). KL deltas vs norm are negative (i.e., prism worse): ΔKL p50≈−17.54 bits and p25≈−12.81 (875–891). Rank milestones under prism unchanged (null), with first_kl_le_1.0 also null (892–897). Verdict: Regressive (KL increases and no earlier ranks).

4.6. Ablation & stress tests
Ablation summary: L_copy_orig=null; L_sem_orig=25; L_copy_nf=null; L_sem_nf=24; ΔL_sem=−1 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:5258–5265). Control prompt (France→Paris): first_control_margin_pos=2; max_control_margin=0.6539; first_control_strong_pos=24 (5292–5297). Important‑word trajectory: at L=25 the top‑k include ‘Berlin’, ‘Germany’, ‘Frankfurt’, ‘Deutschland’, ‘German’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-records.csv:443); early layers 0–3 show no such semantics (e.g., layer 0 row 18 shows unrelated fragments).

4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓ (RMSNorm; next_ln1 strategy) (2434–2441, 2727–2733)
- LayerNorm bias removed ✓ (unembed_bias.present=false; 834–838)
- FP32 unembed promoted ✓ (unembed_dtype=torch.float32; 811–818)
- Punctuation / markup anchoring noted ✓ (position‑window rank1_frac=0.0; quotes cluster pre‑semantics; 7003–7015, row 24)
- Copy‑reflex ✗ (no early copy flags; 2211–2237; rows 2–5)
- Preferred lens honored ✓ (preferred_lens_for_reporting=tuned; used for tuned comparisons; 7092–7107)
- Confirmed semantics reported ✓ (L_semantic_confirmed=25; confirmed_source="raw"; 4730–4736)
- Dual‑lens artifact metrics cited ✓ (lens_artifact_score_v2, JS/L1, Jaccard; 2331–2356, 2427–2431)
- Tuned‑lens audit done ✓ (rotation/temp/positional/head; 7166–7196)
- normalization_provenance present ✓ (ln_source at L0/final; 2439, 2727)
- per‑layer normalizer effect present ✓ (resid_norm_ratio, delta_resid_cos; 2442–2463, 2667–2733)
- deterministic_algorithms true ✓ (5236–5247)
- numeric_health clean ✓ (2936–2942)
- copy_mask plausible ✓ (2125–2143)
- milestones/evaluation_pack citations used ✓ (7114–7133, 7198–7211)
- gate_stability_small_scale reported ✓ (2287–2303)
- position_window stability reported ✓ (7003–7015)

—
Produced by OpenAI GPT-5
