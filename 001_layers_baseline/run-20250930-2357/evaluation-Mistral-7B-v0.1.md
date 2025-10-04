# Evaluation Report: mistralai/Mistral-7B-v0.1

## 1. Overview
Model: "mistralai/Mistral-7B-v0.1" (diagnostics.model) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:805]. The probe analyzes layer-by-layer next-token behavior under the norm lens with RMSNorm architecture and calibration diagnostics. Sidecars for Tuned‑Lens and Prism are present (loaded/compatible) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2121] [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:828–833].

## 2. Method sanity‑check
The run used the norm lens ("use_norm_lens": true) with fp32 unembedding dtype [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:807–809], on an RMSNorm model (first_block_ln1_type/final_ln_type = "RMSNorm") [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:810–811]. The context prompt is:
> "Give the city name only, plain text. The capital of Germany is called simply"  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:817]
It ends with "called simply" (no trailing space).

Copy/soft‑copy configuration and flags are present and consistent: strict copy (τ=0.95, k=1, level "id_subsequence") [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:928–931], threshold sweep τ ∈ {0.7,0.8,0.9,0.95} with stability "none" and null L_copy_strict at all τ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:984–1005], and soft‑copy config threshold 0.5, window_ks [1,2,3] [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:100–112 (copy_soft_config), 880–889, 888–889]. The emitted flag columns mirror these labels (copy_strict@{0.95,0.7,0.8,0.9}, copy_soft_k{1,2,3}@0.5) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1619–1626].

Gold‑token alignment succeeded ("gold_alignment": "ok") [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1071]. Negative control summary is present: first_control_margin_pos = 2, max_control_margin ≈ 0.654 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1651–1652]. Ablation exists: L_sem_orig = 25, L_sem_nf = 24 (ΔL_sem = −1) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1629–1633].

Rank/KL summary indices (baseline lens): first_rank_le_{10,5,1} = {23,25,25} and first_kl_below_{1.0,0.5} = {32,32} [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:932–940]. KL/entropy units are bits throughout. Final-layer head calibration is clean: last_layer_consistency shows kl_to_final_bits = 0.0, top1_agree = true, temp_est = 1.0, and kl_after_temp_bits = 0.0 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1073–1081].

Measurement guidance advises rank‑first reporting and tuned‑lens preference: prefer_ranks = true, suppress_abs_probs = true, preferred_lens_for_reporting = "tuned", use_confirmed_semantics = true [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2125–2135].

Raw‑vs‑Norm window: center_layers = [25,32], radius = 4; norm_only_semantics_layers = [32]; max_kl_norm_vs_raw_bits_window ≈ 8.56 bits; mode = "window" [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1033–1057]. Full raw‑vs‑norm: pct_layers_kl_ge_{1.0,0.5} ≈ {0.242, 0.333}; n_norm_only_semantics_layers = 1; earliest_norm_only_semantic = 32; max_kl_norm_vs_raw_bits ≈ 8.56; lens_artifact_score tier = "high" [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1060–1069]. Given this, treat any pre‑final early semantics cautiously and prefer rank milestones.

Strict copy‑collapse: no hits (L_copy_strict null at all τ), and no soft‑copy k∈{1,2,3} hits under τ_soft=0.5 (earliest layers 0–3 show all copy flags False) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:984–1005] [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:2–5]. Copy‑reflex ✓/✗: ✗ (no early strict or soft copy in L0–L3).

Norm temperature diagnostics present: tau_norm_per_layer listed; snapshots kl_to_final_bits_norm_temp@{25,50,75}% provided [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:889–923, 1092–1103].

Prism sidecar is compatible (present=true, compatible=true) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:828–833]. Tuned‑Lens is loaded and attribution prefers tuned [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2121].

Context prompt sanity: verified ending “called simply” and no trailing space [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:817].

## 3. Quantitative findings
Table (pos, orig only). Bold = semantic collapse (confirmed). L_semantic_confirmed = 25 (source = raw) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1443–1446].

- L 0 — entropy 14.96 bits, top‑1 ‘dabei’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:2]
- L 1 — entropy 14.93 bits, top‑1 ‘biologie’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:3]
- L 2 — entropy 14.83 bits, top‑1 ‘,\r’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:4]
- L 3 — entropy 14.88 bits, top‑1 ‘[…]’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:5]
- L 4 — entropy 14.85 bits, top‑1 ‘[…]’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:6]
- L 5 — entropy 14.83 bits, top‑1 ‘[…]’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:7]
- L 6 — entropy 14.84 bits, top‑1 ‘[…]’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:8]
- L 7 — entropy 14.80 bits, top‑1 ‘[…]’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:9]
- L 8 — entropy 14.82 bits, top‑1 ‘[…]’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:10]
- L 9 — entropy 14.78 bits, top‑1 ‘[…]’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:11]
- L 10 — entropy 14.78 bits, top‑1 ‘[…]’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:12]
- L 11 — entropy 14.74 bits, top‑1 ‘[…]’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:13]
- L 12 — entropy 14.64 bits, top‑1 ‘[…]’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:14]
- L 13 — entropy 14.73 bits, top‑1 ‘[…]’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:15]
- L 14 — entropy 14.65 bits, top‑1 ‘[…]’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:16]
- L 15 — entropy 14.45 bits, top‑1 ‘[…]’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:17]
- L 16 — entropy 14.60 bits, top‑1 ‘[…]’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:18]
- L 17 — entropy 14.63 bits, top‑1 ‘[…]’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:19]
- L 18 — entropy 14.52 bits, top‑1 ‘[…]’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:20]
- L 19 — entropy 14.51 bits, top‑1 ‘[…]’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:21]
- L 20 — entropy 14.42 bits, top‑1 ‘simply’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:22]
- L 21 — entropy 14.35 bits, top‑1 ‘simply’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:23]
- L 22 — entropy 14.39 bits, top‑1 ‘“’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:24]
- L 23 — entropy 14.40 bits, top‑1 ‘simply’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:25]
- L 24 — entropy 14.21 bits, top‑1 ‘simply’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:26]
- **L 25 — entropy 13.60 bits, top‑1 ‘Berlin’** [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:27]
- L 26 — entropy 13.54 bits, top‑1 ‘Berlin’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:28]
- L 27 — entropy 13.30 bits, top‑1 ‘Berlin’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:29]
- L 28 — entropy 13.30 bits, top‑1 ‘Berlin’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:30]
- L 29 — entropy 11.43 bits, top‑1 ‘"’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:31]
- L 30 — entropy 10.80 bits, top‑1 ‘“’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:32]
- L 31 — entropy 10.99 bits, top‑1 ‘"’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:33]
- L 32 — entropy 3.61 bits, top‑1 ‘Berlin’ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:34]

Control margin: first_control_margin_pos = 2; max_control_margin ≈ 0.654 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1651–1652].

Ablation (no‑filler): L_copy_orig = null, L_sem_orig = 25; L_copy_nf = null, L_sem_nf = 24; ΔL_copy = null, ΔL_sem = −1 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1628–1633].

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (strict and soft copy are null). Soft ΔHₖ (k∈{1,2,3}) = n.a. (no soft copy hits).

Confidence milestones (baseline lens): p_top1 > 0.30 at L 32; p_top1 > 0.60 not reached; final-layer p_top1 = 0.382 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:34].

Rank milestones (preferred tuned lens first): tuned le_{10,5,1} = {25,25,32}; baseline le_{10,5,1} = {23,25,25} [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2064–2076, 2068–2072].

KL milestones: first_kl_below_1.0 at L 32; first_kl_below_0.5 at L 32; final‑layer KL ≈ 0 (clean calibration) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:932–935, 1073]. KL decreases with depth only late.

Cosine milestones (norm lens): ge_0.2 at L 11; ge_0.4 at L 25; ge_0.6 at L 26; final cos_to_final ≈ 1.0 at L 32 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1018–1025] [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:34].

Depth fractions: L_semantic_frac ≈ 0.781 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1026].

Copy robustness (threshold sweep): stability = "none"; earliest L_copy_strict at τ=0.70 and τ=0.95 are null; norm_only_flags null [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:984–1005].

Prism sidecar: compatible=true; KL at percentiles is much worse than baseline (p25 ≈ 23.06 vs 10.25; p50 ≈ 27.87 vs 10.33; p75 ≈ 26.55 vs 9.05) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:856–866]. Rank milestones under Prism are null (no improvement) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:845–853]. Verdict: Regressive.

## 4. Qualitative patterns & anomalies
The negative control “Berlin is the capital of” shows Germany dominating with Berlin still present lower in the list: > “Germany, 0.8966; the, 0.0539; … Berlin, 0.00284” [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:14–36]. This is expected and indicates the control behaves correctly.

Important‑word trajectory: Berlin becomes top‑1 at L 25 and remains top‑1 through L 28 and at L 32 (semantic collapse), with filler/quote tokens dominating mid‑late layers (e.g., L 29–31: '"', '“', '"') [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:27–34]. Rank milestones corroborate: rank ≤10 at L 23, ≤5 at L 25, ≤1 at L 25 (baseline), while tuned regresses to ≤1 at L 32 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2064–2072].

Rest‑mass sanity: rest_mass falls from ≈0.9998 at L 0 to ≈0.2298 at L 32; the maximum after semantic onset is at L 25 (≈0.9106) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:2,27,34]. This reflects growing top‑k concentration as the answer emerges; note rest_mass is top‑k coverage only (not a lens fidelity metric).

Rotation vs amplification: cosine rises early (ge_0.2 by L 11) while KL remains high until late (first_kl_below_1.0 at L 32), indicating “early direction, late calibration” [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1018–1025, 932–935]. Final‑head calibration is clean (KL≈0 at final) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1073].

Lens sanity: Raw‑vs‑Norm window flags a norm‑only semantic at L 32 within the window [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1053–1057]; full analysis reports lens_artifact_risk = high and earliest_norm_only_semantic = 32 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1062–1069]. We therefore emphasize rank milestones and the confirmed semantics layer (25, source=raw) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1443–1446].

Temperature robustness: at T=0.1, Berlin rank 1 with p≈0.9996 and entropy ≈0.005 bits [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:670–678]; at T=2.0, Berlin p≈0.036 with entropy ≈12.22 bits [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:737–744].

Tuned‑Lens attribution: prefer_tuned=true; ΔKL_tuned at depth percentiles ≈ {p25: 4.03, p50: 3.75, p75: 7.08} bits; ΔKL_temp ≈ {−0.24, −0.28, 4.23}; thus ΔKL_rot ≈ {4.27, 4.03, 2.85} [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2105–2119, 2121]. Despite KL gains, tuned rank‑1 occurs later (L 32) vs baseline (L 25) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2068–2072, 2064–2067].

Checklist:
✓ RMS lens (RMSNorm) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:810–811]
✓ LayerNorm bias removed/not needed (RMS) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:812]
✓ Entropy rise at unembed: teacher_entropy_bits ≈ 3.611 at final; mid‑depth entropy ≫ teacher (e.g., L 16 entropy ≈14.60) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:18,34]
✓ FP32 unembed dtype [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:809]
✗ Copy‑reflex (strict/soft in L0–3) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:2–5]
✓ Grammatical filler anchoring (top‑1 among {is/the/a/of/quotes/simply} in L20–24) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:22–26]
✓ Preferred lens honored (tuned in summaries) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2125–2135]
✓ Confirmed semantics reported (L=25, source=raw) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1443–1446]
✓ Full dual‑lens metrics cited (pct_layers_kl thresholds, earliest_norm_only, tier=high) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1060–1069]
✓ Tuned‑lens attribution (ΔKL_tuned/ΔKL_temp/ΔKL_rot at 25/50/75%) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2105–2119]

## 5. Limitations & data quirks
High raw‑vs‑norm artifact risk (tier=high) and a norm‑only semantic at L 32 warrant caution; prefer rank milestones and the confirmed raw semantics (L 25) for onset [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1053–1069, 1443–1446]. Rest_mass is top‑k coverage (not fidelity); it decreases post‑collapse but should not be used to compare lenses. Final‑layer KL≈0 indicates good head calibration here, but measurement_guidance still advises suppressing absolute probability comparisons across families [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2125–2135]. Surface‑mass and coverage depend on tokenization; prefer within‑model trends and rank milestones for interpretation.

## 6. Model fingerprint (one sentence)
“Mistral‑7B‑v0.1: confirmed collapse at L 25 (raw), late KL↓ at L 32, final entropy ≈3.61 bits; filler/quotes dominate mid‑late before ‘Berlin’ stabilizes.”

---
Produced by OpenAI GPT-5
