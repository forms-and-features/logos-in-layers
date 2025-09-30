# Evaluation Report: google/gemma-2-27b
**Overview**
Google Gemma‑2‑27B; probe over the Germany→Berlin prompt using the norm lens. The run captures early copy‑reflex, late semantic collapse at the final layer, high raw↔norm divergence, and family‑typical last‑layer head mis‑calibration.

**Method Sanity‑Check**
The prompt ends exactly with “called simply” (no trailing space):
> "Give the city name only, plain text. The capital of Germany is called simply"  [001_layers_baseline/run-latest/output-gemma-2-27b.json:4]

Norm lens and positional handling are enabled and architecture‑aware:
> "use_norm_lens": true, "unembed_dtype": "torch.float32"  [001_layers_baseline/run-latest/output-gemma-2-27b.json:807–809]
> "layer0_position_info": "token_only_rotary_model"  [001_layers_baseline/run-latest/output-gemma-2-27b.json:816]

Gold alignment is OK and ID‑based:
> "gold_alignment": "ok"  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1071]
> "gold_answer": { "string": "Berlin", "first_id": 12514, "answer_ids": [12514] }  [001_layers_baseline/run-latest/output-gemma-2-27b.json:2106–2114]

Copy/collapse configuration and results are present in diagnostics. Strict copy (τ=0.95, k=1, id_subsequence) and soft copy (τ=0.5, k∈{1,2,3}) are defined, with earliest strict and soft copy at L=0 and Δ=46:
> "L_copy": 0, "L_semantic": 46, "delta_layers": 46, "copy_thresh": 0.95, "copy_window_k": 1, "copy_match_level": "id_subsequence"  [001_layers_baseline/run-latest/output-gemma-2-27b.json:938–944]
> "L_copy_soft": { "1": 0, "2": null, "3": null }, "delta_layers_soft": { "1": 46, ... }  [001_layers_baseline/run-latest/output-gemma-2-27b.json:956–965]
> "copy_detector.strict": { "thresh": 0.95, "k": 1, "L_copy_strict": 0 }  [001_layers_baseline/run-latest/output-gemma-2-27b.json:966–971]
> "copy_detector.soft": { "thresh": 0.5, "window_ks": [1,2,3], "L_copy_soft": { "k1": 0, ... } }  [001_layers_baseline/run-latest/output-gemma-2-27b.json:972–983]
> "copy_thresholds": { "tau_list": [0.7,0.8,0.9,0.95], "L_copy_strict": {"0.7":0,"0.8":0,"0.9":0,"0.95":0}, "norm_only_flags": { ... all false }, "stability": "mixed" }  [001_layers_baseline/run-latest/output-gemma-2-27b.json:994–1021]
> "copy_flag_columns": ["copy_strict@0.95", "copy_strict@0.7", "copy_strict@0.8", "copy_strict@0.9", "copy_soft_k1@0.5", ...]  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1596–1603]

Strict copy fires immediately at L0 in the pure CSV (ID‑level subsequence):
> layer=0, copy_collapse=True, copy_strict@{0.95,0.9,0.8,0.7}=True; top‑1 " simply" (p=0.99998)  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2]
Earliest soft‑copy (k1, τ=0.5) is also at L0: "copy_soft_k1@0.5"=True  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2].

Rank and KL summary indices are present (units: bits for KL/entropy):
> "first_kl_below_0.5": null, "first_kl_below_1.0": null, "first_rank_le_1": 46, "first_rank_le_5": 46, "first_rank_le_10": 46  [001_layers_baseline/run-latest/output-gemma-2-27b.json:945–949]

Last‑layer head calibration is off: the final‑row KL is not ~0 and diagnostics flag it with a temperature estimate; measurement guidance enforces rank‑first reporting:
> final row: "kl_to_final_bits" = 1.1352, "is_answer"=True, "p_top1"=0.9841  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48]
> { "kl_to_final_bits": 1.1352, "top1_agree": true, "p_top1_lens": 0.9841, "p_top1_model": 0.4226, "temp_est": 2.6102, "kl_after_temp_bits": 0.5665, "warn_high_last_layer_kl": true }  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1072–1091]
> "measurement_guidance": { "prefer_ranks": true, "suppress_abs_probs": true, "reasons": ["warn_high_last_layer_kl","norm_only_semantics_window","high_lens_artifact_risk"], ... }  [001_layers_baseline/run-latest/output-gemma-2-27b.json:2096–2104]

Raw‑vs‑Norm window sanity shows norm‑only semantics at the final layer and very large raw↔norm divergences:
> "raw_lens_window": { "center_layers": [0,46], "radius": 4, "norm_only_semantics_layers": [46], "max_kl_norm_vs_raw_bits_window": 99.5398 }  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1047–1069]
> rawlens window CSV (L46): norm p_top1=0.9841 (Berlin), raw p_top1=0.8898 (); kl_norm_vs_raw_bits=99.5398  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token-rawlens-window.csv:20–21]
Lens sanity (sampled) flags high artifact risk:
> "raw_lens_check": { "mode": "sample", "summary": { "first_norm_only_semantic_layer": null, "max_kl_norm_vs_raw_bits": 80.1001, "lens_artifact_risk": "high" } }  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1534–1594]

Negative control and ablation present:
> control_summary: { "first_control_margin_pos": 0, "max_control_margin": 0.9911 }  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1628–1631]
> ablation_summary: { "L_copy_orig": 0, "L_sem_orig": 46, "L_copy_nf": 3, "L_sem_nf": 46, "delta_L_copy": 3, "delta_L_sem": 0 }  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1606–1611]

Prism sidecar is compatible (k=512; layers: embed,10,22,33):
> "prism_summary": { "compatible": true, "k": 512, "layers": ["embed",10,22,33] }  [001_layers_baseline/run-latest/output-gemma-2-27b.json:825–835]

Tuned‑Lens loaded; skip‑layers sanity warns and regression is noted:
> "tuned_lens": { "path": "/.../tuned_lenses/gemma-2-27b" }  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1104–1108]
> "skip_layers_sanity": { "m=2": 475106470164.1723, ... }, "tuned_lens_regression": true  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1431–1437]

**Quantitative Findings**
Table (pure next‑token; prompt_id=pos, prompt_variant=orig): Layer, entropy (bits), top‑1 token. Bold marks the semantic layer (first is_answer=True). For reference, gold_answer.string = “Berlin”.

| Layer | Entropy (bits) | Top-1 token |
|---:|---:|---|
| 0 | 0.000 |  simply |
| 1 | 8.758 |  |
| 2 | 8.764 |  |
| 3 | 0.886 |  simply |
| 4 | 0.618 |  simply |
| 5 | 8.520 | ๲ |
| 6 | 8.553 |  |
| 7 | 8.547 |  |
| 8 | 8.529 |  |
| 9 | 8.524 | 𝆣 |
| 10 | 8.345 |  dieſem |
| 11 | 8.493 | 𝆣 |
| 12 | 8.324 |  |
| 13 | 8.222 |  |
| 14 | 7.877 |  |
| 15 | 7.792 |  |
| 16 | 7.975 |  dieſem |
| 17 | 7.786 |  dieſem |
| 18 | 7.300 | ſicht |
| 19 | 7.528 |  dieſem |
| 20 | 6.210 | ſicht |
| 21 | 6.456 | ſicht |
| 22 | 6.378 |  dieſem |
| 23 | 7.010 |  dieſem |
| 24 | 6.497 |  dieſem |
| 25 | 6.995 |  dieſem |
| 26 | 6.220 |  dieſem |
| 27 | 6.701 |  dieſem |
| 28 | 7.140 |  dieſem |
| 29 | 7.574 |  dieſem |
| 30 | 7.330 |  dieſem |
| 31 | 7.565 |  dieſem |
| 32 | 8.874 |  zuſammen |
| 33 | 6.945 |  dieſem |
| 34 | 7.738 |  dieſem |
| 35 | 7.651 |  dieſem |
| 36 | 7.658 |  dieſem |
| 37 | 7.572 |  dieſem |
| 38 | 7.554 |  パンチラ |
| 39 | 7.232 |  dieſem |
| 40 | 8.711 |  展板 |
| 41 | 7.082 |  dieſem |
| 42 | 7.057 |  dieſem |
| 43 | 7.089 |  dieſem |
| 44 | 7.568 |  dieſem |
| 45 | 7.141 |  Geſch |
| 46 | 0.118 |  Berlin |

Bold semantic layer: L46 — "Berlin" with p_top1=0.9841 and is_answer=True  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48].

Control margin (JSON): first_control_margin_pos = 0; max_control_margin = 0.9911  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1628–1631].

Ablation (no‑filler): L_copy_orig=0, L_sem_orig=46; L_copy_nf=3, L_sem_nf=46; ΔL_copy=3, ΔL_sem=0  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1606–1611].

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = 0.00050 − 0.11805 = −0.118  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2,48].
Soft ΔHₖ (k=1) = entropy(L_copy_soft[k]) − entropy(L_semantic) = 0.00050 − 0.11805 = −0.118  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2,48].

Confidence milestones (pure): p_top1 > 0.30 at L0; > 0.60 at L0; final‑layer p_top1 = 0.9841  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2,48].

Rank milestones (diagnostics): rank ≤10 at L46; ≤5 at L46; ≤1 at L46  [001_layers_baseline/run-latest/output-gemma-2-27b.json:947–949].

KL milestones (diagnostics): first_kl_below_1.0 = null; first_kl_below_0.5 = null  [001_layers_baseline/run-latest/output-gemma-2-27b.json:945–946]. KL is not ~0 at final (1.1352 bits) and head calibration is flagged  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1072–1091].

Cosine milestones (JSON): first cos_to_final ≥0.2 at L1; ≥0.4 at L46; ≥0.6 at L46  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1032–1036]. Final cos_to_final = 0.99939  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48].

Depth fractions: L_semantic_frac=1.0; L_copy_strict_frac=0.0; L_copy_soft_k1_frac=0.0  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1039–1045].

Copy robustness (threshold sweep): stability = "mixed"; earliest strict copy layer τ=0.70 → L0, τ=0.95 → L0; norm_only_flags for all τ are false  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1001–1019].

Prism Sidecar Analysis
- Early/mid‑depth KL: clear reductions vs baseline (e.g., L11: baseline 41.852 bits vs Prism 19.432 bits)  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:11; 001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token-prism.csv:3].
- Rank milestones (Prism pure): never reaches rank ≤10/5/1 (all null) while baseline reaches 46  [001_layers_baseline/run-latest/output-gemma-2-27b.json:839–848].
- Sampled depths: L0 baseline top‑1 " simply" vs Prism "assuredly"; Prism cos_to_final negative early (−0.089)  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token-prism.csv:2].
- Copy flags: baseline strict copy at L0; Prism shows no copy flags at L0–4  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2; 001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token-prism.csv:2–6].
Verdict: Regressive (big early KL drop without earlier rank milestones; never achieves rank≤10 under Prism).

Tuned‑Lens Sidecar
- ΔKL medians at percentiles: p25 Δ≈+0.054, p50 Δ≈+0.334, p75 Δ≈+0.342 (KL_norm − KL_tuned; positive means tuned slightly better)  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:11,24,35; 001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token-tuned.csv:24].
- Entropy drift (tuned): at p25/p50/p75, (entropy − teacher_entropy_bits) ≈ 5.580, 4.425, 5.203 bits  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token-tuned.csv:24].
- Rank milestones: tuned reaches rank ≤{10,5,1} at L46 (same as baseline)  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token-tuned.csv:24].
- Surface→meaning: L_surface_to_meaning_norm = 46 (JSON); tuned first answer_mass > echo_mass also at L46  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1021–1023; 001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token-tuned.csv:24].
- Geometry: tuned cos_to_final ≥0.4 at L46 (same as norm)  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token-tuned.csv:24].
- Coverage (K=50, τ≈0.33): L_topk_decay_norm=1 (JSON); tuned shows the same earliest drop <0.33  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1027–1030; 001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token-tuned.csv:2].
- Norm temperature snapshots: KL_temp@{25,50,75}% ≈ 41.30/41.60/41.51 bits  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1092–1103].
Overall: Tuned‑Lens is neutral‑to‑slightly helpful on KL but does not improve rank earliness.

**Qualitative Patterns & Anomalies**
Negative control (Berlin is the capital of): top‑5 do not include “Berlin”.
> [" Germany", 0.8676], [" the", 0.0650], [" and", 0.0065], [" a", 0.0062], [" Europe", 0.0056]  [001_layers_baseline/run-latest/output-gemma-2-27b.json:14–31]

Important‑word trajectory (records): “Berlin” first appears only at the final layer across the three prompt positions; e.g., NEXT after “simply” shows “Berlin” 0.9841 at L46.
> " simply,0.1180, Berlin,0.9841, ..."  [001_layers_baseline/run-latest/output-gemma-2-27b-records.csv:806]
Earlier layers are dominated by historical/orthographic artifacts (e.g., “dieſem”, “ſicht”, “Geſch”) rather than country/capital semantics  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:10,18,45].

Collapse‑layer stability without “one‑word” instruction: Ablation shows L_sem unchanged (46) while copy shifts later (L_copy 0→3), suggesting filler removal disrupts copy‑reflex but not the eventual semantic collapse  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1606–1611].

Rest‑mass sanity: early layers have very high rest_mass (e.g., L1 ≈ 0.982), decaying by the final layer (≈2.0e‑7), indicating sharpening and top‑k concentration near collapse  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:3,48].

Rotation vs amplification: cos_to_final rises only late (≥0.4 and ≥0.6 at L46), while KL to the final head stays high throughout and remains ≈1.14 bits even at L46, indicating an “early direction, late calibration” picture with final‑head calibration issues  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1032–1036,1072–1091].

Head calibration (final layer): diagnostics estimate temp≈2.61 and reduce KL after temp to ≈0.567 bits, consistent with Gemma family behavior; rely on rank‑based milestones for cross‑model comparisons  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1072–1091].

Lens sanity: sampled raw‑vs‑norm shows extreme divergences (max≈80.10 bits), with raw often assigning very high mass to non‑semantic tokens mid‑stack; within the window, the final layer is norm‑only semantic  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1534–1594,1047–1069]. Example sample:
> L12: kl_norm_vs_raw_bits=33.687; p_top1_norm=0.0144 vs p_top1_raw=0.3918; answer_rank_norm=240631 vs raw=402  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1551–1561].

Temperature robustness: T=0.1 → “Berlin” p≈0.9898 (entropy≈0.082); T=2.0 → “Berlin” p≈0.049 (entropy≈12.631)  [001_layers_baseline/run-latest/output-gemma-2-27b.json:670–676,737–744,738–739].

Checklist
- RMS lens? ✓  [001_layers_baseline/run-latest/output-gemma-2-27b.json:810–814]
- LayerNorm bias removed? ✓ (RMSNorm; “not_needed_rms_model”)  [001_layers_baseline/run-latest/output-gemma-2-27b.json:811–813]
- Entropy rise at unembed? ✗ (final entropy low; teacher entropy ≈2.886)  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48]
- FP32 un‑embed promoted? ✓ (casting to fp32 before unembed; unembed_dtype fp32)  [001_layers_baseline/run-latest/output-gemma-2-27b.json:808–809,815]
- Punctuation / markup anchoring? ✓ (non‑Latin/markup tokens mid‑stack)  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:32,40]
- Copy‑reflex? ✓ (strict and soft at L0)  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2]
- Grammatical filler anchoring? ✓ (top‑1 “ simply” in L0/L3/L4)  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2,5,6]

**Limitations & Data Quirks**
- High rest_mass in early layers reflects top‑k coverage limits, not fidelity; interpret via KL/entropy and ranks within‑model only  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:3].
- KL is lens‑sensitive; raw‑vs‑norm shows large divergences (80–100 bits in window). Treat any pre‑final “early semantics” cautiously; prefer rank milestones  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1047–1069,1534–1594].
- Final‑head calibration (warn_high_last_layer_kl=true) means final probabilities are not cross‑family comparable; use rank milestones and KL thresholds within model  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1072–1091,2096–2104].
- Surface‑mass and coverage trends can be tokenizer‑dependent; avoid cross‑model absolute comparisons.

**Model Fingerprint**
Gemma‑2‑27B: strict copy at L0; semantic collapse at L46; final KL≈1.14 bits; “Berlin” stabilizes only at the last layer.

---
Produced by OpenAI GPT-5 
*Run executed on: 2025-09-29 23:35:16*
