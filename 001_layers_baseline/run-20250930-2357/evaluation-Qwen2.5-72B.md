**1. Overview**
- Model: `Qwen/Qwen2.5-72B`; Layers: 80 (diagnostics.model, model_stats)
- Run timestamp: `timestamp-20250930-2357` (run-latest marker). Probe traces layerwise entropy, ranks, KL-to-final, cosine, copy flags, and control margin.

**2. Method Sanity‑Check**
JSON confirms norm lensing with rotary positions and FP32 unembed: "use_norm_lens": true (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:807) and "layer0_position_info": "token_only_rotary_model" (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:816). The context prompt ends exactly with “called simply” (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:817).

Last‑layer head calibration is good: "kl_to_final_bits": 0.000109..., "top1_agree": true, with near‑match between lens and model: "p_top1_lens": 0.33946 vs "p_top1_model": 0.33828; temp_est=1.0; "kl_after_temp_bits" ≈ 0.000109 (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1115–1123).

Copy detection config present and mirrored in CSV: copy_thresh=0.95, copy_window_k=1, copy_match_level=id_subsequence; soft threshold=0.5 with window_ks=[1,2,3]; copy_flag_columns include `copy_strict@{0.95,0.7,0.8,0.9}` and `copy_soft_k{1,2,3}@0.5` (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:976–978,880–888,1028–1054,1332–1340). Gold alignment is "ok" (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1114).

Measurement guidance: prefer ranks and suppress absolute probabilities; preferred lens="norm"; use_confirmed_semantics=false. Reasons include norm‑only semantics and high lens‑artifact risk/score (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1373–1383).

Raw‑vs‑Norm window: radius=4; centers=[78,80]; norm‑only semantics_layers=[80]; max KL(norm vs raw) in window=83.315 bits (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1081–1101). Full raw‑vs‑norm summary: pct_kl≥1.0=0.321, pct_kl≥0.5=0.420, n_norm_only_semantics_layers=1 (earliest=80), max_kl=83.315, tier=high (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1102–1113). Treat any pre‑final or window semantics as lens‑induced unless confirmed.

Threshold sweep: present with stability="none"; earliest L_copy_strict at τ=0.70 and τ=0.95 are null; norm_only_flags all null (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1028–1054). Strict copy not observed; soft detectors at τ_soft=0.5 do not fire in layers 0–3 (pure CSV shows no True flags). Copy‑reflex: ✗ (no strict or soft k1 hits in L0–L3).

Summary indices (bits, ranks): first_kl_below_1.0 = 80; first_kl_below_0.5 = 80; first_rank_le_{10,5,1} = {74,78,80} (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:972–980). Units are bits (KL/entropy fields are suffixed "_bits").

Prism sidecar: present and compatible; metrics block reports baseline rank milestones {74,78,80} while prism yields null for {10,5,1}; KL deltas at percentiles are mixed (baseline p25=12.62, prism p25=9.46; p75 baseline 9.11 vs prism 9.65) (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:825–860).

**3. Quantitative Findings**

| Layer | Entropy (bits) | Top-1 token |
|---:|---:|:---|
| L 0 | 17.214 | 's' |
| L 1 | 17.214 | '下一篇' |
| L 2 | 17.143 | 'ولوج' |
| L 3 | 17.063 | 'شدد' |
| L 4 | 17.089 | '.myapplication' |
| L 5 | 17.007 | 'ستحق' |
| L 6 | 17.031 | '.myapplication' |
| L 7 | 16.937 | '.myapplication' |
| L 8 | 16.798 | '.myapplication' |
| L 9 | 16.120 | 'ستحق' |
| L 10 | 16.501 | '.myapplication' |
| L 11 | 16.718 | '.myapplication' |
| L 12 | 16.778 | 'かもしれ' |
| L 13 | 16.631 | 'かもしれ' |
| L 14 | 16.359 | 'かもしれ' |
| L 15 | 16.517 | 'のではない' |
| L 16 | 16.491 | 'iéndo' |
| L 17 | 16.213 | 'iéndo' |
| L 18 | 16.109 | '有期徒' |
| L 19 | 15.757 | '有期徒' |
| L 20 | 16.129 | '有期徒' |
| L 21 | 16.156 | '有期徒' |
| L 22 | 15.980 | '有期徒' |
| L 23 | 16.401 | '.myapplication' |
| L 24 | 15.999 | 'iéndo' |
| L 25 | 15.351 | 'hế' |
| L 26 | 15.944 | 'iéndo' |
| L 27 | 15.756 | 'iéndo' |
| L 28 | 15.750 | '.myapplication' |
| L 29 | 15.885 | '.myapplication' |
| L 30 | 16.123 | '.myapplication' |
| L 31 | 16.170 | '.myapplication' |
| L 32 | 16.171 | '.myapplication' |
| L 33 | 16.419 | 'hế' |
| L 34 | 16.200 | 'iéndo' |
| L 35 | 16.455 | 'hế' |
| L 36 | 16.408 | 'iéndo' |
| L 37 | 16.210 | 'iéndo' |
| L 38 | 16.490 | 'hế' |
| L 39 | 16.418 | 'iéndo' |
| L 40 | 16.192 | 'iéndo' |
| L 41 | 16.465 | 'hế' |
| L 42 | 16.595 | 'hế' |
| L 43 | 16.497 | 'hế' |
| L 44 | 16.655 | '続きを読む' |
| L 45 | 16.877 | '国际在线' |
| L 46 | 17.002 | '国际在线' |
| L 47 | 17.013 | '主义思想' |
| L 48 | 17.022 | '主义思想' |
| L 49 | 17.022 | ' reuseIdentifier' |
| L 50 | 16.968 | 'uckets' |
| L 51 | 16.972 | ' "' |
| L 52 | 17.009 | '"' |
| L 53 | 16.927 | '"' |
| L 54 | 16.908 | '"' |
| L 55 | 16.942 | '"' |
| L 56 | 16.938 | '"' |
| L 57 | 16.841 | ' "' |
| L 58 | 16.915 | ' "' |
| L 59 | 16.920 | ' "' |
| L 60 | 16.886 | ' '' |
| L 61 | 16.903 | ' '' |
| L 62 | 16.834 | ' "' |
| L 63 | 16.891 | ' "' |
| L 64 | 16.895 | ' "' |
| L 65 | 16.869 | ' "' |
| L 66 | 16.899 | ' "' |
| L 67 | 16.893 | ' "' |
| L 68 | 16.779 | ' "' |
| L 69 | 16.876 | ' "' |
| L 70 | 16.787 | ' "' |
| L 71 | 16.505 | ' "' |
| L 72 | 16.650 | ' "' |
| L 73 | 15.787 | ' "' |
| L 74 | 16.081 | ' "' |
| L 75 | 13.350 | ' "' |
| L 76 | 14.743 | ' "' |
| L 77 | 10.848 | ' "' |
| L 78 | 15.398 | ' "' |
| L 79 | 16.666 | ' "' |
| **L 80** | 4.116 | ' Berlin' |

Control margin (JSON): first_control_margin_pos = 0; max_control_margin = 0.2070367018786783 (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1361–1379).

Ablation (no‑filler): L_copy_orig = null, L_sem_orig = 80; L_copy_nf = null, L_sem_nf = 80; ΔL_copy = null, ΔL_sem = 0 (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1339–1351).

- ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n/a (no L_copy).
- Soft ΔHk (bits): n/a (no L_copy_soft[k]).
- Confidence milestones (pure CSV, pos/orig): p_top1 > 0.30 at layer 77; p_top1 > 0.60: none; final-layer p_top1 = 0.3395 (001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:82).
- Rank milestones (diagnostics, norm lens): rank ≤ 10 at L 74; ≤ 5 at L 78; ≤ 1 at L 80 (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:972–980).
- KL milestones (diagnostics): first_kl_below_1.0 at L 80; first_kl_below_0.5 at L 80; final KL ≈ 0 (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:972–976,1116).
- Cosine milestones (JSON/CSV): cos_to_final ≥ {0.2,0.4} at L 0; ≥ 0.6 at L 53; final cos_to_final ≈ 1.0000 (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1066–1071; pure CSV final row 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:82).
- Normalized depths: L_semantic_frac = 1.00; first_rank_le_5_frac = 0.975 (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1073–1076).

Copy robustness (threshold sweep): stability = "none"; earliest L_copy_strict at τ=0.70 = null; at τ=0.95 = null; norm_only_flags all null (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1028–1054). No strict nor soft copy in early layers.

Prism Sidecar Analysis: Prism is present/compatible but regressive here. It shows no rank milestones (le_1=null) and higher KL at mid/final depth; prism KL at L∈{0,20,40,60,80} ≈ {9.49, 9.46, 9.57, 9.65, 20.68} bits; baseline final KL ≈ 0 (pure‑prism vs pure baseline; see 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token-prism.csv and 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv).

**4. Qualitative Patterns & Anomalies**
Negative control. For “Berlin is the capital of”, top‑5 are “ Germany” (0.77), “ the” (0.086), “ which” (0.049), “ a” (0.012), “ what” (0.0075) — no Berlin; this is the desired direction (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:10–16,18–24,26–32). No semantic leakage from the target prompt here.

Important‑word trajectory. In the records CSV, “Berlin” first becomes top‑k in late layers at NEXT positions. For example, at L=74/76 under prompt tokens near “simply”, Berlin appears among top‑k but not top‑1; by L=80 it is top‑1 at the final positions with p≈0.339 (001_layers_baseline/run-latest/output-Qwen2.5-72B-records.csv:3911,3966,4106). Punctuation and quotes dominate many late layers’ top‑1 (L51–L79), indicative of stylistic tokens anchoring before semantics.

Instruction sensitivity. Removing “simply” does not shift semantic collapse: L_sem_orig=80 and L_sem_nf=80; ΔL_sem=0 (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1339–1351). This suggests minimal reliance on that stylistic cue for this probe.

Rest‑mass sanity. Rest_mass falls by final layer; max after L_semantic is 0.298 (<0.3), not indicative of precision loss (pure CSV, pos/orig; final row shows rest_mass=1−p_top20≈0.2977: 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:82).

Rotation vs amplification. Cosine to final rises early (≥0.2 by L0; ≥0.6 by L53) while KL to final remains high until the end (first KL≤1.0 at L80), matching an “early direction, late calibration” pattern. Since final KL≈0 and last‑layer consistency is good, we trust rank milestones over absolute probabilities per measurement guidance.

Head calibration (final layer). warn_high_last_layer_kl=false; temp_est=1.0; kl_after_temp_bits≈0.000109 (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1115–1123). No family‑specific head calibration issue observed.

Lens sanity. Raw‑vs‑Norm window shows norm‑only semantics at L80 and extreme KL(norm vs raw) ≈ 83.32 bits (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1081–1101). Full raw‑vs‑norm summary tier=high; earliest_norm_only_semantic=80 (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1102–1113). In the window sidecar, at L80 the norm lens has answer_rank=1 with top‑1 “ Berlin”, while the raw lens has p_top1≈1.0 on “,” and answer_rank=28 (001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token-rawlens-window.csv). Treat early semantics as lens‑induced; prefer rank milestones and within‑model trends.

Temperature robustness. At T=0.1, “ Berlin” rank 1 with p≈0.953 (entropy≈0.275); at T=2.0, “ Berlin” p≈0.016 and high entropy≈15.01 (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:736–800).

Checklist
- RMS lens? ✓ (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:807)
- LayerNorm bias removed? n.a. (RMS family; rotary pos)
- Entropy rise at unembed? ✓ (teacher_entropy_bits ≈ 4.136; early layers ≈17 bits; pure CSV)
- FP32 un‑embed promoted? ✓ ("unembed_dtype": "torch.float32"; 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:811)
- Punctuation / markup anchoring? ✓ (quotes/punctuation top‑1 in L51–L79; records/pure CSV)
- Copy‑reflex? ✗ (no strict/soft hits in L0–L3; pure CSV)
- Grammatical filler anchoring? ✓ (" the", quotes frequently top‑1 mid/late; records CSV)
- Preferred lens honored? ✓ (measurement_guidance.preferred_lens_for_reporting=norm; 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1382)
- Confirmed semantics reported? n.a. (L_semantic_confirmed=null; confirmed_source=none; 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1066–1076, confirmed_semantics)
- Full dual‑lens metrics cited? ✓ (pct_kl≥1.0=0.321; n_norm_only=1; earliest=80; tier=high; 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1102–1113)
- Tuned‑lens attribution? n.a. (tuned lens missing; 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1–2, tuned_lens)

**5. Limitations & Data Quirks**
- Final‑layer KL≈0 and top‑1 agree, but raw‑vs‑norm shows high lens divergence (max_kl≈83.3 bits) and norm‑only semantics at L80. Prefer rank milestones; treat any early “semantics” cautiously as lens‑induced.
- Rest_mass just under 0.3 after L_semantic (≈0.298) suggests no precision loss at output, but punctuation‑heavy top‑k indicates stylistic anchoring.
- Prism is regressive here (higher KL, no earlier ranks); avoid using Prism for milestones beyond diagnostic comparison.
- Soft‑copy threshold in this run is 0.5 (not 0.33 default); no soft copy hits occurred. ΔH and Soft ΔHk are n/a due to absent L_copy.
- Surface‑mass and tokenization: echo/answer masses reflect tokenizer idiosyncrasies; use within‑model trends.

**6. Model Fingerprint**
“Qwen2.5‑72B: collapse at L 80; final entropy ≈4.12 bits; ‘Berlin’ appears only at the very end under the norm lens.”

---
Produced by OpenAI GPT-5 

