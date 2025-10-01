# Evaluation Report: Qwen/Qwen2.5-72B

*Run executed on: 2025-09-29 23:35:16*
**Overview**
- Model: Qwen/Qwen2.5-72B (pre-norm; 80 layers). Probe analyzes layer-wise next-token distributions with a norm lens, copy detectors, KL-to-final, cosine trajectory, and surface mass metrics. Model final prediction entropy 4.1356 bits with ‘ Berlin’ top-1.
- Artifacts: Compact JSON + pure-next-token/records CSVs; Prism sidecar present; Tuned-Lens missing. Measurement guidance recommends rank-led reporting.

**Method Sanity-Check**
- Lens and positions: use_norm_lens = true; layer-0 uses rotary/token-only info. “token_only_rotary_model” and norm lens flagged in diagnostics: "use_norm_lens": true [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:807], "layer0_position_info": "token_only_rotary_model" [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:816].
- Prompt string ends exactly with “called simply”: "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply" [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:817].
- Copy/strict/soft configuration present: copy_thresh = 0.95; copy_window_k = 1; copy_match_level = "id_subsequence"; soft threshold = 0.5 with window_ks = [1,2,3] [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:976–989]. Sweep block present with tau_list = {0.7,0.8,0.9,0.95} and stability = "none" [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1028–1054]. CSV/JSON flag columns match: ["copy_strict@0.95","copy_strict@0.7","copy_strict@0.8","copy_strict@0.9","copy_soft_k1@0.5","copy_soft_k2@0.5","copy_soft_k3@0.5"] [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1299–1307].
- Gold alignment ok and ID-based: diagnostics.gold_alignment = "ok" [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1102]; gold_answer: "Berlin" with first_id 19846 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1349–1355].
- Negative control prompt and summary present: control_prompt string for France→Paris with gold_alignment = "ok" [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1316–1330]; control_summary: first_control_margin_pos = 0, max_control_margin = 0.2070 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1331–1333].
- Ablation present: L_copy_orig = null, L_sem_orig = 80; L_copy_nf = null, L_sem_nf = 80; delta_L_sem = 0 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1308–1315]. Both prompt_variant rows appear in pure CSV.
- Summary indices (bits, ranks): first_kl_below_1.0 = 80; first_kl_below_0.5 = 80; first_rank_le_10 = 74; first_rank_le_5 = 78; first_rank_le_1 = 80 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:979–983]. Units for KL/entropy are bits.
- Last-layer head calibration: final-row KL≈0. CSV final row shows kl_to_final_bits = 0.0001091, answer_rank = 1 [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:138]. JSON last_layer_consistency confirms: kl_to_final_bits = 0.0001091; top1_agree = true; p_top1_lens = 0.3395 vs p_top1_model = 0.3383; temp_est = 1.0; kl_after_temp_bits = 0.0001091; warn_high_last_layer_kl = false [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1104–1121].
- Measurement guidance: prefer_ranks = true; suppress_abs_probs = true; reasons: ["norm_only_semantics_window","high_lens_artifact_risk"] [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1340–1347].
- Raw-vs-Norm window: radius = 4; center_layers = [78,80]; norm_only_semantics_layers = [80]; max_kl_norm_vs_raw_bits_window = 83.315 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1081–1100]. Lens sanity (sample mode): lens_artifact_risk = "high"; max_kl_norm_vs_raw_bits = 19.910; first_norm_only_semantic_layer = null [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1237–1298]. Sampled row shows large raw-vs-norm divergence near L61: kl_norm_vs_raw_bits = 19.91, p_top1_raw = 0.792 vs p_top1_norm = 0.000434 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1280–1286].
- Strict copy flags: none. No row with copy_collapse = True in layers 0–3 (or anywhere). Soft copy flags at τ_soft=0.5, k∈{1,2,3}: none in layers 0–3 (and none overall) in pure CSV.

Copy-collapse flag check
- No strict copy collapse observed; no line with copy_collapse = True found in pos/orig rows.
- Soft copy flags: none (k1/k2/k3) at τ_soft=0.5 in pos/orig rows.

**Quantitative Findings**
Table (pos, orig). One row per layer of the answer position:

L 0 — entropy 17.2142 bits, top-1 's'  [row 2]
L 1 — entropy 17.2142 bits, top-1 '下一篇'  [row 3]
L 2 — entropy 17.1425 bits, top-1 'ولوج'  [row 4]
L 3 — entropy 17.0631 bits, top-1 'شدد'  [row 5]
L 4 — entropy 17.0891 bits, top-1 '.myapplication'  [row 6]
L 5 — entropy 17.0072 bits, top-1 'ستحق'  [row 7]
L 6 — entropy 17.0315 bits, top-1 '.myapplication'  [row 8]
L 7 — entropy 16.9372 bits, top-1 '.myapplication'  [row 9]
L 8 — entropy 16.7980 bits, top-1 '.myapplication'  [row 10]
L 9 — entropy 16.1203 bits, top-1 'ستحق'  [row 11]
L 10 — entropy 16.5008 bits, top-1 '.myapplication'  [row 12]
L 11 — entropy 16.7180 bits, top-1 '.myapplication'  [row 13]
L 12 — entropy 16.7781 bits, top-1 'かもしれ'  [row 14]
L 13 — entropy 16.6314 bits, top-1 'かもしれ'  [row 15]
L 14 — entropy 16.3595 bits, top-1 'かもしれ'  [row 16]
L 15 — entropy 16.5170 bits, top-1 'のではない'  [row 17]
L 16 — entropy 16.4908 bits, top-1 'iéndo'  [row 18]
L 17 — entropy 16.2127 bits, top-1 'iéndo'  [row 19]
L 18 — entropy 16.1093 bits, top-1 '有期徒'  [row 20]
L 19 — entropy 15.7574 bits, top-1 '有期徒'  [row 21]
L 20 — entropy 16.1290 bits, top-1 '有期徒'  [row 22]
L 21 — entropy 16.1558 bits, top-1 '有期徒'  [row 23]
L 22 — entropy 15.9799 bits, top-1 '有期徒'  [row 24]
L 23 — entropy 16.4015 bits, top-1 '.myapplication'  [row 25]
L 24 — entropy 15.9989 bits, top-1 'iéndo'  [row 26]
L 25 — entropy 15.3506 bits, top-1 'hế'  [row 27]
L 26 — entropy 15.9435 bits, top-1 'iéndo'  [row 28]
L 27 — entropy 15.7559 bits, top-1 'iéndo'  [row 29]
L 28 — entropy 15.7500 bits, top-1 '.myapplication'  [row 30]
L 29 — entropy 15.8849 bits, top-1 '.myapplication'  [row 31]
L 30 — entropy 16.1225 bits, top-1 '.myapplication'  [row 32]
L 31 — entropy 16.1700 bits, top-1 '.myapplication'  [row 33]
L 32 — entropy 16.1708 bits, top-1 '.myapplication'  [row 34]
L 33 — entropy 16.4191 bits, top-1 'hế'  [row 35]
L 34 — entropy 16.2001 bits, top-1 'iéndo'  [row 36]
L 35 — entropy 16.4550 bits, top-1 'hế'  [row 37]
L 36 — entropy 16.4078 bits, top-1 'iéndo'  [row 38]
L 37 — entropy 16.2100 bits, top-1 'iéndo'  [row 39]
L 38 — entropy 16.4904 bits, top-1 'hế'  [row 40]
L 39 — entropy 16.4177 bits, top-1 'iéndo'  [row 41]
L 40 — entropy 16.1916 bits, top-1 'iéndo'  [row 42]
L 41 — entropy 16.4652 bits, top-1 'hế'  [row 43]
L 42 — entropy 16.5948 bits, top-1 'hế'  [row 44]
L 43 — entropy 16.4974 bits, top-1 'hế'  [row 45]
L 44 — entropy 16.6553 bits, top-1 '続きを読む'  [row 46]
L 45 — entropy 16.8770 bits, top-1 '国际在线'  [row 47]
L 46 — entropy 17.0023 bits, top-1 '国际在线'  [row 48]
L 47 — entropy 17.0133 bits, top-1 '主义思想'  [row 49]
L 48 — entropy 17.0217 bits, top-1 '主义思想'  [row 50]
L 49 — entropy 17.0217 bits, top-1 ' reuseIdentifier'  [row 51]
L 50 — entropy 16.9678 bits, top-1 'uckets'  [row 52]
L 51 — entropy 16.9723 bits, top-1 ' "'  [row 53]
L 52 — entropy 17.0086 bits, top-1 '"'  [row 54]
L 53 — entropy 16.9266 bits, top-1 '"'  [row 55]
L 54 — entropy 16.9081 bits, top-1 '"'  [row 56]
L 55 — entropy 16.9423 bits, top-1 '"'  [row 57]
L 56 — entropy 16.9382 bits, top-1 '"'  [row 58]
L 57 — entropy 16.8408 bits, top-1 ' "'  [row 59]
L 58 — entropy 16.9148 bits, top-1 ' "'  [row 60]
L 59 — entropy 16.9201 bits, top-1 ' "'  [row 61]
L 60 — entropy 16.8861 bits, top-1 ' ''  [row 62]
L 61 — entropy 16.9030 bits, top-1 ' ''  [row 63]
L 62 — entropy 16.8336 bits, top-1 ' "'  [row 64]
L 63 — entropy 16.8908 bits, top-1 ' "'  [row 65]
L 64 — entropy 16.8947 bits, top-1 ' "'  [row 66]
L 65 — entropy 16.8689 bits, top-1 ' "'  [row 67]
L 66 — entropy 16.8994 bits, top-1 ' "'  [row 68]
L 67 — entropy 16.8932 bits, top-1 ' "'  [row 69]
L 68 — entropy 16.7786 bits, top-1 ' "'  [row 70]
L 69 — entropy 16.8758 bits, top-1 ' "'  [row 71]
L 70 — entropy 16.7866 bits, top-1 ' "'  [row 72]
L 71 — entropy 16.5046 bits, top-1 ' "'  [row 73]
L 72 — entropy 16.6499 bits, top-1 ' "'  [row 74]
L 73 — entropy 15.7867 bits, top-1 ' "'  [row 75]
L 74 — entropy 16.0809 bits, top-1 ' "'  [row 76]
L 75 — entropy 13.3499 bits, top-1 ' "'  [row 77]
L 76 — entropy 14.7428 bits, top-1 ' "'  [row 78]
L 77 — entropy 10.8478 bits, top-1 ' "'  [row 79]
L 78 — entropy 15.3978 bits, top-1 ' "'  [row 80]
L 79 — entropy 16.6656 bits, top-1 ' "'  [row 81]
L 80 — entropy 4.1158 bits, top-1 ' Berlin'  [row 82]

- Bold semantic layer: L 80 where is_answer = True [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:138]. For reference, gold_answer.string = "Berlin" [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1350–1355].

Control margin (JSON control_summary)
- first_control_margin_pos = 0; max_control_margin = 0.2070 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1331–1333].

Ablation (no‑filler)
- L_copy_orig = null, L_sem_orig = 80; L_copy_nf = null, L_sem_nf = 80; ΔL_copy = null; ΔL_sem = 0 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1308–1315].

Entropy deltas
- ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n/a (strict L_copy null).
- Soft ΔHk (bits) for k∈{1,2,3}: n/a (all L_copy_soft[k] null).

Confidence milestones (pure CSV)
- p_top1 > 0.30 at layer 77 (p = 0.3091); no layer > 0.60; final-layer p_top1 = 0.3395 [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:77,138].

Rank milestones (JSON)
- rank ≤ 10 at L 74; rank ≤ 5 at L 78; rank ≤ 1 at L 80 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:979–983].

KL milestones (JSON + CSV)
- first_kl_below_1.0 at L 80; first_kl_below_0.5 at L 80 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:979–980]. Final KL≈0: 0.0001091 bits [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:138].

Cosine milestones (JSON)
- First cos_to_final ≥ 0.2: L 0; ≥ 0.4: L 0; ≥ 0.6: L 53; final cos_to_final = 1.0000 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1066–1071; 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:138].

Surface → meaning (JSON)
- L_surface_to_meaning_norm = 80 with answer_mass_at_L = 0.3395 and echo_mass_at_L = 0.0446 (answer_minus_echo = 0.2948; ratio ≈ 7.61) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1055–1057; 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:138].
- Coverage: L_topk_decay_norm = 0; topk_prompt_mass_at_L_norm = 0.0; τ = 0.33 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1061–1065].

Copy robustness (threshold sweep)
- stability = "none"; earliest strict copy at τ=0.70 and τ=0.95 are null (norm_only_flags also null) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1028–1054].

Prism Sidecar Analysis
- Presence/compatibility: present = true; compatible = true; sampled sidecar written [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:825–836].
- Early-depth stability: KL(P_layer||P_final) drops at early/mid depths vs baseline: Δp25 = 3.16 bits, Δp50 = 2.83 bits; Δp75 = −0.54 bits [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:856–871].
- Rank milestones (prism pure): no rank ≤{10,5,1} within stack (all null) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:845–853].
- Top‑1 agreement: no qualitative early-depth flips to the correct answer; prism improves KL but not rank.
- Copy flags: none fire under Prism at early layers (see sidecar header and early rows [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token-prism.csv:1–25]).
- Verdict: Neutral — meaningful early KL reduction without earlier rank milestones.

**Qualitative Patterns & Anomalies**
Negative control (“Berlin is the capital of”):
- Top‑5 shows strong “ Germany” at 0.7695 with low-entropy continuation: " Germany", 0.7695; " the", 0.0864; " which", 0.0491 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:13–23].

Temperature robustness:
- At T = 0.1, “ Berlin” rank 1 (p = 0.9526; entropy 0.275 bits) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:670–679]. At T = 2.0, “ Berlin” remains in top‑k but with p = 0.0162; entropy 15.013 bits [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:736–744].

Important-word trajectory and surface tokens:
- “Berlin” first appears in the answer position top‑5 near L 78 (answer_rank = 5; p_answer = 0.00334) and strengthens by L 79 (rank = 2) before becoming top‑1 at L 80 [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:136–138].
- In the context positions, “ Berlin” enters some top‑5 lists late (e.g., layer 72 at the token “ is”: includes ‘ Berlin’) [001_layers_baseline/run-latest/output-Qwen2.5-72B-records.csv:3856].
- Final head still assigns small mass to prompt words and punctuation alongside the answer: e.g., final top‑k includes " \"", ",", ":", " '" and low mass on " Germany" and " capital" [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1147–1216].

One‑word instruction ablation:
- Removing “simply” does not shift semantics: L_sem_nf = 80 (ΔL_sem = 0) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1308–1315].

Rest‑mass sanity:
- Rest_mass drops sharply only at the final layer (0.2977 at L 80; earlier layers ≫ 0.9), consistent with a late sharpening under the norm lens [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:138].

Rotation vs amplification:
- KL-to-final decreases only at the end (first ≤1.0 bit at L 80), while cos_to_final is already ≥0.4 at L 0 and ≥0.6 by L 53, indicating “early direction, late calibration” under the norm lens [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:979–980,1066–1071]. Final-head calibration is sound (kl≈0, temp_est=1.0) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1104–1111].

Lens sanity cautions:
- Raw‑vs‑norm check: lens_artifact_risk = high; max_kl_norm_vs_raw_bits = 19.91; sample at L 61 shows massive raw/norm divergence (p_top1_raw 0.792 vs p_top1_norm 0.00043) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1237–1290]. Raw‑vs‑norm window marks norm‑only semantics at L 80 with extreme KL window (83.315 bits) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1081–1100]. Prefer rank thresholds and within‑model statements.

Checklist
- RMS lens? Yes (RMSNorm; norm lens used) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:810–814,807].
- LayerNorm bias removed? not_needed_rms_model [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:812].
- Entropy rise at unembed? Final entropy 4.1356 bits; much higher mid‑stack vs teacher (teacher_entropy_bits = 4.1356) — drift noted in table/CSV [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:98].
- FP32 un‑embed promoted? use_fp32_unembed = true; unembed_dtype = torch.float32 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:808–809].
- Punctuation / markup anchoring? Yes — mid‑layers dominated by quotes/commas [table rows L 51–79].
- Copy‑reflex? ✗ — no strict or soft copy flags in L0–3.
- Grammatical filler anchoring? Yes — fillers/punctuation frequently top‑1 in early/mid layers [table L 51–79].

**Limitations & Data Quirks**
- Norm‑only semantics in the late window (L 80) with very high KL(raw||norm) in the window (83.3 bits) and lens_artifact_risk = high indicate lens sensitivity; follow measurement_guidance to prefer rank milestones and within‑model comparisons [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1081–1100,1296–1298,1340–1347].
- KL is lens‑sensitive; although final‑head calibration is good (kl≈0), cross‑model probability comparisons are discouraged per guidance.
- Rest_mass is top‑k coverage only; early high values and late drop should not be used as lens fidelity metrics.
- Surface mass depends on tokenizer idiosyncrasies; prefer within‑model trajectories (answer_minus_echo > 0 at L 80) [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:138].
- Raw‑lens sampling mode = sample, so raw-vs-norm findings are a sampled sanity rather than exhaustive [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1237–1239].

**Model Fingerprint**
- Qwen2.5‑72B: collapse at L 80; final entropy 4.12 bits; ‘ Berlin’ only enters top‑5 at L 78 and becomes top‑1 at L 80.

---
Produced by OpenAI GPT-5
