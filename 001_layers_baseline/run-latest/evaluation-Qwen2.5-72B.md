# Evaluation Report: Qwen/Qwen2.5-72B
1. Overview
Qwen/Qwen2.5-72B (80 layers) probed on CPU. This run captures layer-by-layer entropy, top-1 behavior, ID-level gold alignment to “Berlin”, and final-head calibration using a norm-based logit lens.

2. Method sanity-check
Diagnostics confirm the intended RMS norm lens and rotary positions: “use_norm_lens: true; unembed_dtype: torch.float32” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:807–809], and “context_prompt: ‘Give the city name only, plain text. The capital of Germany is called simply’” with no trailing space [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:817]. Copy rule parameters present: copy_thresh=0.95, copy_window_k=1, copy_match_level=id_subsequence [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:837–839]. Gold alignment is ok [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:845–846]. Negative control fields exist (control_prompt and control_summary) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1032–1050]. Ablation summary present with both variants; L_sem_orig=80, L_sem_nf=80 (ΔL_sem=0) and L_copy fields null [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1024–1031]. For the main table below, rows are filtered to prompt_id=pos, prompt_variant=orig (pure CSV).

Summary indices: first_kl_below_0.5=80, first_kl_below_1.0=80, first_rank_le_1=80, first_rank_le_5=78, first_rank_le_10=74 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:840–844]. Units for KL/entropy are bits. Final-head calibration is good: final CSV kl_to_final_bits=0.000109… [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:138], and diagnostics.last_layer_consistency exists with top1_agree=true; p_top1_lens=0.3395 vs p_top1_model=0.3383; temp_est=1.0 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:846–864]. Lens sanity (raw lens check) mode=sample; lens_artifact_risk=high; max_kl_norm_vs_raw_bits=19.91; first_norm_only_semantic_layer=null [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1018–1022].

Copy-collapse flag check: no rows with copy_collapse=True in layers 0–3 (or anywhere) [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:2–6,134–138]. ✓ rule satisfied (no spurious firing).

3. Quantitative findings
L 0 — entropy 17.214 bits, top-1 ‘s’
L 1 — entropy 17.214 bits, top-1 ‘下一篇’
L 2 — entropy 17.143 bits, top-1 ‘ولوج’
L 3 — entropy 17.063 bits, top-1 ‘شدد’
L 4 — entropy 17.089 bits, top-1 ‘.myapplication’
L 5 — entropy 17.007 bits, top-1 ‘ستحق’
L 6 — entropy 17.031 bits, top-1 ‘.myapplication’
L 7 — entropy 16.937 bits, top-1 ‘.myapplication’
L 8 — entropy 16.798 bits, top-1 ‘.myapplication’
L 9 — entropy 16.120 bits, top-1 ‘ستحق’
L 10 — entropy 16.501 bits, top-1 ‘.myapplication’
L 11 — entropy 16.718 bits, top-1 ‘.myapplication’
L 12 — entropy 16.778 bits, top-1 ‘かもしれ’
L 13 — entropy 16.631 bits, top-1 ‘かもしれ’
L 14 — entropy 16.359 bits, top-1 ‘かもしれ’
L 15 — entropy 16.517 bits, top-1 ‘のではない’
L 16 — entropy 16.491 bits, top-1 ‘iéndo’
L 17 — entropy 16.213 bits, top-1 ‘iéndo’
L 18 — entropy 16.109 bits, top-1 ‘有期徒’
L 19 — entropy 15.757 bits, top-1 ‘有期徒’
L 20 — entropy 16.129 bits, top-1 ‘有期徒’
L 21 — entropy 16.156 bits, top-1 ‘有期徒’
L 22 — entropy 15.980 bits, top-1 ‘有期徒’
L 23 — entropy 16.401 bits, top-1 ‘.myapplication’
L 24 — entropy 15.999 bits, top-1 ‘iéndo’
L 25 — entropy 15.351 bits, top-1 ‘hế’
L 26 — entropy 15.944 bits, top-1 ‘iéndo’
L 27 — entropy 15.756 bits, top-1 ‘iéndo’
L 28 — entropy 15.750 bits, top-1 ‘.myapplication’
L 29 — entropy 15.885 bits, top-1 ‘.myapplication’
L 30 — entropy 16.123 bits, top-1 ‘.myapplication’
L 31 — entropy 16.170 bits, top-1 ‘.myapplication’
L 32 — entropy 16.171 bits, top-1 ‘.myapplication’
L 33 — entropy 16.419 bits, top-1 ‘hế’
L 34 — entropy 16.200 bits, top-1 ‘iéndo’
L 35 — entropy 16.455 bits, top-1 ‘hế’
L 36 — entropy 16.408 bits, top-1 ‘iéndo’
L 37 — entropy 16.210 bits, top-1 ‘iéndo’
L 38 — entropy 16.490 bits, top-1 ‘hế’
L 39 — entropy 16.418 bits, top-1 ‘iéndo’
L 40 — entropy 16.192 bits, top-1 ‘iéndo’
L 41 — entropy 16.465 bits, top-1 ‘hế’
L 42 — entropy 16.595 bits, top-1 ‘hế’
L 43 — entropy 16.497 bits, top-1 ‘hế’
L 44 — entropy 16.655 bits, top-1 ‘続きを読む’
L 45 — entropy 16.877 bits, top-1 ‘国际在线’
L 46 — entropy 17.002 bits, top-1 ‘国际在线’
L 47 — entropy 17.013 bits, top-1 ‘主义思想’
L 48 — entropy 17.022 bits, top-1 ‘主义思想’
L 49 — entropy 17.022 bits, top-1 ‘ reuseIdentifier’
L 50 — entropy 16.968 bits, top-1 ‘uckets’
L 51 — entropy 16.972 bits, top-1 ‘ "’
L 52 — entropy 17.009 bits, top-1 ‘"’
L 53 — entropy 16.927 bits, top-1 ‘"’
L 54 — entropy 16.908 bits, top-1 ‘"’
L 55 — entropy 16.942 bits, top-1 ‘"’
L 56 — entropy 16.938 bits, top-1 ‘"’
L 57 — entropy 16.841 bits, top-1 ‘ "’
L 58 — entropy 16.915 bits, top-1 ‘ "’
L 59 — entropy 16.920 bits, top-1 ‘ "’
L 60 — entropy 16.886 bits, top-1 ‘ ''
L 61 — entropy 16.903 bits, top-1 ‘ ''
L 62 — entropy 16.834 bits, top-1 ‘ "’
L 63 — entropy 16.891 bits, top-1 ‘ "’
L 64 — entropy 16.895 bits, top-1 ‘ "’
L 65 — entropy 16.869 bits, top-1 ‘ "’
L 66 — entropy 16.899 bits, top-1 ‘ "’
L 67 — entropy 16.893 bits, top-1 ‘ "’
L 68 — entropy 16.779 bits, top-1 ‘ "’
L 69 — entropy 16.876 bits, top-1 ‘ "’
L 70 — entropy 16.787 bits, top-1 ‘ "’
L 71 — entropy 16.505 bits, top-1 ‘ "’
L 72 — entropy 16.650 bits, top-1 ‘ "’
L 73 — entropy 15.787 bits, top-1 ‘ "’
L 74 — entropy 16.081 bits, top-1 ‘ "’
L 75 — entropy 13.350 bits, top-1 ‘ "’
L 76 — entropy 14.743 bits, top-1 ‘ "’
L 77 — entropy 10.848 bits, top-1 ‘ "’
L 78 — entropy 15.398 bits, top-1 ‘ "’
L 79 — entropy 16.666 bits, top-1 ‘ "’
**L 80 — entropy 4.116 bits, top-1 ‘ Berlin’**

Control margin (JSON): first_control_margin_pos=0; max_control_margin=0.2070 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1048–1050].

Ablation (no‑filler): L_copy_orig=null, L_sem_orig=80; L_copy_nf=null, L_sem_nf=80; ΔL_copy=null, ΔL_sem=0 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1024–1031].

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n/a − 4.116 (L_copy is null; rely on rank milestones).

Confidence milestones (pure CSV):
p_top1 > 0.30 at layer 77; p_top1 > 0.60: none; final-layer p_top1 = 0.3395 [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:135,138].

Rank milestones (JSON): rank ≤ 10 at layer 74; rank ≤ 5 at layer 78; rank ≤ 1 at layer 80 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:842–844].

KL milestones (JSON/CSV): first_kl_below_1.0 at layer 80; first_kl_below_0.5 at layer 80; final KL ≈ 0 (0.000109 bits) with good last‑layer consistency [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:840–841,846–864; 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:138]. KL decreases with depth and matches the final head.

Cosine milestones (pure CSV): first cos_to_final ≥ 0.2 at layer 0; ≥ 0.4 at layer 0; ≥ 0.6 at layer 53; final cos_to_final = 1.0000 [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:2,54,138].

Prism Sidecar Analysis
Presence: prism_summary.compatible=true [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:819–831]; sidecar CSVs present.
Early-depth stability: at sample depths, Prism KL vs baseline (KL(P_layer||P_final), bits): L0 9.49 vs 9.39; L19 9.40 vs 13.20; L39 9.59 vs 11.91; L59 9.60 vs 9.12; L80 20.68 vs 0.0001 [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token-prism.csv:2,31,57,99,120; 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:2,21,97,117,138].
Rank milestones: Prism never reaches rank ≤ 10/5/1 for ‘Berlin’ (answer_rank stays >10^4) [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token-prism.csv:116–120]. Baseline reaches 10/5/1 at 74/78/80.
Top‑1 agreement: No early/mid-layer Prism→final top‑1 agreements emerge; Prism final top‑1 is not ‘ Berlin’ and is far from the baseline final distribution [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token-prism.csv:120; 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:868–951].
Cosine drift: Prism cos_to_final remains low at sampled layers (e.g., L80 cos≈0.092 vs baseline 1.000) [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token-prism.csv:120; 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:138].
Copy flags: No spurious copy_collapse flips under Prism at early layers [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token-prism.csv:2–14].
Verdict: Regressive — despite some mid-depth KL drops, Prism fails to calibrate to the final head (KL explodes at L80; rank milestones never reached), so it is not helpful for this model.

4. Qualitative patterns & anomalies
The model shows a clear late semantic collapse: ‘Berlin’ first becomes rank‑1 at L80 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:842]. Final‑head calibration is strong (kl_to_final_bits≈0), so final probabilities are reliable within‑model [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:846–864]. The trajectory suggests “early direction, late calibration”: cosine aligns early (cos≈0.59 at L0) while KL remains high (≈9.39 bits) and the answer rank is extremely poor (≈67k), consistent with direction rotation preceding probability amplification (cf. Tuned‑Lens 2303.08112) [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:2].

Negative control: For “Berlin is the capital of”, the model’s next‑token top‑5 includes “ Germany (0.7695), the (0.0864), which (0.0491), a (0.0125), what (0.0075)” — no ‘Berlin’ leakage [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:10–21,29–33].

Records CSV shows the evolution of “important words”: by L74–77 the generic top‑5 often contains punctuation and structural tokens, while ‘Berlin’ begins to appear within the layer top‑k around L74 and grows thereafter (e.g., “simply … Berlin, 0.00095” [001_layers_baseline/run-latest/output-Qwen2.5-72B-records.csv:3911]; later “simply … Berlin, 0.00481” [001_layers_baseline/run-latest/output-Qwen2.5-72B-records.csv:3999]). This is consistent with narrowing to the answer near the top of the stack. Semantically related tokens (‘capital’, country names) co-occur in late layers (e.g., L76 top‑5 includes capital alongside Berlin) [001_layers_baseline/run-latest/output-Qwen2.5-72B-records.csv:3966].

Instruction phrasing: Removing “simply” does not shift the collapse layer (L_sem_nf=80; ΔL_sem=0) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1024–1031]. This suggests low stylistic‑cue sensitivity for this prompt pair.

Rest‑mass sanity: Rest_mass is 0.2977 at the final layer (post‑semantic), within expectations for top‑k coverage; no spike after L_semantic [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:138].

Rotation vs amplification: KL decreases with depth and hits ~0 at L80 while p_answer increases from near‑zero to ~0.339; cosine aligns early (≥0.2 at L0) and strengthens (≥0.6 by L53), indicating an early directional lock-in followed by late calibration and amplitude growth [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:2,54,138].

Head calibration (final layer): warn_high_last_layer_kl=false; temp_est=1.0; kl_after_temp_bits≈0.0001 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:846–864].

Lens sanity: raw vs norm check indicates lens_artifact_risk=high and max_kl_norm_vs_raw_bits≈19.91; no norm‑only semantic layer flagged [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1018–1022]. Treat any pre‑final “early semantics” cautiously and prefer rank milestones.

Temperature robustness (side test): at T=2.0, ‘Berlin’ remains top‑k with p≈0.016; entropy rises to ≈15.01 bits [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:736–743].

Important‑word trajectory: ‘Berlin’ first penetrates any top‑5 around mid‑late layers (e.g., L74 “Berlin, 0.00296” under ‘is’) and stabilizes near the top by L77–80; ‘Germany’ appears in various intermediate layers but is largely overshadowed by structural/punctuation tokens until late [001_layers_baseline/run-latest/output-Qwen2.5-72B-records.csv:3909,3999; 15–210].

Checklist
– RMS lens? ✓ [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:810–812]
– LayerNorm bias removed? “not_needed_rms_model” ✓ [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:812]
– Entropy rise at unembed? No anomalous spike at final; entropy drops to 4.116 bits at L80 [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:138]
– FP32 un‑embed promoted? ✓ use_fp32_unembed=true; unembed_dtype=float32 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:808–809]
– Punctuation / markup anchoring? Present in mid/late layers (quotes/commas dominate top‑1) [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:51–72]
– Copy‑reflex? ✗ (no copy_collapse=True in layers 0–3) [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:2–6]
– Grammatical filler anchoring? Some early dominance by fillers/markup (e.g., “the”, quotes) [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:134–137]

5. Limitations & data quirks
Lens_artifact_risk=high in raw‑vs‑norm check (max_kl_norm_vs_raw_bits≈19.91); treat pre‑final signals cautiously and prefer rank milestones [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1018–1022]. KL is lens‑sensitive; final‑head calibration is good here, but cross‑model probability comparisons should still rely on rank/thresholds. Raw lens check is sample mode (not exhaustive) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:962–967]. Prism sidecar appears regressive for this model (final KL very high; no rank milestones), so baseline lens is preferred for conclusions.

6. Model fingerprint (one sentence)
Qwen‑2.5‑72B: collapse at L 80; final entropy 4.12 bits; ‘Berlin’ only becomes top‑1 at the last layer with early cosine alignment but late probability calibration.

---
Produced by OpenAI GPT-5
*Run executed on: 2025-09-21 17:13:26*
