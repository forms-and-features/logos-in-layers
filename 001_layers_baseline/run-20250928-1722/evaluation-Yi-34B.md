# Evaluation Report: 01-ai/Yi-34B

*Run executed on: 2025-09-28 17:22:48*
## 1. Overview

Yi-34B (01-ai/Yi-34B) evaluated on 2025-09-28 against a one-word factual recall probe (“Berlin”). The run captures layer-by-layer next-token distributions under a norm lens with FP32 unembedding, emitting per-layer entropy, KL-to-final, cosine-to-final, rank milestones, copy/soft-copy flags, and surface/meaning metrics. Sidecars include Prism and Tuned-Lens for calibration/rotation comparisons.

## 2. Method sanity-check

Diagnostics confirm RMSNorm with norm lens enabled and FP32 unembedding; positional info indicates rotary (token-only) at layer 0. The exact context prompt ends with “called simply” (no trailing space). Examples:

- “use_norm_lens": true … "use_fp32_unembed": true … "unembed_dtype": "torch.float32” (RMSNorm; rotary) [001_layers_baseline/run-latest/output-Yi-34B.json:807–817]
- “context_prompt": "Give the city name only, plain text. The capital of Germany is called simply” [001_layers_baseline/run-latest/output-Yi-34B.json:817]

Copy detection config and flags are present and consistent across JSON/CSV:
- “copy_thresh": 0.95, "copy_window_k": 1, "copy_match_level": "id_subsequence” [001_layers_baseline/run-latest/output-Yi-34B.json:915–918]
- “copy_soft_config": { "threshold": 0.5, "window_ks": [1,2,3], "extra_thresholds": [] } [001_layers_baseline/run-latest/output-Yi-34B.json:839–847]
- “copy_flag_columns": ["copy_strict@0.95","copy_soft_k1@0.5","copy_soft_k2@0.5","copy_soft_k3@0.5"] [001_layers_baseline/run-latest/output-Yi-34B.json:1401–1405]

Gold alignment is ID-based and ok for both probes: “gold_alignment": "ok” [001_layers_baseline/run-latest/output-Yi-34B.json:978, 1428]. Control prompt present: France→Paris [001_layers_baseline/run-latest/output-Yi-34B.json:1415–1426]. Ablation summary exists with both variants: “ablation_summary”: { “L_copy_orig": null, “L_sem_orig": 44, “L_copy_nf": null, “L_sem_nf": 44, “delta_L_sem": 0 } [001_layers_baseline/run-latest/output-Yi-34B.json:1407–1414]. For the main table, rows are filtered to prompt_id=pos, prompt_variant=orig (pure CSV).

Summary indices (bits, ranks refer to gold ID): “first_kl_below_0.5": 60, “first_kl_below_1.0": 60, “first_rank_le_1": 44, “first_rank_le_5": 44, “first_rank_le_10": 43 [001_layers_baseline/run-latest/output-Yi-34B.json:918–922]. Units for KL/entropy are bits throughout.

Last-layer head calibration is excellent: “kl_to_final_bits": 0.000278…, “top1_agree": true, “p_top1_lens": 0.5555 vs “p_top1_model": 0.5627, “temp_est": 1.0, “warn_high_last_layer_kl": false [001_layers_baseline/run-latest/output-Yi-34B.json:979–987,997]. Final-row CSV also shows kl_to_final_bits≈0.00028 [001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:60].

Lens sanity (raw vs norm): sampled check indicates high artifact risk and a norm-only semantic layer at L=46 — caution warranted for pre-final “early semantics”: “first_norm_only_semantic_layer": 46, “max_kl_norm_vs_raw_bits": 80.5716, “lens_artifact_risk": "high” [001_layers_baseline/run-latest/output-Yi-34B.json:1396–1399]. Norm temperature diagnostics present with snapshots: “kl_to_final_bits_norm_temp@25%": 12.2128 at L15; @50%: 12.2593 at L30; @75%: 6.7681 at L45 [001_layers_baseline/run-latest/output-Yi-34B.json:999–1010].

Copy-collapse flags: strict never fires (no row with copy_collapse=True). Soft-copy (τ_soft=0.5, k∈{1,2,3}) also never fires in L0–L3 (or elsewhere) in the pure CSV. ✓ rule: not triggered.

## 3. Quantitative findings

Per-layer summary (pos/orig only). Bold marks L_semantic = 44 (first is_answer=True for the gold ID).

| Layer | Entropy (bits) | Top-1 token |
|---:|---:|:---|
| 0 | 15.962 |  Denote |
| 1 | 15.942 | . |
| 2 | 15.932 | . |
| 3 | 15.839 | MTY |
| 4 | 15.826 | MTY |
| 5 | 15.864 | MTY |
| 6 | 15.829 | MTQ |
| 7 | 15.862 | MTY |
| 8 | 15.873 | 其特征是 |
| 9 | 15.836 | 审理终结 |
| 10 | 15.797 | ~\\ |
| 11 | 15.702 | ~\\ |
| 12 | 15.774 | ~\\ |
| 13 | 15.784 | 其特征是 |
| 14 | 15.739 | 其特征是 |
| 15 | 15.753 | 其特征是 |
| 16 | 15.714 | 其特征是 |
| 17 | 15.714 | 其特征是 |
| 18 | 15.716 | 其特征是 |
| 19 | 15.696 | ncase |
| 20 | 15.604 | ncase |
| 21 | 15.609 | ODM |
| 22 | 15.620 | ODM |
| 23 | 15.602 | ODM |
| 24 | 15.548 | ODM |
| 25 | 15.567 | ODM |
| 26 | 15.585 | ODM |
| 27 | 15.227 | ODM |
| 28 | 15.432 | MTU |
| 29 | 15.467 | ODM |
| 30 | 15.551 | ODM |
| 31 | 15.531 |  版的 |
| 32 | 15.455 | MDM |
| 33 | 15.455 | XFF |
| 34 | 15.477 | XFF |
| 35 | 15.471 | Mpc |
| 36 | 15.433 | MDM |
| 37 | 15.454 | MDM |
| 38 | 15.486 | MDM |
| 39 | 15.504 | MDM |
| 40 | 15.528 | MDM |
| 41 | 15.519 | MDM |
| 42 | 15.535 | keV |
| 43 | 15.518 |  " |
| 44 | 15.327 |  Berlin |
| 45 | 15.293 |  Berlin |
| 46 | 14.834 |  Berlin |
| 47 | 14.731 |  Berlin |
| 48 | 14.941 |  Berlin |
| 49 | 14.696 |  Berlin |
| 50 | 14.969 |  Berlin |
| 51 | 14.539 |  Berlin |
| 52 | 15.137 |  Berlin |
| 53 | 14.870 |  Berlin |
| 54 | 14.955 |  Berlin |
| 55 | 14.932 |  Berlin |
| 56 | 14.745 |  Berlin |
| 57 | 14.748 |   |
| 58 | 13.457 |   |
| 59 | 7.191 |   |
| 60 | 2.981 |  Berlin |

Control margin (JSON control_summary): first_control_margin_pos = 1; max_control_margin = 0.58357 [001_layers_baseline/run-latest/output-Yi-34B.json:1431–1432].

Ablation (no-filler): L_copy_orig = null, L_sem_orig = 44; L_copy_nf = null, L_sem_nf = 44; ΔL_copy = null, ΔL_sem = 0 [001_layers_baseline/run-latest/output-Yi-34B.json:1407–1414]. Interpretation: no shift from removing “simply”.

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n/a (L_copy = null). Soft ΔHₖ (k∈{1,2,3}) = n/a (no soft-copy hits).

Confidence milestones (pure CSV, pos/orig): p_top1 > 0.30 at layer 60; p_top1 > 0.60 never; final-layer p_top1 = 0.5555 [001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:60].

Rank milestones (JSON): rank ≤10 at L=43; ≤5 at L=44; ≤1 at L=44 [001_layers_baseline/run-latest/output-Yi-34B.json:920–922].

KL milestones (JSON): first_kl_below_1.0 at L=60; first_kl_below_0.5 at L=60. KL decreases with depth and ≈0 at final; “kl_to_final_bits”: 0.000278… [001_layers_baseline/run-latest/output-Yi-34B.json:918–921,980].

Cosine milestones (pure CSV): cos_to_final ≥ 0.2 at L=1; ≥0.4 at L=44; ≥0.6 at L=51; final cos_to_final ≈ 1.0000 [001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv].

Tuned‑Lens (sidecar) summary: last-layer agreement (kl_after_temp≈0) holds in baseline; tuned reduces KL substantially at mid-depths but delays top‑1 slightly. ΔKL (norm − tuned) at depth percentiles: L≈15: Δ≈5.88 bits; L≈30: Δ≈6.10 bits; L≈45: Δ≈8.39 bits [001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv, …-tuned.csv]. Tuned ranks: first_rank≤{10,5,1} = {44,44,46} vs baseline {43,44,44}. Norm temperature per-layer present with snapshots (see above). Surface→meaning: L_surface_to_meaning_norm = 51, answer_mass_at_L ≈ 0.0597, echo_mass_at_L ≈ 0.0063 [001_layers_baseline/run-latest/output-Yi-34B.json:967–976]; tuned: L_surface_to_meaning_tuned = 46, answer_mass_at_L ≈ 0.2812, echo_mass_at_L ≈ 0.0653 [001_layers_baseline/run-latest/output-Yi-34B.json:1225–1233]. Geometry: L_geom_norm = 46 with cos_to_answer≈0.125, cos_to_prompt_max≈0.096 [001_layers_baseline/run-latest/output-Yi-34B.json:970–973]. Coverage: L_topk_decay_norm = 0; L_topk_decay_tuned = 1 [001_layers_baseline/run-latest/output-Yi-34B.json:973–975,1231–1233]. Teacher entropy drift (baseline pure): at L=30, entropy − teacher_entropy_bits ≈ 12.61 bits; at final, ≈0.04 bits [001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv].

Prism Sidecar Analysis. Presence and compatibility confirmed [001_layers_baseline/run-latest/output-Yi-34B.json:825–833]. Early-depth stability (KL drop): at L15, baseline KL≈13.12 vs Prism≈12.18 (−0.94 bits); at L30, baseline≈13.54 vs Prism≈12.17 (−1.36 bits). At L45, Prism is worse (+1.01 bits), and at final, Prism’s KL≈13.25 bits (not calibrated to the final head) [001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv, …-prism.csv]. Rank milestones under Prism do not reach ≤10 anywhere (none). Cosine drift at mid-depths shows earlier stabilization magnitude than norm at L≈15–30, but diverges near late layers. Copy flags do not spuriously flip (no changes at L0–L3). Verdict: Neutral — early KL improves modestly but no rank milestone benefit; final calibration misaligned.

## 4. Qualitative patterns & anomalies

Negative control (Berlin→country). For “Berlin is the capital of” the model outputs “ Germany” with high confidence: > “ Germany, 0.8398; the, 0.0537; which, 0.0288; what, 0.0120; Europe, 0.0060” [001_layers_baseline/run-latest/output-Yi-34B.json:10–31]. No semantic leakage of “Berlin”.

Important‑word trajectory (records CSV; IMPORTANT_WORDS = ["Germany", "Berlin", "capital", "Answer", "word", "simply"]). Around the last context token (“ simply”, pos=16), ‘Berlin’ becomes top‑1 at L44: > “… simply … ‘ Berlin’, 0.00846” [001_layers_baseline/run-latest/output-Yi-34B-records.csv:788]; it strengthens across adjacent positions at L44–47 (e.g., pos=14 “ is”: ‘Berlin’, 0.01049 [001_layers_baseline/run-latest/output-Yi-34B-records.csv:786]; pos=15 “ called”: ‘Berlin’, 0.00826 [001_layers_baseline/run-latest/output-Yi-34B-records.csv:787]). At L46 pos=16: > “… ‘ Berlin’, 0.03455” [001_layers_baseline/run-latest/output-Yi-34B-records.csv:824]. ‘capital’ co-occurs in late layers (e.g., L45 pos=14 top‑1 “ capital”, 0.00863 [001_layers_baseline/run-latest/output-Yi-34B-records.csv:804]). This suggests late consolidation around the gold token with context-aligned neighbors.

Rotation vs amplification. KL-to-final steadily declines, while cos_to_final rises early (≥0.2 by L1) and crosses ≥0.4 at semantic onset (L44). This is consistent with “early direction, late calibration”: directions point toward the final head early, but calibrated probabilities (low KL, higher p_answer) arrive later. With last-layer KL≈0 and top‑1 agreement true, final-head calibration is good (no Gemma-like mismatch) [001_layers_baseline/run-latest/output-Yi-34B.json:979–987].

Temperature robustness. At T=0.1, ‘Berlin’ is rank‑1 with p≈0.9999996 and near‑zero entropy: > “… ‘ Berlin’, 0.99999964 … entropy = 7.1e−06 bits” [001_layers_baseline/run-latest/output-Yi-34B.json:670–676,671]. At T=2.0, ‘Berlin’ remains top‑1 but low p≈0.0488 with high entropy 12.49 bits [001_layers_baseline/run-latest/output-Yi-34B.json:736–743,738].

Rest‑mass sanity. Rest_mass is high early (top‑20 coverage small) and decreases after semantics; max after L_semantic is 0.981 at L44; final rest_mass ≈ 0.175 [001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:44,60]. As noted, rest_mass is top‑k coverage only, not a lens fidelity metric.

Head calibration (final layer). “warn_high_last_layer_kl": false with “temp_est": 1.0; lens vs model top‑1 probs: 0.5555 vs 0.5627; “kl_to_final_bits": 0.000278 [001_layers_baseline/run-latest/output-Yi-34B.json:979–987].

Lens sanity. Raw‑vs‑norm summary marks “lens_artifact_risk": "high”, “max_kl_norm_vs_raw_bits": 80.57, and “first_norm_only_semantic_layer": 46 [001_layers_baseline/run-latest/output-Yi-34B.json:1396–1399]. Treat any pre‑final “early semantics” cautiously; rely on rank milestones and within‑model trends.

Checklist
- RMS lens? ✓ [001_layers_baseline/run-latest/output-Yi-34B.json:810–817]
- LayerNorm bias removed? ✓ (RMS model; not needed) [001_layers_baseline/run-latest/output-Yi-34B.json:812]
- Entropy rise at unembed? ✓ (final lens entropy 2.98 > teacher 2.94) [001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:60]
- FP32 un-embed promoted? ✓ [001_layers_baseline/run-latest/output-Yi-34B.json:808–809]
- Punctuation / markup anchoring? ✓ early layers (e.g., “.”, special/markup tokens) [001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:1–10]
- Copy-reflex? ✗ (no strict or soft hits in L0–3) [001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv]
- Grammatical filler anchoring? ✗ (early top‑1 not in {“is”, “the”, “a”, “of”})

## 5. Limitations & data quirks

- Rest_mass reflects top‑k coverage, not fidelity; early high values are expected and do not imply distortion.
- Raw‑vs‑norm lens differences are sampled (“mode": "sample”); “lens_artifact_risk": high with a norm‑only semantic layer at L=46 — treat pre‑final semantics cautiously and prefer rank milestones [001_layers_baseline/run-latest/output-Yi-34B.json:1339–1341,1396–1399].
- KL is lens‑sensitive; here final‑layer calibration is good (KL≈0), but cross‑model probability comparisons should still prefer rank‑based milestones.
- Surface‑mass and echo‑mass depend on tokenizer granularity; use within‑model trends rather than absolute comparisons.

## 6. Model fingerprint

Yi‑34B: collapse at L 44; final entropy ≈2.98 bits; early cosine alignment (≥0.2 by L1) with calibrated semantics late.

---
Produced by OpenAI GPT-5
