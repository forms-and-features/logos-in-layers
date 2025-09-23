# Evaluation Report: 01-ai/Yi-34B

1. Overview  
Yi‑34B (60 layers). Run date 2025‑09‑23 (local timestamp recorded). This probe analyzes layer‑by‑layer next‑token predictions with a norm lens, tracking entropy (bits), KL to final (bits), cosine to final, copy flags, and answer rank for the gold token “Berlin”.

2. Method sanity‑check  
Diagnostics confirm the norm lens and positional encoding interpretation:
• "use_norm_lens": true [001_layers_baseline/run-latest/output-Yi-34B.json:807]  
• "layer0_position_info": "token_only_rotary_model" [001_layers_baseline/run-latest/output-Yi-34B.json:816]

The `context_prompt` ends with “called simply” (no trailing space):  
"context_prompt": "Give the city name only, plain text. The capital of Germany is called simply" [001_layers_baseline/run-latest/output-Yi-34B.json:817]

Copy/collapse instrumentation present in diagnostics and CSV:  
• "copy_flag_columns": ["copy_strict@0.95", "copy_soft_k1@0.5", "copy_soft_k2@0.5", "copy_soft_k3@0.5"] [001_layers_baseline/run-latest/output-Yi-34B.json:1077–1081]  
• Strict rule: τ=0.95, k=1, level="id_subsequence" — "copy_thresh": 0.95; "copy_window_k": 1; "copy_match_level": "id_subsequence" [001_layers_baseline/run-latest/output-Yi-34B.json:846–848]. Margin δ=0.10 (run parameter).  
• Soft config: threshold=0.5, window_ks=[1,2,3], extra_thresholds=[] [001_layers_baseline/run-latest/output-Yi-34B.json:833–841].  
• Summary indices: "L_copy": null, "L_copy_H": null, "L_semantic": 44, "delta_layers": null; "L_copy_soft": {1:null,2:null,3:null}; "delta_layers_soft": {1:null,2:null,3:null} [001_layers_baseline/run-latest/output-Yi-34B.json:842–869].  
• Gold alignment: "gold_alignment": "ok" [001_layers_baseline/run-latest/output-Yi-34B.json:898]. Control present: "control_prompt" … and "control_summary" [001_layers_baseline/run-latest/output-Yi-34B.json:1106–1108]. Ablation present: "ablation_summary" with both variants [001_layers_baseline/run-latest/output-Yi-34B.json:1083–1088].

KL/entropy units are bits (column names: kl_to_final_bits; entropies ~log2|V|): final‑layer KL≈0: kl_to_final_bits=0.000278… at L60 [001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:63]. Last‑layer head calibration block present and aligned: { top1_agree=true, p_top1_lens=0.5555, p_top1_model=0.5627, temp_est=1.0, kl_after_temp_bits=0.000278… } [001_layers_baseline/run-latest/output-Yi-34B.json:912–938].

Lens sanity (raw vs norm): mode="sample" with summary { lens_artifact_risk="high", max_kl_norm_vs_raw_bits=80.57, first_norm_only_semantic_layer=46 } [001_layers_baseline/run-latest/output-Yi-34B.json:1072–1075]. Treat any pre‑final “early semantics” cautiously; prefer rank milestones.  
Copy‑collapse flags (strict/soft): no firings in L0–3 (or anywhere) in the pure CSV; earliest strict/soft flags are null.

3. Quantitative findings  
Filter: prompt_id=pos, prompt_variant=orig.

Control margin (JSON): first_control_margin_pos=1; max_control_margin=0.5836 [001_layers_baseline/run-latest/output-Yi-34B.json:1106–1108].

Main table (entropy in bits; top‑1 token). Bold = semantic layer (first is_answer=True).  

| Layer | Entropy (bits) | Top‑1 |
|---|---:|---|
| L 0 | 15.962 |  Denote |
| L 1 | 15.942 | . |
| L 2 | 15.932 | . |
| L 3 | 15.839 | MTY |
| L 4 | 15.826 | MTY |
| L 5 | 15.864 | MTY |
| L 6 | 15.829 | MTQ |
| L 7 | 15.862 | MTY |
| L 8 | 15.873 | 其特征是 |
| L 9 | 15.836 | 审理终结 |
| L 10 | 15.797 | ~\\\\ |
| L 11 | 15.702 | ~\\\\ |
| L 12 | 15.774 | ~\\\\ |
| L 13 | 15.784 | 其特征是 |
| L 14 | 15.739 | 其特征是 |
| L 15 | 15.753 | 其特征是 |
| L 16 | 15.714 | 其特征是 |
| L 17 | 15.714 | 其特征是 |
| L 18 | 15.716 | 其特征是 |
| L 19 | 15.696 | ncase |
| L 20 | 15.604 | ncase |
| L 21 | 15.609 | ODM |
| L 22 | 15.620 | ODM |
| L 23 | 15.602 | ODM |
| L 24 | 15.548 | ODM |
| L 25 | 15.567 | ODM |
| L 26 | 15.585 | ODM |
| L 27 | 15.227 | ODM |
| L 28 | 15.432 | MTU |
| L 29 | 15.467 | ODM |
| L 30 | 15.551 | ODM |
| L 31 | 15.531 |  版的 |
| L 32 | 15.455 | MDM |
| L 33 | 15.455 | XFF |
| L 34 | 15.477 | XFF |
| L 35 | 15.471 | Mpc |
| L 36 | 15.433 | MDM |
| L 37 | 15.454 | MDM |
| L 38 | 15.486 | MDM |
| L 39 | 15.504 | MDM |
| L 40 | 15.528 | MDM |
| L 41 | 15.519 | MDM |
| L 42 | 15.535 | keV |
| L 43 | 15.518 |  " |
| L 44 | 15.327 | ** Berlin** |
| L 45 | 15.293 | ** Berlin** |
| L 46 | 14.834 | ** Berlin** |
| L 47 | 14.731 | ** Berlin** |
| L 48 | 14.941 | ** Berlin** |
| L 49 | 14.696 | ** Berlin** |
| L 50 | 14.969 | ** Berlin** |
| L 51 | 14.539 | ** Berlin** |
| L 52 | 15.137 | ** Berlin** |
| L 53 | 14.870 | ** Berlin** |
| L 54 | 14.955 | ** Berlin** |
| L 55 | 14.932 | ** Berlin** |
| L 56 | 14.745 | ** Berlin** |
| L 57 | 14.748 |   |
| L 58 | 13.457 |   |
| L 59 | 7.191 |   |
| L 60 | 2.981 | ** Berlin** |

Ablation (no‑filler) [JSON]:  
• L_copy_orig=null; L_sem_orig=44; L_copy_nf=null; L_sem_nf=44; ΔL_copy=null; ΔL_sem=0 [001_layers_baseline/run-latest/output-Yi-34B.json:1083–1088].

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n/a (L_copy=null).  
Soft ΔHk (bits) for k∈{1,2,3}: n/a (all L_copy_soft[k]=null).  
Confidence milestones (pure CSV, pos/orig): p_top1>0.30 at L60; p_top1>0.60 not reached; final p_top1=0.5555 [001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:63].  
Rank milestones (JSON): rank≤10 at L43; rank≤5 at L44; rank≤1 at L44 [001_layers_baseline/run-latest/output-Yi-34B.json:851–853].  
KL milestones (JSON): first_kl_below_1.0 at L60; first_kl_below_0.5 at L60 [001_layers_baseline/run-latest/output-Yi-34B.json:849–850]. KL decreases with depth and is ≈0 at final (CSV row 63).  
Cosine milestones (pure CSV): cos_to_final≥0.2 at L1, ≥0.4 at L44, ≥0.6 at L51; final cos=1.0000 [rows 2, 46, 53, 63 in CSV].

Prism Sidecar Analysis  
Prism present and marked compatible (k=512; layers=[embed,14,29,44]) [001_layers_baseline/run-latest/output-Yi-34B.json:824–831]. However, sidecar distributions are not calibrated to the final head:  
• Early/mid KL comparison (baseline vs Prism):  
  L0: 12.01 vs 12.13; L15: 13.12 vs 12.18; L30: 13.54 vs 12.17; L45: 11.16 vs 12.17; L60: 0.0003 vs 13.25 [computed from both pure CSVs].  
• Rank milestones: Prism never reaches rank≤1 (answer_rank stays large; e.g., L60 rank≈27969) while baseline hits rank=1 at L44 and final [Prism CSV; baseline JSON 851–853].  
• Top‑1 agreement: none at sampled depths; baseline has top‑1 Berlin at L≥44; Prism remains off‑manifold.
• Cosine drift: Prism cos_to_final stays near 0/negative at early/mid depths and remains ≈0 at L60 [Prism CSV].  
• Copy flags: no spurious flips under Prism (no copy flags at all).  
Verdict: Regressive — KL does not improve and rank milestones are much later/absent.

4. Qualitative patterns & anomalies  
Negative control (test prompt): “Berlin is the capital of” yields a clean top‑5 with “Germany” dominant:  
> " Germany", 0.8398; " the", 0.0537; " which", 0.0288; " what", 0.0120; " Europe", 0.0060 [001_layers_baseline/run-latest/output-Yi-34B.json:14–31]. No semantic leakage of “Berlin”.

Records (important words). At L43 on the final context token (“ simply”), top‑5 is still non‑answer: “… '"', 0.00133 … 'capital' …” [001_layers_baseline/run-latest/output-Yi-34B-records.csv:749]. At L44, “Berlin” becomes top‑1: “… ('Berlin', 0.00846) …” [001_layers_baseline/run-latest/output-Yi-34B-records.csv:766]; it remains top‑1 at L45–46 with strengthening mass: “… (0.00748)… (0.03455)” [001_layers_baseline/run-latest/output-Yi-34B-records.csv:783,800]. By L50 it sustains dominance with “capital/柏林” co‑activations [001_layers_baseline/run-latest/output-Yi-34B-records.csv:868]. This matches a rotation‑then‑amplification trajectory: cosine to final rises early (≥0.2 at L1) while KL remains high until late; semantics (rank 1) emerge at L44.

“One‑word” ablation: L_sem unchanged (44→44) and L_copy remains null [001_layers_baseline/run-latest/output-Yi-34B.json:1083–1088], suggesting stylistic filler does not shift collapse for this prompt/model.

Rest‑mass sanity: Rest_mass peaks just after semantic onset (0.9807 at L44) and then falls steadily to 0.175 at the final layer [pure CSV rows 46,63]. This pattern is consistent with narrowing the distribution while recording only top‑k mass.

Rotation vs amplification: Cosine increases early (≥0.2 by L1; ≥0.4 at L44), while KL to final remains large until the end; answer rank improves sharply at L44 (rank=1) and p_top1 only exceeds 0.30 at the final layer. This is “early direction, late calibration” consistent with tuned‑lens observations (cf. 2303.08112 for context).

Head calibration (final layer): Aligned — { top1_agree=true, p_top1_lens≈0.556 vs p_top1_model≈0.563, temp_est=1.0, kl_after_temp≈0.00028 } [001_layers_baseline/run-latest/output-Yi-34B.json:912–938]. Prefer rank‑based summaries for cross‑family comparisons.

Lens sanity: raw‑vs‑norm summary flags high artifact risk and a “norm‑only semantics” at L46 (first_norm_only_semantic_layer=46; max_kl_norm_vs_raw_bits≈80.57) [001_layers_baseline/run-latest/output-Yi-34B.json:1072–1075]. Treat any “early semantics” before final with caution; use reported rank milestones.

Temperature robustness (JSON):  
> T=0.1 → “Berlin” p=0.9999996, entropy≈7.1e‑06 [001_layers_baseline/run-latest/output-Yi-34B.json:669–690].  
> T=2.0 → “Berlin” p=0.0488, entropy≈12.49 [001_layers_baseline/run-latest/output-Yi-34B.json:736–784].

Checklist:  
✓ RMS lens (RMSNorm; norm lens applied) [001_layers_baseline/run-latest/output-Yi-34B.json:810–811,807]  
✓ LayerNorm bias removed/irrelevant (rms model) [001_layers_baseline/run-latest/output-Yi-34B.json:812]  
✓ FP32 un‑embed promoted [001_layers_baseline/run-latest/output-Yi-34B.json:809,815]  
✓ Entropy rises at unembed? Final prediction entropy=2.94 bits [001_layers_baseline/run-latest/output-Yi-34B.json:922]  
✗ Copy‑reflex (no strict/soft flags in L0–3).  
Punctuation / filler anchoring: early layers dominated by punctuation/garbage tokens (e.g., '.', 'MTY', 'ODM').  
Grammatical filler anchoring: not observed (no early top‑1 in {“is”, “the”, “a”, “of”}).

5. Limitations & data quirks  
• raw_lens_check.mode="sample"; lens_artifact_risk="high" with first_norm_only_semantic_layer=46; prefer rank milestones, especially for “early semantics”.  
• Rest_mass after L_semantic is high at onset (≈0.98 at L44) because it measures top‑k coverage only; not a fidelity metric.  
• KL is lens‑sensitive; final KL≈0 confirms good last‑layer head calibration for this run; cross‑model probability comparisons should rely on rank thresholds.  
• Prism sidecar appears regressive here; treat Prism outputs as calibration diagnostics only.

6. Model fingerprint (one sentence)  
Yi‑34B: semantic collapse at L 44; final entropy 2.98 bits; “Berlin” only exceeds p_top1=0.30 at the final layer; Prism sidecar regressive.

---
Produced by OpenAI GPT-5 
