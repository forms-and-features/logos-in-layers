# Evaluation Report: Qwen/Qwen2.5-72B
1. Overview

Qwen2.5‑72B (80 blocks; layers indexed 0–80). Run artifacts under run‑latest dated 2025‑09‑23. The probe captures layer‑wise entropy, KL to final, cosine trajectory, copy/semantic collapse flags, and calibrated pure next‑token predictions (pos/ctl, orig/no_filler), with Prism sidecar available.

2. Method sanity‑check

Diagnostics confirm RMSNorm lens with FP32 unembed and token‑only rotary positional handling: “use_norm_lens”: true; “unembed_dtype”: “torch.float32” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:807–809]; “layer0_position_info”: “token_only_rotary_model” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:816]. Context prompt ends exactly with “called simply” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:804–815].

Copy/semantic metrics present: “L_copy”: null; “L_copy_H”: null; “L_semantic”: 80; “delta_layers”: null; “first_rank_le_{10,5,1}”: 74,78,80; “first_kl_below_{1.0,0.5}”: 80,80 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:833–853]. Strict copy rule settings appear: “copy_thresh”: 0.95, “copy_window_k”: 1, “copy_match_level”: “id_subsequence” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:846–848]. Soft detector config present: “copy_soft_config”: { “threshold”: 0.5, “window_ks”: [1,2,3], “extra_thresholds”: [] } and “copy_flag_columns”: [“copy_strict@0.95”, “copy_soft_k1@0.5”, “copy_soft_k2@0.5”, “copy_soft_k3@0.5”] [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:833–843,1077–1083]. Gold alignment is ok for both prompts [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:898,1104], and the control prompt is present with summary margins [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1091–1106]. Ablation summary exists with both prompt variants (orig/no_filler present in CSV) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1083–1090; 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:139]. Main table below filters to prompt_id=pos and prompt_variant=orig.

Units: entropy and KL are in bits (columns ‘entropy’, ‘kl_to_final_bits’). Final‑head calibration is sound: final CSV row has kl_to_final_bits ≈ 0 (0.000109) [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:138] and diagnostics.last_layer_consistency exists with top1_agree=true; p_top1_lens=0.3395 vs p_top1_model=0.3383; temp_est=1.0; kl_after_temp_bits=0.000109; warn_high_last_layer_kl=false [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:899–907,917].

Lens sanity: raw_lens_check.mode=“sample”, lens_artifact_risk=“high”, max_kl_norm_vs_raw_bits=19.9099, first_norm_only_semantic_layer=null [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1015–1016,1072–1074]. Treat any early “semantics” cautiously; prefer rank milestones.

Copy‑collapse flag check (layers 0–3, pos/orig): copy_collapse=False and copy_soft_k1@0.5=False (no strict/soft triggers observed). Earliest is_answer=True at L=80 [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:138].

Soft copy flags: no soft copy triggers (k∈{1,2,3}) across layers (all False in CSV).

3. Quantitative findings

Table (pos, orig). Bold indicates the semantic layer (first is_answer=True; gold token “Berlin”).

| Layer | Entropy (bits) | Top-1 |
|---|---:|---|
| L 0 | 17.2142 | `s` |
| L 1 | 17.2142 | `下一篇` |
| L 2 | 17.1425 | `ولوج` |
| L 3 | 17.0631 | `شدد` |
| L 4 | 17.0891 | `.myapplication` |
| L 5 | 17.0072 | `ستحق` |
| L 6 | 17.0315 | `.myapplication` |
| L 7 | 16.9372 | `.myapplication` |
| L 8 | 16.7980 | `.myapplication` |
| L 9 | 16.1203 | `ستحق` |
| L 10 | 16.5008 | `.myapplication` |
| L 11 | 16.7180 | `.myapplication` |
| L 12 | 16.7781 | `かもしれ` |
| L 13 | 16.6314 | `かもしれ` |
| L 14 | 16.3595 | `かもしれ` |
| L 15 | 16.5170 | `のではない` |
| L 16 | 16.4908 | `iéndo` |
| L 17 | 16.2127 | `iéndo` |
| L 18 | 16.1093 | `有期徒` |
| L 19 | 15.7574 | `有期徒` |
| L 20 | 16.1290 | `有期徒` |
| L 21 | 16.1558 | `有期徒` |
| L 22 | 15.9799 | `有期徒` |
| L 23 | 16.4015 | `.myapplication` |
| L 24 | 15.9989 | `iéndo` |
| L 25 | 15.3506 | `hế` |
| L 26 | 15.9435 | `iéndo` |
| L 27 | 15.7559 | `iéndo` |
| L 28 | 15.7500 | `.myapplication` |
| L 29 | 15.8849 | `.myapplication` |
| L 30 | 16.1225 | `.myapplication` |
| L 31 | 16.1700 | `.myapplication` |
| L 32 | 16.1708 | `.myapplication` |
| L 33 | 16.4191 | `hế` |
| L 34 | 16.2001 | `iéndo` |
| L 35 | 16.4550 | `hế` |
| L 36 | 16.4078 | `iéndo` |
| L 37 | 16.2100 | `iéndo` |
| L 38 | 16.4904 | `hế` |
| L 39 | 16.4177 | `iéndo` |
| L 40 | 16.1916 | `iéndo` |
| L 41 | 16.4652 | `hế` |
| L 42 | 16.5948 | `hế` |
| L 43 | 16.4974 | `hế` |
| L 44 | 16.6553 | `続きを読む` |
| L 45 | 16.8770 | `国际在线` |
| L 46 | 17.0023 | `国际在线` |
| L 47 | 17.0133 | `主义思想` |
| L 48 | 17.0217 | `主义思想` |
| L 49 | 17.0217 | ` reuseIdentifier` |
| L 50 | 16.9678 | `uckets` |
| L 51 | 16.9723 | ` "` |
| L 52 | 17.0086 | `"` |
| L 53 | 16.9266 | `"` |
| L 54 | 16.9081 | `"` |
| L 55 | 16.9423 | `"` |
| L 56 | 16.9382 | `"` |
| L 57 | 16.8408 | ` "` |
| L 58 | 16.9148 | ` "` |
| L 59 | 16.9201 | ` "` |
| L 60 | 16.8861 | ` '` |
| L 61 | 16.9030 | ` '` |
| L 62 | 16.8336 | ` "` |
| L 63 | 16.8908 | ` "` |
| L 64 | 16.8947 | ` "` |
| L 65 | 16.8689 | ` "` |
| L 66 | 16.8994 | ` "` |
| L 67 | 16.8932 | ` "` |
| L 68 | 16.7786 | ` "` |
| L 69 | 16.8758 | ` "` |
| L 70 | 16.7866 | ` "` |
| L 71 | 16.5046 | ` "` |
| L 72 | 16.6499 | ` "` |
| L 73 | 15.7867 | ` "` |
| L 74 | 16.0809 | ` "` |
| L 75 | 13.3499 | ` "` |
| L 76 | 14.7428 | ` "` |
| L 77 | 10.8478 | ` "` |
| L 78 | 15.3978 | ` "` |
| L 79 | 16.6656 | ` "` |
| **L 80** | 4.1158 | ` Berlin` |

Control margin (JSON): first_control_margin_pos=0; max_control_margin=0.2070 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1106–1113].

Ablation (no‑filler): L_copy_orig=null, L_sem_orig=80; L_copy_nf=null, L_sem_nf=80; ΔL_copy=null, ΔL_sem=0 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1083–1090]. Strict and soft copy detectors did not fire; rely on rank milestones where needed.

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n/a (L_copy=null). Soft ΔHₖ (k∈{1,2,3}) = n/a (all L_copy_soft[k]=null).

Confidence milestones (pure CSV, pos/orig): p_top1>0.30 at layer 77 (0.3091) [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:135]; p_top1>0.60 not reached; final‑layer p_top1=0.3395 [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:138].

Rank milestones (JSON): rank≤10 at L=74; rank≤5 at L=78; rank≤1 at L=80 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:851–853].

KL milestones (JSON): first_kl_below_1.0 at L=80; first_kl_below_0.5 at L=80 [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:849–850]. KL decreases to ≈0 by the final layer, consistent with well‑aligned final head [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:138].

Cosine milestones (pure CSV): cos_to_final≥0.2 at L=0; ≥0.4 at L=0; ≥0.6 at L=53 (cos=0.6154) [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:111]; final cos=1.0000 [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:138].

Prism Sidecar Analysis
- Presence: compatible=true, present=true, k=512, layers=[embed,19,39,59]; error=null [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:819–831].
- Early-depth stability (KL to final at sample depths):
  - L0: baseline 9.391 bits vs Prism 9.494 bits (slightly higher)
  - L19/39: baseline 13.205/11.909 vs Prism 9.404/9.593 (≥0.5–3.8 bits drop)
  - L59: baseline 9.116 vs Prism 9.597 (slightly higher)
  [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:1, and 
   001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token-prism.csv:— sampled via matching layers].
- Rank milestones (Prism pure CSV): no layer reaches rank≤10/5/1 (all None; final L80: answer_rank≈106311) [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token-prism.csv:120].
- Top‑1 agreement at sampled depths: neither baseline nor Prism matches final ‘Berlin’ until the end; Prism diverges strongly at L80 (top‑1 ‘onent’, KL≈20.68 bits) [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token-prism.csv:120].
- Cosine drift: Prism cos_to_final is near zero/negative at early/mid layers (e.g., L0≈−0.119; L59≈−0.101) vs baseline positive at several depths (e.g., L59≈0.498) — no earlier stabilization.
- Copy flags: no spurious flips (all False for strict/soft flags at sampled depths).
- Verdict: Regressive. Despite mid‑depth KL reductions, Prism fails to preserve rank/probability calibration and severely degrades final‑layer agreement.

4. Qualitative patterns & anomalies

The gold token appears only at the final layer under the stringent “one‑word” instruction. The trajectory shows rising directional alignment (cosine ≥0.6 by L53) while KL to final remains high until late, indicating “early direction, late calibration.” Final‑head calibration is good (KL≈0), so within‑model probabilities near the end are interpretable; earlier probabilities should be interpreted cautiously given raw‑lens risk.

Negative control: For “Berlin is the capital of”, the model’s top‑5 are Germany (0.7695), “ the” (0.0864), “ which” (0.0491), “ a” (0.0125), “ what” (0.0075) — strong semantic leakage to the country without prompting [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:206–228].

Records CSV shows the evolution of IMPORTANT_WORDS = [“Germany”, “Berlin”, “capital”, “Answer”, “word”, “simply”] (run.py). Around NEXT (pos=15), grammatical filler and punctuation dominate top‑1 deep into the stack. ‘Berlin’ starts to show in late layers near the context tokens: at L74 pos=13 (‘ is’), ‘Berlin’ is top‑2 (p≈0.0030) [001_layers_baseline/run-latest/output-Qwen2.5-72B-records.csv:3909]; at L74 pos=14 (‘ called’), ‘Berlin’ enters top‑5 (p≈0.00157) [001_layers_baseline/run-latest/output-Qwen2.5-72B-records.csv:3910]; at L74 pos=15 (‘ simply’), ‘Berlin’ appears in top‑20 (p≈0.00095) [001_layers_baseline/run-latest/output-Qwen2.5-72B-records.csv:3911]; by L76 pos=15 it rises (p≈0.00138) [001_layers_baseline/run-latest/output-Qwen2.5-72B-records.csv:3966]. This supports a late consolidation around the gold token near the end of the stack.

Instructional phrasing: Test‑prompt variants without the “one‑word” style tend to elevate ‘Berlin’ as a top‑k completion directly (e.g., “Germany’s capital city is called” → ‘ Berlin’, 0.4473) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:229–262], consistent with stylistic guidance affecting when semantics surface in the NEXT position under the lens rather than the model’s factual access.

Rest‑mass sanity: Final rest_mass at L80 is ≈0.298 [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:138]; no spikes indicating precision loss after semantics.

Rotation vs amplification: cos_to_final climbs early (≥0.4 at L0; ≥0.6 by L53) while KL remains >0.5 until L80, and answer_rank improves late (≤10 at L74, ≤1 at L80). This indicates early direction formation with late probability calibration. Given raw‑lens risk=high, prefer rank thresholds to claim “early semantics.”

Head calibration (final layer): warn_high_last_layer_kl=false; temp_est=1.0; kl_after_temp_bits≈0.0001 — no family‑specific calibration anomaly [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:899–907,917].

Lens sanity: raw_lens_check.summary → lens_artifact_risk=high; max_kl_norm_vs_raw_bits=19.9099; first_norm_only_semantic_layer=null [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1072–1074]. Caution that pre‑final “semantics” may be lens‑induced; rely on rank milestones.

Temperature robustness: At T=0.1, ‘Berlin’ rank 1 (p≈0.9526; entropy≈0.275 bits) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:670–688]. At T=2.0, ‘Berlin’ remains present but diluted (p≈0.0162; entropy≈15.013 bits) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:742–760].

Important‑word trajectory — ‘Berlin’ first enters any top‑5 around L74 in neighbor positions and stabilizes only at the end; ‘capital’ is salient as filler near NEXT in mid‑late layers (e.g., L74 pos=15 includes “ capital”, p≈0.00330) [001_layers_baseline/run-latest/output-Qwen2.5-72B-records.csv:3911].

Stylistic ablation: ΔL_sem=0; ΔL_copy=null. Removing “simply” does not change the semantic layer index (both at L80) [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:1083–1090].

Checklist:
✓ RMS lens (RMSNorm)  | ✓ LayerNorm bias removed (not needed)  | ✓ FP32 unembed
✗ Copy‑reflex (no strict/soft triggers at L0–3)  | ✓ Punctuation/filler anchoring (late‑stack quotes/commas dominate NEXT)

5. Limitations & data quirks

Raw‑vs‑norm lens differences: raw_lens_check.mode=sample and lens_artifact_risk=high with max_kl_norm_vs_raw_bits≈19.91, so early “semantics” may be lens‑induced; rely on rank milestones for pre‑final claims. KL is lens‑sensitive; although final KL≈0 here, cross‑model probability comparisons should prefer rank thresholds. Rest_mass≈0.298 at L80 is moderate and not a fidelity metric; use last_layer_consistency and raw_lens_check for calibration and lens sanity.

6. Model fingerprint

Qwen2.5‑72B: semantics at L 80; final entropy 4.12 bits; late rise of ‘Berlin’ with early cosine alignment and no copy‑reflex.

---
Produced by OpenAI GPT-5 
*Run executed on: 2025-09-23 13:07:01*
