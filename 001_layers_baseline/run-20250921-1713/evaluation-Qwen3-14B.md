## 1. Overview
Qwen/Qwen3-14B (14B) evaluated on 2025-09-21 (run-latest: `timestamp-20250921-1713`). The probe performs a layer-by-layer logit‑lens analysis (norm‑lens with RMSNorm fixes) on the positive prompt “The capital of Germany is called simply …”, tracking entropy, KL-to-final, cosine-to-final, rank milestones, and copy/entropy collapse flags.

## 2. Method sanity‑check
Diagnostics confirm a norm lens on a pre‑norm RMS model with FP32 unembed and the expected prompt ending. From JSON diagnostics: `"first_block_ln1_type": "RMSNorm"`, `"final_ln_type": "RMSNorm"`, `"use_norm_lens": true`, `"unembed_dtype": "torch.float32"`, `"layer0_position_info": "token_only_rotary_model"`, and `"context_prompt": "Give the city name only, plain text. The capital of Germany is called simply"` (001_layers_baseline/run-latest/output-Qwen3-14B.json:809,813,816,818,824,826). The context prompt ends with “called simply” (no trailing space). Gold alignment is ID‑based and resolved: `"gold_alignment": "ok"` (001_layers_baseline/run-latest/output-Qwen3-14B.json:845). Copy detection parameters present and correct: `"copy_thresh": 0.95`, `"copy_window_k": 1`, `"copy_match_level": "id_subsequence"` (001_layers_baseline/run-latest/output-Qwen3-14B.json:837–839). Summary indices: `first_kl_below_0.5 = 40`, `first_kl_below_1.0 = 40`, `first_rank_le_1 = 36`, `first_rank_le_5 = 33`, `first_rank_le_10 = 32` (001_layers_baseline/run-latest/output-Qwen3-14B.json:840–844). Negative control present with control prompt and summary: `"context_prompt": "… France … called simply"`, `"gold_alignment": "ok"`, `"first_control_margin_pos": 0`, `"max_control_margin": 0.9741542933185612` (001_layers_baseline/run-latest/output-Qwen3-14B.json:1032–1053). Ablation summary exists: `L_sem_orig = 36`, `L_sem_nf = 36` (no change), `L_copy_orig = null`, `L_copy_nf = null` (001_layers_baseline/run-latest/output-Qwen3-14B.json:1024–1031).

Lens sanity (raw vs norm): `"lens_artifact_risk": "high"`, `"max_kl_norm_vs_raw_bits": 17.6735`, `"first_norm_only_semantic_layer": null` (001_layers_baseline/run-latest/output-Qwen3-14B.json:1113–1118). Treat early semantics cautiously and prefer rank milestones. Last‑layer head calibration is clean: `"kl_to_final_bits": 0.0`, `"top1_agree": true`, `"p_top1_lens" = "p_top1_model" = 0.34514`, `"temp_est": 1.0`, and `"warn_high_last_layer_kl": false` (001_layers_baseline/run-latest/output-Qwen3-14B.json:846–918). The pure CSV’s final row also shows `kl_to_final_bits = 0.0` (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:42).

Copy‑collapse flags in early layers are absent: layers 0–3 all have `copy_collapse = False` (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:2–5). Copy-collapse flag check: no row with `copy_collapse = True` in layers 0–3 → ✓ rule did not fire spuriously.

## 3. Quantitative findings

Main prompt filter: `prompt_id = pos`, `prompt_variant = orig` (pure CSV). Table: L i — entropy H bits, top‑1 ‘token’. Bold marks L_semantic (first `is_answer = True`), which is L 36 (“Berlin”).

| Layer | Entropy (bits) | Top-1 |
|---:|---:|:---|
| L 0 | 17.2129 | 梳 |
| L 1 | 17.2120 | 地处 |
| L 2 | 17.2112 | 是一部 |
| L 3 | 17.2099 | tics |
| L 4 | 17.2084 | tics |
| L 5 | 17.2073 | -minded |
| L 6 | 17.2051 | 过去的 |
| L 7 | 17.1863 | � |
| L 8 | 17.1796 | -minded |
| L 9 | 17.1876 | -minded |
| L 10 | 17.1696 |  (?) |
| L 11 | 17.1511 | 时代的 |
| L 12 | 17.1653 | といって |
| L 13 | 17.1153 |  nav |
| L 14 | 17.1407 |  nav |
| L 15 | 17.1487 | 唿 |
| L 16 | 17.1346 | 闯 |
| L 17 | 17.1372 | 唿 |
| L 18 | 17.1009 | ____ |
| L 19 | 17.0753 | ____ |
| L 20 | 16.9323 | ____ |
| L 21 | 16.9860 | 年夜 |
| L 22 | 16.9541 | 年夜 |
| L 23 | 16.8397 | ____ |
| L 24 | 16.7602 | ____ |
| L 25 | 16.7578 | 年夜 |
| L 26 | 16.6685 | ____ |
| L 27 | 16.0316 | ____ |
| L 28 | 15.2344 | ____ |
| L 29 | 14.1869 | 这个名字 |
| L 30 | 7.7892 | 这个名字 |
| L 31 | 5.1617 | ____ |
| L 32 | 0.8160 | ____ |
| L 33 | 0.4813 | ____ |
| L 34 | 0.5948 | ____ |
| L 35 | 0.6679 | ____ |
| **L 36** | 0.3122 | ** Berlin** |
| L 37 | 0.9058 |  ____ |
| L 38 | 1.2121 |  ____ |
| **L 39** | 0.9521 | ** Berlin** |
| **L 40** | 3.5835 | ** Berlin** |

Control margin (JSON `control_summary`): `first_control_margin_pos = 0`, `max_control_margin = 0.9741542933185612` (001_layers_baseline/run-latest/output-Qwen3-14B.json:1047–1053).

Ablation (no‑filler): `L_copy_orig = null`, `L_sem_orig = 36`; `L_copy_nf = null`, `L_sem_nf = 36`; `ΔL_copy = null`, `ΔL_sem = 0` (001_layers_baseline/run-latest/output-Qwen3-14B.json:1024–1031). Interpretation: no stylistic‑cue shift in semantic collapse; rely on rank milestones due to missing L_copy.

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n/a (L_copy = null).

Confidence milestones (pure CSV): p_top1 > 0.30 at L 31; p_top1 > 0.60 at L 32; final-layer p_top1 = 0.3451 (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:32,34,42).

Rank milestones (diagnostics): rank ≤ 10 at L 32; rank ≤ 5 at L 33; rank ≤ 1 at L 36 (001_layers_baseline/run-latest/output-Qwen3-14B.json:842–844).

KL milestones (diagnostics + CSV): first_kl_below_1.0 at L 40; first_kl_below_0.5 at L 40 (001_layers_baseline/run-latest/output-Qwen3-14B.json:840–841). KL decreases with depth and is ≈ 0 at final: `kl_to_final_bits = 0.0` at L 40 (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:42).

Cosine milestones (pure CSV): first `cos_to_final ≥ 0.2` at L 5; `≥ 0.4` at L 29; `≥ 0.6` at L 36; final `cos_to_final = 0.99999` (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:6,30,38,42).

Prism Sidecar Analysis. Prism artifacts are present and marked compatible (k = 512; layers = [embed, 9, 19, 29]) (001_layers_baseline/run-latest/output-Qwen3-14B.json:827–836). However, Prism distributions are regressive relative to baseline:
- Early-depth stability: KL(P_layer||P_final) is higher under Prism at comparable depths and even increases toward final (e.g., L 0: 13.17 bits; L 29: 13.38; L 40: 14.84) vs baseline L 0 ≈ 12.96 → L 40 = 0.0 (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token-prism.csv:2,31,42; 001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:2,32,42).
- Rank milestones: no Prism layer reaches `is_answer = True`; answer ranks remain large (e.g., L 29 answer_rank = 51522; L 40 ≈ 112714) (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token-prism.csv:31,42).
- Top‑1 agreement: no improvements observed at sampled depths; top‑1 tokens diverge substantially.
- Cosine drift: Prism cos_to_final remains near zero or small through mid‑stack (e.g., L 29 cos ≈ 0.0058) (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token-prism.csv:31). Verdict: Regressive.

## 4. Qualitative patterns & anomalies
The model exhibits a late semantic snap: answer rank reaches 1 at L 36 while KL-to-final and cosine both sharply improve, consistent with a “rotation then calibration” picture where direction becomes informative before probabilities align. Notably, `cos_to_final` crosses 0.4 by L 29 while KL remains ≈13 bits, indicating an early directional alignment with late probabilistic calibration—“early direction, late calibration” (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:30–32). Final‑head calibration is good: last-layer KL is 0.0 and top‑1 agreement holds between lens and model head.

Negative control (“Berlin is the capital of”): the model predicts “ Germany” with high confidence and no leakage of “Berlin”: “ Germany, 0.632” (001_layers_baseline/run-latest/output-Qwen3-14B.json:12–20). For the main test prompt family, “Germany’s capital city is called simply” shows strong “ Berlin” likelihood in the top‑5 variants (e.g., “ Berlin, 0.873”) (001_layers_baseline/run-latest/output-Qwen3-14B.json:43–51).

Records evolution (important words). The script tags important words ["Germany", "Berlin", "capital", "Answer", "word", "simply"] (001_layers_baseline/run.py:334). In the pure CSV for the next‑token position, “Berlin” first becomes top‑1 at L 36 and stabilizes by the top layers: “(layer 36, token = ‘ Berlin’, p = 0.953)” [row 38 in CSV]; “(layer 39, token = ‘ Berlin’, p = 0.812)” [row 41]; “(layer 40, token = ‘ Berlin’, p = 0.345)” [row 42] (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:38,41,42). Prior layers are dominated by generic or template tokens (“____”, “这个名字”) until the semantic snap (rows 31–35). This aligns with a pattern of filler/template anchoring before content resolution.

Instruction brevity. The “one-word” instruction is present in the context and the collapse layer is L 36; ablation removes “simply” but leaves L_sem unchanged (ΔL_sem = 0), suggesting minimal stylistic anchoring in this case (001_layers_baseline/run-latest/output-Qwen3-14B.json:1024–1031).

Rest‑mass sanity. Rest_mass falls substantially through the snap and remains moderate thereafter; maximum after L_semantic is 0.236 at L 40 (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:42), not indicating precision loss.

Rotation vs amplification. KL decreases modestly until a sharp drop near L 36, while `cos_to_final` rises earlier (≥0.4 by L 29) and `answer_rank` improves to 1 at L 36—consistent with rotation first, then amplification/calibration. Example quotes:
> “(layer 31, kl_to_final_bits = 12.73, cos = 0.464)”  [row 32 in CSV]
> “(layer 36, token = ‘ Berlin’, p = 0.953, kl = 1.66)”  [row 38 in CSV]

Head calibration (final layer). `warn_high_last_layer_kl = false`; `temp_est = 1.0`; `kl_after_temp_bits = 0.0` (001_layers_baseline/run-latest/output-Qwen3-14B.json:846–918). Treat final probabilities as calibrated within this model.

Lens sanity. Raw‑vs‑norm check flags `lens_artifact_risk = high` with `max_kl_norm_vs_raw_bits ≈ 17.67` and no norm‑only semantics layer (001_layers_baseline/run-latest/output-Qwen3-14B.json:1113–1118). Early “semantics” should be treated cautiously; rely on rank milestones and within‑model trends (we do).

Temperature robustness. At T = 0.1 the distribution is sharply peaked on “ Berlin” (e.g., “ Berlin, 0.974” within the test‑prompt family) and at T = 2.0 the same prompt shows “ Berlin, 0.036” with entropy 13.16 bits (001_layers_baseline/run-latest/output-Qwen3-14B.json:694–760). Entropy rises substantially with temperature.

Important‑word trajectory. “Berlin” first enters top‑1 at L 36 and remains frequently in the top‑k through the final layers; “Germany” dominates the reverse direction prompt (“Berlin is the capital of”) as desired; “capital” and “simply” behave as grammatical scaffolding early, then drop out after the snap. Example: “(layer 36, token = ‘ Berlin’, p = 0.953)” [row 38]; “(layer 39, token = ‘ Berlin’, p = 0.812)” [row 41] (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv).

Checklist:
- RMS lens? ✓ (`RMSNorm`) (001_layers_baseline/run-latest/output-Qwen3-14B.json:809,813)
- LayerNorm bias removed? ✓ (`layernorm_bias_fix = not_needed_rms_model`) (001_layers_baseline/run-latest/output-Qwen3-14B.json:815)
- Entropy rise at unembed? ✓ (final entropy 3.5835 bits; earlier near 0.312 at L 36) (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:38,42)
- FP32 un‑embed promoted? ✓ (`unembed_dtype = torch.float32`) (001_layers_baseline/run-latest/output-Qwen3-14B.json:818)
- Punctuation / markup anchoring? ✓ (early top‑1 includes fillers like “____”, punctuation) (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:18–27,31)
- Copy‑reflex? ✗ (no `copy_collapse = True` in layers 0–3) (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:2–5)
- Grammatical filler anchoring? ✓ (layers 0–5 dominated by fillers/non‑semantic tokens) (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:2–7)

## 5. Limitations & data quirks
- Raw‑vs‑norm lens difference flagged “high” risk (sampled mode), so early pre‑final “semantics” may be lens‑induced. We rely on rank milestones and within‑model trends.
- L_copy is null; ΔH relative to copy cannot be computed. Use rank milestones and the L_sem collapse.
- Rest_mass after L_sem remains ≤ 0.236; not a fidelity metric but monitored for precision artifacts.
- Prism sidecar is marked compatible but regresses KL/rank/cosine; treat Prism outputs as diagnostic only for this run.

## 6. Model fingerprint
Qwen‑3‑14B: semantic collapse at L 36; final entropy 3.58 bits; “Berlin” top‑1 from L 36 onward with KL→0 at final.

---
Produced by OpenAI GPT-5
