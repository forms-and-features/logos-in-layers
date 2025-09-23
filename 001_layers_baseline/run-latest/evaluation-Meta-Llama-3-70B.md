**Overview**

Meta-Llama-3-70B (80 layers). Probe run captured layer-by-layer next-token predictions under a norm-lens with FP32 unembed on CPU. Outputs include per-layer entropy/flags and diagnostics for head calibration, KL-to-final, cosine trajectory, control, ablation, and Prism sidecar.

**Method Sanity-Check**

- Context prompt ends exactly with “called simply”: "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply"  (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:17).
- Norm lens and FP32 unembed active; RMSNorm model with bias fix not needed: "use_norm_lens": true, "use_fp32_unembed": true, "first_block_ln1_type": "RMSNorm" (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:744–753).
- Copy detector configuration present and mirrored in CSV: "copy_thresh": 0.95, "copy_window_k": 1, "copy_match_level": "id_subsequence"; "copy_soft_config": {"threshold": 0.5, "window_ks": [1,2,3], "extra_thresholds": []}; "copy_flag_columns": ["copy_strict@0.95","copy_soft_k1@0.5","copy_soft_k2@0.5","copy_soft_k3@0.5"] (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:895–912, 1034–1040, 1112–1118).
- Summary indices (bits, ranks) present: "first_kl_below_0.5": 80, "first_kl_below_1.0": 80, "first_rank_le_1": 40, "first_rank_le_5": 38, "first_rank_le_10": 38 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:878–887).
- Gold alignment OK and ID-based: "gold_alignment": "ok"; gold_answer.first_id=20437 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:905, 1144–1152).
- Negative control present with summary: "first_control_margin_pos": 0, "max_control_margin": 0.5168457566906 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1121–1127).
- Ablation summary present with both variants: {"L_copy_orig": null, "L_sem_orig": 40, "L_copy_nf": null, "L_sem_nf": 42, "delta_L_copy": null, "delta_L_sem": 2} (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1101–1110). Positive rows appear for both prompt_variant=orig and no_filler in CSV.
- Lens sanity (raw vs norm): mode "sample"; summary: {"first_norm_only_semantic_layer": null, "max_kl_norm_vs_raw_bits": 0.04289584828470766, "lens_artifact_risk": "low"} (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:837–862).
- Last-layer head calibration present: "last_layer_consistency": {"kl_to_final_bits": 0.000729..., "top1_agree": true, "p_top1_lens": 0.4783, "p_top1_model": 0.4690, "temp_est": 1.0, "warn_high_last_layer_kl": false} (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:916–939). Note: the CSV’s per-layer KL-to-final is lens→model; final-row KL≈0 (see layer 80 row with kl_to_final_bits=0.000729) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:82).
- Copy-collapse (strict) in layers 0–3: none observed in pure CSV (all False) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:1–4). Soft-copy flags (k∈{1,2,3} at τ=0.5): none observed (consistent with JSON L_copy_soft all null).

Copy-collapse flag check (pure CSV; pos/orig): first copy_collapse=True — none; ✓ rule did not spuriously fire. Earliest soft flags — none (strict remained null throughout).

**Quantitative Findings**

Main table (pos/orig). L_semantic=40 (first is_answer=True). Units: entropy and KL are bits.

| Layer | Entropy (bits) | Top‑1 token |
|---:|---:|---|
| 0 | 16.9681 | winding |
| 1 | 16.9601 | cepts |
| 2 | 16.9634 | улю |
| 3 | 16.9626 | zier |
| 4 | 16.9586 | alls |
| 5 | 16.9572 | alls |
| 6 | 16.9561 | alls |
| 7 | 16.9533 | NodeId |
| 8 | 16.9594 | inds |
| 9 | 16.9597 | NodeId |
| 10 | 16.9524 | inds |
| 11 | 16.9560 | inds |
| 12 | 16.9564 | lia |
| 13 | 16.9552 | eds |
| 14 | 16.9504 | idders |
| 15 | 16.9533 |  Kok |
| 16 | 16.9522 | /plain |
| 17 | 16.9480 |  nut |
| 18 | 16.9443 |  nut |
| 19 | 16.9475 |  nut |
| 20 | 16.9464 |  nut |
| 21 | 16.9380 |  burge |
| 22 | 16.9378 |  simply |
| 23 | 16.9358 |  bur |
| 24 | 16.9497 |  bur |
| 25 | 16.9375 | � |
| 26 | 16.9383 | � |
| 27 | 16.9372 | za |
| 28 | 16.9328 | /plain |
| 29 | 16.9328 |  plain |
| 30 | 16.9386 | zed |
| 31 | 16.9251 |  simply |
| 32 | 16.9406 |  simply |
| 33 | 16.9271 |  plain |
| 34 | 16.9323 |  simply |
| 35 | 16.9292 |  simply |
| 36 | 16.9397 | simply |
| 37 | 16.9346 | simply |
| 38 | 16.9342 | simply |
| 39 | 16.9349 | simply |
| 40 | 16.9374 | **Berlin** |
| 41 | 16.9362 | " """ |
| 42 | 16.9444 | " """ |
| 43 | 16.9413 | Berlin |
| 44 | 16.9259 | Berlin |
| 45 | 16.9402 | " """ |
| 46 | 16.9552 | " """ |
| 47 | 16.9393 | " """ |
| 48 | 16.9388 | " """ |
| 49 | 16.9369 | " """ |
| 50 | 16.9438 | " """ |
| 51 | 16.9401 | " """ |
| 52 | 16.9220 | Berlin |
| 53 | 16.9330 | Berlin |
| 54 | 16.9424 | Berlin |
| 55 | 16.9419 | Berlin |
| 56 | 16.9210 | Berlin |
| 57 | 16.9335 | Berlin |
| 58 | 16.9411 | Berlin |
| 59 | 16.9441 | Berlin |
| 60 | 16.9229 | Berlin |
| 61 | 16.9396 | Berlin |
| 62 | 16.9509 | Berlin |
| 63 | 16.9458 | Berlin |
| 64 | 16.9263 | Berlin |
| 65 | 16.9334 | " """ |
| 66 | 16.9407 | Berlin |
| 67 | 16.9304 | Berlin |
| 68 | 16.9240 | Berlin |
| 69 | 16.9315 | Berlin |
| 70 | 16.9257 | Berlin |
| 71 | 16.9226 | Berlin |
| 72 | 16.9221 | Berlin |
| 73 | 16.9181 | " """ |
| 74 | 16.9143 | Berlin |
| 75 | 16.9127 | Berlin |
| 76 | 16.9190 | Berlin |
| 77 | 16.9099 | Berlin |
| 78 | 16.9186 | Berlin |
| 79 | 16.9423 | Berlin |

Control margin (JSON): first_control_margin_pos = 0; max_control_margin = 0.5168457566906 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1121–1127).

Ablation (no‑filler): L_copy_orig = null, L_sem_orig = 40; L_copy_nf = null, L_sem_nf = 42; ΔL_copy = null, ΔL_sem = 2 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1101–1110). Interpretation: removing “simply” delays semantics by 2 layers (~2.5% of depth), suggesting mild stylistic anchoring rather than semantic dependence.

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (L_copy null). Soft ΔHk (bits) = n.a. (all L_copy_soft[k] null).

Confidence milestones (pure CSV; pos/orig): p_top1 > 0.30 at layer 80; p_top1 > 0.60 not reached; final-layer p_top1 = 0.4783 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:82).

Rank milestones (JSON): rank ≤ 10 at layer 38; rank ≤ 5 at layer 38; rank ≤ 1 at layer 40 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:882–887).

KL milestones (JSON): first_kl_below_1.0 at layer 80; first_kl_below_0.5 at layer 80 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:878–881). KL decreases slowly with depth and is ≈ 0 at final under head calibration (last_layer_consistency.kl_to_final_bits=0.000729) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:918–920).

Cosine milestones (pure CSV; within-model): first cos_to_final ≥ 0.2 at layer 80; ≥ 0.4 at 80; ≥ 0.6 at 80; final cos_to_final = 0.99999 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:82).

Prism Sidecar Analysis
- Presence: compatible = true (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:758–765). Baseline vs Prism KL(P_layer||P_final) at sampled depths (bits):
  - L0: baseline 10.50 vs Prism 10.67
  - L20: 10.45 vs 11.34
  - L40: 10.42 vs 11.42
  - L60: 10.31 vs 11.47
  - L80: 0.00073 vs 26.88 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:80–83; 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token-prism.csv:1–4,82).
- Rank milestones (Prism pure): not reached (answer_rank never ≤10/5/1; e.g., at L40 answer_rank=1723) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token-prism.csv:42).
- Top‑1 agreement: baseline reaches top‑1 ‘Berlin’ at L40 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:42), while Prism top‑1 at L40 is unrelated (“imax”); at L80 Prism top‑1 is a spurious token (“oldt”, p≈1.0) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token-prism.csv:82).
- Cosine drift: Prism does not show earlier stabilization; cos_to_final remains tiny until an ill‑calibrated final row (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token-prism.csv:0–5,82).
- Copy flags: no spurious flips; copy flags remain False under Prism (scan of Prism pure CSV).
- Verdict: Regressive (KL increases at early/mid layers and final calibration is pathological; rank milestones are later/not achieved).

**Qualitative Patterns & Anomalies**

Negative control. “Berlin is the capital of” predicts “ Germany” with high confidence: > " Germany", 0.8516; “ the”, 0.0791; “ and”, 0.0146; “ modern”, 0.0048; “ Europe”, 0.0031 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:6–28). No semantic leakage of “Berlin”.

Important-word trajectory (records CSV; IMPORTANT_WORDS = ["Germany", "Berlin", "capital", "Answer", "word", "simply"]). Around the final context token (“ simply”, pos=16) ‘Berlin’ first enters the top‑5 at L38 (top‑3, p≈2.50e‑05) and becomes top‑1 at L40: 
> “ Berlin, 2.5017e-05 … Germany, 2.4887e-05” (layer 38, pos=16)  [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-records.csv:817]
> “ Berlin, 2.3917e-05 … simply, 2.3188e-05” (layer 40, pos=16)  [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-records.csv:854]
At the “ Germany” token position (pos=13), ‘Berlin’ also appears in late layers’ top‑5 near L38–40 (e.g., L38 pos=13 has “ Berlin, 2.1217e‑05”) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-records.csv:814]. This suggests late consolidation of the answer with related context words present nearby.

One‑word instruction. Removing “simply” delays the semantic layer by 2 (L_sem 40 → 42), consistent with stylistic cue anchoring rather than core factual retrieval (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1101–1110).

Rest‑mass sanity. Rest_mass is top‑20 remainder; it stays high and is not a fidelity metric. Post‑semantic layers keep rest_mass ≈ 0.9999 (e.g., L40 pos/orig rest_mass=0.9998877) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:42). No spikes suggesting precision loss.

Rotation vs amplification. KL_to_final decreases slowly through mid‑stack and drops at the calibrated final head (first_kl_below_1.0 at L80; last_layer_consistency.kl≈0.00073). Meanwhile, p_answer reaches rank‑1 at L40 and final p_top1≈0.478. Cosine to final remains small until the calibrated final row (first ≥0.2/0.4/0.6 only at L80), indicating early direction remains miscalibrated until the head (“early direction, late calibration”) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:878–920; 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:42,82).

Head calibration. Final‑head calibration is good: top‑1 agrees, kl_to_final_bits≈0.00073, temp_est=1.0, and warn_high_last_layer_kl=false (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:916–939). Prefer rank milestones for cross‑family comparisons.

Lens sanity. RAW vs NORM sampled check indicates low artifact risk; max_kl_norm_vs_raw_bits=0.0429; no “norm‑only semantics” layer (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:856–862). Within‑model trends are therefore reliable.

Temperature robustness. At T=0.1, ‘Berlin’ rank 1 (p=0.9933; entropy 0.058 bits); at T=2.0, ‘Berlin’ rank 1 with low confidence (p=0.0357; entropy 14.46 bits) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:676–708, 737–780).

Checklist
- RMS lens? ✓ (RMSNorm; norm lens true) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:748–755)
- LayerNorm bias removed? ✓ (“not_needed_rms_model”) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:748–755)
- Entropy rise at unembed? n.a. (final calibrated row provided separately; early layers ≈17 bits; final calibrated ~2.59 bits) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:940–973; 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:82)
- FP32 un-embed promoted? ✓ ("use_fp32_unembed": true) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:746–747)
- Punctuation / markup anchoring? ✓ (early records/top‑1 dominated by code/markup shards) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-records.csv:1–20)
- Copy‑reflex? ✗ (no strict or soft copy flags in L0–L3) (pure CSV early rows)
- Grammatical filler anchoring? ✗ (early L0–5 top‑1 not in {“is”, “the”, “a”, “of”})

**Limitations & Data Quirks**

- Rest_mass is top‑k remainder only; it remains high after L_semantic and is not a lens‑fidelity signal (instructions). No spikes observed post‑semantic.
- KL is lens‑sensitive; despite final head calibration being near‑zero KL, mid‑stack KL values are sizable. Prefer rank milestones for cross‑model claims.
- Raw‑vs‑norm sanity used "sample" mode, not full; treat findings as sampled sanity, though artifact risk is low here (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:856–862).
- Prism sidecar appears miscalibrated for this run (KL increases; nonsensical final top‑1), so Prism‑based claims are not used to adjust baseline metrics.

**Model Fingerprint**

Llama‑3‑70B: collapse at L 40; final entropy ~2.59 bits; ‘Berlin’ stabilizes only at the calibrated final head.

---
Produced by OpenAI GPT-5 
