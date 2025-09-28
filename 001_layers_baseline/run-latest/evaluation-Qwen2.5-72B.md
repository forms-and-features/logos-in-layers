# Evaluation Report: Qwen/Qwen2.5-72B

*Run executed on: 2025-09-28 17:22:48*
# Evaluation Report: Qwen2.5-72B

1. Overview

Qwen2.5‑72B; run timestamp 2025‑09‑28 (see `001_layers_baseline/run-latest/timestamp-20250928-1722`). This probe captures layer‑by‑layer next‑token behavior under a norm lens, tracking copy vs semantic collapse, calibration (KL in bits), geometry (cosine to final), and control/ablation diagnostics.

2. Method sanity‑check

Norm lens and FP32 un‑embed are active with RMSNorm: `use_norm_lens=True`, `use_fp32_unembed=True`, `unembed_dtype=torch.float32`, `first_block_ln1_type=RMSNorm`, `final_ln_type=RMSNorm`, `layernorm_bias_fix=not_needed_rms_model`, `layer0_position_info=token_only_rotary_model`, `norm_alignment_fix=using_ln2_rmsnorm_for_post_block` (001_layers_baseline/run-latest/output-Qwen2.5-72B.json). The context prompt ends with “called simply” with no trailing space: `Give the city name only, plain text. The capital of Germany is called simply` (001_layers_baseline/run-latest/output-Qwen2.5-72B.json).

Copy detector configuration is present and mirrored in CSV/JSON: `copy_thresh=0.95`, `copy_window_k=1`, `copy_match_level=id_subsequence`, soft `threshold=0.5`, `window_ks=[1,2,3]`, `extra_thresholds=[]`; flags `['copy_strict@0.95','copy_soft_k1@0.5','copy_soft_k2@0.5','copy_soft_k3@0.5']` (001_layers_baseline/run-latest/output-Qwen2.5-72B.json). Diagnostics include `L_copy`, `L_copy_H`, `L_semantic`, `delta_layers`, `L_copy_soft`, `delta_layers_soft` (same JSON). Gold alignment is `ok` with ID‑level answer “Berlin” (`first_id=19846`, `variant=with_space`) and control prompt/summary are present: `first_control_margin_pos=0`, `max_control_margin=0.2070` (001_layers_baseline/run-latest/output-Qwen2.5-72B.json).

Units are bits for both entropy and KL (`entropy`; `kl_to_final_bits`; `teacher_entropy_bits`). Final‑head calibration is clean and agrees: `last_layer_consistency={ kl_to_final_bits=0.000109, top1_agree=True, p_top1_lens=0.3395 vs p_top1_model=0.3383, temp_est=1.0, warn_high_last_layer_kl=False }` (001_layers_baseline/run-latest/output-Qwen2.5-72B.json). Lens sanity (raw vs norm) reports `mode=sample`, `lens_artifact_risk=high`, `max_kl_norm_vs_raw_bits=19.9099`, `first_norm_only_semantic_layer=None` (001_layers_baseline/run-latest/output-Qwen2.5-72B.json). Strict and soft copy flags do not fire in layers 0–3 (see Section 4). Ablation summary exists with both `orig` and `no_filler` variants recorded (see Section 3). Summary indices: `first_kl_below_0.5=80`, `first_kl_below_1.0=80`, `first_rank_le_1=80`, `first_rank_le_5=78`, `first_rank_le_10=74` (same JSON).

Copy‑collapse flag check (pure‑next‑token CSV): no row with `copy_collapse=True` in L0–L3; earliest soft flags are also absent (`copy_soft_k1@0.5=False` in L0–L3). Final‑row sanity: “(layer 80, token = ‘Berlin’, p = 0.3395; KL = 0.000109 bits)” 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:138.

3. Quantitative findings

Table below uses only positive rows (`prompt_id=pos`, `prompt_variant=orig`) from `001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv`. Bold marks the first semantic layer (`is_answer=True`).

| Layer summary |
|---|
| L 0 — entropy 17.214 bits, top‑1 's' |
| L 1 — entropy 17.214 bits, top‑1 '下一篇' |
| L 2 — entropy 17.143 bits, top‑1 'ولوج' |
| L 3 — entropy 17.063 bits, top‑1 'شدد' |
| L 4 — entropy 17.089 bits, top‑1 '.myapplication' |
| L 5 — entropy 17.007 bits, top‑1 'ستحق' |
| L 6 — entropy 17.031 bits, top‑1 '.myapplication' |
| L 7 — entropy 16.937 bits, top‑1 '.myapplication' |
| L 8 — entropy 16.798 bits, top‑1 '.myapplication' |
| L 9 — entropy 16.120 bits, top‑1 'ستحق' |
| L 10 — entropy 16.501 bits, top‑1 '.myapplication' |
| L 11 — entropy 16.718 bits, top‑1 '.myapplication' |
| L 12 — entropy 16.778 bits, top‑1 'かもしれ' |
| L 13 — entropy 16.631 bits, top‑1 'かもしれ' |
| L 14 — entropy 16.359 bits, top‑1 'かもしれ' |
| L 15 — entropy 16.517 bits, top‑1 'のではない' |
| L 16 — entropy 16.491 bits, top‑1 'iéndo' |
| L 17 — entropy 16.213 bits, top‑1 'iéndo' |
| L 18 — entropy 16.109 bits, top‑1 '有期徒' |
| L 19 — entropy 15.757 bits, top‑1 '有期徒' |
| L 20 — entropy 16.129 bits, top‑1 '有期徒' |
| L 21 — entropy 16.156 bits, top‑1 '有期徒' |
| L 22 — entropy 15.980 bits, top‑1 '有期徒' |
| L 23 — entropy 16.401 bits, top‑1 '.myapplication' |
| L 24 — entropy 15.999 bits, top‑1 'iéndo' |
| L 25 — entropy 15.351 bits, top‑1 'hế' |
| L 26 — entropy 15.944 bits, top‑1 'iéndo' |
| L 27 — entropy 15.756 bits, top‑1 'iéndo' |
| L 28 — entropy 15.750 bits, top‑1 '.myapplication' |
| L 29 — entropy 15.885 bits, top‑1 '.myapplication' |
| L 30 — entropy 16.123 bits, top‑1 '.myapplication' |
| L 31 — entropy 16.170 bits, top‑1 '.myapplication' |
| L 32 — entropy 16.171 bits, top‑1 '.myapplication' |
| L 33 — entropy 16.419 bits, top‑1 'hế' |
| L 34 — entropy 16.200 bits, top‑1 'iéndo' |
| L 35 — entropy 16.455 bits, top‑1 'hế' |
| L 36 — entropy 16.408 bits, top‑1 'iéndo' |
| L 37 — entropy 16.210 bits, top‑1 'iéndo' |
| L 38 — entropy 16.490 bits, top‑1 'hế' |
| L 39 — entropy 16.418 bits, top‑1 'iéndo' |
| L 40 — entropy 16.192 bits, top‑1 'iéndo' |
| L 41 — entropy 16.465 bits, top‑1 'hế' |
| L 42 — entropy 16.595 bits, top‑1 'hế' |
| L 43 — entropy 16.497 bits, top‑1 'hế' |
| L 44 — entropy 16.655 bits, top‑1 '続きを読む' |
| L 45 — entropy 16.877 bits, top‑1 '国际在线' |
| L 46 — entropy 17.002 bits, top‑1 '国际在线' |
| L 47 — entropy 17.013 bits, top‑1 '主义思想' |
| L 48 — entropy 17.022 bits, top‑1 '主义思想' |
| L 49 — entropy 17.022 bits, top‑1 ' reuseIdentifier' |
| L 50 — entropy 16.968 bits, top‑1 'uckets' |
| L 51 — entropy 16.972 bits, top‑1 ' "' |
| L 52 — entropy 17.009 bits, top‑1 '"' |
| L 53 — entropy 16.927 bits, top‑1 '"' |
| L 54 — entropy 16.908 bits, top‑1 '"' |
| L 55 — entropy 16.942 bits, top‑1 '"' |
| L 56 — entropy 16.938 bits, top‑1 '"' |
| L 57 — entropy 16.841 bits, top‑1 ' "' |
| L 58 — entropy 16.915 bits, top‑1 ' "' |
| L 59 — entropy 16.920 bits, top‑1 ' "' |
| L 60 — entropy 16.886 bits, top‑1 ' '' |
| L 61 — entropy 16.903 bits, top‑1 ' '' |
| L 62 — entropy 16.834 bits, top‑1 ' "' |
| L 63 — entropy 16.891 bits, top‑1 ' "' |
| L 64 — entropy 16.895 bits, top‑1 ' "' |
| L 65 — entropy 16.869 bits, top‑1 ' "' |
| L 66 — entropy 16.899 bits, top‑1 ' "' |
| L 67 — entropy 16.893 bits, top‑1 ' "' |
| L 68 — entropy 16.779 bits, top‑1 ' "' |
| L 69 — entropy 16.876 bits, top‑1 ' "' |
| L 70 — entropy 16.787 bits, top‑1 ' "' |
| L 71 — entropy 16.505 bits, top‑1 ' "' |
| L 72 — entropy 16.650 bits, top‑1 ' "' |
| L 73 — entropy 15.787 bits, top‑1 ' "' |
| L 74 — entropy 16.081 bits, top‑1 ' "' |
| L 75 — entropy 13.350 bits, top‑1 ' "' |
| L 76 — entropy 14.743 bits, top‑1 ' "' |
| L 77 — entropy 10.848 bits, top‑1 ' "' |
| L 78 — entropy 15.398 bits, top‑1 ' "' |
| L 79 — entropy 16.666 bits, top‑1 ' "' |
| **L 80 — entropy 4.116 bits, top‑1 ' Berlin'** |

Control margin (JSON `control_summary`): `first_control_margin_pos = 0`, `max_control_margin = 0.2070`.

Ablation (no‑filler): `L_copy_orig = None`, `L_sem_orig = 80`; `L_copy_nf = None`, `L_sem_nf = 80`; `ΔL_copy = None`, `ΔL_sem = 0` (001_layers_baseline/run-latest/output-Qwen2.5-72B.json).

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n/a (strict copy not detected). Soft ΔHₖ (bits) n/a (`L_copy_soft[k]` all null for k ∈ {1,2,3}).

Confidence milestones (pure CSV): p_top1 > 0.30 at layer 77; p_top1 > 0.60 not reached; final‑layer p_top1 = 0.3395 (001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:138).

Rank milestones (JSON diagnostics): rank ≤ 10 at layer 74; rank ≤ 5 at layer 78; rank ≤ 1 at layer 80.

KL milestones (JSON diagnostics): first_kl_below_1.0 at layer 80; first_kl_below_0.5 at layer 80. KL decreases to ≈0 at the final layer; see CSV final row with `kl_to_final_bits=0.000109` (001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:138).

Cosine milestones (pure CSV): first `cos_to_final ≥ 0.2` at layer 0; `≥ 0.4` at layer 0; `≥ 0.6` at layer 53; final `cos_to_final = 1.0000`.

Norm temperature diagnostics (JSON): `tau_norm_per_layer` present; snapshots `kl_to_final_bits_norm_temp@25% {layer=20, value=9.844}`, `@50% {layer=40, value=9.814}`, `@75% {layer=60, value=7.634}`.

Prism Sidecar Analysis
- Presence: compatible=True (`k=512`, layers sampled at embed/19/39/59). 
- Early‑depth stability: KL(P_layer||P_final) baseline vs Prism at depths [0,20,40,60,80] = [(9.39, 9.49), (12.62, 9.46), (12.40, 9.57), (9.11, 9.65), (0.00011, 20.68)].
- Rank milestones: Prism never reaches rank ≤ {10,5,1} (baseline: 74/78/80).
- Top‑1 agreement: Disagrees at sampled depths 0,20,40,60,80.
- Cosine drift: Baseline vs Prism at [0,20,40] = [(0.588, −0.119), (−0.726, −0.003), (−0.710, −0.130)].
- Copy flags: No flips in L0–L3 (strict/soft all remain False).
- Verdict: Regressive (higher final KL and no earlier rank milestones despite some mid‑depth KL reductions).

4. Qualitative patterns & anomalies

Strict copy‑collapse and soft‑copy flags do not fire in L0–L3 (“copy‑reflex” not observed). Negative control shows no semantic leakage: “Berlin is the capital of” → top‑5: [“ Germany” 0.7695, “ the” 0.0864, “ which” 0.0491, “ a” 0.0125, “ what” 0.0075] (001_layers_baseline/run-latest/output-Qwen2.5-72B.json). Temperature robustness: at T = 0.1, Berlin rank 1 (p = 0.953; entropy 0.275 bits); at T = 2.0, Berlin remains in top‑k with much lower mass (p = 0.016; entropy 15.013 bits) (same JSON).

Important‑word trajectory (records CSV). Berlin appears with small mass in adjacent context positions before final collapse; e.g., at layer 74 (NEXT‑adjacent “simply”), “… (‘Berlin’, 9.53e‑04)” 001_layers_baseline/run-latest/output-Qwen2.5-72B-records.csv:3911; at layer 77, the same slot shows “… (‘Berlin’, 4.81e‑03)” 001_layers_baseline/run-latest/output-Qwen2.5-72B-records.csv:3999. This aligns with a late semantic crystallization (first_rank≤10 at L74; ≤1 at L80).

Collapse‑layer index does not shift under the stylistic ablation (“simply” removed): `L_sem_orig=80`, `L_sem_nf=80` (ΔL_sem=0), indicating minimal dependence on the adverb for this probe (001_layers_baseline/run-latest/output-Qwen2.5-72B.json).

Rest‑mass sanity: after L_semantic the maximum `rest_mass` is 0.298 (001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:138), consistent with limited top‑k coverage but no precision loss spike.

Rotation vs amplification: `kl_to_final_bits` stays high through most of the stack and collapses only at L80 while `cos_to_final` is already ≥0.4 from L0. This indicates “early direction, late calibration.” For example, at L0: `cos_to_final=0.588` with `kl_to_final_bits=9.391`; at L53: `cos_to_final≈0.600` while KL is still large. Final‑head calibration is clean: `top1_agree=True`, `temp_est=1.0`, `kl_after_temp_bits=0.000109`, `warn_high_last_layer_kl=False` (001_layers_baseline/run-latest/output-Qwen2.5-72B.json).

Lens sanity: raw‑vs‑norm check reports `mode=sample`, `lens_artifact_risk=high`, `max_kl_norm_vs_raw_bits=19.91`, `first_norm_only_semantic_layer=None` (001_layers_baseline/run-latest/output-Qwen2.5-72B.json). Treat any apparent pre‑final “early semantics” cautiously; prefer rank milestones and within‑model comparisons.

Checklist
✓ RMS lens
n.a. LayerNorm bias removed (RMS; not needed)
✗ Entropy rise at unembed (entropy drops strongly at L80)
✓ FP32 un‑embed promoted
✗ Punctuation / markup anchoring (no early dominance)
✗ Copy‑reflex (no strict/soft hits in L0–L3)
✗ Grammatical filler anchoring (no early {is, the, a, of})

5. Limitations & data quirks

KL is lens‑sensitive; despite near‑zero final KL, mid‑depth KL remains high—prefer rank milestones for cross‑model claims. Raw‑vs‑norm sanity was sampled and flags `lens_artifact_risk=high`; treat early‑depth interpretations cautiously. Surface‑mass and rest‑mass depend on tokenizer granularity; avoid cross‑model comparisons of absolute masses.

6. Model fingerprint

“Qwen2.5‑72B: collapse at L 80; final entropy 4.12 bits; ‘Berlin’ locks in only at the final head.”

---
Produced by OpenAI GPT-5 
*Run executed on: 2025-09-28 17:22:48*
