# Evaluation — Meta-Llama-3-8B

## 1. Overview
Meta-Llama-3-8B (meta-llama/Meta-Llama-3-8B) evaluated on 2025-09-28 in run-latest.
The probe captures per-layer entropy, KL to final, cosine geometry, and answer rank dynamics with a norm lens, plus Prism and Tuned-Lens sidecars.

## 2. Method sanity-check
> context_prompt: "Give the city name only, plain text. The capital of Germany is called simply" [diagnostics]
> use_norm_lens=true, unembed_dtype=torch.float32, layer0_position_info=token_only_rotary_model [diagnostics]
Positional encoding and norm lens are active; context ends with “called simply” (no trailing space). Last-layer head calibration aligns: kl_to_final_bits=0.0 and top1_agree=true (p_top1_lens=0.5202 vs p_top1_model=0.5202; temp_est=1.0).
> last_layer_consistency: kl_to_final_bits=0.0, top1_agree=True, p_top1_lens=0.5201817154884338, p_top1_model=0.5201817154884338, temp_est=1.0, kl_after_temp_bits=0.0 [diagnostics.last_layer_consistency]
Copy config present: copy_thresh=0.95, copy_window_k=1, copy_match_level=id_subsequence, copy_soft_config={'threshold': 0.5, 'window_ks': [1, 2, 3], 'extra_thresholds': []}; copy_flag_columns=['copy_strict@0.95', 'copy_soft_k1@0.5', 'copy_soft_k2@0.5', 'copy_soft_k3@0.5'].
Gold alignment: ok (gold_answer.first_id=20437).
Control prompt and summary present (first_control_margin_pos=0, max_control_margin=0.5186312557016208); ablation_summary present: {'L_copy_orig': None, 'L_sem_orig': 25, 'L_copy_nf': None, 'L_sem_nf': 25, 'delta_L_copy': None, 'delta_L_sem': 0}.
Raw-lens sanity mode=sample; summary: lens_artifact_risk=high, max_kl_norm_vs_raw_bits=0.07131043489929646, first_norm_only_semantic_layer=25.
Copy-collapse flag: no strict hits; earliest soft k1=None.

## 3. Quantitative findings
Main prompt (prompt_id=pos, prompt_variant=orig). Table shows layer, entropy (bits), and generic top‑1 token:

| Layer | Entropy | Top‑1 |
|---|---:|---|
| L 0 | 16.96 | 'itzer' |
| L 1 | 16.94 | 'mente' |
| L 2 | 16.88 | 'mente' |
| L 3 | 16.89 | 'tones' |
| L 4 | 16.90 | 'interp' |
| L 5 | 16.87 | '�' |
| L 6 | 16.88 | 'tons' |
| L 7 | 16.88 | 'Exited' |
| L 8 | 16.86 | 'надлеж' |
| L 9 | 16.87 | 'biased' |
| L 10 | 16.85 | 'tons' |
| L 11 | 16.85 | 'tons' |
| L 12 | 16.88 | 'LEGAL' |
| L 13 | 16.84 | 'macros' |
| L 14 | 16.84 | 'tons' |
| L 15 | 16.85 | ' simply' |
| L 16 | 16.85 | ' simply' |
| L 17 | 16.85 | ' simply' |
| L 18 | 16.84 | ' simply' |
| L 19 | 16.84 | ' '' |
| L 20 | 16.83 | ' '' |
| L 21 | 16.83 | ' '' |
| L 22 | 16.83 | 'tons' |
| L 23 | 16.83 | 'tons' |
| L 24 | 16.83 | ' capital' |
| L 25 | 16.81 | ' Berlin' |
| L 26 | 16.83 | ' Berlin' |
| L 27 | 16.82 | ' Berlin' |
| L 28 | 16.82 | ' Berlin' |
| L 29 | 16.80 | ' Berlin' |
| L 30 | 16.79 | ' Berlin' |
| L 31 | 16.84 | ':' |
| L 32 | 2.96 | ' Berlin' |

Control margin: first_control_margin_pos=0; max_control_margin=0.5186312557016208 [control_summary].
Ablation (no‑filler): L_copy_orig=None, L_sem_orig=25; L_copy_nf=None, L_sem_nf=25; ΔL_copy=None, ΔL_sem=0.
ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n/a (L_copy=null).
Soft ΔH_k (bits) = k=1: n/a, k=2: n/a, k=3: n/a.
Confidence: p_top1>0.30 at layer 32; p_top1>0.60 at layer None; final-layer p_top1=0.52.
Rank milestones: rank≤10 at layer 24; rank≤5 at layer 25; rank≤1 at layer 25 [diagnostics].
KL milestones: first_kl_below_1.0 at layer 32; first_kl_below_0.5 at layer 32; KL decreases to ≈0 at final (kl_to_final_bits=0.0).
Cosine milestones: first cos_to_final≥0.2 at L 20; ≥0.4 at L 30; ≥0.6 at L 32; final cos_to_final=1.00 [pure CSV].

Prism Sidecar Analysis
Prism compatible=true; sampled layers=['embed', 7, 15, 23].
Early-depth KL: L0: KL_base=11.57 vs KL_prism=12.28; L8: KL_base=11.81 vs KL_prism=17.18; L16: KL_base=11.73 vs KL_prism=20.01; L24: KL_base=11.32 vs KL_prism=21.10.
Prism rank milestone: first_rank≤1 at layer None (baseline 25).
Verdict: Neutral.

## 4. Qualitative patterns & anomalies
> “Berlin is the capital of” →  Germany (0.90),  the (0.05),  and (0.01),  germany (0.00),  modern (0.00) [test_prompts]
Important-word trajectory (top-5 at selected layers): 
> (layer 0) itzer (0.0000), ikers (0.0000), itchens (0.0000), � (0.0000),  Hem (0.0000) | (layer 8) надлеж (0.0001), apı (0.0001), biased (0.0000), �单 (0.0000), Deserializer (0.0000) | (layer 16)  simply (0.0001), enance (0.0001), tons (0.0001), íme (0.0000), tics (0.0000) | (layer 25)  Berlin (0.0001),  capital (0.0001), tons (0.0001),  Germany (0.0001),  simply (0.0001) | (layer 32)  Berlin (0.5202),  " (0.1865),  “ (0.0588),  ' (0.0418), : (0.0302) [pure CSV]
Stylistic ablation: removing “simply” gives ΔL_sem=0 and ΔL_copy=None (no change in semantic onset).
Rest_mass falls steadily; max after L_semantic = 1.00.
Rotation vs amplification: KL drops late (first<1.0 at L 32) while cos_to_final rises earlier (≥0.2 at L 20), and p_answer turns on at L 25 — early direction, late calibration. Final‑head calibration is good (KL≈0).
Head calibration: warn_high_last_layer_kl=False; temp_est=1.0; kl_after_temp_bits=0.0.
Lens sanity: raw_lens_check.summary → lens_artifact_risk=high, max_kl_norm_vs_raw_bits=0.07131043489929646, first_norm_only_semantic_layer=25 (treat early semantics cautiously; prefer rank milestones).
At T=2.0:  Berlin (0.04),  " (0.02),  “ (0.01),  ' (0.01), : (0.01); entropy=13.87 bits [temperature_exploration].

- RMS lens? ✓
- LayerNorm bias removed? n.a. (RMSNorm model)
- Entropy rise at unembed? Yes
- FP32 un-embed promoted? False (unembed_dtype=torch.float32)
- Punctuation / markup anchoring? Mixed
- Copy-reflex? ✗
- Grammatical filler anchoring? ✗

## 5. Limitations & data quirks
KL is lens-sensitive; final KL≈0 indicates good final-head calibration. Raw‑vs‑norm sampling shows high lens_artifact_risk with first_norm_only_semantic_layer=25; treat early semantics cautiously and rely on rank milestones for cross‑model claims. Surface mass and rest_mass reflect tokenizer coverage; use within‑model trends only.
> raw_lens_check.summary: lens_artifact_risk=high; max_kl_norm_vs_raw_bits=0.07131043489929646 [raw_lens_check.summary]

## 6. Model fingerprint (one sentence)
Llama‑3‑8B: collapse at L 25; final entropy 3.0 bits; ‘Berlin’ is rank 1 from L 25.

---
Produced by OpenAI GPT-5 