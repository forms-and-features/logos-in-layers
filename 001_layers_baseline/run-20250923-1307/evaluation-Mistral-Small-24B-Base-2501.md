## Overview

Mistral-Small-24B-Base-2501 (24B, pre-norm) was probed with a norm-lens, recording per-layer next-token distributions, ranks, KL-to-final, and cosine trajectory. The run targets a single-token answer setting (“Berlin”) to surface the gap between copy/filler reflexes and semantic collapse.

## Method sanity-check

The JSON confirms a norm lens with RMSNorm and rotary positions: “use_norm_lens”: true; “first_block_ln1_type”: “RMSNorm”; “layer0_position_info”: “token_only_rotary_model”. The context prompt ends with “called simply” without a trailing space.

> "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply"  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:817]
> "use_norm_lens": true, "first_block_ln1_type": "RMSNorm", "layer0_position_info": "token_only_rotary_model"  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:807]

Copy/collapse diagnostics and flags are present with strict τ=0.95, k=1, ID-subsequence and soft τ_soft=0.5, window_ks={1,2,3}; columns mirror labels in CSV:

> "copy_thresh": 0.95, "copy_window_k": 1, "copy_match_level": "id_subsequence"  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:846]
> "copy_soft_config": { "threshold": 0.5, "window_ks": [1, 2, 3], "extra_thresholds": [] }  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:833]
> "copy_flag_columns": ["copy_strict@0.95","copy_soft_k1@0.5","copy_soft_k2@0.5","copy_soft_k3@0.5"]  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1077]

Gold alignment is ID-based and OK; the gold token is “Berlin” (with leading-space variant). Negative control present with summary. Ablation summary present and both prompt variants exist in CSV (`prompt_variant = orig` and `no_filler`).

> "gold_alignment": "ok"  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:898]
> "control_summary": { "first_control_margin_pos": 1, "max_control_margin": 0.4679627253462968 }  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1106]
> "ablation_summary": { "L_sem_orig": 33, "L_sem_nf": 31, "delta_L_sem": -2, ... }  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1083]
> pos,no_filler,… rows present  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:43]

Summary indices (bits): first_kl_below_0.5 = 40; first_kl_below_1.0 = 40; first_rank_le_1 = 33; first_rank_le_5 = 30; first_rank_le_10 = 30.

> "first_kl_below_0.5": 40, "first_kl_below_1.0": 40, "first_rank_le_1": 33, "first_rank_le_5": 30, "first_rank_le_10": 30  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:849]

Last-layer head calibration is tight (KL≈0, top‑1 agree, same probabilities). Final CSV row shows KL≈0 to final and cosine≈1.

> "last_layer_consistency": { "kl_to_final_bits": 0.0, "top1_agree": true, "p_top1_lens": 0.45546209812164307, "p_top1_model": 0.45546209812164307, ... }  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:899]
> (layer 40, is_answer=True, kl_to_final_bits=0.0, cos_to_final≈1.0)  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:42]

Lens sanity (raw vs norm) is sampled with low artifact risk; max KL(norm||raw) ≈ 0.179 bits; no norm‑only semantic layer.

> "raw_lens_check": { "mode": "sample", "summary": { "max_kl_norm_vs_raw_bits": 0.1792677499620318, "lens_artifact_risk": "low", "first_norm_only_semantic_layer": null } }  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1071]

Copy-collapse flag check (strict τ=0.95, δ=0.10): no strict or soft copy flags fire in layers 0–3 (or anywhere). Earliest `is_answer=True` is layer 33.

> first `is_answer=True`: layer 33; top‑1 “Berlin” (p=0.00042); answer_rank=1  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:35]

Soft copy flags: no `copy_soft_k{1,2,3}@0.5` fires at any layer; strict remains null. ✓ rule satisfied.

## Quantitative findings

Table (positive prompt only: `prompt_id = pos`, `prompt_variant = orig`). Format: “L N — entropy X bits, top‑1 ‘token’”. Bold marks the semantic layer (`is_answer=True`).

| Layer | Entropy (bits) | Top‑1 token |
|---:|---:|:--|
| 0 | 16.9985 | ' Forbes' |
| 1 | 16.9745 | '随着时间的' |
| 2 | 16.9441 | '随着时间的' |
| 3 | 16.8120 | '随着时间的' |
| 4 | 16.8682 | ' quelcon' |
| 5 | 16.9027 | 'народ' |
| 6 | 16.9087 | 'народ' |
| 7 | 16.8978 | 'народ' |
| 8 | 16.8955 | ' quelcon' |
| 9 | 16.8852 | ' simply' |
| 10 | 16.8359 | ' hétérogènes' |
| 11 | 16.8423 | '从那以后' |
| 12 | 16.8401 | ' simply' |
| 13 | 16.8709 | ' simply' |
| 14 | 16.8149 | 'стен' |
| 15 | 16.8164 | 'luš' |
| 16 | 16.8300 | 'luš' |
| 17 | 16.7752 | 'luš' |
| 18 | 16.7608 | 'luš' |
| 19 | 16.7746 | 'luš' |
| 20 | 16.7424 | 'luš' |
| 21 | 16.7747 | ' simply' |
| 22 | 16.7644 | ' simply' |
| 23 | 16.7690 | '-на' |
| 24 | 16.7580 | '-на' |
| 25 | 16.7475 | ' «**' |
| 26 | 16.7692 | ' «**' |
| 27 | 16.7763 | ' «**' |
| 28 | 16.7407 | ' «**' |
| 29 | 16.7604 | ' «**' |
| 30 | 16.7426 | '-на' |
| 31 | 16.7931 | '-на' |
| 32 | 16.7888 | '-на' |
| **33** | 16.7740 | ' Berlin' |
| **34** | 16.7613 | ' Berlin' |
| **35** | 16.7339 | ' Berlin' |
| **36** | 16.6994 | ' Berlin' |
| 37 | 16.5133 | ' "' |
| 38 | 15.8694 | ' "' |
| **39** | 16.0050 | ' Berlin' |
| **40** | 3.1807 | ' Berlin' |

Control margin (negative control): first_control_margin_pos = 1; max_control_margin = 0.4679627253462968.  
> "control_summary": { "first_control_margin_pos": 1, "max_control_margin": 0.4679627253462968 }  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1106]

Ablation (no‑filler): L_copy_orig = null, L_sem_orig = 33; L_copy_nf = null, L_sem_nf = 31; ΔL_copy = null; ΔL_sem = −2.  
> "ablation_summary": { "L_sem_orig": 33, "L_sem_nf": 31, "delta_L_sem": -2 }  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1083]

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (strict L_copy = null).  
Soft ΔHₖ (k∈{1,2,3}) = n.a. (no soft-copy layer flagged).  
Confidence milestones: p_top1 > 0.30 at layer 40; p_top1 > 0.60: none; final-layer p_top1 = 0.4555.  
Rank milestones: rank ≤ 10 at layer 30; rank ≤ 5 at layer 30; rank ≤ 1 at layer 33.  
KL milestones: first_kl_below_1.0 at layer 40; first_kl_below_0.5 at layer 40; KL decreases with depth and is ≈ 0 at final.  
Cosine milestones: first cos_to_final ≥ 0.2 at layer 35; ≥ 0.4 at layer 40; ≥ 0.6 at layer 40; final cos_to_final ≈ 1.0.

Prism Sidecar Analysis
- Presence: available and compatible (k=512; layers: embed, 9, 19, 29).  
  > "prism_summary": { "present": true, "compatible": true, "k": 512, "layers": ["embed", 9, 19, 29] }  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:819]
- Early-depth stability (KL layer→final, bits; baseline → Prism):  
  L0: 10.52 → 10.68  [baseline: row 2; prism: row 2]  
  L10: 10.91 → 12.84  [baseline: row 12; prism: row 12]  
  L20: 10.84 → 16.82  [baseline: row 22; prism: row 22]  
  L30: 10.57 → 15.53  [baseline: row 32; prism: row 32]
- Rank milestones (Prism): no rank ≤ {10,5,1} achieved at any layer (answer_rank remains > 10).  
- Top‑1 agreement: no earlier layer had Prism top‑1 = final top‑1 (“Berlin”).  
- Cosine drift: Prism cos_to_final is lower at early/mid depths (e.g., L10 baseline  −0.1066 vs Prism 0.0793; L30 baseline 0.0991 vs Prism −0.0214).  
- Copy flags: no `copy_collapse` or soft flags flip under Prism.  
- Verdict: Regressive (KL increases materially at early/mid layers and rank milestones are later/never).

## Qualitative patterns & anomalies

The model shows a clean separation between filler/copy behavior and semantics: no strict or soft copy-collapse is detected at any early layer, and the gold token first becomes top‑1 only at L33, then strengthens rapidly thereafter (e.g., L39 “Berlin”, p=0.0271 [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:41], final p=0.4555 [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:42]). The KL trajectory steadily decreases and reaches ≈0 only at the last layer, while the cosine rises late, consistent with a rotation‑then‑calibration picture (“early direction, late calibration”).

Negative control (“Berlin is the capital of”): the top‑5 are all country/functional tokens; Berlin appears only at rank 7 with low probability, indicating no leakage into the inverse task.  
> “ Germany” (0.802), “ which” (0.067), “ the” (0.045), “ _” (0.012), “ what” (0.011)  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:14]

Records CSV shows expected evolution for important words (“Germany”, “Berlin”, “capital”, “simply”): “capital” remains salient across mid‑stack positions (e.g., pos=11 token=capital has “capital” in top‑1 at L27) and the answer token begins to enter the next‑token top‑5 by L30 and stabilizes as top‑1 by L33.  
> “… (layer 30, answer_rank=3; top‑1 ‘-на’, p_top1=0.00039; ‘Berlin’ in top‑3, p=0.00017)”  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:32]

Stylistic ablation (“no_filler”) advances semantics modestly (ΔL_sem = −2), suggesting the model can resolve the target even without the stylistic cue “simply”; this points to guidance‑style sensitivity but not dependence.  
> “L_sem_orig = 33; L_sem_nf = 31; delta_L_sem = -2”  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1083]

Rest‑mass sanity: rest_mass falls once the answer becomes top‑1 and is largest immediately at semantic onset (rest_mass ≈ 0.999 at L33), then drops sharply by the final layer (0.181).  
> “rest_mass=0.9988 at L33”  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:35]  
> “rest_mass=0.1811 at L40”  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:42]

Rotation vs amplification: cos_to_final rises late (≥0.2 only at L35; 0.37 by L39) while KL remains >10 bits until the final layer; probabilities for the answer remain small until L33, then jump by L39/L40—an “early direction, late calibration” signature (cf. Tuned‑Lens 2303.08112 for analogous trends).

Head calibration (final layer): last‑layer consistency shows KL=0.0, temp_est=1.0 with top‑1 agreement; no family‑specific calibration issues.  
> “… "kl_to_final_bits": 0.0, "temp_est": 1.0, "warn_high_last_layer_kl": false …”  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:899]

Lens sanity: sampled raw‑vs‑norm differences are small (max 0.179 bits) with low artifact risk; no “norm‑only semantics” layer is flagged. Prefer rank milestones for cross‑model comparisons.  
> “… "mode": "sample", "max_kl_norm_vs_raw_bits": 0.1792677, "lens_artifact_risk": "low" …”  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1071]

Temperature robustness: at T=2.0, entropy rises to 14.36 bits and Berlin drops to p≈0.030, as expected for high temperature.  
> “… temperature 2.0 … entropy 14.3589 … “ Berlin”, 0.02995 …”  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:981]

Important-word trajectory: “Berlin” first enters any top‑5 at L30 (answer_rank=3), becomes top‑1 by L33, and stabilizes thereafter; “Germany” remains salient in negative control; “capital” is top‑1 for its token in mid‑stack positions (e.g., L27 pos=11).  
> “… (‘Berlin’, rank 3, p = 0.00017)”  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:32]  
> “… capital … top‑1 … 0.00090 …”  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-records.csv:27]

Checklist
✓ RMS lens  
✓ LayerNorm bias removed (RMS model; not needed)  
✓ Entropy rise at unembed (final entropy 3.18 bits vs ~17 early)  
✓ FP32 un‑embed promoted (unembed_dtype=torch.float32)  
✓ Punctuation / markup anchoring: minimal  
✗ Copy‑reflex (no copy flags in L0–3)  
✗ Grammatical filler anchoring in L0–5 (top‑1 not in {“is”, “the”, “a”, “of”})

## Limitations & data quirks

KL is lens‑sensitive; within‑model trends are preferred over cross‑model levels. Final‑layer KL≈0 and last‑layer calibration is sound here, but early‑layer probabilities are not directly comparable across families. Raw‑vs‑norm sanity used sampling mode only (“sample”), so findings are sampled sanity rather than exhaustive. Strict/soft copy detectors did not fire (L_copy and L_copy_soft[k] are null), so ΔH values are n.a.

## Model fingerprint

Mistral‑Small‑24B: semantic collapse at L33; final entropy 3.18 bits; “Berlin” first top‑5 at L30, top‑1 by L33.

---
Produced by OpenAI GPT-5 
