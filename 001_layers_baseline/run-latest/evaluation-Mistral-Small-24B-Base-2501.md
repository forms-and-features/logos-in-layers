# Evaluation Report: mistralai/Mistral-Small-24B-Base-2501
**1. Overview**
Mistral‑Small‑24B‑Base‑2501 (24B) was probed on 2025‑10‑04 (Experiment started: 2025‑10‑04 18:04:23; see timestamp-20251004-1804:1). The run analyzes layer‑by‑layer next‑token behavior with a norm lens calibrated per architecture (pre‑norm; next block ln1) to trace rank milestones, KL to final, cosine trajectory, copy detectors, and control margins.

The probe captures late semantic stabilization for “Berlin” with confirmed collapse at L 33 under the norm lens (raw corroboration), tuned‑lens later at L 39. Final‑head calibration is clean (KL≈0), so probability calibration at the last layer is trustworthy within‑model.

**2. Method Sanity‑Check**
JSON confirms intended prompt and lens: “Give the city name only, plain text. The capital of Germany is called simply” (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4). Positional encoding info indicates a token‑only rotary model at layer 0 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:816). The norm lens is enabled with FP32 unembed weights and RMSNorm stack, using next‑block ln1 as the source (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:807,809; 4049–4055). The prompt ends with “called simply” with no trailing space (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4).

Copy thresholds and flags are wired as expected: strict τ=0.95, k=1, ID‑subsequence, with sweep {0.70,0.80,0.90,0.95} and soft window_ks={1,2,3} at τ_soft=0.5 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:3899–3907,3956–3961,3962–3967,3908–3916,3929–3940,890–897). Flag columns mirror these (copy_strict@{0.95,0.7,0.8,0.9}, copy_soft_k{1,2,3}@0.5) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:5210–5216). Gold‑answer alignment is ID‑based and ok (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:5735–5743,5746–5755).

Normalizer provenance and per‑layer effect are present: strategy="next_ln1", ln_source=blocks[0].ln1 at L0 and blocks[39].ln1 at final; eps_inside_sqrt=true; scale_gamma_used=true; with resid_norm_ratio and delta_resid_cos reported (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4049–4058,4365–4370). Unembedding bias is absent (present=false; l2_norm=0) ensuring bias‑free geometry (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:826–830). Environment determinism is on (torch 2.8, device=cpu, deterministic_algorithms=true, seed=316) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:5195–5206). Numeric health is clean (any_nan=false, any_inf=false, layers_flagged=[]) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4630–4636). Copy mask size is 2931 with a plausible whitespace/punctuation sample (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:3875–3893).

Collapse diagnostics exist: L_copy, L_copy_soft[k], L_semantic, delta_layers, L_copy_soft (per k), and delta_layers_soft are present (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:3895–3922). Summary indices are populated: first_kl_below_{0.5,1.0}=40; first_rank_le_{1,5,10}={33,30,30} (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:3902–3906). Units are bits for KL and teacher entropy (columns kl_to_final_bits, teacher_entropy_bits; pure CSV header). Last‑layer head calibration is perfect: kl_to_final_bits=0.0, top1_agree=true, p_top1_lens=p_top1_model=0.4555, temp_est=1.0, kl_after_temp_bits=0.0 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4639–4646).

Measurement guidance: prefer ranks and suppress absolute probabilities; prefer tuned for reporting; use confirmed semantics; reason=normalization_spike (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:5735–5744). Raw‑vs‑Norm window: radius=4; centers=[30,33,40]; no norm‑only semantic layers; max_kl_norm_vs_raw_bits_window=5.98 bits; mode=window (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4008–4035). Full raw‑vs‑norm: pct_layers_kl_ge_{1.0,0.5}=0.0244; n_norm_only_semantics_layers=0; max_kl_norm_vs_raw_bits=5.98; lens_artifact_score=0.1146 (tier=low) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4036–4045). Prism is present and compatible (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:834–839).

Copy‑collapse flag check (strict τ=0.95; δ=0.10): no True flags in L0–3; soft k1@0.5 also absent in L0–3 (pure CSV rows for L0–3).

**3. Quantitative Findings**
| Layer | Entropy (bits) | Top‑1 token |
|---:|---:|:---|
| L 0 | 16.999 |  Forbes |
| L 1 | 16.975 | 随着时间的 |
| L 2 | 16.944 | 随着时间的 |
| L 3 | 16.812 | 随着时间的 |
| L 4 | 16.868 |  quelcon |
| L 5 | 16.903 | народ |
| L 6 | 16.909 | народ |
| L 7 | 16.898 | народ |
| L 8 | 16.896 |  quelcon |
| L 9 | 16.885 |  simply |
| L 10 | 16.836 |  hétérogènes |
| L 11 | 16.842 | 从那以后 |
| L 12 | 16.840 |  simply |
| L 13 | 16.871 |  simply |
| L 14 | 16.815 | стен |
| L 15 | 16.816 | luš |
| L 16 | 16.830 | luš |
| L 17 | 16.775 | luš |
| L 18 | 16.761 | luš |
| L 19 | 16.775 | luš |
| L 20 | 16.742 | luš |
| L 21 | 16.775 |  simply |
| L 22 | 16.764 |  simply |
| L 23 | 16.769 | -на |
| L 24 | 16.758 | -на |
| L 25 | 16.747 |  «** |
| L 26 | 16.769 |  «** |
| L 27 | 16.776 |  «** |
| L 28 | 16.741 |  «** |
| L 29 | 16.760 |  «** |
| L 30 | 16.743 | -на |
| L 31 | 16.793 | -на |
| L 32 | 16.789 | -на |
| **L 33** | 16.774 |  Berlin |
| L 34 | 16.761 |  Berlin |
| L 35 | 16.734 |  Berlin |
| L 36 | 16.699 |  Berlin |
| L 37 | 16.513 |  " |
| L 38 | 15.869 |  " |
| L 39 | 16.005 |  Berlin |
| L 40 | 3.181 |  Berlin |

Control margin (JSON): first_control_margin_pos=1; max_control_margin=0.468 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:5241–5243).

Ablation (no‑filler): L_copy_orig=null; L_sem_orig=33; L_copy_nf=null; L_sem_nf=31; ΔL_copy=null; ΔL_sem=−2 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:5218–5224). Interpretation: removing “simply” slightly advances semantics (earlier by 2 layers) without inducing copy.

Confidence milestones (pure CSV, pos/orig): p_top1>0.30 at L 40; p_top1>0.60 not reached; final p_top1=0.4555 at L 40 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4639–4643; pure CSV L40).

Rank milestones (preferred=tuned; baseline in parentheses): rank≤10 at L 34 (30), rank≤5 at L 33 (30), rank=1 at L 39 (33) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:5259–5261; 3904–3906).

KL milestones: first_kl_below_1.0 at L 40; first_kl_below_0.5 at L 40; KL decreases with depth and is ≈0 at final (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:3902–3903,4639).

Cosine milestones (JSON): cos_to_final ≥0.2 at L 35, ≥0.4 at L 40, ≥0.6 at L 40; final cos_to_final≈1.0 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:3993–3998; pure CSV L40).

Depth fractions: L_semantic_frac=0.825; first_rank_le_5_frac=0.75 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4000–4006).

Copy robustness (threshold sweep): stability="none"; earliest strict L_copy at τ∈{0.70,0.95} is null; norm_only_flags are null (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:3955–3981).

Prism sidecar: compatible but regressive — KL(P_layer||P_final) increases markedly at all sampled depths (e.g., baseline vs Prism at L20: 10.84→16.82 bits), and Prism never attains rank≤10 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:865–885; prism pure CSV). Verdict: Regressive.

Norm temperature: tau_norm_per_layer present; snapshots of KL after norm‑temp show values at 25/50/75% depth: 10.60, 10.69, 16.71 bits (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4658–4669).

Tuned‑lens attribution (percentiles): at ~25/50/75% depth, ΔKL_tuned={4.19, 4.59, 5.35} bits, ΔKL_temp={0.318, 0.152, −6.15} bits, implying rotation gains ΔKL_rot={3.87, 4.44, 11.50} (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:5716–5730). First_kl_le_1.0 unchanged (40→40) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:5707–5711).

ΔH (strict) = n/a (L_copy_strict=null). Soft ΔHₖ (k∈{1,2,3}) = n/a (all null). Rest_mass falls after semantic collapse; max after L_semantic is 0.9988 at L 33; final rest_mass=0.1811 at L 40 (pure CSV L33/L40). 

**4. Qualitative Patterns & Anomalies**
Negative control (“Berlin is the capital of”): top‑5 = [“ Germany”, 0.802], [“ which”, 0.067], [“ the”, 0.0448], [“ _”, 0.0124], [“ what”, 0.0109]; “ Berlin” appears rank 7 at p=0.00481 — semantic leakage acknowledged (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:10–39).

Important‑word trajectory at NEXT: “Berlin” first enters top‑k by L30–31 (rank 3→2) and becomes rank‑1 at L 33, consolidating through L 36–40; punctuation tokens momentarily occupy top‑1 at L37–L38 before stabilizing (e.g., “ Berlin”, 0.0002636 at L31; “ Berlin”, 0.0004248 at L33) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:33–35). Control prompt is aligned (“France … called simply”; gold=Paris) with positive control margins (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:5226–5232,5241–5243).

Rotation vs amplification: cos_to_final rises by mid‑stack (≥0.2 at L35) while KL remains high until very late (L40), indicating “early direction, late calibration.” Final‑head calibration is clean (kl_to_final_bits=0.0; temp_est=1.0) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:3993–3998,4639–4646). Raw‑vs‑Norm sanity: sample mode shows low risk (max_kl_norm_vs_raw_bits≈0.179; no norm‑only semantics) and full mode reports low lens‑artifact score (0.1146; tier=low) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:5189–5193,4036–4045). 

Temperature robustness: at T=0.1, “Berlin” rank 1 (p=0.9995; entropy≈0.0061); at T=2.0, “Berlin” rank 1 but much lower p=0.0299; entropy≈14.36 bits (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:670–679,736–743).

Stylistic ablation: removing “simply” advances semantic collapse by 2 layers (33→31) without introducing copy (all copy fields null) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:5218–5224,3913–3921).

Rest‑mass sanity: steady decay after L33; no spikes post‑semantic; final rest_mass=0.1811 at L40 (pure CSV L40). 

Checklist
✓ RMS lens? (pre‑norm; next ln1)  
✓ LayerNorm bias removed? (RMS; bias‑free)  
✓ Entropy rise at unembed? (teacher_entropy_bits reported)  
✓ FP32 un‑embed promoted? (unembed_dtype=torch.float32)  
✗ Punctuation/markup anchoring? (brief at L37–L38)  
✗ Copy‑reflex? (no strict/soft hits in L0–3)  
✓ Grammatical filler anchoring? (“simply” appears top‑1 mid‑stack)  
✓ Preferred lens honored (tuned foregrounded; baseline included)  
✓ Confirmed semantics reported (L_semantic_confirmed=33; source=raw)  
✓ Full dual‑lens metrics cited (raw_lens_full tier=low)  
✓ Tuned‑lens attribution (ΔKL_tuned vs ΔKL_temp at 25/50/75%) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:5716–5730)
✓ normalization_provenance present (ln_source at L0/final)  
✓ per‑layer normalizer effect (resid_norm_ratio, delta_resid_cos present)  
✓ unembed bias audited (present=false)  
✓ deterministic_algorithms=true  
✓ numeric_health clean  
✓ copy_mask present and plausible  
✓ layer_map/model_stats present (num_layers=40)  

**5. Limitations & Data Quirks**
- Rest_mass is top‑k coverage, not fidelity; it remains high pre‑semantic and falls late (do not over‑interpret across models). 
- KL is lens‑sensitive; use rank milestones for cross‑model claims. Here final KL≈0 confirms good last‑layer calibration, but early KL magnitudes depend on lens temperature normalization (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4658–4669).
- Raw‑vs‑norm differences were screened in both sample and full modes; lens_artifact_score is low and no norm‑only semantics were found, but max_kl_norm_vs_raw_bits_window reached 5.98 bits in a late window — treat early “semantics” cautiously and prefer rank milestones (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4008–4045).
- Surface‑mass relies on tokenizer vocabulary; compare within‑model only.

**6. Model Fingerprint**
“Mistral‑Small‑24B‑Base‑2501: collapse at L 33 (tuned L 39); final entropy 3.18 bits; ‘Berlin’ emerges by L31 and stabilizes by L33.”

---
Produced by OpenAI GPT-5 
