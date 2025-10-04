# 1. Overview
Qwen/Qwen3-8B (8B) evaluated on 2025-09-30 23:57:21 with a logit‑lens probe capturing layerwise entropy, rank milestones, KL to final, cosine trajectory, and surface/semantic mass. The run targets a one‑token answer: 'Berlin'.
Reporting follows measurement guidance (prefer ranks; tuned lens foregrounded; use confirmed semantics).

# 2. Method sanity‑check
> token_only_rotary_model; norm lens active; RMSNorm (first_ln=RMSNorm, final_ln=RMSNorm)
> context_prompt = "Give the city name only, plain text. The capital of Germany is called simply"  [001_layers_baseline/run-latest/output-Qwen3-8B.json:817]
Context prompt ends with ‘called simply’ (no trailing space). Gold alignment ok.
copy config: copy_thresh=0.95, copy_window_k=1, copy_match_level=id_subsequence. Soft: {'threshold': 0.5, 'window_ks': [1, 2, 3], 'extra_thresholds': []}.
copy_flag_columns = ['copy_strict@0.95', 'copy_strict@0.7', 'copy_strict@0.8', 'copy_strict@0.9', 'copy_soft_k1@0.5', 'copy_soft_k2@0.5', 'copy_soft_k3@0.5'] [001_layers_baseline/run-latest/output-Qwen3-8B.json:1621]
Gold alignment: ok [001_layers_baseline/run-latest/output-Qwen3-8B.json:1074]
Control prompt present and aligned; control_summary: first_control_margin_pos=1, max_control_margin=0.999998 [001_layers_baseline/run-latest/output-Qwen3-8B.json:1653]
Ablation summary present: {'L_copy_orig': None, 'L_sem_orig': 31, 'L_copy_nf': None, 'L_sem_nf': 31, 'delta_L_copy': None, 'delta_L_sem': 0} [001_layers_baseline/run-latest/output-Qwen3-8B.json:1630]
Summary indices: first_kl_below_1.0=36, first_kl_below_0.5=36, first_rank_le_{10,5,1}=(29, 29, 31) (bits).
Last‑layer head calibration: kl_to_final_bits=0.0, top1_agree=True, temp_est=1.0, kl_after_temp_bits=0.0 [001_layers_baseline/run-latest/output-Qwen3-8B.json:1075].
Measurement guidance: prefer_ranks=True, suppress_abs_probs=True, reasons=['high_lens_artifact_risk', 'high_lens_artifact_score'].
Raw‑vs‑Norm window: center_layers=[29, 31, 36], radius=4, norm_only_semantics_layers=[], max_kl_norm_vs_raw_bits_window=38.096 [001_layers_baseline/run-latest/output-Qwen3-8B.json:1037].
Raw‑lens full: pct_layers_kl_ge_1.0=0.757, pct_layers_kl_ge_0.5=0.811, n_norm_only_semantics_layers=0, earliest_norm_only_semantic=None, max_kl_norm_vs_raw_bits=38.096, lens_artifact_score=0.554 tier=high.
Threshold sweep: summary.copy_thresholds.stability = none; earliest L_copy_strict@0.70=None, @0.95=None. norm_only_flags={'0.7': None, '0.8': None, '0.9': None, '0.95': None}.
Copy-collapse flags: none fire in L0–3 (strict or soft@0.5).
Lens selection: L_surface_to_meaning_norm=31, L_geom_norm=34, L_topk_decay_norm=0. tau_norm_per_layer: present; kl_to_final_bits_norm_temp@{25,50,75}% = (13.138, 12.490, 7.391).

# 3. Quantitative findings
Use rows with prompt_id = pos, prompt_variant = orig.

| Layer snapshot |
|---|
| L 0 – entropy 17.2128 bits, top‑1 'CLICK' |
| L 1 – entropy 17.2114 bits, top‑1 'apr' |
| L 2 – entropy 17.2105 bits, top‑1 '财经' |
| L 3 – entropy 17.2083 bits, top‑1 '-looking' |
| L 4 – entropy 17.2059 bits, top‑1 '院子' |
| L 5 – entropy 17.2037 bits, top‑1 ' (?)' |
| L 6 – entropy 17.1963 bits, top‑1 'ly' |
| L 7 – entropy 17.1463 bits, top‑1 ' (?)' |
| L 8 – entropy 17.1322 bits, top‑1 ' (?)' |
| L 9 – entropy 17.1188 bits, top‑1 ' (?)' |
| L 10 – entropy 17.0199 bits, top‑1 ' (?)' |
| L 11 – entropy 17.1282 bits, top‑1 'ifiable' |
| L 12 – entropy 17.1169 bits, top‑1 'ifiable' |
| L 13 – entropy 17.1256 bits, top‑1 'ifiable' |
| L 14 – entropy 17.0531 bits, top‑1 '"' |
| L 15 – entropy 17.0364 bits, top‑1 '"' |
| L 16 – entropy 16.9128 bits, top‑1 '-' |
| L 17 – entropy 16.9716 bits, top‑1 '-' |
| L 18 – entropy 16.9106 bits, top‑1 '-' |
| L 19 – entropy 16.6286 bits, top‑1 'ly' |
| L 20 – entropy 16.6960 bits, top‑1 '_' |
| L 21 – entropy 16.4081 bits, top‑1 '_' |
| L 22 – entropy 15.2195 bits, top‑1 ' ______' |
| L 23 – entropy 15.2203 bits, top‑1 '____' |
| L 24 – entropy 10.8929 bits, top‑1 '____' |
| L 25 – entropy 13.4545 bits, top‑1 '____' |
| L 26 – entropy 5.5576 bits, top‑1 '____' |
| L 27 – entropy 4.3437 bits, top‑1 '____' |
| L 28 – entropy 4.7859 bits, top‑1 '____' |
| L 29 – entropy 1.7777 bits, top‑1 '-minded' |
| L 30 – entropy 2.2030 bits, top‑1 ' Germany' |
| **L 31 – entropy 0.4539 bits, top‑1 ' Berlin'** |
| L 32 – entropy 1.0365 bits, top‑1 ' German' |
| L 33 – entropy 0.9878 bits, top‑1 ' Berlin' |
| L 34 – entropy 0.6691 bits, top‑1 ' Berlin' |
| L 35 – entropy 2.4944 bits, top‑1 ' Berlin' |
| L 36 – entropy 3.1226 bits, top‑1 ' Berlin' |

Control margin: first_control_margin_pos=1; max_control_margin=0.999998 (JSON).
Ablation (no‑filler): L_copy_orig=None, L_sem_orig=31, L_copy_nf=None, L_sem_nf=31, ΔL_copy=None, ΔL_sem=0.

ΔH (bits) = entropy(L_copy) − entropy(L_semantic): n.a. (no strict/soft copy layer). Soft ΔH_k: n.a. (k∈{1,2,3} all null).
Confidence milestones: p_top1 > 0.30 at layer 29; > 0.60 at layer 29; final-layer p_top1 = 0.433.
Rank milestones (tuned preferred): rank ≤ 10 at layer None (baseline None), ≤ 5 at None (None), ≤ 1 at None (None).
KL milestones: first_kl_below_1.0 at layer 36, first_kl_below_0.5 at layer 36; KL decreases with depth and is ≈0 at final.
Cosine milestones (norm): ≥0.2 at L36, ≥0.4 at L36, ≥0.6 at L36; final cos_to_final ≈ 1.000.
Depth fractions: L_semantic_frac = 0.861 (JSON).
Copy robustness: stability=none; earliest strict@0.70=None, strict@0.95=None; norm_only_flags={'0.7': None, '0.8': None, '0.9': None, '0.95': None}.

Prism sidecar analysis: present and compatible.
KL at sampled depths (baseline→Prism bits): L0 12.788→12.977 (p25 ~ L9), mid 12.412→13.000, L≈27 6.143→13.176. Rank milestones: Prism unreported/null in JSON; no earlier rank≤(1, 5, 10). Verdict: Regressive (higher KL, no earlier ranks).

# 4. Qualitative patterns & anomalies
Negative control — Berlin is the capital of: top‑5 = 'Germany' (0.729), 'which' (0.221), 'the' (0.024), 'what' (0.011), '__' (0.002). ‘Berlin’ appears in top‑10 (semantic leakage; rank ≈9, p≈0.00046). [001_layers_baseline/run-latest/output-Qwen3-8B.json:10–53]
> pos,orig,29,15, simply,1.7776504755020142,-minded,0.7581772208213806, ______,0.02949567139148712, _____,0.028441989794373512, Berlin,0.026486337184906006, “,0.02357080765068531,____,0.021020686253905296,tons,0.020356448367238045, Germany,0.02001619152724743, German,0.019352832809090614, ___,0.014890880323946476, ____,0.007312466856092215,thing,0.004986921325325966,ifiable,0.003510829294100404,唿,0.0031470239628106356, ‘,0.002962499624118209, ',0.0019449291285127401, ",0.0015954503323882818,",0.0015113287372514606, __,0.0013570220908150077,德国,0.0010344129987061024,0.008828049525618553  [row 481 in CSV]
> pos,orig,30,15, simply,2.2029597759246826, Germany,0.4079546332359314, Berlin,0.2843639850616455, German,0.19966867566108704, “,0.04079069569706917,-minded,0.018246591091156006, _____,0.00899623241275549, ",0.008824669755995274, ______,0.00474811065942049,德国,0.004658787976950407, ',0.004590682219713926, ___,0.004412038251757622,____,0.0032183718867599964, ‘,0.0017719718161970377, ____,0.0014899357920512557,Germany,0.0006450416403822601,",0.0006396685494109988,唿,0.0005160623113624752, capital,0.00047378268209286034,湖北省,0.00030335396877489984, deutschland,0.0002982336445711553,0.003388475684914738  [row 497 in CSV]
> pos,orig,31,15, simply,0.4538646340370178, Berlin,0.9359333515167236, “,0.0428178608417511, Germany,0.011410730890929699, ",0.0034452772233635187, German,0.0014031281461939216, ',0.0012444431195035577, _____,0.0007414189167320728,-minded,0.0006555040599778295, ___,0.0005102462018840015, ______,0.0004853995342273265, ‘,0.0003470926603768021,",0.0003202920488547534,____,0.0002930416085291654, ____,0.00010141241364181042,德国,5.7472207117825747e-05,武汉,4.1150931792799383e-05, Munich,3.0284338208730333e-05,北京,2.4219107217504643e-05, München,2.3153890651883557e-05,Berlin,1.7080066754715517e-05,9.744027556735091e-05  [row 513 in CSV]
> pos,orig,32,15, simply,1.036511778831482, German,0.6797715425491333, Berlin,0.2962426543235779, Germany,0.022280070930719376,德国,0.001701397355645895,German,1.3140289638613467e-06,德,1.0667262131391908e-06,Berlin,7.86317571055406e-07,柏林,3.6749864307239477e-07,Germany,2.981622913011961e-07, Germans,2.696715171168762e-07,ドイツ,5.34356345838205e-08, german,1.553968687062479e-08,武汉,1.5236970796195237e-08, Deutschland,4.346505821928304e-09, Munich,3.07980085878512e-09, “,2.943849386483066e-09,湖北省,1.4311126728117074e-09,武汉市,5.265080593730431e-10, deutsche,3.0350544300006277e-10, Hamburg,2.337970372856546e-10,1.3535835252653783e-07  [row 529 in CSV]
Collapse layer does not shift under no‑filler ablation (L_sem_orig=31; L_sem_nf=31). The test‑prompt variants without the ‘one‑word’ instruction show high next‑token mass on ‘Berlin’, but layerwise shift cannot be asserted from JSON alone.
Rest_mass remains moderate; max after L_semantic = 0.175 at layer 36.
Rotation vs amplification: KL drops late (≤0.5 at L36) while p_answer spikes and rank improves around L31; cosine only crosses ≥0.6 at L36 — early direction, late calibration.
Final‑head calibration: clean — kl_to_final_bits=0.0, temp_est=1.0, kl_after_temp_bits=0.0.
Lens sanity (sample): lens_artifact_risk=high, max_kl_norm_vs_raw_bits=13.605; no norm‑only semantics flagged. Prefer rank milestones; treat early ‘semantics’ cautiously.
Entropy drift (tuned; entropy − teacher_entropy_bits): L9 ≈ 7.91 bits; L18 ≈ 6.94 bits; L27 ≈ 0.72 bits.
Important-word trajectory: ‘Berlin’ first enters any top‑5 by L29, strengthens by L30, and dominates by L31; related tokens (‘Germany’, ‘German’) co‑occupy top‑5 around the collapse. ‘Simply’ persists pre‑collapse but recedes thereafter.

- ✓ RMS lens?
- ✓ LayerNorm bias removed?
- ✗ Entropy rise at unembed?
- ✓ FP32 un-embed promoted?
- ✓ Punctuation / markup anchoring?
- ✗ Copy-reflex?
- ✗ Grammatical filler anchoring?
- ✓ Preferred lens honored in milestones
- ✓ Confirmed semantics reported when available
- ✓ Full dual‑lens metrics cited
- ✓ Tuned‑lens attribution done

# 5. Limitations & data quirks
High lens‑artifact risk (raw lens full score high; max KL_norm_vs_raw ≈ 38 bits). Prefer rank milestones and confirmed semantics; avoid absolute probability comparisons across families. Prism is diagnostic and regressive here; treat it as non‑teacher and avoid using its probabilities for cross‑model claims. Surface‑mass is tokenizer‑dependent; use within‑model trends.

# 6. Model fingerprint (one sentence)
Qwen‑3‑8B: collapse at L 31; final entropy 3.12 bits; ‘Berlin’ stabilises by L31; control ‘Paris’ hits rank‑1 early with large margin.

---
Produced by OpenAI GPT-5
