# Evaluation Report: meta-llama/Meta-Llama-3-8B
**Overview**
Meta-Llama-3-8B (32 layers). Run artifacts dated 20250923 (see `timestamp-20250923-1651`). This probe captures layer-wise next-token distributions under a norm lens, entropy/KL in bits, copy/semantic collapse, cosine-to-final trajectory, and control margins.

**Method Sanity-Check**
Diagnostics confirm RMS norm lens with rotary positions: "layer0_position_info": "token_only_rotary_model", "use_norm_lens": true, and RMSNorm at first/final LNs: "first_block_ln1_type": "RMSNorm", "final_ln_type": "RMSNorm" (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:810–817). The context prompt ends exactly with “called simply”: "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply" (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:817). Copy/semantic indices and settings are present: "L_copy": null, "L_copy_H": null, "L_semantic": 25, "copy_thresh": 0.95, "copy_window_k": 1, "copy_match_level": "id_subsequence" (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:842–849). Soft detectors: "copy_soft_config": {"threshold": 0.5, "window_ks": [1,2,3], "extra_thresholds": []} and flag labels "copy_flag_columns": ["copy_strict@0.95","copy_soft_k1@0.5","copy_soft_k2@0.5","copy_soft_k3@0.5"] (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:833–841,1077–1082). Gold alignment is ID-based and OK: "gold_alignment": "ok" (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:898). Control prompt and summary exist: context for France/Paris and summary {"first_control_margin_pos": 0, "max_control_margin": 0.5186} (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1091–1109). Ablation present: {"L_copy_orig": null, "L_sem_orig": 25, "L_copy_nf": null, "L_sem_nf": 25, "delta_L_copy": null, "delta_L_sem": 0} (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1083–1090). Summary indices: {"first_kl_below_0.5": 32, "first_kl_below_1.0": 32, "first_rank_le_1": 25, "first_rank_le_5": 25, "first_rank_le_10": 24} (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:849–853). Units are bits for KL/entropy (field names include "_bits"). Last-layer head calibration: per-layer final row in pure CSV at layer 32 shows KL≈0: "..., Berlin,0.520..., ..., kl_to_final_bits,0.0, ... cos_to_final,1.000..." (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:36), and diagnostics block corroborates perfect agreement: {"kl_to_final_bits": 0.0, "top1_agree": true, "p_top1_lens": 0.52018, "p_top1_model": 0.52018, "temp_est": 1.0} (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:899–907). Lens sanity: mode "sample" with summary {"first_norm_only_semantic_layer": 25, "max_kl_norm_vs_raw_bits": 0.0713, "lens_artifact_risk": "high"} (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1071–1075). Treat any “early semantics” before final cautiously and prefer rank milestones.
Copy-collapse flags in layers 0–3 (pos/orig) are all False for both strict τ=0.95, k=1 and soft τ_soft=0.5, k∈{1,2,3} (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:2–5). First strict/soft copy flag: none observed (strict/soft null). Copy-reflex not triggered.

First row with is_answer=True (strict ID-level): layer 25 with top-1 Berlin: "..., 25, …, top1 ' Berlin', …, is_answer,True, … answer_rank,1 …" (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:29). Earliest soft-copy flags (k1–k3) do not fire.

**Quantitative Findings**
Main table (pos, orig; pure next-token):
- L 0 – entropy 16.9568 bits, top-1 'itzer'
- L 1 – entropy 16.9418 bits, top-1 'mente'
- L 2 – entropy 16.8764 bits, top-1 'mente'
- L 3 – entropy 16.8936 bits, top-1 'tones'
- L 4 – entropy 16.8991 bits, top-1 'interp'
- L 5 – entropy 16.8731 bits, top-1 '�'
- L 6 – entropy 16.8797 bits, top-1 'tons'
- L 7 – entropy 16.8806 bits, top-1 'Exited'
- L 8 – entropy 16.8624 bits, top-1 'надлеж'
- L 9 – entropy 16.8666 bits, top-1 'biased'
- L 10 – entropy 16.8506 bits, top-1 'tons'
- L 11 – entropy 16.8541 bits, top-1 'tons'
- L 12 – entropy 16.8770 bits, top-1 'LEGAL'
- L 13 – entropy 16.8430 bits, top-1 'macros'
- L 14 – entropy 16.8351 bits, top-1 'tons'
- L 15 – entropy 16.8467 bits, top-1 ' simply'
- L 16 – entropy 16.8471 bits, top-1 ' simply'
- L 17 – entropy 16.8477 bits, top-1 ' simply'
- L 18 – entropy 16.8392 bits, top-1 ' simply'
- L 19 – entropy 16.8399 bits, top-1 ' '
- L 20 – entropy 16.8304 bits, top-1 ' '
- L 21 – entropy 16.8338 bits, top-1 ' '
- L 22 – entropy 16.8265 bits, top-1 'tons'
- L 23 – entropy 16.8280 bits, top-1 'tons'
- L 24 – entropy 16.8299 bits, top-1 ' capital'
- L 25 – entropy 16.8142 bits, top-1 ' Berlin'  ← semantic layer (is_answer=True)
- L 26 – entropy 16.8285 bits, top-1 ' Berlin'
- L 27 – entropy 16.8194 bits, top-1 ' Berlin'
- L 28 – entropy 16.8194 bits, top-1 ' Berlin'
- L 29 – entropy 16.7990 bits, top-1 ' Berlin'
- L 30 – entropy 16.7946 bits, top-1 ' Berlin'
- L 31 – entropy 16.8378 bits, top-1 ':'

Control margin (from JSON): first_control_margin_pos = 0; max_control_margin = 0.5186 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1106–1109).

Ablation (no‑filler): L_copy_orig = null, L_sem_orig = 25; L_copy_nf = null, L_sem_nf = 25; ΔL_copy = null; ΔL_sem = 0 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1083–1090). Interpretation: removing “simply” did not shift collapse.

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (L_copy = null). Soft ΔHk (k∈{1,2,3}) = n.a. (all L_copy_soft[k] = null). Note: soft detector never fired while strict stayed null.

Confidence milestones (pure CSV): p_top1 > 0.30 at layer 32; p_top1 > 0.60 not reached; final-layer p_top1 = 0.5202 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:36).

Rank milestones (diagnostics): rank ≤ 10 at layer 24; rank ≤ 5 at layer 25; rank ≤ 1 at layer 25 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:851–853).

KL milestones (diagnostics/pure): first_kl_below_1.0 at layer 32; first_kl_below_0.5 at layer 32 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:849–850). Baseline per-layer KL decreases modestly with depth and hits ≈0 only at final (e.g., L0: 11.57 bits; L31: 10.73 bits; L32: 0.0) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:2,35,36). Treat final p_top1 as calibrated within‑model.

Cosine milestones (pure CSV): first cos_to_final ≥ 0.2 at L20; ≥ 0.4 at L30; ≥ 0.6 at L32; final cos_to_final = 1.0000 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:2,30,34,36).

Prism Sidecar Analysis
- Presence: available and compatible (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:819–823).
- Early-depth stability: Prism KL(P_layer||P_final) is higher than baseline at sampled depths (e.g., L0: 12.28 vs 11.57; L24: 21.10 vs 11.32) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token-prism.csv:2; 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:28).
- Rank milestones: no earlier improvements; first_rank_le_{10,5,1} not achieved at sampled Prism depths (no rows with answer_rank ≤ 10).
- Top‑1 agreement: no notable Prism→final agreements that baseline lacked at early layers; final remains Berlin.
- Cosine drift: no earlier stabilization than baseline at sampled depths; cos_to_final remains low early and < 0.2.
- Copy flags: none fire under Prism; no flips relative to baseline.
- Verdict: Regressive (KL increases at early/mid layers; no earlier rank milestones).

**Qualitative Patterns & Anomalies**
Norm‑only semantics risk is high; JSON reports first_norm_only_semantic_layer = 25 with lens_artifact_risk = "high" and max_kl_norm_vs_raw_bits ≈ 0.071 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1071–1075). Prefer rank milestones over absolute probabilities for pre‑final layers.
Negative control — “Berlin is the capital of”: top‑5 " Germany" 0.8955, " the" 0.0525, " and" 0.0075, " germany" 0.00336, " modern" 0.00297; Berlin still appears 6th: 0.00289 → semantic leakage: Berlin rank 6 (p = 0.00289) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:10–36).
Important‑word trajectory (pos/orig, NEXT position): “ simply” dominates mid‑stack before semantics, e.g., " top1 ' simply' … p=7.00e‑05" (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-records.csv:307). “ capital” becomes top‑1 at L24: " top1 ' capital', p=8.01e‑05" (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-records.csv:426). Berlin becomes top‑1 at L25: " top1 ' Berlin', p=1.32e‑04" (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-records.csv:443), and remains top‑5 through L31. “Germany” enters top‑5 at L25 and persists through L27 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-records.csv:443,460,477).
Collapse‑layer shift without the “one‑word” instruction: not directly measurable from test‑prompt snapshots (they lack per‑layer traces), but the ablation summary shows ΔL_sem = 0 between orig/no_filler (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1083–1090), suggesting robustness to filler removal for this prompt.
Rest‑mass sanity: rest_mass stays ≈1.0 pre‑final and peaks after L_semantic at 0.99955 (L25), then drops at final (0.1625) due to concentrated top‑k (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:29,36). Treat as top‑k coverage only, not lens fidelity.
Rotation vs amplification: cos_to_final rises while KL remains high (e.g., L24 cos=0.282 with KL≈11.32), then calibrates at final (KL→0), indicating early direction, late calibration (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:28).
Head calibration (final): last_layer_consistency shows perfect alignment and no temperature correction needed: temp_est=1.0; kl_after_temp_bits=0.0; warn_high_last_layer_kl=false (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:899–907,917).
Lens sanity: mode "sample"; summary as above. Example sample at L25 shows norm identifies answer rank 1 while raw rank 3, reflecting norm‑only semantics risk: "answer_rank_norm": 1 vs "answer_rank_raw": 3 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1058–1066).
Temperature robustness: at T=0.1, Berlin rank 1 (p=0.99996, entropy≈0.00057 bits); at T=2.0, Berlin rank 1 (p=0.0366, entropy≈13.87 bits) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:669–676,737–744).
Checklist: RMS lens ✓ (RMSNorm; norm lens enabled); LayerNorm bias removed? ✓ ("not_needed_rms_model"); Entropy rise at unembed? ✗ (final entropy 2.961 bits vs lens rows ≈16.8 bits); FP32 un‑embed promoted? ✓ ("mixed_precision_fix": "casting_to_fp32_before_unembed", "unembed_dtype": "torch.float32") (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:809,815). Punctuation/markup anchoring? ✓ (late layers top‑1 include ':' and quotes; e.g., final‑1: ':' top‑1) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:35). Copy‑reflex? ✗ (no early strict/soft flags in L0–3). Grammatical filler anchoring? ✗ (no {is,the,a,of} as top‑1 in L0–5).

**Limitations & Data Quirks**
- High lens‑artifact risk (norm‑only semantics at L25); prefer rank milestones and within‑model trends. Raw‑vs‑norm sampling is "sample" mode, not exhaustive (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1015–1075).
- Pre‑final KL remains ≈10–12 bits; treat absolute p_top1 from lens rows cautiously and avoid cross‑family comparisons. Use final row (layer 32) for calibrated probabilities.
- Rest_mass ≈1.0 across most layers reflects top‑k truncation, not fidelity; final drop reflects concentrated mass in top tokens.

**Model Fingerprint**
Llama‑3‑8B: semantic collapse at L 25; final p_top1(Berlin)=0.520; KL→0 only at L32; mid‑stack “simply/ capital” precedes Berlin.

---
Produced by OpenAI GPT-5 
*Run executed on: 2025-09-23 16:51:10*
