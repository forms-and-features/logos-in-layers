**Overview**
- Model: meta-llama/Meta-Llama-3-8B (8B); run timestamp 2025-09-23 (see `001_layers_baseline/run-latest/timestamp-20250923-1307`).
- Probe captures layerwise next-token distributions via a norm lens, copy vs semantic collapse, KL-to-final, cosine-to-final, calibrated ranks, and control margins.

**Method Sanity‑Check**
The JSON confirms the intended RMSNorm lens on a token‑only rotary model and records the exact context prompt. Diagnostics show: “use_norm_lens”: true and “layer0_position_info”: “token_only_rotary_model” (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:807,816). The `context_prompt` matches and ends with “called simply” (no trailing space) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:817). Lens configuration is explicit: first and final norms are RMSNorm; LN bias not needed for RMS; normalization alignment and fp32 unembed cast are recorded (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:810–815). Copy metrics present with strict τ=0.95, k=1, ID‑level subsequence; soft τ_soft=0.5 with window_ks {1,2,3}, no extras, and `copy_flag_columns` mirror these labels (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:846–848,833–841,1077–1082). Summary indices exist: first_kl_below_0.5=32, first_kl_below_1.0=32, first_rank_le_1=25, first_rank_le_5=25, first_rank_le_10=24 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:849–853). Units: CSV and diagnostics use bits for KL/entropy; last‑layer head calibration present with kl_to_final_bits=0.0, top1_agree=true, p_top1_lens=p_top1_model=0.5202, temp_est=1.0 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:900–908,101–108). Lens sanity: raw vs norm check is mode=sample with lens_artifact_risk="high", max_kl_norm_vs_raw_bits=0.0713, and first_norm_only_semantic_layer=25 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1072–1074). Gold alignment is OK both for Berlin and control Paris (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:898,1104–1120). Negative control present with `control_summary` (first_control_margin_pos=0; max_control_margin=0.5186) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1106–1108). Ablation summary exists; `L_sem_orig=25`, `L_sem_nf=25` (ΔL_sem=0), `L_copy` and soft `L_copy_k{1,2,3}` are null (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1083–1089). Copy‑collapse flags in the pure CSV are absent in layers 0–3 (copy_collapse=False; copy_strict@0.95=False; copy_soft_k1@0.5=False) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:2–5) — no early copy reflex. Earliest soft copy (k=1) is not observed (no True anywhere). For the main table below, rows are filtered to `prompt_id=pos`, `prompt_variant=orig` (pure CSV).

**Quantitative Findings**
Table (pos/orig; per layer: entropy in bits; top‑1 token):
- L 0 — entropy 16.9568 bits, top‑1 ‘itzer’
- L 1 — entropy 16.9418 bits, top‑1 ‘mente’
- L 2 — entropy 16.8764 bits, top‑1 ‘mente’
- L 3 — entropy 16.8936 bits, top‑1 ‘tones’
- L 4 — entropy 16.8991 bits, top‑1 ‘interp’
- L 5 — entropy 16.8731 bits, top‑1 ‘�’
- L 6 — entropy 16.8797 bits, top‑1 ‘tons’
- L 7 — entropy 16.8806 bits, top‑1 ‘Exited’
- L 8 — entropy 16.8624 bits, top‑1 ‘надлеж’
- L 9 — entropy 16.8666 bits, top‑1 ‘biased’
- L 10 — entropy 16.8506 bits, top‑1 ‘tons’
- L 11 — entropy 16.8541 bits, top‑1 ‘tons’
- L 12 — entropy 16.8770 bits, top‑1 ‘LEGAL’
- L 13 — entropy 16.8430 bits, top‑1 ‘macros’
- L 14 — entropy 16.8351 bits, top‑1 ‘tons’
- L 15 — entropy 16.8467 bits, top‑1 ‘ simply’
- L 16 — entropy 16.8471 bits, top‑1 ‘ simply’
- L 17 — entropy 16.8477 bits, top‑1 ‘ simply’
- L 18 — entropy 16.8392 bits, top‑1 ‘ simply’
- L 19 — entropy 16.8399 bits, top‑1 ‘ ’’
- L 20 — entropy 16.8304 bits, top‑1 ‘ ’’
- L 21 — entropy 16.8338 bits, top‑1 ‘ ’’
- L 22 — entropy 16.8265 bits, top‑1 ‘tons’
- L 23 — entropy 16.8280 bits, top‑1 ‘tons’
- L 24 — entropy 16.8299 bits, top‑1 ‘ capital’
- **L 25 — entropy 16.8142 bits, top‑1 ‘ Berlin’ (first is_answer=True)**
- L 26 — entropy 16.8285 bits, top‑1 ‘ Berlin’
- L 27 — entropy 16.8194 bits, top‑1 ‘ Berlin’
- L 28 — entropy 16.8194 bits, top‑1 ‘ Berlin’
- L 29 — entropy 16.7990 bits, top‑1 ‘ Berlin’
- L 30 — entropy 16.7946 bits, top‑1 ‘ Berlin’
- L 31 — entropy 16.8378 bits, top‑1 ‘:’
- L 32 — entropy 2.9610 bits, top‑1 ‘ Berlin’ (final) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:36)

Control margin (JSON): first_control_margin_pos=0; max_control_margin=0.5186 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1106–1108).

Ablation (no‑filler) (JSON): L_copy_orig=null, L_sem_orig=25; L_copy_nf=null, L_sem_nf=25; ΔL_copy=null; ΔL_sem=0 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1083–1089). As L_copy is null (strict and soft), ΔH cannot be computed reliably; rely on rank milestones.

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n/a (L_copy=null).
Soft ΔHk (bits) for k∈{1,2,3}: n/a (all L_copy_soft[k]=null).

Confidence milestones (pure CSV): p_top1 > 0.30 at layer 32; p_top1 > 0.60 not reached; final-layer p_top1 = 0.5202 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:36).

Rank milestones (diagnostics): rank ≤ 10 at L 24; rank ≤ 5 at L 25; rank ≤ 1 at L 25 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:851–853,852–853,851).

KL milestones (diagnostics): first_kl_below_1.0 at L 32; first_kl_below_0.5 at L 32 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:850,849). KL decreases with depth and is ≈ 0 at final (kl_to_final_bits=0.0; last layer) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:900–908; pure CSV final row 36).

Cosine milestones (pure CSV): first cos_to_final ≥ 0.2 at L 20; ≥ 0.4 at L 30; ≥ 0.6 at L 32; final cos_to_final ≈ 1.0000 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:36).

Prism Sidecar Analysis
- Presence: compatible=true (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:823). 
- Early‑depth stability (KL to final, sampled layers): baseline vs Prism KL at L={0,8,16,24} are {11.57, 11.81, 11.73, 11.32} vs {12.28, 17.18, 20.01, 21.10} bits, respectively (strongly higher under Prism). 
- Rank milestones: Prism pure CSV never reaches rank ≤ {10,5,1} (answer_rank stays >10 throughout; final answer_rank=190) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token-prism.csv:34). Baseline reaches rank=1 at L 25.
- Top‑1 agreement: at sampled layers L={0,8,16,24}, top‑1 tokens differ (e.g., L16 baseline ‘ simply’ vs Prism ‘frau’).
- Cosine drift: Prism cos_to_final is smaller and can be negative mid‑stack; baseline stabilizes directionally earlier (cos ≥ 0.2 by L 20) while Prism remains low.
- Copy flags: no spurious flips; `copy_collapse` remains False across depths.
- Verdict: Regressive (KL increases markedly and rank milestones are not achieved earlier; final Prism distribution is far from the model’s final head [kl_to_final_bits≈16 bits at L32], see 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token-prism.csv:34).

**Qualitative Patterns & Anomalies**
The negative control “Berlin is the capital of” assigns high mass to “ Germany” with Berlin still visible in top‑10: “ Germany, 0.8955 … Berlin, 0.0029” (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:14–16,34–36). Semantic leakage: Berlin rank 6 (p = 0.0029).

Records (pos/orig) show the trajectory of “Berlin” and nearby words through the context tokens. “Berlin” enters the top‑5 by L22 at the ‘ is’ slot and strengthens through L24–L25, where ‘ simply’/answer slots make Berlin the top item: “L25 … ‘ simply … Berlin, 0.0001323 …” (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-records.csv:482–483,499–501,518). Close geographical distractors (e.g., Washington, Frankfurt, Stuttgart) appear around L24–L26 with smaller mass (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-records.csv:499–501,535).

Removing the stylistic cue (“simply”) does not shift the semantic collapse layer (ΔL_sem = 0), suggesting robustness to that filler; copy detectors remain null (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1083–1089). Rest‑mass is top‑k coverage only; it remains high pre‑final and drops at the last layer (final rest_mass=0.1625; max after L_sem ≈ 0.9996), indicating no precision loss but limited top‑20 coverage before the final head (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:36).

Rotation vs amplification: cosine rises before probability calibration (cos_to_final ≥ 0.2 by L20) while KL remains ≈11 bits, indicating early directional alignment with late calibration (“early direction, late calibration”). By the last layer, KL≈0 and rank=1 with p_answer≈0.52 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:36; 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:900–906).

Head calibration (final layer) is clean: `warn_high_last_layer_kl=false`; `p_top1_lens` matches `p_top1_model` and `temp_est=1.0` (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:900–908).

Lens sanity: raw‑vs‑norm summary flags “lens_artifact_risk”: high and “first_norm_only_semantic_layer”: 25; thus treat any pre‑final “early semantics” cautiously and prefer rank milestones (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1072–1074). A sampled row shows modest norm vs raw divergence (e.g., layer 25: kl_norm_vs_raw_bits=0.0539; is_answer_norm=true vs raw=false), consistent with norm‑only semantics (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1071–1075 and raw_lens_check samples around 216–236).

Temperature robustness: at T=0.1, Berlin rank 1 with p≈0.99996 and entropy≈0.00057 bits (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:670–676,671); at T=2.0, Berlin remains rank 1 with p≈0.0366 and entropy≈13.87 bits (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:737–743,738).

Important‑word trajectory: “Berlin” first enters any top‑5 by L22 and stabilizes by L25; “capital” is prominent around L22–L24; filler/punctuation (“the”, quotes, colons) appear mid‑stack but recede by L25+ (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-records.csv:465–501,516–518).

Checklist
- RMS lens? ✓ (RMSNorm detected; use_norm_lens=true) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:807,810–811)
- LayerNorm bias removed? ✓ (not needed for RMS) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:812)
- Entropy rise at unembed? ✓ (final entropy drops from ~16.8→2.96 bits) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:36)
- FP32 un‑embed promoted? ✓ (`use_fp32_unembed=false` but `unembed_dtype="torch.float32"`; mixed_precision_fix recorded) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:808–815)
- Punctuation / markup anchoring? ✓ (early top‑1 includes punctuation/code‑isms, e.g., “ "($(""#” at L3) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:5)
- Copy‑reflex? ✗ (no strict or soft flags in L0–3) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:2–5)
- Grammatical filler anchoring? ✓ (e.g., ‘ the’, quotes, colon dominate mid‑stack; L15–L18 ‘ simply’)

**Limitations & Data Quirks**
- Rest_mass is top‑k coverage only; high values after L_sem (≈0.9996) do not indicate poor lens fidelity and should not be used as a fidelity metric.
- KL is lens‑sensitive; cross‑model claims should use rank milestones. Here final KL≈0, but early “norm‑only semantics” (first_norm_only_semantic_layer=25) and “lens_artifact_risk=high” warrant caution on pre‑final probabilities; prefer rank thresholds (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1072–1074).
- Raw‑vs‑norm check is `mode=sample`; treat it as sampled sanity rather than exhaustive.
- Prism sidecar is compatible but regressive here (higher KL, no earlier ranks), so do not adjust metrics based on Prism; treat it as a calibration diagnostic only.

**Model Fingerprint**
“Llama‑3‑8B: semantics at L 25; final entropy 2.96 bits; mid‑stack filler ‘ simply’ before ‘ Berlin’ becomes top‑1.”

---
Produced by OpenAI GPT-5 
