# Evaluation Report: Qwen/Qwen3-14B
**1. Overview**
Qwen3-14B (Qwen/Qwen3-14B), probed on the Germany→Berlin prompt. The run captures layer-by-layer next-token distributions, entropy and KL-to-final, copy vs semantic collapse, cosine-to-final trajectories, and control/ablation checks.

**2. Method Sanity‑Check**
The context prompt matches and ends with “called simply” (no trailing space):
> "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply"  [001_layers_baseline/run-latest/output-Qwen3-14B.json:4]
Norm lens is applied with RMSNorm and fp32 unembed: use_norm_lens=True, unembed_dtype=torch.float32, layernorm_bias_fix=not_needed_rms_model, layer0_norm_fix=using_real_ln1_on_embeddings, mixed_precision_fix=casting_to_fp32_before_unembed (diagnostics). Final-head calibration is correct: final-row KL≈0 and JSON consistency reports zero KL after temp:
> "kl_to_final_bits": 0.0 … "temp_est": 1.0, "kl_after_temp_bits": 0.0, "warn_high_last_layer_kl": false  [001_layers_baseline/run-latest/output-Qwen3-14B.json]
> (layer 40, p_top1 = 0.3451, kl_to_final_bits = 0.0)  [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:42]
Copy detectors present; strict rule is ID‑level contiguous subsequence (k=1) at τ=0.95, δ=0.10. JSON shows copy configuration and flags: copy_flag_columns = [copy_strict@0.95, 0.7, 0.8, 0.9; copy_soft_k{1,2,3}@0.5]; copy_soft_config = {threshold: 0.5, window_ks: [1,2,3]}. Gold alignment is ok.
Measurement guidance advises ranks over absolute probabilities:
> "measurement_guidance": { "prefer_ranks": true, "suppress_abs_probs": true, reasons: ["high_lens_artifact_risk"] }  [001_layers_baseline/run-latest/output-Qwen3-14B.json]
Raw‑vs‑Norm window: center_layers = [33, 36, 40], radius = 4, norm_only_semantics_layers = [], max_kl_norm_vs_raw_bits_window = 98.58 bits, mode="window" (diagnostics.raw_lens_window).
Lens sanity summary block is not present (raw_lens_check.summary = null); treat early semantics cautiously and prefer rank milestones per guidance.
Threshold sweep exists with τ ∈ {0.70,0.80,0.90,0.95}: stability = "none"; earliest L_copy_strict at τ=0.70 and τ=0.95 are null; norm_only_flags at all τ are null.
Strict copy-reflex check (layers 0–3): copy_collapse=False and copy_soft_k1@0.5=False across L0–L3 in the pure CSV; no early copy reflex ✓.
Control present and aligned: control_prompt uses France→Paris; control_summary = { first_control_margin_pos = 0, max_control_margin = 0.9742 }.
Ablation (no‑filler) present with both variants in CSV (prompt_variant = orig and no_filler).
Summary indices (bits and ranks): first_kl_below_1.0 = 40; first_kl_below_0.5 = 40; first_rank_le_10 = 32; first_rank_le_5 = 33; first_rank_le_1 = 36.

**3. Quantitative Findings**
Table (pos, orig; one row per layer):
- L 0 – entropy 17.21 bits, top-1 ‘梳’
- L 1 – entropy 17.21 bits, top-1 ‘地处’
- L 2 – entropy 17.21 bits, top-1 ‘是一部’
- L 3 – entropy 17.21 bits, top-1 ‘tics’
- L 4 – entropy 17.21 bits, top-1 ‘tics’
- L 5 – entropy 17.21 bits, top-1 ‘-minded’
- L 6 – entropy 17.21 bits, top-1 ‘过去的’
- L 7 – entropy 17.19 bits, top-1 ‘�’
- L 8 – entropy 17.18 bits, top-1 ‘-minded’
- L 9 – entropy 17.19 bits, top-1 ‘-minded’
- L 10 – entropy 17.17 bits, top-1 ‘(?)’
- L 11 – entropy 17.15 bits, top-1 ‘时代的’
- L 12 – entropy 17.17 bits, top-1 ‘といって’
- L 13 – entropy 17.12 bits, top-1 ‘nav’
- L 14 – entropy 17.14 bits, top-1 ‘nav’
- L 15 – entropy 17.15 bits, top-1 ‘唿’
- L 16 – entropy 17.13 bits, top-1 ‘闯’
- L 17 – entropy 17.14 bits, top-1 ‘唿’
- L 18 – entropy 17.10 bits, top-1 ‘____’
- L 19 – entropy 17.08 bits, top-1 ‘____’
- L 20 – entropy 16.93 bits, top-1 ‘____’
- L 21 – entropy 16.99 bits, top-1 ‘年夜’
- L 22 – entropy 16.95 bits, top-1 ‘年夜’
- L 23 – entropy 16.84 bits, top-1 ‘____’
- L 24 – entropy 16.76 bits, top-1 ‘____’
- L 25 – entropy 16.76 bits, top-1 ‘年夜’
- L 26 – entropy 16.67 bits, top-1 ‘____’
- L 27 – entropy 16.03 bits, top-1 ‘____’
- L 28 – entropy 15.23 bits, top-1 ‘____’
- L 29 – entropy 14.19 bits, top-1 ‘这个名字’
- L 30 – entropy 7.79 bits, top-1 ‘这个名字’
- L 31 – entropy 5.16 bits, top-1 ‘____’
- L 32 – entropy 0.82 bits, top-1 ‘____’
- L 33 – entropy 0.48 bits, top-1 ‘____’
- L 34 – entropy 0.59 bits, top-1 ‘____’
- L 35 – entropy 0.67 bits, top-1 ‘____’
- **L 36 – entropy 0.31 bits, top-1 ‘Berlin’**
- L 37 – entropy 0.91 bits, top-1 ‘____’
- L 38 – entropy 1.21 bits, top-1 ‘____’
- L 39 – entropy 0.95 bits, top-1 ‘Berlin’
- L 40 – entropy 3.58 bits, top-1 ‘Berlin’

Gold answer alignment: { string = “Berlin”, first_id = 19846 } (ID‑level). Control margin: first_control_margin_pos = 0; max_control_margin = 0.9742.

Ablation (no‑filler): L_copy_orig = null, L_sem_orig = 36; L_copy_nf = null, L_sem_nf = 36; ΔL_copy = null; ΔL_sem = 0. Interpretation: no shift from removing “simply”.

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (L_copy = null). Soft ΔHk (k∈{1,2,3}) = n.a. (no soft copy layer).

Confidence milestones (pure CSV): p_top1 > 0.30 at L 31; p_top1 > 0.60 at L 32; final-layer p_top1 = 0.3451 [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:42].
Rank milestones (diagnostics): rank ≤ 10 at L 32; rank ≤ 5 at L 33; rank ≤ 1 at L 36.
KL milestones (diagnostics, bits): first_kl_below_1.0 at L 40; first_kl_below_0.5 at L 40; KL decreases with depth and is 0.0 at final (head‑aligned).
Cosine milestones: ge_0.2 at L 5; ge_0.4 at L 29; ge_0.6 at L 36; final cos_to_final = 0.99999.
Depth fractions: L_semantic_frac = 0.90; first_rank_le_5_frac = 0.825.

Copy robustness (threshold sweep): stability = none; earliest strict copy at τ=0.70 = null, at τ=0.95 = null; norm_only_flags all null.

Prism Sidecar Analysis. Presence/compatibility = true. Early-depth stability: Prism KL(P_layer||P_final) is larger than baseline at p25/p50/p75 with deltas ≈ −0.26, −0.25, −0.71 bits (baseline − prism), indicating regression (diagnostics.prism_summary.metrics). Rank milestones under Prism never reach ≤10/5/1 (None), later than baseline (32/33/36). Sample depths: baseline KL bits at L={0,10,20,30,40} ≈ {12.96, 12.91, 12.98, 12.73, 0.00}; Prism ≈ {13.17, 13.17, 13.23, 13.44, 14.84}; cosine drifts are smaller under Prism (e.g., L30 cos 0.058 vs 0.464 baseline). Copy flags do not spuriously flip (no copy_collapse under Prism). Verdict: Regressive.

Tuned‑Lens Comparison. Rank milestones are later with tuned (le_10=33, le_5=34, le_1=39) than baseline (32/33/36). KL medians (bits) at depth percentiles: Δ = KL_norm − KL_tuned ≈ {p25: 4.52, p50: 4.56, p75: 4.10} from sidecar CSVs, indicating sizable KL reduction but not earlier rank collapse.

Teacher entropy drift (baseline, bits): at L20 drift = +13.35; L30 = +4.21; L36 = −3.27; L40 ≈ 0.00 — entropy approaches the teacher at the end while peaking above mid‑stack.

Raw‑vs‑Norm window sidecar: largest kl_norm_vs_raw_bits_window = 98.58 at L 37; norm_only_semantics_layers = [] [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token-rawlens-window.csv:19].

**4. Qualitative Patterns & Anomalies**
Negative control behaves: “Berlin is the capital of” → top‑5 includes “ Germany” (0.632) and function words; “Berlin” does not appear, as expected: > “ Germany, 0.632”  [001_layers_baseline/run-latest/output-Qwen3-14B.json:14–16].
Important‑word trajectory (NEXT position): “Berlin” first enters any top‑5 at L 33 and remains through final; “Germany” does not appear in top‑5 at NEXT across layers; filler/style tokens dominate early layers, then “这个名字/这座城市” precede “Berlin” before semantic collapse.
Collapse‑layer instruction sensitivity: removing “simply” does not shift semantics (L_sem stays 36) per ablation_summary.
Rest‑mass sanity (pure CSV): rest_mass is small at semantic collapse (0.00088 at L 36) and reaches a modest 0.236 at the final layer, reflecting top‑20 coverage rather than fidelity.
Rotation vs amplification: cosine rises early (ge_0.2 by L 5) while KL remains ≈13 bits and p_answer near 0 — an “early direction, late calibration” pattern; calibration consolidates near L 32–36 when rank and probabilities sharpen. Example: “cos_to_final ≈ 0.238, kl_to_final_bits ≈ 12.93” (layer 5)  [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:7]. Final‑head calibration is good (CSV KL=0; JSON last_layer_consistency KL=0.0, temp_est=1.0).
Lens sanity: raw‑vs‑norm window shows very large norm/raw discrepancies around L≈33–38 (max ≈98.58 bits) with raw top‑1 already “ Berlin” while norm stays different at several layers [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token-rawlens-window.csv:17–20]. Measurement guidance flags high lens‑artifact risk; prefer rank milestones.
Temperature robustness: T=0.1 → “ Berlin” p=0.974 (entropy 0.173); T=2.0 → “ Berlin” p=0.036 (entropy 13.161)  [001_layers_baseline/run-latest/output-Qwen3-14B.json].
Important‑word milestones (NEXT): “Berlin” first top‑5 at L 33, stabilises by L 36; “Germany” absent in top‑5 at NEXT; “capital” does not feature at NEXT.
Stylistic ablation: ΔL_sem = 0 and copy remains null; effect points to robust semantics rather than instruction anchoring.

Checklist
✓ RMS lens  
✓ LayerNorm bias removed  
✓ Entropy rise at unembed  
✓ FP32 un-embed promoted (unembed_dtype=torch.float32)  
✗ Punctuation / markup anchoring (only sporadic)  
✗ Copy-reflex (no early strict/soft hits 0–3)  
✗ Grammatical filler anchoring (no {“is”, “the”, “a”, “of”} as top‑1 in L0–5)

**5. Limitations & Data Quirks**
- Rest_mass is top‑k coverage, not fidelity; it stays ≤0.236 post‑semantics and shouldn’t be used as a lens metric.
- KL is lens‑sensitive; raw‑vs‑norm window shows very large discrepancies (max ≈98.58 bits). Per measurement_guidance, prefer rank milestones for cross‑model claims.
- raw_lens_check.summary is absent (mode="sample" via window only); treat norm‑only “early semantics” cautiously even though norm_only_semantics_layers is empty in the window.
- Surface‑mass (echo/answer) depends on tokenizer; keep comparisons within‑model. Final‑head calibration is good (warn_high_last_layer_kl=false).

**6. Model Fingerprint**
Qwen3‑14B: collapse at L 36; final entropy 3.58 bits; “Berlin” enters top‑5 at L 33 and is top‑1 by L 36.

---
Produced by OpenAI GPT-5 
*Run executed on: 2025-09-29 23:35:16*
