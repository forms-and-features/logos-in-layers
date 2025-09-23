# Evaluation Report: mistralai/Mistral-7B-v0.1

*Run executed on: 2025-09-23 16:51:10*
## 1. Overview

Mistral‑7B‑v0.1 (7B, pre‑norm) — run outputs dated timestamp-20250923-1651 — was probed with a norm‑lens pure next‑token analysis over all 32 layers. The probe traces copy/filler collapse vs. semantic emergence for the gold answer and reports calibration (KL to final), cosine trajectory, and rank/entropy milestones.

## 2. Method sanity‑check

The JSON confirms the intended norm lens and positional setup: "use_norm_lens": true and FP32 unembed are recorded ("unembed_dtype": "torch.float32"). Positional encoding is rotary as expected for Mistral. Examples:
> "use_norm_lens": true  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:807]\
> "layer0_position_info": "token_only_rotary_model"  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:816]

Context prompt ends with “called simply” (no trailing space):
> "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply"  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4]

Copy/collapse diagnostics and flags are present with strict and soft detectors, and labels match the CSV: copy_thresh=0.95, copy_window_k=1, match_level=id_subsequence; soft threshold=0.5 with window_ks {1,2,3}. Examples:
> "copy_thresh": 0.95; "copy_window_k": 1; "copy_match_level": "id_subsequence"  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:846–848]\
> "copy_soft_config": { "threshold": 0.5, "window_ks": [1,2,3], "extra_thresholds": [] }  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:833–841]\
> "copy_flag_columns": ["copy_strict@0.95", "copy_soft_k1@0.5", "copy_soft_k2@0.5", "copy_soft_k3@0.5"]  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1077–1081]

Gold‑token alignment is ID‑based and ok; the gold token block is present:
> "gold_alignment": "ok"  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:898]\
> "gold_answer": { "string": "Berlin", "first_id": 8430, "answer_ids": [8430] }  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1110–1118]

Negative control is present with summary:
> "control_prompt" … "The capital of France is called simply"  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1092]\
> "control_summary": { "first_control_margin_pos": 2, "max_control_margin": 0.6539100579732349 }  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1106]

Ablation summary is present and both prompt variants appear in the pure CSV (orig/no_filler):
> "ablation_summary": { "L_copy_orig": null, "L_sem_orig": 25, "L_copy_nf": null, "L_sem_nf": 24, "delta_L_sem": -1 }  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1083–1090]

Summary indices (bits, ranks) from diagnostics:
> "first_kl_below_0.5": 32; "first_kl_below_1.0": 32; "first_rank_le_1": 25; "first_rank_le_5": 25; "first_rank_le_10": 23  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:849–853]

Last‑layer head calibration is clean and present; final KL≈0:
> "last_layer_consistency": { "kl_to_final_bits": 0.0, "top1_agree": true, "p_top1_lens": 0.3822, "p_top1_model": 0.3822, "temp_est": 1.0, "kl_after_temp_bits": 0.0, "warn_high_last_layer_kl": false }  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:899–919]

Lens sanity (raw vs norm): sampled mode with high artifact risk — prefer rank milestones:
> "raw_lens_check": … "summary": { "first_norm_only_semantic_layer": null, "max_kl_norm_vs_raw_bits": 1.1738671535074732, "lens_artifact_risk": "high" }  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1015; 1072–1074]

Copy‑collapse flag check (pure CSV, pos/orig): no strict copy in layers 0–3; earliest copy_collapse=False at all depths; soft flags (k1–k3) never fire. Earliest strict: n/a; earliest soft k1/k2/k3: n/a.

Prism sidecar is present and compatible:
> "prism_summary": { "present": true, "compatible": true, "k": 512, "layers": ["embed", 7, 15, 23] }  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:820–832]

## 3. Quantitative findings

Main rows: prompt_id=pos; prompt_variant=orig.

- L 0 — entropy 14.961 bits, top-1 'dabei'
- L 1 — entropy 14.929 bits, top-1 'biologie'
- L 2 — entropy 14.825 bits, top-1 ','
- L 3 — entropy 14.877 bits, top-1 '[…]'
- L 4 — entropy 14.854 bits, top-1 '[…]'
- L 5 — entropy 14.827 bits, top-1 '[…]'
- L 6 — entropy 14.838 bits, top-1 '[…]'
- L 7 — entropy 14.805 bits, top-1 '[…]'
- L 8 — entropy 14.821 bits, top-1 '[…]'
- L 9 — entropy 14.776 bits, top-1 '[…]'
- L 10 — entropy 14.782 bits, top-1 '[…]'
- L 11 — entropy 14.736 bits, top-1 '[…]'
- L 12 — entropy 14.642 bits, top-1 '[…]'
- L 13 — entropy 14.726 bits, top-1 '[…]'
- L 14 — entropy 14.653 bits, top-1 '[…]'
- L 15 — entropy 14.450 bits, top-1 '[…]'
- L 16 — entropy 14.600 bits, top-1 '[…]'
- L 17 — entropy 14.628 bits, top-1 '[…]'
- L 18 — entropy 14.520 bits, top-1 '[…]'
- L 19 — entropy 14.510 bits, top-1 '[…]'
- L 20 — entropy 14.424 bits, top-1 'simply'
- L 21 — entropy 14.347 bits, top-1 'simply'
- L 22 — entropy 14.387 bits, top-1 '“'
- L 23 — entropy 14.395 bits, top-1 'simply'
- L 24 — entropy 14.212 bits, top-1 'simply'
- **L 25 — entropy 13.599 bits, top-1 'Berlin'**
- L 26 — entropy 13.541 bits, top-1 'Berlin'
- L 27 — entropy 13.296 bits, top-1 'Berlin'
- L 28 — entropy 13.296 bits, top-1 'Berlin'
- L 29 — entropy 11.427 bits, top-1 '"'
- L 30 — entropy 10.797 bits, top-1 '“'
- L 31 — entropy 10.994 bits, top-1 '"'
- L 32 — entropy 3.611 bits, top-1 'Berlin'

Control margin (JSON): first_control_margin_pos=2; max_control_margin=0.6539.\
> "control_summary": { "first_control_margin_pos": 2, "max_control_margin": 0.6539100579732349 }  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1106]

Ablation (no‑filler): L_copy_orig=null; L_sem_orig=25; L_copy_nf=null; L_sem_nf=24; ΔL_copy=n/a; ΔL_sem=−1.\
> "ablation_summary": …  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1083–1090]

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n/a (no strict copy).\
Soft ΔHₖ (bits): k=1/2/3 n/a (no soft copy).

Confidence milestones (pure CSV): p_top1 > 0.30 at L 32; p_top1 > 0.60: none; final-layer p_top1 = 0.3822.

Rank milestones (JSON): rank ≤10 at L 23; ≤5 at L 25; ≤1 at L 25.\
> "first_rank_le_10": 23; "first_rank_le_5": 25; "first_rank_le_1": 25  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:851–853]

KL milestones (JSON): first_kl_below_1.0 at L 32; first_kl_below_0.5 at L 32. KL decreases with depth and is ≈0 at the final layer (pure CSV L0 KL=10.17 bits; final KL=0.0).\
> L0 "kl_to_final_bits": 10.1748 … L32 "kl_to_final_bits": 0.0  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:2; 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:899]

Cosine milestones (pure CSV): first cos_to_final ≥0.2 at L 11; ≥0.4 at L 25; ≥0.6 at L 26; final cos_to_final ≈ 1.0.

### Prism Sidecar Analysis

Presence: compatible=true; k=512; layers=[embed,7,15,23].\
> "prism_summary": …  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:820–832]

Early‑depth stability (KL to final, baseline → prism at sampled depths):
- L0: 10.17 → 10.40 bits (Δ −0.23)
- L8: 10.25 → 23.06 bits (Δ −12.81)
- L16: 10.33 → 27.87 bits (Δ −17.54)
- L24: 9.05 → 26.55 bits (Δ −17.50)

Rank milestones (prism pure CSV, pos/orig): first_rank_le_10/5/1 = n/a (never reached); baseline: 23/25/25. Verdict: Regressive (KL increases strongly; no earlier rank milestones).

Top‑1 agreement at sampled depths shows prism deviates (e.g., L24 baseline top‑1 'simply' vs prism 'minecraft').

Cosine drift: prism cos_to_final is lower/negative at early/mid layers (e.g., L24: 0.357 → −0.376), indicating no earlier stabilization.

Copy flags: no baseline strict/soft flags; prism shows no flips of strict copy calls.

## 4. Qualitative patterns & anomalies

Negative control. The test prompt “Berlin is the capital of” produces a clean country completion with Berlin still present at small mass: > "Germany", 0.8966; "the", 0.0539; "both", 0.0044; "a", 0.0038; "Europe", 0.0031; … "Berlin", 0.00284  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:14–36]. No leakage of the city into the top‑5.

Important‑word trajectory. The script tracks IMPORTANT_WORDS = ["Germany", "Berlin", "capital", "Answer", "word", "simply"]  [001_layers_baseline/run.py:335]. In records.csv, Berlin first enters any top‑5 around L22, then strengthens at L24 and becomes dominant by L25: > “… Berlin, 0.00202” (L22, pos=14) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-records.csv:390]; > “… Berlin, 0.01081” (L24, pos=15) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-records.csv:425]; > “… Berlin, 0.08978” (L25, pos=14) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-records.csv:441]. “Germany” appears near the top‑5 throughout the same window (e.g., L24 rows) and “capital” co‑occurs strongly around L24–L25 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-records.csv:424–426].

Instruction cueing. Removing the filler (“no_filler”) slightly advances semantics (ΔL_sem = −1), suggesting a mild guidance‑style effect rather than substantive semantics change [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1083–1090]. The strict copy detectors (τ=0.95, δ=0.10) and soft detectors (τ_soft=0.5, k∈{1,2,3}) do not fire; no early copy‑reflex is observed.

Rotation vs amplification. Cosine aligns relatively early (≥0.2 by L11), while KL remains high until very late and the answer rank only reaches 1 at L25. This fits an “early direction, late calibration” pattern (cf. Tuned‑Lens, arXiv:2303.08112): direction aligns before probabilities calibrate. Final head calibration is clean (kl_to_final_bits=0, temp_est=1.0) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:899–919].

Lens sanity. Raw‑vs‑norm check is in sampled mode with high artifact risk and max_kl_norm_vs_raw_bits≈1.17; early semantics should be treated cautiously and rank milestones preferred [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1072–1074]. No “norm‑only semantics” flag (null) was observed.

Temperature robustness. At T=0.1: Berlin rank 1, p≈0.9996, entropy≈0.005 bits [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:670–676]. At T=2.0: Berlin remains top‑1 but much flatter (p≈0.036), entropy≈12.22 bits [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:737–743; 738].

Rest‑mass sanity. Rest_mass spikes to 0.91 at L25 (the semantic layer), reflecting limited top‑k coverage at that depth; this is a coverage measure, not lens fidelity [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:27]. Final rest_mass ≈0.230 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:34].

Prism sidecar. Prism is regressive for this model/setup: KL to final increases substantially at early/mid depths and rank milestones never improve vs baseline.

Checklist:
- RMS lens? ✓ (RMSNorm; rotary)  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:810; 816]
- LayerNorm bias removed? ✓ ("layernorm_bias_fix": "not_needed_rms_model")
- Entropy rise at unembed? n.a.
- FP32 un‑embed promoted? ✓ ("unembed_dtype": "torch.float32"; "mixed_precision_fix": "casting_to_fp32_before_unembed")
- Punctuation / markup anchoring? ✓ (early layers show punctuation/markup as top‑1)
- Copy‑reflex? ✗ (no strict/soft flags in L0–3)
- Grammatical filler anchoring? ✗ (L0–5 top‑1 not in {is, the, a, of})

## 5. Limitations & data quirks

- Rest_mass is top‑k coverage only; high values (e.g., 0.91 at L25) indicate limited coverage, not lens mis‑scale.
- KL is lens‑sensitive; rely on rank milestones (≤10/5/1) for cross‑model claims. Here final‑head calibration is clean (warn_high_last_layer_kl=false), so final probabilities are internally consistent.
- Raw‑vs‑norm check ran in sampled mode with high artifact risk; early “semantics” should be interpreted cautiously and within‑model only.
- Prism sidecar appears incompatible with this run’s calibration (strong KL regressions); treat Prism comparisons as diagnostic only.

## 6. Model fingerprint

Mistral‑7B‑v0.1: collapse at L 25; final entropy 3.61 bits; “Berlin” consolidates from L24 and is rank 1 by L25.

---
Produced by OpenAI GPT-5
