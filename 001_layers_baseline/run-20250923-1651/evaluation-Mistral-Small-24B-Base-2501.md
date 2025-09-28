# Evaluation Report: mistralai/Mistral-Small-24B-Base-2501

## 1. Overview
This run evaluates mistralai/Mistral-Small-24B-Base-2501 (pre-norm, 40 layers) on a one-token capital question with a norm-lens logit-lens pipeline. Start time: “Experiment started: 2025-09-23 16:51:10”  [001_layers_baseline/run-latest/timestamp-20250923-1651:1].
The probe records layer-wise entropy, KL to final logits, copy/filler collapse flags, answer rank/probability, cosine-to-final, and a negative control.

## 2. Method sanity-check
Diagnostics confirm the intended setup: RMSNorm lens with normalized residuals and rotary positions only at layer 0, and FP32 unembed math. Examples:
> "use_norm_lens": true  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:807]
> "layer0_position_info": "token_only_rotary_model"  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:816]

Context prompt ends with “called simply” (no trailing space):
> "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply"  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:817]

Copy/soft-copy configuration and flags are present and aligned with CSV labels:
> "copy_thresh": 0.95, "copy_window_k": 1, "copy_match_level": "id_subsequence"  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:846–848]
> "copy_soft_config": { "threshold": 0.5, "window_ks": [1,2,3], "extra_thresholds": [] }  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:833–841]
> "copy_flag_columns": ["copy_strict@0.95","copy_soft_k1@0.5","copy_soft_k2@0.5","copy_soft_k3@0.5"]  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1077–1081]

Summary indices (bits/ranks):
> "first_kl_below_0.5": 40, "first_kl_below_1.0": 40  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:849–850]
> "first_rank_le_1": 33, "first_rank_le_5": 30, "first_rank_le_10": 30  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:851–853]

Last-layer head calibration exists and is perfect-aligned: 
> "kl_to_final_bits": 0.0, "top1_agree": true, "p_top1_lens": 0.4554620981, "p_top1_model": 0.4554620981, "temp_est": 1.0, "warn_high_last_layer_kl": false  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:900–907,917]

Lens sanity (raw vs norm):
> "mode": "sample"; "max_kl_norm_vs_raw_bits": 0.1792677; "lens_artifact_risk": "low"; "first_norm_only_semantic_layer": null  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1016,1072–1074]

Gold alignment is ID-based and ok for both prompts:
> "gold_alignment": "ok"  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:898]
> control gold_alignment: "ok"  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1104–1105]
Negative control prompt and summary present:
> "context_prompt": "Give the city name only, plain text. The capital of France is called simply"  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1092]
> "first_control_margin_pos": 1, "max_control_margin": 0.4679627253462968  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1106–1109]
Ablation summary present:
> "L_sem_orig": 33, "L_sem_nf": 31, "delta_L_sem": -2  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1085,1087,1089]; both variants appear in CSV (e.g., pos,no_filler rows)  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:43].

Copy-collapse flags: no strict or soft copy fired anywhere in L0–L3 (or later) for pos/orig; earliest copy_collapse=True row does not exist in this run (✓ rule did not spuriously fire).

Units: entropy and KL columns are in bits (column name "kl_to_final_bits" in CSV; entropy in bits by construction).

## 3. Quantitative findings
Table below is filtered to prompt_id=pos, prompt_variant=orig (pure-next-token CSV). Bold marks the first semantic layer (is_answer=True). Gold answer string: “Berlin”  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1111–1115].

| Layer | Entropy (bits) | Top‑1 token |
|---|---:|---|
| L 0 | 16.999 | Forbes |
| L 1 | 16.975 | 随着时间的 |
| L 2 | 16.944 | 随着时间的 |
| L 3 | 16.812 | 随着时间的 |
| L 4 | 16.868 | quelcon |
| L 5 | 16.903 | народ |
| L 6 | 16.909 | народ |
| L 7 | 16.898 | народ |
| L 8 | 16.896 | quelcon |
| L 9 | 16.885 | simply |
| L 10 | 16.836 | hétérogènes |
| L 11 | 16.842 | 从那以后 |
| L 12 | 16.840 | simply |
| L 13 | 16.871 | simply |
| L 14 | 16.815 | стен |
| L 15 | 16.816 | luš |
| L 16 | 16.830 | luš |
| L 17 | 16.775 | luš |
| L 18 | 16.761 | luš |
| L 19 | 16.775 | luš |
| L 20 | 16.742 | luš |
| L 21 | 16.775 | simply |
| L 22 | 16.764 | simply |
| L 23 | 16.769 | -на |
| L 24 | 16.758 | -на |
| L 25 | 16.747 | «** |
| L 26 | 16.769 | «** |
| L 27 | 16.776 | «** |
| L 28 | 16.741 | «** |
| L 29 | 16.760 | «** |
| L 30 | 16.743 | -на |
| L 31 | 16.793 | -на |
| L 32 | 16.789 | -на |
| **L 33** | **16.774** | **Berlin** |
| L 34 | 16.761 | Berlin |
| L 35 | 16.734 | Berlin |
| L 36 | 16.699 | Berlin |
| L 37 | 16.513 | " |
| L 38 | 15.869 | " |
| L 39 | 16.005 | Berlin |
| L 40 | 3.181 | Berlin |

Control margin (from JSON control_summary): first_control_margin_pos=1; max_control_margin=0.4679627253462968  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1106–1109].

Ablation (no‑filler): L_copy_orig=null; L_sem_orig=33; L_copy_nf=null; L_sem_nf=31; ΔL_copy=n/a; ΔL_sem=−2  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1083–1090]. Interpretation: removing “simply” advances semantics by 2 layers (≈5% of depth), consistent with weaker stylistic anchoring.

ΔH(bits)=entropy(L_copy)−entropy(L_semantic)=n/a (L_copy null). Soft ΔHk(bits): n/a for k∈{1,2,3} (no soft-copy layer).

Confidence milestones (pure CSV, generic top‑1): p_top1>0.30 at layer 40; p_top1>0.60 not reached; final-layer p_top1=0.4554620981  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:42].

Rank milestones (diagnostics): rank≤10 at L30; rank≤5 at L30; rank≤1 at L33.
> "first_rank_le_1": 33  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:851]
> "first_rank_le_5": 30, "first_rank_le_10": 30  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:852–853]

KL milestones (diagnostics): first_kl_below_1.0 at L40; first_kl_below_0.5 at L40.
> "first_kl_below_1.0": 40, "first_kl_below_0.5": 40  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:849–850]
KL decreases toward depth and equals 0.0 at final (perfect head alignment):
> "kl_to_final_bits": 0.0  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:900]

Cosine milestones (pure CSV): cos_to_final≥0.2 at L35; ≥0.4 at L40; ≥0.6 at L40; final cos_to_final≈1.0  [pure CSV rows: 37–42].

Prism Sidecar Analysis
- Presence: compatible=true (k=512; layers=[embed,9,19,29])  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:823,825–829].
- Early-depth stability (KL in bits; baseline→Prism): L0 10.522→10.679; L10 10.913→12.842; L20 10.844→16.821; L30 10.565→15.531. No early KL drop.
- Rank milestones (Prism pure CSV): none achieved (no rank≤10/5/1) vs baseline {10:30,5:30,1:33}.
- Top‑1 agreement: at L∈{0,10,20,30}, Prism never matches the final top‑1 (“Berlin”) where baseline disagreed.
- Cosine drift: baseline cos_to_final at {0,10,20,30} = {0.134, −0.107, −0.022, 0.099}; Prism = {−0.070, 0.079, −0.110, −0.021}; Prism shows no earlier stabilization; final Prism cos is ≈0.090 vs baseline ≈1.0.
- Copy flags: no flips (no strict/soft copy True in either file).
- Verdict: Regressive — KL increases at early/mid depths and rank milestones do not improve under Prism.

## 4. Qualitative patterns & anomalies
Negative control (“Berlin is the capital of”): top‑5 are “ Germany” 0.8021, “ which” 0.0670, “ the” 0.0448, “ _” 0.0124, “ what” 0.0109; Berlin still appears later: “ Berlin”, 0.00481 (rank 7) — semantic leakage  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:10–34].

Important‑word trajectory: the filler word “simply” is often generic top‑1 at mid‑stack (e.g., L9 and L12–L13) before semantics: > “…, simply,5.98e-05 …”  [pure CSV row 11] and > “…, simply,0.0001521 …”  [row 14]. “Berlin” first enters top‑5 by L30 (top‑3; p≈1.69e‑4)  [row 32], becomes top‑1 at L33  [row 35], and remains top‑1 through depth (except punctuation spikes at L37–38)  [rows 37–41].

Rest‑mass sanity: rest_mass falls markedly near the end; maximum after L_semantic is ≈0.9988 at L33, reaching 0.181 at final  [pure CSV rows 35 and 42].

Rotation vs amplification: direction aligns before calibration. At L35, cos_to_final≈0.225 while KL_to_final≈10.14 bits and p_top1 is still low  [pure CSV row 37]. By L40, KL=0.0 and cos_to_final≈1.0 with p_top1=0.455  [pure CSV row 42; 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:900].

Head calibration (final layer): perfect agreement with the model head; temp_est=1.0; kl_after_temp_bits=0.0; warn_high_last_layer_kl=false  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:900–907,917].

Lens sanity: raw‑vs‑norm summary reports lens_artifact_risk=low; max_kl_norm_vs_raw_bits=0.1793; first_norm_only_semantic_layer=null  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1072–1074]. Treat early “semantics” cautiously but risk appears low.

Temperature robustness: at T=0.1, Berlin rank 1 (p≈0.9995; entropy≈0.0061 bits)  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:670–678]; at T=2.0, Berlin rank 1 (p≈0.0299; entropy≈14.36 bits)  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:737–744].

Stylistic ablation: semantics happen earlier without “simply” (L_sem_nf=31 vs 33; ΔL_sem=−2)  [..json:1083–1090], consistent with a mild guidance‑style anchoring effect rather than content semantics.

Checklist
- RMS lens? ✓ (RMSNorm detected)  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:810–811]
- LayerNorm bias removed? ✓ (not needed on RMS)  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:812]
- Entropy rise at unembed? ✗ (entropy drops to 3.18 bits at L40)  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:922]
- FP32 un‑embed promoted? ✓ (unembed_dtype="torch.float32"; auto‑cast; use_fp32_unembed=false)  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:808–809,815]
- Punctuation / markup anchoring? ✓ (quotes/colon dominate at L37–38)  [pure CSV rows 39–40]
- Copy‑reflex? ✗ (no strict/soft copy flags fired in L0–3)  [pure CSV rows 2–6]
- Grammatical filler anchoring? ✗ (no early top‑1 in {is,the,a,of})

## 5. Limitations & data quirks
- Rest_mass remains >0.3 for many layers after L_semantic, only dropping near the end (e.g., 0.9988 at L33 → 0.181 at L40); rest_mass is top‑k coverage only and not a lens‑fidelity metric.
- KL is lens‑sensitive; here final‑head calibration is perfect (KL=0.0), so within‑model probabilities are reliable, but prefer rank milestones for cross‑model claims.
- Raw‑vs‑norm check ran in sample mode; treat it as a sampled sanity rather than exhaustive.
- Prism sidecar appears regressive for this model/config; do not use Prism‑adjusted trends for conclusions.

## 6. Model fingerprint
Mistral‑Small‑24B‑Base‑2501: collapse at L 33; final entropy 3.18 bits; “Berlin” first enters top‑5 at L 30 and stabilizes by L 33.

---
Produced by OpenAI GPT-5 
