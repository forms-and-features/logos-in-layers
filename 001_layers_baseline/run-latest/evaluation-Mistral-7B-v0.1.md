# Evaluation Report: mistralai/Mistral-7B-v0.1

*Run executed on: 2025-09-29 23:35:16*
## 1. Overview

This evaluation analyzes Mistral‑7B‑v0.1 under a layer‑by‑layer norm‑lens probe on the prompt “Give the city name only, plain text. The capital of Germany is called simply”. It captures copy/filler behavior, semantic collapse, calibration vs the final head (KL in bits), cosine trajectory, and control/ablation effects.

## 2. Method sanity‑check

The run uses the norm lens on an RMSNorm model with rotary positions at L0. Diagnostics confirm this, and the context prompt ends with “called simply”. For example: “use_norm_lens”: true and “layer0_position_info”: “token_only_rotary_model” with the exact prompt string.

> "use_norm_lens": true  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:807]
> "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply"  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:817]

Norm/RMS details and unembed precision are recorded:

> "first_block_ln1_type": "RMSNorm", "final_ln_type": "RMSNorm"  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:810-811]
> "unembed_dtype": "torch.float32", "use_fp32_unembed": false  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:808-809]

Copy/collapse configuration and presence of required fields are confirmed:

> "copy_thresh": 0.95, "copy_window_k": 1, "copy_match_level": "id_subsequence"  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:928-931]
> "copy_soft_config": { "threshold": 0.5, "window_ks": [1,2,3], "extra_thresholds": [] }  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:880-888]
> "copy_flag_columns": ["copy_strict@0.95", "copy_strict@0.7", "copy_strict@0.8", "copy_strict@0.9", "copy_soft_k1@0.5", "copy_soft_k2@0.5", "copy_soft_k3@0.5"]  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1584-1591]

Gold‑token alignment is ID‑based and ok. Control prompt and summary exist. Ablation summary exists and both prompt variants are present in CSVs (pos/orig and pos/no_filler in the raw‑vs‑norm window sidecar).

> "gold_alignment": "ok"  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1059]
> "control_summary": { "first_control_margin_pos": 2, "max_control_margin": 0.6539100579732349 }  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1616-1619]
> "ablation_summary": { "L_sem_orig": 25, "L_sem_nf": 24, ... }  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1593-1600]

Strict copy (k=1, τ=0.95, δ=0.10) never fires in layers 0–3; soft copy (τ_soft=0.5 here) also never fires in early layers. There is no strict “copy_collapse = True” row anywhere in the pure CSV.

> pos,orig,0…3 rows show "copy_collapse,False" and all soft flags False  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:2-5]

KL/entropy units are bits. Rank/KL milestones are present in diagnostics for summary reporting.

> "first_kl_below_0.5": 32, "first_kl_below_1.0": 32, "first_rank_le_1": 25, "first_rank_le_5": 25, "first_rank_le_10": 23  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:931-935]

Final‑head calibration is clean (KL≈0, top‑1 agreement, temp=1.0). Prefer ranks is still advised by measurement guidance due to raw‑vs‑norm cautions.

> "last_layer_consistency": { "kl_to_final_bits": 0.0, "top1_agree": true, "p_top1_lens": 0.38216…, "p_top1_model": 0.38216…, "temp_est": 1.0, "kl_after_temp_bits": 0.0, "warn_high_last_layer_kl": false }  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1061-1079]
> "measurement_guidance": { "prefer_ranks": true, "suppress_abs_probs": true, "reasons": ["norm_only_semantics_window","high_lens_artifact_risk"] }  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2070-2077]

Raw‑vs‑Norm window sanity: L≈{25,32} (radius=4) shows a norm‑only semantic layer at L=32 and large norm vs raw divergence within the window.

> "raw_lens_window": { "center_layers": [25,32], "radius": 4, "norm_only_semantics_layers": [32], "max_kl_norm_vs_raw_bits_window": 8.5569, "mode": "window" }  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1033-1057]
> raw‑lens check summary: "lens_artifact_risk": "high", "max_kl_norm_vs_raw_bits": 1.1739  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1578-1582]

Threshold sweep sanity (strict): stability="none"; earliest L_copy_strict at τ∈{0.70,0.95} are null; norm_only_flags null.

> "copy_thresholds": { … "L_copy_strict": {"0.7": null, …, "0.95": null}, "stability": "none" }  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:980-1006]

Copy‑collapse flag check: no row has copy_collapse=True → ✓ no spurious fires.
Soft‑copy flags: no early hits for k∈{1,2,3} → strict null, soft null.

## 3. Quantitative findings

Positive prompt, original variant only (pos, orig). Table shows per‑layer entropy (bits) and generic top‑1 token (not necessarily the answer before semantic collapse). The semantic layer (first is_answer=True) is bolded.

| Layer | Entropy (bits) | Top‑1 token |
|---:|---:|:--|
| 0 | 14.961 | ‘dabei’ |
| 1 | 14.929 | ‘biologie’ |
| 2 | 14.825 | ‘,\r’ |
| 3 | 14.877 | ‘[…]’ |
| 4 | 14.854 | ‘[…]’ |
| 5 | 14.827 | ‘[…]’ |
| 6 | 14.838 | ‘[…]’ |
| 7 | 14.805 | ‘[…]’ |
| 8 | 14.821 | ‘[…]’ |
| 9 | 14.776 | ‘[…]’ |
| 10 | 14.782 | ‘[…]’ |
| 11 | 14.736 | ‘[…]’ |
| 12 | 14.642 | ‘[…]’ |
| 13 | 14.726 | ‘[…]’ |
| 14 | 14.653 | ‘[…]’ |
| 15 | 14.450 | ‘[…]’ |
| 16 | 14.600 | ‘[…]’ |
| 17 | 14.628 | ‘[…]’ |
| 18 | 14.520 | ‘[…]’ |
| 19 | 14.510 | ‘[…]’ |
| 20 | 14.424 | ‘simply’ |
| 21 | 14.347 | ‘simply’ |
| 22 | 14.387 | ‘“’ |
| 23 | 14.395 | ‘simply’ |
| 24 | 14.212 | ‘simply’ |
| 25 | 13.599 | ‘Berlin’ |
| 26 | 13.541 | ‘Berlin’ |
| 27 | 13.296 | ‘Berlin’ |
| 28 | 13.296 | ‘Berlin’ |
| 29 | 11.427 | ‘"’ |
| 30 | 10.797 | ‘“’ |
| 31 | 10.994 | ‘"’ |
| 32 | 3.611 | ‘Berlin’ |

Bold semantic layer (ID‑level gold token “Berlin”): L_semantic = 25.

Control margin (control_summary): first_control_margin_pos = 2; max_control_margin = 0.6539.  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1616-1619]

Ablation (no‑filler):
- L_copy_orig = null, L_sem_orig = 25  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1593-1596]
- L_copy_nf = null, L_sem_nf = 24  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1596-1597]
- ΔL_copy = n.a. (strict/soft null), ΔL_sem = −1  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1598-1599]

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (no strict/soft copy). Soft ΔHₖ = n.a. for k∈{1,2,3}.

Confidence milestones (pure CSV):
- p_top1 > 0.30 at layer 32; p_top1 > 0.60 at none; final‑layer p_top1 = 0.3822  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:34]

Rank milestones (diagnostics): rank ≤10 at L=23; ≤5 at L=25; ≤1 at L=25  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:931-935]

KL milestones (diagnostics): first_kl_below_1.0 at L=32; first_kl_below_0.5 at L=32; final‑row KL≈0 with last‑layer head calibration (see below).  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:931-932]

Cosine milestones (JSON): first cos_to_final ≥0.2 at L=11, ≥0.4 at L=25, ≥0.6 at L=26; final cos_to_final ≈ 1.0 (pure CSV L32).  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1018-1023; 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:34]

Depth fractions: L_semantic_frac = 0.781; first_rank_le_5_frac = 0.781.  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1025-1027]

Copy robustness (strict sweep): stability = "none"; earliest L_copy_strict at τ=0.70 and τ=0.95 are null; norm_only_flags null.  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:980-1006]

Prism sidecar analysis: present but regressive vs baseline.
- Early‑depth stability (KL): Prism KL is much worse at percentiles (p25=23.06, p50=27.87, p75=26.55) vs baseline (10.25/10.33/9.05). Δ is −12.81/−17.54/−17.50 bits.  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:862-871]
- Rank milestones: Prism first_rank_le_{10,5,1} = null/null/null (baseline 23/25/25).  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:840-848]
- Copy flags: no strict copy flips observed (no copy_collapse in Prism pure CSV).
- Verdict: Regressive.

Tuned‑Lens (if relevant): summary shows earlier KL thresholds but later first_rank_le_1; treat as sidecar only.
- ΔKL medians (baseline − tuned): +4.03 (p25), +3.75 (p50), +7.08 (p75).  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1688-1706]
- first_kl_le_1.0: baseline 32 vs tuned 26 (Δ=−6)  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1717-1723]
- L_surface_to_meaning_norm=32 (answer_mass=0.3822, echo_mass=0.0610).  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1007-1015]

## 4. Qualitative patterns & anomalies

Negative control. For “Berlin is the capital of”, the model’s top‑5 include Germany as top‑1 and “Berlin” still appears (semantic leakage):

> “… "Germany", 0.8966 … "Berlin", 0.00284 …”  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:14-36]

Important‑word trajectory (records CSV; IMPORTANT_WORDS=["Germany","Berlin","capital","Answer","word","simply"]). Around the NEXT position (pos=16, token “simply”), Berlin first appears in the top‑20 by L22 and strengthens through L24; Berlin is already salient at adjacent positions (pos=14/15) before full semantic collapse.

> L22 pos=16: “… Berlin, 0.001114 … capital … Germany …”  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-records.csv:392]
> L23 pos=16: “… simply … Berlin, 0.001216 …”  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-records.csv:409]
> L24 pos=15: “called … Berlin, 0.010807 …”  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-records.csv:425]
> L24 pos=16: “… simply … Berlin, 0.003887 …”  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-records.csv:426]
> L22 pos=14/15: “is/called … Berlin in top‑k …”  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-records.csv:390-391]

Instructional variants. Removing the adverb (“no‑filler”) slightly advances semantics (ΔL_sem = −1) per ablation_summary; collapse‑layer shift for “no one‑word instruction” prompts is not layer‑tracked (test prompts are single‑step snapshots), but qualitative top‑k show the model often supplies quoting/punctuation or “called/known” without the one‑word nudge.

> “In Germany the capital city is simply” → top‑1 “called”, Berlin present but not dominant  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:200-234]

Rest‑mass sanity. Rest_mass falls sharply after semantic onset: it is highest just after L_semantic (L25: 0.911) and decreases to 0.230 by the final layer (L32).  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:27,34]

Rotation vs amplification. Cos_to_final rises early (≥0.4 at L25; ≥0.6 at L26), while KL to final remains high at L25 (8.017 bits) and only nears zero at L32. This is consistent with “early direction, late calibration”: direction aligns by mid‑stack, calibration (probability mass) consolidates late.

> L25 (semantic onset): p_answer=0.0335, answer_rank=1, cos_to_final=0.4245, kl_to_final_bits=8.0175  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:27]
> L32 (final): p_answer=0.3822, cos_to_final≈1.0, kl_to_final_bits=0.0  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:34]

Head calibration (final layer). Last‑layer consistency shows KL≈0 with top‑1 agreement and temp_est=1.0; no warning flag.

> “… "kl_to_final_bits": 0.0, "top1_agree": true, "temp_est": 1.0, … "warn_high_last_layer_kl": false”  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1061-1079]

Lens sanity. Raw‑lens sampling indicates high artifact risk overall and a norm‑only semantic layer within the local window (L32). Prefer rank‑based milestones and within‑model comparisons for early‑depth interpretations.

> raw_lens_check.summary: { "max_kl_norm_vs_raw_bits": 1.174, "lens_artifact_risk": "high" }  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1578-1582]
> window: norm_only_semantics_layers=[32], max_kl_norm_vs_raw_bits_window=8.5569  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1053-1057]

Temperature robustness. At T=2.0, entropy rises (12.22 bits) and “Berlin” remains the top token with low mass (≈0.036), as punctuation and quotation marks gain probability.

> “temperature": 2.0 … "entropy": 12.2199 … "Berlin", 0.03598 …”  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:737-743]

Checklist:
- RMS lens? ✓  [RMSNorm; norm lens on]  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:807,810-811]
- LayerNorm bias removed? ✓ (not needed on RMS)  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:812]
- Entropy rise at unembed? ✓ (final entropy 3.611 bits; calibrated)  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1429]
- FP32 un‑embed promoted? ✓ (analysis dtype fp32)  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:809]
- Punctuation / markup anchoring? ✓ (quotes/“simply” mid‑stack)  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:22-24,30-33]
- Copy‑reflex? ✗ (no strict or soft hits in L0–3)  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:2-5]
- Grammatical filler anchoring? ✓ (simply/quotes dominate L20–24)  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:20-24]

## 5. Limitations & data quirks

- Measurement guidance flags “norm_only_semantics_window” and “high_lens_artifact_risk”; prefer ranks and within‑model comparisons of probabilities.  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2071-2077]
- Raw‑vs‑norm window shows norm‑only semantics at L32 and large norm↔raw KL in the window (8.56 bits), cautioning that apparent early semantics can be lens‑induced.  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1033-1057]
- Rest_mass > 0.3 immediately after L_semantic (L25) reflects broad distribution under the lens; it declines to 0.230 by the final layer (not indicative of final mis‑scale).  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:27,34]
- KL is lens‑sensitive; final‑head calibration is good here (KL≈0), so use rank milestones for cross‑model claims and KL trends qualitatively.
- Surface‑mass relies on tokenizer vocab; prefer within‑model trends and rank milestones rather than cross‑model mass comparisons.

## 6. Model fingerprint

“Mistral‑7B‑v0.1: semantic collapse at L 25; final entropy 3.611 bits; cosine ≥0.4 by L 25 and ≈1.0 at L 32; control margin positive early (pos 2), with no early copy‑reflex.”

---
Produced by OpenAI GPT-5 

