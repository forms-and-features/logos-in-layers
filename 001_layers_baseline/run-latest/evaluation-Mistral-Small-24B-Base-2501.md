**Overview**

This evaluation analyzes mistralai/Mistral-Small-24B-Base-2501 (40-layer pre-norm) under a logit-lens probe that decodes layer-by-layer next-token distributions. The run uses the normalized RMS stream with FP32 unembedding, and confirms semantic collapse on the gold token at L 33 (confirmed by raw lens within a ±2 window).

> "model": "mistralai/Mistral-Small-24B-Base-2501"  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:805]


**Method Sanity‑Check**

The intended norm lens and positional encodings are active: use_norm_lens is true and layer0 reports token-only rotary; RMSNorm is detected at first and final LNs, and logits are cast to FP32 pre-unembed. The context prompt ends exactly in “called simply” (no trailing space).

> "use_norm_lens": true  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:807]
>
> "layer0_position_info": "token_only_rotary_model"  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:816]

> "first_block_ln1_type": "RMSNorm"; "final_ln_type": "RMSNorm"  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:810–811]
>
> "unembed_dtype": "torch.float32"; "mixed_precision_fix": "casting_to_fp32_before_unembed"  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:809,815]

> "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply"  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:817]

Copy detection and thresholds are present and consistent across JSON and CSV: strict ID‑level contiguous subsequence at τ=0.95 with k=1 and soft windowed copy at τ_soft=0.5 with k∈{1,2,3}. All strict τ∈{0.70,0.80,0.90,0.95} and soft flags appear in the CSV header.

> "copy_thresh": 0.95; "copy_window_k": 1; "copy_match_level": "id_subsequence"  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:936–938]
>
> "copy_soft_config": { "threshold": 0.5, "window_ks": [1,2,3] }  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:880–885]

> "copy_flag_columns": ["copy_strict@0.95","copy_strict@0.7","copy_strict@0.8","copy_strict@0.9","copy_soft_k1@0.5","copy_soft_k2@0.5","copy_soft_k3@0.5"]  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1627–1635]

Gold‑token alignment is ID‑based and successful.

> "gold_alignment": "ok"  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1081]

Negative control is present and summarized.

> "first_control_margin_pos": 1; "max_control_margin": 0.4679627253462968  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1660–1661]

Ablation is present and yields L_sem_nf=31 vs L_sem_orig=33 (ΔL_sem = −2). Both prompt_variant rows exist in the CSV (orig and no_filler).

> "ablation_summary": { "L_sem_orig": 33, "L_sem_nf": 31, "delta_L_sem": -2 }  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1636–1642]
>
> pos,no_filler,... (CSV rows exist)  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:43]

Summary indices (bits for KL/entropy): first_kl_below_1.0 = 40; first_kl_below_0.5 = 40; rank milestones first_rank_le_{10,5,1} = {30,30,33}. Final‑head calibration is good: final CSV row has kl_to_final_bits = 0.0, with top‑1 agreement and temp_est = 1.0.

> "first_kl_below_0.5": 40; "first_kl_below_1.0": 40; "first_rank_le_1": 33; "first_rank_le_5": 30; "first_rank_le_10": 30  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:939–943]
>
> "kl_to_final_bits": 0.0; "top1_agree": true; "temp_est": 1.0; "kl_after_temp_bits": 0.0  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1083–1090]

Measurement guidance prefers tuned for reporting and to use confirmed semantics; ranks/probabilities are acceptable within‑model.

> "preferred_lens_for_reporting": "tuned"; "use_confirmed_semantics": true  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:2146–2149]

Raw‑vs‑Norm windowed check (center_layers=[30,33,40], radius=4) shows no norm‑only semantic layers but a sizable local norm‑vs‑raw divergence; full scan rates lens‑artifact risk as low.

> "center_layers": [30,33,40]; "radius": 4; "max_kl_norm_vs_raw_bits_window": 5.980131892728737  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1042–1066]
>
> "pct_layers_kl_ge_1.0": 0.02439; "n_norm_only_semantics_layers": 0; "max_kl_norm_vs_raw_bits": 5.9801; "tier": "low"  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1070–1078]

Threshold sweep sanity is present with stability="none" and null strict L_copy at τ∈{0.70,0.95}; this matches CSV copy flags (all false in early layers).

> "copy_thresholds"... "stability": "none"; "L_copy_strict": {"0.7": null, ..., "0.95": null}  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:988–1013]

Strict copy‑collapse never fires (k=1, τ=0.95); soft k∈{1,2,3} at τ_soft=0.5 also never fires in L0–L3 (and globally). Thus, no copy reflex in early layers. Prism sidecar is present/compatible but lacks rank milestones (nulls in diagnostics).

> copy_collapse=False in early rows; no "copy_soft_k1@0.5,True" hits  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:2–7]
>
> "prism_summary": { "present": true, "compatible": true, ... "rank_milestones": { "prism": { "le_10": null, ... }}}  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:825–876]

Lens selection and milestones: confirmed semantics is present with source=raw and L_semantic_confirmed=33; baseline L_surface_to_meaning_norm=40 with answer_mass≈0.455 and echo_mass≈0.0042. Geometry shows L_geom_norm=31; coverage L_topk_decay_norm=0. Norm‑temperature diagnostics (per‑layer τ) are present with KL_temp snapshots at {10,20,30}.

> "L_semantic_confirmed": 33; "confirmed_source": "raw"  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1559–1566]
>
> "L_surface_to_meaning_norm": 40; "answer_mass_at_L_surface_norm": 0.45546; "echo_mass_at_L_surface_norm": 0.00424  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1015–1017]

> "L_geom_norm": 31; "L_topk_decay_norm": 0  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1018,1021]
>
> "kl_to_final_bits_norm_temp@50%": { "layer": 20, "value": 10.6917 }  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1103–1108]


**Quantitative Findings**

Table (pos, orig; entropy in bits; generic top‑1 token quoted; bolded row = confirmed semantic collapse):

| Layer | Entropy | Top‑1 |
|---|---|---|
| L  0 | 16.999 bits |  Forbes |
| L  1 | 16.975 bits | 随着时间的 |
| L  2 | 16.944 bits | 随着时间的 |
| L  3 | 16.812 bits | 随着时间的 |
| L  4 | 16.868 bits |  quelcon |
| L  5 | 16.903 bits | народ |
| L  6 | 16.909 bits | народ |
| L  7 | 16.898 bits | народ |
| L  8 | 16.896 bits |  quelcon |
| L  9 | 16.885 bits |  simply |
| L 10 | 16.836 bits |  hétérogènes |
| L 11 | 16.842 bits | 从那以后 |
| L 12 | 16.840 bits |  simply |
| L 13 | 16.871 bits |  simply |
| L 14 | 16.815 bits | стен |
| L 15 | 16.816 bits | luš |
| L 16 | 16.830 bits | luš |
| L 17 | 16.775 bits | luš |
| L 18 | 16.761 bits | luš |
| L 19 | 16.775 bits | luš |
| L 20 | 16.742 bits | luš |
| L 21 | 16.775 bits |  simply |
| L 22 | 16.764 bits |  simply |
| L 23 | 16.769 bits | -на |
| L 24 | 16.758 bits | -на |
| L 25 | 16.747 bits |  «** |
| L 26 | 16.769 bits |  «** |
| L 27 | 16.776 bits |  «** |
| L 28 | 16.741 bits |  «** |
| L 29 | 16.760 bits |  «** |
| L 30 | 16.743 bits | -на |
| L 31 | 16.793 bits | -на |
| L 32 | 16.789 bits | -на |
| **L 33** | **16.774 bits** | ** Berlin** |
| L 34 | 16.761 bits |  Berlin |
| L 35 | 16.734 bits |  Berlin |
| L 36 | 16.699 bits |  Berlin |
| L 37 | 16.513 bits | " """ |
| L 38 | 15.869 bits | " """ |
| L 39 | 16.005 bits |  Berlin |
| L 40 | 3.181 bits |  Berlin |

Control margin (JSON control_summary): first_control_margin_pos = 1; max_control_margin = 0.468.  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1660–1661]

Ablation (no‑filler): L_copy_orig = null; L_sem_orig = 33; L_copy_nf = null; L_sem_nf = 31; ΔL_copy = null; ΔL_sem = −2. Large negative ΔL_sem indicates slightly earlier collapse without the “simply” filler, not later.  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1636–1642]

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (strict and soft L_copy null). Soft ΔH_k (k∈{1,2,3}) = n.a. (no soft copy). Confidence milestones: p_top1 > 0.30 at L 40; p_top1 > 0.60 not reached; final p_top1 = 0.4555 (Berlin).  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:42]

Rank milestones (preferred lens tuned first; baseline in parentheses):
- rank ≤ 10 at layer 34 (30)
- rank ≤ 5 at layer 35 (30)
- rank ≤ 1 at layer 39 (33)

> tuned vs baseline deltas: {le_10: 4, le_5: 5, le_1: 6}  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:2077–2116]

KL milestones: first_kl_below_1.0 = 40; first_kl_below_0.5 = 40; KL decreases toward 0 at final, matching last‑layer consistency (kl_to_final_bits=0.0).  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:939–943,1083]

Cosine milestones (norm lens): ge_0.2 at L 35; ge_0.4 at L 40; ge_0.6 at L 40; final cos_to_final ≈ 1.00.  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1026–1030]

Depth fractions: L_semantic_frac = 0.825.  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1033–1035]

Copy robustness (threshold sweep): stability = "none"; earliest L_copy_strict at τ=0.70 and τ=0.95 are null; norm_only_flags all null.  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:988–1013]

Prism Sidecar Analysis. Prism is present and compatible but regressive on calibration: KL at percentiles is higher than baseline (negative deltas), and rank milestones are null.
- Early‑depth stability: baseline vs prism KL deltas {p25: −1.93, p50: −5.98, p75: −4.97} (negative = prism worse).  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:856–870]
- Rank milestones: prism le_{10,5,1} = null; skip rank analysis.  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:850–876]
- Copy flags: no spurious flips (no copy flags set in prism CSV early layers).
- Verdict: Regressive (higher KL, no helpful rank shifts).


**Qualitative Patterns & Anomalies**

Negative control. The test prompt “Berlin is the capital of” yields a strong “ Germany” with p≈0.802; Berlin also appears (semantic leakage) in the top‑10 at low mass.

> " Germany", 0.8021 … " Berlin", 0.00481  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:12–31]

Important‑word trajectory. The expected answer appears in top‑k progressively: at L 30 Berlin is rank‑3 (p≈1.69e‑4), L 31 rank‑2 (p≈2.64e‑4), and becomes rank‑1 by L 33 (p≈4.25e‑4), then strengthens (e.g., L 36 p≈0.00263) before final consolidation (L 40 p≈0.4555).

> " Berlin", 0.000169 … (L 30, rank 3)  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:32]
>
> " Berlin", 0.000263 … (L 31, rank 2)  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:33]
>
> " Berlin", 0.000425 … (L 33, rank 1)  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:35]

Punctuation/mark‑up anchoring emerges late: quotes become dominant at L 37–38 before returning to the answer in L 39–40.

> top‑1 '"' at L 37 (p≈0.00405), and L 38 (p≈0.02536)  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:39–40]

Collapse‑layer shift without “one‑word” instruction. The ablation shows earlier semantics without filler (L_sem_nf=31 vs 33), indicating minor guidance‑style sensitivity rather than filler‑copy.  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1636–1642]

Rest‑mass sanity. Rest_mass remains high through mid‑depths and drops sharply at the final layer (0.18), with the maximum after L_semantic ≈ 0.999 at L 33.  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:33,42]

Rotation vs amplification. Cosine to final rises by mid‑depth (ge_0.2 at L 35) while KL remains high until very late (first KL<1.0 at L 40), i.e., early direction, late calibration; final head calibration is clean (KL≈0).  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1026–1030,939–943,1083]

Head calibration. Last‑layer agreement is perfect within this run (top‑1 agree, temp_est=1.0, kl_after_temp_bits=0.0), so probabilities can be compared within‑model.  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1083–1090]

Lens sanity. Full raw‑vs‑norm summary reports tier=low, no norm‑only semantic layers, but max_kl_norm_vs_raw_bits≈5.98 indicates that apparent “early semantics” should still be judged via rank milestones (which we use).  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1066,1070–1078]

Temperature robustness. At T=0.1 the model is extremely confident (Berlin p≈0.9995; entropy≈0.006); at T=2.0 entropy rises to ≈14.36 bits and Berlin p≈0.030, as expected.  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:670–742]

Checklist:
- RMS lens? ✓ (RMSNorm detected)
- LayerNorm bias removed? ✓ (not needed for RMS)  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:812]
- Entropy rise at unembed? ✗ (no spike; final entropy ≈ teacher)
- FP32 un‑embed promoted? ✓ (unembed_dtype=float32; cast before unembed)  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:809,815]
- Punctuation / markup anchoring? ✓ late (L 37–38 quotes)
- Copy‑reflex? ✗ (no strict or soft copy hits L0–L3)
- Grammatical filler anchoring? ✗ (top‑1 in L0–5 are not {is,the,a,of})
- Preferred lens honored in milestones? ✓ (tuned first, baseline in parentheses)
- Confirmed semantics reported? ✓ (L_semantic_confirmed=33, source=raw)
- Full dual‑lens metrics cited? ✓ (pct_layers_kl≥{0.5,1.0}, norm‑only=0, tier=low)
- Tuned‑lens attribution done? ✓ (ΔKL_tuned, ΔKL_temp, ΔKL_rot at ~25/50/75%)  [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:2122–2136]


**Limitations & Data Quirks**

Rest_mass stays >0.3 after L_semantic until the final layer (e.g., ≈0.999 at L 33), reflecting top‑k coverage rather than lens fidelity; interpret probabilities comparatively within‑model only. Final KL≈0 suggests good head calibration here, but KL trends are lens‑sensitive; rely on rank milestones for cross‑model claims. Raw‑vs‑norm full check rates artifact tier low with no norm‑only semantics; still, the single‑model norm‑vs‑raw divergence (max≈5.98 bits) warrants caution and preferring confirmed semantics and rank thresholds when interpreting “early meaning”. Tokenization differences can confound absolute surface‑mass; prefer within‑model trends and ranks.


**Model Fingerprint**

Mistral‑Small‑24B‑Base‑2501: collapse at L 33 (confirmed raw); final entropy ≈3.18 bits; "Berlin" becomes top‑1 by mid‑late stack and consolidates at L 40 (p≈0.455).

---
Produced by OpenAI GPT-5 

