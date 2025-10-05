# Evaluation Report: google/gemma-2-9b

*Run executed on: 2025-10-04 18:04:23*
**1. Overview**

This evaluates google/gemma-2-9b (42 layers) using the norm lens on CPU, run on 2025-10-04. The probe tracks copy-reflex vs. semantic collapse, KL-to-final, cosine-to-final, and surface mass at NEXT for the positive prompt and a France→Paris control.

**2. Method Sanity-Check**

Diagnostics confirm the intended prompt and lens: context ends with “called simply” and norm-lens is active. For reference: 001_layers_baseline/run-latest/output-gemma-2-9b.json:4 and :826 show
> "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply"  [001_layers_baseline/run-latest/output-gemma-2-9b.json:4]
> "use_norm_lens": true  [001_layers_baseline/run-latest/output-gemma-2-9b.json:819]

Copy detectors and thresholds are present. JSON reports: "copy_thresh": 0.95, "copy_window_k": 1, "copy_match_level": "id_subsequence"; strict sweep τ ∈ {0.70,0.80,0.90,0.95} with earliest L_copy_strict = 0 for all τ and stability "mixed"; soft-copy config threshold 0.5 with window_ks {1,2,3} and earliest L_copy_soft[k1] = 0. Flags mirror these in CSVs via copy_flag_columns. Quotes:
- L_copy/L_semantic and strict/soft: "L_copy": 0, "L_semantic": 42; "L_copy_soft": {"1": 0} [001_layers_baseline/run-latest/output-gemma-2-9b.json:5634–5666]
- Threshold sweep: "L_copy_strict": {"0.7": 0, ..., "0.95": 0}, "stability": "mixed" [001_layers_baseline/run-latest/output-gemma-2-9b.json:5694–5719]
- copy_flag_columns present [001_layers_baseline/run-latest/output-gemma-2-9b.json:6973–6980]

Normalizer provenance: pre-norm model using next ln1 with epsilon inside sqrt and scaled γ; per-layer ln_source switches to ln_final at the last layer. Quotes:
- strategy: "strategy": "next_ln1" [001_layers_baseline/run-latest/output-gemma-2-9b.json:5785]
- layer 0 ln_source and metrics: "blocks[0].ln1", resid_norm_ratio 0.757…, delta_resid_cos 0.929… [001_layers_baseline/run-latest/output-gemma-2-9b.json:5791–5798]
- final ln_source: "ln_final" at layer 42 [001_layers_baseline/run-latest/output-gemma-2-9b.json:6266–6284]

Unembedding bias: present=false; l2_norm=0.0 (cosines are bias‑free) [001_layers_baseline/run-latest/output-gemma-2-9b.json:826–833].

Environment & determinism: torch 2.8, device=cpu, deterministic_algorithms=true, seed=316 [001_layers_baseline/run-latest/output-gemma-2-9b.json:6962–6971].

Numeric health: any_nan=false, any_inf=false, layers_flagged=[] [001_layers_baseline/run-latest/output-gemma-2-9b.json:6458–6466].

Copy mask: size=4668; sample of ignored tokens shows newline spans [001_layers_baseline/run-latest/output-gemma-2-9b.json:5608–5631].

Gold alignment: gold_answer and control gold are OK [001_layers_baseline/run-latest/output-gemma-2-9b.json:7001–7003, 7462–7470].

Control summary: first_control_margin_pos=18; max_control_margin=0.8677 [001_layers_baseline/run-latest/output-gemma-2-9b.json:7472–7480].

Summary indices (bits): first_kl_below_1.0=null; first_kl_below_0.5=null; first_rank_le_{10,5,1}=42 [001_layers_baseline/run-latest/output-gemma-2-9b.json:5641–5645].

Last‑layer head calibration: kl_to_final_bits=1.0129, top1_agree=true, p_top1_lens vs p_top1_model mismatch; temp_est=2.61, kl_after_temp_bits=0.3499, warn_high_last_layer_kl=true [001_layers_baseline/run-latest/output-gemma-2-9b.json:6401–6432]. Treat absolute probabilities cautiously; prefer ranks.

Measurement guidance: prefer_ranks=true, suppress_abs_probs=true; reasons include "warn_high_last_layer_kl", "norm_only_semantics_window", "high_lens_artifact_risk" [001_layers_baseline/run-latest/output-gemma-2-9b.json:8038–8047]. Preferred lens for reporting is "norm"; use_confirmed_semantics=true.

Raw‑vs‑Norm checks (window): center_layers=[0,42], radius=4, norm_only_semantics_layers=[42], max_kl_norm_vs_raw_bits_window=92.316 [001_layers_baseline/run-latest/output-gemma-2-9b.json:5747–5783].

Lens sanity (full): pct_layers_kl_ge_1.0=0.302, pct_layers_kl_ge_0.5=0.372, n_norm_only_semantics_layers=1, earliest_norm_only_semantic=42, max_kl_norm_vs_raw_bits=92.316, lens_artifact_score=0.581 (tier=high) [001_layers_baseline/run-latest/output-gemma-2-9b.json:5771–5783]. Early semantics should be treated as lens‑sensitive; rely on rank milestones and confirmed semantics.

Copy‑collapse flag check (strict): first row with copy_collapse=True is layer 0 with top‑1 ‘ simply’ (p≈0.9999993), next ‘simply’ (p≈7.7e‑07) — ✓ rule satisfied (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:row for layer 0).

Soft copy: earliest copy_soft_k1@0.5=True at layer 0; k2/k3 are null in this run [001_layers_baseline/run-latest/output-gemma-2-9b.json:5652–5675].

Lens selection and milestones (norm): L_surface_to_meaning=42 with answer_mass≈0.93 and echo_mass≈1.15e‑08; L_geom=42; L_topk_decay (K=50, τ=0.33)=28; cosine milestones ge_{0.2,0.4,0.6} at layers {1,42,42}; tau_norm_per_layer present; kl_to_final_bits_norm_temp@{25,50,75}% = {23.09, 21.27, 5.79} [001_layers_baseline/run-latest/output-gemma-2-9b.json:5719–5769, 6035–6073, 6467–6482].

Prism sidecar: compatible=true with metrics present; see Section 3 for verdict [001_layers_baseline/run-latest/output-gemma-2-9b.json:834–882].

Tuned‑lens: loaded; prefer_tuned=false; attribution provided; rank milestones unchanged vs baseline [001_layers_baseline/run-latest/output-gemma-2-9b.json:7335–7368, 7488–7511].

Negative control prompt check: “Berlin is the capital of” has top‑5 as (" Germany", 0.8766), (" the", 0.0699), (" modern", 0.0077), (" a", 0.0053), (" ", 0.0034); “ Berlin” appears with low mass [001_layers_baseline/run-latest/output-gemma-2-9b.json:10–31, 46–47]. No leakage.

Activation of venv, positional encodings, and RMSNorm policy follow script defaults; context_prompt has no trailing space after “simply” in run.py [001_layers_baseline/run.py:216–223].

Overall, sanity checks pass, with a known Gemma family final‑head calibration caveat.

**3. Quantitative Findings**

Table (pos, orig): Layer → entropy (bits), top‑1 token. Bold indicates the semantic collapse layer (confirmed norm lens).

| Layer | Entropy (bits) | Top‑1 |
|---|---:|---|
| 0 | 1.672e-05 | ` simply` |
| 1 | 6.942e-08 | ` simply` |
| 2 | 3.138e-05 | ` simply` |
| 3 | 4.300e-04 | ` simply` |
| 4 | 2.116e-03 | ` simply` |
| 5 | 2.333e-03 | ` simply` |
| 6 | 1.279e-01 | ` simply` |
| 7 | 3.357e-02 | ` simply` |
| 8 | 9.842e-02 | ` simply` |
| 9 | 1.021e-01 | ` simply` |
| 10 | 2.814e-01 | ` simply` |
| 11 | 3.330e-01 | ` simply` |
| 12 | 1.093e-01 | ` simply` |
| 13 | 1.374e-01 | ` simply` |
| 14 | 1.658e-01 | ` simply` |
| 15 | 7.349e-01 | ` simply` |
| 16 | 3.568 | ` simply` |
| 17 | 3.099 | ` simply` |
| 18 | 3.337 | ` simply` |
| 19 | 1.382 | ` simply` |
| 20 | 3.163 | ` simply` |
| 21 | 1.866 | ` simply` |
| 22 | 2.190 | ` simply` |
| 23 | 3.181 | ` simply` |
| 24 | 1.107 | ` simply` |
| 25 | 2.119 | ` the` |
| 26 | 2.371 | ` the` |
| 27 | 1.842 | ` the` |
| 28 | 1.227 | ` "` |
| 29 | 0.316 | ` "` |
| 30 | 0.134 | ` "` |
| 31 | 0.0461 | ` "` |
| 32 | 0.0625 | ` "` |
| 33 | 0.0427 | ` "` |
| 34 | 0.0900 | ` "` |
| 35 | 0.0234 | ` "` |
| 36 | 0.0741 | ` "` |
| 37 | 0.0825 | ` "` |
| 38 | 0.0335 | ` "` |
| 39 | 0.0469 | ` "` |
| 40 | 0.0362 | ` "` |
| 41 | 0.1767 | ` "` |
| 42 | 0.3701 | ` Berlin` |

Control margin (JSON): first_control_margin_pos = 18; max_control_margin = 0.8677 [001_layers_baseline/run-latest/output-gemma-2-9b.json:7472–7480].

Ablation (no‑filler): L_copy_orig=0, L_sem_orig=42; L_copy_nf=0, L_sem_nf=42; ΔL_copy=0, ΔL_sem=0 [001_layers_baseline/run-latest/output-gemma-2-9b.json:6982–6990].

- ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = 1.672e‑05 − 0.3701 ≈ −0.3701.
- Soft ΔH₁ (bits) = entropy(L_copy_soft[k1]) − entropy(L_semantic) = same as above; k2/k3 are null in this run.
- Confidence milestones (generic top‑1): p_top1 > 0.30 at L0; p_top1 > 0.60 at L0; final‑layer p_top1 ≈ 0.93 (prefer ranks per guidance).
- Rank milestones (diagnostics): rank ≤ 10 at L42; rank ≤ 5 at L42; rank ≤ 1 at L42 [001_layers_baseline/run-latest/output-gemma-2-9b.json:5643–5645].
- KL milestones (diagnostics, bits): first_kl_below_1.0 = null; first_kl_below_0.5 = null [001_layers_baseline/run-latest/output-gemma-2-9b.json:5641–5642]. Final KL≠0; see calibration caveat above.
- Cosine milestones (JSON): cos_to_final ≥ 0.2 at L1; ≥ 0.4 at L42; ≥ 0.6 at L42; final cos_to_final ≈ 0.9993.
- Coverage: L_topk_decay (K=50, τ=0.33) = 28 (norm lens).

Copy robustness (threshold sweep): stability="mixed"; earliest strict copy at τ=0.70 and 0.95 both at L0; norm_only_flags false at all τ [001_layers_baseline/run-latest/output-gemma-2-9b.json:5694–5731].

Prism Sidecar Analysis: compatible=true but regressive. KL(P_layer||P_final) increases markedly vs baseline (e.g., p50: baseline 15.17 bits vs prism 25.51 bits; Δ≈−10.33) and rank milestones remain null under prism [001_layers_baseline/run-latest/output-gemma-2-9b.json:848–882]. Verdict: Regressive.

Tuned‑Lens Comparison: prefer_tuned=false; rank milestones unchanged (Δ=0). KL deltas at percentiles indicate no consistent improvement beyond temperature: at p25 ΔKL_tuned≈−0.28 with ΔKL_rot≈+7.88; at p50 ΔKL_tuned≈−10.52 with ΔKL_rot≈−4.42; at p75 ΔKL_tuned≈+0.20 with ΔKL_rot≈+4.13 [001_layers_baseline/run-latest/output-gemma-2-9b.json:7480–7511]. No earlier first_rank milestones.

**4. Qualitative Patterns & Anomalies**

Early layers exhibit a strong copy‑reflex on the immediate prompt token “ simply” (L0–L7), then drift to grammatical fillers (“ the”) and punctuation/quotes mid‑stack before semantic collapse at the final layer. The strict copy rule (ID‑level, τ=0.95, δ=0.10) triggers at L0; soft‑copy (k=1, τ_soft=0.5) also fires at L0. Example: at L25 (NEXT) the top‑5 are the→simply→space→quote→a (001_layers_baseline/run-latest/output-gemma-2-9b-records.csv:553), indicating filler/markup anchoring precedes semantics.

Negative control (Berlin is the capital of): Top‑1 is “ Germany” with high mass; “ Berlin” remains low in the list (" Germany", 0.8766)…(" Berlin", 0.00187) [001_layers_baseline/run-latest/output-gemma-2-9b.json:13–16, 46–47]. No semantic leakage.

Important‑word trajectory (NEXT): “Berlin” does not enter top‑10/5 until L42 (diagnostics rank milestones). “simply” dominates early layers; mid‑depths are governed by punctuation and articles (e.g., quotes ‘"’ at L28–L41; table above). This pattern is consistent with early surface form copying, then syntactic framing, then late semantics.

Collapse‑layer stability: Removing the stylistic filler (“simply”) does not change collapse (L_sem stays 42); ΔL_sem=0, ΔL_copy=0. This argues against strong guidance‑style anchoring in this setup (001_layers_baseline/run-latest/output-gemma-2-9b.json:6982–6990).

Rest mass: remains tiny at NEXT; max after L_semantic (L=42) is ≈1.1e‑05 (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:row L42), suggesting good top‑k coverage.

Rotation vs amplification: cos_to_final rises early (≥0.2 by L1), but KL stays high until the end and final‑layer KL≠0. This is “early direction, late calibration”. Last‑layer calibration is off (kl_to_final_bits≈1.01; temp_est≈2.61; kl_after_temp_bits≈0.35) [001_layers_baseline/run-latest/output-gemma-2-9b.json:6401–6420]. Prefer rank milestones over absolute probabilities.

Lens sanity: raw‑vs‑norm indicates high artifact risk (full score tier=high; earliest norm‑only semantics at L42), though windowed checks also flag large KL discrepancies near the ends [001_layers_baseline/run-latest/output-gemma-2-9b.json:5747–5783]. Treat pre‑final “semantics” cautiously; rely on confirmed semantics and ranks (which both place collapse at 42).

Temperature robustness: Not explicitly varied here; tuned‑lens attribution separates temperature vs rotation gains and does not advance rank milestones (prefer_tuned=false).

Checklist
- RMS lens? ✓ (RMSNorm; next ln1)  [001_layers_baseline/run-latest/output-gemma-2-9b.json:5785–5798]
- LayerNorm bias removed? ✓ (not needed; RMS)  [001_layers_baseline/run-latest/output-gemma-2-9b.json:819–833]
- Entropy rise at unembed? n.a.
- FP32 un‑embed promoted? ✓ (unembed_dtype torch.float32)  [001_layers_baseline/run-latest/output-gemma-2-9b.json:822–823]
- Punctuation / markup anchoring? ✓ (quotes ‘"’ from L28–L41; table)
- Copy‑reflex? ✓ (strict at L0; soft k1 at L0)  [001_layers_baseline/run-latest/output-gemma-2-9b.json:5634–5666]
- Grammatical filler anchoring? ✓ (" the" top‑1 at L25–L27)
- Preferred lens honored? ✓ (preferred_lens_for_reporting = "norm")  [001_layers_baseline/run-latest/output-gemma-2-9b.json:8046]
- Confirmed semantics reported? ✓ (L_semantic_confirmed = 42)  [001_layers_baseline/run-latest/output-gemma-2-9b.json:6780–6787]
- Full dual‑lens metrics cited? ✓ (raw_lens_full tier=high)  [001_layers_baseline/run-latest/output-gemma-2-9b.json:5771–5783]
- Tuned‑lens attribution done? ✓ (ΔKL_tuned, ΔKL_temp, ΔKL_rot at 25/50/75%)  [001_layers_baseline/run-latest/output-gemma-2-9b.json:7480–7511]
- normalization_provenance present? ✓ (ln_source verified at 0 and 42)  [001_layers_baseline/run-latest/output-gemma-2-9b.json:5791–5798, 6266–6284]
- per‑layer normalizer effect present? ✓ (resid_norm_ratio, delta_resid_cos)  [same lines]
- unembed bias audited? ✓ (bias‑free cosine guaranteed)  [001_layers_baseline/run-latest/output-gemma-2-9b.json:826–833]
- deterministic_algorithms = true? ✓  [001_layers_baseline/run-latest/output-gemma-2-9b.json:6966]
- numeric_health clean? ✓  [001_layers_baseline/run-latest/output-gemma-2-9b.json:6458–6466]
- copy_mask plausible? ✓ (newline spans)  [001_layers_baseline/run-latest/output-gemma-2-9b.json:5608–5631]
- layer_map present? ✓  [001_layers_baseline/run-latest/output-gemma-2-9b.json:6091–6284]

**5. Limitations & Data Quirks**

- Final‑head calibration (Gemma family): kl_to_final_bits≈1.01 and warn_high_last_layer_kl=true. Per guidance, prefer ranks and within‑model trends over absolute probabilities; do not compare absolute p across families.
- Raw‑vs‑norm lens differences: raw_lens_check.mode="sample" for sampled rows, but raw_lens_full is present with lens_artifact_score tier=high and a norm‑only semantics layer at 42. Treat any “early semantics” as potentially lens‑induced; rely on confirmed semantics and rank milestones.
- Surface mass and rest_mass depend on tokenizer vocabulary; prefer within‑model trends (rest_mass≈1.1e‑05 at L42).

**6. Model Fingerprint**

Gemma‑2‑9B: collapse at L 42 (confirmed); final KL≈1.01 bits; mid‑stack dominated by quotes, answer only rank‑1 at the final layer.

---
Produced by OpenAI GPT-5 
