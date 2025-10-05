**Overview**

Qwen3-8B (Qwen/Qwen3-8B), 36 layers, evaluated 2025-10-04 (Experiment started: 2025-10-04 18:04:23) in this run. The probe traces surface-to-meaning dynamics at NEXT using a norm lens with per-layer RMS normalization and reports copy-reflex, rank/entropy/KL/cosine milestones, and calibration checks. Confirmed semantic collapse occurs at L 31 under the baseline norm lens; tuned-lens is used as the preferred reporting lens for milestones, with baseline included for context.

> "use_norm_lens": true  (001_layers_baseline/run-latest/output-Qwen3-8B.json:807)


**Method Sanity-Check**

The intended RMS norm lens and prompts are applied. The script defines the main positive context ending with “called simply” without a trailing space: "Give the city name only, plain text. The capital of Germany is called simply" (001_layers_baseline/run.py:254), and the JSON diagnostics echoes the exact context_prompt (001_layers_baseline/run-latest/output-Qwen3-8B.json:817). Normalizer provenance confirms pre-norm architecture with next-ln1 selection and epsilon inside sqrt throughout; layer 0 sources `blocks[0].ln1` and the final layer uses `ln_final` (001_layers_baseline/run-latest/output-Qwen3-8B.json:7222, 7227–7233, 7538–7546, 7742–7746). Unembed bias is absent (present=false, l2_norm=0.0) so geometry/cosines are bias‑free (001_layers_baseline/run-latest/output-Qwen3-8B.json:826–830). Determinism and environment are recorded (torch 2.8.0, device=cpu, dtype=float32, deterministic_algorithms=true, seed=316) (001_layers_baseline/run-latest/output-Qwen3-8B.json:8315–8325). Numeric health is clean (no NaN/Inf; no flagged layers) (001_layers_baseline/run-latest/output-Qwen3-8B.json:7748–7754). The copy mask is present with a plausible punctuation-heavy ignore set and size 6112 (001_layers_baseline/run-latest/output-Qwen3-8B.json:7056–7068, 7070).

Gold-token alignment is ID-based and ok (001_layers_baseline/run-latest/output-Qwen3-8B.json:8850–8873). Diagnostics include the required indices and copy detector configuration: L_semantic=31; first_rank thresholds (≤10/≤5/≤1) = 29/29/31; first_kl_below_{1.0,0.5}=36/36; copy_thresh=0.95 with match_level=id_subsequence and window_k=1 (001_layers_baseline/run-latest/output-Qwen3-8B.json:7074–7083). Strict-copy threshold sweep is present with stability="none" and null L_copy across τ∈{0.70,0.95}; all norm_only_flags null (001_layers_baseline/run-latest/output-Qwen3-8B.json:7132–7157). The emitted copy_flag_columns mirror these labels, including soft windows at τ_soft=0.5 and k∈{1,2,3} (001_layers_baseline/run-latest/output-Qwen3-8B.json:8328–8336, 889–897). Measurement guidance requests rank-forward reporting, suppressing absolute probability comparisons, preferring tuned lens, and using confirmed semantics (001_layers_baseline/run-latest/output-Qwen3-8B.json:8850–8861).

Last-layer head calibration is excellent: kl_to_final_bits=0.0 with exact top‑1 agreement and temp_est=1.0 (001_layers_baseline/run-latest/output-Qwen3-8B.json:7757–7764). Raw‑vs‑Norm window/full checks: window radius=4 centered at layers [29,31,36], max_kl_norm_vs_raw_bits_window ≈ 38.10 bits (001_layers_baseline/run-latest/output-Qwen3-8B.json:7206–7209). Full scan flags high lens‑artifact risk (pct_layers_kl_ge_1.0=0.7568; pct_layers_kl_ge_0.5=0.8108; n_norm_only_semantics_layers=0; max_kl_norm_vs_raw_bits≈38.10; tier=high) (001_layers_baseline/run-latest/output-Qwen3-8B.json:7210–7220). Given this, prefer rank milestones over absolute probabilities for pre‑final layers.

Copy-reflex screen (layers 0–3): strict copy_collapse=False and soft k1@τ_soft also False at L0–L3 (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:2–5). No early copy reflex is detected.

Control prompt and summary present; negative control margin turns positive at layer 1 with a very large maximum margin (001_layers_baseline/run-latest/output-Qwen3-8B.json:8345–8363). Ablation (no_filler) is present; L_sem_nf=31 equals L_sem_orig=31 (ΔL_sem=0) and strict copy remains null in both (001_layers_baseline/run-latest/output-Qwen3-8B.json:8337–8344).


**Quantitative Findings**

Preferred lens for milestones: tuned (per measurement_guidance), but the table below reflects the baseline norm-lens pure NEXT for pos/orig only. Confirmed semantics: L_semantic_confirmed=31 (confirmed_source=raw) alongside L_semantic_norm=31 (001_layers_baseline/run-latest/output-Qwen3-8B.json:8134–8142).

| Layer | Entropy (bits) | Top‑1 token |
|---|---:|---|
| L 0 | 17.2128 | 'CLICK' |
| L 1 | 17.2114 | 'apr' |
| L 2 | 17.2105 | '财经' |
| L 3 | 17.2083 | '-looking' |
| L 4 | 17.2059 | '院子' |
| L 5 | 17.2037 | ' (?)' |
| L 6 | 17.1963 | 'ly' |
| L 7 | 17.1463 | ' (?)' |
| L 8 | 17.1322 | ' (?)' |
| L 9 | 17.1188 | ' (?)' |
| L 10 | 17.0199 | ' (?)' |
| L 11 | 17.1282 | 'ifiable' |
| L 12 | 17.1169 | 'ifiable' |
| L 13 | 17.1256 | 'ifiable' |
| L 14 | 17.0531 | '"' |
| L 15 | 17.0364 | '"' |
| L 16 | 16.9128 | '-' |
| L 17 | 16.9716 | '-' |
| L 18 | 16.9106 | '-' |
| L 19 | 16.6286 | 'ly' |
| L 20 | 16.6960 | '_' |
| L 21 | 16.4081 | '_' |
| L 22 | 15.2195 | ' ______' |
| L 23 | 15.2203 | '____' |
| L 24 | 10.8929 | '____' |
| L 25 | 13.4545 | '____' |
| L 26 | 5.5576 | '____' |
| L 27 | 4.3437 | '____' |
| L 28 | 4.7859 | '____' |
| L 29 | 1.7777 | '-minded' |
| L 30 | 2.2030 | ' Germany' |
| **L 31** | 0.4539 | ' Berlin' |
| L 32 | 1.0365 | ' German' |
| L 33 | 0.9878 | ' Berlin' |
| L 34 | 0.6691 | ' Berlin' |
| L 35 | 2.4944 | ' Berlin' |
| L 36 | 3.1226 | ' Berlin' |

- Control margin: first_control_margin_pos=1; max_control_margin=0.999997735 (001_layers_baseline/run-latest/output-Qwen3-8B.json:8360–8363).
- Ablation (no-filler): L_copy_orig=null, L_sem_orig=31; L_copy_nf=null, L_sem_nf=31; ΔL_copy=n/a, ΔL_sem=0 (001_layers_baseline/run-latest/output-Qwen3-8B.json:8337–8344).

ΔH (bits) = entropy(L_copy) − entropy(L_semantic): n/a (strict and soft copy layers are null). Soft ΔHₖ: n/a (k∈{1,2,3} all null) (001_layers_baseline/run-latest/output-Qwen3-8B.json:7090–7100, 7139–7144).

Confidence milestones (pure CSV):
- p_top1 > 0.30 at L 29; p_top1 > 0.60 at L 29; final-layer p_top1 = 0.4334 (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:31, 38).

Rank milestones (diagnostics):
- rank ≤ 10 at L 29; rank ≤ 5 at L 29; rank ≤ 1 at L 31 (001_layers_baseline/run-latest/output-Qwen3-8B.json:7081–7083). Preferred lens gate is tuned; tuned milestones are later: first_rank_le_1 at L 34; le_5 at L 31; le_10 at L 30 (001_layers_baseline/run-latest/output-Qwen3-8B.json:8376–8380).

KL milestones (diagnostics):
- first_kl_below_1.0 at L 36; first_kl_below_0.5 at L 36; KL decreases toward final and is ≈ 0 at final (001_layers_baseline/run-latest/output-Qwen3-8B.json:7079–7080, 7757).

Cosine milestones (diagnostics):
- first cos_to_final ≥ {0.2, 0.4, 0.6} all at L 36; final cos_to_final ≈ 1.0 (001_layers_baseline/run-latest/output-Qwen3-8B.json:7170–7174, 38 in CSV shows cos_to_final≈1.0 at L 36).

Depth fractions: L_semantic_frac ≈ 0.861; first_rank_le_5_frac ≈ 0.806 (001_layers_baseline/run-latest/output-Qwen3-8B.json:7177–7179).

Copy robustness (threshold sweep): stability="none", L_copy_strict at τ=0.70 and 0.95 both null; no norm-only flip (001_layers_baseline/run-latest/output-Qwen3-8B.json:7132–7157). Baseline copy_flag_columns include strict and soft labels (001_layers_baseline/run-latest/output-Qwen3-8B.json:8328–8336).

Prism sidecar: present and compatible (k=512; layers [embed,8,17,26]) (001_layers_baseline/run-latest/output-Qwen3-8B.json:834–846, 839–845). KL at percentiles worsens vs baseline at early/mid depths (Δp25≈−0.36, Δp50≈−0.59, Δp75≈−7.03 bits); first_kl_le_1.0 remains null under Prism (001_layers_baseline/run-latest/output-Qwen3-8B.json:865–886). Rank milestones under Prism are null (no le_{10,5,1}) (001_layers_baseline/run-latest/output-Qwen3-8B.json:854–862). Verdict: Regressive relative to baseline (higher KL, no earlier ranks).

Tuned‑lens: loaded and preferred for reporting with rotation gains beyond temperature. Attribution at percentiles: ΔKL_tuned ≈ {4.14, 4.00, 1.40} bits and ΔKL_temp ≈ {−0.52, −0.08, −1.25} leading to ΔKL_rot ≈ {4.66, 4.08, 2.64} (001_layers_baseline/run-latest/output-Qwen3-8B.json: attributions under tuned_lens; 841–end; see 8850–8861 prefer_tuned=true). Rank deltas (tuned − baseline): le_10 +1, le_5 +2, le_1 +3 (001_layers_baseline/run-latest/output-Qwen3-8B.json: last tuned summary metrics block around 8467–8685).


**Qualitative Patterns & Anomalies**

Important-word trajectory. Berlin first enters any top‑5 at L 29 and becomes rank‑1 by L 31; it stabilises as the persistent top‑1 from L 33 onward (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:31, 33–36). At L 29 the model still prefers a non‑answer surface token while “Berlin” sits at rank 4 with p=0.0265 (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:31: “… Berlin,0.02649 … answer_rank=4 …”).

Rotation vs amplification. KL to final decreases with depth and reaches 0 at the head, while p_answer and cosine to final rise late: at L 18, entropy≈16.91 bits and KL≈12.41 bits with very low p_answer, cos_to_final negative (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:20). By L 31, p_top1≈0.936 and is_answer=True with KL≈1.06 bits (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:33), and cosine aligns by the final layer (cos_to_final≈1.0 at L 36; 001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:38). This pattern suggests a late calibration phase where direction aligns near the top but normalization and rank settle closer to collapse.

Negative control. For the test prompt “Berlin is the capital of”, top‑5 includes “ Germany” with p=0.7286 and “ Berlin” remains in the list at p≈4.59e‑4 — semantic leakage: Berlin rank 9 (p=0.000459) (001_layers_baseline/run-latest/output-Qwen3-8B.json:10–16, 46–48).

Rest‑mass sanity. Rest_mass declines strongly by L 31 and remains low thereafter; the maximum after L_semantic appears at the final layer with rest_mass≈0.175 (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:38). Treat rest_mass as coverage only, not fidelity.

Head calibration. Final‑layer agreement is perfect (kl_to_final_bits=0.0; temp_est=1.0; warn_high_last_layer_kl=false) (001_layers_baseline/run-latest/output-Qwen3-8B.json:7757–7774). Probability calibration at the head is sound; cross‑family probability comparisons are still avoided per measurement_guidance.

Lens sanity. Raw‑vs‑Norm “full” check marks high lens‑artifact risk with max_kl_norm_vs_raw_bits≈38.10 bits and tier="high"; first_norm_only_semantic_layer=null (001_layers_baseline/run-latest/output-Qwen3-8B.json:7210–7220). The sampled raw‑vs‑norm rows also show large mid‑depth KL gaps (e.g., L 28 ≈13.60 bits) (001_layers_baseline/run-latest/output-Qwen3-8B.json:8295–8303). Consistent with measurement_guidance, statements emphasize rank thresholds and within‑model trends.

Temperature robustness. At T=0.1, Berlin rank 1 with p≈0.99915 and entropy≈0.0099 bits; at T=2.0, Berlin still leads but p≈0.0419 and entropy≈13.40 bits (001_layers_baseline/run-latest/output-Qwen3-8B.json:714–734, 736–760).

Stylistic ablation. Removing “simply” leaves L_sem unchanged (31→31); no strict/soft copy emergence, pointing to minimal stylistic anchoring for this prompt (001_layers_baseline/run-latest/output-Qwen3-8B.json:8337–8344).

Checklist
- RMS lens? ✓ (pre_norm; next_ln1) (001_layers_baseline/run-latest/output-Qwen3-8B.json:7222–7225)
- LayerNorm bias removed? ✓/n.a. (RMSNorm; bias not needed) (001_layers_baseline/run-latest/output-Qwen3-8B.json:810–816)
- Entropy rise at unembed? ✗ (final entropy≈teacher entropy; 001_layers_baseline/run-latest/output-Qwen3-8B.json:8159–8166)
- FP32 un‑embed promoted? ✓ (unembed_dtype=torch.float32) (001_layers_baseline/run-latest/output-Qwen3-8B.json:809)
- Punctuation / markup anchoring? ✓ (early top‑1 underscores/punctuation around L 20–28; table above)
- Copy‑reflex? ✗ (no strict/soft hits at L0–3) (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:2–5)
- Grammatical filler anchoring? ✓ (" the" and quotes appear in early top‑k; 001_layers_baseline/run-latest/output-Qwen3-8B.json:22–27, 65–71)
- Preferred lens honored in milestones? ✓ (measurement_guidance preferred_lens_for_reporting=tuned) (001_layers_baseline/run-latest/output-Qwen3-8B.json:8859)
- Confirmed semantics reported? ✓ (L_semantic_confirmed=31; source=raw) (001_layers_baseline/run-latest/output-Qwen3-8B.json:8134–8142)
- Full dual‑lens metrics cited? ✓ (raw_lens_full tier=high; pct/n_norm_only/max_kl quoted) (001_layers_baseline/run-latest/output-Qwen3-8B.json:7210–7220)
- Tuned‑lens attribution done? ✓ (ΔKL_tuned/ΔKL_temp/ΔKL_rot at ~25/50/75%) (tuned block)
- normalization_provenance present? ✓ (strategy=next_ln1) (001_layers_baseline/run-latest/output-Qwen3-8B.json:7222–7225)
- per‑layer normalizer effect metrics present? ✓ (resid_norm_ratio, delta_resid_cos listed per layer) (001_layers_baseline/run-latest/output-Qwen3-8B.json:7227–7233)
- unembed bias audited? ✓ (present=false; l2_norm=0.0) (001_layers_baseline/run-latest/output-Qwen3-8B.json:826–830)
- deterministic_algorithms = true? ✓ (001_layers_baseline/run-latest/output-Qwen3-8B.json:8321)
- numeric_health clean? ✓ (no NaN/Inf; no flagged layers) (001_layers_baseline/run-latest/output-Qwen3-8B.json:7748–7754)
- copy_mask present and plausible? ✓ (size=6112; punctuation sample) (001_layers_baseline/run-latest/output-Qwen3-8B.json:7056–7068, 7070)
- layer_map present? ✓ (001_layers_baseline/run-latest/output-Qwen3-8B.json:7538–7560, 7742–7746)


**Limitations & Data Quirks**

- High raw‑vs‑norm artifact risk (tier=high; max_kl_norm_vs_raw_bits≈38.10) suggests pre‑final “early semantics” under the norm lens may be lens‑induced; rank milestones and confirmed semantics are preferred for interpretation (001_layers_baseline/run-latest/output-Qwen3-8B.json:7210–7220).
- Rest_mass is a top‑k coverage measure; after L_semantic it reaches ≈0.175 at L 36 (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:38) and should not be used for lens fidelity.
- KL is lens‑sensitive; final‑head calibration is clean here (kl_to_final_bits=0), but cross‑model probability claims are avoided per measurement_guidance (001_layers_baseline/run-latest/output-Qwen3-8B.json:7757–7764, 8850–8861).
- Surface‑mass and exact p thresholds vary with tokenizer; within‑model rank trends are favored over absolute mass values.


**Model Fingerprint**

Qwen3‑8B: collapse at L 31 (confirmed), final entropy ≈3.12 bits; “Berlin” first in top‑5 at L 29 and stabilizes as rank‑1 by L 33.

---
Produced by OpenAI GPT-5 

