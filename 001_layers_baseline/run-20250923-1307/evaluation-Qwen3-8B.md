# Evaluation Report: Qwen/Qwen3-8B

*Run executed on: 2025-09-23 13:07:01*
**1. Overview**

Qwen/Qwen3-8B was probed on a single‑token next‑word task using a normalized residual-stream (norm lens) with fp32 unembedding and per‑layer KL to the final head. The run captures copy/filler collapse vs. semantic emergence, per‑layer entropy and ranks, and calibration diagnostics, with a stylistic ablation (no‑filler) and a Prism sidecar for comparison.

**2. Method Sanity‑Check**

Diagnostics confirm the intended setup: the norm lens is enabled and positional encoding handling matches a rotary model. For example: “use_norm_lens: true” (001_layers_baseline/run-latest/output-Qwen3-8B.json:807) and “layer0_position_info: token_only_rotary_model” (001_layers_baseline/run-latest/output-Qwen3-8B.json:816). The prompt ends exactly with “called simply” (001_layers_baseline/run-latest/output-Qwen3-8B.json:817).

Copy detector configuration is present and consistent across JSON/CSV: “copy_thresh: 0.95”, “copy_window_k: 1”, “copy_match_level: id_subsequence” (001_layers_baseline/run-latest/output-Qwen3-8B.json:846–848); soft config “threshold: 0.5; window_ks: [1,2,3]” (001_layers_baseline/run-latest/output-Qwen3-8B.json:833–841); and the CSV mirrors with “copy_flag_columns: [copy_strict@0.95, copy_soft_k1@0.5, copy_soft_k2@0.5, copy_soft_k3@0.5]” (001_layers_baseline/run-latest/output-Qwen3-8B.json:1077–1081). Gold‑token alignment is ID‑based and OK: “gold_alignment: ok” (001_layers_baseline/run-latest/output-Qwen3-8B.json:898). Negative control is present with summary stats: “first_control_margin_pos: 1; max_control_margin: 0.9999977350234983” (001_layers_baseline/run-latest/output-Qwen3-8B.json:1108–1113). Ablation summary exists and positive rows are present for both variants in the CSV (see pos/orig and pos/no_filler rows, e.g., 001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:33,76).

Summary indices (bits for KL/entropy): first_kl_below_0.5 = 36; first_kl_below_1.0 = 36; first_rank_le_1 = 31; first_rank_le_5 = 29; first_rank_le_10 = 29 (001_layers_baseline/run-latest/output-Qwen3-8B.json:849–853). Last‑layer head calibration is excellent: the CSV shows final kl_to_final_bits = 0.0 at L36 (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:38), and the dedicated snapshot agrees: “kl_to_final_bits: 0.0; top1_agree: true; temp_est: 1.0; warn_high_last_layer_kl: false” (001_layers_baseline/run-latest/output-Qwen3-8B.json:899–917). Lens sanity (sample mode) flags high artifact risk: “lens_artifact_risk: high; max_kl_norm_vs_raw_bits: 13.6049; first_norm_only_semantic_layer: null” (001_layers_baseline/run-latest/output-Qwen3-8B.json:1072–1074).

Copy‑collapse flag check (strict τ=0.95, δ=0.10): no layer fires in L0–L3 (and none elsewhere). Soft copy flags (τ_soft=0.5, k∈{1,2,3}) never fire. First row with copy_collapse=True: n.a. (none observed). Earliest soft flags: none.

Main table below filters to prompt_id = pos and prompt_variant = orig.

**3. Quantitative Findings**

Per‑layer summary (pure next‑token; entropy in bits; top‑1 token):

| Layer | Entropy (bits) | Top‑1 token |
|---:|---:|:---|
| L 0 | 17.2128 | 'CLICK' |
| L 1 | 17.2114 | 'apr' |
| L 2 | 17.2105 | '财经' |
| L 3 | 17.2083 | '-looking' |
| L 4 | 17.2059 | '院子' |
| L 5 | 17.2037 | '(?)' |
| L 6 | 17.1963 | 'ly' |
| L 7 | 17.1463 | '(?)' |
| L 8 | 17.1322 | '(?)' |
| L 9 | 17.1188 | '(?)' |
| L 10 | 17.0199 | '(?)' |
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
| L 22 | 15.2195 | '______' |
| L 23 | 15.2203 | '____' |
| L 24 | 10.8929 | '____' |
| L 25 | 13.4545 | '____' |
| L 26 | 5.5576 | '____' |
| L 27 | 4.3437 | '____' |
| L 28 | 4.7859 | '____' |
| L 29 | 1.7777 | '-minded' |
| L 30 | 2.2030 | 'Germany' |
| **L 31** | 0.4539 | 'Berlin' |
| L 32 | 1.0365 | 'German' |
| L 33 | 0.9878 | 'Berlin' |
| L 34 | 0.6691 | 'Berlin' |
| L 35 | 2.4944 | 'Berlin' |
| L 36 | 3.1226 | 'Berlin' |

Control margin (JSON): first_control_margin_pos = 1; max_control_margin = 0.9999977350 (001_layers_baseline/run-latest/output-Qwen3-8B.json:1119–1122).

Ablation (no‑filler): L_copy_orig = null, L_sem_orig = 31; L_copy_nf = null, L_sem_nf = 31; ΔL_copy = null, ΔL_sem = 0 (001_layers_baseline/run-latest/output-Qwen3-8B.json:1083–1090). Interpretation: no detectable shift from removing the filler “simply”.

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (L_copy = null). Soft ΔHk (bits) = n.a. for k ∈ {1,2,3} (L_copy_soft[k] = null).

Confidence milestones (pure CSV): p_top1 > 0.30 at L29; p_top1 > 0.60 at L29; final‑layer p_top1 = 0.4334 (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:38).

Rank milestones (JSON): rank ≤ 10 at L29; rank ≤ 5 at L29; rank ≤ 1 at L31 (001_layers_baseline/run-latest/output-Qwen3-8B.json:851–853).

KL milestones (JSON): first_kl_below_1.0 at L36; first_kl_below_0.5 at L36; KL decreases with depth and is ≈ 0 at the final layer (001_layers_baseline/run-latest/output-Qwen3-8B.json:849–850,899–907).

Cosine milestones (pure CSV): first cos_to_final ≥ 0.2 at L36, ≥ 0.4 at L36, ≥ 0.6 at L36; final cos_to_final = 1.0000 (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:38).

Prism Sidecar Analysis
- Presence: compatible = true (001_layers_baseline/run-latest/output-Qwen3-8B.json:819–831). Prism CSVs are present.
- Early‑depth stability (KL to final, bits): baseline vs Prism at L0/9/18/27 = 12.79/12.61/12.41/6.14 vs 12.94/12.98/13.00/13.18 — KL increases under Prism, especially mid‑stack.
- Rank milestones (Prism): no layer reaches rank ≤ 10 (none observed in Prism pure CSV; e.g., answer_rank ≫ 10 at L31–36: 001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token-prism.csv:33–38).
- Top‑1 agreement: baseline top‑1 “Berlin” by L31 (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:33) vs Prism “ertools/reflect/…” at L31–34 (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token-prism.csv:33–36) — disagreement at all sampled depths.
- Cosine drift: Prism never crosses 0.2; final cos_to_final ≈ 0.037 (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token-prism.csv:38) vs baseline 1.0 at final.
- Copy flags: no spurious flips — all copy flags remain False under Prism.
- Verdict: Regressive (higher KL at early/mid depths and later/no rank milestones).

**4. Qualitative Patterns & Anomalies**

Negative control shows expected behaviour with slight semantic leakage: for “Berlin is the capital of” the top‑5 is dominated by “Germany/which/the/what/__” and includes Berlin at low mass — “Germany, 0.7286” (001_layers_baseline/run-latest/output-Qwen3-8B.json:14–15) and “Berlin, 0.00046” (001_layers_baseline/run-latest/output-Qwen3-8B.json:46–47). Semantic leakage: Berlin rank 9 (p = 0.00046).

Important‑word trajectory: Berlin first enters any top‑5 at L29 (“… ‘Berlin’, 0.02649”  [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:31]), becomes top‑2 at L30 (“… ‘Berlin’, 0.28436”  [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:32]), and is top‑1 by L31 (“… ‘Berlin’, 0.93593”  [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:33]). “Germany” is top‑1 at L30 (“… ‘Germany’, 0.40795”  [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:32]) and remains in the top‑5 through late layers. Early layers are dominated by punctuation/markup (quotes/underscores/dashes), consistent with filler anchoring rather than early semantics.

Collapse‑layer stability: removing the filler (“no_filler”) does not shift semantics (L_sem_nf = 31; ΔL_sem = 0; 001_layers_baseline/run-latest/output-Qwen3-8B.json:1083–1090). The strict copy detector never fires in either variant, and soft windows do not trigger, indicating minimal copy reflex in this setup.

Rest‑mass sanity: Rest_mass does not monotonically fall; it is minimal near L31 (≈0.005) and rises to 0.175 at the final layer (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:33,38), reflecting broader top‑k distribution at the end; rest_mass is top‑k coverage only and not a lens‑fidelity metric.

Rotation vs amplification: KL_to_final decreases only late (first ≤1.0 at L36), while cos_to_final snaps into alignment only at the final layer (≥0.2/0.4/0.6 all at L36), suggesting “late direction, late calibration” rather than a long gradual rotation. Given the high raw‑vs‑norm discrepancy (“lens_artifact_risk: high”  [001_layers_baseline/run-latest/output-Qwen3-8B.json:1072–1074]), treat apparent pre‑final movements cautiously and prefer rank milestones.

Head calibration (final): Final‑head alignment is clean — “kl_to_final_bits: 0.0; top1_agree: true; temp_est: 1.0” (001_layers_baseline/run-latest/output-Qwen3-8B.json:899–907); no temperature or transform correction is indicated.

Temperature robustness: At T = 0.1, Berlin rank 1 (p = 0.99915; entropy = 0.0099) (001_layers_baseline/run-latest/output-Qwen3-8B.json:670–681). At T = 2.0, Berlin remains rank 1 (p = 0.04186; entropy = 13.3983) (001_layers_baseline/run-latest/output-Qwen3-8B.json:736–762). Entropy rises strongly with temperature.

Checklist
- RMS lens? ✓ (“first_block_ln1_type: RMSNorm”; “use_norm_lens: true” — 001_layers_baseline/run-latest/output-Qwen3-8B.json:810,807)
- LayerNorm bias removed? ✓ (“not_needed_rms_model” — 001_layers_baseline/run-latest/output-Qwen3-8B.json:812)
- Entropy rise at unembed? ✓ (entropy increases again after L31; final entropy 3.1226 — 001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:38)
- FP32 un‑embed promoted? ✓ (unembed_dtype: torch.float32; use_fp32_unembed: false — 001_layers_baseline/run-latest/output-Qwen3-8B.json:809,808)
- Punctuation / markup anchoring? ✓ (early layers dominated by quotes/underscores/dashes; see table rows L14–L23)
- Copy‑reflex? ✗ (no strict or soft copy flags in L0–L3 or elsewhere)
- Grammatical filler anchoring? ✗ (no “is/the/a/of” dominance in L0–L5 top‑1)

**5. Limitations & Data Quirks**

Raw‑vs‑norm checks are “sample” mode only and flag high lens‑artifact risk (max_kl_norm_vs_raw_bits ≈ 13.6), so any pre‑final “early semantics” should be treated cautiously and rank milestones preferred for claims. Final‑row KL≈0 means last‑layer calibration is good for this model; nonetheless, KL is lens‑sensitive and should be used qualitatively. Copy layer (strict and soft) is null for this run, so ΔH metrics based on L_copy are not available.

**6. Model Fingerprint**

Qwen‑3‑8B: collapse at L 31; final entropy 3.1226 bits; “Berlin” enters top‑5 at L 29 and stabilizes by L 31; Prism sidecar regressive.

---
Produced by OpenAI GPT-5

