**Overview**
Google Gemma‑2‑9B (9B) evaluated with the project’s logit‑lens pipeline on the Germany→Berlin prompt. The probe records per‑layer entropy, top‑1 tokens, answer rank/probability, KL to final, cosine to final, and copy‑collapse flags for the pure next‑token.

**Method Sanity‑Check**
Diagnostics confirm the RMS norm‑lens and rotary positions are applied: “use_norm_lens: true” (001_layers_baseline/run-latest/output-gemma-2-9b.json:807) and “layer0_position_info: token_only_rotary_model” (001_layers_baseline/run-latest/output-gemma-2-9b.json:816). The context_prompt ends with “called simply” (001_layers_baseline/run-latest/output-gemma-2-9b.json:4). Copy detector config is present with strict τ=0.95, k=1 and soft τ_soft=0.5, k∈{1,2,3}: “copy_thresh: 0.95, copy_window_k: 1, copy_match_level: id_subsequence” (001_layers_baseline/run-latest/output-gemma-2-9b.json:846–848) and “copy_soft_config: { threshold: 0.5, window_ks: [1,2,3], extra_thresholds: [] }” (001_layers_baseline/run-latest/output-gemma-2-9b.json:833–841). The CSV flag columns mirror these (“copy_strict@0.95”, “copy_soft_k1@0.5”, “copy_soft_k2@0.5”, “copy_soft_k3@0.5”) (001_layers_baseline/run-latest/output-gemma-2-9b.json:1077–1081). Gold answer alignment is ID‑level and ok (001_layers_baseline/run-latest/output-gemma-2-9b.json:1091–1100,1110–1117). Negative control prompt is present with a summary (001_layers_baseline/run-latest/output-gemma-2-9b.json:1091–1092,1106–1109). Ablation summary exists and both prompt variants are present (“L_copy_orig=0, L_sem_orig=42; L_copy_nf=0, L_sem_nf=42; delta_L_copy=0, delta_L_sem=0”) (001_layers_baseline/run-latest/output-gemma-2-9b.json:1083–1089). For summary indices: first_kl_below_0.5 = null, first_kl_below_1.0 = null, first_rank_le_1 = 42, first_rank_le_5 = 42, first_rank_le_10 = 42 (001_layers_baseline/run-latest/output-gemma-2-9b.json:849–853). Units for KL/entropy are bits (fields: kl_to_final_bits, entropy). Last‑layer head calibration shows non‑zero final KL: kl_to_final_bits = 1.0129 bits with top1_agree=true, p_top1_lens=0.9298 vs p_top1_model=0.3943, temp_est=2.61, kl_after_temp_bits=0.3499, warn_high_last_layer_kl=true (001_layers_baseline/run-latest/output-gemma-2-9b.json:899–907,917). Lens sanity (raw vs norm): lens_artifact_risk = “high”, max_kl_norm_vs_raw_bits = 12.9056, first_norm_only_semantic_layer = null (001_layers_baseline/run-latest/output-gemma-2-9b.json:1071–1075). Copy‑collapse fires in early layers on the pure CSV (see below); flag copy‑reflex ✓ in Section 4.

Copy‑collapse flag check (first True): layer 0, top‑1 “simply” p1=0.9999993, p2=7.73e‑07 (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2) — ✓ rule satisfied. Soft copy earliest: k1 at layer 0; k2/k3 remain null in diagnostics (001_layers_baseline/run-latest/output-gemma-2-9b.json:860–864).

**Quantitative Findings**
Main rows (filter: prompt_id=pos, prompt_variant=orig; token=⟨NEXT⟩):

| Layer | Entropy (bits) | Top‑1 token |
|---:|---:|:---|
| 0 | 0.000 | simply |
| 1 | 0.000 | simply |
| 2 | 0.000 | simply |
| 3 | 0.000 | simply |
| 4 | 0.002 | simply |
| 5 | 0.002 | simply |
| 6 | 0.128 | simply |
| 7 | 0.034 | simply |
| 8 | 0.098 | simply |
| 9 | 0.102 | simply |
| 10 | 0.281 | simply |
| 11 | 0.333 | simply |
| 12 | 0.109 | simply |
| 13 | 0.137 | simply |
| 14 | 0.166 | simply |
| 15 | 0.735 | simply |
| 16 | 3.568 | simply |
| 17 | 3.099 | simply |
| 18 | 3.337 | simply |
| 19 | 1.382 | simply |
| 20 | 3.163 | simply |
| 21 | 1.866 | simply |
| 22 | 2.190 | simply |
| 23 | 3.181 | simply |
| 24 | 1.107 | simply |
| 25 | 2.119 | the |
| 26 | 2.371 | the |
| 27 | 1.842 | the |
| 28 | 1.227 | " |
| 29 | 0.316 | " |
| 30 | 0.134 | " |
| 31 | 0.046 | " |
| 32 | 0.063 | " |
| 33 | 0.043 | " |
| 34 | 0.090 | " |
| 35 | 0.023 | " |
| 36 | 0.074 | " |
| 37 | 0.083 | " |
| 38 | 0.033 | " |
| 39 | 0.047 | " |
| 40 | 0.036 | " |
| 41 | 0.177 | " |
| 42 | 0.370 | **Berlin** |

Control margin (JSON control_summary): first_control_margin_pos = 18; max_control_margin = 0.8677237033843427 (001_layers_baseline/run-latest/output-gemma-2-9b.json:1106–1109).

Ablation (no‑filler). From JSON ablation_summary: L_copy_orig = 0, L_sem_orig = 42; L_copy_nf = 0, L_sem_nf = 42; ΔL_copy = 0; ΔL_sem = 0 (001_layers_baseline/run-latest/output-gemma-2-9b.json:1083–1089). Interpretation: no shift from removing “simply”; semantics land only at the final layer.

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = 0.000017 − 0.370067 ≈ −0.3701. Soft ΔH₁ (bits) = 0.000017 − 0.370067 ≈ −0.3701; k=2,3 not present (null in diagnostics).

Confidence milestones (pure CSV, generic top‑1): p_top1 > 0.30 at layer 0; p_top1 > 0.60 at layer 0; final‑layer p_top1 = 0.9298 (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:49).

Rank milestones (JSON diagnostics): rank ≤ 10 at layer 42; rank ≤ 5 at layer 42; rank ≤ 1 at layer 42 (001_layers_baseline/run-latest/output-gemma-2-9b.json:851–853).

KL milestones (JSON diagnostics + CSV): first_kl_below_1.0 = null; first_kl_below_0.5 = null (001_layers_baseline/run-latest/output-gemma-2-9b.json:849–850). KL decreases late but remains ≈1.01 bits at final (001_layers_baseline/run-latest/output-gemma-2-9b.json:900), indicating final‑head calibration issues; rely on ranks for semantics.

Cosine milestones (pure CSV): cos_to_final ≥ 0.2 first at L1; ≥ 0.4 at L42; ≥ 0.6 at L42; final cos_to_final = 0.9993 (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:49).

Prism Sidecar Analysis
- Presence: prism sidecar present and compatible (001_layers_baseline/run-latest/output-gemma-2-9b.json:819–830).
- Early‑depth stability (KL vs final at sampled depths): baseline KL drops to ~1.01 bits by L42, while Prism stays very high (e.g., L0 14.39 vs Prism 28.48; L21 15.17 vs 25.51; L31 1.87 vs 26.01; L42 1.01 vs 28.73).
- Rank milestones: Prism never reaches rank ≤ 10 (first_rank_le_{10,5,1} = None) while baseline reaches rank 1 at L42.
- Top‑1 agreement: Prism diverges at all sampled depths; at L42, Prism p_top1≈0.0112 vs baseline 0.9298.
- Cosine drift: Prism cos_to_final remains <0.2 at all sampled depths; baseline reaches 0.9993 at L42.
- Copy flags: Prism does not spuriously flip copy_collapse; baseline shows early copy on “simply”.
- Verdict: Regressive — Prism inflates KL and delays/no‑shows rank milestones, providing no calibration benefit here.

**Qualitative Patterns & Anomalies**
Strict copy‑reflex is pronounced at the context boundary: at L0 the next token is the filler “simply” with p=0.9999993 and massive margin over p2 (7.7e‑07) (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2). This persists through mid‑stack before drifting to punctuation (many layers 28–41 have top‑1 '"') and only at the final layer does the model switch to the answer token “Berlin” (p=0.9298; answer_rank=1) (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:49). KL declines sharply very late while cosine rises early then saturates at the end, an “early direction, late calibration” signature; given warn_high_last_layer_kl=true and temp_est≈2.61, absolute probabilities should be taken cautiously (001_layers_baseline/run-latest/output-gemma-2-9b.json:900–907,917).

Negative control: “Berlin is the capital of” top‑5 shows no Berlin leakage; top candidates are “ Germany” (0.8766), “ the” (0.0699), “ modern” (0.0077), “ a” (0.0053), “ ” (0.0034) (001_layers_baseline/run-latest/output-gemma-2-9b.json:12–31). No semantic leakage is observed.

Important‑word trajectory: important words are {Germany, Berlin, capital, Answer, word, simply} (001_layers_baseline/run.py:335). Across layers for the next‑token position, “simply” dominates early (L0–L24) before punctuation takeover (L28–L41), and “Berlin” only appears and stabilizes at the final layer (rank 1, p=0.9298) (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2–49). Semantically close geography tokens do not meaningfully enter the top‑5 before L42 in the pure CSV.

Instructional brevity (one‑word cue) removal via ablation does not shift collapse or semantics (ΔL_copy=0; ΔL_sem=0), suggesting style prompts here do not anchor semantics for this model (001_layers_baseline/run-latest/output-gemma-2-9b.json:1083–1089).

Rest‑mass sanity: rest_mass remains near zero throughout the final layers; max after L_semantic ≈ 1.11e‑05 (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:49), suggesting no precision loss.

Rotation vs amplification: cosine alignment reaches ≥0.2 by L1 but KL stays high until the end; combined with the final‑head calibration warning, this indicates correct direction emerges earlier than calibrated probabilities (“early direction, late calibration”).

Head calibration (final layer): warn_high_last_layer_kl=true with temp_est≈2.61 reduces KL to ~0.35 bits under temperature scaling but still non‑negligible (001_layers_baseline/run-latest/output-gemma-2-9b.json:906–907). Treat final‑row probabilities as family‑specific calibration; rely on rank milestones.

Lens sanity: raw‑vs‑norm check reports lens_artifact_risk = high and max_kl_norm_vs_raw_bits ≈ 12.91 (001_layers_baseline/run-latest/output-gemma-2-9b.json:1071–1075). Early “semantics” prior to final layer should be treated cautiously; within‑model rank milestones are preferred.

Checklist (✓/✗/n.a.)
- RMS lens? ✓
- LayerNorm bias removed? ✓ (“not_needed_rms_model”)
- Entropy rise at unembed? ✓ (final entropy 0.370 bits at L42)
- FP32 un‑embed promoted? ✓ (“unembed_dtype: torch.float32”, “casting_to_fp32_before_unembed”)
- Punctuation / markup anchoring? ✓ (top‑1 '"' across L28–41)
- Copy‑reflex? ✓ (copy_collapse=True in L0–4; strict τ,δ satisfied)
- Grammatical filler anchoring? ✓ (early top‑1 includes {“simply”, “the”})

**Limitations & Data Quirks**
Final‑layer KL≈1 bit and warn_high_last_layer_kl=true imply final‑head calibration; prefer rank milestones and within‑model trends over absolute probabilities (001_layers_baseline/run-latest/output-gemma-2-9b.json:899–907,917). Raw‑vs‑norm lens sanity flags high artifact risk (max_kl_norm_vs_raw_bits≈12.91), so any pre‑final “early semantics” should be treated cautiously. Rest_mass after L_semantic is <0.3 and near zero; no evidence of precision loss.

**Model Fingerprint**
Gemma‑2‑9B: collapse at L0 on “simply”; semantics only at L42; final entropy 0.370 bits; punctuation plateau L28–41.

---
Produced by OpenAI GPT-5
