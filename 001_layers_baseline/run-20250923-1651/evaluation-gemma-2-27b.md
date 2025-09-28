## 1. Overview

This evaluation covers google/gemma-2-27b (27B). The probe tracks layer-by-layer next-token behavior with an RMSNorm lens, measuring copy collapse, entropy, rank milestones, KL to the final head, and cosine alignment to the final logits direction, with a negative control and an ablation (no-filler) variant.

## 2. Method sanity-check

Positional encoding and norm-lens usage are confirmed in diagnostics: “use_norm_lens: true … layer0_position_info: token_only_rotary_model; … first_block_ln1_type: RMSNorm” [001_layers_baseline/run-latest/output-gemma-2-27b.json:807–814]. The `context_prompt` ends with “called simply” and has no trailing space: “Give the city name only, plain text. The capital of Germany is called simply” [001_layers_baseline/run-latest/output-gemma-2-27b.json:810].

Copy-collapse flags are present in CSV (strict and soft); earliest strict `copy_collapse=True` occurs at layer 0 with top-1/2: “ simply” (p=0.999976), “ merely” (p≈7.52e-06) [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2]. Strict rule parameters and labels are recorded in JSON: `copy_thresh=0.95`, `copy_window_k=1`, `copy_match_level="id_subsequence"`; soft-detector config `copy_soft_config` uses `threshold=0.5`, `window_ks=[1,2,3]`, `extra_thresholds=[]`; `copy_flag_columns` mirrors these labels: ["copy_strict@0.95","copy_soft_k1@0.5","copy_soft_k2@0.5","copy_soft_k3@0.5"] [001_layers_baseline/run-latest/output-gemma-2-27b.json:833–849,846–847,1049–1056]. Gold-token alignment is ID-based and OK: `gold_alignment: "ok"` [001_layers_baseline/run-latest/output-gemma-2-27b.json:866].

Summary indices (diagnostics): `first_kl_below_0.5=null`, `first_kl_below_1.0=null`, `first_rank_le_1=46`, `first_rank_le_5=46`, `first_rank_le_10=46` (units: KL/entropy in bits) [001_layers_baseline/run-latest/output-gemma-2-27b.json:838–857]. Last-layer head calibration exists and is not ≈0: `kl_to_final_bits=1.1352`, `top1_agree=true`, `p_top1_lens=0.9841` vs `p_top1_model=0.4226`, `temp_est=2.6102`, `kl_after_temp_bits=0.5665`, `warn_high_last_layer_kl=true` [001_layers_baseline/run-latest/output-gemma-2-27b.json:875–902]. Prefer rank-based statements over absolute probabilities.

Lens sanity (raw vs norm): mode=`sample`; `lens_artifact_risk: "high"`, `max_kl_norm_vs_raw_bits=80.10`, `first_norm_only_semantic_layer=null` [001_layers_baseline/run-latest/output-gemma-2-27b.json:1073–1086]. Treat any pre-final “early semantics” cautiously and prefer rank milestones.

Negative control is present with `control_prompt` and `control_summary`: `first_control_margin_pos=0`, `max_control_margin=0.9910899` [001_layers_baseline/run-latest/output-gemma-2-27b.json:1091–1116]. Ablation summary exists and both variants appear in the CSV (orig/no_filler): `L_copy_orig=0`, `L_sem_orig=46`, `L_copy_nf=3`, `L_sem_nf=46`, `delta_L_copy=3`, `delta_L_sem=0` [001_layers_baseline/run-latest/output-gemma-2-27b.json:1083–1090; 001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:49–51]. For the main table below, rows are filtered to `prompt_id=pos`, `prompt_variant=orig`.

Copy-collapse flag check (strict): first `copy_collapse=True` at layer 0; token₁=“ simply” (p=0.999976), token₂=“ merely” (p≈7.52e-06) — ✓ rule satisfied [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2]. Soft copy: earliest `copy_soft_k1@0.5=True` at layer 0; `k2`/`k3` never fire in pos/orig.

## 3. Quantitative findings

Per-layer (pos, orig). Format: “L n — entropy X bits, top‑1 ‘token’”. Bold indicates the semantic layer (first `is_answer=True`).

| Layer | Entropy (bits) | Top‑1 |
|---:|---:|:---|
| 0 | 0.000 | ‘ simply’ |
| 1 | 8.758 | ‘’ |
| 2 | 8.764 | ‘’ |
| 3 | 0.886 | ‘ simply’ |
| 4 | 0.618 | ‘ simply’ |
| 5 | 8.520 | ‘๲’ |
| 6 | 8.553 | ‘’ |
| 7 | 8.547 | ‘’ |
| 8 | 8.529 | ‘’ |
| 9 | 8.524 | ‘𝆣’ |
| 10 | 8.345 | ‘ dieſem’ |
| 11 | 8.493 | ‘𝆣’ |
| 12 | 8.324 | ‘’ |
| 13 | 8.222 | ‘’ |
| 14 | 7.877 | ‘’ |
| 15 | 7.792 | ‘’ |
| 16 | 7.975 | ‘ dieſem’ |
| 17 | 7.786 | ‘ dieſem’ |
| 18 | 7.300 | ‘ſicht’ |
| 19 | 7.528 | ‘ dieſem’ |
| 20 | 6.210 | ‘ſicht’ |
| 21 | 6.456 | ‘ſicht’ |
| 22 | 6.378 | ‘ dieſem’ |
| 23 | 7.010 | ‘ dieſem’ |
| 24 | 6.497 | ‘ dieſem’ |
| 25 | 6.995 | ‘ dieſem’ |
| 26 | 6.220 | ‘ dieſem’ |
| 27 | 6.701 | ‘ dieſem’ |
| 28 | 7.140 | ‘ dieſem’ |
| 29 | 7.574 | ‘ dieſem’ |
| 30 | 7.330 | ‘ dieſem’ |
| 31 | 7.565 | ‘ dieſem’ |
| 32 | 8.874 | ‘ zuſammen’ |
| 33 | 6.945 | ‘ dieſem’ |
| 34 | 7.738 | ‘ dieſem’ |
| 35 | 7.651 | ‘ dieſem’ |
| 36 | 7.658 | ‘ dieſem’ |
| 37 | 7.572 | ‘ dieſem’ |
| 38 | 7.554 | ‘ パンチラ’ |
| 39 | 7.232 | ‘ dieſem’ |
| 40 | 8.711 | ‘ 展板’ |
| 41 | 7.082 | ‘ dieſem’ |
| 42 | 7.057 | ‘ dieſem’ |
| 43 | 7.089 | ‘ dieſem’ |
| 44 | 7.568 | ‘ dieſem’ |
| 45 | 7.141 | ‘ Geſch’ |
| **46** | **0.118** | **‘ Berlin’** |

Semantic layer: L_semantic = 46 (first `is_answer=True`), with “ Berlin” top‑1, p_top1=0.9841 and `answer_rank=1` [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48].

Control margin (JSON): `first_control_margin_pos=0`, `max_control_margin=0.9910899` [001_layers_baseline/run-latest/output-gemma-2-27b.json:1110–1116].

Ablation (no‑filler) [JSON]: L_copy_orig=0, L_sem_orig=46; L_copy_nf=3, L_sem_nf=46; ΔL_copy=+3, ΔL_sem=0 [001_layers_baseline/run-latest/output-gemma-2-27b.json:1083–1090].

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = 0.00050 − 0.11805 = −0.11755 [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2,48].

Soft ΔHₖ (bits): k=1 uses L=0 ⇒ same as strict (−0.11755). k=2,3 are null (no soft copy firing) [001_layers_baseline/run-latest/output-gemma-2-27b.json:854–861].

Confidence milestones (pure CSV): p_top1>0.30 at L=3; p_top1>0.60 at L=3; final-layer p_top1=0.9841 [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2–6,48].

Rank milestones (JSON): rank≤10 at L=46; rank≤5 at L=46; rank≤1 at L=46 [001_layers_baseline/run-latest/output-gemma-2-27b.json:838–845].

KL milestones and final-head note: `first_kl_below_1.0=null`, `first_kl_below_0.5=null`; final KL is not ≈0 (1.1352 bits), consistent with family head calibration issues [001_layers_baseline/run-latest/output-gemma-2-27b.json:836–843,875–902]. Prefer ranks over final absolute probabilities.

Cosine milestones (pure CSV): first `cos_to_final≥0.2` at L=1; `≥0.4` at L=46; `≥0.6` at L=46; final `cos_to_final=0.99939` [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48].

Prism Sidecar Analysis
- Presence: prism_summary.compatible=true, present=true, k=512, layers=[embed,10,22,33] [001_layers_baseline/run-latest/output-gemma-2-27b.json:817–833].
- Early-depth stability (KL to final): baseline vs prism at sampled depths (L=0,11,23,34):
  - Baseline KL: 16.85, 41.85, 43.15, 42.51 bits; Prism KL: 19.43, 19.43, 19.42, 19.43 bits [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv; 001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token-prism.csv]. No helpful early drop.
  - Final KL: baseline 1.135; prism 20.172 bits (miscalibrated) [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48; …-prism.csv].
- Rank milestones (prism): never reaches rank≤10/5/1 (null) vs baseline all at L=46 [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token-prism.csv].
- Top‑1 agreement: baseline disagrees with final at early depths; prism also disagrees and remains very low-confidence (p_top1≈1.6e‑5 at L∈{0,11,23,34}) [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token-prism.csv].
- Cosine drift: prism cos_to_final is slightly negative (≈−0.09..−0.11) at sampled depths, indicating a poor alignment vs the final direction [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token-prism.csv].
- Copy flags: baseline strict copy at L0 flips to False under Prism (mass spread), plausibly from the transform reducing concentrated copy mass [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2; …-prism.csv].
- Verdict: Regressive (higher KL, no earlier rank milestones, degraded calibration).

## 4. Qualitative patterns & anomalies

The model exhibits a clear copy-reflex at layer 0: “ simply” dominates (p≈0.99998), satisfying the strict copy rule, while soft copy (k=1) also fires at L0 [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2]. Despite early cosine alignment (`cos≈0.33` by mid‑stack), KL to final remains very high until the last layer, indicating “early direction, late calibration.” Final‑head calibration issues are explicit for Gemma‑2‑27B: `top1_agree=true` but `p_top1_lens=0.9841` vs `p_top1_model=0.4226`, `temp_est=2.610`, `kl_after_temp_bits=0.5665`, and `warn_high_last_layer_kl=true` [001_layers_baseline/run-latest/output-gemma-2-27b.json:875–902].

Negative control: “Berlin is the capital of” produces a clean country prediction; top‑5 are “ Germany” (0.8676), “ the” (0.0650), “ and” (0.0065), “ a” (0.0062), “ Europe” (0.0056) — no “Berlin” appears in the top‑5 [001_layers_baseline/run-latest/output-gemma-2-27b.json:13–31].

Records and important‑word trajectory (SCRIPT IMPORTANT_WORDS = ["Germany","Berlin","capital","Answer","word","simply"]). Within the context, “ Germany” is strongly salient early at its token position (e.g., L0 pos=13 top‑1 “ Germany”, p≈0.4365 [001_layers_baseline/run-latest/output-gemma-2-27b-records.csv:15]; L3 pos=13 top‑1 “ Germany”, p≈0.5790 [001_layers_baseline/run-latest/output-gemma-2-27b-records.csv:66]). At NEXT (pos=16), “ Berlin” only becomes top‑1 at the final layer: “ Berlin, 0.9841” (L46, pos=16) [001_layers_baseline/run-latest/output-gemma-2-27b-records.csv:806]. Semantically close city distractors appear only as minor mass in the final prediction (e.g., “ Munich”, p≈0.00581; also “ BERLIN/berlin” variants) [001_layers_baseline/run-latest/output-gemma-2-27b.json:957–981].

Instructional variations (test prompts): removing the filler cue or “simply” in phrasing still yields high-confidence “ Berlin” (e.g., “Germany’s capital city is called”, p≈0.598; “The capital city of Germany is named”, p≈0.376) [001_layers_baseline/run-latest/output-gemma-2-27b.json:214–231,240–254]. While these probe only final predictions (not layer-wise collapse), the ablation run shows ΔL_sem=0, ΔL_copy=+3, indicating stylistic filler mainly affects early copy behavior rather than semantic emergence [001_layers_baseline/run-latest/output-gemma-2-27b.json:1083–1090].

Rest‑mass sanity: no top‑k coverage issues after semantics; at L=46, `rest_mass≈2.0e‑7` [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48].

Rotation vs amplification: KL decreases sharply only at the end (still 1.135 bits at final), while `answer_rank` jumps to 1 and `p_answer`/`p_top1` surge; cosine rises modestly early (≈0.33 mid‑stack) before snapping to ≈1.0 at L=46. This is consistent with “early direction, late calibration.” Given `warn_high_last_layer_kl=true`, prefer rank-based milestones for cross-family claims [001_layers_baseline/run-latest/output-gemma-2-27b.json:875–902,1073–1086].

Head calibration (final): `temp_est=2.610` reduces KL to 0.5665 bits (`kl_after_temp_bits`) but still not ≈0; `cfg_transform` fields are present but null for this run [001_layers_baseline/run-latest/output-gemma-2-27b.json:875–902].

Lens sanity: raw-vs-norm check (mode=sample) flags `lens_artifact_risk="high"` with `max_kl_norm_vs_raw_bits=80.10`; no `first_norm_only_semantic_layer` [001_layers_baseline/run-latest/output-gemma-2-27b.json:1079–1086]. Treat any apparent early semantics cautiously; rely on ranks and within-model cosine trends.

Temperature robustness (final head): at T=0.1, “ Berlin” rank 1 (p≈0.9898; entropy≈0.082 bits); at T=2.0, “ Berlin” rank 1 (p≈0.0492; entropy≈12.631 bits) [001_layers_baseline/run-latest/output-gemma-2-27b.json:669–697,736–760].

Checklist
- RMS lens? ✓ (“RMSNorm” in first/final LN) [001_layers_baseline/run-latest/output-gemma-2-27b.json:811–814]
- LayerNorm bias removed? ✓ (“not_needed_rms_model”) [001_layers_baseline/run-latest/output-gemma-2-27b.json:813]
- Entropy rise at unembed? ✓ (pure L46≈0.118 → final≈2.886 bits) [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48; 001_layers_baseline/run-latest/output-gemma-2-27b.json:922]
- FP32 un‑embed promoted? ✓ (`unembed_dtype:"torch.float32"`; `use_fp32_unembed:false`) [001_layers_baseline/run-latest/output-gemma-2-27b.json:808–809]
- Punctuation / markup anchoring? ✓ (quotes/punctuation in finals) [001_layers_baseline/run-latest/output-gemma-2-27b.json:944–961]
- Copy‑reflex? ✓ (copy_collapse=True in L0) [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2]
- Grammatical filler anchoring? ✗ for {is,the,a,of} as top‑1 in L0–5 (dominant cue is “ simply”).

## 5. Limitations & data quirks

- Final‑layer KL is not ≈0 (`kl_to_final_bits=1.135`), and `warn_high_last_layer_kl=true`; treat final probabilities as family‑specific calibration and rely on rank milestones and within‑model trends [001_layers_baseline/run-latest/output-gemma-2-27b.json:875–902].
- Raw‑vs‑norm lens `mode=sample` with `lens_artifact_risk="high"` and `max_kl_norm_vs_raw_bits=80.10`; treat early‑depth findings as sanity checks rather than exhaustive [001_layers_baseline/run-latest/output-gemma-2-27b.json:1079–1086].
- Rest_mass is low after semantics (≈2e‑7 at L=46); no evidence of precision loss from top‑k coverage in this run [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48].

## 6. Model fingerprint (one sentence)

Gemma‑2‑27B: collapse at L 46; final entropy ≈0.118 bits (lens) with “ Berlin” top‑1; strong early copy of “ simply” at L 0; final‑head calibration gap persists (KL≈1.135 bits).

---
Produced by OpenAI GPT-5 

