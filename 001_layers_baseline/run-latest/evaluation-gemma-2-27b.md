**Overview**

- Model: google/gemma-2-27b (pre_norm; 46 layers). The probe analyzes the first unseen token after the prompt and records per-layer entropy, calibration (KL to final), copy/collapse flags, cosine-to-final geometry, and answer rank. Artifacts: `output-gemma-2-27b.json`, `*-pure-next-token.csv`, `*-records.csv`, plus Prism/Tuned sidecars.

**Method Sanity‑Check**

The context prompt uses the intended “called simply” ending with no trailing space: “Give the city name only, plain text. The capital of Germany is called simply” (diagnostics.context_prompt) [001_layers_baseline/run-latest/output-gemma-2-27b.json:716]. The run applies the RMS norm lens with fp32 unembed and rotary token‑only pos‑embed handling: “use_norm_lens: true; unembed_dtype: torch.float32; layer0_position_info: token_only_rotary_model” [001_layers_baseline/run-latest/output-gemma-2-27b.json:784].

Strict copy detector configuration is ID‑level contiguous subsequence with τ=0.95, k=1, δ=0.10 (script defaults), and soft‑copy uses τ_soft=0.5 with window_ks={1,2,3} as recorded in `copy_soft_config` and mirrored in flags: “copy_soft_config: { threshold: 0.5, window_ks: [1,2,3], extra_thresholds: [] }” and “copy_flag_columns: [copy_strict@0.95, copy_soft_k1@0.5, copy_soft_k2@0.5, copy_soft_k3@0.5]” [001_layers_baseline/run-latest/output-gemma-2-27b.json:1358,1387]. Gold alignment is OK: `gold_answer = { string: "Berlin", pieces: ["▁Berlin"], first_id: 12514 }` and `diagnostics.gold_alignment: "ok"` [001_layers_baseline/run-latest/output-gemma-2-27b.json:1775,1348]. Negative control is present with summary margins: `control_summary = { first_control_margin_pos: 0, max_control_margin: 0.9910899400710897 }` [001_layers_baseline/run-latest/output-gemma-2-27b.json:1416]. Ablation is present: `ablation_summary = { L_copy_orig: 0, L_sem_orig: 46, L_copy_nf: 3, L_sem_nf: 46, delta_L_copy: 3, delta_L_sem: 0 }` [001_layers_baseline/run-latest/output-gemma-2-27b.json:1393].

Copy‑collapse flags fire strictly at layer 0: “copy_collapse=True; copy_strict@0.95=True; copy_soft_k1@0.5=True” [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2]. ✓ rule satisfied (token is the prompt word “ simply”). Soft‑copy earliest hit: layer 0 (k=1).

Summary indices (diagnostics): `first_kl_below_0.5 = null`, `first_kl_below_1.0 = null`, `first_rank_le_1 = 46`, `first_rank_le_5 = 46`, `first_rank_le_10 = 46` [001_layers_baseline/run-latest/output-gemma-2-27b.json:904]. Units for KL/entropy are bits; the pure CSV includes `teacher_entropy_bits` for drift. Final‑head calibration check shows non‑zero last‑layer KL and a temperature mismatch: `kl_to_final_bits=1.1352`, `top1_agree=true`, `p_top1_lens=0.9841` vs `p_top1_model=0.4226`, `temp_est=2.6102`, `kl_after_temp_bits=0.5665`, `warn_high_last_layer_kl=true` [001_layers_baseline/run-latest/output-gemma-2-27b.json:966]. Treat final probabilities cautiously (known Gemma pattern).

Lens sanity (raw vs norm): sampled `raw_lens_check.summary` reports `lens_artifact_risk: "high"` with `max_kl_norm_vs_raw_bits=80.10` and `first_norm_only_semantic_layer=null` [001_layers_baseline/run-latest/output-gemma-2-27b.json:1361]. Caution about early semantics; prefer rank milestones.

**Quantitative Findings**

- Table (pos, orig). One row per layer: L i — entropy (bits), top‑1 token. Bold = semantic layer (first `is_answer=True`).

| Layer | Entropy (bits) | Top‑1 |
|---|---:|---|
| L 0 | 0.000 | ' simply' |
| L 1 | 8.758 | '' |
| L 2 | 8.764 | '' |
| L 3 | 0.886 | ' simply' |
| L 4 | 0.618 | ' simply' |
| L 5 | 8.520 | '๲' |
| L 6 | 8.553 | '' |
| L 7 | 8.547 | '' |
| L 8 | 8.529 | '' |
| L 9 | 8.524 | '𝆣' |
| L 10 | 8.345 | ' dieſem' |
| L 11 | 8.493 | '𝆣' |
| L 12 | 8.324 | '' |
| L 13 | 8.222 | '' |
| L 14 | 7.877 | '' |
| L 15 | 7.792 | '' |
| L 16 | 7.975 | ' dieſem' |
| L 17 | 7.786 | ' dieſem' |
| L 18 | 7.300 | 'ſicht' |
| L 19 | 7.528 | ' dieſem' |
| L 20 | 6.210 | 'ſicht' |
| L 21 | 6.456 | 'ſicht' |
| L 22 | 6.378 | ' dieſem' |
| L 23 | 7.010 | ' dieſem' |
| L 24 | 6.497 | ' dieſem' |
| L 25 | 6.995 | ' dieſem' |
| L 26 | 6.220 | ' dieſem' |
| L 27 | 6.701 | ' dieſem' |
| L 28 | 7.140 | ' dieſem' |
| L 29 | 7.574 | ' dieſem' |
| L 30 | 7.330 | ' dieſem' |
| L 31 | 7.565 | ' dieſem' |
| L 32 | 8.874 | ' zuſammen' |
| L 33 | 6.945 | ' dieſem' |
| L 34 | 7.738 | ' dieſem' |
| L 35 | 7.651 | ' dieſem' |
| L 36 | 7.658 | ' dieſem' |
| L 37 | 7.572 | ' dieſem' |
| L 38 | 7.554 | ' パンチラ' |
| L 39 | 7.232 | ' dieſem' |
| L 40 | 8.711 | ' 展板' |
| L 41 | 7.082 | ' dieſem' |
| L 42 | 7.057 | ' dieſem' |
| L 43 | 7.089 | ' dieſem' |
| L 44 | 7.568 | ' dieſem' |
| L 45 | 7.141 | ' Geſch' |
| **L 46** | 0.118 | ' Berlin' |

Control margin (JSON): `first_control_margin_pos = 0`, `max_control_margin = 0.9911` [001_layers_baseline/run-latest/output-gemma-2-27b.json:1416].

Ablation (no‑filler): `L_copy_orig = 0`, `L_sem_orig = 46`, `L_copy_nf = 3`, `L_sem_nf = 46`, so `ΔL_copy = +3`, `ΔL_sem = 0` [001_layers_baseline/run-latest/output-gemma-2-27b.json:1393].

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = 0.00050 − 0.11805 = −0.1176 [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2,48].
Soft ΔH₁ (k=1) = entropy(L_copy_soft[1]) − entropy(L_semantic) = 0.00050 − 0.11805 = −0.1176 (soft k=2,3: null) [001_layers_baseline/run-latest/output-gemma-2-27b.json:1350; 001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2,48].

Confidence milestones (pure CSV): p_top1 > 0.30 at layer 0; p_top1 > 0.60 at layer 0; final p_top1 = 0.9841 [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2,48].

Rank milestones (diagnostics): rank ≤ 10 at layer 46; rank ≤ 5 at layer 46; rank ≤ 1 at layer 46 [001_layers_baseline/run-latest/output-gemma-2-27b.json:904].

KL milestones (diagnostics): first_kl_below_1.0 = null; first_kl_below_0.5 = null; KL decreases to a non‑zero value at final (1.135 bits) [001_layers_baseline/run-latest/output-gemma-2-27b.json:904,966]. Final‑head calibration warning present; prefer rank statements across families.

Cosine milestones (pure CSV): cos_to_final ≥ 0.2 at L 1; ≥ 0.4 at L 46; ≥ 0.6 at L 46; final cos_to_final = 0.9994 [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48].

Prism Sidecar Analysis

- Presence: Prism artifacts compatible=true [001_layers_baseline/run-latest/output-gemma-2-27b.json:725].
- Early‑depth stability (KL vs final, baseline→Prism): L0 16.85→19.43; L≈25% (L11) 41.85→19.43; L≈50% (L23) 43.15→19.42; L≈75% (L34) 42.51→19.43 bits [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2; 001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token-prism.csv:1].
- Rank milestones (Prism): no rank≤10/5/1 before final; at L46 `answer_rank=165699` (no semantics) [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token-prism.csv:48].
- Top‑1 agreement at sampled depths: top‑1 differs at all sampled layers (e.g., L0 ' simply' vs ' assuredly') [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2; prism:1].
- Cosine drift: Prism cos_to_final is negative at early/mid layers (e.g., L0 −0.089, L23 −0.095) vs baseline positive (~0.33), indicating a different projection geometry; no earlier stabilization [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token-prism.csv:1].
- Copy flags: baseline strict copy fires at L0; Prism `copy_collapse=False` at L0, plausibly due to Prism transform changing the local top‑1 neighborhood and probabilities [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2; prism:1].
- Verdict: Regressive for rank/semantics (no earlier `first_rank≤1` and altered surface behavior), though it substantially lowers KL at mid‑depths (~−22 bits).

**Qualitative Patterns & Anomalies**

Negative control (“Berlin is the capital of”): top‑5 are “ Germany (0.8676), the (0.0650), and (0.0065), a (0.0062), Europe (0.0056)” — Berlin does not appear (no leakage) [001_layers_baseline/run-latest/output-gemma-2-27b.json:6]. For stylistic rephrasings (e.g., “Germany’s capital city is called simply”), Berlin is top‑1 with p≈0.52–0.62 [001_layers_baseline/run-latest/output-gemma-2-27b.json:208,231,287].

Important‑word trajectory (records): early positions are dominated by literal/punctuation/filler tokens (e.g., pos 2: “ the” p≈0.9999; pos 3: “ city” p≈0.996; pos 12: “ of” p≈0.99999) [001_layers_baseline/run-latest/output-gemma-2-27b-records.csv:3,4,13]. At NEXT, L0 predicts the copy token “ simply” with p=0.99998, while Berlin only becomes top‑1 at the final head (L46, p=0.9841) [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2,48]. Mid‑stack NEXT top‑1s frequently include non‑semantic artifacts (e.g., “ dieſem”, symbols, multilingual tokens), with `cos_to_final` rising gradually (≥0.2 by L1) while KL stays very high, indicating “early direction, late calibration.”

One‑word instruction ablation: removing “simply” delays/relocates copy from L0→L3 (ΔL_copy=+3) but leaves semantics unchanged (L_semantic=46 both) [001_layers_baseline/run-latest/output-gemma-2-27b.json:1393]. This suggests stylistic anchoring affects surface reflexes, not answer emergence.

Rest‑mass sanity: For NEXT, rest_mass is tiny when the model is confident (L0 ~4.8e−06; L46 ~2.0e−07), and large during high‑entropy mid‑layers (e.g., L29 rest_mass ~0.86), falling again near the final [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2,47,48]. No post‑final spikes observed.

Rotation vs amplification: KL to final remains large across most depths and is not ≈0 at the final head (1.135 bits). Yet `cos_to_final` steadily increases to 0.9994 at L46, and `answer_rank` flips to 1 only at L46, consistent with meaningful direction emerging late relative to calibration. Final‑head calibration warning: `temp_est=2.6102`, `kl_after_temp=0.5665` [001_layers_baseline/run-latest/output-gemma-2-27b.json:966]. Given `raw_lens_check.summary.lens_artifact_risk="high"` with `max_kl_norm_vs_raw_bits=80.10` and `first_norm_only_semantic_layer=null` [001_layers_baseline/run-latest/output-gemma-2-27b.json:1361], treat any pre‑final “semantics” as lens‑sensitive and favor ranks.

Temperature robustness: At T=0.1, Berlin rank 1 with p=0.9898 and entropy=0.082 bits; at T=2.0, Berlin still rank 1 with p=0.0492 and entropy=12.631 bits [001_layers_baseline/run-latest/output-gemma-2-27b.json:670,737].

Checklist
- RMS lens? ✓ [001_layers_baseline/run-latest/output-gemma-2-27b.json:778]
- LayerNorm bias removed? n.a. (RMS) [001_layers_baseline/run-latest/output-gemma-2-27b.json:776]
- Entropy rise at unembed? ✓ (mid‑stack entropies 6–9 bits; final 0.118) [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:33,48]
- FP32 un‑embed promoted? ✓ (`unembed_dtype: torch.float32`) [001_layers_baseline/run-latest/output-gemma-2-27b.json:772]
- Punctuation / markup anchoring? ✓ (multiple filler/punct top‑1s mid‑stack) [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:29–41]
- Copy‑reflex? ✓ (strict/soft at L0) [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2]
- Grammatical filler anchoring? ✓ (“is”, “the”, punctuation dominate early positions) [001_layers_baseline/run-latest/output-gemma-2-27b-records.csv:2,9,14]

**Limitations & Data Quirks**

- Final‑head calibration is off for Gemma‑2‑27B (`warn_high_last_layer_kl=true; kl_to_final_bits=1.135; temp_est=2.610; kl_after_temp=0.567`), so treat final probabilities as family‑specific; rely on rank and within‑model trends [001_layers_baseline/run-latest/output-gemma-2-27b.json:966].
- `raw_lens_check` is `mode: sample` and flags `lens_artifact_risk: high` with very large `max_kl_norm_vs_raw_bits` (80.10), so early “semantics” may be lens‑induced; cross‑lens differences are advisory [001_layers_baseline/run-latest/output-gemma-2-27b.json:1325].
- Surface‑mass metrics depend on tokenizer; absolute masses are not comparable across families; use within‑model trends.

**Model Fingerprint**

“Gemma‑2‑27B: collapse at L 46; final entropy 0.118 bits; ‘Berlin’ appears only at the final head.”

---
Produced by OpenAI GPT-5 

