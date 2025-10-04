# Evaluation Report: google/gemma-2-27b

*Run executed on: 2025-09-30 23:57:21*
**Overview**

Google Gemma‑2‑27B (46 layers; pre‑norm) was probed layer‑by‑layer on the “Germany → Berlin” prompt. The run captures copy‑reflex at the input token and a very late semantic collapse at the final layer. Confirmed semantics (tuned corroboration) occur at L 46.

**Method Sanity‑Check**

Diagnostics indicate the intended norm lens and rotary positional encoding path were used: “use_norm_lens: true … layer0_position_info: token_only_rotary_model”  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1080–1090, 1097–1103]. The context prompt ends exactly with “called simply” (no trailing space): “context_prompt … The capital of Germany is called simply”  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1099–1103].

Copy detection configuration and outputs are present: “copy_thresh: 0.95, copy_window_k: 1, copy_match_level: id_subsequence” with strict sweep entries and soft‑copy config/window_ks recorded  [001_layers_baseline/run-latest/output-gemma-2-27b.json:941–951, 966–983, 994–1012]. The JSON lists `copy_flag_columns` that mirror the CSV flags  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1630–1638]. Gold‑token alignment is OK  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1083] and ablation summary is present with both `orig` and `no_filler` variants  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1639–1646]. For the main table and milestones below, rows are filtered to `prompt_id = pos`, `prompt_variant = orig` in the pure CSV.

Measurement guidance requests rank‑first reporting: “prefer_ranks: true, suppress_abs_probs: true … reasons: [‘warn_high_last_layer_kl’, ‘norm_only_semantics_window’, ‘high_lens_artifact_risk’, ‘high_lens_artifact_score’] … preferred_lens_for_reporting: "norm" … use_confirmed_semantics: true”  [001_layers_baseline/run-latest/output-gemma-2-27b.json:2150–2162].

Raw‑vs‑Norm window and full checks flag norm‑only semantics and high lens artefact risk: “center_layers: [0, 46], radius: 4 … norm_only_semantics_layers: [46] … max_kl_norm_vs_raw_bits_window: 99.54”  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1047–1069]; and in full: “pct_layers_kl_ge_1.0: 0.9787 … n_norm_only_semantics_layers: 1 … earliest_norm_only_semantic: 46 … max_kl_norm_vs_raw_bits: 99.54 … tier: "high"”  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1071–1081]. Treat early semantics cautiously and prefer rank milestones.

Strict copy thresholds sweep shows earliest `L_copy_strict` at τ∈{0.70,0.95} equals 0 (stability: “mixed”; norm_only_flags false)  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1001–1019]. The pure CSV confirms strict copy at L0 with contiguous ID‑level subsequence: “layer 0 … copy_collapse = True … copy_strict@0.95 = True … copy_soft_k1@0.5 = True”  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2].

Last‑layer head calibration is non‑negligible (Gemma family pattern): “kl_to_final_bits: 1.1352 … top1_agree: true; p_top1_lens: 0.9841 vs p_top1_model: 0.4226; temp_est: 2.61; kl_after_temp_bits: 0.5665; warn_high_last_layer_kl: true”  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1084–1103]. The pure CSV’s final row corroborates non‑zero final KL  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48]. Prefer ranks over absolute probabilities across families.

Prism sidecar is compatible but diagnostic‑only: “k: 512; layers: [embed, 10, 22, 33]; KL delta at percentiles ≈ +22.6/+23.7/+23.1 bits (baseline minus Prism)” with no rank milestones achieved by Prism  [001_layers_baseline/run-latest/output-gemma-2-27b.json:825–870].

Confirmed semantics are present and preferred for reporting: “L_semantic_confirmed: 46 (confirmed_source: "tuned")”  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1451–1458].

**Quantitative Findings**

Layer‑by‑layer (pos/orig; entropy in bits; top‑1 token). Bold indicates the confirmed semantic layer.

| Layer | Entropy | Top‑1 |
|---:|---:|:---|
| 0 | 0.00050 | simply |
| 1 | 8.75823 |  |
| 2 | 8.76449 |  |
| 3 | 0.88567 | simply |
| 4 | 0.61827 | simply |
| 5 | 8.52026 | ๲ |
| 6 | 8.55309 |  |
| 7 | 8.54697 |  |
| 8 | 8.52874 |  |
| 9 | 8.52380 | 𝆣 |
| 10 | 8.34524 | dieſem |
| 11 | 8.49276 | 𝆣 |
| 12 | 8.32442 |  |
| 13 | 8.22249 |  |
| 14 | 7.87661 |  |
| 15 | 7.79248 |  |
| 16 | 7.97484 | dieſem |
| 17 | 7.78555 | dieſem |
| 18 | 7.29993 | ſicht |
| 19 | 7.52777 | dieſem |
| 20 | 6.20999 | ſicht |
| 21 | 6.45600 | ſicht |
| 22 | 6.37844 | dieſem |
| 23 | 7.01041 | dieſem |
| 24 | 6.49704 | dieſem |
| 25 | 6.99488 | dieſem |
| 26 | 6.21981 | dieſem |
| 27 | 6.70072 | dieſem |
| 28 | 7.14012 | dieſem |
| 29 | 7.57415 | dieſem |
| 30 | 7.33021 | dieſem |
| 31 | 7.56517 | dieſem |
| 32 | 8.87356 | zuſammen |
| 33 | 6.94474 | dieſem |
| 34 | 7.73832 | dieſem |
| 35 | 7.65066 | dieſem |
| 36 | 7.65774 | dieſem |
| 37 | 7.57239 | dieſem |
| 38 | 7.55355 | パンチラ |
| 39 | 7.23244 | dieſem |
| 40 | 8.71052 | 展板 |
| 41 | 7.08169 | dieſem |
| 42 | 7.05652 | dieſem |
| 43 | 7.08893 | dieſem |
| 44 | 7.56833 | dieſem |
| 45 | 7.14057 | Geſch |
| **46** | **0.11805** | **Berlin** |

Control margin (ctl JSON): first_control_margin_pos = 0; max_control_margin = 0.9911  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1662–1665].

Ablation (no‑filler): L_copy_orig = 0; L_sem_orig = 46; L_copy_nf = 3; L_sem_nf = 46; ΔL_copy = 3; ΔL_sem = 0  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1639–1646].

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = 0.00050 − 0.11805 = −0.11755  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2, 48]. Soft ΔH₁ (k=1) = −0.11755 (L_copy_soft[1] = 0)  [001_layers_baseline/run-latest/output-gemma-2-27b.json:956–963].

Confidence milestones (generic top‑1 from pure CSV): p_top1 > 0.30 at L 0; p_top1 > 0.60 at L 0; final‑layer p_top1 = 0.9841 (Berlin)  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2, 48].

Rank milestones (norm lens; diagnostics): rank ≤ 10 at L 46; rank ≤ 5 at L 46; rank ≤ 1 at L 46  [001_layers_baseline/run-latest/output-gemma-2-27b.json:947–949].

KL milestones: first_kl_below_1.0 = null; first_kl_below_0.5 = null; final KL ≈ 1.135 bits (non‑zero), decreasing is not the primary signal here  [001_layers_baseline/run-latest/output-gemma-2-27b.json:945–947, 1084–1087].

Cosine milestones (norm): first cos_to_final ≥ 0.2 at L 1; ≥ 0.4 at L 46; ≥ 0.6 at L 46; final cos_to_final = 0.9994  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1032–1037; 001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48].

Copy robustness (strict sweep): stability = “mixed”; earliest L_copy_strict at τ=0.70 → 0; at τ=0.95 → 0; norm_only_flags all false  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1001–1019].

Prism sidecar analysis: compatible=true, but rank milestones not achieved (null). Early‑depth KL is much lower under Prism (baseline p25/p50/p75 ≈ 42.01/43.15/42.51 bits vs Prism ≈ 19.43/19.42/19.43; deltas ≈ 22.57/23.73/23.08) with no earlier ranks and no copy flips to strict True  [001_layers_baseline/run-latest/output-gemma-2-27b.json:825–870]. Verdict: Regressive for semantics (KL lower but no rank‑1), consistent with Prism’s diagnostic role.

**Qualitative Patterns & Anomalies**

The model exhibits a strong copy‑reflex on the adverb “simply” at L0 (strict τ=0.95) with soft‑copy at early layers, e.g., “layer 0 … copy_collapse = True … copy_strict@0.95 = True … copy_soft_k1@0.5 = True”  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2]. Berlin only becomes rank‑1 at the final layer with high within‑lens confidence but a non‑zero final KL vs the model’s head: “L46 … p_answer = 0.9841 … kl_to_final_bits = 1.1352”  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48]. This is a family‑typical final‑head calibration gap for Gemma; follow measurement guidance and prefer rank‑based claims.

Negative control (“Berlin is the capital of”): the model correctly predicts the country without leakage; top‑5 begins with “Germany (0.868), the (0.065), and (0.0065), a (0.0062), Europe (0.0056)”  [001_layers_baseline/run-latest/output-gemma-2-27b.json:11–20]. No “Berlin” appears, so no semantic leakage.

Important‑word trajectory (records CSV): early layers are dominated by filler/copy (“simply” is pervasive up to mid‑stack) and later by odd orthographic tokens (“dieſem”, “ſicht”), before collapsing to “Berlin” at L46: “… L10 top‑1 ‘dieſem’ … L20 ‘ſicht’ … L45 ‘Geſch’ … L46 ‘Berlin’ (0.984)”  [001_layers_baseline/run-latest/output-gemma-2-27b-records.csv:188, 358, 528, 806]. This suggests surface‑form anchoring and orthography‑heavy features prior to semantic collapse.

Prompt ablation (no “simply”) delays strict copy from L0→L3 without affecting semantics (L_sem remains 46): “L_copy_orig: 0 … L_copy_nf: 3 … L_sem_nf: 46 … ΔL_copy = 3; ΔL_sem = 0”  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1639–1646]. This supports the interpretation of grammatical‑style anchoring rather than changed semantics.

Rest‑mass sanity: final rest_mass is ≈ 2.0e‑07 at L46, consistent with concentrated mass after collapse  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48].

Rotation vs amplification: cosine to final rises early (≥0.2 by L1) while KL remains very high until the end (first_kl_below_1.0 = null)  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1032–1037, 945–947]. This is “early direction, late calibration”; final‑head calibration remains non‑trivial (warn_high_last_layer_kl=true; temp_est≈2.61; kl_after_temp_bits≈0.567)  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1084–1103].

Lens sanity: raw‑vs‑norm indicates high artefact risk and norm‑only semantics at L46 (“max_kl_norm_vs_raw_bits: 99.54 … earliest_norm_only_semantic: 46 … tier: high”)  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1071–1081]. Accordingly, statements emphasize ranks and within‑model trends.

Temperature robustness (teacher‑head): at T=0.1, Berlin rank 1 (p≈0.990; entropy≈0.082); at T=2.0, Berlin remains rank‑1 with much lower margin (p≈0.049; entropy≈12.63)  [001_layers_baseline/run-latest/output-gemma-2-27b.json:669–744].

Checklist: RMS lens ✓; LayerNorm bias removed n/a (RMS model)  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1087–1091]; Entropy rise at unembed ✓ (mid‑stack entropies ≫ teacher entropy)  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:10–32]; FP32 un‑embed promoted ✓ (“unembed_dtype": "torch.float32")  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1086–1088]; Punctuation/filler anchoring ✓ (“simply” early; orthographic tokens mid)  [001_layers_baseline/run-latest/output-gemma-2-27b-records.csv:69, 188]; Copy‑reflex ✓ (strict at L0; soft at L0–4)  [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2, 5–6]; Preferred lens honored ✓ (norm; confirmed semantics used)  [001_layers_baseline/run-latest/output-gemma-2-27b.json:2150–2162]; Confirmed semantics reported ✓ (source: tuned)  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1451–1458]; Full dual‑lens metrics cited ✓ (pct_layers_kl_ge_1.0, n_norm_only_semantics_layers, earliest_norm_only_semantic, tier)  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1071–1081]; Tuned‑lens attribution ✓ (ΔKL_tuned/ΔKL_temp/ΔKL_rot at 25/50/75%) shows negative ΔKL_rot  [001_layers_baseline/run-latest/output-gemma-2-27b.json:2129–2147].

**Limitations & Data Quirks**

Final KL is not ≈0 (warn_high_last_layer_kl=true); treat final probabilities as family‑specific calibration and prefer rank‑based milestones  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1084–1103]. Raw‑vs‑norm artefact risk is high with norm‑only semantics at L46; rely on confirmed semantics and rank milestones for onset  [001_layers_baseline/run-latest/output-gemma-2-27b.json:1047–1081]. Surface‑mass metrics depend on tokenizer idiosyncrasies; cross‑model mass comparisons are not advised.

**Model Fingerprint**

Gemma‑2‑27B: collapse at L 46; final entropy ≈ 0.118 bits; “Berlin” first appears rank 1 only at the last layer.

---
Produced by OpenAI GPT-5 
