**Overview**
- Model: Qwen/Qwen3-8B (pre_norm; 36 layers). The probe runs a layer-by-layer logit-lens pass and captures entropy, copy flags, rank/KL milestones, cosine-to-final, and calibration diagnostics. Confirmed norm lens with rotary positional handling and FP32 unembedding during analysis.
- Scope: Pos prompt (“The capital of Germany … simply”), control prompt (France→Paris), stylistic ablation (no “simply”), and optional Prism sidecar.

**Method Sanity-Check**
The JSON confirms the norm lens and rotary handling: “use_norm_lens: true” and “layer0_position_info: token_only_rotary_model” (001_layers_baseline/run-latest/output-Qwen3-8B.json:807,816). The context prompt ends exactly with “called simply” (001_layers_baseline/run-latest/output-Qwen3-8B.json:4). Copy detector configuration is present and consistent with the strict ID-level rule (τ=0.95, δ=0.10, k=1) and soft windows: “copy_thresh: 0.95”, “copy_window_k: 1”, “copy_match_level: id_subsequence”, and “copy_soft_config { threshold: 0.5, window_ks: [1,2,3] }” (001_layers_baseline/run-latest/output-Qwen3-8B.json:846–848,833–836). The CSV header includes the expected flag columns and rest_mass (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:1). Gold-token alignment is ID-based and ok (001_layers_baseline/run-latest/output-Qwen3-8B.json:1110–1120). Ablation summary exists with both orig and no_filler rows present in the pure CSV (001_layers_baseline/run-latest/output-Qwen3-8B.json:1083–1090; 001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:77–78). Control prompt and summary are present (001_layers_baseline/run-latest/output-Qwen3-8B.json:1091–1109).

Summary indices (bits/ranks) are present: first_kl_below_0.5=36, first_kl_below_1.0=36, first_rank_le_1=31, first_rank_le_5=29, first_rank_le_10=29 (001_layers_baseline/run-latest/output-Qwen3-8B.json:849–853). Last-layer head calibration is good: kl_to_final_bits=0.0, top1_agree=true, p_top1_lens=p_top1_model=0.4334, temp_est=1.0 (001_layers_baseline/run-latest/output-Qwen3-8B.json:899–917). Lens sanity: mode=sample; lens_artifact_risk=high with max_kl_norm_vs_raw_bits=13.60 and no norm-only semantic layer (001_layers_baseline/run-latest/output-Qwen3-8B.json:1015–1075). Copy-collapse (strict) never fires in layers 0–3; soft flags do not fire either (pure CSV; see Section 4 checklist). Copy flag columns match JSON: [copy_strict@0.95, copy_soft_k1@0.5, k2, k3] (001_layers_baseline/run-latest/output-Qwen3-8B.json:1077–1081).

Copy-collapse flag check: no row with copy_collapse=True for pos/orig; earliest strict/soft copy flags are absent. ✓ rule did not fire spuriously.

**Quantitative Findings**
Table (pos, orig only): L n — entropy (bits), top‑1 token.

| Layer | Entropy (bits) | Top-1 token |
|---|---:|---|
| L 0 | 17.212843 | `CLICK` |
| L 1 | 17.211401 | `apr` |
| L 2 | 17.210529 | `财经` |
| L 3 | 17.208303 | `-looking` |
| L 4 | 17.205887 | `院子` |
| L 5 | 17.203651 | ` (?)` |
| L 6 | 17.196342 | `ly` |
| L 7 | 17.146254 | ` (?)` |
| L 8 | 17.132240 | ` (?)` |
| L 9 | 17.118841 | ` (?)` |
| L 10 | 17.019901 | ` (?)` |
| L 11 | 17.128208 | `ifiable` |
| L 12 | 17.116917 | `ifiable` |
| L 13 | 17.125601 | `ifiable` |
| L 14 | 17.053112 | `"` |
| L 15 | 17.036381 | `"` |
| L 16 | 16.912832 | `-` |
| L 17 | 16.971632 | `-` |
| L 18 | 16.910591 | `-` |
| L 19 | 16.628599 | `ly` |
| L 20 | 16.695965 | `_` |
| L 21 | 16.408072 | `_` |
| L 22 | 15.219478 | ` ______` |
| L 23 | 15.220285 | `____` |
| L 24 | 10.892860 | `____` |
| L 25 | 13.454483 | `____` |
| L 26 | 5.557553 | `____` |
| L 27 | 4.343688 | `____` |
| L 28 | 4.785875 | `____` |
| L 29 | 1.777650 | `-minded` |
| L 30 | 2.202960 | ` Germany` |
| **L 31** | 0.453865 | ` Berlin` |
| L 32 | 1.036512 | ` German` |
| L 33 | 0.987818 | ` Berlin` |
| L 34 | 0.669073 | ` Berlin` |
| L 35 | 2.494367 | ` Berlin` |
| L 36 | 3.122584 | ` Berlin` |

Control margin (JSON): first_control_margin_pos=1; max_control_margin=0.9999977 (001_layers_baseline/run-latest/output-Qwen3-8B.json:1106–1109).

Ablation (no‑filler): L_copy_orig=null, L_sem_orig=31; L_copy_nf=null, L_sem_nf=31; ΔL_copy=n.a., ΔL_sem=0 (001_layers_baseline/run-latest/output-Qwen3-8B.json:1083–1090). Interpretation: no stylistic‑cue sensitivity on semantic collapse.

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (no strict copy). Soft ΔHk (k∈{1,2,3}) = n.a. (no soft copy). Note: k=1/2/3 soft detectors never fired while strict was null.

Confidence milestones (pure CSV): p_top1>0.30 at L 29; p_top1>0.60 at L 29; final p_top1=0.4334 (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:33–38).

Rank milestones (JSON): rank≤10 at L 29; rank≤5 at L 29; rank≤1 at L 31 (001_layers_baseline/run-latest/output-Qwen3-8B.json:851–853).

KL milestones (JSON): first_kl_below_1.0 at L 36; first_kl_below_0.5 at L 36 (001_layers_baseline/run-latest/output-Qwen3-8B.json:849–850). KL decreases with depth and is ≈0 at final (last_layer_consistency.kl_to_final_bits=0.0; 001_layers_baseline/run-latest/output-Qwen3-8B.json:899–907).

Cosine milestones (pure CSV): first cos_to_final≥0.2 at L 36; ≥0.4 at L 36; ≥0.6 at L 36; final cos_to_final≈1.00 (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:38).

Prism Sidecar Analysis
- Presence: compatible=true; k=512; layers=[embed,8,17,26] (001_layers_baseline/run-latest/output-Qwen3-8B.json:819–831).
- Early-depth stability (KL): baseline vs Prism at sampled depths (L=0,9,18,27) shows Prism KL ≥ baseline (e.g., L0: 12.94 vs 12.79 bits; L27: 13.18 vs 6.14 bits). Baseline: 001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv; Prism: 001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token-prism.csv.
- Rank milestones: Prism never reaches rank≤10/5/1 (all None) while baseline is (29,29,31).
- Top‑1 agreement: no corrective agreements observed at sampled depths; Prism remains far from final.
- Cosine drift: Prism cos_to_final stays negative at early/mid layers (e.g., L0 −0.096; L27 −0.195), not earlier stabilization than baseline.
- Copy flags: no spurious flips; copy flags remain false under Prism.
- Verdict: Regressive (higher KL and no earlier rank milestones).

**Qualitative Patterns & Anomalies**
Negative control (“Berlin is the capital of”): top‑5 = “ Germany” (0.7286), “ which” (0.2207), “ the” (0.0237), “ what” (0.0114), “ __” (0.0023). Semantic leakage: Berlin rank 9 (p = 0.00046) (001_layers_baseline/run-latest/output-Qwen3-8B.json:10–31,46–48).

Important-word trajectory (records CSV): “Berlin” first enters any top‑5 for the next token at L 29 and stabilises by L 31; “Germany” appears in next‑token top‑5 at L 30. Representative next‑token rows: L 30 shows top‑5 including “ Germany” and “ Berlin” (001_layers_baseline/run-latest/output-Qwen3-8B-records.csv:594); at L 31 the top‑1 becomes “ Berlin” (001_layers_baseline/run-latest/output-Qwen3-8B-records.csv:610). Other positions in the prompt (“is”, “called”) increasingly carry “Berlin” in their top‑5 from L≈31 onward (e.g., 001_layers_baseline/run-latest/output-Qwen3-8B-records.csv:609,628–630,655–657).

Collapse timing: removing the “simply” filler does not shift the collapse layer (L_sem=31 in both orig and no_filler; ΔL_sem=0), indicating minimal reliance on that stylistic cue (001_layers_baseline/run-latest/output-Qwen3-8B.json:1083–1090).

Rest‑mass sanity: rest_mass remains low at the first semantic layer (L 31: 0.0050) and rises to 0.175 at the final layer, indicating increased coverage needed for the more diffuse final distribution but no alarming precision loss (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:33–38).

Rotation vs amplification: KL_to_final decreases only late (first <1 bit at L 36), while p_answer jumps to rank 1 at L 31; cosine-to-final reaches high values only at the last layer (≥0.6 at L 36). This suggests early direction remains miscalibrated (late calibration), with strong rank improvement before distributional alignment. Final‑head calibration is clean (kl_to_final_bits=0), so within‑model probabilities at the last layer are usable for this run (001_layers_baseline/run-latest/output-Qwen3-8B.json:849–907).

Head calibration (final layer): last_layer_consistency shows temp_est=1.0 and kl_after_temp_bits=0.0; warn_high_last_layer_kl=false (001_layers_baseline/run-latest/output-Qwen3-8B.json:899–917).

Lens sanity: raw-vs-norm sampling flags lens_artifact_risk=high and max_kl_norm_vs_raw_bits=13.60; no norm-only semantic layer indicated (001_layers_baseline/run-latest/output-Qwen3-8B.json:1071–1075). Treat pre‑final “early semantics” cautiously and prefer rank milestones (we do).

Temperature robustness (JSON test prompts): at nominal decoding variants with “called simply”, “ Berlin” is top‑1 with high mass (e.g., 0.772/0.690) while variants without “simply” or with paraphrase still keep “ Berlin” dominant among top‑k (001_layers_baseline/run-latest/output-Qwen3-8B.json:57–75,150–159). The temperature sweep notes entropy 13.40 bits at T=2.0 with “ Berlin” still top‑1 at 4.2% (001_layers_baseline/run-latest/output-Qwen3-8B.json:480–860).

Checklist
- RMS lens? ✓ (RMSNorm; norm lens enabled) (001_layers_baseline/run-latest/output-Qwen3-8B.json:807,810–816)
- LayerNorm bias removed? n.a. (RMS model; “not_needed_rms_model”) (001_layers_baseline/run-latest/output-Qwen3-8B.json:812)
- Entropy rise at unembed? ✓ (L 35: 2.49 → L 36: 3.12 bits) (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:37–38)
- FP32 un‑embed promoted? ✓ (“unembed_dtype: torch.float32”; “casting_to_fp32_before_unembed”) (001_layers_baseline/run-latest/output-Qwen3-8B.json:809,815)
- Punctuation / markup anchoring? ✓ (underscores/quotes dominate mid‑stack; e.g., L 22–28) (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:24–31)
- Copy‑reflex? ✗ (no strict/soft copy flags in L 0–3) (pure CSV)
- Grammatical filler anchoring? ✗ (early top‑1 are not in {is, the, a, of})

**Limitations & Data Quirks**
- Lens‑artifact risk is “high” (sampled check; max_kl_norm_vs_raw_bits=13.60); prefer rank milestones over absolute probabilities for pre‑final layers (001_layers_baseline/run-latest/output-Qwen3-8B.json:1071–1075).
- No strict or soft copy events; ΔH and Soft ΔHk are undefined.
- Rest_mass after L_sem remains ≤0.18; not a fidelity metric but suggests reasonable top‑k coverage in final layers.
- Prism sidecar appears regressive for this run (higher KL; no earlier ranks), so we treat baseline lens as primary and Prism as diagnostic only.

**Model Fingerprint**
“Qwen‑3‑8B: semantic collapse at L 31; final entropy 3.12 bits; ‘Berlin’ stable top‑1 from L 31; KL→final drops to 0 only at L 36.”

---
Produced by OpenAI GPT-5

