# Evaluation Report: mistralai/Mistral-Small-24B-Base-2501
**1. Overview**
Mistral‑Small‑24B‑Base‑2501 (24B). Run timestamp from artifact: 2025‑09‑28 20:02:40 CEST (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1).
Probe captures layer‑wise next‑token distributions with a normalization (norm) lens, KL to the model’s final head, geometry to final direction, and copy/semantic collapse flags for the “Germany→Berlin” probe.

**2. Method Sanity‑Check**
Diagnostics confirm the norm lens and rotary positions are applied: “use_norm_lens: true … layer0_position_info: token_only_rotary_model” (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:18–27). Context prompt ends with “called simply” and no trailing space (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:28).
Implementation flags and calibration present: “layernorm_bias_fix: not_needed_rms_model … norm_alignment_fix: using_ln2_rmsnorm_for_post_block … mixed_precision_fix: casting_to_fp32_before_unembed” (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:23–26). Copy config and labels are recorded: “copy_thresh: 0.95, copy_window_k: 1, copy_match_level: id_subsequence … copy_flag_columns: ["copy_strict@0.95","copy_soft_k1@0.5","copy_soft_k2@0.5","copy_soft_k3@0.5"]” (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:106–113, 1380–1386). Soft‑copy configuration: “threshold: 0.5, window_ks: [1,2,3]” (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:50–57). Gold alignment is OK (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:29, 958, 1720–1729).
Control present: “control_prompt … The capital of France … gold_alignment: ok” (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:15–29). Control summary: “first_control_margin_pos: 1, max_control_margin: 0.4679627253462968” (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:30–33).
Ablation summary exists: “L_copy_orig: null, L_sem_orig: 33 … L_copy_nf: null, L_sem_nf: 31 … delta_L_sem: −2” (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:7–14).
Strict and soft copy flags are present in the pure CSV header (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:1). No strict or soft copy flag fires in layers 0–3 (scan; earliest soft k1: none). Lens sanity: raw‑vs‑norm “mode: sample … lens_artifact_risk: low; max_kl_norm_vs_raw_bits: 0.1793; first_norm_only_semantic_layer: null” (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4–13, 59–63).
Summary indices (bits): “first_kl_below_0.5: 40, first_kl_below_1.0: 40, first_rank_le_1: 33, first_rank_le_5: 30, first_rank_le_10: 30” (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:109–113). Last‑layer head calibration is perfect: “kl_to_final_bits: 0.0 … p_top1_lens: 0.4555 … temp_est: 1.0 … warn_high_last_layer_kl: false” (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:31–38, 48). Final CSV row also shows KL=0 and “Berlin” top‑1 with p_top1=0.4555 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:42).

Copy‑collapse flag check (strict): none fired in L0–L3. ✓ rule satisfied (no spurious early copy). Soft k1: none in L0–L3.

**3. Quantitative Findings**
Main table (pos, orig). Each row: L — entropy (bits), top‑1 ‘token’.

| Layer | Entropy (bits) | Top‑1 token |
|---:|---:|:--|
| 0 | 16.999 | Forbes |
| 1 | 16.975 | 随着时间的 |
| 2 | 16.944 | 随着时间的 |
| 3 | 16.812 | 随着时间的 |
| 4 | 16.868 |  quelcon |
| 5 | 16.903 | народ |
| 6 | 16.909 | народ |
| 7 | 16.898 | народ |
| 8 | 16.896 |  quelcon |
| 9 | 16.885 |  simply |
| 10 | 16.836 |  hétérogènes |
| 11 | 16.842 | 从那以后 |
| 12 | 16.840 |  simply |
| 13 | 16.871 |  simply |
| 14 | 16.815 | стен |
| 15 | 16.816 | luš |
| 16 | 16.830 | luš |
| 17 | 16.775 | luš |
| 18 | 16.761 | luš |
| 19 | 16.775 | luš |
| 20 | 16.742 | luš |
| 21 | 16.775 |  simply |
| 22 | 16.764 |  simply |
| 23 | 16.769 | -на |
| 24 | 16.758 | -на |
| 25 | 16.747 |  «** |
| 26 | 16.769 |  «** |
| 27 | 16.776 |  «** |
| 28 | 16.741 |  «** |
| 29 | 16.760 |  «** |
| 30 | 16.743 | -на |
| 31 | 16.793 | -на |
| 32 | 16.789 | -на |
| **33** | **16.774** | **Berlin** |
| 34 | 16.761 | Berlin |
| 35 | 16.734 | Berlin |
| 36 | 16.699 | Berlin |
| 37 | 16.513 | " |
| 38 | 15.869 | " |
| 39 | 16.005 | Berlin |
| 40 | 3.181 | Berlin |

Bold semantic layer (ID‑level gold): L_semantic = 33 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:104, 111; pure CSV first is_answer=True at 33 [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:35]).

Control margin (control_summary): first_control_margin_pos = 1; max_control_margin = 0.4679627253462968 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:30–33).

Ablation (no‑filler): L_copy_orig = null, L_sem_orig = 33; L_copy_nf = null, L_sem_nf = 31; ΔL_copy = null, ΔL_sem = −2 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:7–14).

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n/a (no strict copy). Soft ΔHₖ (k∈{1,2,3}) = n/a (no soft copy). Notably, at L_semantic=33 the top‑k rest_mass remains high (0.9988) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:35).

Confidence milestones: p_top1 > 0.30 at layer 40; p_top1 > 0.60 at n/a; final‑layer p_top1 = 0.4555 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:42).

Rank milestones (diagnostics): rank ≤ 10 at layer 30, rank ≤ 5 at layer 30, rank ≤ 1 at layer 33 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:111–113).

KL milestones (diagnostics): first_kl_below_1.0 at 40; first_kl_below_0.5 at 40; KL decreases with depth and is ≈0 at final (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:109–110; 30–38). Final CSV KL=0.0 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:42).

Cosine milestones (pure CSV): first cos_to_final ≥0.2 at L=35; ≥0.4 at L=40; ≥0.6 at L=40; final cos_to_final = 1.0000 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:37, 42).

Prism Sidecar Analysis. Present and compatible (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:36–49), but regressive: at L≈{0,10,20,30} KL_base vs KL_prism are {10.52→10.68, 10.91→12.84, 10.84→16.82, 10.57→15.53}; rank milestones (≤10,≤5,≤1) never achieved under Prism for this probe; cos_to_final is smaller at early/mid depths; copy flags do not flip (no early hits). Verdict: Regressive (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:2, 42; 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token-prism.csv).

Tuned‑Lens Sidecar. Last‑layer agreement holds via diagnostics (KL≈0 after temp; temp_est=1.0) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:31–38). ΔKL at depth percentiles (approx L=10/20/30): Δ = KL_norm − KL_tuned ≈ {4.19, 4.59, 5.35} bits (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv; -tuned.csv). Entropy drift at L≈20 shrinks under tuned (≈13.56→6.90 bits). Rank earliness under tuned is later (rank≤10 at 34; ≤5 at 35; ≤1 at 39), indicating calibration improvements without earlier rank collapse.

**4. Qualitative Patterns & Anomalies**
Negative control: “Berlin is the capital of” → top‑5: [“ Germany”, 0.802], [“ which”, 0.067], [“ the”, 0.0448], [“ _”, 0.0124], [“ what”, 0.0109] (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:14–31). Berlin still appears in the top‑10: “ Berlin”, 0.00481 — semantic leakage: Berlin rank ~8 (p = 0.00481) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:38–41).
Records (NEXT slot) show a late transition from filler/punctuation and multilingual shards to the target token: Berlin first enters top‑k around L≈30 and stabilizes as top‑1 by L=33–36, briefly ceding to quotes at L=37–38, then returning as top‑1 by L=39–40: “…, Berlin, 0.000169 …” [L30], “ Berlin, 0.000424 … (rank=1)” [L33], “ Berlin, 0.0271” [L39] (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:32, 35, 41). Important‑word trajectory aligns with this: “Berlin” rises from sporadic top‑k presence (L≈30–32) to stable dominance post‑L33; earlier layers are dominated by non‑English tokens and punctuation.
Collapse‑layer stability vs instruction: Removing “simply” advances semantics slightly (ΔL_sem = −2; 33→31), suggesting minor stylistic anchoring rather than core semantics (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:7–14).
Rest‑mass sanity: Rest_mass falls sharply only at the final layer (0.1811), while remaining high through L_semantic (max after L_semantic = 0.9988 at L=33), indicating very low top‑k coverage pre‑final and emphasizing calibration drift prior to the head (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:35, 42). Rotation vs amplification: KL decreases only at the very end (first <1.0 at L=40), while cos_to_final rises modestly (≥0.2 at L=35) then locks at the final head. This pattern suggests “early direction, late calibration” — the trajectory aligns toward the final target direction before the distribution calibrates (norm temperature snapshots also large: e.g., KL_temp@50% = 10.69 bits) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:54–60).
Head calibration (final): Perfect last‑layer agreement (kl_to_final_bits=0.0; temp_est=1.0; warn_high_last_layer_kl=false) supports using final probabilities for within‑model interpretation (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:31–38, 48).
Lens sanity: raw‑vs‑norm “mode: sample … lens_artifact_risk: low; max_kl_norm_vs_raw_bits: 0.1793; first_norm_only_semantic_layer: null” indicates low risk of norm‑only semantics (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4–13, 59–63). Temperature exploration in test prompts shows robust picking of “ Berlin” across paraphrases (e.g., “Germany’s capital city is called” → 0.6203; “Germany has its capital at” → 0.7161) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:26–69, 118–127).
Important‑word trajectory — “Berlin first enters any top‑5 at layer 30, stabilises by layer 33. ‘Germany’ remains salient in test prompts; ‘capital’ is present in the context, but not a NEXT candidate; punctuation/quotes dominate in L37–38” — e.g., “…, Berlin, 0.000169 …” [L30]; “ "", 0.02536 …” [L38] (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:32, 40).

Checklist: ✓ RMS lens; ✓ LayerNorm bias removed (RMS model); ✓ Entropy rise at unembed (final entropy 3.18 bits vs ≫10 bits mid‑stack); ✓ FP32 un‑embed promoted (unembed_dtype: torch.float32); ✓ Punctuation/markup anchoring (L37–38 quotes); Copy‑reflex: ✗ (no early strict/soft flags in L0–3); Grammatical filler anchoring: ✓ (tokens like “ is”, “ the”, “ of” appear as top‑k across early layers; see CSV rows around L9–14).

**5. Limitations & Data Quirks**
Rest_mass > 0.3 after L_semantic (0.9988 at L=33) indicates low top‑k coverage under the norm lens despite correct rank; treat probabilities pre‑final with caution and prefer rank milestones. KL is lens‑sensitive; here, final‑head calibration is perfect, but mid‑depth KL remains high. Raw‑vs‑norm checks were “sample” mode only (not exhaustive). Surface‑mass comparisons are tokenizer‑dependent; prefer within‑model trends and rank thresholds.

**6. Model Fingerprint**
Mistral‑Small‑24B‑Base‑2501: collapse at L 33; final entropy 3.18 bits; “Berlin” enters top‑k at L≈30 and is top‑1 by L 33.

---
Produced by OpenAI GPT-5 
*Run executed on: 2025-09-28 17:22:48*
