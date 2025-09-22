# Evaluation Report: mistralai/Mistral-7B-v0.1
**Overview**

Mistral‑7B‑v0.1 (32‑layer, pre‑norm; diagnostics model "mistralai/Mistral-7B-v0.1"). The probe traces copy vs. semantic collapse on the pure next token, tracking entropy, KL to final, answer rank, and cosine trajectory per layer. Results are filtered to `prompt_id = pos`, `prompt_variant = orig` for the main table.

**Method Sanity‑Check**

Diagnostics confirm a norm‑lens on an RMSNorm stack with rotary positions: "use_norm_lens": true [output-Mistral-7B-v0.1.json:807], "first_block_ln1_type": "RMSNorm"; "final_ln_type": "RMSNorm" [output-Mistral-7B-v0.1.json:810]. Positional info is rotary: "layer0_position_info": "token_only_rotary_model" [output-Mistral-7B-v0.1.json:816]. Context prompt ends with “called simply” (no trailing space): "context_prompt": "… is called simply" [output-Mistral-7B-v0.1.json:817].

Implementation flags present: "unembed_dtype": "torch.float32" with fp32 cast fix [output-Mistral-7B-v0.1.json:809,815]. Copy/semantic indices are emitted: "L_copy": null, "L_copy_H": null, "L_semantic": 25, "delta_layers": null [output-Mistral-7B-v0.1.json:833–836]. Copy rule params: "copy_thresh": 0.95, "copy_window_k": 1, "copy_match_level": "id_subsequence" [output-Mistral-7B-v0.1.json:837–839]. Rank/KL milestones: "first_kl_below_0.5": 32, "first_kl_below_1.0": 32, "first_rank_le_1": 25, "first_rank_le_5": 25, "first_rank_le_10": 23 [output-Mistral-7B-v0.1.json:840–844]. Units: the CSV column is `kl_to_final_bits` (bits), and entropies are in bits; model architecture: "architecture": "pre_norm" [output-Mistral-7B-v0.1.json:960].

Gold alignment is ID‑based and OK: "gold_alignment": "ok" [output-Mistral-7B-v0.1.json:845]. Negative control present: `control_prompt` and `control_summary` with margins [output-Mistral-7B-v0.1.json:1032–1050]. Ablation present: `ablation_summary` with `L_sem_nf = 24`, `delta_L_sem = -1` [output-Mistral-7B-v0.1.json:1024–1031]. For main analysis, rows are filtered to `pos, orig`.

Last‑layer head calibration is clean: "kl_to_final_bits": 0.0; top‑1 agree; `temp_est = 1.0`; `warn_high_last_layer_kl = false` [output-Mistral-7B-v0.1.json:847–865]. The pure CSV’s final row also shows `kl_to_final_bits = 0.0` and `p_top1 = 0.3822…` for "Berlin" [output-Mistral-7B-v0.1-pure-next-token.csv:34].

Lens sanity (raw vs norm): mode "sample" with "lens_artifact_risk": "high" and "max_kl_norm_vs_raw_bits": 1.1739; no norm‑only semantics layer [output-Mistral-7B-v0.1.json:1018–1022]. Treat any pre‑final “early semantics” cautiously and prefer rank milestones.

Copy‑collapse flags in early layers 0–3 are all False in the pure CSV (no early copy reflex) [output-Mistral-7B-v0.1-pure-next-token.csv:2–5].

Copy‑collapse flag check: no `copy_collapse = True` rows were found in layers 0–3 → ✓ rule did not spuriously fire.

**Quantitative Findings**

Table (pure next‑token; `prompt_id = pos`, `prompt_variant = orig`). Bold = semantic layer (first `is_answer = True`).

| Layer | Entropy (bits) | Top‑1 |
|---|---:|---|
| L 0 | 14.9614 | dabei |
| L 1 | 14.9291 | biologie |
| L 2 | 14.8254 | ,↵ |
| L 3 | 14.8771 | […] |
| L 4 | 14.8538 | […] |
| L 5 | 14.8265 | […] |
| L 6 | 14.8378 | […] |
| L 7 | 14.8049 | […] |
| L 8 | 14.8210 | […] |
| L 9 | 14.7755 | […] |
| L 10 | 14.7816 | […] |
| L 11 | 14.7363 | […] |
| L 12 | 14.6418 | […] |
| L 13 | 14.7261 | […] |
| L 14 | 14.6531 | […] |
| L 15 | 14.4497 | […] |
| L 16 | 14.5998 | […] |
| L 17 | 14.6278 | […] |
| L 18 | 14.5197 | […] |
| L 19 | 14.5104 | […] |
| L 20 | 14.4242 | simply |
| L 21 | 14.3474 | simply |
| L 22 | 14.3874 | “ |
| L 23 | 14.3953 | simply |
| L 24 | 14.2124 | simply |
| **L 25** | 13.5986 | Berlin |
| L 26 | 13.5409 | Berlin |
| L 27 | 13.2964 | Berlin |
| L 28 | 13.2962 | Berlin |
| L 29 | 11.4269 | " |
| L 30 | 10.7970 | “ |
| L 31 | 10.9943 | " |
| L 32 | 3.6110 | Berlin |

Control margin (JSON): first_control_margin_pos = 2; max_control_margin = 0.6539 [output-Mistral-7B-v0.1.json:1048–1049].

Ablation (no‑filler): L_copy_orig = null; L_sem_orig = 25; L_copy_nf = null; L_sem_nf = 24; ΔL_copy = null; ΔL_sem = −1 [output-Mistral-7B-v0.1.json:1024–1031]. Interpretation: semantics arrives 1 layer earlier without the filler, suggesting low stylistic‑cue dependence for this prompt.

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n/a (L_copy = null).

Confidence milestones (pure CSV): p_top1 > 0.30 at layer 32; p_top1 > 0.60 at n/a; final‑layer p_top1 = 0.3822 [output-Mistral-7B-v0.1-pure-next-token.csv:34].

Rank milestones (diagnostics): rank ≤ 10 at layer 23; rank ≤ 5 at layer 25; rank ≤ 1 at layer 25 [output-Mistral-7B-v0.1.json:842–844].

KL milestones (diagnostics): first_kl_below_1.0 at layer 32; first_kl_below_0.5 at layer 32 [output-Mistral-7B-v0.1.json:840–841]. KL decreases with depth and is ≈ 0 at final (see also CSV final row `kl_to_final_bits = 0.0`) [output-Mistral-7B-v0.1-pure-next-token.csv:34].

Cosine milestones (pure CSV): first cos_to_final ≥ 0.2 at layer 11; ≥ 0.4 at layer 25; ≥ 0.6 at layer 26; final cos_to_final = 0.9999999 [derived from pure CSV].

Prism Sidecar Analysis
- Presence: present and compatible [output-Mistral-7B-v0.1.json:819–824].
- Early‑depth stability (KL to final, bits): baseline vs Prism at L0/8/16/24: baseline 10.17/10.25/10.33/9.05 vs Prism 10.40/23.06/27.87/26.55 (higher KL under Prism; regressive).
- Rank milestones (Prism pure CSV): no layer reaches rank ≤ 10 (none found) [derived from Prism pure CSV].
- Top‑1 agreement: Prism diverges at final (L32 Prism top‑1 non‑semantic; answer_rank = 49) [output-Mistral-7B-v0.1-pure-next-token-prism.csv:32 block].
- Cosine drift: Prism cos_to_final remains low/negative mid‑stack and only modest at L32 (≈0.17), indicating no early stabilization.
- Copy flags: no spurious `copy_collapse` flips in layers 0–3 (all False).
- Verdict: Regressive (KL increases substantially; no earlier or equal rank milestones; degraded final agreement).

**Qualitative Patterns & Anomalies**

Negative control: for “Berlin is the capital of”, top‑5: Germany 0.8966, the 0.0539, both 0.0044, a 0.0038, Europe 0.0031 [output-Mistral-7B-v0.1.json:10–24]. Berlin still appears in the list (semantic leakage: Berlin rank ≈6, p ≈ 0.00284) [output-Mistral-7B-v0.1.json:15].

Important‑word trajectory (records): by L25, the context token “Germany” has top‑5 including “Berlin” (ranked among capital‑related tokens): "Germany, … Berlin, 0.0034" [output-Mistral-7B-v0.1-records.csv:440]. Across layers 20–24, the pure next‑token top‑1 remains prompt‑style fillers (“simply”, quotation marks), then the answer becomes top‑1 at L25, stabilising through L28 (table above).

Collapse dynamics: From L20–L24, top‑1 focuses on filler punctuation and the word “simply,” indicating grammatical/form anchoring before semantic commitment. Semantic collapse occurs sharply at L25 (rank 1; p_answer ≈ 0.0335), with cosine crossing 0.4 at the same layer and 0.6 at L26, while KL remains high until the final layer—an “early direction, late calibration” signature consistent with tuned‑lens observations (cf. KL→0 only at L32).

Rest‑mass sanity: Rest_mass falls after L24; maximum after L_semantic is 0.9106 at L25, then decreases, reaching 0.2298 at the final layer [derived from pure CSV]. No spike indicative of precision loss post‑semantic collapse.

Rotation vs amplification: p_answer rises from near‑zero to ≈0.0335 at L25 as answer_rank hits 1, cosine increases (≥0.4 at 25, ≥0.6 at 26), while KL remains >8 bits at L25 and only collapses to 0 at L32. This suggests direction alignment precedes calibration of the final head.

Head calibration (final): `warn_high_last_layer_kl = false`, with `temp_est = 1.0` and `kl_after_temp_bits = 0.0` [output-Mistral-7B-v0.1.json:853–864]. Treat final probabilities as well‑calibrated for within‑model interpretation.

Lens sanity: raw vs norm check reports `lens_artifact_risk = "high"`, `max_kl_norm_vs_raw_bits = 1.1739`, no `first_norm_only_semantic_layer` [output-Mistral-7B-v0.1.json:1018–1022]. Use rank milestones (first_rank_le_{10,5,1}) for early‑layer claims.

Temperature robustness: at T = 0.1, Berlin rank 1 (p = 0.9996), entropy = 0.0050 bits [output-Mistral-7B-v0.1.json:670–683]. At T = 2.0, Berlin rank drops with p = 0.0360 and entropy = 12.2200 bits [output-Mistral-7B-v0.1.json:736–744, 737–738].

Checklist
- RMS lens? ✓ (RMSNorm stack; norm lens true) [output-Mistral-7B-v0.1.json:807,810–811]
- LayerNorm bias removed? ✓ (not needed for RMSNorm) [output-Mistral-7B-v0.1.json:812]
- Entropy rise at unembed? n.a. (not separately instrumented here)
- FP32 un‑embed promoted? ✓ (fp32 cast; unembed_dtype fp32) [output-Mistral-7B-v0.1.json:809,815]
- Punctuation / markup anchoring? ✓ (L20–L24)
- Copy‑reflex? ✗ (layers 0–3 `copy_collapse = False`)
- Grammatical filler anchoring? ✓ (top‑1 ∈ {“simply”, quotes} for L20–L24)

**Limitations & Data Quirks**

`rest_mass` reflects top‑k coverage, not fidelity; it remains high early and declines late (max after L_semantic = 0.9106 at L25). `raw_lens_check` is `mode = sample` with `lens_artifact_risk = high`; treat early “semantics” cautiously and prefer rank thresholds. KL is lens‑sensitive; despite final KL ≈ 0 (clean head calibration), cross‑model comparisons should lean on rank milestones. Prism sidecar is present but regressive for this model; do not use Prism probabilities for calibration here.

**Model Fingerprint**

Mistral‑7B‑v0.1: semantic collapse at L 25; final entropy 3.611 bits; “Berlin” stabilises top‑1 from L25, with fillers (“simply”, quotes) dominating L20–L24.

---
Produced by OpenAI GPT-5
*Run executed on: 2025-09-21 17:13:26*
