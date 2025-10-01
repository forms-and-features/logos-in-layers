# EVAL — Meta‑Llama‑3‑8B

**Overview**
- Model `meta-llama/Meta-Llama-3-8B`, 32 layers; probe targets first unseen token after the context.
- Captures layerwise entropy, top‑1/rank milestones, copy/echo signals, KL to final, and geometry vs final head; final entropy 2.961 bits (baseline lens).

**Method Sanity‑Check**
- Norm lens active and rotary positions noted: "use_norm_lens": true [JSON:807], "layer0_position_info": "token_only_rotary_model" [JSON:816].
- Context prompt ends with ‘called simply’:     "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply", [JSON:4].
- Gold alignment ok; last‑layer head agrees (KL≈0): "gold_alignment": "ok" [JSON:1063], "kl_to_final_bits": 0.0; "top1_agree": true [JSON:1065].
- Measurement guidance: prefer ranks, suppress abs probs: "prefer_ranks": true, "suppress_abs_probs": true [JSON:2074].
- Copy config present: "copy_thresh": 0.95 [JSON:928], "copy_window_k": 1 [JSON:929], "copy_match_level": "id_subsequence" [JSON:930], soft "threshold": 0.5 with ks [1, 2, 3] [JSON:880]; flags ['copy_strict@0.95', 'copy_strict@0.7', 'copy_strict@0.8', 'copy_strict@0.9', 'copy_soft_k1@0.5', 'copy_soft_k2@0.5', 'copy_soft_k3@0.5'] [JSON:1588].
- Raw‑vs‑Norm window: centers [25, 32], radius 4, norm‑only semantics at layers [25, 27, 28, 29, 30], max KL(norm||raw) in window 5.257 [JSON:1033].
- Lens sanity (sample): lens_artifact_risk = high with first_norm_only_semantic_layer = 25 and max_kl_norm_vs_raw_bits = 0.071 [JSON:1526].
- Threshold sweep: stability = none; earliest L_copy_strict@0.70=None, @0.95=None (norm_only_flags null).
- Copy‑collapse flags: none fired in L0–3 or later (strict/soft all null). Window check at L25 shows norm semantic ‘Berlin’ vs raw ‘ in’: [RAWLENS:10] / [RAWLENS:11] — norm‑only semantics present; treat early semantics cautiously.

**Quantitative Findings**
| Layer | Entropy (bits) | Top‑1 token |
|---:|---:|---|
| L 0 | 16.9568 | `itzer` |
| L 1 | 16.9418 | `mente` |
| L 2 | 16.8764 | `mente` |
| L 3 | 16.8936 | `tones` |
| L 4 | 16.8991 | `interp` |
| L 5 | 16.8731 | `�` |
| L 6 | 16.8797 | `tons` |
| L 7 | 16.8806 | `Exited` |
| L 8 | 16.8624 | `надлеж` |
| L 9 | 16.8666 | `biased` |
| L 10 | 16.8506 | `tons` |
| L 11 | 16.8541 | `tons` |
| L 12 | 16.8770 | `LEGAL` |
| L 13 | 16.8430 | `macros` |
| L 14 | 16.8351 | `tons` |
| L 15 | 16.8467 | ` simply` |
| L 16 | 16.8471 | ` simply` |
| L 17 | 16.8477 | ` simply` |
| L 18 | 16.8392 | ` simply` |
| L 19 | 16.8399 | ` '` |
| L 20 | 16.8304 | ` '` |
| L 21 | 16.8338 | ` '` |
| L 22 | 16.8265 | `tons` |
| L 23 | 16.8280 | `tons` |
| L 24 | 16.8299 | ` capital` |
| **L 25** | 16.8142 | `** Berlin**` |
| L 26 | 16.8285 | ` Berlin` |
| L 27 | 16.8194 | ` Berlin` |
| L 28 | 16.8194 | ` Berlin` |
| L 29 | 16.7990 | ` Berlin` |
| L 30 | 16.7946 | ` Berlin` |
| L 31 | 16.8378 | `:` |
| L 32 | 2.9610 | ` Berlin` |

Control margin: first_control_margin_pos = 0, max_control_margin = 0.518631 [JSON:1620].
Ablation (no‑filler): L_copy_orig = None, L_sem_orig = 25, L_copy_nf = None, L_sem_nf = 25, ΔL_copy = None, ΔL_sem = 0 [JSON:1597].
ΔH (bits) = n.a. (no strict or soft copy layer). Soft ΔHk (k∈{1,2,3}) = n.a. (no soft copy layer).
Confidence milestones: p_top1>0.30 at layer 32, p_top1>0.60 at layer none, final p_top1 = 0.520 [CSV:36].
Rank milestones: ≤10 at 24, ≤5 at 25, ≤1 at 25 [JSON:933].
KL milestones: first KL<1.0 at 32, KL<0.5 at 32; final KL≈0 [JSON:1065].
Cosine milestones: cos_to_final ≥0.2 at 20, ≥0.4 at 30, ≥0.6 at 32; final cos_to_final ≈ 1.0 [CSV:36]. Depth fractions: L_semantic_frac = 0.781 [JSON:1018].
Copy robustness: none (L_copy_strict @0.70=None, @0.95=None; norm_only_flags all null).
Prism Sidecar Analysis
- Presence: compatible=true (k=512, layers=['embed', 7, 15, 23]) [JSON:825].
- Early-depth KL (baseline vs Prism, bits): p25 11.807→17.176, p50 11.728→20.014, p75 11.321→21.102 [JSON:856].
- Rank milestones: baseline le_1=25, Prism le_1=None (none) [JSON:839].
- Verdict: Regressive (KL increases ≥5–9 bits at early/mid depths; no earlier rank milestones).

Tuned‑Lens (side‑by‑side diagnostics)
- ΔKL medians (norm − tuned): p25 4.352, p50 4.308, p75 3.947 [JSON:1582].
- Last-layer agreement: kl_after_temp_bits ≈ 0 [JSON:1065]. Entropy drift at mid-depth (e.g., L16): entropy 16.8471 vs teacher 2.9610 (drift +13.886) [CSV:20].
- Rank earliness: baseline first_rank≤1 at 25; tuned shows ≤1 at 32 (later) [JSON:839].
- Geometry/coverage: L_geom_norm=26, L_surface_to_meaning_norm=32 with answer_mass 0.520, echo_mass 0.024 [JSON:1007].
- Norm temp taus present; snapshots: @25%=11.983, @50%=11.959, @75%=11.108 [JSON:1084].

**Qualitative Patterns & Anomalies**
Negative control: “Berlin is the capital of” shows ‘Berlin’ rank 6 (p = 0.0029) → semantic leakage [JSON:10].
Important‑word trajectory: top‑1 ‘ capital’ appears at L 24; ‘Berlin’ first top‑1 at L 25 and stays through L 30; punctuation resurfaces at L 31 before final head [CSV:28] → [CSV:29] → [CSV:35].
Rest‑mass sanity: rest_mass falls steadily; max after L_semantic = 0.999554 at layer 25 [CSV:29].
Rotation vs amplification: cosine rises early (≥0.2 by L20) while KL remains ≫1 until L32, indicating early direction with late calibration; treat early ‘meaning’ as lens‑induced given norm‑only semantics at L25 (and 27–30).
Head calibration: final KL≈0 with top‑1 agreement and temp_est=1.0 [JSON:1065].
Lens sanity: raw‑lens check flags high artifact risk; first_norm_only_semantic_layer=25; caution on early semantics [JSON:1585]. Window rows show norm ‘Berlin’ at L25 vs raw ‘ in’ [RAWLENS:10] / [RAWLENS:11].
Temperature robustness: at T=0.1, ‘Berlin’ rank 1 (p≈0.99996); at T=2.0, ‘Berlin’ remains rank 1 (p≈0.0366); entropy shifts 0.0006→13.8732 bits [JSON:670]/[JSON:737].
Stylistic ablation: removing ‘simply’ leaves L_semantic unchanged (ΔL_sem=0), suggesting low dependence on the adverb for this probe [JSON:1597].

Checklist
- RMS lens? ✓ (RMSNorm; normalized lens on)
- LayerNorm bias removed? ✓ (not needed for RMS)
- Entropy rise at unembed? ✓ (final entropy 2.961 vs ≈16.8 mid‑stack)
- FP32 un‑embed promoted? use_fp32_unembed=False with unembed_dtype=torch.float32 [JSON:808].
- Punctuation / markup anchoring? ✓ (e.g., ':', quotes at L31)
- Copy‑reflex? ✗ (no strict or soft hits in L0–3)
- Grammatical filler anchoring? ✓ (‘ simply’ dominates L15–18 top‑1)

**Limitations & Data Quirks**
- High lens‑artifact risk with norm‑only semantics (L25, 27–30); prefer rank milestones and within‑model trends.
- KL is lens‑sensitive; final KL≈0 here, but cross‑family comparisons should rely on rank milestones and qualitative KL trends.
- Surface‑mass depends on tokenizer; avoid cross‑model absolute comparisons.
- Raw‑vs‑norm window mode present; treat as targeted, not exhaustive; max KL(norm||raw) in window 5.257.

**Model Fingerprint**
Meta‑Llama‑3‑8B: collapse at L 25; final entropy 3.0 bits; ‘Berlin’ stabilizes as top‑1 by L25 and persists to L30 before final head.

---
Produced by OpenAI GPT-5
