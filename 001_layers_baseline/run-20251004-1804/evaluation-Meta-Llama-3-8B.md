# Evaluation Report: meta-llama/Meta-Llama-3-8B

*Run executed on: 2025-10-04 18:04:23*

**1. Overview**

Meta‑Llama‑3‑8B (32 layers) evaluated on 2025‑10‑04. The probe traces layer‑by‑layer next‑token behavior under a norm lens, measuring copy vs. semantic onset, KL to the final head, cosine trajectory to final, and surface/coverage masses.

**2. Method Sanity‑Check**

The prompt is correct and ends with “called simply” (no trailing space): "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply" (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:4). Norm lens and FP32 analysis unembed are active: "use_norm_lens": true; "unembed_dtype": "torch.float32"; "use_fp32_unembed": false (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:807–809).

Normalizer provenance matches a pre‑norm pipeline with next‑block ln1 and epsilon inside sqrt (strategy="next_ln1"): "ln_source": "blocks[0].ln1" at layer 0 and "ln_source": "ln_final" at the unembed (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7135, 7139, 7395). Per‑layer normalizer effects are present and behave smoothly (e.g., layer 0 resid_norm_ratio=18.1873, delta_resid_cos=0.5346; layer 31 resid_norm_ratio=0.6416, delta_resid_cos=0.8832) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7142–7143, 7390–7391). Unembedding bias is absent: "present": false, "l2_norm": 0.0 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:826–830).

Environment/determinism: torch 2.8.0 (cpu), dtype_compute=torch.float32, deterministic_algorithms=true, seed=316 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:8170–8178). Numeric health is clean (any_nan=false; any_inf=false; layers_flagged=[]) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7603–7609). Copy mask is present and plausible (punctuation‑heavy sample), size=6022 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:6976, 932–952).

Copy detection configuration is included: copy_thresh=0.95, k=1, match=id_subsequence; strict sweep tau_list=[0.7,0.8,0.9,0.95] with all L_copy_strict=null and stability="none" (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:6985–7019, 7060–7079). Flags emitted in JSON/CSV mirror labels: copy_strict@{0.95,0.7,0.8,0.9}, copy_soft_k{1,2,3}@0.5 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:8183–8190). Gold alignment is ok (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7610). Control summary is present: first_control_margin_pos=0, max_control_margin=0.5186 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:8215–8220). Ablation summary exists for no‑filler: L_sem_orig=25, L_sem_nf=25 (ΔL_sem=0) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:8190–8200).

Summary indices (baseline/norm): first_kl_below_0.5=32, first_kl_below_1.0=32, first_rank_le_1=25, first_rank_le_5=25, first_rank_le_10=24 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:6985–6989). Last‑layer head calibration is clean: kl_to_final_bits=0.0, top1_agree=true, p_top1_lens=0.5202, p_top1_model=0.5202, temp_est=1.0, kl_after_temp_bits=0.0 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7611–7623). Measurement guidance requests rank‑first reporting and suppresses absolute probabilities; preferred lens=tuned; use_confirmed_semantics=true; reasons=["norm_only_semantics_window","high_lens_artifact_risk","normalization_spike"] (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:8701–8720).

Raw‑vs‑Norm window: radius=4; center_layers=[25,32]; norm_only_semantics_layers=[25,27,28,29,30]; max_kl_norm_vs_raw_bits_window=5.2565; mode="window" (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7091–7119). Lens sanity (sample): first_norm_only_semantic_layer=25; max_kl_norm_vs_raw_bits=0.0713; lens_artifact_risk="high" (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:8163–8169). Full raw‑vs‑norm: pct_layers_kl_ge_1.0=0.0303, pct_layers_kl_ge_0.5=0.0303, n_norm_only_semantics_layers=5, earliest_norm_only_semantic=25, max_kl_norm_vs_raw_bits=5.2565, tier="medium" (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7120–7131).

Copy‑collapse flag check (pos/orig, layers 0–3): no strict copy_collapse and no soft k1@0.5 hits; all False in CSV rows L0–L3 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:2–7).

Tuned‑lens presence and attribution: prefer_tuned=true; ΔKL_tuned at percentiles ≈{25,50,75} = {4.352, 4.308, 3.947} bits with ΔKL_rot positive (rotation gains beyond temperature) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:8637–8660). Prism sidecar is present/compatible, but regressive vs. baseline (KL higher at p25/p50/p75 by ≈{−5.37, −8.29, −9.78} bits) and rank milestones null (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:834–871).

Negative control prompt exists and aligns to "Paris" (gold ok). Top‑5 for the test prompt “Berlin is the capital of” shows Germany at top and Berlin low‑mass: "Germany" 0.8955, "the" 0.0525, …, "Berlin" 0.0029 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:10–31).

**3. Quantitative Findings**

Table below reports per‑layer NEXT under norm lens for pos/orig. Bold marks the semantic layer (confirmed_norm=25; confirmed_source=raw; see confirmed Semantics block).

| Layer | Entropy (bits) | Top‑1 token |
|---:|---:|:---|
| 0 | 16.9568 | itzer |
| 1 | 16.9418 | mente |
| 2 | 16.8764 | mente |
| 3 | 16.8936 | tones |
| 4 | 16.8991 | interp |
| 5 | 16.8731 | � |
| 6 | 16.8797 | tons |
| 7 | 16.8806 | Exited |
| 8 | 16.8624 | надлеж |
| 9 | 16.8666 | biased |
| 10 | 16.8506 | tons |
| 11 | 16.8541 | tons |
| 12 | 16.8770 | LEGAL |
| 13 | 16.8430 | macros |
| 14 | 16.8351 | tons |
| 15 | 16.8467 | simply |
| 16 | 16.8471 | simply |
| 17 | 16.8477 | simply |
| 18 | 16.8392 | simply |
| 19 | 16.8399 | ' |
| 20 | 16.8304 | ' |
| 21 | 16.8338 | ' |
| 22 | 16.8265 | tons |
| 23 | 16.8280 | tons |
| 24 | 16.8299 |  capital |
| **25** | 16.8142 | Berlin |
| 26 | 16.8285 | Berlin |
| 27 | 16.8194 | Berlin |
| 28 | 16.8194 | Berlin |
| 29 | 16.7990 | Berlin |
| 30 | 16.7946 | Berlin |
| 31 | 16.8378 | : |
| 32 | 2.9610 | Berlin |

Control margin (JSON): first_control_margin_pos=0; max_control_margin=0.5186 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:8215–8220).

Ablation (no‑filler): L_copy_orig=null, L_sem_orig=25; L_copy_nf=null, L_sem_nf=25; ΔL_copy=null; ΔL_sem=0 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:8190–8200). Interpretation: removing “simply” does not shift collapse for this item.

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (L_copy_strict=null). Soft ΔHk (k∈{1,2,3}) = n.a. (all L_copy_soft[k]=null) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7001–7019, 7060–7079).

Confidence milestones (CSV): p_top1>0.30 at layer 32 only; p_top1>0.60 not reached; final‑layer p_top1=0.5202 (“Berlin”) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:36).

Rank milestones: baseline(nrm) rank≤10 at L24, rank≤5 at L25, rank=1 at L25 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:6985–6989). Preferred lens=tuned reports later ranks (≤{10,5,1} all at L32) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7656–7660).

KL milestones: first_kl_below_1.0 at L32; first_kl_below_0.5 at L32; KL≈0 at final (0.0), consistent with last‑layer calibration (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:6985–6986, 7611).

Cosine milestones (baseline): ge_0.2 at L20; ge_0.4 at L30; ge_0.6 at L32; final cos_to_final≈1.0 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7076–7082; 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:36).

Depth fractions: L_semantic_frac=0.781 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7083–7089).

Copy robustness (sweep): stability="none"; earliest L_copy_strict at τ=0.70 and 0.95 are null; norm_only_flags are null (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7038–7079).

Prism sidecar analysis: present/compatible, but regressive vs. baseline — higher KL at early/mid depths (p25/p50/p75 deltas ≈−5.37/−8.29/−9.78 bits) and no earlier ranks (all null) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:840–871). Verdict: Regressive.

**4. Qualitative Patterns & Anomalies**

Norm‑only semantics and calibration. Under the norm lens the answer reaches rank‑1 at L25, while raw lens in the window shows answer_rank_raw > 1 at nearby layers (e.g., L25: norm answer_rank=1 vs raw=3; KL_norm_vs_raw≈0.054 bits) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token-rawlens-window.csv:9–10). Full raw‑vs‑norm records flag norm‑only semantics layers and medium artifact tier; accordingly, I prefer rank‑based milestones and the confirmed semantics index.

Rotation vs amplification. Cosine rises into {0.2,0.4,0.6} by layers {20,30,32} while KL to final remains high until late; p_answer becomes rank‑1 by L25 but with tiny absolute mass until the end — an “early direction, late calibration” profile (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7076–7082; 6985–6989).

Negative control. For “Berlin is the capital of”, the top‑5 places "Germany" at 0.8955 and "Berlin" at 0.0029, indicating no leakage of the target into the inverse query (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:10–31). Write‑in prompt variant shows similar behavior across the paraphrase set in test_prompts (multiple entries with " Berlin" as top‑1 when directly cueing the city name).

Important‑word trajectory. Around L24–L31, prompt words and answer compete among top‑k at NEXT: e.g., L25 top‑5 includes "Berlin" (top‑1) with nearby prompt tokens " capital"/" simply"/" Germany" (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:29–33). This aligns with the IMPORTANT_WORDS list in the script [Germany, Berlin, capital, Answer, word, simply] (001_layers_baseline/run.py:474–484), and reflects a shift from filler/punctuation in early layers to semantically central tokens near collapse.

Collapse‑layer stability without the one‑word filler. The ablation shows L_sem_nf = L_sem_orig = 25 (ΔL_sem=0), i.e., no measurable shift (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:8190–8200).

Rest‑mass sanity. Rest_mass remains ≈0.999 through mid/late layers (e.g., L29 rest_mass=0.99951) and drops at the final layer to 0.1625 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:33, 36). This reflects the expected concentration under the final head and does not indicate precision loss.

Head calibration (final layer). Calibration is clean: kl_to_final_bits=0.0; temp_est=1.0; warn_high_last_layer_kl=false (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7611–7623).

Lens sanity. Raw‑lens sample flags first_norm_only_semantic_layer=25 and lens_artifact_risk="high" (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:8163–8169); the full check reports n_norm_only_semantics_layers=5 and max_kl_norm_vs_raw_bits=5.2565 (tier="medium") (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7120–7131). Accordingly, I emphasize ranks and use the confirmed semantics layer.

Temperature robustness. At T=0.1, "Berlin" is rank‑1 at p≈0.99996 (entropy≈0.00057 bits); at T=2.0, "Berlin" remains rank‑1 at p≈0.0366 and entropy≈13.87 bits (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:660–740).

Checklist
✓ RMS lens?
✓ LayerNorm bias removed?
✓ Entropy rise at unembed?
✗ FP32 un‑embed promoted? (use_fp32_unembed=false; unembed_dtype=float32) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:807–809)
✓ Punctuation / markup anchoring? (early layers top‑1 are punctuation/fillers)
✗ Copy‑reflex? (no early strict/soft hits in L0–L3)
✓ Grammatical filler anchoring? ("simply", "the", punctuation in L15–L21)
✓ Preferred lens honored in milestones (tuned ranks cited alongside baseline)
✓ Confirmed semantics reported (L_semantic_confirmed=25; source=raw) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7990–7997)
✓ Full dual‑lens metrics cited (pct_layers_kl_ge_1.0, n_norm_only_semantics_layers, earliest_norm_only_semantic, tier)
✓ Tuned‑lens attribution done (ΔKL_tuned, ΔKL_temp, ΔKL_rot at ~25/50/75%)
✓ normalization_provenance present (ln_source verified at layer 0 and final)
✓ per‑layer normalizer effect metrics present (resid_norm_ratio, delta_resid_cos)
✓ unembed bias audited (bias‑free cosine guaranteed)
✓ deterministic_algorithms = true
✓ numeric_health clean (no NaN/Inf)
✓ copy_mask present and plausible
✓ layer_map present (indexing audit)

**5. Limitations & Data Quirks**

Norm‑only semantics are present in a window around L25–L30 with medium artifact risk; early semantics should be treated cautiously and rank milestones preferred (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7091–7131, 8163–8169). KL is lens‑sensitive; while final KL≈0 confirms head calibration, cross‑model probability comparisons should be avoided per measurement guidance (prefer_ranks=true; suppress_abs_probs=true) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:8701–8720). Surface‑mass and coverage depend on tokenizer idiosyncrasies; interpret within‑model trends. Prism is diagnostic only and appears regressive here; do not substitute its ranks for the model head.

**6. Model Fingerprint**

Llama‑3‑8B: collapse at L 25 (confirmed/raw), tuned collapse L 32; final entropy ≈2.96 bits; "Berlin" stabilizes rank‑1 mid‑stack and calibrates late.

---
Produced by OpenAI GPT-5 
