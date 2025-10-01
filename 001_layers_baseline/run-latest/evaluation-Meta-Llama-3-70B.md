**1. Overview**
- Model: `meta-llama/Meta-Llama-3-70B`; Layers: 80 (diagnostics.model, model_stats)
- Run timestamp: `timestamp-20250930-2357` (run-latest marker). Probe traces layerwise entropy, ranks, KL-to-final, cosine, copy flags, and control margin.

**2. Method Sanity‑Check**
JSON confirms RMSNorm lensing and rotary positions with calibrated FP32 unembed: "use_norm_lens": true and "layer0_position_info": "token_only_rotary_model" (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:807,816). The context prompt ends exactly with “called simply” (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:817). Final‑head alignment is good: "kl_to_final_bits": 0.000729..., "top1_agree": true (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1127–1132).

Gold alignment is OK (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1125,1396–1406). Implementation flags and indices present: "use_norm_lens", "use_fp32_unembed", "unembed_dtype" (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:807–811); "L_semantic": 40; copy detector config with strict k=1, τ=0.95 and soft ks∈{1,2,3} (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:980–1018). Copy flag labels mirror JSON/CSV: copy_strict@{0.95,0.7,0.8,0.9}, copy_soft_k{1,2,3}@0.5 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1344–1352).

Measurement guidance: prefer ranks and suppress abs probs due to norm‑only semantics window; preferred lens = norm; use_confirmed_semantics = true (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1385–1394). Summary indices: first_kl_below_1.0 = 80; first_kl_below_0.5 = 80; first_rank_le_{10,5,1} = {38,38,40} (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:972–980).

Raw‑vs‑Norm window: radius=4, centers=[38,40,80], norm‑only semantics at layers [79,80], max window KL(norm vs raw)=1.2437 bits (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1082–1111). Full raw‑vs‑norm: pct_kl≥1.0=0.0123, pct_kl≥0.5=0.0247, n_norm_only_semantics_layers=2, earliest=79, max_kl=1.2437, tier=medium (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1113–1123). Treat early semantics cautiously; report rank milestones.

Threshold sweep: present with stability="none"; earliest L_copy_strict at τ=0.70 and τ=0.95 are null; norm_only_flags all null (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1028–1054). Strict copy collapse not observed in layers 0–3 and soft‑copy k1@0.5 also never fires there (pure CSV shows no True flags).

Last‑layer head calibration: final pure row has kl_to_final_bits ≈ 0.00073 and is_answer=True (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:82). Warn flag is false (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1144).

Prism sidecar: present and compatible; shows higher KL vs baseline at key depths (p25/p50/p75 deltas ≈ −0.89/−1.00/−1.16 bits baseline − prism; negative means prism worse) and no rank milestones (le_1=null) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:825–860,845–853,856–860).

Copy config and flags used: copy_thresh=0.95; copy_window_k=1; copy_match_level=id_subsequence; copy_soft_config: threshold=0.5, window_ks=[1,2,3], extra_thresholds=[] (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:965–987,880–888,1000–1018,1344–1352). When L_copy_strict is null, Δ is defined to use earliest L_copy_soft[k]; here all soft L_copy_soft[k] are null, so Δ is n/a.

Control prompt present with alignment OK; control summary: first_control_margin_pos=0, max_control_margin=0.5168457566906 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1361–1379).

**3. Quantitative Findings**

| Layer | Entropy (bits) | Top-1 token |
|---:|---:|:---|
| L 0 | 16.968 | ' winding' |
| L 1 | 16.960 | 'cepts' |
| L 2 | 16.963 | 'улю' |
| L 3 | 16.963 | 'zier' |
| L 4 | 16.959 | 'alls' |
| L 5 | 16.957 | 'alls' |
| L 6 | 16.956 | 'alls' |
| L 7 | 16.953 | 'NodeId' |
| L 8 | 16.959 | 'inds' |
| L 9 | 16.960 | 'NodeId' |
| L 10 | 16.952 | 'inds' |
| L 11 | 16.956 | 'inds' |
| L 12 | 16.956 | 'lia' |
| L 13 | 16.955 | 'eds' |
| L 14 | 16.950 | 'idders' |
| L 15 | 16.953 | ' Kok' |
| L 16 | 16.952 | '/plain' |
| L 17 | 16.948 | ' nut' |
| L 18 | 16.944 | ' nut' |
| L 19 | 16.948 | ' nut' |
| L 20 | 16.946 | ' nut' |
| L 21 | 16.938 | ' burge' |
| L 22 | 16.938 | ' simply' |
| L 23 | 16.936 | ' bur' |
| L 24 | 16.950 | ' bur' |
| L 25 | 16.937 | '�' |
| L 26 | 16.938 | '�' |
| L 27 | 16.937 | 'za' |
| L 28 | 16.933 | '/plain' |
| L 29 | 16.933 | ' plain' |
| L 30 | 16.939 | 'zed' |
| L 31 | 16.925 | ' simply' |
| L 32 | 16.941 | ' simply' |
| L 33 | 16.927 | ' plain' |
| L 34 | 16.932 | ' simply' |
| L 35 | 16.929 | ' simply' |
| L 36 | 16.940 | ' simply' |
| L 37 | 16.935 | ' simply' |
| L 38 | 16.934 | ' simply' |
| L 39 | 16.935 | ' simply' |
| **L 40** | 16.937 | ' Berlin' |
| L 41 | 16.936 | ' "' |
| L 42 | 16.944 | ' "' |
| L 43 | 16.941 | ' Berlin' |
| L 44 | 16.926 | ' Berlin' |
| L 45 | 16.940 | ' "' |
| L 46 | 16.955 | ' "' |
| L 47 | 16.939 | ' "' |
| L 48 | 16.939 | ' "' |
| L 49 | 16.937 | ' "' |
| L 50 | 16.944 | ' "' |
| L 51 | 16.940 | ' "' |
| L 52 | 16.922 | ' Berlin' |
| L 53 | 16.933 | ' Berlin' |
| L 54 | 16.942 | ' Berlin' |
| L 55 | 16.942 | ' Berlin' |
| L 56 | 16.921 | ' Berlin' |
| L 57 | 16.934 | ' Berlin' |
| L 58 | 16.941 | ' Berlin' |
| L 59 | 16.944 | ' Berlin' |
| L 60 | 16.923 | ' Berlin' |
| L 61 | 16.940 | ' Berlin' |
| L 62 | 16.951 | ' Berlin' |
| L 63 | 16.946 | ' Berlin' |
| L 64 | 16.926 | ' Berlin' |
| L 65 | 16.933 | ' "' |
| L 66 | 16.941 | ' Berlin' |
| L 67 | 16.930 | ' Berlin' |
| L 68 | 16.924 | ' Berlin' |
| L 69 | 16.932 | ' Berlin' |
| L 70 | 16.926 | ' Berlin' |
| L 71 | 16.923 | ' Berlin' |
| L 72 | 16.922 | ' Berlin' |
| L 73 | 16.918 | ' "' |
| L 74 | 16.914 | ' Berlin' |
| L 75 | 16.913 | ' Berlin' |
| L 76 | 16.919 | ' Berlin' |
| L 77 | 16.910 | ' Berlin' |
| L 78 | 16.919 | ' Berlin' |
| L 79 | 16.942 | ' Berlin' |
| L 80 | 2.589 | ' Berlin' |

Control margin (JSON): first_control_margin_pos = 0; max_control_margin = 0.5168457566906 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1376–1379).

Ablation (no‑filler): L_copy_orig = null, L_sem_orig = 40; L_copy_nf = null, L_sem_nf = 42; ΔL_copy = null, ΔL_sem = 2 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1353–1360).

- ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n/a (no L_copy).
- Soft ΔHk (bits): n/a (no L_copy_soft[k]).
- Confidence milestones (pure CSV, pos/orig): p_top1 > 0.30 at layer 80; p_top1 > 0.60: none; final-layer p_top1 = 0.4783 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:82).
- Rank milestones (diagnostics, norm lens): rank ≤ 10 at L 38; ≤ 5 at L 38; ≤ 1 at L 40 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:976–980).
- KL milestones (diagnostics): first_kl_below_1.0 at L 80; first_kl_below_0.5 at L 80; final KL ≈ 0 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:972–976,1127).
- Cosine milestones (JSON): cos_to_final ≥ {0.2,0.4,0.6} at L 80; final cos_to_final ≈ 0.99999 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1066–1071; pure CSV final row 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:82).
- Normalized depths: L_semantic_frac = 0.50; first_rank_le_5_frac = 0.475 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1073–1076).

Copy robustness (threshold sweep): stability = "none"; earliest L_copy_strict at τ=0.70 and τ=0.95 are null; norm_only_flags all null (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1028–1054). No soft‑copy flags fired early.

Prism Sidecar Analysis
- Early‑depth KL: baseline vs prism at L≈{0,20,40,60} shows higher KL under prism: {10.50→10.67, 10.45→11.34, 10.42→11.42, 10.31→11.47} bits (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:1; 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token-prism.csv:1).
- Rank milestones: prism never achieves rank ≤ {10,5,1} (diagnostics.prism_summary.metrics.rank_milestones.prism all null) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:845–853). Baseline: le_1 at 40.
- Top‑1 agreement/copy flags: no prism copy flips observed; copy flags absent in both.
- Verdict: Regressive relative to baseline (higher KL and no earlier rank milestones).

**4. Qualitative Patterns & Anomalies**
Negative control (“Berlin is the capital of”): top‑5 are " Germany" 0.8516, " the" 0.0791, " and" 0.0146, " modern" 0.0048, " Europe" 0.0031 — consistent, no leakage of "Berlin" (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:13–31). Important‑word trajectory: under the norm lens at NEXT, "simply" holds top‑1 through L39 before semantic flip; at L38 Berlin enters top‑5 and Germany appears ("... Berlin", p=2.50e−05; " Germany", p=2.49e−05) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:40], at L39 Berlin rises to top‑2 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:39), and at L40 Berlin becomes rank‑1 with is_answer=True (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:42). Germany remains visible in top‑5 around L38–41 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:40–43).

Rotation vs amplification: KL to final stays ≈10.4 bits through mid‑depth while cos_to_final rises from ≈0.084 (L38) to ≈0.097 (L41), then saturates to ~0.99999 at L80, illustrating “early direction, late calibration” (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:40–43,82). Final‑head calibration is clean: warn_high_last_layer_kl=false and kl_after_temp_bits≈0.00073 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1127–1145).

Lens sanity: sampled raw‑vs‑norm risk is low (max_kl_norm_vs_raw_bits=0.043; lens_artifact_risk="low") (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1338–1342), but the full scan flags two late norm‑only semantics layers (79–80) and tier="medium" (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1106–1123). We therefore honor measurement_guidance.prefer_ranks=true (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1385–1394) and foreground rank milestones.

Temperature robustness: at T=0.1, Berlin rank 1 with p≈0.9933; at T=2.0, Berlin remains rank 1 with p≈0.0357; entropy rises from ≈0.058 to ≈14.464 bits (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:670–676,737–744).

Rest‑mass sanity: rest_mass decreases sharply by the final layer (min after L_semantic = 0.1074 at L80), but is large mid‑stack (max after L_semantic ≈0.9999 at L46), reflecting top‑k coverage rather than fidelity (pure CSV per‑layer rest_mass; final row 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:82).

Checklist
- RMS lens? ✓ (RMSNorm; "first_block_ln1_type": "RMSNorm") (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:810)
- LayerNorm bias removed? ✓ ("layernorm_bias_fix": "not_needed_rms_model") (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:812)
- Entropy rise at unembed? ✓ (final teacher_entropy_bits≈2.597; final entropy≈2.589) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:82)
- FP32 un‑embed promoted? ✓ ("use_fp32_unembed": true) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:808)
- Punctuation / markup anchoring? ✓ (mid‑layers top‑1 often quotes/punctuation around L41–51 in pure CSV table)
- Copy‑reflex? ✗ (no strict/soft copy flags in L0–3)
- Grammatical filler anchoring? ✓ (" simply" dominates L35–39)
- Preferred lens honored? ✓ (measurement_guidance.preferred_lens_for_reporting = "norm") (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1392)
- Confirmed semantics reported? ✓ (L_semantic_confirmed=40, source=raw) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1160–1167)
- Full dual‑lens metrics cited? ✓ (pct_layers_kl≥{1.0,0.5}, n_norm_only_semantics, earliest, score.tier)
- Tuned‑lens attribution done? n.a. (tuned_lens.status="missing") (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1158–1160)

**5. Limitations & Data Quirks**
- Rest_mass is top‑k coverage only and is very high mid‑stack; not a fidelity metric. We rely on KL/ranks (final KL≈0.00073 bits; 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1127).
- Raw‑vs‑norm “full” reports tier=medium with late norm‑only semantics at {79,80}; we therefore emphasize rank milestones and confirmed semantics (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1106–1123).
- Copy thresholds show no strict/soft L_copy; ΔH and soft ΔHk are n/a.
- Prism is diagnostic and regressive here (higher KL, no earlier ranks); do not adjust core metrics based on prism.

**6. Model Fingerprint**
“Llama‑3‑70B: collapse at L 40; final entropy ≈2.59 bits; ‘Berlin’ becomes rank‑1 at L40 and dominates by L80.”

---
Produced by OpenAI GPT-5 
