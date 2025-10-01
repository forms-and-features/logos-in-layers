**1. Overview**
- Model: Meta-Llama-3-8B (32 layers) — results in `output-Meta-Llama-3-8B.*`
- Probe: layer-by-layer logit lens on the prompt “Give the city name only, plain text. The capital of Germany is called simply”. Captures rank milestones, KL-to-final, cosine-to-final, and copy/collapse signals.

**2. Method Sanity-Check**
- Lens config and prompt: Diagnostics confirm norm lens is active (`"use_norm_lens": true`) and the prompt ends exactly with “called simply” (no trailing space):
  > "use_norm_lens": true  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:807]
  > "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply"  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:817]
- Family/normalization: RMSNorm model; LN bias fix not needed and LN2 alignment in post-block path:
  > "first_block_ln1_type": "RMSNorm", "final_ln_type": "RMSNorm", "layernorm_bias_fix": "not_needed_rms_model", "norm_alignment_fix": "using_ln2_rmsnorm_for_post_block"  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:810-814]
- Copy detector configs (strict + soft) present and mirrored in flags:
  > copy_thresholds: τ ∈ {0.7, 0.8, 0.9, 0.95}; all `L_copy_strict[τ] = null`; stability = "none"  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1000-1006]
  > copy_soft_config: threshold = 0.5, window_ks = [1,2,3] (extra_thresholds = [])  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:880-888]
  > copy_flag_columns: ["copy_strict@0.95", "copy_strict@0.7", "copy_strict@0.8", "copy_strict@0.9", "copy_soft_k1@0.5", "copy_soft_k2@0.5", "copy_soft_k3@0.5"]  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:958-966]
- Gold alignment: ok.  > "gold_alignment": "ok"  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1075]
- Required summary indices (norm lens):
  > first_kl_below_0.5 = 32; first_kl_below_1.0 = 32; first_rank_le_1 = 25; first_rank_le_5 = 25; first_rank_le_10 = 24  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:931-935]
- Last-layer head calibration: final KL≈0; perfect lens↔model agreement and Temp=1.0:
  > "kl_to_final_bits": 0.0, "top1_agree": true, "p_top1_lens": 0.52018, "p_top1_model": 0.52018, "kl_after_temp_bits": 0.0, "warn_high_last_layer_kl": false  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1077-1095]
- Measurement guidance (must honor): prefers ranks; suppress abs probs; preferred lens = tuned; use confirmed semantics:
  > { "prefer_ranks": true, "suppress_abs_probs": true, reasons: ["norm_only_semantics_window", "high_lens_artifact_risk"], "preferred_lens_for_reporting": "tuned", "use_confirmed_semantics": true }  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:2128-2138]
- Raw-vs-Norm window (NEXT): radius=4; centers=[25,32]. Norm-only semantic layers in window: [25,27,28,29,30]. Max KL(norm||raw) in window = 5.2565 bits:
  > "raw_lens_window": { … "center_layers": [25,32], "norm_only_semantics_layers": [25,27,28,29,30], "max_kl_norm_vs_raw_bits_window": 5.256533119384461 }  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1033-1061]
- Raw-vs-Norm full: pct KL≥1.0 = 0.0303; pct KL≥0.5 = 0.0303; n_norm_only_semantics_layers = 5; earliest_norm_only_semantic = 25; max_kl_norm_vs_raw_bits = 5.2565; lens_artifact_score = 0.4182 (tier = medium):
  > "raw_lens_full": { … "pct_layers_kl_ge_1.0": 0.030303…, "n_norm_only_semantics_layers": 5, "earliest_norm_only_semantic": 25, "max_kl_norm_vs_raw_bits": 5.256533…, "score": {"tier": "medium"} }  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1063-1073]
- Copy-reflex check (layers 0–3): strict remains False and soft k1@0.5 also False in L0–L3 (see pure CSV rows 2–5). No early copy-reflex ✓.
- Control prompt present; control summary exists:
  > first_control_margin_pos = 0; max_control_margin = 0.5186312557016208  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:160-166]
- Ablation (no-filler) present, both variants emitted; main table uses `prompt_id = pos`, `prompt_variant = orig`:
  > { "L_copy_orig": null, "L_sem_orig": 25, "L_copy_nf": null, "L_sem_nf": 25, "delta_L_sem": 0 }  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:146-156]

Prism sidecar: compatible=true but regressive vs norm (KL higher; no rank milestones):
  > prism KL p25=17.176 vs baseline p25=11.807 (Δ=−5.369); first_kl_le_1.0 prism=null  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:862-876]

Tuned-Lens: loaded; gate prefers tuned; attribution shows rotation gains (ΔKL_rot > 0):
  > prefer_tuned: true; ΔKL_rot at p25=4.528, p50=4.540, p75=3.734  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:2107-2126]
  > tuned collapse: first_rank_le_1 = 32 (later than baseline 25)  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:2067-2076]

**3. Quantitative Findings**
Per-layer NEXT token (pos,orig): “L i — entropy X bits, top‑1 'token'”
- L 0 — entropy 16.9568 bits, top‑1 'itzer'  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:2]
- L 1 — 16.9418 bits, 'mente'  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:3]
- L 2 — 16.8764 bits, 'mente'  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:4]
- L 3 — 16.8936 bits, 'tones'  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:5]
- L 4 — 16.8991 bits, 'interp'  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:6]
- L 5 — 16.8731 bits, '�'  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:7]
- L 6 — 16.8797 bits, 'tons'  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:8]
- L 7 — 16.8806 bits, 'Exited'  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:9]
- L 8 — 16.8624 bits, 'надлеж'  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:10]
- L 9 — 16.8677 bits, '‘'  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:11]
- L 10 — 16.8582 bits, '‘'  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:12]
- L 11 — 16.8530 bits, '‘'  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:13]
- L 12 — 16.8500 bits, '‘'  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:14]
- L 13 — 16.8477 bits, ' '
- L 14 — 16.8447 bits, ' '
- L 15 — 16.8427 bits, ' '
- L 16 — 16.8376 bits, ' '
- L 17 — 16.8369 bits, ' '
- L 18 — 16.8329 bits, ' '
- L 19 — 16.8354 bits, ' '
- L 20 — 16.8304 bits, "'"  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:24]
- L 21 — 16.8338 bits, "'"  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:25]
- L 22 — 16.8265 bits, 'tons'  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:26]
- L 23 — 16.8280 bits, 'tons'  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:27]
- L 24 — 16.8299 bits, ' capital'  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:28]
- L 25 — 16.8142 bits, ' Berlin'  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:29]
- L 26 — 16.8285 bits, ' Berlin'  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:30]
- L 27 — 16.8194 bits, ' Berlin'  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:31]
- L 28 — 16.8194 bits, ' Berlin'  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:32]
- L 29 — 16.7990 bits, ' Berlin'  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:33]
- L 30 — 16.7946 bits, ' Berlin'  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:34]
- L 31 — 16.8378 bits, ':'  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:35]
- L 32 — 2.9610 bits, ' Berlin'  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:36]

Bolded semantic layer: L 25 (confirmed; source = raw). The norm lens gives first rank=1 at L 25 and confirmed semantics at L 25:
- > "first_rank_le_1": 25; … "L_semantic_confirmed": 25, "confirmed_source": "raw"  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:933,1447-1450]

Control margin (JSON control_summary):
- first_control_margin_pos = 0; max_control_margin = 0.5186  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:160-166]

Ablation (no‑filler): L_copy_orig = null; L_sem_orig = 25; L_copy_nf = null; L_sem_nf = 25; ΔL_copy = null; ΔL_sem = 0  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:146-156].

Below-table metrics
- ΔH (bits) = entropy(L_copy) − entropy(L_semantic): not available (L_copy_strict null; L_copy_soft[k] null).
- Soft ΔHk (bits): not available (no soft-copy layer).
- Confidence milestones (p_top1, norm lens): p_top1 > 0.30 at L 32; p_top1 > 0.60: n/a; final p_top1 = 0.5202  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:36].
- Rank milestones (norm lens): rank ≤ 10 at L 24; rank ≤ 5 at L 25; rank ≤ 1 at L 25  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:931-935].
- KL milestones (norm lens): first_kl_below_1.0 at L 32; first_kl_below_0.5 at L 32; final KL≈0 with perfect last-layer agreement  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:931-935,1077-1095]. KL decreases late and is ≈0 at final.
- Cosine milestones (norm lens): cos_to_final ≥0.2 at L 20; ≥0.4 at L 30; ≥0.6 at L 32; final cos_to_final ≈ 1.0  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1018-1024; 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:36].
- Normalized depths: L_semantic_frac = 0.781; first_rank_le_5_frac = 0.781  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1025-1031].

Copy robustness (threshold sweep)
- stability = "none"; earliest L_copy_strict at τ=0.70 and τ=0.95 are both null; no norm-only strict-copy flags  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1000-1006].

Prism Sidecar Analysis
- Presence: compatible=true but KL is worse than baseline at early/mid depths (e.g., p25: 17.18 vs 11.81; Δ = −5.37 bits) and no prism rank milestones (all null)  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:862-876].
- Verdict: Regressive (higher KL, no earlier ranks).

Tuned-Lens Summary (respect preferred_lens_for_reporting="tuned")
- Rank earliness: tuned first_rank_le_1/5/10 = 32/32/32 (later than norm’s 25/25/24)  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:2067-2076].
- Attribution (rotation vs temperature): ΔKL_tuned ≈ {p25: 4.35, p50: 4.31, p75: 3.95}; ΔKL_temp ≈ {−0.18, −0.23, +0.21}; ΔKL_rot positive at all reported percentiles ⇒ rotation gains beyond temperature  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:2108-2123].
- Last-layer agreement check (teacher): KL_after_temp_bits ≈ 0.0 and lens↔model agree  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1077-1095].
- Entropy drift (norm, mid-depth example): at L 24, entropy 16.8299 vs teacher 11.3211 ⇒ drift ≈ +5.509 bits  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:28].

**4. Qualitative Patterns & Anomalies**
The answer emerges abruptly: rank‑1 for 'Berlin' at L 25 and remains stable through L 30–31 with very low absolute p_top1 (measurement_guidance prefers ranks), then the final layer calibrates both probability and entropy sharply (p_top1 ≈ 0.52; entropy ≈ 2.96 bits)  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:29-36]. Pre‑collapse layers exhibit filler/punctuation drift in the top‑1 (e.g., ' capital' at L 24) with rising cos_to_final (≥0.2 by L 20; ≥0.4 by L 30), consistent with “early direction, late calibration” — cosine aligns sooner than KL, which drops only at the end  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1018-1024; 931-935].

Negative control ("Berlin is the capital of"): top‑5 shows clean country prediction; 'Berlin' still appears in the list (rank 6, p ≈ 0.0029) — semantic leakage:
> [" Germany", 0.8955], [" the", 0.0525], [" and", 0.0075], [" germany", 0.0034], [" modern", 0.0030], [" Berlin", 0.0029]  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:810-800 → 801-803-851-859; first test prompt block 200-235]

Important-word trajectory (NEXT). 'Berlin' first becomes rank‑1 at L 25 and remains rank‑1 or in top‑2 through L 30; 'Germany' and prompt-themed punctuation remain in the top‑5 mass until final layer reduces uncertainty: e.g., L 25 shows ' Berlin' (p ≈ 1.32e‑4), ' capital', ' Germany' in the top‑5  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:29]; final layer concentrates mass heavily on ' Berlin' (0.5202) with quotes and punctuation trailing  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1469-1490]. Semantically close tokens like ' Deutschland' and lowercase ' berlin' appear in final top‑k with small mass  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1489-1499].

Instruction sensitivity (ablation). Removing “simply” leaves L_semantic unchanged (ΔL_sem = 0), suggesting little stylistic anchoring in this setup  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:146-156].

Rest‑mass sanity. Top‑k rest_mass stays ≈1.0 through mid‑stack (very diffuse), then drops sharply at L 32 to 0.1625 as uncertainty collapses; no spikes after L_semantic  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:29-36].

Rotation vs amplification. cos_to_final rises by L 20 and continues to increase (≥0.4 by L 30), while KL to final remains high until the end; the final layer both rotates and calibrates distributional mass (KL≈0, entropy ≈ 2.96 bits). This matches “early direction, late calibration.”

Head calibration (final). No warning; Temp_est=1.0; KL_after_temp_bits=0.0; top‑1 agreements between lens and model: see last_layer_consistency  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1077-1095].

Lens sanity. Windowed check flags norm‑only semantics at layers [25,27,28,29,30], max KL(norm||raw)=5.2565 bits; full check: 5 norm‑only semantic layers, earliest at 25, lens_artifact_score tier=medium  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1033-1073]. Given measurement_guidance, we favor rank milestones over absolute probabilities and treat pre‑final “early semantics” cautiously.

Temperature robustness (teacher exploration). At T=0.1, 'Berlin' is rank‑1 with high mass; at T=2.0, 'Berlin' remains top‑1 but with small mass; entropy rises substantially (see temperature_exploration entries, e.g., final: p≈0.0366 at T=2.0; entropy≈13.87 bits)  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1688-1762].

Checklist
- RMS lens? ✓  [RMSNorm types; 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:810-814]
- LayerNorm bias removed? ✓ (not needed for RMS)  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:812]
- Entropy rise at unembed? ✓ Final entropy ≈ 2.96 bits, concentrated mass on answer  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1466-1474]
- FP32 un-embed promoted? use_fp32_unembed=false; unembed_dtype="torch.float32"  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:809]
- Punctuation / filler anchoring? ✓ Pre‑collapse top‑1s include punctuation/fillers (e.g., ' capital', quotes)  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:24,28,31]
- Copy‑reflex? ✗ (no strict/soft hits in L0–L3)  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:2-5]
- Grammatical filler anchoring? ✓ (' the', quotes, punctuation frequent in L0–L5 and mid‑stack)
- Preferred lens honored? ✓ Tuned metrics reported, but note tuned rank earliness regresses.
- Confirmed semantics reported? ✓ L_semantic_confirmed=25 (source=raw)  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1447-1450]
- Full dual‑lens metrics cited? ✓ (pct_layers_kl≥1.0, n_norm_only_semantics_layers, earliest_norm_only_semantic, tier)
- Tuned‑lens attribution done? ✓ (ΔKL_tuned, ΔKL_temp, ΔKL_rot at ~25/50/75%)

**5. Limitations & Data Quirks**
- Measurement guidance flags norm‑only semantics and medium lens‑artifact score; treat early semantics cautiously and prefer rank milestones over absolute probabilities  [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1033-1073,2128-2138].
- Final‑head calibration is excellent here (KL≈0), but KL is lens‑sensitive; use ranks for cross‑model claims.
- Surface mass and rest_mass rely on tokenizer‑level coverage; prefer within‑model trends.
- Prism sidecar is diagnostic only and is regressive on this run (higher KL, no earlier ranks).
- Soft/strict copy detectors did not trigger; ΔH tied to copy is not computable for this run.

**6. Model Fingerprint**
“Llama‑3‑8B: collapse at L 25 (confirmed raw); final entropy 2.96 bits; tuned lens collapses later (L 32).”

---
Produced by OpenAI GPT-5 

