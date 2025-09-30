**1. Overview**
mistralai/Mistral-7B-v0.1 (7B), run timestamp 2025-09-28 17:22 (see `001_layers_baseline/run-latest/timestamp-20250928-1722`). The probe captures layer-by-layer next-token distributions under a norm lens with RMSNorm-aware scaling, reporting entropy, rank milestones, KL to final, cosine-to-final, and copy/echo surface mass.

**2. Method Sanity‑Check**
JSON confirms the intended norm lens and positional handling: “use_norm_lens”: true and FP32 shadow unembed configured (“unembed_dtype”: "torch.float32") [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:807,809]. The model is RMSNorm pre‑norm (“first_block_ln1_type”: "RMSNorm", “final_ln_type”: "RMSNorm", “architecture”: "pre_norm") [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:810,811,1309], with layer‑0 position info reported as “token_only_rotary_model” [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:816]. Context ends exactly with “called simply” [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4]. Norm‑temperature diagnostics are present (tau curve and KL_temp snapshots at 25/50/75%) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:848,971–979].

Copy flags and configuration are present and mirrored in CSV: `copy_flag_columns` = ["copy_strict@0.95","copy_soft_k1@0.5","copy_soft_k2@0.5","copy_soft_k3@0.5"] [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1372–1377]; strict rule: τ=0.95, k=1, `copy_match_level` = "id_subsequence" [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1009–1016]. Soft‑copy config present (threshold 0.5, window_ks = {1,2,3}) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:835–844]. Gold alignment is ok [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:949–952]. Negative control prompt and summary are present [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1400–1416]. Ablation summary exists with both variants [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1379–1386].

Summary indices (bits): first_kl_below_0.5 = 32; first_kl_below_1.0 = 32; first_rank≤1 = 25; first_rank≤5 = 25; first_rank≤10 = 23 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1001–1007]. Final‑head calibration is good: final CSV KL→final = 0.0 and `last_layer_consistency` shows top‑1 agreement, temp_est = 1.0, kl_after_temp_bits = 0.0, warn_high_last_layer_kl = false [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:34, 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:954–969]. Lens sanity: raw‑vs‑norm check ran in sampled mode with `lens_artifact_risk` = "high" and `first_norm_only_semantic_layer` = null; max_kl_norm_vs_raw_bits = 1.1739 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1311,1368–1370].

Copy‑collapse flag check (strict, τ=0.95, δ=0.10): no `copy_collapse=True` rows in layers 0–3; likewise, no `copy_soft_k1@0.5=True` in layers 0–3 (all False in CSV rows 2–5) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:2–5].

First `copy_collapse=True`: none observed; ✓ rule did not fire spuriously.
Earliest soft‑copy flags (k1/k2/k3): none (all null/False across layers) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1018–1038].

**3. Quantitative Findings**
Table (pos, orig only; entropy in bits; top‑1 token). Bold = semantic layer (first `is_answer=True`).

- L 0 – entropy 14.961 bits, top‑1 'dabei'
- L 1 – entropy 14.929 bits, top‑1 'biologie'
- L 2 – entropy 14.825 bits, top‑1 ',\r'
- L 3 – entropy 14.877 bits, top‑1 '[…]'
- L 4 – entropy 14.854 bits, top‑1 '[…]'
- L 5 – entropy 14.827 bits, top‑1 '[…]'
- L 6 – entropy 14.838 bits, top‑1 '[…]'
- L 7 – entropy 14.805 bits, top‑1 '[…]'
- L 8 – entropy 14.821 bits, top‑1 '[…]'
- L 9 – entropy 14.776 bits, top‑1 '[…]'
- L 10 – entropy 14.782 bits, top‑1 '[…]'
- L 11 – entropy 14.736 bits, top‑1 '[…]'
- L 12 – entropy 14.642 bits, top‑1 '[…]'
- L 13 – entropy 14.726 bits, top‑1 '[…]'
- L 14 – entropy 14.653 bits, top‑1 '[…]'
- L 15 – entropy 14.450 bits, top‑1 '[…]'
- L 16 – entropy 14.600 bits, top‑1 '[…]'
- L 17 – entropy 14.628 bits, top‑1 '[…]'
- L 18 – entropy 14.520 bits, top‑1 '[…]'
- L 19 – entropy 14.510 bits, top‑1 '[…]'
- L 20 – entropy 14.424 bits, top‑1 'simply'
- L 21 – entropy 14.347 bits, top‑1 'simply'
- L 22 – entropy 14.387 bits, top‑1 '“'
- L 23 – entropy 14.395 bits, top‑1 'simply'
- L 24 – entropy 14.212 bits, top‑1 'simply'
- **L 25 – entropy 13.599 bits, top‑1 'Berlin'**
- L 26 – entropy 13.541 bits, top‑1 'Berlin'
- L 27 – entropy 13.296 bits, top‑1 'Berlin'
- L 28 – entropy 13.296 bits, top‑1 'Berlin'
- L 29 – entropy 11.427 bits, top‑1 '"'
- L 30 – entropy 10.797 bits, top‑1 '“'
- L 31 – entropy 10.994 bits, top‑1 '"'
- L 32 – entropy 3.611 bits, top‑1 'Berlin'

Gold answer: “Berlin” [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1665–1671].

Control margin (JSON `control_summary`): first_control_margin_pos = 2; max_control_margin = 0.6539100579732349 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1420–1423].

Ablation (no‑filler): L_copy_orig = null; L_sem_orig = 25; L_copy_nf = null; L_sem_nf = 24; ΔL_copy = null; ΔL_sem = −1 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1379–1386].

ΔH = entropy(L_copy) − entropy(L_semantic) = n.a. (strict copy not detected).
Soft ΔHₖ = n.a. for k ∈ {1,2,3} (no soft hits).

Confidence milestones (pure CSV): p_top1 > 0.30 at layer 32; p_top1 > 0.60 not reached; final-layer p_top1 = 0.3822 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:34].
Rank milestones (diagnostics): rank ≤ 10 at L 23; rank ≤ 5 at L 25; rank ≤ 1 at L 25 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1003–1007].
KL milestones (diagnostics): first_kl_below_1.0 at L 32; first_kl_below_0.5 at L 32 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1001–1002]. KL decreases with depth and is ≈ 0 at final [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:34].
Cosine milestones (pure CSV): cos_to_final ≥ 0.2 at L 11; ≥ 0.4 at L 25; ≥ 0.6 at L 26; final cos_to_final = 0.99999988 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:27–30,34].

Prism Sidecar Analysis (present, compatible; k=512, layers=[embed,7,15,23]) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:825–835].
- Early-depth KL vs baseline (KL(P_layer||P_final), bits):
  - L0: baseline 10.17 vs prism 10.40 (Δ −0.23)
  - L8: baseline 10.25 vs prism 23.06 (Δ −12.81)
  - L16: baseline 10.33 vs prism 27.87 (Δ −17.54)
  - L24: baseline 9.05 vs prism 26.55 (Δ −17.50)
  [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:2,10,18,26; 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token-prism.csv:1,9,17,25]
- Prism rank milestones: no first_rank≤{10,5,1} within sampled depths (none observed) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token-prism.csv:1–33].
- Top‑1 agreement at sampled depths remains non‑semantic (e.g., L23 prism top‑1 “minecraft”, p=0.817) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token-prism.csv:24].
- Cosine drift: prism cos_to_final is unstable/negative at mid‑depths (e.g., L24 −0.3757 vs baseline +0.3565) [same lines as above].
- Copy flags: no spurious flips under Prism (no True flags) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token-prism.csv:1–33].
- Verdict: Regressive (KL larger by >0.5 bits early/mid and no earlier rank milestones).

Tuned‑Lens (present). Last‑layer agreement holds (`last_layer_consistency`, temp_est=1.0) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:954–969]. ΔKL medians at L≈{25,50,75}% (L=8,16,24): Δ = {4.03, 3.75, 7.08} bits (KL_norm − KL_tuned) [computed from pure CSVs]. Entropy drift (tuned): +7.83 bits at L8; +6.01 bits at L16; +0.28 bits at L24 (entropy − teacher_entropy_bits). Rank earliness did not improve: first_rank≤{10,5} at L25 but first_rank≤1 at L32 (later than baseline L25) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token-tuned.csv]. Surface→meaning: L_surface_to_meaning_norm = 32 with (answer_mass, echo_mass) = (0.3822, 0.0610) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:939–946,954]; tuned alternatives show L_surface_to_meaning_tuned ∈ {27,26,25}, e.g., 25 with (0.8768, 0.0286) and L_geom_tuned = 25 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1183–1189,1605–1611]. Coverage: L_topk_decay_{norm,tuned} = {0,1} at τ=0.33 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:945–947,1119–1120]. Norm‑temp snapshots: KL_temp@{25,50,75}% = {10.49, 10.61, 4.83} bits [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:971–979].

**4. Qualitative Patterns & Anomalies**
Negative control: “Berlin is the capital of” → top‑5: Germany 0.8966; the 0.0539; both 0.00436; a 0.00380; Europe 0.00311 (Berlin also appears with p=0.00284) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:8–33]. Semantic leakage: Berlin rank 6 (p = 0.00284) [same block].

Records and surface semantics: The answer emerges at L25 with a semantically coherent top‑5 (“Berlin”, “Germany”, “Frankfurt”, “Deutschland”, “German”) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:27]. After L29, punctuation/quotes briefly dominate top‑1 (", “, ") before the final head reasserts “Berlin” at L32 (p=0.3822) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:29–34]. Berlin first enters any top‑5 at L25 and stabilises by L32.

Collapse timing vs instruction: Removing “simply” slightly advances semantics (L_sem 25→24; ΔL_sem = −1), consistent with mild stylistic anchoring rather than deeper semantics [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1379–1386].

Rest‑mass sanity: Rest_mass is high at first semantic hit (L25: 0.9106) and then falls steadily to 0.2298 at the final layer [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:27,34]. This reflects top‑k coverage, not lens fidelity.

Rotation vs amplification: KL to final declines only near the top (first <1 bit at L32) while `cos_to_final` rises earlier (≥0.2 by L11) and p_answer becomes non‑zero at L25. This indicates “early direction, late calibration”: direction aligns mid‑stack but probabilities calibrate only late [cos milestones above; KL milestones from JSON]. Final‑head calibration is clean (temp_est=1.0; kl_after_temp_bits=0.0) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:961–968].

Lens sanity: raw‑vs‑norm `lens_artifact_risk` = high (sampled), max_kl_norm_vs_raw_bits = 1.1739; `first_norm_only_semantic_layer` = null [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1368–1370]. Treat early “semantics” cautiously; prefer rank milestones and within‑model trends.

Checklist: ✓ RMS lens; ✓ LayerNorm bias removed (RMS model: not needed); ✓ Entropy rises at unembed captured via FP32 shadow path (use_fp32_unembed=false but unembed_dtype is fp32); ✓ FP32 un‑embed promoted? AUTO path present [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:808–816]; ✓ Punctuation/markup anchoring observed (L29–31); ✗ Copy‑reflex (none in L0–3); ✓ Grammatical filler anchoring (top‑1 “simply” around L20–24) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:22–26].

**5. Limitations & Data Quirks**
- Lens‑sensitive KL: rely on rank milestones for cross‑model claims; treat KL trends qualitatively. Final KL≈0 indicates good final‑head calibration here.
- Raw‑vs‑norm check is sampled (`mode`: sample implied by `raw_lens_check`), `lens_artifact_risk` = high; interpret any pre‑final semantics cautiously [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1311,1368–1370].
- Surface‑mass depends on tokenizer; prefer within‑model trends. Rest_mass is top‑k coverage only; high values post‑semantic do not imply mis‑scale.
- Tuned‑lens regresses rank‑1 earliness (first_rank≤1 at L32); treat TL here as a calibration sidecar rather than a semantic lens.

**6. Model Fingerprint**
Mistral‑7B‑v0.1: semantic at L 25; final entropy 3.61 bits; quotes briefly top‑1 at L29–31 before “Berlin” reasserts at L32.

---
Produced by OpenAI GPT-5 

*Run executed on: 2025-09-28 17:22:48*
