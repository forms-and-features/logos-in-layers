# 1. Overview

Meta-Llama-3-70B (80 layers) probed on 2025-09-21 (run-latest). The sweep records per-layer next-token distributions, entropy (bits), KL to final, cosine-to-final, top-k tokens, and answer alignment for the prompt ending in “called simply.”

# 2. Method sanity-check

Diagnostics confirm the intended norm lens and positional handling are active, with FP32 unembed and RMSNorm-aware alignment:
- “use_norm_lens: true” (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:807)
- “unembed_dtype: "torch.float32"” (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:809)
- “layer0_position_info: "token_only_rotary_model"” (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:816)
- “norm_alignment_fix: "using_ln2_rmsnorm_for_post_block"” (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:813)

Prompt ends with “called simply” and no trailing space: “context_prompt: "Give the city name only, plain text. The capital of Germany is called simply"” (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:4).

Copy-collapse configuration present and ID-level, contiguous subsequence with k=1, τ=0.95, δ=0.10; no entropy fallback. Quoted from diagnostics:
- “copy_thresh: 0.95” (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:837)
- “copy_window_k: 1” (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:838)
- “copy_match_level: "id_subsequence"” (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:839)

Gold-token alignment is ID-based and resolved: “gold_alignment: "ok"” (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:845). Negative control present: “control_prompt” and “control_summary” blocks exist (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1032,1047).

Ablation summary exists and reports a semantic delay without “simply”: “L_sem_orig: 40 … L_sem_nf: 42 … delta_L_sem: 2” (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1026,1028,1030). Positive rows appear for both variants in the pure CSV (e.g., rows starting with `pos,orig,…` and `pos,no_filler,…`). For the main table below, rows are filtered to `prompt_id = pos`, `prompt_variant = orig`.

Summary indices (units = bits for KL and entropy):
- first_kl_below_0.5 = 80; first_kl_below_1.0 = 80 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:840-841)
- first_rank_le_1 = 40; first_rank_le_5 = 38; first_rank_le_10 = 38 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:842-844)

Last-layer head calibration is good: final CSV `kl_to_final_bits` ≈ 0 (0.000729) at L80 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:82), and diagnostics include a consistency block with “top1_agree: true … temp_est: 1.0 … warn_high_last_layer_kl: false” (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:846-854,864).

Lens sanity (raw vs norm): mode = “sample”; summary shows “lens_artifact_risk: "low"”, “max_kl_norm_vs_raw_bits: 0.04289584828470766”, and no norm-only semantics (first_norm_only_semantic_layer: null) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:962-963,1019-1021). Treat early “semantics” as genuine within this run but still prefer rank milestones over raw probabilities.

Copy-collapse flag check (pure CSV, layers 0–3): no `copy_collapse = True` found; e.g., L0–L3 show `copy_collapse=False` (rows 2–5 in CSV). Therefore: ✗ fired spuriously (no firing observed); rule did not trigger.

# 3. Quantitative findings

Table (pos/orig only): “L N — entropy X bits, top‑1 ‘token’”. Bold indicates the first semantic layer (ID-level gold token).

- L 0 — entropy 16.968 bits, top-1 ' winding'
- L 1 — entropy 16.960 bits, top-1 'cepts'
- L 2 — entropy 16.963 bits, top-1 'улю'
- L 3 — entropy 16.963 bits, top-1 'zier'
- L 4 — entropy 16.959 bits, top-1 'alls'
- L 5 — entropy 16.957 bits, top-1 'alls'
- L 6 — entropy 16.956 bits, top-1 'alls'
- L 7 — entropy 16.953 bits, top-1 'NodeId'
- L 8 — entropy 16.959 bits, top-1 'inds'
- L 9 — entropy 16.960 bits, top-1 'NodeId'
- L 10 — entropy 16.952 bits, top-1 'inds'
- L 11 — entropy 16.956 bits, top-1 'inds'
- L 12 — entropy 16.956 bits, top-1 'lia'
- L 13 — entropy 16.955 bits, top-1 'eds'
- L 14 — entropy 16.950 bits, top-1 'idders'
- L 15 — entropy 16.953 bits, top-1 ' Kok'
- L 16 — entropy 16.952 bits, top-1 '/plain'
- L 17 — entropy 16.948 bits, top-1 ' nut'
- L 18 — entropy 16.944 bits, top-1 ' nut'
- L 19 — entropy 16.948 bits, top-1 ' nut'
- L 20 — entropy 16.946 bits, top-1 ' nut'
- L 21 — entropy 16.938 bits, top-1 ' burge'
- L 22 — entropy 16.938 bits, top-1 ' simply'
- L 23 — entropy 16.936 bits, top-1 ' bur'
- L 24 — entropy 16.950 bits, top-1 ' bur'
- L 25 — entropy 16.937 bits, top-1 '�'
- L 26 — entropy 16.938 bits, top-1 '�'
- L 27 — entropy 16.937 bits, top-1 'za'
- L 28 — entropy 16.933 bits, top-1 '/plain'
- L 29 — entropy 16.933 bits, top-1 ' plain'
- L 30 — entropy 16.939 bits, top-1 'zed'
- L 31 — entropy 16.925 bits, top-1 ' simply'
- L 32 — entropy 16.941 bits, top-1 ' simply'
- L 33 — entropy 16.927 bits, top-1 ' plain'
- L 34 — entropy 16.932 bits, top-1 ' simply'
- L 35 — entropy 16.929 bits, top-1 ' simply'
- L 36 — entropy 16.940 bits, top-1 ' simply'
- L 37 — entropy 16.935 bits, top-1 ' simply'
- L 38 — entropy 16.934 bits, top-1 ' simply'
- L 39 — entropy 16.935 bits, top-1 ' simply'
- **L 40 — entropy 16.937 bits, top-1 ' Berlin'**
- L 41 — entropy 16.936 bits, top-1 ' "'
- L 42 — entropy 16.944 bits, top-1 ' "'
- L 43 — entropy 16.941 bits, top-1 ' Berlin'
- L 44 — entropy 16.926 bits, top-1 ' Berlin'
- L 45 — entropy 16.940 bits, top-1 ' "'
- L 46 — entropy 16.955 bits, top-1 ' "'
- L 47 — entropy 16.939 bits, top-1 ' "'
- L 48 — entropy 16.939 bits, top-1 ' "'
- L 49 — entropy 16.937 bits, top-1 ' "'
- L 50 — entropy 16.944 bits, top-1 ' "'
- L 51 — entropy 16.940 bits, top-1 ' "'
- L 52 — entropy 16.922 bits, top-1 ' Berlin'
- L 53 — entropy 16.933 bits, top-1 ' Berlin'
- L 54 — entropy 16.942 bits, top-1 ' Berlin'
- L 55 — entropy 16.942 bits, top-1 ' Berlin'
- L 56 — entropy 16.921 bits, top-1 ' Berlin'
- L 57 — entropy 16.934 bits, top-1 ' Berlin'
- L 58 — entropy 16.941 bits, top-1 ' Berlin'
- L 59 — entropy 16.944 bits, top-1 ' Berlin'
- L 60 — entropy 16.923 bits, top-1 ' Berlin'
- L 61 — entropy 16.940 bits, top-1 ' Berlin'
- L 62 — entropy 16.951 bits, top-1 ' Berlin'
- L 63 — entropy 16.946 bits, top-1 ' Berlin'
- L 64 — entropy 16.926 bits, top-1 ' Berlin'
- L 65 — entropy 16.933 bits, top-1 ' "'
- L 66 — entropy 16.941 bits, top-1 ' Berlin'
- L 67 — entropy 16.930 bits, top-1 ' Berlin'
- L 68 — entropy 16.924 bits, top-1 ' Berlin'
- L 69 — entropy 16.932 bits, top-1 ' Berlin'
- L 70 — entropy 16.926 bits, top-1 ' Berlin'
- L 71 — entropy 16.923 bits, top-1 ' Berlin'
- L 72 — entropy 16.922 bits, top-1 ' Berlin'
- L 73 — entropy 16.918 bits, top-1 ' "'
- L 74 — entropy 16.914 bits, top-1 ' Berlin'
- L 75 — entropy 16.913 bits, top-1 ' Berlin'
- L 76 — entropy 16.919 bits, top-1 ' Berlin'
- L 77 — entropy 16.910 bits, top-1 ' Berlin'
- L 78 — entropy 16.919 bits, top-1 ' Berlin'
- L 79 — entropy 16.942 bits, top-1 ' Berlin'
- L 80 — entropy 2.589 bits, top-1 ' Berlin'

Control margin (JSON control_summary): first_control_margin_pos = 0; max_control_margin = 0.5168457566906 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1048-1049).

Ablation (no-filler): L_copy_orig = null, L_sem_orig = 40; L_copy_nf = null, L_sem_nf = 42; ΔL_copy = null; ΔL_sem = 2 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1024-1030). Interpretation: removing “simply” delays semantics by 2 layers (~2.5% of depth), a small but present stylistic-cue sensitivity.

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (no copy-collapse under τ=0.95, δ=0.10).

Confidence milestones (pure CSV):
- p_top1 > 0.30 at layer 80; p_top1 > 0.60 at layer n.a.; final-layer p_top1 = 0.4783 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:82).

Rank milestones (diagnostics):
- rank ≤ 10 at layer 38; rank ≤ 5 at layer 38; rank ≤ 1 at layer 40 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:842-844).

KL milestones (diagnostics + CSV):
- first_kl_below_1.0 at layer 80; first_kl_below_0.5 at layer 80 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:840-841). KL decreases toward the final and is ≈ 0 at L80 (0.000729) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:82).

Cosine milestones (pure CSV):
- first cos_to_final ≥ 0.2: L80; ≥ 0.4: L80; ≥ 0.6: L80; final cos_to_final = 0.999989 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:82).

Prism Sidecar Analysis
- Presence: compatible = true (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:829-836).
- Early-depth stability (KL to final, bits): baseline vs Prism at sampled depths:
  - L0: 10.50 vs 10.67; L20: 10.45 vs 11.34; L40: 10.42 vs 11.42; L60: 10.31 vs 11.47; L80: 0.0007 vs 26.88 (computed from CSVs).
- Rank milestones (Prism): none achieved (no rank ≤10/5/1 over depth), vs baseline reaching rank 1 at L40 (computed from CSVs and diagnostics).
- Top‑1 agreement at sampled depths: baseline disagrees with final at L0/20/40/60; Prism also disagrees at all sampled depths.
- Cosine drift: baseline stays low until a sharp jump at L80; Prism remains near zero throughout early/mid layers (no earlier stabilization).
- Copy flags under Prism: no `copy_collapse=True` flips observed.
- Verdict: Regressive (KL increases and rank milestones are later/absent under Prism).

# 4. Qualitative patterns & anomalies

The model exhibits a classic late consolidation: answer rank reaches 1 at L40 while probabilities remain modest until the final head, where both entropy drops (to 2.59 bits) and p_top1 rises to 0.478. Cosine-to-final stays near zero through most of the stack and only saturates at L80, indicating late alignment of the distributional direction, not just calibration. This is “late direction, late calibration,” in contrast to early-direction signatures sometimes seen in tuned-lens studies (cf. Tuned-Lens 2303.08112) — here both direction and calibration concentrate at the top.

Negative control shows clean behavior with no leakage: “Berlin is the capital of” → top-1 “ Germany” (0.8516), then “ the” (0.0791) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:10-20). Quoted top‑5: > “ Germany, 0.8516; the, 0.0791; and, 0.0146; modern, 0.0048; Europe, 0.0031” (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:14-31). No semantic leakage of “Berlin.”

Records and pure CSVs together show that “Berlin” first enters any top‑5 by L38 and becomes top‑1 by L40; at L38 the top‑5 includes both “ Berlin” and “ Germany” while the generic filler “ simply” is still top‑1 (row 40 in pure CSV). Important-word trajectory: “simply” dominates top‑1 from L22–39; “Berlin” appears by L38, and stabilizes as frequent top‑1 beyond L43. This suggests prompt-style tokens (filler) anchor mid‑stack predictions before factual content wins out.

The collapse layer index shifts slightly when the one‑word instruction is ablated: L_semantic increases from 40 to 42 (ΔL_sem = +2), indicating a small reliance on the stylistic cue “simply” (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1026,1028,1030).

Rest-mass sanity: rest_mass declines substantially near the top head (e.g., L80 rest_mass = 0.1074; 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:82) and is high mid‑stack (max after L_semantic ≈ 0.9999 at L46, row 48), consistent with diffuse mid‑stack distributions and sharp final concentration. Do not use rest_mass as fidelity; it is top‑k coverage only.

Rotation vs amplification: KL-to-final decreases with depth and reaches ≈0 at L80, while answer rank improves (≤10 at L38, 1 at L40) and p_top1 peaks only at L80, with cosine-to-final also peaking there. This indicates joint rotation and sharpening near the head rather than a long early linear trajectory with only late rescaling.

Head calibration: “warn_high_last_layer_kl: false” with “temp_est: 1.0 … kl_to_final_bits: 0.000729” (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:846-854,864). No correction needed; treat final probabilities as calibrated for within‑model use.

Lens sanity: raw-vs-norm sampling shows “lens_artifact_risk: "low" … max_kl_norm_vs_raw_bits: 0.0429 … first_norm_only_semantic_layer: null” (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1019-1021), so early semantics are unlikely to be lens-induced in this run; still prefer rank thresholds for robustness.

Temperature robustness: at T=0.1, “Berlin” rank 1 (p = 0.9933; entropy 0.058 bits) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:669-681). At T=2.0, “Berlin” remains top‑1 but with p = 0.0357 and entropy 14.46 bits (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:736-776). Entropy rises dramatically with temperature as expected.

Checklist:
- RMS lens? ✓ (RMSNorm; norm_alignment_fix uses ln2 for post‑block)
- LayerNorm bias removed? n.a. (RMS model; “not_needed_rms_model”)
- Entropy rise at unembed? ✗; instead, sharp entropy drop at final head
- FP32 un-embed promoted? ✓ (“use_fp32_unembed: true” via FP32 shadow unembed, 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:808-809)
- Punctuation / markup anchoring? ✓ (mid‑stack top‑1 often quotes/“simply”)
- Copy-reflex? ✗ (no `copy_collapse=True` in L0–3)
- Grammatical filler anchoring? ✗ for {is,the,a,of} in L0–5; mid‑stack dominated by “ simply” nonetheless (L22–39)

# 5. Limitations & data quirks

- No copy-collapse under τ=0.95, δ=0.10; ΔH (L_copy vs L_semantic) is not computable; rely on rank milestones. 
- KL is lens-sensitive; even with low artifact risk, prefer qualitative KL trends and rank thresholds for cross‑model claims.
- Raw‑vs‑norm lens sanity is “sample” mode, not exhaustive; conclusions are supported but sampled.
- Rest_mass is not fidelity; its very high mid‑stack values reflect top‑k coverage, not calibration.

# 6. Model fingerprint

“Llama‑3‑70B: semantics at L 40; final entropy 2.59 bits; ‘Berlin’ becomes top‑1 by L 40 and probability consolidates only at the final head.”

---
Produced by OpenAI GPT-5
