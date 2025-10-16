# Evaluation Report: meta-llama/Meta-Llama-3-70B

*Run executed on: 2025-10-16 07:26:19*
**EVAL**

**1. Overview**
- Model: Meta-Llama-3-70B (run-latest); run timestamp: 2025-10-16 19:54:12 CEST. File: 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json
- Probe measures copy-reflex vs semantic onset via norm-lens logit lens, tracking per-layer KL-to-final, ranks, cosine, and entropy, with raw-vs-norm lens diagnostics and stability gates (decoding-point, repeatability, small rescalings).

**2. Method Sanity‑Check**
- Prompt & indexing: context prompt ends with “called simply” (no trailing space): "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply"  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:4. Positive rows exist: "Germany→Berlin,0,pos,orig,0,..."  001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:2.
- Normalizer provenance: arch pre_norm; strategy primary next_ln1 with ablation post_ln2_vs_next_ln1: "arch": "pre_norm", "primary": "next_ln1"  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7429–7433. Per-layer ln_source: L0 uses blocks[0].ln1  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7416–7422; final uses ln_final (layer 80)  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8158–8164.
- Per-layer normalizer effect: early spikes are present but flagged: "normalization_spike": true  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:818–819. Early per-layer shows elevated resid_norm_ratio/delta_resid_cos (e.g., layer 1 delta_resid_cos≈0.689)  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7427–7433, 7429–7431.
- Unembed bias: "unembed_bias": { "present": false, "l2_norm": 0.0 }  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:834–841. Cosine metrics are bias-free by construction.
- Environment & determinism: device cpu; dtype_compute torch.bfloat16; deterministic_algorithms true; seed 316  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9536–9545.
- Repeatability (forward-of-two): skipped due to deterministic env; gate null: "mode": "skipped_deterministic", pass1.layer=40, pass2.layer=null, delta_layers=null, topk_jaccard_at_primary_layer=null  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8779–8791.
- Decoding-point ablation (pre-norm): gate.decoding_point_consistent=false; at L_semantic_norm (L=40): rank1_agree=true, jaccard@10=0.333  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8669–8680, 8673–8680, 8721–8723.
- Numeric health: any_nan=false, any_inf=false, layers_flagged=[]  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8653–8661.
- Copy mask: ignored_token_ids sample: [0,1,2,3,4,5,6,7,8,9,10,11,…]  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1008–1016. (Size field not emitted; sample matches tokenizer special/punctuation IDs.)
- Gold alignment: gold_alignment_rate=1.0; pieces ["ĠBerlin"], string "Berlin"  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8839–8846, 880–885.
- Repeatability (decode micro-check §1.39) and forward-of-two (§1.54): {max_rank_dev=0.0, p95_rank_dev=0.0, top1_flip_rate=0.0}  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8653–8661.
- Norm trajectory: shape "spike" (slope≈0.0498, r2≈0.937)  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9148–9154.
- Measurement guidance: {prefer_ranks=true, suppress_abs_probs=true, preferred_lens_for_reporting="norm", use_confirmed_semantics=true} with reasons including "decoding_point_sensitive"  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9719–9730.
- Semantic margin: {delta_abs=0.002, p_uniform≈7.80e-06, L_semantic_margin_ok_norm=80, margin_ok_at_L_semantic_norm=false}  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9586–9593.
- Gate-stability: min_both_gates_pass_frac=0.0; at L_semantic_norm, both_gates_pass_frac=0.0 (calibration-sensitive)  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7169–7177, 7199–7205.
- Position-window: grid=[], rank1_frac=null at L_semantic_norm=40 (no multi-position generalization measured)  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9637–9643.
- Micro-suite: evaluation_pack.micro_suite.aggregates present; n=5; L_semantic_confirmed_median=40; no missing facts (n_missing=0)  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9856–9860.

**3. Quantitative Findings (Layer‑by‑Layer)**
- Positive baseline rows (pos/orig):
  - L 0 — entropy 16.97 bits; top-1 “winding”; answer_rank=115765  (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:2)
  - L 38 — entropy 16.93 bits; top-1 “simply”; answer_rank=3  (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:40)
  - L 40 — entropy 16.94 bits; top-1 “ Berlin”; answer_rank=1  (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:42)
  - L 80 — entropy 2.59 bits; top-1 “ Berlin”; answer_rank=1  (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:82)
- Semantic layer: L 40 — L_semantic_confirmed=40 (source=raw); decoding-point sensitive (pre_norm with gate=false). "L_semantic_confirmed": {"layer": 40, "source": "raw"}  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9746–9750; "decoding_point_consistent": false  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8721–8723.
- Control margins: first_control_margin_pos=0; max_control_margin≈0.517; first_control_strong_pos=80  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9593–9599.
- Micro-suite: median L_semantic_confirmed=40; median L_semantic_norm=51 (IQR 49–60); n_missing=0; example citation (Germany→Berlin) row_index=40  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9626–9636, 9807–9815.
- Entropy drift: entropy_gap_bits_p50≈14.34 bits; teacher_entropy_bits≈2.597 bits  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:976–980 (percentiles), 7198–7201 (teacher), 9161–9164 (final entropy).

**4. Qualitative Findings**

4.1. Copy vs semantics (Δ‑gap)
No strict or soft copy collapse is detected at any threshold: L_copy_strict and L_copy_soft[k∈{1,2,3}] are null; stability label “none”  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7069–7077, 7081–7095, 7094–7095. Δ̂ is therefore undefined (depth_fractions.delta_hat=null)  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7148–7156, 9750–9756. Copy-reflex ✓ is not triggered by early layers (0–3) in the pure CSV (no copy flags on rows)  001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:2.
Copy-thresholds stability tag: tau∈{0.70,0.80,0.90,0.95} all null; norm_only_flags[τ]=null  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7169–7177, 7182–7193.

4.2. Lens sanity: Raw‑vs‑Norm
Artifact scores (legacy/v2) are ~0.332/~0.344, tier=medium  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7423–7427, 9754–9761. Symmetric metrics are strong: js_divergence_p50≈0.00245; l1_prob_diff_p50≈0.092; first_js_le_0.1=0; first_l1_le_0.5=0  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9756–9763. Top‑K overlap: jaccard_raw_norm_p50≈0.515; first ≥0.5 at L=11  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9759–9761. Prevalence: pct_layers_kl_ge_1.0≈0.0123; n_norm_only_semantics_layers=2; earliest at 79  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7198–7205, 7228–7234. Caution: norm‑only “semantics” occur late (79–80), far from candidate L=40, so early semantics are unlikely lens‑induced.
Lens-consistency at targets is moderate/strong (norm vs raw): at L=40, jaccard@10≈0.818; spearman_top50≈0.658  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8749–8756.

4.3. Tuned‑Lens analysis
Tuned lens artifacts are not present for this run: "tuned_lens": {"status": "missing"}  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9597–9603. Per guidance, semantics are reported under the norm lens; last‑layer agreement is excellent (final KL≈0.00073; top1_agree=true)  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8802–8807.

4.4. KL, ranks, cosine, entropy milestones
KL thresholds (norm): first_kl_below_1.0 = 80; first_kl_below_0.5 = 80  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7044–7045. Final‑head calibration is excellent: kl_to_final_bits≈0.00073; top1_agree=true  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8802–8807.
Ranks (norm lens): first_rank_le_10=38; first_rank_le_5=38; first_rank_le_1=40  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7046–7048. Cosine milestones (norm): ge_{0.2,0.4,0.6} all at 80  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7141–7148. Entropy declines sharply only at the end: final entropy≈2.597 bits  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9148–9154; teacher_entropy_bits≈2.597 and entropy_gap_bits_p50≈14.34 bits  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7247, 9787–9789.
Margin gate reminder: margin_ok_at_L_semantic_norm=false, so rank‑1 at L=40 is weak under uniform‑margin gate; we therefore prefer confirmed semantics at the same layer  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9589–9593, 9746–9750. Stability reminder: run‑of‑two not available (skipped_deterministic); gate_stability_small_scale indicates calibration sensitivity (both_gates_pass_frac=0.0)  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8779–8791, 7169–7177.

4.5. Prism (shared‑decoder diagnostic)
Prism artifacts are present as sidecar CSVs; JSON deltas indicate regressive behavior at p50: ΔKL_p50≈−1.00 bits and no rank milestone improvement  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:846–906. Verdict: Regressive.

4.6. Ablation & stress tests
Style ablation (no‑filler) mildly delays semantics: L_sem_nf=42 vs L_sem_orig=40 (ΔL_sem=2)  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9539–9546. Negative/control prompt summary: first_control_margin_pos=0; max_control_margin≈0.517; first_control_strong_pos=80  001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9593–9599. Important‑word trajectory: by L≈31 and L≈40, “Berlin” appears among salient candidates across prompt positions (e.g., L=31 pos=13 includes “ Berlin”; L=40 pos=16 top-1=” Berlin”)  001_layers_baseline/run-latest/output-Meta-Llama-3-70B-records.csv:680; 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:42.

4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓
- LayerNorm bias removed ✓
- FP32 unembed promoted ✓
- Punctuation / markup anchoring noted ✓
- Copy‑reflex ✗ (no early copy collapse)
- Preferred lens honored ✓ (norm)
- Confirmed semantics reported ✓ (L=40)
- Dual‑lens artefact metrics (incl. v2, JS/Jaccard/L1) cited ✓
- Tuned‑lens audit done ✓ (status missing; baseline used)
- normalization_provenance present ✓ (ln_source @ L0/ln_final)
- per‑layer normalizer effect present ✓ (resid_norm_ratio, delta_resid_cos)
- deterministic_algorithms true ✓
- numeric_health clean ✓
- copy_mask plausible ✓
- milestones.csv or evaluation_pack.citations used ✓ (citations provided with line numbers)
- gate_stability_small_scale reported ✓
- position_window stability reported ✓ (null/no positions)

---
Produced by OpenAI GPT-5
