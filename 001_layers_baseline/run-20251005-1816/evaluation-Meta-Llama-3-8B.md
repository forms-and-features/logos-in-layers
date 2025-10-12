# Evaluation Report: meta-llama/Meta-Llama-3-8B

*Run executed on: 2025-10-05 18:16:50*

**Overview**
- Model: Meta-Llama-3-8B; run timestamp 2025-10-05. The probe tracks copy vs. semantic onset using rank and KL milestones, cosine geometry, entropy drift, and lens diagnostics (raw vs. norm; tuned lens audit).

**Method sanity-check**
- Prompt & indexing: context ends with “called simply” and no trailing space: 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:4. Positive rows present: 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:2.
- Normalizer provenance: strategy “next_ln1” with pre-norm architecture, L0 uses blocks[0].ln1 and final uses ln_final: 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7235, 7241, 7530.
- Per-layer normalizer effect: normalization_spike=true; early spike at L0 resid_norm_ratio=18.187 and delta_resid_cos=0.5346: 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:833, 7243.
- Unembed bias: present=false, l2_norm=0.0 (cosines are bias‑free): 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:824–829.
- Environment & determinism: device="cpu", deterministic_algorithms=true, seed=316, torch=2.8.0+cu128: 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:8396–8401.
- Numeric health: any_nan=false, any_inf=false, layers_flagged=[]: 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7732–7740.
- Copy mask: size=6022; ignored_token_ids sample includes control/punctuation tokens: 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:934–940, 6977.
- Gold alignment: ok=true, variant=with_space, pieces=["ĠBerlin"]: 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7745–7752.
- Repeatability (1.39): status=skipped (reason=deterministic_env): 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7738–7742.
- Norm trajectory: shape="spike", slope≈0.113, r2≈0.953: 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:8227–8233.
- Measurement guidance: prefer_ranks=true; suppress_abs_probs=true; preferred_lens_for_reporting=tuned; use_confirmed_semantics=true: 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:9018–9024.

**Quantitative findings (layer‑by‑layer)**
- L 0 — entropy 16.9568 bits, top‑1 “itzer” [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:2]
- L 20 — entropy 16.8304 bits, top‑1 “',” [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:24]
- L 25 — entropy 16.8142 bits, top‑1 “Berlin” (answer_rank=1; answer_logit_gap≈0.4075) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:29]
- L 30 — entropy 16.7946 bits, top‑1 “Berlin” [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:34]
- L 32 — entropy 2.9610 bits, top‑1 “Berlin” (final KL=0) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:36; 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7759–7766]

- Semantic layer: L_semantic_norm=25 and L_semantic_confirmed=25 (confirmed_source=raw). Bolded layer in table uses confirmed semantics per guidance: 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:8202–8209, 9045–9052; 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-milestones.csv:3–4.
- Control margin: first_control_margin_pos=0; max_control_margin≈0.5186: 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:8449–8456.
- Entropy drift: entropy_gap_bits_p25≈13.87, p50≈13.88, p75≈13.91 (vs. teacher 2.961): 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:9086–9089, 7146.
- Normalizer snapshots: L0 resid_norm_ratio≈18.19; final ln_final resid_norm_ratio≈2.25, delta_resid_cos≈0.989: 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7243, 7530–7536.

**Qualitative findings**

4.1. Copy vs semantics (Δ‑gap)
The probe shows no early copy‑reflex: strict copy L_copy_strict is null across τ∈{0.70,0.80,0.90,0.95} and soft copy L_copy_soft is absent; stability tag “none” (no consistent threshold crossings) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7048–7068]. Early layers 0–3 in the pure CSV have copy flags false [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:2–5]. Δ̂ unavailable (delta_hat=null) and thus not reported [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:9050–9056].

4.2. Lens sanity: Raw‑vs‑Norm
Raw vs norm shows medium artefact risk with lens_artifact_score_v2≈0.459 and tier “medium” [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7226–7234, 7218–7224]. Symmetric/robust metrics indicate modest but non‑negligible discrepancies: js_divergence_p50≈0.0168, l1_prob_diff_p50≈0.2403, first_js_le_0.1=0, first_l1_le_0.5=0 [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:9058–9066]. Top‑K overlap has jaccard_raw_norm_p50≈0.408 and first_jaccard_raw_norm_ge_0.5 at layer 3 [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:9066–9070]. Prevalence: pct_layers_kl_ge_1.0≈0.03, n_norm_only_semantics_layers=5, earliest_norm_only_semantic=25; window shows norm‑only semantics at {25,27,28,29,30} with max KL≈5.26 bits near L_semantic [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7098–7120]. Given norm‑only near candidate semantics, prefer rank milestones and confirmed semantics for onset.

4.3. Tuned‑Lens analysis
The tuned lens is not “calibration‑only” (tuned_is_calibration_only=false) and is the preferred lens hint for semantics [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:9178–9184, 9022–9024]. Attribution shows rotation contributes the bulk of KL reductions: ΔKL_rot_p25≈1.91/ΔKL_temp_p25≈−0.13; ΔKL_rot_p50≈2.34/ΔKL_temp_p50≈−0.05; ΔKL_rot_p75≈2.61/ΔKL_temp_p75≈0.04; interaction_p50≈1.98 [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:8968–9010]. Rank earliness shifts later under tuned (first_rank_le_1 baseline=25 vs tuned=32) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:8918–8934]. Positional generalization is mixed (pos_ood_ge_0.96≈4.32 vs pos_in_dist_le_0.92≈5.50; gap≈−1.17) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:9108–9116]. Head mismatch is clean (kl_bits_tuned_final=0.0; tau_star_modelcal=1.0) and last‑layer consistency is exact (kl_to_final_bits=0.0; top1_agree=true) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:9116–9120, 7759–7766]. Baseline (norm) is reported alongside tuned where relevant per guidance.

4.4. KL, ranks, cosine, entropy milestones
Norm lens milestones: first_kl_below_1.0=32; first_kl_below_0.5=32; first_rank_le_10=24, le_5=25, le_1=25 [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:6986–6990]. Cosine milestones (norm): ge_0.2=20, ge_0.4=30, ge_0.6=32 [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7083–7088]. Final‑head calibration is exact (final KL≈0), so late‑layer alignment is strong [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7759–7766]. Entropy decreases sharply only at the final head (teacher≈2.961 vs. per‑layer entropy_bits ≈16–17 until L32), consistent with deep calibration; drift summary p25/p50/p75≈13.87/13.88/13.91 bits [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:9086–9089].

4.5. Prism (shared‑decoder diagnostic)
Prism present and compatible; however, KL increases substantially (e.g., p50 baseline≈11.73 vs prism≈20.01; Δ≈−8.29 bits) and rank milestones under prism are null, indicating degraded behavior [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:852–880]. Verdict: Regressive.

4.6. Ablation & stress tests
Style ablation: L_sem_orig=25 and L_sem_nf=25 (ΔL_sem=0), suggesting insensitivity to filler removal [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:8424–8432]. Control prompts: first_control_margin_pos=0; max_control_margin≈0.5186 [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:8449–8456]. Negative test prompt “Berlin is the capital of” yields top‑1 “ Germany”; “ Berlin” also appears in the top‑k list [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:10–16, 15–16]. Important‑word trajectory (records CSV) shows “Berlin” rising to top‑1 on internal tokens before final head (e.g., layer 24 pos=14 top‑1 “Berlin”): 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-records.csv:499; and strong presence at layers 22–23 (rows 465, 482–483).

4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓
- LayerNorm bias removed ✓
- FP32 unembed promoted ✓
- Punctuation / markup anchoring noted ✓
- Copy‑reflex ✗
- Preferred lens honored ✓
- Confirmed semantics reported ✓
- Dual‑lens artefact metrics (incl. v2, JS/Jaccard/L1) cited ✓
- Tuned‑lens audit (rotation/temp/positional/head) ✓
- normalization_provenance present (ln_source @ L0/final) ✓
- per‑layer normalizer effect present (resid_norm_ratio, delta_resid_cos) ✓
- deterministic_algorithms true ✓
- numeric_health clean ✓
- copy_mask plausible ✓
- milestones.csv or evaluation_pack.citations used ✓

---
Produced by OpenAI GPT-5 
