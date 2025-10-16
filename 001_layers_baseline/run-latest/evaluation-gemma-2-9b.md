# Evaluation Report: google/gemma-2-9b

*Run executed on: 2025-10-16 07:26:19*
**Overview**

- Model: `gemma-2-9b` (42 layers); run artifacts dated 2025‑10‑16. Probe measures copy vs. semantics onset, KL/rank/cosine/entropy trajectories, and lens diagnostics (norm/raw/tuned; Prism diagnostic).
- Final‑head calibration check indicates non‑zero final KL; guidance flags and norm‑only risk are active, so ranks/KL are preferred over absolute probabilities.

**Method sanity‑check**

- Prompt & indexing: context prompt ends with “called simply” with no trailing space: "context_prompt": "... is called simply" (001_layers_baseline/run-latest/output-gemma-2-9b.json:4). Positive rows present for `prompt_id=pos, prompt_variant=orig` (e.g., pure CSV L0 and L42: rows 2 and 49).
- Normalizer provenance: strategy primary=`next_ln1`, ablation=`post_ln2_vs_next_ln1@targets` (001_layers_baseline/run-latest/output-gemma-2-9b.json:5953–5956). Per‑layer ln_source: L0=`blocks[0].ln1` (5959–5964); final=`ln_final` (6337–6344).
- Per‑layer normalizer effect: norm trajectory shape="spike" with small‑scale flag present (8605–8610; 5787–5803). Early large changes captured by `resid_norm_ratio`/`delta_resid_cos`, but flagged and occurring well before semantics; no candidate layer is flagged in `numeric_health.layers_flagged` (6607–6613).
- Unembed bias: present=false; l2_norm=0.0 (834–838). Cosine metrics are computed bias‑free by design.
- Environment & determinism: device=`cpu`, torch=2.8.0, deterministic_algorithms=true, seed=316 (provenance.env block; 6607–6621). Reproducibility is adequate.
- Repeatability (forward‑of‑two): mode=`skipped_deterministic`; pass1.layer=42, pass2.layer=null, delta_layers=null; topk_jaccard_at_primary_layer=null; gate.repeatability_forward_pass=null (6769–6788). Caution: forward‑of‑two check not executed due to deterministic mode.
- Decoding‑point ablation (pre‑norm): arch=`pre_norm`; gate.decoding_point_consistent=false (6622–6681). Example at `L_semantic_norm` (layer 42): rank1_agree=false, jaccard@10=0.538 (6641–6656). Interpretation: early semantics are decoding‑point sensitive; prefer confirmed semantics.
- Numeric health: any_nan=false, any_inf=false, layers_flagged=[] (6607–6613).
- Copy mask: size=4668 with plausible whitespace/punctuation/system IDs (5643–5646 and 5673–5699 show detector config; sample IDs present earlier in file).
- Gold alignment: ok=true; pieces=["▁Berlin"], first_id=12514; gold_alignment_rate=1.0 (6757–6767; 10915–10916).
- Repeatability (decode micro‑check §1.39) and forward‑of‑two (§1.54): max_rank_dev=0.0, p95_rank_dev=0.0, top1_flip_rate=0.0 (6614–6621; 10908–10913). No instability detected in one pass; forward‑of‑two skipped.
- Norm trajectory: shape="spike", slope=0.0517, r2=0.987, n_spikes=1 (10918–10923).
- Measurement guidance: prefer_ranks=true, suppress_abs_probs=true; reasons include warn_high_last_layer_kl, norm_only_semantics_window, high_lens_artifact_risk, normalization_spike, decoding_point_sensitive, pos_window_low_stability; preferred_lens_for_reporting=`norm`; use_confirmed_semantics=true (10854–10871).
- Semantic margin: δ_abs=0.002, p_uniform=3.9e‑06; margin_ok_at_L_semantic_norm=true; L_semantic_confirmed_margin_ok_norm=42 (5768–5775).
- Gate‑stability: min_both_gates_pass_frac=1.0 at L=42 (5787–5803). Not calibration‑sensitive under small rescalings.
- Position‑window: grid=[0.2,0.4,0.6,0.8,0.92,0.98]; rank1_frac=0.167 at L_sem=42 (8591–8604). Position‑fragile (<0.50).
- Micro‑suite: aggregates present; n=5, n_missing=0; L_semantic_confirmed_median=42; delta_hat_median=1.0 (evaluation_pack.micro_suite; 10976–11020).

**Quantitative findings (layer‑by‑layer)**

- L 0 — entropy 1.672e‑05 bits; top‑1 ‘ simply’ (copy); teacher_entropy_bits 2.9367; answer_rank=1468; KL_to_final=14.392 bits (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2).
- L 10 — entropy 0.7365 bits; top‑1 ‘ Germany’; answer_rank high; cosine to prompt remains high; consecutive Jaccard rising (001_layers_baseline/run-latest/output-gemma-2-9b-records.csv:122 and nearby rows 140–177).
- L 28 — entropy trending down; top‑K still prompt‑biased; topk_prompt_mass moderate (JSON summary: L_topk_decay_norm=28, topk_prompt_mass_at_L_norm=0.0471; 5744–5746).
- L 41 — sharp normalization contraction; resid_norm_ratio≈0.126; delta_resid_cos≈0.987 (6330–6334).
- L 42 — entropy 0.3701 bits; top‑1 ‘ Berlin’; answer_rank=1; KL_to_final=1.013 bits; cos_to_final≈0.9993 (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:49).
- Control margins: first_control_margin_pos=18; max_control_margin=0.868; first_control_strong_pos=42 (9044–9050).
- Micro‑suite (medians): L_semantic_confirmed=42; Δ̂=1.0 across facts; example fact citation ‘Germany→Berlin’ row 42 (10976–11020).
- Entropy drift: entropy_gap_bits median ≈ −2.80 bits (evaluation_pack.entropy; 10924–10928), consistent with steady sharpening through depth.

Bolded semantics: L 42 — L_semantic_confirmed (tuned; margin_ok) — decoding‑point sensitive (6622–6681; milestones: 10883–10887; milestones CSV row 3).

**Qualitative findings**

4.1. Copy vs semantics (Δ‑gap)

Early layers exhibit a copy reflex: strict copy onset at L_copy_strict=0; soft copy k=1 at L=0 (5645–5667). Semantic onset at L_semantic_norm=42 and confirmed at L=42, giving Δ̂=1.0 of depth (depth_fractions.delta_hat=1.0; 5756–5766 and 10888–10892). Threshold stability is “mixed”; earliest strict copy at τ=0.70 and τ=0.95 is L=0 with no norm‑only flags (5718–5736). Summary: a prompt‑echo phase persists across early layers, with semantics only at the final layer.

4.2. Lens sanity: Raw‑vs‑Norm

Artifact risk is high: lens_artifact_score_v2=0.591 (tier=high), JS_p50=0.0063, L1_p50=0.0292, first_js_le_0.1=0, first_l1_le_0.5=0 (10894–10906). Top‑K overlap median jaccard_raw_norm@50=0.639; first_jaccard_raw_norm_ge_0.5 at L=3 (10900–10906). Prevalence: pct_layers_kl_ge_1.0≈0.302; one norm‑only semantics layer at L=42 (raw_lens_full; 5828–5830). Lens‑consistency at the semantic target is moderate: jaccard@10≈0.429, spearman_top50≈0.442 (6683–6721). Caution: semantics appear only under the norm/tuned lens at L=42; treat early semantics as potentially lens‑induced and prefer confirmed semantics and rank/KL milestones.

4.3. Tuned‑Lens analysis

Guidance: tuned_is_calibration_only=false; preferred_semantics_lens_hint=`norm` (10958–10960). Attribution: ΔKL from rotation vs temperature shows modest improvements primarily from temperature at p25/p50 (delta_kl_temp_p25≈−0.742, p50≈−0.013; rotation p50≈0; interaction p50≈0.0026; 10929–10939). Last‑layer head mismatch improves with model‑calibrated temperature: KL_tuned_final 1.082 → 0.406 bits; τ*≈2.85 (10953–10957). Positional generalization is near‑neutral (pos_ood_ge_0.96=0.0; pos_in_dist_le_0.92≈0.00028; 10940–10951). Last‑layer agreement baseline shows high KL but top‑1 agree (6790–6795). Verdict: tuned lens is a calibration aid with small net impact on ranks; confirm semantics at L=42 under norm/tuned.

4.4. KL, ranks, cosine, entropy milestones

KL: first_kl_below_1.0=null; first_kl_below_0.5=null (5645–5656). Final‑row KL not ≈0; last‑layer consistency warns high KL (6790–6810). Ranks: first_rank_le_{10,5,1}=42 (5645–5656). Cosine: ge_0.2 at L=1; ge_{0.4,0.6} only at L=42 (5749–5754). Entropy: monotonic decrease and substantial drift vs teacher (entropy_gap_bits_p50≈−2.80; 10924–10928). Margin gate: uniform margin passes at L=42 (5768–5775). Stability: forward‑of‑two skipped; decoding‑point gate fails (6622–6681); position‑window rank‑1 fraction is low (8591–8604). Treat the onset as decoding‑point sensitive and position‑fragile; rely on confirmed semantics and normalized depth.

4.5. Prism

Presence/compatibility: present=true, compatible=true (845–853). KL deltas are strongly negative (baseline better): Δp50≈−10.33 bits; rank milestones unchanged/absent under Prism (876–897). Verdict: Regressive — Prism increases KL and does not produce earlier rank milestones.

4.6. Ablation & stress tests

No‑filler ablation: L_copy_orig=0 → L_copy_nf=0; L_sem_orig=42 → L_sem_nf=42; ΔL_copy=0; ΔL_sem=0 (9010–9017). Control prompt “Berlin is the capital of” shows high mass on “ Germany” (1–31), consistent with expected control behaviour (1–32). Important‑word trajectories in records show sustained dominance of ‘ Germany’ in early/mid layers (e.g., records CSV rows 15, 35, 54, 71, 88), with semantics only at L=42 (pure CSV row 49).

4.7. Checklist (✓/✗/n.a.)

- RMS lens ✓
- LayerNorm bias removed ✓
- FP32 unembed promoted ✓
- Punctuation / markup anchoring noted ✓
- Copy‑reflex ✓
- Preferred lens honored ✓
- Confirmed semantics reported ✓
- Dual‑lens artefact metrics (incl. lens_artifact_score_v2, JS/Jaccard/L1) cited ✓
- Tuned‑lens audit done (rotation/temp/positional/head) ✓
- normalization_provenance present (ln_source @ L0/final) ✓
- per‑layer normalizer effect present (resid_norm_ratio, delta_resid_cos) ✓
- deterministic_algorithms true (noted) ✓
- numeric_health clean ✓
- copy_mask plausible ✓
- milestones.csv and evaluation_pack.citations used ✓
- gate_stability_small_scale reported ✓
- position_window stability reported ✓

---
Produced by OpenAI GPT-5
