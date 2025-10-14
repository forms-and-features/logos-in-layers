# Evaluation Report: google/gemma-2-9b

*Run executed on: 2025-10-13 22:23:35*

**1. Overview**
- Model: `google/gemma-2-9b` (42 layers). Run date: 2025-10-13 22:23:35 (`timestamp-20251013-2223`:1).
- Probe measures surface copy vs. semantic emergence and tracks KL-to-final, rank milestones, cosine alignment, and entropy trajectories with lens diagnostics (raw vs. norm; tuned-lens audit).

**2. Method sanity-check**
- Prompt & indexing: context ends with “called simply” (no trailing space): "Give the city name only, plain text. The capital of Germany is called simply" (001_layers_baseline/run-latest/output-gemma-2-9b.json:820). Positive rows present (e.g., `pos,orig` at L0 and L42) [rows 2 and 49 in `output-gemma-2-9b-pure-next-token.csv`].
- Normalizer provenance: `arch=pre_norm`, `strategy=next_ln1`; L0 uses `blocks[0].ln1`, final uses `ln_final` (001_layers_baseline/run-latest/output-gemma-2-9b.json:5950,5960; 6410,6420).
- Per-layer normalizer effect: normalization spike flagged (diagnostics.flags.normalization_spike=true) (001_layers_baseline/run-latest/output-gemma-2-9b.json:834). Early deltas are smooth before L_semantic (e.g., L0 `resid_norm_ratio≈0.757`, `delta_resid_cos≈0.929`; L42 `resid_norm_ratio≈0.220`, `delta_resid_cos≈0.745`) (001_layers_baseline/run-latest/output-gemma-2-9b.json:5960; 6414–6420).
- Unembed bias: `present=false`, `l2_norm=0.0` (001_layers_baseline/run-latest/output-gemma-2-9b.json:835–846).
- Environment & determinism: `device=cpu`, torch 2.8, `deterministic_algorithms=true`, `seed=316` (001_layers_baseline/run-latest/output-gemma-2-9b.json:8898–8910).
- Numeric health: `any_nan=false`, `any_inf=false`, `layers_flagged=[]` (001_layers_baseline/run-latest/output-gemma-2-9b.json:6603–6610).
- Copy mask: `size=4668`; sample ignored tokens are whitespace/newlines (001_layers_baseline/run-latest/output-gemma-2-9b.json:5636–5650).
- Gold alignment: `ok=true`, pieces `▁Berlin` / variant `with_space` (001_layers_baseline/run-latest/output-gemma-2-9b.json:6688–6737).
- Repeatability (1.39): skipped due to deterministic env (`status=skipped`) (001_layers_baseline/run-latest/output-gemma-2-9b.json:6608–6614). Flag as unmeasured, not unstable.
- Norm trajectory: `shape="spike"`, `r2≈0.987` (001_layers_baseline/run-latest/output-gemma-2-9b.json:10798–10810,10828–10833).
- Measurement guidance: `prefer_ranks=true`, `suppress_abs_probs=true`, reasons include `warn_high_last_layer_kl`, `norm_only_semantics_window`, `high_lens_artifact_risk`, `normalization_spike`, `pos_window_low_stability`; preferred lens `norm`; `use_confirmed_semantics=true` (001_layers_baseline/run-latest/output-gemma-2-9b.json:10765–10781).
- Semantic margin: `{delta_abs=0.002, p_uniform=3.90625e-06, margin_ok_at_L_semantic_norm=true}` and `L_semantic_confirmed_margin_ok_norm=42` (001_layers_baseline/run-latest/output-gemma-2-9b.json:10641–10648).
- Gate-stability (small rescalings): at L=42, `uniform_margin_pass_frac=1.0`, `top2_gap_pass_frac=1.0`, `both_gates_pass_frac=1.0`; `min_both_gates_pass_frac=1.0` (001_layers_baseline/run-latest/output-gemma-2-9b.json:5786–5803).
- Position-window: grid `[0.2,0.4,0.6,0.8,0.92,0.98]`, `rank1_frac=0.1667` at L_semantic (position-fragile) (001_layers_baseline/run-latest/output-gemma-2-9b.json:10676–10706).
- Micro-suite: aggregates present, `n=5`, `n_missing=0`; medians: `L_semantic_confirmed=42`, `Δ̂=1.0` (001_layers_baseline/run-latest/output-gemma-2-9b.json:10804–10840).

**3. Quantitative findings (layer-by-layer)**
- Positive (`pos,orig`) snapshots from `output-gemma-2-9b-pure-next-token.csv`:
  - L 0 — entropy 0.0000167 bits, top‑1 ‘simply’ (row 2).
  - L 3 — entropy 0.00043 bits, top‑1 ‘simply’ (row 5).
  - L 10 — entropy 0.281 bits, top‑1 ‘simply’ (row 12).
  - L 21 — entropy 1.866 bits, top‑1 ‘simply’ (row 28).
  - L 32 — entropy 0.0625 bits, top‑1 punctuation (row 39).
  - L 42 — entropy 0.370 bits, top‑1 ‘Berlin’; answer_rank=1; answer_logit_gap≈2.59 (row 49).
- Semantic layer: L 42 — bolded as confirmed (margin_ok; preferred lens=norm). See milestones (001_layers_baseline/run-latest/output-gemma-2-9b-milestones.csv:3–4).
- Controls: `first_control_margin_pos=18`, `max_control_margin≈0.868`, `first_control_strong_pos=42` (001_layers_baseline/run-latest/output-gemma-2-9b.json:8955–8966).
- Micro‑suite: median `L_semantic_confirmed=42`, median `Δ̂=1.0`; e.g., Germany→Berlin cites row_index 42 (001_layers_baseline/run-latest/output-gemma-2-9b.json:10690–10728).
- Entropy drift: entropy gap percentiles vs teacher are negative (p25≈−2.89, p50≈−2.80, p75≈−1.63 bits) (001_layers_baseline/run-latest/output-gemma-2-9b.json:10820–10837).
- Normalizer snapshot: L0 (`resid_norm_ratio≈0.76`, `delta_resid_cos≈0.93`) vs L42 (`≈0.22`, `≈0.75`) (001_layers_baseline/run-latest/output-gemma-2-9b.json:5960; 6414–6420).

**4. Qualitative findings**

4.1. Copy vs semantics (Δ‑gap)
- Copy‑reflex ✓: early layers show copy collapse/soft‑copy (e.g., L0–L3 have `copy_collapse=True`) [rows 2,5 in `output-gemma-2-9b-pure-next-token.csv`].
- Depth gap: Δ̂ from surface‑copy to semantics = 1.0 (evaluation_pack.milestones.depth_fractions.delta_hat=1.0) (001_layers_baseline/run-latest/output-gemma-2-9b.json:10787–10803).
- Threshold stability: earliest strict copy at τ=0.70 and τ=0.95 both at L0; `norm_only_flags=false` and `stability="mixed"` (001_layers_baseline/run-latest/output-gemma-2-9b.json:5710–5732).

4.2. Lens sanity: Raw‑vs‑Norm
- Artifact risk: `lens_artifact_score_v2=0.5906`, tier=high (001_layers_baseline/run-latest/output-gemma-2-9b.json:6012–6020; 10804–10812).
- Symmetric metrics: `js_divergence_p50≈0.0063`, `l1_prob_diff_p50≈0.029`; earliest `first_js_le_0.1=0`, `first_l1_le_0.5=0` (001_layers_baseline/run-latest/output-gemma-2-9b.json:5834–5850; 10807–10812).
- Top‑K overlap: `jaccard_raw_norm_p50≈0.639`, `first_jaccard_raw_norm_ge_0.5=3` (001_layers_baseline/run-latest/output-gemma-2-9b.json:5847–5860).
- Prevalence: `pct_layers_kl_ge_1.0≈0.30`, `n_norm_only_semantics_layers=1`, `earliest_norm_only_semantic=42` (001_layers_baseline/run-latest/output-gemma-2-9b.json:5827–5847).
- Lens‑consistency at L=42 (norm vs raw): `jaccard@10≈0.43`, `jaccard@50≈0.25`, `spearman_top50≈0.44` (001_layers_baseline/run-latest/output-gemma-2-9b.json:6614–6648).
- Caution: early “semantics” near the candidate layer can be lens‑dependent; prefer rank/KL milestones and confirmed semantics.

4.3. Tuned‑Lens analysis
- Preference: `tuned_is_calibration_only=false`, but guidance prefers `norm` for semantics; `preferred_semantics_lens_hint=norm` (001_layers_baseline/run-latest/output-gemma-2-9b.json:10682–10690; 10775–10781).
- Attribution: ΔKL by component — `delta_kl_rot_p25≈−0.586`, `delta_kl_rot_p50≈0.00005`, `delta_kl_rot_p75≈0.058`; `delta_kl_temp_p25≈−0.742`, `p50≈−0.013`, `p75≈0.136`; interaction p50≈0.0026 (001_layers_baseline/run-latest/output-gemma-2-9b.json:10840–10866; 10588–10610).
- Rank earliness: unchanged — `first_rank_le_10/5/1 = 42` (baseline and tuned) (001_layers_baseline/run-latest/output-gemma-2-9b.json:10520–10544).
- Positional generalization: `pos_ood_ge_0.96=0.0`, `pos_in_dist_le_0.92≈0.00028`, small gap (001_layers_baseline/run-latest/output-gemma-2-9b.json:10854–10866; 10596–10610).
- Head mismatch: `kl_bits_tuned_final≈1.08 → 0.406` after `tau_star_modelcal≈2.85` (001_layers_baseline/run-latest/output-gemma-2-9b.json:10866–10874).
- Last‑layer agreement: `top1_agree=true` but `warn_high_last_layer_kl=true` (001_layers_baseline/run-latest/output-gemma-2-9b.json:6700–6720).

4.4. KL, ranks, cosine, entropy milestones
- KL: baseline norm lens does not cross 1.0 bit early; tuned lens reaches `first_kl_below_1.0=42` (001_layers_baseline/run-latest/output-gemma-2-9b.json:9060–9135; 9160–9220).
- Ranks: `first_rank_le_{10,5,1}=42` (preferred lens=norm; tuned unchanged) (001_layers_baseline/run-latest/output-gemma-2-9b.json:10520–10544).
- Cosine: `ge_0.2=1`, `ge_0.4=42`, `ge_0.6=42` under norm (001_layers_baseline/run-latest/output-gemma-2-9b.json:5748–5756).
- Entropy: drift is negative vs teacher across depth (percentiles above). Entropy is not strictly monotonic; later layers calibrate the head while ranks settle very late.
- Gates: uniform‑margin gate passes at L 42; top‑2 gate not asserted. Stability: no run‑of‑two; treat as potentially unstable in other settings. Advisory: position‑fragile and high artifact tier → emphasize ranks/KL over absolute probabilities.

4.5. Prism — shared‑decoder diagnostic
- Presence/compatibility: present (`k=512`, auto) (001_layers_baseline/run-latest/output-gemma-2-9b.json:844–862).
- KL deltas: prism increases KL strongly at p25/p50/p75 (e.g., Δp50≈−10.33 bits where negative means worse) with no earlier rank milestones (001_layers_baseline/run-latest/output-gemma-2-9b.json:904–938).
- Verdict: Regressive.

4.6. Ablation & stress tests
- Style ablation: `L_copy_orig=0`, `L_sem_orig=42`, `L_copy_nf=0`, `L_sem_nf=42`, so `ΔL_sem=0` (001_layers_baseline/run-latest/output-gemma-2-9b.json:8921–8936).
- Negative/control prompts: control summary indicates early control margin (`first_control_margin_pos=18`) and strong by L=42; test prompt “Berlin is the capital of” shows ‘Germany’ dominant and ‘Berlin’ present in top‑10 (001_layers_baseline/run-latest/output-gemma-2-9b.json:8955–8966; 10–52).
- Important‑word trajectory: at L 42, the answer is rank‑1 with a large positive logit gap (row 49 in `output-gemma-2-9b-pure-next-token.csv`).

4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓
- LayerNorm bias removed ✓
- FP32 unembed promoted ✓
- Punctuation / markup anchoring noted ✓
- Copy‑reflex ✓
- Preferred lens honored ✓
- Confirmed semantics reported ✓
- Dual‑lens artefact metrics (incl. v2; JS/Jaccard/L1) cited ✓
- Tuned‑lens audit done (rotation/temp/positional/head) ✓
- normalization_provenance present (ln_source @ L0/final) ✓
- per‑layer normalizer effect present (resid_norm_ratio, delta_resid_cos) ✓
- deterministic_algorithms true ✓
- numeric_health clean ✓
- copy_mask plausible ✓
- milestones.csv and evaluation_pack.citations used ✓
- gate_stability_small_scale reported ✓
- position_window stability reported ✓

---
Produced by OpenAI GPT-5
