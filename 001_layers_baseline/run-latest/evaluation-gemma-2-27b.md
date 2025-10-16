# Evaluation Report: google/gemma-2-27b

*Run executed on: 2025-10-16 07:26:19*
## EVAL

**Overview**
- Model: `google/gemma-2-27b` (pre-norm, RMSNorm) — single-fact probe on next-token prediction. The run measures copy-reflex vs semantic onset with rank/KL/cosine/entropy trajectories and dual-lens diagnostics (raw vs norm; tuned calibration). Model/device and deterministic setup are recorded in `provenance.env` (cpu, torch 2.8, deterministic true; seed 316) 001_layers_baseline/run-latest/output-gemma-2-27b.json:9065,9067,9069.

**Method Sanity‑Check**
- Prompt & indexing: Context prompt ends with “called simply” (no trailing space): "Give the city name only, plain text. The capital of Germany is called simply" 001_layers_baseline/run-latest/output-gemma-2-27b.json:4. Positive rows exist (e.g., `prompt_id=pos`, `prompt_variant=orig`) in pure CSV [row 1] 001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:1.
- Normalizer provenance: `arch: "pre_norm"`; strategy `primary: "next_ln1"` with ablation `post_ln2_vs_next_ln1@targets` 001_layers_baseline/run-latest/output-gemma-2-27b.json:5965-5966. Sources at edges: L0 uses `blocks[0].ln1` 001_layers_baseline/run-latest/output-gemma-2-27b.json:5973; final uses `ln_final` 001_layers_baseline/run-latest/output-gemma-2-27b.json:6387.
- Per‑layer normalizer effect: Early layers show large `delta_resid_cos` values (e.g., 0.81–0.87 at L5–L16) and a spike flag is set (`flags.normalization_spike: true`) 001_layers_baseline/run-latest/output-gemma-2-27b.json:6022,6121,842. Treat early semantics as potentially sensitive to normalization choice.
- Unembed bias: `unembed_bias.present: false`, `l2_norm: 0.0`; cosines are bias‑free 001_layers_baseline/run-latest/output-gemma-2-27b.json:834.
- Environment & determinism: `device: "cpu"`, `deterministic_algorithms: true`, `seed: 316` 001_layers_baseline/run-latest/output-gemma-2-27b.json:808,9067,9069. Reproducibility OK.
- Repeatability (forward‑of‑two): Mode `skipped_deterministic`; pass1.layer=46, pass2.layer=null, `delta_layers`: null; `topk_jaccard_at_primary_layer`: null; `gate.repeatability_forward_pass`: null 001_layers_baseline/run-latest/output-gemma-2-27b.json:6842-6858. Caution: repeatability-forward was skipped; prefer confirmed/run‑of‑two milestones if available.
- Decoding‑point ablation (pre‑norm): `gate.decoding_point_consistent: false` 001_layers_baseline/run-latest/output-gemma-2-27b.json:6753. At `L_semantic_norm` target (L46): `rank1_agree: false`, `jaccard@10: 0.0526` 001_layers_baseline/run-latest/output-gemma-2-27b.json:6695-6734. Mark semantic onset as decoding‑point sensitive.
- Numeric health: `any_nan: false`, `any_inf: false`, `layers_flagged: []` 001_layers_baseline/run-latest/output-gemma-2-27b.json:6680-6685.
- Copy mask: `ignored_token_ids` present with a long list (e.g., 108–151, 226–231, 245–281, …) 001_layers_baseline/run-latest/output-gemma-2-27b.json:960-1060. Size not explicitly recorded; sample appears plausible for whitespace/punctuation/control tokens for the tokenizer.
- Gold alignment: `gold_alignment.ok: true`; `gold_alignment_rate: 1.0`; gold pieces: `▁Berlin` 001_layers_baseline/run-latest/output-gemma-2-27b.json:6830,6899,6906-6914.
- Repeatability (decode micro‑check §1.39) and forward‑of‑two (§1.54): `max_rank_dev: 0.0`, `p95_rank_dev: 0.0`, `top1_flip_rate: 0.0` 001_layers_baseline/run-latest/output-gemma-2-27b.json:6687-6693. Fine under deterministic run; forward-of-two skipped (see above).
- Norm trajectory: `shape: "spike"`, `slope: 0.0878`, `r2: 0.846`, `n_spikes: 16` 001_layers_baseline/run-latest/output-gemma-2-27b.json:8678-8688,10997-11001.
- Measurement guidance: `prefer_ranks: true`, `suppress_abs_probs: true`; `preferred_lens_for_reporting: "norm"`; `use_confirmed_semantics: true`; reasons include `warn_high_last_layer_kl`, `norm_only_semantics_window`, `decoding_point_sensitive`, `pos_window_low_stability` 001_layers_baseline/run-latest/output-gemma-2-27b.json:10931-10955.
- Semantic margin: `delta_abs: 0.002`, `p_uniform: 3.90625e-06`, `margin_ok_at_L_semantic_norm: true`; `L_semantic_confirmed_margin_ok_norm: 46` 001_layers_baseline/run-latest/output-gemma-2-27b.json:10808-10816.
- Gate‑stability: `min_both_gates_pass_frac: 1.0` and at L46 both gates pass (`uniform_margin_pass_frac: 1.0`, `top2_gap_pass_frac: 1.0`) 001_layers_baseline/run-latest/output-gemma-2-27b.json:5792-5810.
- Position‑window: grid `[0.2, 0.4, 0.6, 0.8, 0.92, 0.98]`; `L_semantic_norm: 46`; `rank1_frac: 0.1667` 001_layers_baseline/run-latest/output-gemma-2-27b.json:10842-10852. Position‑fragile (<0.50).
- Micro‑suite: Aggregates present (`n=5`, `n_missing=0`). Medians: `L_semantic_confirmed_median=46`, `delta_hat_median=1.0` 001_layers_baseline/run-latest/output-gemma-2-27b.json:11055-11150.

**Quantitative Findings (Layer‑by‑Layer)**
- L 0 — entropy 0.00050 bits, top‑1 ‘simply’ [row 1] 001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:1.
- L 3 — entropy 0.88567 bits, top‑1 ‘simply’ (copy flags set at this depth) [row 4] 001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:4.
- L 44 — entropy 7.56833 bits, top‑1 ‘dieſem’ [row 46] 001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:46.
- L 45 — entropy 7.14057 bits, top‑1 ‘Geſch’ [row 47] 001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:47.
- L 46 — entropy 0.11805 bits, top‑1 ‘Berlin’ (rank‑1) [row 48] 001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48.

- Control margins: `first_control_margin_pos: 0`, `first_control_strong_pos: 46` (max control top‑2 logit gap 4.71) 001_layers_baseline/run-latest/output-gemma-2-27b.json:11024-11036. Given `suppress_abs_probs=true`, avoid absolute p‑margin interpretation.
- Micro‑suite (medians): `L_semantic_confirmed_median = 46`, `delta_hat_median = 1.0`; example fact row [Germany→Berlin → row 46] 001_layers_baseline/run-latest/output-gemma-2-27b.json:10827-10839.
- Entropy drift: median entropy gap vs teacher `entropy_gap_bits_p50 = 4.68` 001_layers_baseline/run-latest/output-gemma-2-27b.json:11003-11005. Late calibration aligns with rank‑1 at L46 while earlier layers show higher entropy.
- Normalizer snapshots: early `delta_resid_cos` are large (e.g., 0.81–0.87 in mid‑early layers) and a normalization spike is flagged 001_layers_baseline/run-latest/output-gemma-2-27b.json:6022,6121,842.

**Qualitative Findings**

Copy vs semantics (Δ‑gap)
- Copy‑reflex ✓: Pure CSV shows `copy_collapse=True` / `copy_strict@τ` hits within layers ≤3 (e.g., row with copy flags at L3) 001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:4. With strict copy null at τ=0.95 but early soft copy at k=1, overall Δ̂ = 1.0 of depth (from evaluation_pack) 001_layers_baseline/run-latest/output-gemma-2-27b.json:10967-10970. Copy thresholds stability “mixed”; earliest strict copy at τ=0.70 and 0.80 is L0 in summary and L3 in tuned summaries (no norm‑only flags) 001_layers_baseline/run-latest/output-gemma-2-27b.json:5716-5726,7192-7200.
- Semantic onset: Bold layer is L46 as confirmed semantics (see below). Depth fraction `semantic_frac: 1.0` and `delta_hat: 1.0` 001_layers_baseline/run-latest/output-gemma-2-27b.json:5761-5772,10967-10970.

Lens sanity: Raw‑vs‑Norm
- Artifact tier: `lens_artifact_score_v2: 1.0`, `risk_tier: "high"` 001_layers_baseline/run-latest/output-gemma-2-27b.json:10970-10984. Symmetric metrics: `js_divergence_p50: 0.8653`, `l1_prob_diff_p50: 1.893`, with no early small‑divergence onsets (`first_js_le_0.1 = 0`, `first_l1_le_0.5 = 0`) 001_layers_baseline/run-latest/output-gemma-2-27b.json:10976-10981.
- Top‑K overlap: `jaccard_raw_norm_p50: 0.5625`, `first_jaccard_raw_norm_ge_0.5: 1` 001_layers_baseline/run-latest/output-gemma-2-27b.json:10980-10981. Prevalence: `pct_layers_kl_ge_1.0: 0.979`, `n_norm_only_semantics_layers: 1`, `earliest_norm_only_semantic: 46` 001_layers_baseline/run-latest/output-gemma-2-27b.json:5834,5836-5837.
- Lens‑consistency at semantics: norm vs raw at target layer yields `jaccard@10: 0.0`, `spearman_top50: 0.238` 001_layers_baseline/run-latest/output-gemma-2-27b.json:6669-6680. Caution: early “semantics” are view‑dependent; prefer rank milestones and confirmed semantics.

Tuned‑Lens analysis
- Preference: `tuned_is_calibration_only: true`; we treat tuned lens as calibration aid, and report semantics under norm; still, confirmed semantics uses tuned for verification 001_layers_baseline/run-latest/output-gemma-2-27b.json:11021-11031; 10931-10955.
- Attribution (ΔKL): rotation small/near‑zero at p50 (`-0.029` bits), temperature dominant (`+0.528` bits), mild interaction (`+0.028` bits) 001_layers_baseline/run-latest/output-gemma-2-27b.json:11021-11031.
- Rank earliness: No earlier tuned rank milestones vs norm (both `first_rank_le_1 = 46`) per summaries 001_layers_baseline/run-latest/output-gemma-2-27b.json:7050-7140.
- Positional generalization: `pos_in_dist_le_0.92: 0.435`, `pos_ood_ge_0.96: 0.519`, `pos_ood_gap: 0.084` 001_layers_baseline/run-latest/output-gemma-2-27b.json:11031-11045.
- Head mismatch: `kl_bits_tuned_final: 1.133` → `0.548` after `tau_star_modelcal: 3.008` 001_layers_baseline/run-latest/output-gemma-2-27b.json:11045-11053. Last‑layer consistency: `kl_to_final_bits: 1.135`, `warn_high_last_layer_kl: true` 001_layers_baseline/run-latest/output-gemma-2-27b.json:6862-6880.

KL, ranks, cosine, entropy milestones
- KL thresholds: `first_kl_below_1.0: 46` (tuned summaries include runs with 46; baseline norm shows no earlier <1.0 in these summaries) 001_layers_baseline/run-latest/output-gemma-2-27b.json:7200-7286. Final KL not ~0; last‑head calibration warning is active 001_layers_baseline/run-latest/output-gemma-2-27b.json:6862-6880.
- Ranks: Preferred lens (norm): `first_rank_le_{10,5,1} = 46` 001_layers_baseline/run-latest/output-gemma-2-27b.json:7050-7140. Given decoding‑point inconsistency, treat early rank interpretations as sensitive.
- Cosine milestones (norm): `ge_0.2: 1`, `ge_0.4: 46`, `ge_0.6: 46` 001_layers_baseline/run-latest/output-gemma-2-27b.json:5754-5761.
- Entropy: Median entropy gap vs teacher at p50 is 4.68 bits, consistent with high early uncertainty then sharp late calibration 001_layers_baseline/run-latest/output-gemma-2-27b.json:11003-11005.
- Margin gate reminder: `margin_ok_at_L_semantic_norm: true` 001_layers_baseline/run-latest/output-gemma-2-27b.json:10808-10816. Stability: forward-of-two skipped; position-window rank‑1 fraction 0.17 marks position‑fragility.

Prism (shared‑decoder diagnostic)
- Presence/compatibility: present, compatible, `k=512` with sampled layers 001_layers_baseline/run-latest/output-gemma-2-27b.json:847-856.
- KL deltas: large KL drops vs baseline at p25/p50/p75 (~22.6/23.7/23.1 bits) but rank milestones remain unavailable/unchanged (`first_rank_le_1: null`) 001_layers_baseline/run-latest/output-gemma-2-27b.json:847-890.
- Verdict: Neutral — clear KL reductions early, but no improvement in rank milestones under the diagnostic lens.

Ablation & stress tests
- With/without filler: `L_copy_orig: 0`, `L_sem_orig: 46`, `L_copy_nf: 3`, `L_sem_nf: 46`, `delta_L_copy: 3`, `delta_L_sem: 0` 001_layers_baseline/run-latest/output-gemma-2-27b.json:111-124. No stylistic sensitivity on semantic onset (ΔL_sem=0).
- Control prompt (France→Paris): alignment ok; control summary shows early margin (pos 0) and strong control by L46 001_layers_baseline/run-latest/output-gemma-2-27b.json:11007-11036.

Checklist (✓/✗/n.a.)
- RMS lens ✓
- LayerNorm bias removed ✓
- FP32 unembed promoted ✓
- Punctuation / markup anchoring noted ✓
- Copy‑reflex ✓
- Preferred lens honored ✓
- Confirmed semantics reported ✓
- Dual‑lens artefact metrics (incl. v2, JS/Jaccard/L1) cited ✓
- Tuned‑lens audit done (rotation/temp/positional/head) ✓
- normalization_provenance present (ln_source @ L0/final) ✓
- per‑layer normalizer effect present (resid_norm_ratio, delta_resid_cos) ✓
- deterministic_algorithms true ✓
- numeric_health clean ✓
- copy_mask plausible ✓
- milestones.csv or evaluation_pack.citations used for quotes ✓
- gate_stability_small_scale reported ✓
- position_window stability reported ✓

**Semantic Onset (headline)**
- Bolded layer: L46 — confirmed semantics at the final layer under norm lens with tuned‑lens confirmation and uniform‑margin gate OK (decoding‑point sensitive; position‑fragile). Citations: milestones rows at L46 001_layers_baseline/run-latest/output-gemma-2-27b-milestones.csv:3-4; measurement guidance and margin 001_layers_baseline/run-latest/output-gemma-2-27b.json:10931-10955,10808-10816; decoding‑point gate 001_layers_baseline/run-latest/output-gemma-2-27b.json:6753.

---
Produced by OpenAI GPT-5 
