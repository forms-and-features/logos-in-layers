# Cross-Model Probe: Capital-Fact (Rank/KL Lens)

This cross-evaluation synthesizes the latest runs under `001_layers_baseline/run-latest/` across 10 models. All models are pre-norm with `next_ln1` normalization. Guidance is honored per each model’s `measurement_guidance` (favor ranks; suppress absolute probabilities when flagged) and the final-row calibration caveat (avoid inferring probability regressions when last-layer KL is non-zero).

- Preferred semantics lens per model: gemma-2-9b, gemma-2-27b, Qwen2.5-72B, Llama‑3‑70B → `norm`; Qwen3‑8B/14B, Llama‑3‑8B, Mistral‑7B, Mistral‑Small‑24B, Yi‑34B → `tuned` hint (we still cite baseline `norm` for context).
- All architecture detectors report `arch=pre_norm`, `strategy=next_ln1`.

Citations include milestones and CSV line references (e.g., `output-<MODEL>-milestones.csv:4`; `output-<MODEL>-pure-next-token.csv:33`).

---

## 1. Result Synthesis

- Semantic onset (depth‑normalized):
  - Early within-depth: Meta‑Llama‑3‑70B ≈ 0.50 (40/80; confirmed; source=raw) (milestone: `output-Meta-Llama-3-70B-milestones.csv:3`).
  - Mid/late: Yi‑34B ≈ 0.73 (44/60; confirmed; source=tuned) (milestone: `output-Yi-34B-milestones.csv:3`).
  - Late: Llama‑3‑8B ≈ 0.78 (25/32; confirmed; source=raw) (milestone: `output-Meta-Llama-3-8B-milestones.csv:3`); Mistral‑7B ≈ 0.78 (25/32; confirmed) (milestone: `output-Mistral-7B-v0.1-milestones.csv:3`); Mistral‑Small‑24B ≈ 0.83 (33/40; confirmed) (milestone: `output-Mistral-Small-24B-Base-2501-milestones.csv:3`); Qwen3‑8B ≈ 0.86 (31/36; confirmed; source=raw) (milestone: `output-Qwen3-8B-milestones.csv:4`); Qwen3‑14B ≈ 0.90 (36/40; confirmed; source=raw) (milestone: `output-Qwen3-14B-milestones.csv:3`).
  - Final‑layer only: gemma‑2‑9b = 1.00 (42/42); gemma‑2‑27b = 1.00 (46/46); Qwen2.5‑72B = 1.00 (80/80). Milestones confirm semantics at the final layer (e.g., `output-gemma-2-27b-milestones.csv:4`, `output-gemma-2-9b-milestones.csv:4`, `output-Qwen2.5-72B-milestones.csv:3`).
- Δ̂ medians (when reported): Qwen3‑8B shows a small positive median Δ̂≈0.056 (evaluation_pack micro‑suite aggregates; cites fact rows at 31/104/180/252/328 in JSON). Most others report null or Δ̂=1.0 when semantics occurs only at the final layer (gemma‑2‑9b/27b).
- Tuned‑lens audit:
  - Gemma family exhibits substantial head mismatch and calibrated final heads: `tau_star_modelcal≈2.85–3.01`; last‑layer KL falls after τ⋆ (e.g., gemma‑2‑27b: KL_last≈1.135 → KL_after_temp≈0.567; JSON `diagnostics.last_layer_consistency`).
  - Qwen3, Llama‑3‑8B, Mistral family show τ⋆≈1.0 with near‑zero final KL; tuned lens preferred as a reporting lens hint but acts beyond pure calibration (rotation contributes; e.g., Qwen3‑8B rotation ΔKL p50≈0.92; JSON tuned audit).
- Head mismatch & last‑layer KL: Gemma shows non‑zero last‑layer KL (calibrated head), while Llama‑3‑70B, Qwen3‑8B/14B, Mistral, Yi have near‑zero last‑layer KL.

---

## 2. Copy Reflex (layers 0–3)

Criterion: any `copy_collapse` OR `copy_soft_k1@τ_soft` true in layers 0–3 (pos/orig) → “copy‑reflex”.

- Copy‑reflex present: gemma‑2‑9b; gemma‑2‑27b. Example early rows show prompt echo tokens as top‑1 in L0–L3 (e.g., gemma‑2‑27b: `output-gemma-2-27b-pure-next-token.csv:1,4,5`).
- No early copy‑reflex: Qwen2.5‑72B; Qwen3‑8B; Qwen3‑14B; Meta‑Llama‑3‑8B; Meta‑Llama‑3‑70B; Mistral‑7B; Mistral‑Small‑24B; Yi‑34B. Example citations: Qwen3‑14B layers 0–3 (no copy flags) `output-Qwen3-14B-pure-next-token.csv:2–5` (quote a single line: `output-Qwen3-14B-pure-next-token.csv:2`).
- Control strength (lexical leakage): When `first_control_strong_pos` present, we mark strong; otherwise treat positive control margins as weak calibration noise.
  - Strong: Qwen3‑8B (`first_control_strong_pos=30`), Qwen3‑14B (=36), Meta‑Llama‑3‑8B (=25), Mistral‑Small‑24B (=31), Yi‑34B (=42), gemma‑2‑27b (=46), gemma‑2‑9b (=42).
  - Weak/none: Qwen2.5‑72B (no strong pos reported).

---

## 3. Lens Artefact Risk

Reported from `diagnostics.raw_lens_full` (score v2, percentiles, overlap, prevalence):

- High‑tier families/variants: Qwen3‑8B (score_v2≈0.704; js_p50≈0.358; jaccard_p50≈0.282; pct_kl≥1.0≈0.757); Qwen3‑14B (≈0.704; js_p50≈0.513); Qwen2.5‑72B (≈0.743); Yi‑34B (≈0.943); Mistral‑7B (≈0.670); Meta‑Llama‑3‑8B (≈0.459). Normalization spikes noted in guidance for all of these.
- Low/medium: Meta‑Llama‑3‑70B (tier=low; score_v2≈0.344; js_p50≈0.0024; pct_kl≥1.0≈0.012); Mistral‑Small‑24B (tier=low; score_v2≈0.185).
- Norm‑only semantics: flagged near/at final layers for gemma‑2‑9b/27b and Qwen2.5‑72B (e.g., earliest_norm_only_semantic at the last layer for gemma‑2‑27b). When present, we restrict claims to rank/KL milestones near those depths.

Short quotes:
- Qwen3‑8B risk block (tier=high) in JSON; jaccard_p50≈0.282; pct_layers_kl_ge_1.0≈0.757.
- Meta‑Llama‑3‑70B low tier: js_p50≈0.0024; pct_layers_kl_ge_1.0≈0.012.

---

## 4. Confirmed Semantics

We prioritize `L_semantic_strong_run2` > `L_semantic_strong` > `L_semantic_confirmed` > `L_semantic_norm` and apply the uniform‑ and top‑2 margin gates when available.

- Confirmed layers (with source):
  - Qwen3‑8B: L=31 (confirmed; source=raw) — `output-Qwen3-8B-milestones.csv:4`; exemplar fact row (Germany→Berlin) `output-Qwen3-8B-pure-next-token.csv:33`.
  - Qwen3‑14B: L=36 (confirmed; source=raw) — `output-Qwen3-14B-milestones.csv:3`; exemplar fact row `output-Qwen3-14B-pure-next-token.csv:38`.
  - Meta‑Llama‑3‑70B: L=40 (confirmed; source=raw) — `output-Meta-Llama-3-70B-milestones.csv:3`; exemplar fact row `output-Meta-Llama-3-70B-pure-next-token.csv:42`. Uniform‑margin gate fails at `L_semantic_norm`; treat early onset as weak.
  - Meta‑Llama‑3‑8B: L=25 (confirmed; source=raw) — `output-Meta-Llama-3-8B-milestones.csv:3`; exemplar `output-Meta-Llama-3-8B-pure-next-token.csv:29`. Margin gate passes only later at `L_semantic_margin_ok_norm`.
  - Mistral‑7B: L=25 (confirmed) — `output-Mistral-7B-v0.1-milestones.csv:3`; exemplar `output-Mistral-7B-v0.1-pure-next-token.csv:30`.
  - Mistral‑Small‑24B: L=33 (confirmed) — `output-Mistral-Small-24B-Base-2501-milestones.csv:3`; exemplar `output-Mistral-Small-24B-Base-2501-pure-next-token.csv:35`.
  - Yi‑34B: L=44 (confirmed; source=tuned) — `output-Yi-34B-milestones.csv:3`; exemplar `output-Yi-34B-pure-next-token.csv:46`.
  - Gemma‑2‑9B: L=42 (confirmed; source from milestones) — `output-gemma-2-9b-milestones.csv:4`; exemplar `output-gemma-2-9b-pure-next-token.csv:49`.
  - Gemma‑2‑27B: L=46 (confirmed) — `output-gemma-2-27b-milestones.csv:4`; exemplar `output-gemma-2-27b-pure-next-token.csv:48`.
  - Qwen2.5‑72B: L=80 (confirmed window absent; `L_semantic_norm=80`) — `output-Qwen2.5-72B-milestones.csv:3`; exemplar `output-Qwen2.5-72B-pure-next-token.csv:138`.

Where `summary.semantic_margin` indicates `margin_ok_at_L_semantic_norm=false` (e.g., Llama‑3‑70B, Llama‑3‑8B, Mistral‑Small‑24B), we label the onset “weak rank‑1” and prefer later gated confirmations.

---

## 5. Entropy & Confidence

Entropy drift `(entropy_bits − teacher_entropy_bits)` tends to shrink as rank improves and KL falls:

- High median drift families: Llama‑3‑8B/70B, Mistral‑Small‑24B, Qwen2.5‑72B, Yi‑34B (p50 ≈ 12–14 bits). As semantics emerges, layerwise entropy drops (see exemplars at confirmed layers above) while final heads remain calibrated.
- Lower/irregular drift: gemma‑2‑9b shows negative median drift (p50≈−2.80 bits), consistent with a calibrated head and very late semantics; gemma‑2‑27b shows modest positive drift (p50≈4.68 bits). These patterns align with the known Gemma final‑head calibration (non‑zero last‑layer KL).

Short quotes: Qwen3‑8B entropy gap (p50) reported in JSON; Llama‑3‑70B p50≈14.34; gemma‑2‑9b p50≈−2.80.

---

## 6. Normalization & Numeric Health

- Strategy: All models `pre_norm` with `next_ln1` lens provenance. RMSNorm epsilon is inside sqrt and γ selection follows the architecture-aware rule implemented in SCRIPT.
- Early spikes: All models log normalization spikes (e.g., large `resid_norm_ratio` and high `delta_resid_cos`) prior to semantic onset; e.g., Qwen3‑8B L0 `delta_resid_cos≈0.99`; Mistral‑7B reports very high early `resid_norm_ratio`; Llama‑3‑8B shows large ratio at L0. None coincide with numeric health failures.
- Numeric health: `any_nan=false`, `any_inf=false`, `layers_flagged=[]` across models.

---

## 7. Repeatability

`diagnostics.repeatability` is `skipped` for all runs (deterministic environment). No `{max_rank_dev, p95_rank_dev, top1_flip_rate}` are reported; treat near‑threshold single‑layer differences cautiously, especially near lens transitions.

---

## 8. Family Patterns

- Qwen (3‑8B/14B vs 2.5‑72B): Qwen3 variants converge late (≈0.86–0.90 depth) with high raw‑vs‑norm artefact scores and strong control margins; Qwen2.5‑72B pushes semantics to the final layer (1.00 depth) with mixed raw‑vs‑norm divergence and no strong control position. All show near‑zero last‑layer KL with τ⋆≈1.0; tuned lens acts beyond calibration (rotation contributes) for Qwen3.
- Gemma (2‑9B/27B): Both exhibit copy‑reflex in L0–L3 and confirmed semantics only at the last layer. High lens‑artefact tier and non‑zero last‑layer KL indicate final‑head calibration (τ⋆≈2.85–3.01); treat final‑row probability differences as calibration artefacts and emphasize ranks/KL thresholds.
- Llama/Mistral/Yi: Llama‑3‑70B is a within‑family outlier with earlier semantic onset (~0.50 depth) and low artefact tier; Llama‑3‑8B is later (~0.78) with higher lens risk. Mistral family converges late (≈0.78–0.83) with high (7B) vs low (24B) artefact tiers. Yi‑34B converges mid‑late (~0.73) with high artefact risk and a tuned‑confirmed onset.

---

## 10. Prism Summary Across Models

Prism acts as a shared‑decoder diagnostic; we compare prism vs norm lens `kl_to_final_bits` at sampled depths (≈25/50/75% of layers; pos/orig; fact_index=0). Negative deltas indicate Prism lowers KL relative to norm.

- Helpful: gemma‑2‑27b (large KL reductions at all samples; e.g., ≈−24 bits at mid‑depth). Yi‑34B and Qwen2.5‑72B show moderate early/mid KL reductions but degrade at the last layer (norm‑only semantics and calibrated heads make prism diverge at the end).
- Neutral: Mistral‑Small‑24B (small early reduction, mid increase; ranks/milestones unchanged). 
- Regressive: Llama‑3‑8B, Mistral‑7B, Qwen3‑8B/14B (KL increases across sampled depths; no rank‑milestone advances). Llama‑3‑70B shows mild increases.

Concrete citations (layer, prism − norm KL):
- gemma‑2‑27b: L≈22 (≈50%): prism KL well below norm; see `output-gemma-2-27b-pure-next-token-prism.csv` vs `...-pure-next-token.csv` near the same layer; semantic citation row `output-gemma-2-27b-pure-next-token.csv:48`.
- Qwen2.5‑72B: at final layer, prism fails to recover answer; KL large and rank ≫1 (`output-Qwen2.5-72B-pure-next-token-prism.csv:120`).

---

## 11. Within‑Family Similarities and Differences

- Qwen family: Qwen3‑8B (31/36) and Qwen3‑14B (36/40) confirm semantics late (0.86–0.90), both high artefact tier with strong control margins and tuned lens preferred for reporting. Qwen2.5‑72B reaches semantics only at the final layer (1.00) with no strong control position and mixed raw‑vs‑norm overlap; all show near‑zero last‑layer KL and τ⋆≈1.0, indicating little head mismatch.
- Gemma family: 2‑9B and 2‑27B display copy‑reflex in L0–L3 and semantics only at the last layer with high artefact scores and norm‑only semantics near the end. Both have non‑zero last‑layer KL and τ⋆≫1, consistent with calibrated final heads; tuned lens is calibration‑only in 27B and mostly calibration in 9B.

---

## 13. Misinterpretations in Existing EVALS

- Qwen2.5‑72B Prism conclusion needs nuance: the single‑model EVAL states Prism is regressive at the final layer (`output-Qwen2.5-72B-pure-next-token-prism.csv:120`), which is correct, but it overlooks early/mid reductions in KL relative to norm (negative prism−norm deltas around mid‑depth). Cross‑model context suggests “mixed”: early helpful, late regressive.
- Several EVALs quote detailed entropies and logit gaps near semantics (e.g., `output-Qwen3-8B-pure-next-token.csv:33`), which is acceptable; however, per `measurement_guidance.suppress_abs_probs=true`, any absolute probability claims (if added in future edits) should be avoided in favor of ranks/KL. No hard violations found in the current files.
- Where `summary.semantic_margin.margin_ok_at_L_semantic_norm=false` (e.g., Llama‑3‑70B, Llama‑3‑8B, Mistral‑Small‑24B), EVALs should consistently label these onsets as weak; most do, but ensure all instances reflect the gate outcome.

---

Short exemplar quotes per model (semantic layer rows; positive/original prompts):
- Qwen3‑8B — `output-Qwen3-8B-milestones.csv:4`; `output-Qwen3-8B-pure-next-token.csv:33`.
- Qwen3‑14B — `output-Qwen3-14B-milestones.csv:3`; `output-Qwen3-14B-pure-next-token.csv:38`.
- Meta‑Llama‑3‑70B — `output-Meta-Llama-3-70B-milestones.csv:3`; `output-Meta-Llama-3-70B-pure-next-token.csv:42`.
- Meta‑Llama‑3‑8B — `output-Meta-Llama-3-8B-milestones.csv:3`; `output-Meta-Llama-3-8B-pure-next-token.csv:29`.
- Mistral‑7B — `output-Mistral-7B-v0.1-milestones.csv:3`; `output-Mistral-7B-v0.1-pure-next-token.csv:30`.
- Mistral‑Small‑24B — `output-Mistral-Small-24B-Base-2501-milestones.csv:3`; `output-Mistral-Small-24B-Base-2501-pure-next-token.csv:35`.
- Yi‑34B — `output-Yi-34B-milestones.csv:3`; `output-Yi-34B-pure-next-token.csv:46`.
- gemma‑2‑9b — `output-gemma-2-9b-milestones.csv:4`; `output-gemma-2-9b-pure-next-token.csv:49`.
- gemma‑2‑27b — `output-gemma-2-27b-milestones.csv:4`; `output-gemma-2-27b-pure-next-token.csv:48`.
- Qwen2.5‑72B — `output-Qwen2.5-72B-milestones.csv:3`; `output-Qwen2.5-72B-pure-next-token.csv:138`.

---

**Produced by OpenAI GPT-5**

