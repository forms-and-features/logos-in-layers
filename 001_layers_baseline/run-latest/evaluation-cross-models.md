# CROSS‑EVAL: Layer‑by‑layer probes across models

This cross‑model review synthesizes the latest run under `001_layers_baseline/run-latest` using the project’s rank/KL-first guidance and lens artefact controls. All layer indices refer to the model’s own depth. Unless noted otherwise, semantics use the lens indicated by `measurement_guidance.preferred_lens_for_reporting` and confirmed milestones when present. Quotes include source file and line numbers for verification.

## 1. Result Synthesis

- Timing (confirmed semantics). Within‑family patterns are consistent and roughly proportional to depth:
  - Meta‑Llama‑3‑8B: semantics at L25/32 (confirmed_source=raw). Quote: `L_semantic_confirmed=25` (output-Meta-Llama-3-8B-milestones.csv:4); JSON confirms (`"L_semantic_confirmed": 25`) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:8209).
  - Meta‑Llama‑3‑70B: semantics at L40/80 (confirmed_source=raw). Quote: (output-Meta-Llama-3-70B-milestones.csv:4); JSON (`"L_semantic_confirmed": 40`) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8717).
  - Mistral‑7B‑v0.1: L25/32 (raw). Quote: (output-Mistral-7B-v0.1-milestones.csv:4); JSON (`25`) (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3362).
  - Mistral‑Small‑24B‑Base‑2501: L33/40 (raw). Quote: (output-Mistral-Small-24B-Base-2501-milestones.csv:4); JSON (`33`) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:5259).
  - Qwen3‑8B: L31/36 (raw). Quote: (output-Qwen3-8B-milestones.csv:4); JSON (`31`) (001_layers_baseline/run-latest/output-Qwen3-8B.json:8366).
  - Qwen3‑14B: L36/40 (raw). Quote: (output-Qwen3-14B-milestones.csv:4); JSON (`36`) (001_layers_baseline/run-latest/output-Qwen3-14B.json:8437).
  - Qwen2.5‑72B: L80/80 (no tuned confirmation). Quote: (output-Qwen2.5-72B-milestones.csv:4–5 show norm at 80; none tuned) (output-Qwen2.5-72B-milestones.csv:4) and JSON (`"source": "none"`) (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9214–9222).
  - Yi‑34B: L44/60 (confirmed_source=tuned). Quote: (output-Yi-34B-milestones.csv:5) and JSON (`44`) (001_layers_baseline/run-latest/output-Yi-34B.json:4208).
  - Gemma‑2‑9B: L42/42 (tuned). Quote: (output-gemma-2-9b-milestones.csv:5) and JSON (`42`) (001_layers_baseline/run-latest/output-gemma-2-9b.json:7880).
  - Gemma‑2‑27B: L46/46 (tuned). Quote: (output-gemma-2-27b-milestones.csv:5) and JSON (`46`) (001_layers_baseline/run-latest/output-gemma-2-27b.json:7101).
- Rank milestones (norm lens unless tuned is preferred). Examples:
  - Llama‑3‑8B first_rank_le: {10:24, 5:25, 1:25} (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:6988–6990).
  - Qwen3‑8B: {10:29, 5:29, 1:31} (001_layers_baseline/run-latest/output-Qwen3-8B.json:7082–7084).
  - Yi‑34B (tuned preferred): baseline {10:41, 5:43, 1:44} and tuned shifts later to le_1=46 (001_layers_baseline/run-latest/output-Yi-34B.json:4077–4079, 4466–4468).
- Δ̂ surface‑to‑meaning gap, when defined: Gemma family exhibits maximal Δ̂≈1.0 of depth due to copy at L0 and semantics at final layer. Quote: `depth_fractions.delta_hat=1.0` (001_layers_baseline/run-latest/output-gemma-2-27b.json:7971–7977; 001_layers_baseline/run-latest/output-gemma-2-9b.json:7904–7912).
- Tuned‑lens audit (rotation vs temperature) shows rotation contributes the majority of KL reductions in most non‑Gemma cases, while Gemma’s tuned lens is calibration‑only at 27B:
  - Llama‑3‑8B: ΔKL_rot_p50≈+2.34, ΔKL_temp_p50≈−0.05 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:8968–8980).
  - Qwen3‑14B: ΔKL_rot_p50≈+1.69, ΔKL_temp_p50≈−0.03 (001_layers_baseline/run-latest/output-Qwen3-14B.json:9203–9220).
  - Yi‑34B: ΔKL_rot_p50≈+3.28, ΔKL_temp_p50≈−0.66 (001_layers_baseline/run-latest/output-Yi-34B.json:4994–5006).
  - Gemma‑2‑27B: tuned_is_calibration_only=true, ΔKL_rot_p50≈−0.03, ΔKL_temp_p50≈+0.48 (001_layers_baseline/run-latest/output-gemma-2-27b.json:7873–7890, 7902).
- Head mismatch (final‑head calibration). Reported τ⋆ for model‑calibration and tuned final‑KL before/after:
  - Gemma‑2‑27B: τ⋆≈2.85; KL_final 1.1352 → 0.5131 bits (001_layers_baseline/run-latest/output-gemma-2-27b.json:7898–7902).
  - Gemma‑2‑9B: τ⋆≈2.70; KL_final 1.0129 → 0.3391 bits (001_layers_baseline/run-latest/output-gemma-2-9b.json:7822–7824).
  - Llama‑3‑8B, Qwen3‑8B/14B, Mistral‑7B/24B, Yi‑34B: τ⋆=1.0; tuned final‑KL≈0 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:8991–8994; 001_layers_baseline/run-latest/output-Qwen3-8B.json:9152–9155; 001_layers_baseline/run-latest/output-Qwen3-14B.json:9227–9230; 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4144–4147; 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:6049–6052; 001_layers_baseline/run-latest/output-Yi-34B.json:5018–5021).

## 2. Copy Reflex (Layers 0–3)

- Gemma family shows a clear copy‑reflex at the input token: strict and soft copy hit at L0 in both sizes. Quotes:
  - 27B: `copy_strict@0.95=True` at L0 (output-gemma-2-27b-milestones.csv:2) and `copy_soft_k1@0.5=True` (output-gemma-2-27b-pure-next-token.csv:2).
  - 9B: same pattern at L0 (output-gemma-2-9b-milestones.csv:2–3; output-gemma-2-9b-pure-next-token.csv:2).
- No early copy flags in layers 0–3 for Llama‑3‑8B, Mistral‑7B, Mistral‑24B, Qwen3‑8B/14B, Yi‑34B. Example quotes: Llama‑3‑8B (rows 2–5 all `False`) (output-Meta-Llama-3-8B-pure-next-token.csv:2–5); Qwen3‑8B (rows 2–5 all `False`) (output-Qwen3-8B-pure-next-token.csv:2–5).

## 3. Lens Artefact Risk

- We report artifact v2 score, symmetric raw‑vs‑norm stats, and prevalence. Summary rows from `artifact-audit.csv`:
  - Llama‑3‑8B: v2≈0.459 (tier=medium), JS_p50≈0.0168, L1_p50≈0.2403 (output-Meta-Llama-3-8B-artifact-audit.csv:2). JSON agrees (`risk_tier=medium`) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:9068–9070).
  - Llama‑3‑70B: v2≈0.344 (tier=medium), JS_p50≈0.00245, L1_p50≈0.0918 (output-Meta-Llama-3-70B-artifact-audit.csv:2); JSON `medium` (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9012).
  - Mistral‑7B: v2≈0.624 (tier=high), JS_p50≈0.159, L1_p50≈0.989 (output-Mistral-7B-v0.1-artifact-audit.csv:2); JSON `high` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4222).
  - Mistral‑24B: v2≈0.185 (tier=low), JS_p50≈0.0353, L1_p50≈0.3473 (output-Mistral-Small-24B-Base-2501-artifact-audit.csv:2); JSON `low` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:6124).
  - Qwen3‑8B: v2≈0.704 (tier=high), JS_p50≈0.358, L1_p50≈1.134 (output-Qwen3-8B-artifact-audit.csv:2); JSON `high` (001_layers_baseline/run-latest/output-Qwen3-8B.json:9229).
  - Qwen3‑14B: v2≈0.704 (tier=high), JS_p50≈0.513, L1_p50≈1.432 (output-Qwen3-14B-artifact-audit.csv:2); JSON `high` (001_layers_baseline/run-latest/output-Qwen3-14B.json:9304).
  - Qwen2.5‑72B: v2≈0.743 (tier=high), JS_p50≈0.105, L1_p50≈0.615 (output-Qwen2.5-72B-artifact-audit.csv:2); JSON `high` (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9224).
  - Yi‑34B: v2≈0.944 (tier=high), JS_p50≈0.369, L1_p50≈1.089 (output-Yi-34B-artifact-audit.csv:2); JSON `high` (001_layers_baseline/run-latest/output-Yi-34B.json:5096).
  - Gemma‑2‑27B: v2≈1.000 (tier=high), JS_p50≈0.865, L1_p50≈1.893 (output-gemma-2-27b-artifact-audit.csv:2); JSON `high` (001_layers_baseline/run-latest/output-gemma-2-27b.json:7967–7971).
  - Gemma‑2‑9B: v2≈0.591 (tier=high), JS_p50≈0.0063, L1_p50≈0.0292, but `pct_layers_kl_ge_1.0≈0.302`; early norm‑only semantics present (001_layers_baseline/run-latest/output-gemma-2-9b.json:7900–7911).
- Risk tier takeaways: treat Gemma and Qwen families as high artefact risk; avoid absolute probabilities and prefer rank/KL thresholds. Llama‑3 remains medium; Mistral‑24B is low.

## 4. Confirmed Semantics

- Confirmed semantic layers and sources:
  - Llama‑3‑8B: L25 (source=raw). Quote: (output-Meta-Llama-3-8B-milestones.csv:5 shows confirmed row for raw) and JSON (`25`) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:9046–9050).
  - Llama‑3‑70B: L40 (raw) (output-Meta-Llama-3-70B-milestones.csv:5; 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9010–9016).
  - Mistral‑7B: L25 (raw) (output-Mistral-7B-v0.1-milestones.csv:5; 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4200–4203).
  - Mistral‑24B: L33 (raw) (output-Mistral-Small-24B-Base-2501-milestones.csv:5; 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:6102–6106).
  - Qwen3‑8B: L31 (raw) (output-Qwen3-8B-milestones.csv:5; 001_layers_baseline/run-latest/output-Qwen3-8B.json:9207–9211).
  - Qwen3‑14B: L36 (raw) (output-Qwen3-14B-milestones.csv:5; 001_layers_baseline/run-latest/output-Qwen3-14B.json:9282–9286).
  - Qwen2.5‑72B: none confirmed (output-Qwen2.5-72B-milestones.csv:5; 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9202–9220).
  - Yi‑34B: L44 (tuned) (output-Yi-34B-milestones.csv:5; 001_layers_baseline/run-latest/output-Yi-34B.json:5074–5079).
  - Gemma‑2‑9B: L42 (tuned) (output-gemma-2-9b-milestones.csv:5; 001_layers_baseline/run-latest/output-gemma-2-9b.json:7866–7880).
  - Gemma‑2‑27B: L46 (tuned; calibration‑only lens) (output-gemma-2-27b-milestones.csv:5; 001_layers_baseline/run-latest/output-gemma-2-27b.json:7955–7962, 7902).

## 5. Entropy & Confidence

- Entropy drift (gap = entropy_bits − teacher_entropy_bits) shows large positive gaps in Llama/Mistral/Qwen2.5/Yi and small or negative gaps in Gemma‑2‑9B:
  - Llama‑3‑8B: p25≈13.87, p50≈13.88, p75≈13.91 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:9086–9089).
  - Mistral‑24B: p50≈13.59 (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:6142–6145).
  - Qwen2.5‑72B: p50≈12.50 (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9242–9245).
  - Yi‑34B: p50≈12.59 (001_layers_baseline/run-latest/output-Yi-34B.json:5114–5117).
  - Gemma‑2‑9B: p50≈−2.80 (001_layers_baseline/run-latest/output-gemma-2-9b.json:7919–7921).
- Within models, drift shrinks as rank improves and KL to final drops near L_semantic_confirmed (e.g., Llama‑3‑8B has `first_kl_below_1.0=32` with very low final KL, alongside the L25 rank‑1 milestone) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7852–7854, 8467–8469).

## 6. Normalization & Numeric Health

- Norm strategy is uniform across these runs: `strategy = next_ln1` (pre‑norm) for all inspected models (e.g., Llama‑3‑8B at 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7233; Qwen3‑8B at 001_layers_baseline/run-latest/output-Qwen3-8B.json:7330; Yi‑34B at 001_layers_baseline/run-latest/output-Yi-34B.json:2812; Mistral‑24B at 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4164; Gemma‑2‑9B at 001_layers_baseline/run-latest/output-gemma-2-9b.json:5903).
- Early normalization spikes are flagged in all models (`flags.normalization_spike=true`), often with large `resid_norm_ratio` at L0 (e.g., Mistral‑7B `≈115.17` and `delta_resid_cos≈0.308`) (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2393–2399). Numeric health is clean in all cases (no NaN/Inf, no flagged layers; e.g., Llama‑3‑8B at 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7734–7739).

## 7. Repeatability

- Repeatability was skipped under deterministic settings for all models; no variance metrics are available. Example: `flag: "skipped"` with null `{max_rank_dev,p95_rank_dev,top1_flip_rate}` (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:9070–9074; also present for Qwen, Mistral, Gemma, Yi).

## 8. Family Patterns

- Qwen (3‑8B, 3‑14B, 2.5‑72B):
  - Semantics emerge late (≈0.86–1.0 of depth). Quotes: L36/40 (output-Qwen3-14B-milestones.csv:4), L31/36 (output-Qwen3-8B-milestones.csv:4), L80/80 (output-Qwen2.5-72B-milestones.csv:4).
  - Artefact tier is high with large symmetric divergences (v2≈0.70–0.74; JS/L1 elevated) (output-Qwen3-14B-artifact-audit.csv:2; output-Qwen3-8B-artifact-audit.csv:2; output-Qwen2.5-72B-artifact-audit.csv:2).
  - Tuned audit shows rotation dominates KL reduction (ΔKL_rot_p50≈0.88–1.69) with mild or negative temperature effects (001_layers_baseline/run-latest/output-Qwen3-8B.json:9128–9140; output-Qwen3-14B.json:9203–9215). Control margins vary but gold alignment is perfect across prompts (e.g., output-Qwen3-8B.json:9198–9211).
- Gemma (2‑9B, 2‑27B):
  - Strong copy‑reflex at L0 and semantics at the terminal layer (Δ̂≈1.0). Quotes: copy at L0 (output-gemma-2-27b-milestones.csv:2; output-gemma-2-9b-milestones.csv:2–3); semantics at last (output-gemma-2-27b-milestones.csv:4; output-gemma-2-9b-milestones.csv:4).
  - High artefact tier with family‑level final‑head calibration issues (`warn_high_last_layer_kl=true`), so cross‑model final‑row probability comparisons are avoided (001_layers_baseline/run-latest/output-gemma-2-27b.json:6667; output-gemma-2-9b.json:6595).
  - Tuned lens: 27B is calibration‑only (`tuned_is_calibration_only=true`), 9B is not; head τ⋆≈2.7–2.85 and tuned final‑KL remains >0 (001_layers_baseline/run-latest/output-gemma-2-27b.json:7898–7902; output-gemma-2-9b.json:7822–7824).

## 10. Prism Summary Across Models

- We treat Prism as a shared‑decoder diagnostic and compare KL percentiles vs baseline norm lens:
  - Llama‑3‑8B: ΔKL_p50≈−8.29 bits (regressive); no rank milestone improvement (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:852–880).
  - Llama‑3‑70B: ΔKL_p50≈−1.00 (regressive); no milestone change (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:864–900).
  - Mistral‑7B: ΔKL_p50≈−17.54 (regressive) (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:852–890).
  - Mistral‑24B: ΔKL_p50≈−5.98 (regressive) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:852–890).
  - Qwen3‑8B: ΔKL_p50≈−0.59 with much worse tail (regressive) (001_layers_baseline/run-latest/output-Qwen3-8B.json:852–890).
  - Qwen3‑14B: ΔKL_p50≈−0.25 (slightly regressive) (001_layers_baseline/run-latest/output-Qwen3-14B.json:852–890).
  - Yi‑34B: ΔKL_p50≈+1.36 (helpful, small), no rank shift (001_layers_baseline/run-latest/output-Yi-34B.json:852–890).
  - Gemma‑2‑9B: ΔKL_p50≈−10.33 (regressive) (001_layers_baseline/run-latest/output-gemma-2-9b.json:870–906).
  - Gemma‑2‑27B: ΔKL_p50≈+23.73 (helpful, large), though overall family remains high artefact (001_layers_baseline/run-latest/output-gemma-2-27b.json:870–906).

## 11. Within‑Family Notes (Qwen, Gemma)

- Qwen:
  - 8B vs 14B show similar late semantics (31/36 vs 36/40) under raw; 72B (2.5) hits only at the terminal layer without tuned confirmation. Quotes: (output-Qwen3-8B-milestones.csv:4), (output-Qwen3-14B-milestones.csv:4), (output-Qwen2.5-72B-milestones.csv:4–5).
  - Artefact risk is consistently high across sizes, with elevated JS/L1 and large `pct_layers_kl_ge_1.0` (e.g., 0.7568 at 8B and 0.7568 at 14B) (output-Qwen3-8B-artifact-audit.csv:2; output-Qwen3-14B-artifact-audit.csv:2).
  - Tuned lenses reduce KL mainly via rotation with minimal temperature effect; tuned tends to push rank‑1 slightly later (e.g., 14B le_1: 36→39) (001_layers_baseline/run-latest/output-Qwen3-14B.json:8081–8083, 9198–9206).
- Gemma:
  - Both sizes copy at L0 and confirm semantics only at the final layer; Δ̂≈1.0. Quotes: (output-gemma-2-27b-milestones.csv:2,4–5); (output-gemma-2-9b-milestones.csv:2–5).
  - Head calibration differs: 27B needs τ⋆≈2.85 and is calibration‑only; 9B τ⋆≈2.70 and tuned improves alignment but leaves residual final KL>0 (001_layers_baseline/run-latest/output-gemma-2-27b.json:7898–7902; output-gemma-2-9b.json:7822–7826).
  - Prism diverges by size: regressive at 9B (ΔKL_p50≈−10.33), strongly helpful at 27B (ΔKL_p50≈+23.73) (001_layers_baseline/run-latest/output-gemma-2-9b.json:870–906; output-gemma-2-27b.json:870–906).

## 13. Misinterpretations in Existing EVALS

- Qwen3‑14B Prism labeled “Neutral” despite summarized negatives. The JSON provides explicit deltas (ΔKL_p50≈−0.25), which is mildly regressive; recommending to mark as “Slightly Regressive” rather than neutral. Quote: “Verdict: Neutral (no qualitative milestone shift reported).” (001_layers_baseline/run-latest/evaluation-Qwen3-14B.md:108). Supporting JSON: (001_layers_baseline/run-latest/output-Qwen3-14B.json:852–890).
- Gemma‑2‑27B EVAL prints absolute `p_answer` at the final layer in a high‑risk setting (family‑level head calibration; `warn_high_last_layer_kl=true`). Guidance suggests avoiding absolute probabilities. Quote: milestones include `p_answer=0.984…` (output-gemma-2-27b-milestones.csv:4) while `measurement_guidance.suppress_abs_probs=true` (001_layers_baseline/run-latest/output-gemma-2-27b.json:7926–7940).
- Minor omission: Several single‑model EVALs describe Prism as “diagnostic‑only” without noting explicit KL percentile deltas that are present in JSON (e.g., Llama‑3‑70B shows ΔKL_p50≈−1.00). Including these small but consistent signs of regression would improve completeness. Quote: “No explicit KL delta … Verdict: Neutral” (001_layers_baseline/run-latest/evaluation-Qwen3-14B.md:106–110); supporting JSON (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:864–900).

---
**Produced by OpenAI GPT-5**

