**Cross-Model Evaluation (Logit Lens, Capitals Prompt)**

This cross-eval synthesizes layer-by-layer probing across models using the latest run outputs. All models set `prefer_ranks=true` and `suppress_abs_probs=true` in measurement guidance; analysis emphasizes rank/KL milestones and avoids absolute probabilities. Depth comparisons are normalized by each model’s `num_layers`. Quotes cite CSV line numbers via each model’s `*-milestones.csv` and the `pure-next-token.csv` row indices provided in `evaluation_pack.citations.layers`.

References to prior work for context: logit-lens and lens calibration (arXiv:2304.14997), tuned lens (arXiv:2309.08600), and intermediate representation decoding stability (arXiv:2402.11147). These inform our caution around view-dependent “early semantics” and head calibration.

**Models Covered**
- google/gemma-2-27b (46 layers)
- google/gemma-2-9b (42)
- meta-llama/Meta-Llama-3-70B (80)
- meta-llama/Meta-Llama-3-8B (32)
- mistralai/Mistral-7B-v0.1 (32)
- mistralai/Mistral-Small-24B-Base-2501 (40)
- Qwen/Qwen2.5-72B (80)
- Qwen/Qwen3-14B (40)
- Qwen/Qwen3-8B (36)
- 01-ai/Yi-34B (60)

---
**1. Result Synthesis**

- Normalized semantic onset (prefer L_semantic_strong_run2 → L_semantic_strong → L_semantic_confirmed → L_semantic_norm):
  - Meta-Llama-3-70B: L≈40/80 (0.50). Confirmed under raw lens at L40; citation `output-Meta-Llama-3-70B-milestones.csv:3`, pure row `output-Meta-Llama-3-70B-pure-next-token.csv:40`. Lens-consistency p50 at targets is strong (jaccard@10≈0.74; JSON diagnostics.lens_consistency).
  - Meta-Llama-3-8B: L≈25/32 (0.78). Confirmed at L25; `output-Meta-Llama-3-8B-milestones.csv:3`, pure row `...-pure-next-token.csv:25`. Lens-consistency p50 modest (jaccard@10≈0.25), calibration-sensitive gates (both_gates_pass_frac=0.0 at L25).
  - Mistral-7B-v0.1: L≈25/32 (0.78). Confirmed at L25; `output-Mistral-7B-v0.1-milestones.csv:3`, pure row `...-pure-next-token.csv:25`. Gates calibration-sensitive (both=0.0 at L25); lens-consistency p50 around jaccard@10≈0.43.
  - Mistral-Small-24B: L≈33/40 (0.83). Confirmed at L33; `output-Mistral-Small-24B-Base-2501-milestones.csv:3`, pure row `...-pure-next-token.csv:33`. Low artefact tier; gates still calibration-sensitive (both=0.0 at L33).
  - Qwen3-14B: L≈36/40 (0.90). Confirmed at L36; `output-Qwen3-14B-milestones.csv:3`, pure row `...-pure-next-token.csv:36`. Gates robust (both=1.0 at L36), but artefact tier high; position window fragile (rank1_frac=0.0).
  - Qwen3-8B: L≈31/36 (0.86). Confirmed at L31; `output-Qwen3-8B-milestones.csv:3`, pure row `...-pure-next-token.csv:31`. Gates robust (both=1.0 at L31), but lens-consistency p50 low and position fragile.
  - Qwen2.5-72B: L≈80/80 (1.00). Confirmed at L80; `output-Qwen2.5-72B-milestones.csv:3`, pure row `...-pure-next-token.csv:80`. Gates calibration-sensitive (both=0.0 at L80), position-window not measured.
  - Yi-34B: L≈44/60 (0.73). Confirmed at L44; `output-Yi-34B-milestones.csv:3`, pure row `...-pure-next-token.csv:44`. Gates robust (both=1.0 at L44) but artefact tier high; lens-consistency p50 at targets is low (jaccard@10≈0.05).
  - Gemma-2-27B: L≈46/46 (1.00). Confirmed at L46; `output-gemma-2-27b-milestones.csv:3`, pure row `...-pure-next-token.csv:46`. Warn-high-last-layer-KL=true, head mismatch (τ⋆≈3.01). High artefact tier and norm-only semantics at the final layer.
  - Gemma-2-9B: L≈42/42 (1.00). Confirmed at L42; `output-gemma-2-9b-milestones.csv:3`, pure row `...-pure-next-token.csv:42`. Warn-high-last-layer-KL=true (τ⋆≈2.85), high artefact tier.

- Δ̂ medians across facts (when available): Gemma‑2‑27B≈1.00 and Gemma‑2‑9B≈1.00 (evaluation_pack.micro_suite.aggregates), Qwen3‑8B≈0.0556; others null. These confirm “late semantics” in Gemma and later‑than‑middepth on Qwen3‑8B.

- Tuned audit (rotation vs temperature; pos_ood_gap): strong rotation benefits for Meta‑Llama‑3‑8B (ΔKL_rot_p50≈2.56; pos_ood_gap≈−0.77), Mistral‑7B (≈2.21; ≈−0.77), Mistral‑24B (≈1.76; ≈+0.66), Qwen3‑14B (≈1.97; ≈−0.49), Qwen3‑8B (≈0.92; ≈−1.23), Yi‑34B (≈3.50; ≈+0.85). Gemma‑2‑27B’s tuned lens is calibration‑only; treat tuned as a calibration aid rather than a semantics lens.

- Head mismatch (τ⋆, final‑KL before/after): Gemma‑2‑27B (τ⋆≈3.01; tuned final‑KL drops from ≈1.13 to ≈0.55), Gemma‑2‑9B (τ⋆≈2.85; ≈1.08→≈0.41). Others show τ⋆≈1.0 with near‑zero final KL; treat inter‑model final‑row probability differences as head‑calibration artefacts when `warn_high_last_layer_kl=true` (notably Gemma family).

---
**2. Copy Reflex (Layers 0–3)**

- Definition: mark as “copy‑reflex” if `copy_collapse` OR `copy_soft_k1@0.5=True` in layers 0–3 of `*-pure-next-token.csv`.
- Outliers: Copy‑reflex appears only in Gemma‑2 models.
  - Gemma‑2‑27B: row 2, L0 shows `copy_collapse=True`, `copy_soft_k1@0.5=True` (output-gemma-2-27b-pure-next-token.csv:2). L3 has soft‑copy as well (e.g., row 5). Others do not trigger.
  - Gemma‑2‑9B: rows 2–4, L0–L2 show `copy_collapse=True` and `copy_soft_k1@0.5=True` (output-gemma-2-9b-pure-next-token.csv:2–4).
- All other models: no copy‑reflex under the default soft‑window test in layers 0–3 (programmatic scan over `*-pure-next-token.csv`).
- Control strength: lexical leakage is strong when `first_control_strong_pos` is present — seen in all but Qwen2.5‑72B. Examples: Meta‑Llama‑3‑8B L25, Mistral‑7B L24, Qwen3‑14B L36, Yi‑34B L42 (`output-<MODEL>.json` control_summary).

---
**3. Lens Artefact Risk**

- Summary metrics (evaluation_pack.artifact; diagnostics.raw_lens_full):
  - High tier: Qwen3‑14B (lens_artifact_score_v2≈0.704; js_p50≈0.513; l1_p50≈1.432; pct_layers_kl_ge_1.0≈0.756), Qwen3‑8B (≈0.704; ≈0.358; ≈1.134; ≈0.757), Yi‑34B (≈0.943; ≈0.369; ≈1.089; ≈0.656), Qwen2.5‑72B (≈0.743; ≈0.105; ≈0.615; ≈0.321), Mistral‑7B (≈0.670; ≈0.074; ≈0.505; ≈0.242), Gemma‑2‑27B (≈1.000; ≈0.865; ≈1.893; ≈0.979), Gemma‑2‑9B (≈0.591; ≈0.006; ≈0.029; ≈0.302).
  - Medium tier: Meta‑Llama‑3‑70B (≈0.344; ≈0.0024; ≈0.092; ≈0.012), Meta‑Llama‑3‑8B (≈0.459; ≈0.0168; ≈0.240; ≈0.030).
  - Low tier: Mistral‑Small‑24B (≈0.185; ≈0.035; ≈0.347; ≈0.024).
- Norm‑only semantics: present near or at candidate depths for several models (e.g., Meta‑Llama‑3‑8B window includes L25; earliest=25, n=5). Treat early semantics as view‑dependent when `n_norm_only_semantics_layers>0` near the target.
- Lens‑consistency at candidate layers: low p50 values (norm vs raw) indicate view‑dependent “early semantics”. Examples: Meta‑Llama‑3‑8B p50 jaccard@10≈0.25; Qwen3‑8B≈0.176; Yi‑34B≈0.053. Meta‑Llama‑3‑70B is stronger (≈0.742).

---
**4. Confirmed Semantics**

- Preferred milestones (per RULES): L_semantic_strong_run2 → L_semantic_strong → L_semantic_confirmed → L_semantic_norm. When `L_semantic_confirmed` exists, report its `source`.
- Confirmed layers and citations:
  - Meta‑Llama‑3‑70B: L40 confirmed (source=raw); `output-Meta-Llama-3-70B-milestones.csv:3`, pure row 40. Margin gate at L_semantic_norm fails; treat as rank‑confirmed, not margin‑confirmed.
  - Meta‑Llama‑3‑8B: L25 confirmed; `output-Meta-Llama-3-8B-milestones.csv:3`, pure row 25. Gates calibration‑sensitive at L25 (both=0.0) and position‑fragile (rank1_frac=0.0).
  - Mistral‑7B‑v0.1: L25 confirmed; `output-Mistral-7B-v0.1-milestones.csv:3`, pure row 25. Gates calibration‑sensitive (both=0.0) and position‑fragile (rank1_frac=0.0).
  - Mistral‑Small‑24B: L33 confirmed; `output-Mistral-Small-24B-Base-2501-milestones.csv:3`, pure row 33. Low artefact tier; position‑fragile.
  - Qwen3‑14B: L36 confirmed; `output-Qwen3-14B-milestones.csv:3`, pure row 36. Gates robust (both=1.0) but artefact tier high; position‑fragile.
  - Qwen3‑8B: L31 confirmed; `output-Qwen3-8B-milestones.csv:3`, pure row 31. Gates robust (both=1.0) but lens‑consistency low and position‑fragile.
  - Qwen2.5‑72B: L80 confirmed; `output-Qwen2.5-72B-milestones.csv:3`, pure row 80. Gates calibration‑sensitive; position‑window not measured.
  - Yi‑34B: L44 confirmed; `output-Yi-34B-milestones.csv:3`, pure row 44. Gates robust but artefact tier high; lens‑consistency low.
  - Gemma‑2‑27B: L46 confirmed; `output-gemma-2-27b-milestones.csv:3`, pure row 46. Warn‑high‑last‑layer‑KL=true; margin passes at final.
  - Gemma‑2‑9B: L42 confirmed; `output-gemma-2-9b-milestones.csv:3`, pure row 42. Warn‑high‑last‑layer‑KL=true; margin passes at final.
- Stability gates (small rescalings): treat any target layer with `both_gates_pass_frac<0.75` as calibration‑sensitive. This applies to Meta‑Llama‑3‑70B/8B, Mistral‑7B/24B, Qwen2.5‑72B. Prefer run‑of‑two onsets (when present) for stability.
- Position‑window: rank‑1 fraction at the semantic layer is <0.5 for all models that ran the window; annotate as position‑fragile and avoid broad claims beyond the measured next‑token position.

---
**5. Entropy & Confidence**

- Within models, entropy gaps versus the teacher shrink through depth as rank improves and KL falls (evaluation_pack.entropy). Representative p50 gaps: Meta‑Llama‑3‑8B≈13.88 bits, Meta‑Llama‑3‑70B≈14.34, Mistral‑7B≈10.99, Mistral‑24B≈13.59, Qwen3‑8B≈13.79, Qwen3‑14B≈13.40, Qwen2.5‑72B≈12.50, Yi‑34B≈12.59. Gemma‑2‑9B shows negative gaps (p50≈−2.80), consistent with strong copy dominance early and only final‑layer semantics.

---
**6. Normalization & Numeric Health**

- All models are pre‑norm with RMSNorm; norm lens uses `next_ln1` (post‑block uses the next block’s ln1; last layer uses ln_final). Epsilon placement is correct (inside sqrt). Early layers show pronounced normalization spikes (high `resid_norm_ratio` and large `delta_resid_cos`), flagged across all models. No numeric pathologies observed (`any_nan=false`, `any_inf=false`, no `layers_flagged`).

---
**7. Repeatability**

- Repeatability self‑tests were skipped (`deterministic_env` across models). With no `{max_rank_dev,p95_rank_dev,top1_flip_rate}`, treat near‑threshold rank differences (e.g., first_rank_le_5 vs le_1 within ±1 layer) cautiously.

---
**8. Family Patterns**

- Qwen family (Qwen3‑8B/14B; Qwen2.5‑72B): later semantics by depth (≈0.86–1.00), high artefact tier and low lens‑consistency at targets (p50 jaccard@10 often ≤0.33), robust gates for Qwen3 (both=1.0) but calibration‑sensitive at 72B final (both=0.0). Control leakage is strong for Qwen3 (first_control_strong_pos present) and weak/absent for Qwen2.5‑72B.
- Gemma‑2 family (9B/27B): copy‑reflex in L0–L3, semantics only at the final layer (Δ̂≈1.00). Warn‑high‑last‑layer‑KL=true with τ⋆≈3; treat cross‑model probability differences as head‑calibration artefacts. Tuned lens is calibration‑only at 27B; Prism is helpful at 27B (KL p50 ↓) but regressive at 9B.
- Mistral family: 7B and 24B both show semantics in the top ~20–30% of depth; 24B has lower artefact tier but both are calibration‑sensitive at the semantic layer (both_gates_pass_frac=0.0). Control margins become strong just before semantics.
- Llama‑3 family: 70B exhibits the earliest semantics by normalized depth (~0.5) with strong lens‑consistency and medium artefact tier; 8B reaches semantics later (~0.78) with lower lens‑consistency and calibration‑sensitive gates.

---
**10. Prism Summary Across Models**

- Classify by ΔKL_p50 (baseline minus prism; positive=Helpful, negative=Regressive):
  - Helpful: Gemma‑2‑27B (Δ≈+23.73), Qwen2.5‑72B (≈+2.83), Yi‑34B (≈+1.36).
  - Neutral: Qwen3‑14B (≈−0.25; near‑zero), Meta‑Llama‑3‑70B (≈−1.00; small), Mistral‑24B (≈−5.98; moderate).
  - Regressive: Meta‑Llama‑3‑8B (≈−8.29), Mistral‑7B (≈−17.54), Qwen3‑8B (≈−0.59), Gemma‑2‑9B (≈−10.33).
- Rank‑milestone shifts are rarely available for Prism (nulls across models); use KL deltas as diagnostic only. Treat Prism as shared‑decoder analysis rather than a semantics lens.

---
**11. Within‑Family Similarities and Differences (Qwen, Gemma)**

- Qwen: Qwen3‑8B (L≈31/36) and Qwen3‑14B (≈36/40) both achieve confirmed semantics late by depth with strong small‑scale gate stability (both=1.0) but high artefact tier and low lens‑consistency at targets. Qwen2.5‑72B hits semantics only at the final layer (L80/80), lacks gate stability at target (both=0.0), and has no position‑window audit; Prism is helpful only at 72B. Control strongness appears in Qwen3 but is absent for 72B.
- Gemma: Both 9B and 27B show copy‑reflex in L0–L3 and semantics only at the final layer (Δ̂≈1.00). 27B’s tuned lens is calibration‑only with strong head mismatch (τ⋆≈3), and Prism reduces KL (helpful). 9B’s head mismatch is also large (τ⋆≈2.85), but Prism increases KL (regressive). Artefact tier is high for both, and position‑window rank‑1 fractions are low (≈0.167).

---
**13. Misinterpretations in Existing EVALS**

- Mistral‑24B: The EVAL implies “positional generalization” without clarifying that `rank1_frac=0.0` at the semantic layer (position‑fragile per RULES). Prefer stating that the position audit indicates fragility at L33 (`output-Mistral-Small-24B-Base-2501.json` summary.position_window).
- Llama‑3‑8B: The EVAL downplays normalization spikes (“trend down without a pre‑semantic explosion”) despite large early `delta_resid_cos` (e.g., ≈0.99 at L3; `output-Meta-Llama-3-8B.json` normalization_provenance). Clarify that spikes are flagged and contribute to medium artefact tier.
- Qwen2.5‑72B: The EVAL narrative discusses rank improvements near L74–L79; however, confirmed semantics only occurs at L80 and gates are calibration‑sensitive at that target (both_gates_pass_frac=0.0). Keep the focus on confirmed L80 and avoid implying stronger pre‑final certainty (use `output-Qwen2.5-72B-milestones.csv:3`, pure row 80).

---
Notes and caveats
- RMS‑lens can distort absolute probabilities; cross‑family comparisons use rank/KL milestones and normalized depths only.
- Single‑prompt probing can shift copy‑collapse; stylistic ablations should be used for robustness (see prompts and `position_window`).
- Final‑row caveat: when `warn_high_last_layer_kl=true` (Gemma family), treat final probability differences as head calibration, not regressions.
- Normalization gates differ across families only if a model were post‑norm (none here); timing claims stay normalized by depth.

---
**Produced by OpenAI GPT-5**

