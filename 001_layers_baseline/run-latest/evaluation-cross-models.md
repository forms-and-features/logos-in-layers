# Cross-Model Evaluation (Logit-Lens Baseline)

This report synthesizes layer-by-layer probe results across 10 open‑weight models. All claims follow each model’s `measurement_guidance` and gate rules. Where `suppress_abs_probs=true`, we avoid absolute probabilities and lead with rank and KL thresholds. File citations include concrete JSON/CSV line numbers per model.

**Models covered**
- Gemma 2: `gemma-2-27b`, `gemma-2-9b`
- Llama 3: `Meta-Llama-3-70B`, `Meta-Llama-3-8B`
- Mistral: `Mistral-Small-24B-Base-2501`, `Mistral-7B-v0.1`
- Qwen: `Qwen2.5-72B`, `Qwen3-14B`, `Qwen3-8B`
- Yi: `Yi-34B`


## 1. Result Synthesis

Within-model timing (normalized by depth) shows consistent late consolidation for pre‑norm families, with family‑specific variation:
- Gemma 2 consolidates at the end of the stack. Confirmed semantics at L=46/46 (1.00) for 27B and L=42/42 (1.00) for 9B (milestones: 001_layers_baseline/run-latest/output-gemma-2-27b-milestones.csv:3; 001_layers_baseline/run-latest/output-gemma-2-9b-milestones.csv:3). Median Δ̂ across facts = 1.00 for both (001_layers_baseline/run-latest/output-gemma-2-27b.json:11136; 001_layers_baseline/run-latest/output-gemma-2-9b.json:10806).
- Llama 3 exhibits earlier consolidation in 8B (25/32=0.78) but mid‑stack in 70B (40/80=0.50). Confirmed layers: 8B at 25 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-milestones.csv:3), 70B at 40 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-milestones.csv:3).
- Mistral tightens late‑mid stack: 24B at 33/40 (0.83) and 7B at 25/32 (0.78) (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-milestones.csv:3; 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-milestones.csv:3).
- Qwen trends later with size: 3‑8B at 31/36 (0.86), 3‑14B at 36/40 (0.90), and 2.5‑72B at the final 80/80 (1.00) (001_layers_baseline/run-latest/output-Qwen3-8B-milestones.csv:3; 001_layers_baseline/run-latest/output-Qwen3-14B-milestones.csv:3; 001_layers_baseline/run-latest/output-Qwen2.5-72B-milestones.csv:3).
- Yi‑34B consolidates at 44/60 (0.73) (001_layers_baseline/run-latest/output-Yi-34B-milestones.csv:3).

Δ̂ across facts (median when available): Gemma‑2‑27B/9B = 1.00 (fully late) (001_layers_baseline/run-latest/output-gemma-2-27b.json:11136; 001_layers_baseline/run-latest/output-gemma-2-9b.json:10806); Qwen3‑8B shows a small positive gap, Δ̂≈0.0556 (001_layers_baseline/run-latest/output-Qwen3-8B.json:12314). Others do not report Δ̂ medians in this run.

Tuned‑lens audit highlights differences by family/scale:
- Gemma‑2‑27B: `tuned_is_calibration_only=true`, rotation contributes ≤0 while temperature accounts for most ΔKL at p50 (δKL_rot_p50≈−0.03, δKL_temp_p50≈0.53); τ⋆≈3.01; final‑KL drops from 1.133→0.548 bits after τ⋆ (001_layers_baseline/run-latest/output-gemma-2-27b.json:10736,10760–10768).
- Gemma‑2‑9B: calibration‑not‑only; δKL_rot_p50≈0.00, δKL_temp_p50≈−0.01; τ⋆≈2.85; final‑KL 1.082→0.406 bits (001_layers_baseline/run-latest/output-gemma-2-9b.json:10890–10906).
- Llama‑3‑8B: rotation dominates (δKL_rot_p50≈2.56; interaction≈2.34), tuned preferred; τ⋆=1.0 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11896–11920, 12098–12110).
- Mistral‑7B/24B and Qwen‑3/ Yi‑34B: rotation dominates (p50 ≈2–5 bits), τ⋆=1.0; tuned preferred for semantics (see each model’s audit summary and tuned_audit blocks: Mistral‑7B 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:7009–7033; Mistral‑24B 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:9160–9184; Qwen3‑8B 001_layers_baseline/run-latest/output-Qwen3-8B.json:12186–12224; Qwen3‑14B 001_layers_baseline/run-latest/output-Qwen3-14B.json:12024–12092; Yi‑34B 001_layers_baseline/run-latest/output-Yi-34B.json:7912–7931).
- Qwen‑2.5‑72B: tuned lens missing; use norm (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9768–9776, 9937–9960).

Head mismatch (final‑head calibration): Gemma shows non‑trivial last‑layer KL reduced by model‑calibration (e.g., 27B τ⋆≈3.01 with KL drop; 001_layers_baseline/run-latest/output-gemma-2-27b.json:10736–10768). Families with τ⋆=1.0 report near‑zero last‑layer KL (e.g., Llama‑3‑8B: 0.0; 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7974–7998).


## 2. Copy Reflex (Layers 0–3)

We mark “copy‑reflex” when `copy_collapse=True` OR `copy_soft_k1@τ_soft=True` in layers 0–3 of the positive/original prompt (pure CSV):
- Present: Gemma‑2‑9B (L0–L4, strict/soft copy true; 001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2–6) and Gemma‑2‑27B (L0/L3/L4 soft‑copy; 001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2,5,6).
- Absent at L≤3: Llama‑3‑8B (flags false at 0–3; 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:2–5), Llama‑3‑70B (0–3 false; 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:2–5), Mistral‑7B (0–3 false; 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:2–5), Mistral‑24B (0–3 false; 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:2–5), Qwen‑3 (8B/14B) (0–3 false; 001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:2–5), Yi‑34B (0–3 false; 001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:2–5).

Control strength: lexical leakage is strong where `first_control_strong_pos` is set — e.g., Gemma‑2‑27B (46), Gemma‑2‑9B (42), Llama‑3‑8B (25), Llama‑3‑70B (80), Mistral‑7B (24), Mistral‑24B (31), Qwen‑3‑8B (30), Qwen‑3‑14B (36), Yi‑34B (42); Qwen‑2.5‑72B has no strong‑control position (001_layers_baseline/run-latest/output-*.json:9117–9124; 9044–9051; 10229–10236; 9590–9597; 5384–5389; 7276–7283; 10382–10387; 10452–10457; 6227–6232; 9792–9797).


## 3. Lens Artefact Risk

Raw‑vs‑Norm metrics and audit CSVs indicate varying risk tiers; we report score_v2, JS/L1, Jaccard, KL prevalence, and norm‑only semantics:
- Gemma‑2‑27B (high): score_v2=1.00; JS p50≈0.865; L1 p50≈1.893; Jaccard p50≈0.563; pct_kl≥1.0≈0.979; 1 norm‑only semantics layer at L=46 (001_layers_baseline/run-latest/output-gemma-2-27b.json:11106–11120).
- Gemma‑2‑9B (high): score_v2≈0.591; JS p50≈0.0063; L1 p50≈0.029; Jaccard p50≈0.639; pct_kl≥1.0≈0.302; 1 norm‑only semantics layer at L=42 (001_layers_baseline/run-latest/output-gemma-2-9b.json:10956–10986).
- Llama‑3‑8B (medium): score_v2≈0.459; JS p50≈0.0168; L1 p50≈0.240; Jaccard p50≈0.408; pct_kl≥1.0≈0.030; 5 norm‑only semantics layers earliest at L=25 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:12136–12178).
- Llama‑3‑70B (medium): score_v2≈0.344; JS p50≈0.0024; L1 p50≈0.092; Jaccard p50≈0.515; pct_kl≥1.0≈0.012; 2 norm‑only semantics layers (earliest L=79) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9734–9768).
- Mistral‑24B (low): score_v2≈0.185; JS p50≈0.035; L1 p50≈0.347; Jaccard p50≈0.538; pct_kl≥1.0≈0.024; no norm‑only semantics (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:9101–9159).
- Mistral‑7B (high): score_v2≈0.670; JS p50≈0.074; L1 p50≈0.505; Jaccard p50≈0.408; pct_kl≥1.0≈0.242; 1 norm‑only semantics layer at L=32 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:7201–7260).
- Qwen‑3‑8B (high): score_v2≈0.704; JS p50≈0.358; L1 p50≈1.134; Jaccard p50≈0.282; pct_kl≥1.0≈0.757 (001_layers_baseline/run-latest/output-Qwen3-8B.json:12203–12235).
- Qwen‑3‑14B (high): score_v2≈0.704; JS p50≈0.513; L1 p50≈1.432; Jaccard p50≈0.250; pct_kl≥1.0≈0.756 (001_layers_baseline/run-latest/output-Qwen3-14B.json:12276–12312).
- Qwen‑2.5‑72B (high): score_v2≈0.743; JS p50≈0.105; L1 p50≈0.615; Jaccard p50≈0.316; pct_kl≥1.0≈0.321; norm‑only semantics at L=80 (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9960–9971).
- Yi‑34B (high): score_v2≈0.943; JS p50≈0.369; L1 p50≈1.089; Jaccard p50≈0.111; pct_kl≥1.0≈0.656; many norm‑only semantics layers incl. L=44 (001_layers_baseline/run-latest/output-Yi-34B.json:8072–8096).

Lens‑consistency at candidate layers: very low for Gemma‑2‑27B (J@10/J@50≈0.0 at L=46; 001_layers_baseline/run-latest/output-gemma-2-27b.json:6846–6880), moderate for Qwen‑3‑14B (p50 J@10≈0.33; J@50≈0.39; 001_layers_baseline/run-latest/output-Qwen3-14B.json:8138–8160), and low‑to‑moderate for Yi‑34B (p50 J@10≈0.176; J@50≈0.205; 001_layers_baseline/run-latest/output-Yi-34B.json:3924–3970). Treat early semantics in high‑risk tiers as view‑dependent.


## 4. Confirmed Semantics

We prioritize `L_semantic_strong_run2`, else `L_semantic_strong`, else `L_semantic_confirmed`, else `L_semantic_norm`, following gates:
- Confirmed sources and layers: Gemma‑2‑27B (L=46, source=tuned; 001_layers_baseline/run-latest/output-gemma-2-27b-milestones.csv:4), Gemma‑2‑9B (L=42, tuned; 001_layers_baseline/run-latest/output-gemma-2-9b-milestones.csv:4), Llama‑3‑8B (L=25, raw; 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-milestones.csv:4), Llama‑3‑70B (L=40, raw; 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-milestones.csv:3), Mistral‑24B (L=33, raw; 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-milestones.csv:4), Mistral‑7B (L=25, raw; 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-milestones.csv:3), Qwen‑3‑8B (L=31, raw; 001_layers_baseline/run-latest/output-Qwen3-8B-milestones.csv:4), Qwen‑3‑14B (L=36, raw; 001_layers_baseline/run-latest/output-Qwen3-14B-milestones.csv:4), Yi‑34B (L=44, tuned; 001_layers_baseline/run-latest/output-Yi-34B-milestones.csv:4). Qwen‑2.5‑72B has no confirmed source (001_layers_baseline/run-latest/output-Qwen2.5-72B-milestones.csv:4).
- Uniform‑margin and Top‑2 gates: many models pass uniform margin at the reported layer (e.g., Gemma‑2‑27B/9B `margin_ok_at_L_semantic_norm=true`; 001_layers_baseline/run-latest/output-gemma-2-27b.json:10808–10816; 001_layers_baseline/run-latest/output-gemma-2-9b.json:10730–10738). Where `margin_ok_at_L_semantic_norm=false` (e.g., Llama‑3‑8B L=32; Mistral‑24B L=33), we annotate weak rank‑1 per guidance (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11896–11920; 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:8960–8966).
- Gate stability under small rescalings: stable (both_gates_pass_frac=1.0) for Gemma‑2‑27B and Qwen‑3 (8B/14B) (001_layers_baseline/run-latest/output-gemma-2-27b.json:5792–5807; 001_layers_baseline/run-latest/output-Qwen3-8B.json:7226–7241; 001_layers_baseline/run-latest/output-Qwen3-14B.json:7229–7244); fragile for Llama‑3‑8B, Mistral‑24B, and Qwen‑2.5‑72B (both_gates_pass_frac=0.0; 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7132–7147; 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4046–4065; 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7400–7415).
- Position‑window stability: rank‑1 fraction <0.50 in most reported cases (e.g., Gemma‑2‑27B 0.167; Llama‑3‑8B 0.0; Mistral‑24B 0.0) indicating position‑fragile onsets (001_layers_baseline/run-latest/output-gemma-2-27b.json:8648–8684; 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11996–12018; 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:6818–6840).

Example fact citations (positive/original):
- Gemma‑2‑27B: Germany→Berlin at L=46 (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48); Qwen‑3‑8B at L=31 (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:33).


## 5. Entropy & Confidence

Within‑model trends match expectations: as rank improves and KL to final falls, the entropy gap relative to the teacher narrows or is concentrated late in the stack.
- Llama‑3‑70B: entropy_gap_bits p50≈14.34 (tight IQR), collapsing by the end (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9786–9789). Llama‑3‑8B shows a similar high p50≈13.88 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:12099–12102).
- Mistral‑7B/24B: p50≈10.99 and ≈13.59, respectively (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:7253–7256; 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:9153–9156).
- Qwen‑3‑8B/14B: p50≈13.79/13.40 (001_layers_baseline/run-latest/output-Qwen3-8B.json:12255–12258; 001_layers_baseline/run-latest/output-Qwen3-14B.json:12328–12331). Qwen‑2.5‑72B concentrates entropy reduction at L≈80 (p50≈12.50; 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9989–9992).
- Gemma‑2‑27B: smaller p50≈4.68 with very late consolidation (001_layers_baseline/run-latest/output-gemma-2-27b.json:11003–11006).


## 6. Normalization & Numeric Health

All models are detected as pre‑norm RMS stacks; we use next‑block `ln1` for decoding and verify the RMS epsilon fix is on as per SCRIPT. Numeric health is clean across runs (`any_nan=false`, `any_inf=false`, `layers_flagged=[]`; see each `diagnostics.numeric_health`). Early normalization spikes are flagged broadly (e.g., Gemma‑2‑27B `flags.normalization_spike=true`; 001_layers_baseline/run-latest/output-gemma-2-27b.json:842), and large early `resid_norm_ratio`/`delta_resid_cos` values are common before semantics (e.g., Mistral‑7B L2 `resid_norm_ratio≈259`, `delta_resid_cos≈0.854`; 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2948–2968). Where decoding‑point gates fail (e.g., Gemma‑2‑27B, Qwen‑2.5‑72B), we treat early onsets as normalization‑choice sensitive (001_layers_baseline/run-latest/output-gemma-2-27b.json:6702–6768; 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9920–9937).


## 7. Repeatability

Deterministic replays are numerically stable for all models (`max_rank_dev=0.0`, `p95_rank_dev=0.0`, `top1_flip_rate=0.0`; see each model’s `evaluation_pack.repeatability`). However, forward‑of‑two repeatability is unassessed (mode `skipped_deterministic`) and thus we downgrade near‑threshold claims per guidance (e.g., Gemma‑2‑27B: 001_layers_baseline/run-latest/output-gemma-2-27b.json:6858–6880; Llama‑3‑70B: 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9776–9791).


## 8. Family Patterns

- Gemma: strong copy‑reflex in early layers (0–3) and all‑the‑way‑to‑final consolidation (Δ̂≈1.0). High artefact tier, decoding‑point sensitive, but gates (uniform/top‑2) are robust at the final layer. 27B’s tuned lens is calibration‑only (τ⋆≈3), unlike 9B.
- Llama: earlier/mid consolidation relative to depth (8B≈0.78 vs 70B≈0.50), medium artefact tier, and mixed gate stability (8B calibration‑sensitive under small rescalings; 70B also fragile). Decoding‑point consistent for 8B, sensitive for 70B.
- Mistral: late‑mid consolidation (24B≈0.83; 7B≈0.78). Artefact tier splits: 24B low vs 7B high. Both show calibration sensitivity under small rescalings.
- Qwen: later consolidation with scale (8B≈0.86; 14B≈0.90; 72B=1.00). High artefact tier and decoding‑point sensitivity common; 72B shows semantics only at the final layer and is calibration‑sensitive under small rescalings.
- Yi: mid‑late consolidation (≈0.73), high artefact tier with many norm‑only semantics layers and low lens‑consistency at targets; tuned preferred.


## 10. Prism Summary Across Models

Prism is a shared‑decoder diagnostic; we classify by KL deltas and rank shifts at sampled depths:
- Helpful: Gemma‑2‑27B (ΔKL_p50≈+23.7 bits reduction; 001_layers_baseline/run-latest/output-gemma-2-27b.json:838–876), Qwen‑2.5‑72B (ΔKL_p50≈+2.83; 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:852–890). Rank milestones under Prism were not established (null) in this run.
- Neutral to Slightly Regressive: Qwen‑3‑14B (ΔKL_p50≈−0.25; 001_layers_baseline/run-latest/output-Qwen3-14B.json:845–918).
- Regressive: Llama‑3‑8B (ΔKL_p50≈−8.29; 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:800–872), Mistral‑24B (ΔKL_p50≈−5.98; 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:847–918). In these, Prism increased KL without improving early rank.


## 11. Intra‑Family Similarities and Differences

- Gemma 2 (27B vs 9B): Both are pre‑norm RMS with strong early copy reflex (pure CSV shows copy flags true at L≤3; 001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2–5; 001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2–5) and consolidate at the final layer (milestones L=46/42). 27B’s tuned lens is calibration‑only with τ⋆≈3 and substantial final‑head mismatch reduction, while 9B’s tuned lens behaves as a mild calibrator with small component attributions. Both show decoding‑point sensitivity at the semantic layer.
- Llama 3 (8B vs 70B): Both medium artefact tiers; 8B is decoding‑point consistent at L=25 but calibration‑sensitive (both_gates_pass_frac=0.0), while 70B is decoding‑point sensitive at L=40 and also calibration‑sensitive. Normalized timing differs: ~0.78 vs 0.50; both exhibit position fragility or lack of position audit (rank1_frac 0.0 for 8B; null for 70B).
- Mistral (7B vs 24B): Similar normalized timing (0.78 vs 0.83) but different artefact tiers (high vs low). Both show decoding‑point sensitivity at non‑primary targets and calibration sensitivity under small rescalings. 24B’s raw‑vs‑norm overlap (J@50≈0.538) is stronger than 7B’s (≈0.408).
- Qwen (3‑8B vs 3‑14B vs 2.5‑72B): Gradually later consolidation with scale (0.86 → 0.90 → 1.00). All show high artefact tier and decoding‑point sensitivity at least at some targets; 72B has no confirmed semantics and only norm‑lens semantics at the final layer.
- Yi‑34B: High artefact tier with many norm‑only semantics layers; tuned preferred. Consolidation is mid‑late (≈0.73) with position fragility at the semantic layer.


## 13. Misinterpretations in Existing EVALS

- evaluation-Mistral-Small-24B-Base-2501.md: “Semantic margin … margin_ok_at_L_semantic_norm=false; no confirmed‑margin override.” This is correct at L=33, but the text later treats L=33 as “bold” without explicitly marking decoding‑point sensitivity at the alternate target. Prefer: annotate `decoding_point_consistent=false` at first_rank target (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4855–4925).
- evaluation-Qwen2.5-72B.md: The narrative emphasizes final‑row semantics (L=80) without reiterating that forward‑of‑two was unassessed (mode=`skipped_deterministic`). Prefer: add an explicit “repeatability_forward_disagreement: unassessed” note (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8985–9002).
- evaluation-Meta-Llama-3-8B.md: The report states tuned as preferred lens and highlights rotation‑driven ΔKL, but it could better foreground the calibration‑sensitivity gate failure at L=25 (both_gates_pass_frac=0.0). Prefer: flag “calibration‑sensitive” prominently when summarizing the semantic layer (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7132–7147).


---
**Produced by OpenAI GPT-5**
