# Evaluation Report: google/gemma-2-27b

*Run executed on: 2025-10-13 22:23:35*

## EVAL

**1. Overview**
google/gemma-2-27b was probed with a logit‑lens pipeline to measure copy vs. semantic onset, KL‑to‑final calibration, ranks, cosine geometry, and entropy trajectories, alongside raw‑vs‑norm lens diagnostics and tuned‑lens audits. The run targets the fact “Germany→Berlin” with a constrained prompt ending in “called simply” (001_layers_baseline/run-latest/output-gemma-2-27b.json:4).

**2. Method sanity‑check**
- Prompt & indexing: context ends with “called simply” ("context_prompt": "Give the city name only, plain text. The capital of Germany is called simply"  001_layers_baseline/run-latest/output-gemma-2-27b.json:4). Positive rows exist for `prompt_id=pos`, `prompt_variant=orig` (e.g., L0 row)  001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2.
- Normalizer provenance: "arch": "pre_norm", "strategy": "next_ln1"; L0 uses "blocks[0].ln1" and final uses "ln_final"  001_layers_baseline/run-latest/output-gemma-2-27b.json:5964, 5965, 5969, 6383.
- Per‑layer normalizer effect: normalization trajectory flagged as "spike" with n_spikes=16 (001_layers_baseline/run-latest/output-gemma-2-27b.json:10908–10911). Early large `resid_norm_ratio`/`delta_resid_cos` values occur well before semantics (e.g., L1–L6 in normalization_provenance), and this is explicitly flagged in guidance as "normalization_spike" (001_layers_baseline/run-latest/output-gemma-2-27b.json:10850).
- Unembed bias: "present": false, "l2_norm": 0.0 (bias‑free cosines)  001_layers_baseline/run-latest/output-gemma-2-27b.json:834–837.
- Environment & determinism: device "cpu", torch "2.8.0+cu128", `deterministic_algorithms`: true, seed 316  001_layers_baseline/run-latest/output-gemma-2-27b.json:8976–8982.
- Numeric health: any_nan=false, any_inf=false, layers_flagged=[]  001_layers_baseline/run-latest/output-gemma-2-27b.json:6676–6682.
- Copy mask: sample shows whitespace runs (e.g., "\n\n\n\n\n\n\n\n") and size=4668  001_layers_baseline/run-latest/output-gemma-2-27b.json:5637, 5647.
- Gold alignment: `ok=true` with pieces ["▁Berlin"] and `gold_alignment_rate=1.0`  001_layers_baseline/run-latest/output-gemma-2-27b.json:6769–6772, 6810.
- Repeatability (1.39): skipped due to `deterministic_env`; no {max_rank_dev, p95_rank_dev, top1_flip_rate} (001_layers_baseline/run-latest/output-gemma-2-27b.json:6683–6686, 10892–10898).
- Norm trajectory: shape="spike", slope≈0.088, r2≈0.846, n_spikes=16  001_layers_baseline/run-latest/output-gemma-2-27b.json:10908–10911.
- Measurement guidance: {"prefer_ranks": true, "suppress_abs_probs": true, "preferred_lens_for_reporting": "norm", "use_confirmed_semantics": true} with reasons including "warn_high_last_layer_kl", "norm_only_semantics_window", "low_lens_consistency_at_semantic", and "pos_window_low_stability"  001_layers_baseline/run-latest/output-gemma-2-27b.json:10842–10859.
- Semantic margin: delta_abs=0.002 (logit units), p_uniform=3.90625e‑06, margin_ok_at_L_semantic_norm=true  001_layers_baseline/run-latest/output-gemma-2-27b.json:5772–5777.
- Gate‑stability (small rescalings): at L_semantic_norm, uniform/top‑2/both pass‑fractions are 1.0; min_both_gates_pass_frac=1.0 (stable)  001_layers_baseline/run-latest/output-gemma-2-27b.json:5799–5806.
- Position‑window: grid=[0.2,0.4,0.6,0.8,0.92,0.98], rank1_frac=0.1667 at L_semantic_norm (position‑fragile)  001_layers_baseline/run-latest/output-gemma-2-27b.json:8575–8587.
- Micro‑suite: aggregates present; n=5, n_missing=0; medians L_semantic_confirmed=46, Δ̂=1.0  001_layers_baseline/run-latest/output-gemma-2-27b.json:11013–11018.

**3. Quantitative findings (layer‑by‑layer)**
- L 0 — entropy 0.0005 bits, top‑1 "simply"  001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2.
- L 3 — entropy 0.8857 bits, top‑1 "simply"  001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:5.
- L 4 — entropy 0.6183 bits, top‑1 "simply"  001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:6.
- L 46 — entropy 0.1180 bits, top‑1 "Berlin"; answer_rank=1  001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48.

Control: first_control_margin_pos=0; first_control_strong_pos=46; max control margin reported (magnitude omitted per guidance)  001_layers_baseline/run-latest/output-gemma-2-27b.json:9032–9038.

Micro‑suite: median L_semantic_confirmed=46 and median Δ̂=1.0 across 5 facts (n_missing=0); concrete citation for baseline fact at L46  001_layers_baseline/run-latest/output-gemma-2-27b.json:11013–11018 and 001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48.

Entropy drift: across the depth the median entropy gap vs. teacher is ≈4.68 bits (p50)  001_layers_baseline/run-latest/output-gemma-2-27b.json:10914–10916. Last‑layer calibration remains imperfect: final‑head KL=1.135 bits; after temperature, 0.567 bits  001_layers_baseline/run-latest/output-gemma-2-27b.json:6774–6781.

Confidence/margin snapshots: at L46, logit gap to next token is ≈4.12 (answer_logit_gap)  001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48. Normalizer snapshots: L0 resid_norm_ratio≈0.786, delta_resid_cos≈0.572; L46 resid_norm_ratio≈0.053, delta_resid_cos≈0.662  001_layers_baseline/run-latest/output-gemma-2-27b.json:5968–5974, 6383–6389.

Semantics layer (preferred): L_semantic_confirmed=46 (confirmed_source=tuned; window Δ=2)  001_layers_baseline/run-latest/output-gemma-2-27b.json:8468–8474. Bolded below; uniform‑margin gate passes, but no run‑of‑two is reported.

**4. Qualitative findings**

4.1. Copy vs semantics (Δ‑gap)
Copy‑reflex ✓. Layer 0 shows strict copy/collapse on the trailing word "simply" (copy flags assert at L0)  001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2 and 001_layers_baseline/run-latest/output-gemma-2-27b-milestones.csv:2–3. Using earliest strict copy at τ=0.70 and τ=0.95 yields L_copy_strict=0 for both; stability="mixed"; norm_only_flags all false  001_layers_baseline/run-latest/output-gemma-2-27b.json:5716–5731. Δ̂ across depth is 1.0 (semantic at last layer; copy at L0)  001_layers_baseline/run-latest/output-gemma-2-27b.json:10877–10881.

4.2. Lens sanity: Raw‑vs‑Norm
Artifact risk is high: lens_artifact_score=0.987 and lens_artifact_score_v2=1.0; tier=high  001_layers_baseline/run-latest/output-gemma-2-27b.json:10884–10896, 5958–5960. Symmetric metrics indicate substantial divergence between raw and norm: js_divergence_p50≈0.865; l1_prob_diff_p50≈1.893; first_js_le_0.1=0; first_l1_le_0.5=0  001_layers_baseline/run-latest/output-gemma-2-27b.json:10886–10891. Top‑K overlap: jaccard_raw_norm_p50=0.5625; first_jaccard_raw_norm_ge_0.5 at L1  001_layers_baseline/run-latest/output-gemma-2-27b.json:10890–10891. Prevalence: pct_layers_kl_ge_1.0≈0.979; 1 norm‑only semantics layer at L46  001_layers_baseline/run-latest/output-gemma-2-27b.json:10892–10895, 5826–5833. At the semantic target, norm vs raw lens consistency is very low (jaccard@10=0.0; @50=0.0; spearman_top50≈0.238)  001_layers_baseline/run-latest/output-gemma-2-27b.json:6696–6724. Caution: early “semantics” are likely view‑dependent; prefer rank milestones and confirmed semantics.

4.3. Tuned‑Lens analysis
Tuned lens is calibration‑only ("tuned_is_calibration_only": true); continue to use the norm lens for semantic onset while using tuned for calibration sanity  001_layers_baseline/run-latest/output-gemma-2-27b.json:10947–10948. Attribution: rotation reduces KL slightly (ΔKL_rot p50≈−0.029), while temperature increases it (ΔKL_temp p50≈+0.528); interaction p50≈+0.028; overall tuned ΔKL p50≈+0.519  001_layers_baseline/run-latest/output-gemma-2-27b.json:10919–10927. Positional generalization is modest: pos_in_dist_le_0.92≈0.435; pos_ood_ge_0.96≈0.519; gap≈0.084  001_layers_baseline/run-latest/output-gemma-2-27b.json:10930–10940. Head mismatch: final KL drops from ≈1.133 bits to ≈0.548 after τ⋆ calibration; τ⋆≈3.01  001_layers_baseline/run-latest/output-gemma-2-27b.json:10942–10945. Last‑layer agreement: top‑1 agrees; KL_to_final≈1.135; after temp≈0.567; `warn_high_last_layer_kl=true`  001_layers_baseline/run-latest/output-gemma-2-27b.json:6774–6796.

4.4. KL, ranks, cosine, entropy milestones
- KL: baseline `first_kl_below_{1.0,0.5}` are null (no early approach to the final head); `warn_high_last_layer_kl=true`  001_layers_baseline/run-latest/output-gemma-2-27b.json:5656–5657, 6791.
- Ranks: preferred lens (norm): first_rank_le_{10,5,1}=46 (baseline and tuned summaries concur)  001_layers_baseline/run-latest/output-gemma-2-27b.json:5658–5660, 7015–7017.
- Cosine: milestones (norm) — ge_0.2 at L1; ge_0.4 at L46; ge_0.6 at L46  001_layers_baseline/run-latest/output-gemma-2-27b.json:5753–5761.
- Entropy: entropy trajectory follows a spike‑like shape; median entropy gap vs teacher ≈4.68 bits (p50)  001_layers_baseline/run-latest/output-gemma-2-27b.json:10908–10916. This aligns with late calibration and the high KL vs final head.
Reminder: we bold the confirmed semantics layer below; uniform‑margin gate passes at L46, but there is no run‑of‑two and position‑window stability is low (rank1_frac≈0.17), so the onset is position‑fragile.

4.5. Prism (shared‑decoder diagnostic)
Present and compatible (k=512; sampled layers: embed,10,22,33)  001_layers_baseline/run-latest/output-gemma-2-27b.json:845–865. KL drops sharply vs baseline (ΔKL p50≈23.73 bits) with no earlier rank‑1 milestone reported (baseline le_1=46; prism le_1=null)  001_layers_baseline/run-latest/output-gemma-2-27b.json:872–906. Verdict: Helpful — clear KL reduction without evidence of regressive ranks; use diagnostically alongside norm lens.

4.6. Ablation & stress tests
No‑filler ablation: L_copy shifts from 0→3 (ΔL_copy=+3), L_sem unchanged at 46 (ΔL_sem=0)  001_layers_baseline/run-latest/output-gemma-2-27b.json:8996–9004. Negative/control prompts: control summary shows first_control_margin_pos=0 and first_control_strong_pos=46 (001_layers_baseline/run-latest/output-gemma-2-27b.json:9032–9038). Test prompt "Berlin is the capital of" predicts " Germany" at the next token (coherent reversal)  001_layers_baseline/run-latest/output-gemma-2-27b.json:9–19.
Important‑word trajectory: at L46 the positive prompt’s top‑1 is " Berlin" (answer)  001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48; early layers (e.g., L0–L4) anchor to the prompt suffix "simply" (rows 2,5,6).

4.7. Checklist (✓/✗/n.a.)
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
- deterministic_algorithms true (or caution noted) ✓
- numeric_health clean ✓
- copy_mask plausible ✓
- milestones.csv or evaluation_pack.citations used for quotes ✓
- gate_stability_small_scale reported ✓
- position_window stability reported ✓

---
Produced by OpenAI GPT-5 
