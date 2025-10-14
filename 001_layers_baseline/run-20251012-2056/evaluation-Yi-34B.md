# Evaluation Report: 01-ai/Yi-34B

# **Overview**
01-ai/Yi-34B was probed on 2025-10-12 (Experiment started: 2025-10-12 20:56:18). The script 001_layers_baseline/run.py measures copy vs. semantic emergence via rank/KL/cosine/entropy trajectories and runs lens diagnostics (norm vs. raw; tuned-lens; prism).

# **Method sanity-check**
- Prompt & indexing: context prompt ends with “called simply” and no trailing space: "Give the city name only, plain text. The capital of Germany is called simply" (001_layers_baseline/run-latest/output-Yi-34B.json:812). Positive/original rows exist (e.g., L0/L44/L60 in pure CSV at rows 2, 46, 63; 001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:2, 001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:46, 001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:63).
- Normalizer provenance: arch=pre_norm; strategy=next_ln1. L0 ln_source="blocks[0].ln1" and last ln_source="ln_final" (001_layers_baseline/run-latest/output-Yi-34B.json:2735, 001_layers_baseline/run-latest/output-Yi-34B.json:3401).
- Per-layer normalizer effect: normalization spike flagged; norm_trajectory shape="spike" (n_spikes=4; r2=0.926) (001_layers_baseline/run-latest/output-Yi-34B.json:5579). Early snapshots show resid_norm_ratio, delta_resid_cos well-behaved at candidate layer (e.g., L44 resid_norm_ratio=0.2027, delta_resid_cos=0.9265; 001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:46).
- Unembed bias: present=false; l2_norm=0.0; cosines are bias‑free (001_layers_baseline/run-latest/output-Yi-34B.json:835–844).
- Environment & determinism: device=cpu; dtype_compute=torch.bfloat16; deterministic_algorithms=true; seed=316 (001_layers_baseline/run-latest/output-Yi-34B.json:6017–6031). Reproducibility OK.
- Numeric health: any_nan=false; any_inf=false; layers_flagged=[] (001_layers_baseline/run-latest/output-Yi-34B.json:3759–3766).
- Copy mask: ignored_token_ids sample: [97, 98, 99, 100, 114, …] (001_layers_baseline/run-latest/output-Yi-34B.json:980–1008). Field ‘size’ not reported; sample looks plausible for tokenizer control/punct tokens.
- Gold alignment: ok=true; variant=with_space; pieces=["▁Berlin"]; gold_alignment_rate=1.0 (001_layers_baseline/run-latest/output-Yi-34B.json:3793–3821).
- Repeatability (1.39): skipped due to deterministic_env (001_layers_baseline/run-latest/output-Yi-34B.json:3769–3773, 001_layers_baseline/run-latest/output-Yi-34B.json:7890–7896).
- Norm trajectory: shape="spike"; slope=0.074; r2=0.926; n_spikes=4 (001_layers_baseline/run-latest/output-Yi-34B.json:5579–5584).
- Measurement guidance: prefer_ranks=true; suppress_abs_probs=true; preferred_lens_for_reporting=tuned; use_confirmed_semantics=true (001_layers_baseline/run-latest/output-Yi-34B.json:7837–7848).
- Semantic margin: { delta_abs=0.002, p_uniform=1.5625e-05, margin_ok_at_L_semantic_norm=true } (001_layers_baseline/run-latest/output-Yi-34B.json:2630–2660).
- Micro‑suite: evaluation_pack.micro_suite present; n=5; n_missing=0; median L_semantic_confirmed=44 (001_layers_baseline/run-latest/output-Yi-34B.json:7968–8010, 001_layers_baseline/run-latest/output-Yi-34B.json:7752–7757).

# **Quantitative findings (layer‑by‑layer)**

| Layer | Entropy (bits) | Top‑1 token | Notes |
|---|---:|---|---|
| 0 | 15.9623 | ‘Denote’ | pos/orig; “⟨NEXT⟩” position (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:2) |
| **44** | 15.3273 | ‘Berlin’ | first rank‑1 at L=44; confirmed semantics (tuned); answer_rank=1 (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:46; 001_layers_baseline/run-latest/output-Yi-34B.json:7870–7876) |
| 60 | 2.9812 | ‘Berlin’ | final layer; last‑head KL≈0; top‑1 agree (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:63; 001_layers_baseline/run-latest/output-Yi-34B.json:3787–3807) |

- Control margin: first_control_margin_pos=1; max_control_margin=0.5836; first_control_strong_pos=42 (001_layers_baseline/run-latest/output-Yi-34B.json:6065–6075).
- Micro‑suite: median L_semantic_confirmed=44; Δ̂ median not reported; example fact row “France→Paris” at row 163 (001_layers_baseline/run-latest/output-Yi-34B.json:7752–7757, 001_layers_baseline/run-latest/output-Yi-34B.json:8011–8034).
- Entropy drift: entropy_gap_bits p25=12.286, p50=12.586, p75=12.775 (001_layers_baseline/run-latest/output-Yi-34B.json:7905–7907).
- Confidence margins & normalizer snapshots: at L44, answer_logit_gap≈0.673; delta_resid_cos=0.926 (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:46). At L60, answer_logit_gap≈1.426; delta_resid_cos=0.894 (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:63).

# **Qualitative findings**

Yi‑34B shows a clean semantic transition around two‑thirds depth under the norm/tuned lens, with no early copy reflex and a strong final‑head agreement. However, raw‑vs‑norm divergence is large with norm‑only semantics near the candidate layer, so rank milestones and confirmed semantics are preferred for reporting.

## 4.1. Copy vs semantics (Δ‑gap)
Copy reflex not detected: strict L_copy_strict at τ∈{0.70,0.80,0.90,0.95} is null; soft windows k∈{1,2,3} also null; stability="none" (001_layers_baseline/run-latest/output-Yi-34B.json:4740–4776, 001_layers_baseline/run-latest/output-Yi-34B.json:2689–2707). First rank‑1 semantics occurs at L=44 (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:46). Δ̂ not reported (evaluation_pack.milestones.depth_fractions.delta_hat=null; 001_layers_baseline/run-latest/output-Yi-34B.json:7868–7876).

## 4.2. Lens sanity: Raw‑vs‑Norm
Lens‑artifact risk is high: lens_artifact_score=0.793; lens_artifact_score_v2=0.943; tier=high (001_layers_baseline/run-latest/output-Yi-34B.json:2848–2860, 001_layers_baseline/run-latest/output-Yi-34B.json:7878–7890). Symmetric metrics are poor: js_divergence_p50=0.369; l1_prob_diff_p50=1.089; first_js_le_0.1=0; first_l1_le_0.5=0 (001_layers_baseline/run-latest/output-Yi-34B.json:7878–7890). Top‑K overlap is low: jaccard_raw_norm_p50=0.111; first_jaccard_raw_norm_ge_0.5=1 (001_layers_baseline/run-latest/output-Yi-34B.json:2689–2720, 001_layers_baseline/run-latest/output-Yi-34B.json:7878–7890). Prevalence: pct_layers_kl_ge_1.0=0.656; n_norm_only_semantics_layers=14; earliest_norm_only_semantic=44 (001_layers_baseline/run-latest/output-Yi-34B.json:2682–2707, 001_layers_baseline/run-latest/output-Yi-34B.json:7878–7890). Caution: early semantics near L=44 may be lens‑induced; prefer ranks and confirmed semantics.

## 4.3. Tuned‑Lens analysis
Preference: tuned_is_calibration_only=false; preferred_semantics_lens_hint=tuned (001_layers_baseline/run-latest/output-Yi-34B.json:7920–7931). Attribution: ΔKL (tuned vs components) shows rotation dominating temperature — delta_kl_rot_p25=3.156, p50=3.502, p75=3.687; delta_kl_temp_p25=−1.026, p50=−0.664, p75=0.605; interaction p50=3.330 (001_layers_baseline/run-latest/output-Yi-34B.json:7900–7920). Positional generalization: pos_in_dist_le_0.92=5.594; pos_ood_ge_0.96=6.447; gap=0.853 (001_layers_baseline/run-latest/output-Yi-34B.json:7920–7931). Head mismatch is minimal: tau_star_modelcal=1.0; KL_tuned_final=0.00130→0.00130 after τ⋆ (001_layers_baseline/run-latest/output-Yi-34B.json:7924–7931). Last‑layer agreement holds: top1_agree=true; KL_to_final≈0.00028 bits (001_layers_baseline/run-latest/output-Yi-34B.json:3787–3807). Reporting uses tuned lens for semantics, with norm baseline cited.

## 4.4. KL, ranks, cosine, entropy milestones
- KL: multiple tuned summaries show first_kl_below_1.0 at L≈44 or later and first_kl_below_0.5 at L=60 (e.g., 001_layers_baseline/run-latest/output-Yi-34B.json:5328–5394, 001_layers_baseline/run-latest/output-Yi-34B.json:4960–5032). Final KL is ≈0 (001_layers_baseline/run-latest/output-Yi-34B.json:3787–3807).
- Ranks: preferred (tuned) and baseline (norm) both reach rank‑1 by L=44; confirmed semantics layer=44 (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:46; 001_layers_baseline/run-latest/output-Yi-34B.json:7868–7876). Margin gate holds at L=44 (001_layers_baseline/run-latest/output-Yi-34B.json:2630–2660).
- Cosine: milestones under norm — ge_0.2 at L=1; ge_0.4 at L=44; ge_0.6 at L=51 (001_layers_baseline/run-latest/output-Yi-34B.json:2608–2616).
- Entropy: teacher_entropy_bits≈2.941; large positive entropy gaps p50≈12.586 bits suggest early uncertainty with late calibration (001_layers_baseline/run-latest/output-Yi-34B.json:2701, 001_layers_baseline/run-latest/output-Yi-34B.json:7905–7907).

## 4.5. Prism (shared‑decoder diagnostic)
Present/compatible (k=512; layers sampled: embed,14,29,44). KL deltas vs baseline norm: p50 drops by ≈1.36 bits; rank milestones under prism are unavailable (null) so no earlier rank‑1 is demonstrated (001_layers_baseline/run-latest/output-Yi-34B.json:848–901). Verdict: Neutral — KL improves modestly but without rank milestone shifts.

## 4.6. Ablation & stress tests
Ablation (no_filler): L_sem_orig=44; L_sem_nf=44; ΔL_sem=0 (001_layers_baseline/run-latest/output-Yi-34B.json:6086–6093). Control prompt “Berlin is the capital of” strongly favors “Germany” (entropy≈0.98; top‑1 “ Germany”) (001_layers_baseline/run-latest/output-Yi-34B.json:406–438). Important‑word trajectory: the prompt token “Germany” is present across early layers (e.g., records rows 15/32/49/66 for token position 13; 001_layers_baseline/run-latest/output-Yi-34B-records.csv:15, 001_layers_baseline/run-latest/output-Yi-34B-records.csv:32, 001_layers_baseline/run-latest/output-Yi-34B-records.csv:49, 001_layers_baseline/run-latest/output-Yi-34B-records.csv:66). “Berlin” emerges as top‑1 at L=44 and remains at L=60 (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:46, 001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:63).

## 4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓ (pre_norm; next_ln1; eps inside sqrt) (001_layers_baseline/run-latest/output-Yi-34B.json:2841–2864)
- LayerNorm bias removed ✓ (unembed_bias.present=false) (001_layers_baseline/run-latest/output-Yi-34B.json:835–844)
- FP32 unembed promoted ✓ (use_fp32_unembed=true) (001_layers_baseline/run-latest/output-Yi-34B.json:810–818)
- Punctuation / markup anchoring noted ✓ (early top‑1 are punctuation/markup; e.g., L1/L2 top‑1 “.”,“ODM”) (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:3, 001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:4)
- Copy‑reflex ✗ (no strict/soft copy milestones) (001_layers_baseline/run-latest/output-Yi-34B.json:4740–4776)
- Preferred lens honored ✓ (tuned; use_confirmed_semantics=true) (001_layers_baseline/run-latest/output-Yi-34B.json:7837–7848)
- Confirmed semantics reported ✓ (L_semantic_confirmed=44 via tuned) (001_layers_baseline/run-latest/output-Yi-34B.json:5480–5486, 001_layers_baseline/run-latest/output-Yi-34B-milestones.csv:4)
- Dual‑lens artefact metrics (incl. v2, JS/Jaccard/L1) cited ✓ (001_layers_baseline/run-latest/output-Yi-34B.json:2689–2720, 001_layers_baseline/run-latest/output-Yi-34B.json:7878–7890)
- Tuned‑lens audit done (rotation/temp/pos/head) ✓ (001_layers_baseline/run-latest/output-Yi-34B.json:7900–7931)
- normalization_provenance present ✓ (ln_source @ L0/final) (001_layers_baseline/run-latest/output-Yi-34B.json:2735, 001_layers_baseline/run-latest/output-Yi-34B.json:3401)
- per‑layer normalizer effect present ✓ (resid_norm_ratio, delta_resid_cos in CSV/JSON) (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:46)
- deterministic_algorithms true ✓ (001_layers_baseline/run-latest/output-Yi-34B.json:6017–6031)
- numeric_health clean ✓ (001_layers_baseline/run-latest/output-Yi-34B.json:3759–3766)
- copy_mask plausible ✓ (token id sample shown) (001_layers_baseline/run-latest/output-Yi-34B.json:980–1008)
- milestones.csv or evaluation_pack.citations used ✓ (001_layers_baseline/run-latest/output-Yi-34B-milestones.csv:3–4)

---
Produced by OpenAI GPT-5 
*Run executed on: 2025-10-12 20:56:18*
