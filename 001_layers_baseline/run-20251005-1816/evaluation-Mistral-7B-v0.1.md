# Evaluation Report: mistralai/Mistral-7B-v0.1

*Run executed on: 2025-10-05 18:16:50*
## 1. Overview

This EVAL reviews Mistral-7B-v0.1 (32 layers) using the project’s layer-by-layer probe to distinguish prompt copying from emergent semantics and to track KL-to-final, rank, cosine, and entropy trajectories, with lens diagnostics. The run corresponds to the latest artifacts under run-latest; the context prompt ends with “called simply” and gold answer “Berlin” (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4, 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:5).

## 2. Method sanity-check

- Prompt & indexing: "Give the city name only, plain text. The capital of Germany is called simply" (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4). Positive/original rows present (e.g., L0, L25, L32) in `output-Mistral-7B-v0.1-pure-next-token.csv` [rows 2, 27, 34].
- Normalizer provenance: `strategy = "next_ln1"` with pre-norm arch (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2385–2387). Sources: L0 using `blocks[0].ln1` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2390) and final using `ln_final` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2677–2683).
- Per-layer normalizer effect: normalization spike flagged (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:831–834). Residual metrics are tracked per-layer (e.g., `resid_norm_ratio=115.17` at L0, `delta_resid_cos=0.308` at L0; 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2393–2395) and remain high through early layers; this is flagged by the pipeline.
- Unembed bias: `present=false`, `l2_norm=0.0` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:826–830). Cosine metrics are bias-free by construction.
- Environment & determinism: CPU, `torch=2.8.0+cu128`, `deterministic_algorithms=true`, `seed=316` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3547–3556).
- Numeric health: `any_nan=false`, `any_inf=false`, `layers_flagged=[]` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2888–2893).
- Copy mask: `size=1179`; sample includes punctuation tokens (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2128–2135).
- Gold alignment: `{ok=true, variant=with_space, pieces=[▁Berlin]}` with rate `1.0` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2898–2909, 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2942).
- Repeatability (1.39): skipped due to deterministic env; `{max_rank_dev, p95_rank_dev, top1_flip_rate} = null` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2894–2896, 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4224–4228).
- Norm trajectory: shape `spike`, slope `0.148`, r2 `0.963`, spikes `26` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4234–4239).
- Measurement guidance: `{prefer_ranks=true, suppress_abs_probs=true, preferred_lens_for_reporting="tuned", preferred_semantics_lens_hint="tuned", use_confirmed_semantics=true}` with reasons including `high_lens_artifact_risk` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4173–4186).

## 3. Quantitative findings (layer-by-layer)

Positive/original prompt rows from `output-Mistral-7B-v0.1-pure-next-token.csv`:
- L 0 — entropy 14.96 bits, top‑1 ‘dabei’ [row 2 in CSV].
- L 11 — entropy 14.74 bits, top‑1 ‘[…]’ [row 13 in CSV].
- L 25 — entropy 13.60 bits, top‑1 ‘Berlin’ [row 27 in CSV].
- L 26 — entropy 13.54 bits, top‑1 ‘Berlin’ [row 28 in CSV].
- L 32 — entropy 3.61 bits, top‑1 ‘Berlin’ [row 34 in CSV].

Semantic layer: bolding confirmed semantics per guidance. L_semantic_confirmed = 25, confirmed_source = raw (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4200–4203).

Control margin: `first_control_margin_pos=2`, `max_control_margin=0.6539` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3602–3608).

Entropy drift: entropy gaps vs teacher are large and stable—p25=10.60 bits, p50=10.99 bits, p75=11.19 bits (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4241–4243).

Confidence margins and normalizer snapshots: at L25, `answer_logit_gap=0.0146`, `resid_norm_ratio=9.97`, `delta_resid_cos=0.902` [row 27 in pure CSV; 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:265–2681 context for metrics]. At L26, `answer_vs_top1_gap≈0.508` [row 28 in CSV].

## 4. Qualitative findings

The model transitions from diffuse surface features to stable semantics in the last quarter of depth. Ranks and cosine indicate semantic consolidation around L25–L26, with ‘Berlin’ becoming top‑1 and cosine-to-final exceeding 0.6 by L26, followed by final‑head calibration at the last layer. Entropy falls steadily from ~15 bits toward the teacher value (~3.61 bits), consistent with a narrowing distribution as the representation aligns with the unembedding head.

### 4.1. Copy vs semantics (Δ‑gap)

No early copy‑reflex detected: `L_copy_strict` is null across τ∈{0.70,0.80,0.90,0.95}, and soft windows are null (stability="none") (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2216–2228). Early layers 0–3 show `copy_collapse=False` and `copy_soft_k1@0.5=False` [rows 2–5 in pure CSV]. The earliest semantic layer is L25 (confirmed raw), yielding Δ̂ undefined (no copy baseline; evaluation_pack.depth_fractions.delta_hat=null) (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4204–4208).

Threshold stability: earliest strict copy at τ=0.70 and τ=0.95 are null; `norm_only_flags[τ]` also null (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2216–2226).

### 4.2. Lens sanity: Raw‑vs‑Norm

Artifact risk is high. Legacy `lens_artifact_score=0.545` and new `lens_artifact_score_v2=0.670` with `tier="high"` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4211–4222). Symmetric metrics: `js_divergence_p50=0.074`, `l1_prob_diff_p50=0.505`, `first_js_le_0.1=0`, `first_l1_le_0.5=0` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4213–4216). Top‑K overlap: `jaccard_raw_norm_p50=0.408`, `first_jaccard_raw_norm_ge_0.5=19` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4217–4218). Prevalence: `pct_layers_kl_ge_1.0=0.242`, one norm‑only semantics layer at L32 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4219–4222; 2259–2270). Caution: early semantics near the candidate layer may be lens‑induced; defer to rank milestones and confirmed semantics.

### 4.3. Tuned‑Lens analysis

Preference: guidance prefers tuned and to use confirmed semantics (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4173–4186). Attribution: `ΔKL_rot` is the dominant term (p25=1.98, p50=2.20, p75=2.94), `ΔKL_temp` mixed (p50≈−0.24), with positive interaction (`ΔKL_interaction_p50=2.18`); overall `ΔKL_tuned_p50≈4.10` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4246–4255). Rank earliness varies across tuned variants: one tuned summary shows `first_rank_le_1=32` (later than norm), another shows `first_rank_le_1=25` (unchanged) (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2998–3007, 3111–3120). Positional generalization: `pos_in_dist_le_0.92=4.91`, `pos_ood_ge_0.96=4.10`, `pos_ood_gap≈−0.81` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4265–4267). Head mismatch and calibration: `kl_bits_tuned_final=0.0` and unchanged after τ⋆ with `tau_star_modelcal=1.0` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4269–4275). Last‑layer agreement is perfect: `kl_to_final_bits=0.0`, `top1_agree=true` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2910–2916). Baseline (norm) semantics at L25 remain the default for onset; tuned is treated as a calibration aid where it doesn’t improve rank earliness.

### 4.4. KL, ranks, cosine, entropy milestones

- KL: `first_kl_below_1.0=32`, `first_kl_below_0.5=32` under baseline summary (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2138–2146). Final KL≈0 per last-layer consistency (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2911).
- Ranks: preferred lens is tuned; tuned summaries report `first_rank_le_10∈{22,25}`, `first_rank_le_5∈{24,25}`, `first_rank_le_1∈{25,32}` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2998–3007, 3116–3120). Baseline norm `L_semantic_norm=25` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4200–4203, 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-milestones.csv:3).
- Cosine: norm‑lens milestones `ge_0.2=11`, `ge_0.4=25`, `ge_0.6=26` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2240–2250).
- Entropy: monotonic decrease toward teacher entropy with large gaps throughout mid‑depth (`p50≈10.99` bits) (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4241–4243).

### 4.5. Prism (shared‑decoder diagnostic)

Prism artifacts are present and compatible (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:835–840). Rank milestones for prism are null in summary (no earlier `le_1`) (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:836–855 context). Verdict: Neutral — no clear early KL drop or rank improvement in the summary.

### 4.6. Ablation & stress tests

No‑filler ablation: `L_sem_orig=25`, `L_sem_nf=24` → `ΔL_sem=−1` (≈3% of 32 layers), indicating low stylistic sensitivity (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3568–3576).

Control prompts: control summary present with early positive margin (`first_control_margin_pos=2`) and `max_control_margin≈0.654` (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3602–3608).

Important‑word trajectory (records CSV):
- “called” shows top‑1 ‘Berlin’ at L24 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-records.csv:425).
- “is” shows ‘Berlin’ in top‑2 at L23 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-records.csv:407–408).
- The final token slot “simply” includes both ‘Berlin’ and ‘Germany’ in its top‑k by L22–24 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-records.csv:392, 409, 426).

### 4.7. Checklist (✓/✗/n.a.)

- RMS lens ✓
- LayerNorm bias removed ✓
- FP32 unembed promoted ✓ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3549–3552)
- Punctuation / markup anchoring noted ✓ (copy mask sample includes punctuation; 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2128–2134)
- Copy‑reflex ✗ (no strict/soft hit; 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2216–2228)
- Preferred lens honored ✓ (tuned for reporting; confirmed semantics used; 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4173–4186, 4200–4203)
- Confirmed semantics reported ✓ (L=25 raw; 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4200–4203)
- Dual‑lens artefact metrics (incl. v2, JS/Jaccard/L1) cited ✓ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4211–4218)
- Tuned‑lens audit done (rotation/temp/positional/head) ✓ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4246–4275)
- normalization_provenance present (ln_source @ L0/final) ✓ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2390, 2677–2683)
- per‑layer normalizer effect present (resid_norm_ratio, delta_resid_cos) ✓ (e.g., 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2393–2395)
- deterministic_algorithms true ✓ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3552)
- numeric_health clean ✓ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2888–2893)
- copy_mask plausible ✓ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2128–2135)
- milestones.csv or evaluation_pack.citations used for quotes ✓ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-milestones.csv:3, 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4278–4286)

---
Produced by OpenAI GPT-5 

