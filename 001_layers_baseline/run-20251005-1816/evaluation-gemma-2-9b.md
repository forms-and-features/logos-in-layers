# Evaluation Report: google/gemma-2-9b

*Run executed on: 2025-10-05 18:16:50*
## 1. Overview

Gemma 2 9B (google/gemma-2-9b), evaluated on 2025-10-05 (timestamp-20251005-1816). The probe traces layer-by-layer dynamics of copy vs. semantics using KL-to-final, rank milestones, cosine alignment, and entropy, with raw-vs-norm lens diagnostics and tuned-lens audits.

## 2. Method sanity-check

- Prompt & indexing: context ends cleanly with “called simply” (no trailing space). Quote: "context_prompt: '... The capital of Germany is called simply'"  001_layers_baseline/run-latest/output-gemma-2-9b.json:4. Positive rows present: e.g., pos,orig L0/L42 in pure CSV [row 2; row 49 in 001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv].
- Normalizer provenance: strategy "next_ln1" with RMSNorm; layer 0 ln_source "blocks[0].ln1" and final uses "ln_final". Quotes: "strategy": "next_ln1"  001_layers_baseline/run-latest/output-gemma-2-9b.json:5901; "ln_source": "blocks[0].ln1"  001_layers_baseline/run-latest/output-gemma-2-9b.json:5910; "norm": "ln_final"  001_layers_baseline/run-latest/output-gemma-2-9b.json:6549.
- Per-layer normalizer effect: spike flagged by pipeline; treat early-layer effects with caution. Quote: "flags.normalization_spike": true  001_layers_baseline/run-latest/output-gemma-2-9b.json:838; "norm_trajectory.shape": "spike"  001_layers_baseline/run-latest/output-gemma-2-9b.json:7912.
- Unembed bias: absent. Quote: "unembed_bias": { "present": false, "l2_norm": 0.0 }  001_layers_baseline/run-latest/output-gemma-2-9b.json:826–834. Cosines are bias‑free by construction.
- Environment & determinism: CPU fp32 compute, deterministic algorithms true, seed=316. Quote: "device": "cpu", "dtype_compute": "torch.float32", "deterministic_algorithms": true, "seed": 316  001_layers_baseline/run-latest/output-gemma-2-9b.json:7214–7223.
- Numeric health: clean. Quote: "any_nan": false, "any_inf": false, "layers_flagged": []  001_layers_baseline/run-latest/output-gemma-2-9b.json:6554–6561.
- Copy mask: ignored_token_ids list present (sample: 108–121, …); plausible punctuation/control span for tokenizer. Quote: "copy_mask": { "ignored_token_ids": [108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, …]  001_layers_baseline/run-latest/output-gemma-2-9b.json:944–962.
- Gold alignment: OK for positive and control prompts. Quote: "gold_alignment": { "ok": true, "variant": "with_space", "pieces": ["▁Berlin"] }  001_layers_baseline/run-latest/output-gemma-2-9b.json:6565–6581.
- Repeatability (1.39): skipped in deterministic env. Quote: "repeatability": { "status": "skipped", "reason": "deterministic_env" }  001_layers_baseline/run-latest/output-gemma-2-9b.json:6561–6564.
- Norm trajectory: spike pattern with high fit. Quote: "shape": "spike", "slope": 0.0517, "r2": 0.987  001_layers_baseline/run-latest/output-gemma-2-9b.json:7912–7920.
- Measurement guidance: prefer ranks, suppress abs probs; preferred lens=norm; use_confirmed_semantics=true. Quote: { "prefer_ranks": true, "suppress_abs_probs": true, "preferred_lens_for_reporting": "norm", "use_confirmed_semantics": true }  001_layers_baseline/run-latest/output-gemma-2-9b.json:7850–7866.

## 3. Quantitative findings (layer‑by‑layer)

Short table (pos, orig; NEXT token only; tokens shown without probabilities per guidance):
- L 0 — entropy 1.672e-05 bits, top‑1 'simply'  [row 2 in 001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv]
- L 3 — entropy 4.300e-04 bits, top‑1 'simply'  [row 5 in 001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv]
- L 21 — entropy 1.8665 bits, top‑1 'simply'  [row 28 in 001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv]
- L 32 — entropy 0.0625 bits, top‑1 '"' (punctuation)  [row 39 in 001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv]
- L 42 — entropy 0.3701 bits, top‑1 'Berlin'  [row 49 in 001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv]

Semantic layer (preferred lens = norm) is L 42 and is confirmed by tuned lens at the same depth. Quote: "L_semantic_norm": 42; "L_semantic_confirmed": { "layer": 42, "source": "tuned" }  001_layers_baseline/run-latest/output-gemma-2-9b.json:7022–7029, 7866–7880.

Control margin: first_control_margin_pos=18; max_control_margin=0.8677. Quote: { "first_control_margin_pos": 18, "max_control_margin": 0.8677237033843427 }  001_layers_baseline/run-latest/output-gemma-2-9b.json:7269–7273. The control NEXT at L 42 is 'Paris' [row 148 in 001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv].

Entropy drift vs teacher: entropy_gap_bits p25/p50/p75 = −2.89/−2.80/−1.63 bits (teacher=2.9367 bits). Quote: "teacher_entropy_bits": 2.9366900987731905  001_layers_baseline/run-latest/output-gemma-2-9b.json:5797; and "entropy_gap_bits_p25": −2.8923, "p50": −2.8026, "p75": −1.6322  001_layers_baseline/run-latest/output-gemma-2-9b.json:7918–7923.

Snapshots of normalizer effect near endpoints: L0 resid_norm_ratio=0.757, delta_resid_cos=0.929; final stream uses ln_final. Quote: layer 0 { "resid_norm_ratio": 0.757178..., "delta_resid_cos": 0.929459... }  001_layers_baseline/run-latest/output-gemma-2-9b.json:5915–5919; final { "norm": "ln_final" }  001_layers_baseline/run-latest/output-gemma-2-9b.json:6549.

## 4. Qualitative findings

### 4.1. Copy vs semantics (Δ‑gap)
Copy‑reflex ✓. Early layers show strict copy collapse of the trailing prompt token ('simply') at L=0–3 (copy_collapse and strict detectors set) [rows 2–5 in 001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv]. The earliest strict copy at τ=0.70 and τ=0.95 occurs at L=0; stability tag is "mixed". Quote: "L_copy_strict": { "0.7": 0, …, "0.95": 0 }, "stability": "mixed"  001_layers_baseline/run-latest/output-gemma-2-9b.json:5698–5718. Using the preferred rank framing, semantics arrives only at the final layer, L_semantic_norm=42 (confirmed). Depth fraction Δ̂ = 1.0 (evaluation pack). Quote: "depth_fractions": { "delta_hat": 1.0 }  001_layers_baseline/run-latest/output-gemma-2-9b.json:7879–7890.

Interpretation: a classic copy‑then‑decide pattern with very late decision. Δ̂ near 1.0 indicates maximal lag between first copy‑like behavior and semantic commitment in this setup.

### 4.2. Lens sanity: Raw‑vs‑Norm
Artifact tier is high under the v2 score; JS/L1 medians are small but a norm‑only semantic layer exists at L=42 (window/full). Quotes: "lens_artifact_score_v2": 0.5906, "tier": "high"  001_layers_baseline/run-latest/output-gemma-2-9b.json:5797–5810; "js_divergence_p50": 0.00625, "l1_prob_diff_p50": 0.0292, "first_js_le_0.1": 0, "first_l1_le_0.5": 0  001_layers_baseline/run-latest/output-gemma-2-9b.json:5810–5824; top‑K overlap p50=0.6393 with first ≥0.5 at L=3  001_layers_baseline/run-latest/output-gemma-2-9b.json:5798–5805. Prevalence: pct_layers_kl_ge_1.0=0.3023; n_norm_only_semantics_layers=1; earliest=42  001_layers_baseline/run-latest/output-gemma-2-9b.json:5778–5797.

Caution: with high artifact risk and a norm‑only semantic designation at the final layer, prefer rank milestones and confirmed semantics; avoid absolute probabilities (guidance sets suppress_abs_probs=true).

### 4.3. Tuned‑Lens analysis
Preference: not calibration‑only (tuned_is_calibration_only=false) but preferred lens for semantics remains norm (hint=norm). Quote: { "tuned_is_calibration_only": false, "preferred_semantics_lens_hint": "norm" }  001_layers_baseline/run-latest/output-gemma-2-9b.json:7940–7950, 7895–7908.

Attribution: rotation vs temperature deltas are small near median (ΔKL_rot_p50≈+0.0008, ΔKL_temp_p50≈−0.0110; interaction ≈+0.0041). Quote: "rotation_vs_temperature" percentiles  001_layers_baseline/run-latest/output-gemma-2-9b.json:7908–7920.

Rank earliness: unchanged vs norm (first_rank_le_{10,5,1} all at L=42 for both). Quote: tuned summaries show "first_rank_le_1": 42  001_layers_baseline/run-latest/output-gemma-2-9b.json:6648–6674.

Head mismatch: τ* model calibration reduces tuned final KL from 1.013 bits to 0.339 bits. Quote: { "kl_bits_tuned_final": 1.0128875, "kl_bits_tuned_final_after_tau_star": 0.3391450, "tau_star_modelcal": 2.7020 }  001_layers_baseline/run-latest/output-gemma-2-9b.json:7948–7950.

Last‑layer agreement: top‑1 agrees but final KL is high pre‑calibration; avoid inferring absolute p from final head. Quote: "last_layer_consistency": { "top1_agree": true, "kl_to_final_bits": 1.0129, "warn_high_last_layer_kl": true }  001_layers_baseline/run-latest/output-gemma-2-9b.json:6577–6617.

### 4.4. KL, ranks, cosine, entropy milestones
- KL: under norm, first_kl_below_{1.0,0.5} = null; tuned achieves first_kl_below_1.0 at L=42 (still no ≤0.5). Quotes: norm: "first_kl_below_1.0": null  001_layers_baseline/run-latest/output-gemma-2-9b.json:5642–5644; tuned: "first_kl_below_1.0": 42  001_layers_baseline/run-latest/output-gemma-2-9b.json:6783–6798.
- Ranks: preferred lens (norm) reaches first_rank_le_{10,5,1} at L=42 (confirmed semantics). Quote: { "first_rank_le_10": 42, "first_rank_le_5": 42, "first_rank_le_1": 42 }  001_layers_baseline/run-latest/output-gemma-2-9b.json:5644–5646.
- Cosine: cos milestones (norm) ge_0.2 at L=1; ge_0.4/ge_0.6 at L=42. Quote: "cos_milestones": { "norm": { "ge_0.2": 1, "ge_0.4": 42, "ge_0.6": 42 } }  001_layers_baseline/run-latest/output-gemma-2-9b.json:5739–5754.
- Entropy: non‑monotonic; very low near L~32 (punctuation decoding), then rises slightly by L=42; consistently below teacher (negative gap quantiles). Quotes: L32 entropy_bits=0.0625 [row 39 in 001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv]; teacher_entropy_bits=2.9367  001_layers_baseline/run-latest/output-gemma-2-9b.json:5797; entropy_gap p50=−2.8026  001_layers_baseline/run-latest/output-gemma-2-9b.json:7919–7921.

### 4.5. Prism (shared‑decoder diagnostic)
Present and compatible. KL deltas are negative (Prism increases KL substantially) with no earlier rank milestones; verdict: Regressive. Quote: p50 KL baseline 15.17 → prism 25.51 (Δ=−10.33) and rank milestones prism = null  001_layers_baseline/run-latest/output-gemma-2-9b.json:859–886.

### 4.6. Ablation & stress tests
Prompt ablation (no_filler): stable — ΔL_copy=0; ΔL_sem=0. Quote: { "L_copy_orig": 0, "L_sem_orig": 42, "L_copy_nf": 0, "L_sem_nf": 42 }  001_layers_baseline/run-latest/output-gemma-2-9b.json:7235–7248.

Control prompt: consistent; the control NEXT at L=42 is 'Paris' with a large control margin (see control_summary and [row 148 in 001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv]).

Important‑word trajectory: the prompt token 'Germany' remains strongly identified across early layers (e.g., L0–L3) [rows 15, 35, 54, 71 in 001_layers_baseline/run-latest/output-gemma-2-9b-records.csv]. The answer 'Berlin' only becomes the NEXT top‑1 at L=42 [row 49 in 001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv].

### 4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓ ("final_ln_type": "RMSNorm"  001_layers_baseline/run-latest/output-gemma-2-9b.json:811)
- LayerNorm bias removed ✓ (unembed bias absent  001_layers_baseline/run-latest/output-gemma-2-9b.json:826–834)
- FP32 unembed promoted ✓ ("unembed_dtype": "torch.float32", "mixed_precision_fix": "casting_to_fp32_before_unembed"  001_layers_baseline/run-latest/output-gemma-2-9b.json:809,815)
- Punctuation / markup anchoring noted ✓ (L32 punctuation top‑1) [row 39 in 001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv]
- Copy‑reflex ✓ (early strict copy and copy_collapse in L0–3) [rows 2–5 in 001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv]
- Preferred lens honored ✓ (norm; measurement_guidance)  001_layers_baseline/run-latest/output-gemma-2-9b.json:7850–7866
- Confirmed semantics reported ✓ (confirmed at L=42, source=tuned)  001_layers_baseline/run-latest/output-gemma-2-9b.json:7866–7880
- Dual‑lens artefact metrics cited ✓ (v2 score, JS/Jaccard/L1, prevalence)  001_layers_baseline/run-latest/output-gemma-2-9b.json:5778–5824, 5798–5810
- Tuned‑lens audit done ✓ (rotation/temp/positional/head)  001_layers_baseline/run-latest/output-gemma-2-9b.json:7895–7950
- normalization_provenance present ✓ (ln_source @ L0/final)  001_layers_baseline/run-latest/output-gemma-2-9b.json:5910, 6549
- per‑layer normalizer effect present ✓ (resid_norm_ratio, delta_resid_cos)  001_layers_baseline/run-latest/output-gemma-2-9b.json:5915–5919
- deterministic_algorithms true ✓ (provenance.env)  001_layers_baseline/run-latest/output-gemma-2-9b.json:7216–7223
- numeric_health clean ✓  001_layers_baseline/run-latest/output-gemma-2-9b.json:6554–6561
- copy_mask plausible ✓  001_layers_baseline/run-latest/output-gemma-2-9b.json:944–962
- milestones.csv used for quick citation ✓  001_layers_baseline/run-latest/output-gemma-2-9b-milestones.csv:1–4

---
Produced by OpenAI GPT-5 

