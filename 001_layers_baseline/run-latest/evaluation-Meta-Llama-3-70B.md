# Evaluation Report: meta-llama/Meta-Llama-3-70B

*Run executed on: 2025-10-11 21:50:12*

## 1. Overview
Meta-Llama-3-70B (80 layers; pre-norm) evaluated on 2025-10-11. The probe tracks copy vs. semantic emergence using rank and KL milestones, cosine alignment, and entropy trajectories, with dual-lens diagnostics (raw vs. norm) and final-head calibration checks.

## 2. Method sanity‑check
- Prompt & indexing: context prompt ends with “called simply”. “Give the city name only, plain text. The capital of Germany is called simply” (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:816). Positive rows for `prompt_id=pos`, `prompt_variant=orig` are present, e.g., Germany→Berlin at layer 40 [row 42 in CSV].
- Normalizer provenance: `strategy: "next_ln1"` with pre‑norm arch. First/last sources: `per_layer[0].ln_source = blocks[0].ln1` and final `ln_source = ln_final` (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7387, 714, 815).
- Per‑layer normalizer effect: early spikes are flagged and large before semantics, e.g., layer 1 `resid_norm_ratio=11.54`, `delta_resid_cos=0.689` (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7431–7440). `flags.normalization_spike: true` (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:834–841).
- Unembed bias: `present=false`, `l2_norm=0.0`; cosines are bias‑free (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:828–836).
- Environment & determinism: `device=cpu`, `torch=2.8.0`, `deterministic_algorithms=true`, `seed=316` (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9363–9412). Reproducibility OK.
- Numeric health: `any_nan=false`, `any_inf=false`, `layers_flagged=[]` (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8615–8621).
- Copy mask: `ignored_token_ids` present (e.g., 0–14, 25–31, …) — plausible for special/control tokens (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:988–1018).
- Gold alignment: `ok=true`, pieces `["ĠBerlin"]`; `gold_alignment_rate=1.0` (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8626–8671).
- Repeatability (1.39): skipped in deterministic env (`status: "skipped"`) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8622–8625).
- Norm trajectory: `shape: "spike"`, `slope≈0.0498`, `r2≈0.937`, `n_spikes=15` (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8975–8990).
- Measurement guidance: `prefer_ranks=true`, `suppress_abs_probs=true`, `preferred_lens_for_reporting="norm"`, `use_confirmed_semantics=true` (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9522–9535).
- Semantic margin: `delta_abs=0.002`, `p_uniform≈7.8e‑06`, `margin_ok_at_L_semantic_norm=false` (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9424–9431).
- Micro‑suite: aggregates present with `n=5`, `n_missing=0`; median `L_semantic_confirmed=40`, median `L_semantic_norm=51` (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9511–9522, 9424–9432).

## 3. Quantitative findings (layer‑by‑layer)
Positive prompt only (`prompt_id=pos`, `prompt_variant=orig`).

| Layer | Entropy (bits) | Top‑1 token | Answer rank | KL to final (bits) | cos_to_final |
|---|---:|---|---:|---:|---:|
| 0 | 16.968 | “winding” | 115765 | 10.502 | 0.0043 |  [row 2 in CSV]
| 20 | 16.946 | “nut” | 6991 | 10.450 | −0.0258 |  [row 22 in CSV]
| **40** | 16.937 | “Berlin” | 1 | 10.420 | 0.0968 |  [row 42 in CSV]
| 60 | 16.923 | “Berlin” | 1 | 10.310 | 0.0742 |  [row 62 in CSV]
| 80 | 2.589 | “Berlin” | 1 | 0.00073 | 0.99999 |  [row 82 in CSV]

Bolded layer is the confirmed semantic layer per guidance: `L_semantic_confirmed=40` (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8871–8876, 9550–9558). Margin gate at `L_semantic_norm` fails (`margin_ok_at_L_semantic_norm=false`; 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9424–9431), so confirmed semantics are used.

Control margins: `first_control_margin_pos=0`, `max_control_margin≈0.517` (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9414–9420).

Micro‑suite: median `L_semantic_confirmed=40`, median `L_semantic_norm=51` (IQR 49–60); 0 missing facts. Example citation: Germany→Berlin `L_semantic_confirmed_row=40` [row 42 in CSV] (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9598–9608).

Entropy drift: median entropy gap vs teacher is ~14.34 bits (`entropy_gap_bits_p50`) with teacher entropy ≈2.597 bits (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9560–9590, 7212).

Confidence and normalizer snapshots at L=40: `answer_logit_gap≈0.0309`, `resid_norm_ratio≈1.23`, `delta_resid_cos≈0.977` [row 42 in CSV].

## 4. Qualitative findings

### 4.1. Copy vs semantics (Δ‑gap)
No strict or soft copy collapse is detected in early layers: `L_copy_strict` is null at τ∈{0.70,0.80,0.90,0.95} and soft‑k≈{1,2,3} also null; stability `"none"` (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7099–7130). Early rows 0–5 show `copy_collapse=False` and `copy_soft_k1@0.5=False` [rows 2–7 in CSV]. With no copy milestone, Δ̂ is undefined (`delta_hat=null`; 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9551–9558).

### 4.2. Lens sanity: Raw‑vs‑Norm
Artifact score (legacy/v2): 0.332 / 0.344, tier=medium (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9558–9565). Symmetric metrics are strong: `js_divergence_p50≈0.00245`, `l1_prob_diff_p50≈0.092`; `first_js_le_0.1=0`, `first_l1_le_0.5=0` (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9560–9565). Top‑K overlap: `jaccard_raw_norm_p50≈0.515`, first ≥0.5 at layer 11 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9564–9566). Prevalence: `pct_layers_kl_ge_1.0≈0.0123`, `n_norm_only_semantics_layers=2`, earliest at 79 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9566–9565, 7161–7220). Caution: norm‑only semantics occur late (79–80), far from the candidate semantic layer (40), so early semantics are unlikely lens‑induced.

### 4.3. Tuned‑Lens analysis
Tuned lens is not present for this run (`tuned_lens.status: "missing"`; 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9418–9423, 8860–8866). Per guidance, the norm lens is preferred for semantics and calibration is verified via last‑layer consistency below.

### 4.4. KL, ranks, cosine, entropy milestones
Final‑head calibration is excellent: `kl_to_final_bits≈0.00073`, `top1_agree=true` (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8638–8660). Layer‑wise KL remains >1 bit until late (e.g., 25/50/75% depths show ≈10.69/10.63/4.35 bits; 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8660–8670), then collapses at the end. Ranks (norm lens): `first_rank_le_10=38`, `le_5=38`, `le_1=40` (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:832–847). Cosine milestones (norm): ge_{0.2,0.4,0.6} all at 80 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7137–7146). Entropy stays ~16.9 bits through mid‑depth and drops to ~2.60 bits at the final row (teacher) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8985–8990, 7212). Margin gate reminder: although rank‑1 occurs at L=40, `margin_ok_at_L_semantic_norm=false`; thus we report confirmed semantics.

### 4.5. Prism (shared‑decoder diagnostic)
Present and compatible at sampled depths (embed, 19, 39, 59). KL deltas are negative (baseline − prism ≈ −0.9 to −1.16 bits at p25–p75), indicating worse KL under prism; rank milestones unchanged/null (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:841–870). Verdict: Regressive.

### 4.6. Ablation & stress tests
Style ablation (no‑filler) mildly delays semantics: `L_sem_nf=42` vs `L_sem_orig=40` (`ΔL_sem=2`, 2.5% of depth) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9380–9401). Control prompts show healthy separation: `first_control_margin_pos=0`, `max_control_margin≈0.517` (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9414–9420). Test prompt “Berlin is the capital of” yields top‑1 “ Germany” (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:805–846), consistent with correct directionality. Important‑word trajectory: “Berlin” appears among salient tokens by L≈31 and around semantics (e.g., layer 31 includes “ Berlin” among high‑rank tokens; 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-records.csv:680; layer 40 shows “ Berlin” across positions; 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-records.csv:851–853).

### 4.7. Checklist (✓/✗/n.a.)
✓ RMS lens
✓ LayerNorm bias removed
✓ FP32 unembed promoted
✓ Punctuation / markup anchoring noted
✓ Copy‑reflex
✓ Preferred lens honored
✓ Confirmed semantics reported
✓ Dual‑lens artefact metrics (incl. v2, JS/Jaccard/L1) cited
✓ Tuned‑lens audit done (rotation/temp/positional/head) n.a. (tuned missing)
✓ normalization_provenance present (ln_source @ L0/final)
✓ per‑layer normalizer effect present (resid_norm_ratio, delta_resid_cos)
✓ deterministic_algorithms true
✓ numeric_health clean
✓ copy_mask plausible
✓ milestones.csv / evaluation_pack.citations used for quotes

---
Produced by OpenAI GPT-5
