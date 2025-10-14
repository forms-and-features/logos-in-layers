# Evaluation Report: Qwen/Qwen2.5-72B

*Run executed on: 2025-10-12 20:56:18*

## EVAL

**Overview**
- Model: `Qwen/Qwen2.5-72B`; single‑fact probe using the positive prompt “Give the city name only, plain text. The capital of Germany is called simply” (no trailing space) 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:4.
- Run timestamp: 2025‑10‑12 20:56 (see `timestamp-20251012-2056`). Probe measures copy vs. semantic onset, KL/rank/cosine/entropy trajectories, and lens diagnostics (raw vs. normalized; Prism sidecar; tuned lens availability).

**Method sanity‑check**
- Prompt & indexing: Context prompt ends with “called simply” 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:4. Positive rows exist (e.g., `prompt_id=pos,prompt_variant=orig,layer=80`) 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:138.
- Normalizer provenance: `arch="pre_norm"`, `strategy="next_ln1"` 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7619-7621. L0 `ln_source=blocks[0].ln1` 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7624; final uses `ln_final` 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8344.
- Per‑layer normalizer effect: Metrics present (`resid_norm_ratio`, `delta_resid_cos`); notable early spikes (e.g., L2 ratio≈8.94) 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:2 and in provenance (L2–L5 spikes) 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7642-7669. Flag present: `normalization_spike=true` 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:839-842.
- Unembed bias: `present=false`, `l2_norm=0.0` — cosines are bias‑free 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:834-837.
- Environment & determinism: `device="cpu"`, Torch 2.8, `deterministic_algorithms=true`, `seed=316` 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9587-9593. Repeatability self‑test skipped due to deterministic env 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8848-8851.
- Numeric health: `any_nan=false`, `any_inf=false`, `layers_flagged=[]` 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8838-8846.
- Copy mask: Size `6244`; plausible list dominated by punctuation/specials (e.g., “+ , - . / :”) 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7248-7255.
- Gold alignment: `ok=true`, variant `with_space`, pieces `ĠBerlin` 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8852-8862. Rate = 1.0 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8898-8899.
- Repeatability (ranks): skipped; `{max_rank_dev,p95_rank_dev,top1_flip_rate} = null` 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9810-9814.
- Norm trajectory: shape `spike`, slope≈0.064, r²≈0.923 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9200-9205.
- Measurement guidance: `{prefer_ranks=true, suppress_abs_probs=true, preferred_lens_for_reporting=norm, use_confirmed_semantics=false}` with reasons [`norm_only_semantics_window`, `high_lens_artifact_risk`, `high_lens_artifact_score`, `normalization_spike`] 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9760-9772.
- Semantic margin: `{delta_abs=0.002, p_uniform≈6.6e‑06, margin_ok_at_L_semantic_norm=true}` 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9652-9657.
- Micro‑suite: Present with `n=5`, `n_missing=3`, `L_semantic_norm_median=80` and notes 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9774-9788,9826-9830.

**Quantitative findings (layer‑by‑layer)**
- Preferred lens: norm (per measurement guidance). Bolded semantic layer reflects L_semantic_norm (confirmed window absent).

| Layer | Entropy (bits) | Top‑1 token | KL→final (bits) | Answer rank |
|---|---:|---|---:|---:|
| 0 | 17.21 | “s” | 9.39 | 67377 | 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:1
| 20 | 16.13 | “有期徒” | 9.84 | — | 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:22
| 40 | 16.19 | “iéndo” | 9.81 | — | 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:98
| 60 | 16.89 | “'” | 7.63 | 44533 | 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:118
| **80** | 4.12 | “Berlin” | 0.0001 | 1 | 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:138

- Control margin: `first_control_margin_pos=0`, `max_control_margin≈0.207` 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9640-9644.
- Micro‑suite: median `L_semantic_norm=80`, IQR [80, 80]; example fact citation “Germany→Berlin” at row 80; “Japan→Tokyo” at row 566 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9191-9196.
- Entropy drift: median gaps vs teacher entropy ≈12.5 bits (p50; p25≈12.03, p75≈12.77) 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9826-9829.
- Normalizer effect snapshots: early `resid_norm_ratio` spikes (e.g., L2) and large `delta_resid_cos` near depth 70–80 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8200-8280,8325-8349.

**Qualitative findings**

4.1. Copy vs semantics (Δ‑gap)
The probe finds no strict or soft copy reflex in early layers; all `copy_strict@{0.70,0.80,0.90,0.95}` are null and soft‐K windows are null with `stability="none"` 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7330-7349. Rank milestones show semantics only at the final layer (`first_rank_le_1=80`, `first_rank_le_5=78`, `first_rank_le_10=74`) 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7264-7268. With no detected copy milestone, Δ̂ is not defined (evaluation_pack `delta_hat=null`) 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9790-9794,9798-9807.

4.2. Lens sanity: Raw‑vs‑Norm
Artifact risk is high: `lens_artifact_score_v2≈0.743` (tier=high), `js_divergence_p50≈0.105`, `l1_prob_diff_p50≈0.615`, with no early stabilization (`first_js_le_0.1=0`, `first_l1_le_0.5=0`) 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9798-9805. Top‑K overlap is modest (`jaccard_raw_norm_p50≈0.316`, no layer ≥0.5) 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9803. Prevalence shows `pct_layers_kl_ge_1.0≈0.321` and a norm‑only semantic at L80 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7420-7423. Caution: prefer ranks and the final‑layer confirmation; early semantics could be lens‑induced in families with high raw‑vs‑norm divergence.

4.3. Tuned‑Lens analysis
Tuned lens is missing for this model (`status="missing"`) 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9646-9650. Baseline last‑layer consistency is excellent (`kl_to_final_bits≈1.1e‑4`, `top1_agree=true`) 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8864-8872.

4.4. KL, ranks, cosine, entropy milestones
KL thresholds reach below 1.0 and 0.5 bits at L80 (`first_kl_below_1.0=80`, `first_kl_below_0.5=80`) 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7264-7266; final‑head calibration is strong (near‑zero KL) 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8865-8872. Ranks: `first_rank_le_10=74`, `first_rank_le_5=78`, `first_rank_le_1=80` under norm lens 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7266-7268. Cosine milestones (norm) occur late: `ge_0.6` at layer 53; `ge_0.2`/`ge_0.4` not reached 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7361-7366. Entropy declines sharply by L80 with large positive drift vs teacher entropy throughout (p50 gap≈12.5 bits) 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9826-9829. Uniform‑margin gate passes at L80 (`margin_ok_at_L_semantic_norm=true`) 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9652-9657.

4.5. Prism (shared‑decoder diagnostic)
Prism sidecars are present. Compared to norm lens, early‑layer KL is similar/slightly higher, and late‑layer behavior diverges: at L80, Prism does not recover the answer (answer rank ≫1, KL large) 001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token-prism.csv:120. Verdict: Regressive (higher KL late and no improvement in rank milestones).

4.6. Ablation & stress tests
No‑filler ablation leaves semantics unchanged: `L_sem_orig=80`, `L_sem_nf=80`, `ΔL_sem=0` 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9605-9612. Control prompts are provided; for “Berlin is the capital of”, the model’s top suggestion is the country name (plausible) 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:10-17. Important‑word trajectory: by L79–L80, “Berlin” rises to top‑k at multiple prompt positions (e.g., pos=13/14/15) 001_layers_baseline/run-latest/output-Qwen2.5-72B-records.csv:4066,4070,4106.

4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓ 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:812-817
- LayerNorm bias removed ✓ 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:834-837
- FP32 unembed promoted ✓ 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:818-821
- Punctuation/markup anchoring noted ✓ 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:15-29
- Copy‑reflex ✗ 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7330-7349
- Preferred lens honored ✓ 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9769-9772
- Confirmed semantics reported n.a. 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9096-9100
- Dual‑lens artifact metrics cited ✓ 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9798-9807
- Tuned‑lens audit done n.a. 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9646-9650
- normalization_provenance present ✓ 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7618-7624,8343-8349
- per‑layer normalizer effect present ✓ 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8200-8280
- deterministic_algorithms true ✓ 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:9587-9593
- numeric_health clean ✓ 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:8838-8846
- copy_mask plausible ✓ 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:7248-7255,990-1010
- milestones.csv used for quotes ✓ 001_layers_baseline/run-latest/output-Qwen2.5-72B-milestones.csv:3

---
Produced by OpenAI GPT-5
