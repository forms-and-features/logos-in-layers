# Evaluation Report: Qwen/Qwen3-14B

*Run executed on: 2025-10-05 18:16:50*

**1. Overview**
- Model: `Qwen/Qwen3-14B` run on 2025‑10‑05 (see `001_layers_baseline/run-latest/timestamp-20251005-1816`).
- Probe measures copy vs. semantics with layer‑wise KL to final head, rank milestones, cosine/entropy trajectories, and raw‑vs‑norm lens diagnostics (plus tuned‑lens and prism sidecars).

**2. Method Sanity‑Check**
- Prompt & indexing: context prompt ends with “called simply” and no trailing space: `"context_prompt": "Give the city name only, plain text. The capital of Germany is called simply"` (001_layers_baseline/run-latest/output-Qwen3-14B.json:817). Positive rows exist (e.g., `pos,orig,0,...`) (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:2).
- Normalizer provenance: `"strategy": "next_ln1"` (001_layers_baseline/run-latest/output-Qwen3-14B.json:7342); endpoints show `layer 0 → blocks[0].ln1` (…:7346) and `layer 40 → ln_final` (…:7706).
- Per‑layer normalizer effect: normalization spike flagged (`"normalization_spike": true`) (001_layers_baseline/run-latest/output-Qwen3-14B.json:833); norm trajectory `shape: "spike"` (001_layers_baseline/run-latest/output-Qwen3-14B.json:8456).
- Unembed bias: `"present": false` (001_layers_baseline/run-latest/output-Qwen3-14B.json:827). Cosines are bias‑free.
- Environment & determinism: device `cpu`, torch `2.8.0+cu128`, `deterministic_algorithms: true`, `seed: 316` (001_layers_baseline/run-latest/output-Qwen3-14B.json:8622,8625,8627,8629). Reproducibility OK.
- Numeric health: `any_nan=false`, `any_inf=false`, `layers_flagged=[]` (001_layers_baseline/run-latest/output-Qwen3-14B.json:7964).
- Copy mask: sample shows punctuation/markup (e.g., `!", #, $, %`) with `size: 6112` (001_layers_baseline/run-latest/output-Qwen3-14B.json:7057; 001_layers_baseline/run-latest/output-Qwen3-14B.json:7075). Plausible for tokenizer filtering.
- Gold alignment: `ok: true`, variant `with_space`, pieces `["ĠBerlin"]` (001_layers_baseline/run-latest/output-Qwen3-14B.json:7974). `gold_alignment_rate: 1.0` (001_layers_baseline/run-latest/output-Qwen3-14B.json:8018).
- Repeatability (1.39): status `skipped` due to `deterministic_env` (001_layers_baseline/run-latest/output-Qwen3-14B.json:7963). No rank‑dev metrics.
- Norm trajectory: `shape: "spike"`, slope 0.1107, r2 0.9046 (001_layers_baseline/run-latest/output-Qwen3-14B.json:8456).
- Measurement guidance: `prefer_ranks=true`, `suppress_abs_probs=true`, `preferred_lens_for_reporting="tuned"`, `use_confirmed_semantics=true` (001_layers_baseline/run-latest/output-Qwen3-14B.json:9256).

**3. Quantitative Findings (Layer‑by‑Layer)**
- Table (positive prompt only, `pos`/`orig`; see CSV rows):
  - L 0 — entropy 17.213 bits, top‑1 ‘梳’ [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:2]
  - L 10 — entropy 17.170 bits, top‑1 ‘(?)’ [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:12]
  - L 20 — entropy 16.932 bits, top‑1 ‘____’ [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:22]
  - L 30 — entropy 7.789 bits, top‑1 ‘这个名字’ [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:32]
  - L 36 — entropy 0.312 bits, top‑1 ‘Berlin’ [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:38]
  - L 40 — entropy 3.584 bits, top‑1 ‘Berlin’ [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:42]
- Semantic layer: bolding confirmed layer L 36 (confirmed_source=raw); preferred lens for reporting is tuned, but `use_confirmed_semantics=true` (001_layers_baseline/run-latest/output-Qwen3-14B.json:9282; 001_layers_baseline/run-latest/output-Qwen3-14B.json:9256).
- Control margin: `first_control_margin_pos=0`, `max_control_margin=0.97415` (001_layers_baseline/run-latest/output-Qwen3-14B.json:8679).
- Entropy drift: entropy gap percentiles p25=4.206, p50=13.402, p75=13.586 bits (001_layers_baseline/run-latest/output-Qwen3-14B.json:9330).
- Normalizer effect snapshots (KL to final under norm‑temp): @25% L10=12.986; @50% L20=13.204; @75% L30=12.489 bits (001_layers_baseline/run-latest/output-Qwen3-14B.json:8007).

**4. Qualitative Findings**

4.1. Copy vs. semantics (Δ‑gap)
The early layers show no copy reflex under strict or soft detectors. Strict copy `L_copy_strict` is null at τ∈{0.70,0.95} and `stability: "none"` (001_layers_baseline/run-latest/output-Qwen3-14B.json:7158). No `copy_soft_k1@0.5` hit in layers 0–3 (e.g., all False in rows 2–5 of `…-pure-next-token.csv`). With `L_semantic_norm=36` and no copy milestone, Δ̂ cannot be computed (evaluation pack `delta_hat=null`) (001_layers_baseline/run-latest/output-Qwen3-14B.json:9292). Summary tag: no copy threshold stability issues beyond “none”.

4.2. Lens sanity: Raw‑vs‑Norm
Lens artifact risk is high: `lens_artifact_score_v2=0.7037` (tier=high) (001_layers_baseline/run-latest/output-Qwen3-14B.json:7336) with `js_divergence_p50=0.513` and `l1_prob_diff_p50=1.432` (001_layers_baseline/run-latest/output-Qwen3-14B.json:7230; 001_layers_baseline/run-latest/output-Qwen3-14B.json:7235). Neither `first_js_le_0.1` nor `first_l1_le_0.5` occurs (both zero) (001_layers_baseline/run-latest/output-Qwen3-14B.json:7238; 001_layers_baseline/run-latest/output-Qwen3-14B.json:7239). Top‑K overlap is low (`jaccard_raw_norm_p50=0.25`, no layer ≥0.5) (001_layers_baseline/run-latest/output-Qwen3-14B.json:7243; 001_layers_baseline/run-latest/output-Qwen3-14B.json:7244). Prevalence: `pct_layers_kl_ge_1.0=0.7561`, `n_norm_only_semantics_layers=0` (001_layers_baseline/run-latest/output-Qwen3-14B.json:7222; 001_layers_baseline/run-latest/output-Qwen3-14B.json:7224). Caution: prefer rank milestones and confirmed semantics when interpreting early signals.

4.3. Tuned‑Lens analysis
Guidance prefers tuned for reporting (`preferred_lens_for_reporting="tuned"`), but semantics are confirmed at L 36 from the raw lens (001_layers_baseline/run-latest/output-Qwen3-14B.json:9256; 001_layers_baseline/run-latest/output-Qwen3-14B.json:8283). Attribution indicates rotation drives most KL reductions: ΔKL_rot p25=1.466, p50=1.689, p75=1.772, while temperature contributes slightly negative to neutral (ΔKL_temp p50≈−0.026); interaction p50≈2.725 (001_layers_baseline/run-latest/output-Qwen3-14B.json:9295; 001_layers_baseline/run-latest/output-Qwen3-14B.json:9258). Rank earliness shifts slightly later under tuned (baseline le_1=36 vs tuned le_1=39; le_5: 33→34; le_10: 32→33) (001_layers_baseline/run-latest/output-Qwen3-14B.json:9179). Positional generalization shows mild OOD drop (`pos_in_dist_le_0.92`=5.626 vs `pos_ood_ge_0.96`=4.221; gap −1.404) (001_layers_baseline/run-latest/output-Qwen3-14B.json:9347). Final‑head agreement is perfect: `kl_to_final_bits=0.0`, `top1_agree=true`, no warnings (001_layers_baseline/run-latest/output-Qwen3-14B.json:7992).

4.4. KL, ranks, cosine, entropy milestones
- KL: `first_kl_le_1.0` at L 40 for both baseline and tuned (001_layers_baseline/run-latest/output-Qwen3-14B.json:9198); final KL≈0 (001_layers_baseline/run-latest/output-Qwen3-14B.json:7992).
- Ranks (preferred lens=tuned; baseline in parens): le_10=33 (32), le_5=34 (33), le_1=39 (36) (001_layers_baseline/run-latest/output-Qwen3-14B.json:9179).
- Cosine (norm lens): ge_0.2 at L 5; ge_0.4 at L 29; ge_0.6 at L 36 (001_layers_baseline/run-latest/output-Qwen3-14B.json:7189).
- Entropy: large early‑layer gap (p50≈13.40 bits) narrowing to near‑final calibration; entropy dips sharply by L 36 where semantics are confirmed (001_layers_baseline/run-latest/output-Qwen3-14B.json:9330; 001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:38).

4.5. Prism (shared‑decoder diagnostic)
Prism is present and compatible (k=512) (001_layers_baseline/run-latest/output-Qwen3-14B.json:819). The sidecar CSVs are available (`…-pure-next-token-prism.csv`, `…-records-prism.csv`). No explicit KL delta vs. baseline milestones are summarized in JSON; treating Prism as diagnostic‑only per guidance. Verdict: Neutral (no qualitative milestone shift reported).

4.6. Ablation & stress tests
No‑filler ablation leaves semantics unchanged: `L_sem_orig=36`, `L_sem_nf=36` (ΔL_sem=0) (001_layers_baseline/run-latest/output-Qwen3-14B.json:8647). Control summary present with strong separation (`max_control_margin≈0.974`, first at pos 0) (001_layers_baseline/run-latest/output-Qwen3-14B.json:8679). Important‑word trajectory: the prompt token “Germany” is traced across depth (e.g., `pos,orig,0,12,Germany,…` at 001_layers_baseline/run-latest/output-Qwen3-14B-records.csv:14). The correct target “Berlin” emerges as top‑1 by L 36 on the next‑token stream (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:38).

4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓
- LayerNorm bias removed ✓
- FP32 unembed promoted ✓ (`unembed_dtype: torch.float32`)
- Punctuation / markup anchoring noted ✓ (final head shows quotes/punct among top‑k)
- Copy‑reflex ✗ (no strict/soft hits)
- Preferred lens honored ✓ (tuned for ranks; raw for confirmed L_semantics)
- Confirmed semantics reported ✓ (L 36, source=raw)
- Dual‑lens artifact metrics cited ✓ (v2 + JS/Jaccard/L1)
- Tuned‑lens audit done (rotation/temp/positional/head) ✓
- normalization_provenance present (ln_source @ L0/final) ✓
- per‑layer normalizer effect present (resid_norm_ratio, delta_resid_cos) ✓
- deterministic_algorithms true ✓
- numeric_health clean ✓
- copy_mask plausible ✓
- evaluation_pack citations used ✓

---
Produced by OpenAI GPT-5 
