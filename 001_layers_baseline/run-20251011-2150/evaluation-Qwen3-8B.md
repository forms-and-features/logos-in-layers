# Evaluation Report: Qwen/Qwen3-8B

*Run executed on: 2025-10-11 21:50:12*
**EVAL**

**Overview**
- Model: Qwen3-8B. Probe measures copy-reflex vs. semantic emergence and tracks KL-to-final, rank milestones, cosine alignment, and entropy across layers with Raw/Norm/Tuned lens diagnostics (incl. artifact metrics). [001_layers_baseline/run-latest/output-Qwen3-8B.json:10030]
- High-level outputs include per-layer CSVs for the positive prompt and controls; evaluation_pack provides artifact, entropy, tuned-lens audit, and citations. [001_layers_baseline/run-latest/output-Qwen3-8B.json:11734]

**Method sanity-check**
- Prompt & indexing: Positive context ends with “called simply” (no trailing space) and targets Berlin; control prompt mirrors format for Paris. “Give the city name only, plain text. The capital of Germany is called simply” [001_layers_baseline/run-latest/output-Qwen3-8B.json:4]. Control: “Give the city name only, plain text. The capital of France is called simply” [001_layers_baseline/run-latest/output-Qwen3-8B.json:10029]. Positive rows exist with `prompt_id=pos, prompt_variant=orig` (e.g., layer 31) [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:33].
- Normalizer provenance: Strategy `next_ln1` for pre-norm; L0 uses `blocks[0].ln1` and final uses `ln_final`. “strategy: "next_ln1"” [001_layers_baseline/run-latest/output-Qwen3-8B.json:7336]. “ln_source: "blocks[0].ln1" … ln_source: "ln_final"” [001_layers_baseline/run-latest/output-Qwen3-8B.json:7348,7672].
- Per-layer normalizer effect: Early `resid_norm_ratio`/`delta_resid_cos` are moderate without pre-semantic spikes; a run-level flag notes a normalization spike overall. “delta_resid_cos: 0.9923 … 0.8263 …” [001_layers_baseline/run-latest/output-Qwen3-8B.json:7351,7361]. Flag: “"normalization_spike": true” [001_layers_baseline/run-latest/output-Qwen3-8B.json:834].
- Unembed bias: Not present; `l2_norm=0.0`. “"unembed_bias": { "present": false, "l2_norm": 0.0 }” [001_layers_baseline/run-latest/output-Qwen3-8B.json:831-838].
- Environment & determinism: `device: "cpu"`, torch 2.8.0, deterministic algorithms true, seed 316. “"deterministic_algorithms": true, "seed": 316” [001_layers_baseline/run-latest/output-Qwen3-8B.json:10004,10006].
- Numeric health: Clean; no NaN/Inf; no layers flagged. “"any_nan": false, "any_inf": false, "layers_flagged": []” [001_layers_baseline/run-latest/output-Qwen3-8B.json:7920-7960].
- Copy mask: Ignored token-id list present (size not explicitly reported). Sample: “[0, 1, 2, 3, 4, 5, 6, 7…]” [001_layers_baseline/run-latest/output-Qwen3-8B.json:944-952]. Appears plausible for tokenizer control tokens.
- Gold alignment: Gold answer and pieces match (“Berlin”, “ĠBerlin”); alignment ok. “"ok": true, … "first_id": 19846, "pieces": ["ĠBerlin"]” [001_layers_baseline/run-latest/output-Qwen3-8B.json:7932-7950]. Rate 1.0 [001_layers_baseline/run-latest/output-Qwen3-8B.json:7912-7960].
- Repeatability (1.39): Skipped due to deterministic env; no variance metrics. “"status": "skipped", "reason": "deterministic_env"” [001_layers_baseline/run-latest/output-Qwen3-8B.json:7912-7916].
- Norm trajectory: Shape “spike”; slope 0.129, r2 0.922, n_spikes 15. “"shape": "spike", … "r2": 0.9215, "n_spikes": 15” [001_layers_baseline/run-latest/output-Qwen3-8B.json:11780-11788].
- Measurement guidance: prefer ranks, suppress abs probs; preferred lens tuned; use confirmed semantics. “"prefer_ranks": true, "suppress_abs_probs": true, … "preferred_lens_for_reporting": "tuned", … "use_confirmed_semantics": true” [001_layers_baseline/run-latest/output-Qwen3-8B.json:11720-11728].
- Semantic margin: δ_abs=0.002, p_uniform≈6.6e-06; margin gate passes at L_semantic_norm=31 and confirmed margin-ok=31. “"margin_ok_at_L_semantic_norm": true, … "L_semantic_confirmed_margin_ok_norm": 31” [001_layers_baseline/run-latest/output-Qwen3-8B.json:7201-7204].
- Micro‑suite: Present with aggregates; n=5 facts, n_missing=0; median confirmed L=31, median Δ̂=0.0556. “"L_semantic_confirmed_median": 31, … "delta_hat_median": 0.0556, … "n": 5” [001_layers_baseline/run-latest/output-Qwen3-8B.json:11813-11824].

**Quantitative findings (layer‑by‑layer)**
- Preferred lens for reporting is tuned, but semantics are confirmed at L=31 (source=raw); we bold the confirmed layer per guidance. “"L_semantic_confirmed": { "layer": 31, "source": "raw" }” [001_layers_baseline/run-latest/output-Qwen3-8B.json:11748-11756].
- Control margin: first control-margin position=1; max control margin reported. “"first_control_margin_pos": 1, "max_control_margin": 0.9999977…” [001_layers_baseline/run-latest/output-Qwen3-8B.json:10054-10060].
- Micro‑suite robustness: median L_semantic_confirmed=31; median Δ̂=0.0556; no missing facts. Example citation (Germany→Berlin): “L_semantic_norm_row: 31, L_semantic_confirmed_row: 31” [001_layers_baseline/run-latest/output-Qwen3-8B.json:11794-11800].
- Entropy drift: teacher entropy ≈3.123 bits; entropy-gap medians large (p50≈13.79 bits across suite). “"teacher_entropy_bits": 3.1225837” [001_layers_baseline/run-latest/output-Qwen3-8B.json:7248-7254]. “"entropy_gap_bits_p50": 13.7880” [001_layers_baseline/run-latest/output-Qwen3-8B.json:11798-11806].

Short table (positive rows; norm lens values shown; ranks preferred; no probabilities quoted):
- L 0 — entropy 17.213 bits; top‑1 “CLICK”; KL_norm_temp≈12.786 bits; answer rank 14864. [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:1]
- L 9 — entropy 17.119 bits; top‑1 “(?)”; KL_norm_temp≈13.138 bits; answer rank 17918. [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:11]
- L 18 — entropy 16.911 bits; top‑1 “-”; KL_norm_temp≈12.490 bits; answer rank 7274. [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:20]
- L 27 — entropy 4.344 bits; top‑1 “____”; KL_norm_temp≈7.391 bits; answer rank 35. [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:29]
- L 31 — entropy 0.454 bits; top‑1 “Berlin”; KL_norm_temp≈2.728 bits; answer rank 1. [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:33]
- L 36 — entropy 3.123 bits; top‑1 “Berlin”; KL_norm_temp≈0.000 bits; answer rank 1. [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:38]

Bold semantic layer: L 31 (confirmed).

**Qualitative findings**

4.1. Copy vs semantics (Δ‑gap)
- No strict or soft copy detected at any τ in baseline or tuned summaries. “"L_copy_strict": {"0.7": null, … "0.95": null} … "stability": "none"” [001_layers_baseline/run-latest/output-Qwen3-8B.json:7144-7167,7168-7186]. With no copy milestone, Δ̂ is n/a (“delta_hat: null”) [001_layers_baseline/run-latest/output-Qwen3-8B.json:11752-11760].
- Strict copy earliest at τ=0.70 and τ=0.95: null; no norm‑only flags. “first strict copy null at all τ; norm_only_flags: null” [001_layers_baseline/run-latest/output-Qwen3-8B.json:7144-7186]. This suggests semantics emerge without a prior prompt‑echo reflex.

4.2. Lens sanity: Raw‑vs‑Norm
- Artifact tier: high. “"lens_artifact_score_v2": 0.704…, "risk_tier": "high"” [001_layers_baseline/run-latest/output-Qwen3-8B.json:11760-11768].
- Symmetric metrics: JS p50≈0.358; L1 p50≈1.134; no early low‑divergence crossings (first_js≤0.1=0; first_L1≤0.5=0). [001_layers_baseline/run-latest/output-Qwen3-8B.json:7224-7244].
- Top‑K overlap: Jaccard@50 p50≈0.282; first ≥0.5 = 0. [001_layers_baseline/run-latest/output-Qwen3-8B.json:7244-7276].
- Prevalence: pct layers KL≥1.0 ≈0.757; no norm‑only semantics layers; earliest null. [001_layers_baseline/run-latest/output-Qwen3-8B.json:7231-7240]. Caution: early semantics may be lens‑affected; rely on ranks and confirmed L.

4.3. Tuned‑Lens analysis
- Preference: Not calibration‑only; preferred semantics lens hint is tuned. “"tuned_is_calibration_only": false, "preferred_semantics_lens_hint": "tuned"” [001_layers_baseline/run-latest/output-Qwen3-8B.json:11802-11810]. We still anchor semantics at confirmed L=31.
- Attribution: Rotation drives most gains vs. temperature. “ΔKL_rot p25/p50/p75 ≈ 0.81/0.92/1.01; ΔKL_temp p25/p50/p75 ≈ 0.001/0.025/0.048; interaction p50≈2.816” [001_layers_baseline/run-latest/output-Qwen3-8B.json:11798-11810].
- Rank earliness: Norm first_rank_le_1=31 [001_layers_baseline/run-latest/output-Qwen3-8B.json:7087]; tuned first_rank_le_1 ranges 30–34 across summaries; for the base prompt it is 34 [001_layers_baseline/run-latest/output-Qwen3-8B.json:10070-10074]. Net effect: tuned does not make rank‑1 earlier for the base prompt.
- Positional generalization: pos_ood_ge_0.96≈3.77; in‑dist≤0.92≈5.00; ood gap negative (−1.228), indicating some degradation out of distribution. [001_layers_baseline/run-latest/output-Qwen3-8B.json:11798-11810].
- Head mismatch: Last‑layer agreement is exact with τ* = 1.0; KL after calibration 0.0 bits. “"kl_bits_tuned_final": 0.0 → after_tau_star: 0.0; "tau_star_modelcal": 1.0” [001_layers_baseline/run-latest/output-Qwen3-8B.json:11806-11812]. Baseline final‑head consistency is also perfect. “"kl_to_final_bits": 0.0, "top1_agree": true” [001_layers_baseline/run-latest/output-Qwen3-8B.json:7928-7956].

4.4. KL, ranks, cosine, entropy milestones
- KL: first_kl_below_1.0 = 36 and first_kl_below_0.5 = 36 (norm), indicating final‑head calibration arrives only at the end. [001_layers_baseline/run-latest/output-Qwen3-8B.json:7085-7086].
- Ranks: preferred lens (tuned) first_rank_le_10/5/1 = 30/31/34 for the base summary; baseline (norm) is 29/29/31. [001_layers_baseline/run-latest/output-Qwen3-8B.json:10070-10074,7087-7089]. Margin gate passes at L=31 (confirmed), so rank‑1 is trustworthy at that layer. [001_layers_baseline/run-latest/output-Qwen3-8B.json:7201-7204].
- Cosine: cos milestones (norm) all at 36 (≥0.2/0.4/0.6), suggesting cosine rises late. [001_layers_baseline/run-latest/output-Qwen3-8B.json:7182-7190].
- Entropy: strong early‑to‑mid entropy gaps relative to the teacher distribution (suite medians p50≈13.79 bits) [001_layers_baseline/run-latest/output-Qwen3-8B.json:11798-11806], dropping near L≈27–31 (e.g., L27 entropy 4.344; L31 0.454) [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:29,33]. This aligns with improving KL and ranks late rather than early calibration.

4.5. Prism (shared‑decoder diagnostic)
- Present and compatible. Layers sampled: embed/8/17/26. [001_layers_baseline/run-latest/output-Qwen3-8B.json:848-860].
- KL deltas at percentiles (prism−baseline) are negative (higher KL), notably p50≈−0.588 bits and p75≈−7.033 bits. “delta p50: -0.5880, p75: -7.0330” [001_layers_baseline/run-latest/output-Qwen3-8B.json:872-890]. Rank milestones show no improvement (null deltas). Verdict: Regressive.

4.6. Ablation & stress tests
- No‑filler ablation: L_sem_orig=31 and L_sem_nf=31 → ΔL_sem=0 (style‑robust). “"L_sem_orig": 31, "L_sem_nf": 31, "delta_L_sem": 0” [001_layers_baseline/run-latest/output-Qwen3-8B.json:10020-10028].
- Negative/control prompt: control summary present with strong control margin signal (see above). “"first_control_margin_pos": 1” [001_layers_baseline/run-latest/output-Qwen3-8B.json:10054-10060].
- Important‑word trajectory (records): At L31 on the prompt token “simply,” top‑1 is “Berlin” (rank‑1) [001_layers_baseline/run-latest/output-Qwen3-8B-records.csv:610]. At L32 an intermediate position shows “Berlin” again leading (consistent semantic consolidation) [001_layers_baseline/run-latest/output-Qwen3-8B-records.csv:628].

4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓; LayerNorm bias removed ✓; FP32 unembed promoted ✓ [001_layers_baseline/run-latest/output-Qwen3-8B.json:811,831-838].
- Punctuation/markup anchoring noted ✓ (control/negative prompts present) [001_layers_baseline/run-latest/output-Qwen3-8B.json:435-439,10020-10028].
- Copy‑reflex ✗ (none detected) [001_layers_baseline/run-latest/output-Qwen3-8B.json:7144-7186].
- Preferred lens honored ✓ (tuned used for audit; confirmed semantics reported) [001_layers_baseline/run-latest/output-Qwen3-8B.json:11720-11728,11748-11756].
- Confirmed semantics reported ✓ (L=31) [001_layers_baseline/run-latest/output-Qwen3-8B.json:11748-11756].
- Dual‑lens artifact metrics cited ✓ (v2 score, JS/L1/Jaccard; prevalence) [001_layers_baseline/run-latest/output-Qwen3-8B.json:7224-7276,11760-11768].
- Tuned‑lens audit done ✓ (rotation/temp/positional/head) [001_layers_baseline/run-latest/output-Qwen3-8B.json:11798-11812].
- normalization_provenance present ✓ (ln_source @ L0/final) [001_layers_baseline/run-latest/output-Qwen3-8B.json:7348,7672].
- per‑layer normalizer effect present ✓ (resid_norm_ratio, delta_resid_cos) [001_layers_baseline/run-latest/output-Qwen3-8B.json:7351-7361].
- deterministic_algorithms true ✓ (or caution) [001_layers_baseline/run-latest/output-Qwen3-8B.json:10004].
- numeric_health clean ✓ [001_layers_baseline/run-latest/output-Qwen3-8B.json:7920-7960].
- copy_mask plausible ✓ (token id list sample) [001_layers_baseline/run-latest/output-Qwen3-8B.json:944-952].
- milestones.csv or citations used ✓ (evaluation_pack.citations; pure CSV rows 29/33/38 etc.) [001_layers_baseline/run-latest/output-Qwen3-8B.json:11788-11806].

---
Produced by OpenAI GPT-5

