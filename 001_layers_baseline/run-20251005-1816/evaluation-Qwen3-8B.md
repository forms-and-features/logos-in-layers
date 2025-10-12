# Evaluation Report: Qwen/Qwen3-8B

*Run executed on: 2025-10-05 18:16:50*
## EVAL

**Overview**
- Model: Qwen3-8B; experiment start: 2025-10-05 18:16:50 (`001_layers_baseline/run-latest/timestamp-20251005-1816`).
- Probe measures copy vs. semantics onset across layers using norm/raw lenses; tracks KL-to-final, ranks, cosine, and entropy trajectories; includes lens diagnostics (artifact v2, JS/L1/Jaccard), repeatability, and tuned‑lens audit.

**Method Sanity‑Check**
- Prompt & indexing: context ends with “called simply” (no trailing space): "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply" [001_layers_baseline/run-latest/output-Qwen3-8B.json:4]. Positive rows exist: e.g., `pos,orig,0…` and `pos,orig,31…` [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:2,33].
- Normalizer provenance: `strategy = next_ln1`; first/last sources: `blocks[0].ln1` → `ln_final` [001_layers_baseline/run-latest/output-Qwen3-8B.json:7332,7658].
- Per‑layer normalizer effect: early values present but no special flags beyond global “spike” trajectory; e.g., L0 `resid_norm_ratio=0.4108`, `delta_resid_cos=0.9923`; L31 `resid_norm_ratio=0.2518`, `delta_resid_cos=0.7561` [001_layers_baseline/run-latest/output-Qwen3-8B.json:7335,7618]. Norm trajectory: `shape=spike, slope=0.1288, r2=0.9215, n_spikes=15` [001_layers_baseline/run-latest/output-Qwen3-8B.json:9219-9224].
- Unembed bias: `present=False`, `l2_norm=0.0`; unembed cast to fp32 (`unembed_dtype: torch.float32`, `mixed_precision_fix: casting_to_fp32_before_unembed`) [001_layers_baseline/run-latest/output-Qwen3-8B.json:826,809,815]. Cosines are bias‑free by design.
- Environment & determinism: device `cpu`, `dtype_compute=torch.float32`, `deterministic_algorithms=true`, `seed=316`, `torch_version=2.8.0+cu128` [001_layers_baseline/run-latest/output-Qwen3-8B.json:8553-8561]. Reproducibility OK.
- Numeric health: `any_nan=False`, `any_inf=False`, `layers_flagged=[]` [001_layers_baseline/run-latest/output-Qwen3-8B.json:7930-7940].
- Copy mask: `size=6112`; large ignored‑IDs list shown (tokenizer plausibility) [001_layers_baseline/run-latest/output-Qwen3-8B.json:7069-7076,938-1460].
- Gold alignment: `{ok:true, variant:with_space, pieces:["ĠBerlin"]}`, `gold_alignment_rate=1.0` [001_layers_baseline/run-latest/output-Qwen3-8B.json:7902-7912,7946].
- Repeatability (1.39): `max_rank_dev/p95_rank_dev/top1_flip_rate = null` (flag: skipped) [001_layers_baseline/run-latest/output-Qwen3-8B.json:9270-9277,9306].
- Norm trajectory: `shape=spike` with high fit (`r2=0.9215`) [001_layers_baseline/run-latest/output-Qwen3-8B.json:9219-9224].
- Measurement guidance: `{prefer_ranks:true, suppress_abs_probs:true, preferred_lens_for_reporting:tuned, use_confirmed_semantics:true}`; reasons: `high_lens_artifact_risk`, `high_lens_artifact_score`, `normalization_spike` [001_layers_baseline/run-latest/output-Qwen3-8B.json:9181-9193].

**Quantitative Findings (Layer‑by‑Layer)**
- Table (positive prompt, `prompt_id=pos`, `prompt_variant=orig`; pure next‑token lens):
  - L 0 — entropy 17.2128 bits, top‑1 ‘CLICK’ [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:2]
  - L 10 — entropy 17.0199 bits, top‑1 ‘(?)’ [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:12]
  - L 20 — entropy 16.6960 bits, top‑1 ‘_’ [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:22]
  - L 31 — entropy 0.4539 bits, top‑1 ‘Berlin’, answer_rank=1 [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:33]
  - L 36 — entropy 3.1226 bits, top‑1 ‘Berlin’ [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:38]
- Semantic layer: bold confirmed semantics at L 31 (confirmed_source=raw; L_semantic_norm=31) [001_layers_baseline/run-latest/output-Qwen3-8B-milestones.csv:3-4].
- Control margin: `first_control_margin_pos=1`, `max_control_margin≈1.0` [001_layers_baseline/run-latest/output-Qwen3-8B.json:8618-8624].
- Entropy drift (vs teacher, bits): p25=1.663, p50=13.788, p75=14.006 [001_layers_baseline/run-latest/output-Qwen3-8B.json:9241-9248]. Representative: sharp drop at L31 to 0.454 bits [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:33].
- Confidence margins and normalizer snapshot at L31: `answer_logit_gap=0.1074`, `answer_vs_top1_gap=0.0959`, `resid_norm_ratio=0.2518`, `delta_resid_cos=0.7561` [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:33; 001_layers_baseline/run-latest/output-Qwen3-8B.json:7618].

**Qualitative Findings**

4.1. Copy vs semantics (Δ‑gap)
There is no early copy reflex: strict copy is null across τ∈{0.70,0.80,0.90,0.95} and soft‐k∈{1,2,3} also null (stability: none) [001_layers_baseline/run-latest/output-Qwen3-8B.json:8668-8708,8888-8928]. The semantic onset is late and decisive at L31 with rank‑1 ‘Berlin’ [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:33]. Δ̂ is not defined (no copy milestone) [001_layers_baseline/run-latest/output-Qwen3-8B.json:9205-9213].

4.2. Lens sanity: Raw‑vs‑Norm
Artifact risk is high: `lens_artifact_score_v2=0.704 (tier=high)` [001_layers_baseline/run-latest/output-Qwen3-8B-artifact-audit.csv:2]. Symmetric/robust metrics show substantial raw–norm drift: `js_divergence_p50=0.358`, `l1_prob_diff_p50=1.134`, with no early convergence by JS≤0.1 or L1≤0.5 (`first_js_le_0.1=0`, `first_l1_le_0.5=0`) [001_layers_baseline/run-latest/output-Qwen3-8B.json:9199-9236]. Top‑K overlap is low: `jaccard_raw_norm_p50=0.282`, `first_jaccard_raw_norm_ge_0.5=0` [001_layers_baseline/run-latest/output-Qwen3-8B.json:7262-7320]. Prevalence: `pct_layers_kl_ge_1.0=0.757`, `n_norm_only_semantics_layers=0`, `earliest_norm_only_semantic=null` [001_layers_baseline/run-latest/output-Qwen3-8B.json:7212-7256]. Caution: treat early “semantics” as potentially lens‑induced; rely on rank milestones and confirmed semantics.

4.3. Tuned‑Lens analysis
Preference: tuned lens is not “calibration‑only” and is the preferred semantics lens; confirmed semantics is used (`preferred_semantics_lens_hint=tuned`, `tuned_is_calibration_only=false`, `use_confirmed_semantics=true`) [001_layers_baseline/run-latest/output-Qwen3-8B.json:9169-9193,9100-9115]. Attribution: ΔKL components (tuned − norm) indicate large rotation effects (p25=0.764, p50=0.878, p75=1.000 bits) with small temperature effects (p50≈0.008) and notable interaction (ΔKL_interaction_p50=2.712) [001_layers_baseline/run-latest/output-Qwen3-8B.json:9249-9271]. Rank earliness shifts later under tuned (preferred lens): baseline first_rank_le_1=31 vs tuned=34; le_5: 29→31; le_10: 29→30 [001_layers_baseline/run-latest/output-Qwen3-8B.json:9069-9096]. Positional: `pos_in_dist_le_0.92=4.51`, `pos_ood_ge_0.96=3.63`, `pos_ood_gap=-0.882` [001_layers_baseline/run-latest/output-Qwen3-8B.json:9040-9056]. Head mismatch: final‑layer agreement is clean (`kl_bits_tuned_final=0.0` → after τ⋆ still 0.0; `tau_star_modelcal=1.0`) [001_layers_baseline/run-latest/output-Qwen3-8B.json:9056-9066]. Last‑layer consistency confirms no final‑head calibration issues (`kl_to_final_bits=0.0`, `top1_agree=true`) [001_layers_baseline/run-latest/output-Qwen3-8B.json:7914-7930].

4.4. KL, ranks, cosine, entropy milestones
KL: first KL≤1.0 occurs at L36 for both baseline and tuned (Δ=0) [001_layers_baseline/run-latest/output-Qwen3-8B.json:9171-9179]. Ranks (preferred lens=tuned; baseline in parentheses): first_rank_le_10=30 (29), le_5=31 (29), le_1=34 (31) [001_layers_baseline/run-latest/output-Qwen3-8B.json:9069-9096]. Cosine: tuned cos milestones reach ≥0.2/0.4/0.6 only at L36 [001_layers_baseline/run-latest/output-Qwen3-8B.json:8718-8726,8821-8829]. Entropy: large positive drift vs teacher across the bulk of layers (p50≈13.79 bits) with sharp collapse at L31 (0.454 bits) [001_layers_baseline/run-latest/output-Qwen3-8B.json:9241-9248; 001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:33].

4.5. Prism (shared‑decoder diagnostic)
Present (`output-Qwen3-8B-pure-next-token-prism.csv`). At L31, Prism shows very high KL and poor rank relative to baseline norm: `kl_to_final_bits≈14.78`, `answer_rank≈27292` vs norm lens `kl_to_final_bits≈1.06`, `answer_rank=1` [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token-prism.csv:33; 001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:33]. Verdict: Regressive (higher KL and later/no rank‑1).

4.6. Ablation & stress tests
No‑filler ablation leaves semantics unchanged: `L_sem_orig=31`, `L_sem_nf=31`, `ΔL_sem=0` [001_layers_baseline/run-latest/output-Qwen3-8B.json:8573-8586]. Control prompt (“France… called simply”) aligned and healthy; control summary: `first_control_margin_pos=1`, `max_control_margin≈1.0` [001_layers_baseline/run-latest/output-Qwen3-8B.json:8588-8624]. Important‑word trajectory (records CSV): “Germany” leads just before semantics and “Berlin” becomes dominant by L31: e.g., L30 shows ‘Germany’ then ‘Berlin’ [row 594], L31 ‘Berlin’ top‑1 [row 610], L34 ‘Berlin’ stronger [row 686] in `001_layers_baseline/run-latest/output-Qwen3-8B-records.csv`.

4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓ (RMSNorm; epsilon inside sqrt) [001_layers_baseline/run-latest/output-Qwen3-8B.json:810-813,7335-7659]
- LayerNorm bias removed ✓ (`layernorm_bias_fix: not_needed_rms_model`) [001_layers_baseline/run-latest/output-Qwen3-8B.json:812]
- FP32 unembed promoted ✓ (`unembed_dtype: torch.float32`; cast before unembed) [001_layers_baseline/run-latest/output-Qwen3-8B.json:809,815]
- Punctuation / markup anchoring noted ✓ (quotes/underscores dominate pre‑semantics) [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:2,12,22]
- Copy‑reflex ✗ (no strict/soft copy milestones) [001_layers_baseline/run-latest/output-Qwen3-8B.json:8668-8708,8888-8928]
- Preferred lens honored ✓ (reporting under tuned; confirmed semantics used) [001_layers_baseline/run-latest/output-Qwen3-8B.json:9181-9193,9200-9213]
- Confirmed semantics reported ✓ (L31, source=raw) [001_layers_baseline/run-latest/output-Qwen3-8B-milestones.csv:3-4]
- Dual‑lens artifact metrics cited ✓ (v2, JS, L1, Jaccard, prevalence) [001_layers_baseline/run-latest/output-Qwen3-8B.json:7212-7320,9199-9236]
- Tuned‑lens audit done ✓ (rotation/temp/positional/head) [001_layers_baseline/run-latest/output-Qwen3-8B.json:9040-9066,9249-9271]
- normalization_provenance present ✓ (ln_source @ L0/final) [001_layers_baseline/run-latest/output-Qwen3-8B.json:7335,7658]
- per‑layer normalizer effect present ✓ (`resid_norm_ratio`, `delta_resid_cos`) [001_layers_baseline/run-latest/output-Qwen3-8B.json:7335-7760]
- deterministic_algorithms true ✓ [001_layers_baseline/run-latest/output-Qwen3-8B.json:8557]
- numeric_health clean ✓ [001_layers_baseline/run-latest/output-Qwen3-8B.json:7930-7940]
- copy_mask plausible ✓ (`size=6112`) [001_layers_baseline/run-latest/output-Qwen3-8B.json:7069-7076]
- milestones.csv / evaluation_pack.citations used ✓ [001_layers_baseline/run-latest/output-Qwen3-8B-milestones.csv:3-4; 001_layers_baseline/run-latest/output-Qwen3-8B.json:9298-9310]

---
Produced by OpenAI GPT-5

