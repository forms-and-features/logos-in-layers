# Evaluation Report: meta-llama/Meta-Llama-3-8B

*Run executed on: 2025-10-11 21:50:12*
**EVAL**

**Overview**
- Model: Meta-Llama-3-8B; run date: 2025-10-11 21:50 (timestamp-20251011-2150).
- Probe measures copy vs. semantic onset via ranks/KL/cosine/entropy across layers; includes raw-vs-norm lens diagnostics and a tuned-lens audit.

**Method Sanity-check**
- Prompt & indexing: context ends with “called simply” and no trailing space: “Give the city name only, plain text. The capital of Germany is called simply” (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:819). Positive rows exist, e.g., `prompt_id=pos`, `prompt_variant=orig` at layer 0 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:2).
- Normalizer provenance: strategy "next_ln1" (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7248); per-layer ln source L0=blocks[0].ln1 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7251) and final uses ln_final at layer 32 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7540).
- Per-layer normalizer effect: early spikes flagged; L0 resid_norm_ratio=18.19, delta_resid_cos=0.535 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7255–7256); L25 resid_norm_ratio=1.50, delta_resid_cos=0.922 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7480–7481). `normalization_spike=true` (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:839).
- Unembed bias: present=false; l2_norm=0.0 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:831–833). Cosines are bias-free by construction.
- Environment & determinism: device=cpu; torch=2.8.0+cu128; deterministic_algorithms=true; seed=316 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:9843,9846,9848,9850).
- Numeric health: any_nan=false; any_inf=false; layers_flagged=[] (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7750–7754).
- Copy mask: `ignored_token_ids` include control/punctuation (e.g., 0–14, 25–31) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:940–952). Plausible for tokenizer.
- Gold alignment: ok; variant=with_space; first_id=20437 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7761–7764). gold_alignment_rate=1.0 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7804).
- Repeatability: skipped due to deterministic_env; metrics null (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7756–7757,11612–11615).
- Norm trajectory: shape="spike"; slope=0.113; r2=0.953; n_spikes=18 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11621–11625).
- Measurement guidance: prefer_ranks=true; suppress_abs_probs=true; preferred_lens_for_reporting=tuned; use_confirmed_semantics=true; reasons include high_lens_artifact_risk and normalization_spike (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11560–11572).
- Semantic margin: δ_abs=0.002; p_uniform≈7.8e-06; margin_ok_at_L_semantic_norm=false; L_semantic_margin_ok_norm=32 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7106–7110).
- Micro-suite: evaluation_pack.micro_suite present; n=5, n_missing=0; L_semantic_confirmed_median=25 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11679–11688,11727–11728).

**Quantitative Findings (layer-by-layer)**
- L 0 — entropy 16.96 bits; top‑1 “itzer”; copy flags false (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:2).
- L 20 — entropy 16.83 bits; top‑1 “'” (quote) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:24).
- L 25 — entropy 16.81 bits; top‑1 “ Berlin”; answer_rank=1 (confirmed) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:29).
- L 32 — entropy 2.96 bits; top‑1 “ Berlin”; answer_rank=1 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:36).
- Semantic layer: confirmed at L 25 (preferred lens=tuned, source=raw for confirmation) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11586–11591). Note: margin gate fails at L_semantic_norm (see Method; treat norm milestone as weak).
- Control margin: first_control_margin_pos=0; max_control_margin=0.5186 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:9899–9900).
- Micro‑suite: median L_semantic_confirmed=25; Δ̂ median n/a; e.g., France→Paris cited at row 90 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11694–11696).
- Entropy drift: gap medians (teacher→model) p25=13.87, p50=13.88, p75=13.91 bits (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11627–11630).
- Normalizer snapshots: resid_norm_ratio drops from 18.19 (L0) to 1.50 (L25) while delta_resid_cos rises to 0.922 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7255–7256,7480–7481).

**Qualitative Findings**

4.1. Copy vs semantics (Δ‑gap)
- Copy‑reflex ✗: no strict or soft copy milestones at τ∈{0.70,0.80,0.90,0.95}; all null; stability="none" (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7051–7069).
- Earliest strict copy at τ=0.70 and τ=0.95: null (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7062–7069). Norm‑only flags: null across τ (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7069–7076).
- Δ̂ = (L_sem − L_copy)/n_layers not defined (no copy). Milestone fraction for semantics ≈0.781 (evaluation_pack) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11591–11599).

4.2. Lens sanity: Raw‑vs‑Norm
- Artifact tier: lens_artifact_score_v2=0.459 (medium risk) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11599). Window summary flags risk="high" (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:9829).
- Symmetric/robust metrics: js_divergence_p50=0.0168; l1_prob_diff_p50=0.2403; first_js_le_0.1=0; first_l1_le_0.5=0 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11599–11605).
- Top‑K overlap: jaccard_raw_norm_p50=0.408; first_jaccard_raw_norm_ge_0.5 at layer 3 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11604–11605).
- Prevalence: pct_layers_kl_ge_1.0=0.0303; n_norm_only_semantics_layers=5; earliest_norm_only_semantic=25 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11606–11608). Caution: norm‑only layers near semantics → prefer rank milestones and confirmed semantics.

4.3. Tuned‑Lens analysis
- Preference: tuned_is_calibration_only=false; preferred_semantics_lens_hint=tuned (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11437–11438).
- Attribution: ΔKL_rot_p25/p50/p75 ≈ 2.08/2.56/2.86; ΔKL_temp_p25/p50/p75 ≈ −0.13/−0.04/+0.08; interaction_p50≈2.34 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11410–11417).
- Rank earliness: tuned first_rank_le_{10,5,1} = 32/32/32 vs baseline 24/25/25 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11368–11378). Tuned delays rank‑1 relative to norm baseline.
- Positional generalization: pos_ood_ge_0.96≈4.87; pos_in_dist_le_0.92≈5.64; gap≈−0.77 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11424–11433).
- Head mismatch: tau_star_modelcal=1.0; kl_bits_tuned_final=0.0 → 0.0 after τ* (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11433–11436). Last‑layer consistency: kl_to_final_bits=0.0; top1_agree=true; temp_est=1.0 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7772–7779).

4.4. KL, ranks, cosine, entropy milestones
- KL: first_kl_below_1.0 at layer 32 (baseline and tuned) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11401–11404). Final KL≈0 with top‑1 agreement (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7772–7774).
- Ranks: preferred lens=tuned has first_rank_le_{10,5,1}=32/32/32; baseline(norm)=24/25/25 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:11368–11378). Margin gate: L_semantic_norm is weak (margin_ok=false) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7109).
- Cosine: norm ge_{0.2,0.4,0.6} at layers 20, 30, 32 respectively (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:7096–7105).
- Entropy: strong late drop (gap medians ≳13.8 bits; see Quantitative). Early layers high‑entropy despite normalization spikes; calibration aligns at final layer with KL≈0.

4.5. Prism (shared‑decoder diagnostic)
- Present/compatible (k=512) (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:841–848). KL deltas are negative (higher KL) at p50 (≈+8.29 bits vs baseline), and no rank‑milestone improvement (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:872–892). Verdict: Regressive.

4.6. Ablation & stress tests
- Ablation (no_filler): L_sem_orig=25; L_sem_nf=25; ΔL_sem=0 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:9864–9871).
- Negative/control prompts: control_summary first_control_margin_pos=0; max_control_margin≈0.519 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:9899–9900). For the test prompt “Berlin is the capital of”, the expected country appears rank‑1 and “Berlin” also appears in top‑10 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:12–22).
- Important‑word trajectory (records CSV): prompt token “Germany” at pos=13 shown across layers (e.g., 001_layers_baseline/run-latest/output-Meta-Llama-3-8B-records.csv:13). Semantic onset for NEXT token confirmed at layer 25 “ Berlin” (001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:29).

4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓
- LayerNorm bias removed ✓
- FP32 unembed promoted ✓
- Punctuation / markup anchoring noted ✗
- Copy‑reflex ✗
- Preferred lens honored ✓
- Confirmed semantics reported ✓
- Dual‑lens artefact metrics cited ✓
- Tuned‑lens audit done ✓
- normalization_provenance present ✓
- per‑layer normalizer effect present ✓
- deterministic_algorithms true ✓
- numeric_health clean ✓
- copy_mask plausible ✓
- milestones.csv or evaluation_pack.citations used ✓

---
Produced by OpenAI GPT-5

