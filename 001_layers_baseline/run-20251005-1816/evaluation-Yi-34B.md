# Evaluation Report: 01-ai/Yi-34B

*Run executed on: 2025-10-05 18:16:50*
## EVAL — Yi-34B

**Overview**
- Model: 01-ai/Yi-34B (60 layers). Run date: 2025-10-05 (timestamp-20251005-1816).
- Probe measures copy reflex vs semantic onset with rank/KL/cosine/entropy trajectories and lens diagnostics (raw vs norm vs tuned), including artifact/consistency checks.

**Method sanity-check**
- Prompt & indexing: context ends with “called simply” and no trailing space: "Give the city name only, plain text. The capital of Germany is called simply"  001_layers_baseline/run-latest/output-Yi-34B.json:808. Positive rows present, e.g., `pos,orig,0` and `pos,orig,44` in pure CSV (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:2, 46).
- Normalizer provenance: strategy=`next_ln1` and pre-norm arch. L0 ln_source=`blocks[0].ln1`; final uses `ln_final`  001_layers_baseline/run-latest/output-Yi-34B.json:2814, 2830, 3388.
- Per-layer normalizer effect: early spikes are flagged (`normalization_spike: true`) and large at L0 (resid_norm_ratio≈6.14, delta_resid_cos≈0.57) well before semantics  001_layers_baseline/run-latest/output-Yi-34B.json:828, 2830.
- Unembed bias: present=false; l2_norm=0.0. Cosines bias-free  001_layers_baseline/run-latest/output-Yi-34B.json:821-829.
- Environment & determinism: device=`cpu`, compute=`torch.bfloat16`, `deterministic_algorithms=true`, seed=316  001_layers_baseline/run-latest/output-Yi-34B.json:4395-4412.
- Numeric health: any_nan=false, any_inf=false, layers_flagged=[]  001_layers_baseline/run-latest/output-Yi-34B.json:3733-3739.
- Copy mask: size=1513; e.g., id 97 appears in ignored_token_ids list  001_layers_baseline/run-latest/output-Yi-34B.json:2498-2506, 1160.
- Gold alignment: ok, variant=with_space, pieces=["▁Berlin"], gold_alignment_rate=1.0  001_layers_baseline/run-latest/output-Yi-34B.json:3744-3791, 3788.
- Repeatability (1.39): skipped due to deterministic env  001_layers_baseline/run-latest/output-Yi-34B.json:3739-3746; evaluation_pack.flag=skipped  001_layers_baseline/run-latest/output-Yi-34B.json:5100-5110.
- Norm trajectory: shape="spike", slope=0.074, r2=0.926, n_spikes=4  001_layers_baseline/run-latest/output-Yi-34B.json:4210-4220, 5110-5117.
- Measurement guidance: prefer_ranks=true, suppress_abs_probs=true; preferred_lens_for_reporting=tuned; use_confirmed_semantics=true  001_layers_baseline/run-latest/output-Yi-34B.json:5047-5064.

**Quantitative findings (layer-by-layer, pos/orig)**

| Layer | entropy_bits | top-1 token | answer_rank | is_answer | Notes |
|---|---|---|---:|:---:|---|
| 0 | 15.9623 | “Denote” | 34634 | False | high entropy; early noise  001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:2 |
| 30 | 15.5507 | “ODM” | 54886 | False | KL@50%≈12.26 bits  001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:32 |
| 43 | 15.5179 | top‑k includes “Berlin” | — | — | “Berlin” appears in top‑k for preceding token “simply”  001_layers_baseline/run-latest/output-Yi-34B-records.csv:770 |
| 44 | 15.3273 | “Berlin” | 1 | True | bold semantic layer (confirmed via tuned)  001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:46 |
| 60 | 2.9812 | “Berlin” | 1 | True | final‑head agreement; KL≈0.00028 bits  001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:63 |

- Semantic layer: L_semantic_norm=44 and L_semantic_confirmed=44 (confirmed_source=tuned)  001_layers_baseline/run-latest/output-Yi-34B.json:4200-4208, 5074-5079.
- Control margin: first_control_margin_pos=1; max_control_margin=0.5836  001_layers_baseline/run-latest/output-Yi-34B.json:4448-4456.
- Entropy drift (gap = entropy_bits − teacher_entropy_bits): p25=12.286, p50=12.586, p75=12.775  001_layers_baseline/run-latest/output-Yi-34B.json:5114-5117.
- Normalizer effect snapshots: L0 resid_norm_ratio=6.14, delta_resid_cos=0.57 (flagged); stabilizes by mid‑depth; no pre‑L_sem spikes beyond flagged  001_layers_baseline/run-latest/output-Yi-34B.json:2830, 3090-3160.

**Qualitative findings**

4.1. Copy vs semantics (Δ‑gap)
Yi‑34B shows no early copy reflex: strict copy L_copy_strict is null at τ∈{0.70,0.95} and soft windows k∈{1,2,3} are all null; stability="none"  001_layers_baseline/run-latest/output-Yi-34B.json:2564-2612. Thus L_copy is undefined. Semantic onset is at L=44, giving Δ̂ not available (evaluation_pack.depth_fractions.delta_hat=null)  001_layers_baseline/run-latest/output-Yi-34B.json:5069-5079. Copy thresholds carry no norm‑only flags at any τ  001_layers_baseline/run-latest/output-Yi-34B.json:2587-2602.

4.2. Lens sanity: Raw‑vs‑Norm
Lens artifact metrics are high: lens_artifact_score_v2=0.943 (tier=high), js_divergence_p50=0.3687, l1_prob_diff_p50=1.0891; first_js_le_0.1=0 and first_l1_le_0.5=0  001_layers_baseline/run-latest/output-Yi-34B.json:2796-2812, 2651-2662. Top‑K overlap is low (jaccard_raw_norm_p50=0.1111; first_jaccard_raw_norm_ge_0.5=1) and prevalence is high (pct_layers_kl_ge_1.0=0.656; n_norm_only_semantics_layers=14; earliest_norm_only_semantic=44)  001_layers_baseline/run-latest/output-Yi-34B.json:2669-2712. Caution: early semantics may be lens‑induced; prefer rank milestones and confirmed semantics near L≈44.

4.3. Tuned‑Lens analysis
- Preference: tuned lens loaded; tuned_is_calibration_only=false; preferred_semantics_lens_hint=tuned  001_layers_baseline/run-latest/output-Yi-34B.json:5018-5026, 4994-5007.
- Attribution: ΔKL_rot/temp @ p25/p50/p75 = {+2.82, +3.28, +3.50} and {−1.04, −0.66, +0.75}; interaction p50=+3.18; ΔKL_tuned_p50=+6.26 (drop vs norm)  001_layers_baseline/run-latest/output-Yi-34B.json:4986-5010.
- Rank earliness: first_rank_le_1 shifts later under tuned (46) vs norm (44); le_5 unchanged (44), le_10 +1 (44 vs 43)  001_layers_baseline/run-latest/output-Yi-34B.json:4951-4986.
- Positional generalization: pos_in_dist_le_0.92=5.66; pos_ood_ge_0.96=6.26; gap=0.603  001_layers_baseline/run-latest/output-Yi-34B.json:5008-5060.
- Head mismatch: tau_star_modelcal=1.0; tuned final KL≈0.00028 bits (unchanged after τ*)  001_layers_baseline/run-latest/output-Yi-34B.json:5056-5064. Last‑layer consistency: top1_agree=true; kl_to_final_bits=0.000278  001_layers_baseline/run-latest/output-Yi-34B.json:3756-3781.
- Baseline vs tuned semantics: L_semantic_norm=44; L_semantic_tuned=46; confirmed semantics at 44 from tuned audit (Δ_window=2)  001_layers_baseline/run-latest/output-Yi-34B.json:4188-4206.

4.4. KL, ranks, cosine, entropy milestones
- KL: first_kl_below_1.0=60; first_kl_below_0.5=60  001_layers_baseline/run-latest/output-Yi-34B.json:2488-2506. Final KL≈0 (0.000278 bits) with top‑1 agreement  001_layers_baseline/run-latest/output-Yi-34B.json:3756-3781.
- Ranks (preferred=tuned; baseline in parens): le_10=44 (43), le_5=44 (44), le_1=46 (44)  001_layers_baseline/run-latest/output-Yi-34B.json:4951-4986.
- Cosine milestones: norm lens ge_{0.2,0.4,0.6} at {1,44,51}; tuned lens ge_{0.2,0.4,0.6} all at 1  001_layers_baseline/run-latest/output-Yi-34B.json:2602-2612, 4561-4567.
- Entropy: large positive drift vs teacher (gap p50≈12.59 bits)  001_layers_baseline/run-latest/output-Yi-34B.json:5114-5117; aligns with late KL collapse and final calibration.

4.5. Prism (shared‑decoder diagnostic)
Prism present/compatible (k=512, sampled layers embed/14/29/44). Rank milestones unchanged (le_1/le_5/le_10 all null deltas); KL deltas are mixed: p25 −0.94, p50 −1.36, p75 +1.01 (sign indicates prism < norm)  001_layers_baseline/run-latest/output-Yi-34B.json:835-909. Verdict: Neutral (KL improvements ≈1 bit at median without earlier ranks).

4.6. Ablation & stress tests
- Filler ablation: L_sem_orig=44; L_sem_nf=44 (ΔL_sem=0)  001_layers_baseline/run-latest/output-Yi-34B.json:4414-4420.
- Control prompt: first_control_margin_pos=1; max_control_margin=0.5836  001_layers_baseline/run-latest/output-Yi-34B.json:4448-4456.
- Important‑word trajectory: Before semantics, “Berlin” appears in top‑k for tokens leading into the answer (e.g., layer 43, token “simply”: includes “Berlin”)  001_layers_baseline/run-latest/output-Yi-34B-records.csv:770; at L=44 for tokens “is/called/simply” the top‑1 is “Berlin”  001_layers_baseline/run-latest/output-Yi-34B-records.csv:786-788.

4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓
- LayerNorm bias removed ✓
- FP32 unembed promoted ✓
- Punctuation / markup anchoring noted ✓ (copy_mask present; prompt punctuation filtered)
- Copy‑reflex ✗ (no early strict/soft copy)
- Preferred lens honored ✓ (tuned)
- Confirmed semantics reported ✓ (L=44 from tuned audit)
- Dual‑lens artifact metrics cited ✓ (v2/JS/Jaccard/L1)
- Tuned‑lens audit done ✓ (rotation/temp/positional/head)
- normalization_provenance present ✓ (ln_source @ L0/final)
- per‑layer normalizer effect present ✓ (resid_norm_ratio, delta_resid_cos)
- deterministic_algorithms true ✓
- numeric_health clean ✓
- copy_mask plausible ✓ (size=1513)
- milestones/evaluation_pack citations used ✓

---
Produced by OpenAI GPT-5

