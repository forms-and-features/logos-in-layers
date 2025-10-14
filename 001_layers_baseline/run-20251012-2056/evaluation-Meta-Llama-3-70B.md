# Evaluation Report: meta-llama/Meta-Llama-3-70B

## EVAL

**1. Overview**

This report evaluates meta-llama/Meta-Llama-3-70B (single-fact probe) from the 2025-10-12 20:56 run. The probe measures copy vs semantic onset using rank and KL milestones, cosine and entropy trajectories, plus lens diagnostics (raw vs norm, prism) and normalizer provenance.

**2. Method sanity-check**

- Prompt & indexing: context prompt ends with “called simply” and no trailing space; positive/original rows exist. “context_prompt: … is called simply” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:4]. “prompt_id: "pos"; prompt_variant: "orig"” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8690–8691].
- Normalizer provenance: strategy “next_ln1” (pre-norm). First/last sources: “ln_source: blocks[0].ln1” and “ln_source: ln_final” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7409, first/last entries printed programmatically].
- Per-layer normalizer effect: normalization spikes flagged by guidance (“reasons: … normalization_spike”) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9554–9559]. No additional pre-L_semantic spike exceptions required beyond this flag.
- Unembed bias: absent; “unembed_bias: {"present": false, "l2_norm": 0.0}” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:910–913 printed programmatically]. Cosines are bias-free.
- Environment & determinism: cpu; torch 2.8.0; deterministic_algorithms true; seed 316. “provenance.env … 'device': 'cpu', 'torch_version': '2.8.0+cu128', 'deterministic_algorithms': True, 'seed': 316” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json provenance.env printed programmatically].
- Numeric health: clean; “any_nan: false, any_inf: false, layers_flagged: []” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8632–8639].
- Copy mask: size 6022; sample token ids start [0, 1, 2, …]; plausible for tokenizer [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json copy_mask printed programmatically].
- Gold alignment: ok; “variant: with_space, first_id: 20437, pieces: ['ĠBerlin']” and rate 1.0 [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9407, gold_alignment_rate nearby].
- Repeatability (1.39): skipped due to deterministic env; “repeatability: {'status': 'skipped', 'reason': 'deterministic_env'}” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8639].
- Norm trajectory: “shape: spike, slope ≈0.05, r2 ≈0.94, n_spikes: 15” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8992–9002].
- Measurement guidance: prefer_ranks=true; suppress_abs_probs=true; preferred_lens_for_reporting=norm; use_confirmed_semantics=true [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9554–9566].
- Semantic margin: “delta_abs: 0.002, p_uniform: 7.8e-06, margin_ok_at_L_semantic_norm: false; L_semantic_margin_ok_norm: 80” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7159–7165].
- Micro-suite: present with aggregates; n=5, n_missing=0; “L_semantic_confirmed_median: 40” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9639–9646 and evaluation_pack.micro_suite aggregates printed programmatically].

**3. Quantitative findings (layer-by-layer)**

Short table (positive/original prompts; norm lens; ranks not probabilities):

- L 0 — entropy 16.97 bits; top‑1 token “winding”; answer_rank 115765 [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:2].
- L 38 — entropy 16.93 bits; top‑1 token “simply”; answer_rank 3 [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:40].
- L 40 — entropy 16.94 bits; top‑1 token “ Berlin”; answer_rank 1 (semantic onset) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:42].
- L 80 — entropy 2.59 bits; top‑1 token “ Berlin”; answer_rank 1 (final calibration) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:82].

Semantic layer: Bold confirmed L 40 (preferred lens=norm; confirmed source=raw). “L_semantic_confirmed: 40; source: raw” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9577–9591]. Note the uniform‑margin gate fails at L 40 (“margin_ok_at_L_semantic_norm: false”) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9455–9460].

Control margin: first_control_margin_pos=0; max_control_margin recorded (value suppressed per guidance) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9431].

Micro‑suite: median L_semantic_confirmed = 40 across 5 facts (n_missing=0) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json evaluation_pack.micro_suite aggregates printed programmatically]. Concrete citation: “France→Paris at L 80 answer_rank=1” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:250].

Entropy drift: median (entropy_bits − teacher_entropy_bits) ≈ 14.34 bits (p50), with p25≈14.33, p75≈14.35 [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9618–9621]. Early layers high entropy, dropping sharply by the final layer.

Normalizer effect snapshot: at L 38, resid_norm_ratio≈1.43, delta_resid_cos≈0.968 [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:40]. At L 80, resid_norm_ratio≈1.25, delta_resid_cos≈0.981 [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:82].

Confidence margins: answer_rank=1 at L 40 with small answer_vs_top1_gap thereafter; final‑head KL ≈ 0.0007 bits indicates tight calibration [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8656–8673].

**4. Qualitative findings**

4.1. Copy vs semantics (Δ‑gap)

No early copy-reflex detected: strict copy is null at τ∈{0.70,0.95} and soft copy k∈{1,2,3} also absent in layers 0–3 (CSV shows copy flags False) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:2–6]. The earliest strict copy L_copy_strict is null for all τ; stability tag “none”; norm_only_flags are null [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7102]. Semantic onset occurs at L 40 (confirmed). Δ̂ not reported (copy layer missing) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9577–9591].

4.2. Lens sanity: Raw‑vs‑Norm

Artifact score tier is medium; lens_artifact_score_v2≈0.344 [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7210–7217]. Symmetric metrics are good: js_divergence_p50≈0.00245 and l1_prob_diff_p50≈0.092; first_js_le_0.1=0 and first_l1_le_0.5=0 (i.e., near‑final agreement only) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7217–7230]. Top‑K overlap is strong mid‑to‑late (Jaccard@50 p50≈0.515; first ≥0.5 at L 11) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7230–7233]. Prevalence is low: pct_layers_kl_ge_1.0≈0.012; there are 2 norm‑only semantics layers with earliest at L 79 [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7211–7215]. Caution: norm‑only appears near the end (L 79–80) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7178–7207], far from L 40, so early semantics are unlikely lens‑induced; prefer ranks (per guidance) regardless.

4.3. Tuned‑Lens analysis

Not present for this run (no tuned sidecars or audit in JSON; only a path placeholder) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9438–9443]. Prefer norm lens for semantics per measurement_guidance.

4.4. KL, ranks, cosine, entropy milestones

KL milestones: first_kl_below_1.0=80; first_kl_below_0.5=80; final KL≈0 (0.0007 bits) with last‑layer agreement [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7043–7045, 8656–8673]. Rank milestones (norm lens): first_rank_le_10=38, first_rank_le_5=38, first_rank_le_1=40 [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7045–7047]. Cosine milestones (norm): ge_0.2/0.4/0.6 all at L 80 [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7140–7148]. Entropy declines late and tracks KL/rank consolidation (median entropy gap ≈14.34 bits) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9618–9621]. Margin gate reminder: uniform‑margin gate fails at L 40 (“margin_ok_at_L_semantic_norm: false”) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9455–9460].

4.5. Prism (shared‑decoder diagnostic)

Present and compatible (k=512, layers [embed,19,39,59]) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:844–858]. KL deltas show worse agreement vs baseline (p50 delta ≈ −1.00 bits), and no earlier ranks (le_1 unchanged) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:875–892]. Verdict: Regressive.

4.6. Ablation & stress tests

No‑filler ablation: L_sem_orig=40 → L_sem_nf=42 (ΔL_sem=+2; ~2.5% of depth) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9397–9407]. Controls: “control_summary … first_control_margin_pos: 0; first_control_strong_pos: 80” (max_control_margin recorded; value suppressed per guidance) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:9431]. Important‑word trajectory (records CSV): “Berlin” begins to appear in top‑K by mid‑layers (e.g., L 31, 33, 34, 36) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-records.csv:680, 719, 738, 776].

4.7. Checklist (✓/✗/n.a.)

- RMS lens ✓
- LayerNorm bias removed ✓
- FP32 unembed promoted ✓
- Punctuation / markup anchoring noted ✓
- Copy‑reflex ✗
- Preferred lens honored ✓
- Confirmed semantics reported ✓
- Dual‑lens artefact metrics (incl. v2/JS/Jaccard/L1) cited ✓
- Tuned‑lens audit done (rotation/temp/positional/head) n.a.
- normalization_provenance present (ln_source @ L0/final) ✓
- per‑layer normalizer effect present (resid_norm_ratio, delta_resid_cos) ✓
- deterministic_algorithms true ✓
- numeric_health clean ✓
- copy_mask plausible ✓
- milestones.csv or evaluation_pack.citations used for quotes ✓

---
Produced by OpenAI GPT-5

*Run executed on: 2025-10-12 20:56:18*
