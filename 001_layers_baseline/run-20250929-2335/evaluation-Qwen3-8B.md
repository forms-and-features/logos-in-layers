1. Overview

Qwen/Qwen3-8B (36 layers). Probe analyzes next-token trajectories under a norm lens for the prompt “Give the city name only, plain text. The capital of Germany is called simply”, tracking rank, KL-to-final (bits), entropy (bits), copy flags, cosine-to-final, and surface-vs-meaning mass.

2. Method sanity‑check

Diagnostics confirm the intended norm lens and rotary/token-only layer‑0 view: “use_norm_lens: true” [001_layers_baseline/run-latest/output-Qwen3-8B.json:807] and “layer0_position_info: token_only_rotary_model” [001_layers_baseline/run-latest/output-Qwen3-8B.json:816]. Context prompt ends with “called simply” and no trailing space [001_layers_baseline/run-latest/output-Qwen3-8B.json:4,817]. Gold alignment is ok [001_layers_baseline/run-latest/output-Qwen3-8B.json:1062]. Copy detector configuration present: copy_thresh=0.95, copy_window_k=1, match_level="id_subsequence" [001_layers_baseline/run-latest/output-Qwen3-8B.json:932–934], and strict‑sweep metadata with stability=“none” [001_layers_baseline/run-latest/output-Qwen3-8B.json:980–1010]. Copy flag columns listed in JSON/CSVs [001_layers_baseline/run-latest/output-Qwen3-8B.json:1928]. Negative control prompt and summary present [001_layers_baseline/run-latest/output-Qwen3-8B.json:1605–1617]. Ablation summary exists with both variants; L_sem_orig=31, L_sem_nf=31 (ΔL_sem=0) and both prompt_variant rows appear in pure CSV (e.g., “pos,no_filler,0,14,⟨NEXT⟩,…”) [001_layers_baseline/run-latest/output-Qwen3-8B.json:1600–1604; 001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:39].

KL/entropy units are bits (columns: “entropy”, “kl_to_final_bits”, “teacher_entropy_bits”). Final‑head calibration is clean: last_layer_consistency.kl_to_final_bits=0.0; top1_agree=true; p_top1_lens=p_top1_model=0.4334; temp_est=1.0; kl_after_temp_bits=0.0 [001_layers_baseline/run-latest/output-Qwen3-8B.json:1063–1091].

Measurement guidance: prefer_ranks=true; suppress_abs_probs=true; reason: “high_lens_artifact_risk” [001_layers_baseline/run-latest/output-Qwen3-8B.json:2077–2081]. Raw‑vs‑Norm checks: window radius=4, center_layers=[29,29,31,36], max_kl_norm_vs_raw_bits_window=38.096 [001_layers_baseline/run-latest/output-Qwen3-8B.json:1037–1061]; sampled raw‑lens sanity: lens_artifact_risk=“high”, max_kl_norm_vs_raw_bits=13.605, first_norm_only_semantic_layer=null [001_layers_baseline/run-latest/output-Qwen3-8B.json:1525,1582–1584].

Summary indices (baseline norm lens): first_kl_below_0.5=36; first_kl_below_1.0=36; first_rank_le_1=31; first_rank_le_5=29; first_rank_le_10=29 [001_layers_baseline/run-latest/output-Qwen3-8B.json:935–939].

Copy‑collapse strict rule (τ=0.95, δ=0.10) did not fire; soft detectors (τ_soft=0.5, k∈{1,2,3}) also did not fire in L0–L3. Earliest strict at τ∈{0.70,0.95} is null in the threshold sweep [001_layers_baseline/run-latest/output-Qwen3-8B.json:980–1010].

Copy‑collapse flag check: no rows with copy_collapse=True in pos/orig for layers 0–36 (scan of pure CSV; first Strict/Soft flags are null). ✓ rule did not fire.

3. Quantitative findings

Gold answer: “Berlin” [001_layers_baseline/run-latest/output-Qwen3-8B.json:2090–2098]. Table below uses only pos/orig from the pure‑next‑token CSV.

| Layer | Entropy (bits) | Top-1 token |
|---|---:|---|
| L 0 | 17.213 | 'CLICK' |
| L 1 | 17.211 | 'apr' |
| L 2 | 17.211 | '财经' |
| L 3 | 17.208 | '-looking' |
| L 4 | 17.206 | '院子' |
| L 5 | 17.204 | ' (?)' |
| L 6 | 17.196 | 'ly' |
| L 7 | 17.146 | ' (?)' |
| L 8 | 17.132 | ' (?)' |
| L 9 | 17.119 | ' (?)' |
| L 10 | 17.020 | ' (?)' |
| L 11 | 17.128 | 'ifiable' |
| L 12 | 17.117 | 'ifiable' |
| L 13 | 17.126 | 'ifiable' |
| L 14 | 17.053 | '"' |
| L 15 | 17.036 | '"' |
| L 16 | 16.913 | '-' |
| L 17 | 16.972 | '-' |
| L 18 | 16.911 | '-' |
| L 19 | 16.629 | 'ly' |
| L 20 | 16.696 | '_' |
| L 21 | 16.408 | '_' |
| L 22 | 15.219 | ' ______' |
| L 23 | 15.220 | '____' |
| L 24 | 10.893 | '____' |
| L 25 | 13.454 | '____' |
| L 26 | 5.558 | '____' |
| L 27 | 4.344 | '____' |
| L 28 | 4.786 | '____' |
| L 29 | 1.778 | '-minded' |
| L 30 | 2.203 | ' Germany' |
| **L 31** | **0.454** | **' Berlin'** |
| L 32 | 1.037 | ' German' |
| L 33 | 0.988 | ' Berlin' |
| L 34 | 0.669 | ' Berlin' |
| L 35 | 2.494 | ' Berlin' |
| L 36 | 3.123 | ' Berlin' |

Control margin: first_control_margin_pos=1; max_control_margin=0.999997735 [001_layers_baseline/run-latest/output-Qwen3-8B.json:1619–1623].

Ablation (no‑filler): L_copy_orig=null, L_sem_orig=31; L_copy_nf=null, L_sem_nf=31; ΔL_copy=null, ΔL_sem=0 [001_layers_baseline/run-latest/output-Qwen3-8B.json:1600–1604].

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (no strict or soft copy layer). Soft ΔHₖ (k∈{1,2,3}) = n.a. (no soft hits).

Confidence milestones (within‑model): p_top1>0.30 at L29; p_top1>0.60 at L29; final‑layer p_top1=0.4334 (CSV last row; KL≈0) [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:2–3; 001_layers_baseline/run-latest/output-Qwen3-8B.json:1063–1071].

Rank milestones (diagnostics): rank≤10 at L29; rank≤5 at L29; rank≤1 at L31 [001_layers_baseline/run-latest/output-Qwen3-8B.json:935–939].

KL milestones: first_kl_below_1.0 at L36; first_kl_below_0.5 at L36; KL decreases with depth and is ≈0 at final (CSV last layer; diagnostics agree) [001_layers_baseline/run-latest/output-Qwen3-8B.json:935–939,1063–1071].

Cosine milestones: ge_0.2 at L36; ge_0.4 at L36; ge_0.6 at L36; final cos_to_final reported in CSV at L36 (use within‑model only) [001_layers_baseline/run-latest/output-Qwen3-8B.json:1001–1009].

Copy robustness (threshold sweep): stability=“none”; earliest strict L_copy at τ=0.70 and τ=0.95 are null (norm_only_flags also null) [001_layers_baseline/run-latest/output-Qwen3-8B.json:980–1010].

Prism Sidecar Analysis

Presence: compatible=true; artifacts loaded [001_layers_baseline/run-latest/output-Qwen3-8B.json:824–846]. Early‑depth stability at sampled layers (L≈0/9/18/27): KL_baseline vs KL_prism = 12.79 vs 12.94; 12.61 vs 12.98; 12.41 vs 13.00; 6.14 vs 13.18 (bits). Cosine: baseline −0.127/−0.273/−0.318/−0.262 vs prism −0.096/−0.086/−0.117/−0.195. Top‑1 agreement with final: none at sampled layers for both. Rank milestones (prism): no rank≤{10,5,1} achieved. Verdict: Regressive (KL increases notably; no earlier ranks).

Tuned‑Lens Comparison

ΔKL medians at depth percentiles (baseline − tuned): p25=4.1395, p50=4.0047, p75=1.3956 bits [001_layers_baseline/run-latest/output-Qwen3-8B.json:2009–2027]. Last‑layer agreement is clean (kl_after_temp_bits≈0) [001_layers_baseline/run-latest/output-Qwen3-8B.json:1063–1071]. Rank earliness (diagnostics metrics): baseline le_1=31 vs tuned le_1=34 (later); le_5: 29 vs 31; le_10: 29 vs 30 [001_layers_baseline/run-latest/output-Qwen3-8B.json:2013–2025]. Surface→meaning: L_surface_to_meaning_norm=31 with (answer_mass≈0.936, echo_mass≈0.011) [001_layers_baseline/run-latest/output-Qwen3-8B.json:1010–1019]; tuned reports similar L_surface_to_meaning_tuned=31 with small echo_mass [001_layers_baseline/run-latest/output-Qwen3-8B.json:1417–1421]. Geometry: L_geom_norm=34; L_geom_tuned=34 [001_layers_baseline/run-latest/output-Qwen3-8B.json:1012–1016,1421–1421]. Coverage: L_topk_decay_norm=0; L_topk_decay_tuned=1 [001_layers_baseline/run-latest/output-Qwen3-8B.json:1017–1019,1417–1419]. Norm‑temp snapshots present (kl_to_final_bits_norm_temp@{25,50,75}%) [001_layers_baseline/run-latest/output-Qwen3-8B.json:1083–1091]; tuned lens shows regression flags (skip_layers_sanity m=2=0.370; tuned_lens_regression=true) [001_layers_baseline/run-latest/output-Qwen3-8B.json:1422–1442]. Entropy drift (mid‑depth): entropy − teacher_entropy_bits ≈ +13.8 bits at L≈18 (expected earlier‑layer higher entropy; within‑model only).

4. Qualitative patterns & anomalies

Negative control shows correct mapping: “Berlin is the capital of → Germany (0.73), which (0.22), the (0.02), what (0.01), __ (0.002)” [001_layers_baseline/run-latest/output-Qwen3-8B.json:10–22,40–47]. Semantic leakage check: “Berlin” appears in the tail here with small mass in top‑10 [001_layers_baseline/run-latest/output-Qwen3-8B.json:40–47].

Important‑word trajectory in records.csv: starting around L28–L30, “simply/called/is/Germany” positions begin to admit “Berlin” into their top candidates, e.g., “simply … Berlin, 0.0036” (layer 28) [001_layers_baseline/run-latest/output-Qwen3-8B-records.csv:562], “called … Berlin, 0.095” (layer 30) [001_layers_baseline/run-latest/output-Qwen3-8B-records.csv:593], “is … Berlin, 0.076” (layer 30) [001_layers_baseline/run-latest/output-Qwen3-8B-records.csv:592], and “Germany … Berlin, 0.0052” (layer 30) [001_layers_baseline/run-latest/output-Qwen3-8B-records.csv:591]. This aligns with the pure NEXT transition to rank‑1 at L31 (table above).

Rest‑mass sanity: rest_mass generally decays into later layers with a maximum after L_semantic of 0.175 at layer 36 (top‑k coverage; not a fidelity metric) [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:2–3].

Rotation vs amplification: KL_to_final declines monotonically late, while answer_rank improves to 1 by L31 and cos_to_final crosses positive late (ge_0.2/0.4/0.6 only at L36). This pattern suggests “early direction, late calibration”: the answer direction emerges around L30–L31 but the distribution calibrates toward the final head only in the last few layers.

Head calibration: final‑head consistency is excellent (kl_to_final_bits=0.0; temp_est=1.0; no warnings) [001_layers_baseline/run-latest/output-Qwen3-8B.json:1063–1071].

Lens sanity: raw‑lens artifact risk is high (sampled max_kl_norm_vs_raw_bits=13.605; window max=38.096), with no norm‑only semantic layer flagged; treat any pre‑final “early semantics” cautiously and prefer ranks over absolute probabilities [001_layers_baseline/run-latest/output-Qwen3-8B.json:1037–1061,1582–1584]. Copy strict norm‑only flags are null across τ∈{0.70,0.80,0.90,0.95} in the threshold sweep [001_layers_baseline/run-latest/output-Qwen3-8B.json:980–1010].

Temperature robustness: at T=2.0, “Berlin” remains top‑1 but with spread and entropy rises (JSON temperature_exploration block) [001_layers_baseline/run-latest/output-Qwen3-8B.json:1200–1280].

Stylistic ablation: removing “simply” leaves L_sem unchanged (31); ΔL_sem=0 [001_layers_baseline/run-latest/output-Qwen3-8B.json:1600–1604].

Checklist: RMS lens ✓; LayerNorm bias removed n/a (RMSNorm) [001_layers_baseline/run-latest/output-Qwen3-8B.json:812–815]; Entropy rise at unembed — final entropy=3.123 bits [001_layers_baseline/run-latest/output-Qwen3-8B.json:1445–1460]; FP32 un‑embed promoted? use_fp32_unembed=false; unembed_dtype="torch.float32" (analysis W_U in fp32 by cast) [001_layers_baseline/run-latest/output-Qwen3-8B.json:809–813]; Punctuation/filler anchoring ✓ (early layers top‑1 punctuation/fillers in table); Copy‑reflex ✗ (no strict/soft hits in L0–L3).

5. Limitations & data quirks

High raw‑vs‑norm divergence (lens_artifact_risk=“high”) cautions that early “semantics” may be lens‑induced; prefer rank milestones and within‑model trends [001_layers_baseline/run-latest/output-Qwen3-8B.json:1582–1584]. Rest_mass remains >0.1 in late layers (max ~0.175 post‑semantic), indicating limited top‑k coverage rather than lens infidelity; do not use as a fidelity metric. KL is lens‑sensitive; final KL≈0 indicates good head calibration here, but cross‑model probability comparisons should rely on rank/thresholds per measurement_guidance [001_layers_baseline/run-latest/output-Qwen3-8B.json:2077–2081]. Raw‑lens window mode is “window” and “sample”, not exhaustive; treat as sampled sanity rather than proof [001_layers_baseline/run-latest/output-Qwen3-8B.json:1037–1061,1525]. Surface‑mass depends on tokenizer; avoid cross‑family absolute mass comparisons.

6. Model fingerprint

Qwen3‑8B: semantics at L 31; final entropy 3.123 bits; “Berlin” enters and stabilizes near L 31.

---
Produced by OpenAI GPT-5

