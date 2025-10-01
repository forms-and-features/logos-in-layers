# Evaluation Report: Qwen/Qwen3-14B

*Run executed on: 2025-09-30 23:57:21*
**1. Overview**

Qwen/Qwen3-14B (40 layers, pre_norm) is probed with a layer-by-layer norm lens to track copy vs. semantic collapse, calibration to the final head, geometry (cosine-to-final), and surface mass. We follow measurement guidance that prefers rank-based reporting and the tuned lens as the primary reference while still presenting baseline metrics for context. Confirmed semantics place the collapse at L 36 (source=raw), with tuned lens collapse later at L 39.

**2. Method Sanity‑Check**

Diagnostics confirm the norm lens was applied on a rotary (token-only) model with RMSNorm and FP32 unembed casting: “use_norm_lens: true … unembed_dtype: torch.float32 … first_block_ln1_type: "RMSNorm" … norm_alignment_fix: "using_ln2_rmsnorm_for_post_block" … layer0_position_info: "token_only_rotary_model"” [L807–L817]. The context_prompt matches (no trailing space): “Give the city name only, plain text. The capital of Germany is called simply” [L817].

Strict/soft copy configs and flags are present and aligned across JSON/CSV: “copy_thresh: 0.95 … copy_window_k: 1 … copy_match_level: "id_subsequence" … copy_soft_config: { threshold: 0.5, window_ks: [1,2,3] }” [L936–L938, L880–L888]; “copy_flag_columns: [ … copy_strict@{0.95,0.7,0.8,0.9}, copy_soft_k{1,2,3}@0.5 ]” [L1624–L1632]. Gold-token alignment is ok [L1078, L2150–L2160]. Negative control and summary present: “control_prompt … France … gold_alignment: ok” [L1641–L1654]; “control_summary: first_control_margin_pos: 0, max_control_margin: 0.974…” [L1656–L1659]. Ablation block present with both variants: “ablation_summary … L_sem_orig: 36 … L_sem_nf: 36 … delta_L_sem: 0” [L1633–L1640].

Rank/KL milestones (units in bits) from diagnostics: first_rank≤{10,5,1} = {32,33,36} [L926–L935], first_kl_below_{1.0,0.5} = {40,40} [L921–L925]. Final-head calibration is clean: “last_layer_consistency: kl_to_final_bits: 0.0 … top1_agree: true … p_top1_lens = p_top1_model … temp_est: 1.0 … kl_after_temp_bits: 0.0 … warn_high_last_layer_kl: false” [L1080–L1098]. Measurement guidance: “prefer_ranks: true … suppress_abs_probs: true … preferred_lens_for_reporting: "tuned" … use_confirmed_semantics: true” [L2138–L2148].

Raw‑vs‑Norm (window): radius=4, center_layers=[33,36,40], norm_only_semantics_layers=[], max_kl_norm_vs_raw_bits_window=98.58 bits, mode="window" [L1041–L1065]. Full dual‑lens summary: pct_layers_kl_ge_{1.0,0.5}={0.756,0.829}, n_norm_only_semantics_layers=0, earliest_norm_only_semantic=null, max_kl_norm_vs_raw_bits=98.58, tier=high [L1066–L1077]. Treat early semantics cautiously; prefer ranks and confirmed semantics. Threshold sweep: “summary.copy_thresholds.stability: none … L_copy_strict@τ ∈ {0.70,0.95} = null” [L988–L1011].

Copy-reflex check (layers 0–3): pure CSV has copy_collapse=False and soft k1@0.5=False in early layers; no copy-reflex detected. “L_copy: null; L_copy_soft k∈{1,2,3}: null” [L911–L925].

Raw‑vs‑Norm sample sanity: “raw_lens_check.summary: lens_artifact_risk: "high"; max_kl_norm_vs_raw_bits: 17.67 bits” [L1618–L1623].

**3. Quantitative Findings**

Table below uses baseline norm lens at NEXT for pos/orig rows only. Bold marks the confirmed semantic layer.

| Layer | Entropy (bits) | Top‑1 |
|---|---:|---|
| L 0 | 17.213 | '梳' |
| L 1 | 17.212 | '地处' |
| L 2 | 17.211 | '是一部' |
| L 3 | 17.210 | 'tics' |
| L 4 | 17.208 | 'tics' |
| L 5 | 17.207 | '-minded' |
| L 6 | 17.205 | '过去的' |
| L 7 | 17.186 | '�' |
| L 8 | 17.180 | '-minded' |
| L 9 | 17.188 | '-minded' |
| L 10 | 17.170 | '(?)' |
| L 11 | 17.151 | '时代的' |
| L 12 | 17.165 | 'といって' |
| L 13 | 17.115 | 'nav' |
| L 14 | 17.141 | 'nav' |
| L 15 | 17.149 | '唿' |
| L 16 | 17.135 | '闯' |
| L 17 | 17.137 | '唿' |
| L 18 | 17.101 | '____' |
| L 19 | 17.075 | '____' |
| L 20 | 16.932 | '____' |
| L 21 | 16.986 | '年夜' |
| L 22 | 16.954 | '年夜' |
| L 23 | 16.840 | '____' |
| L 24 | 16.760 | '____' |
| L 25 | 16.758 | '年夜' |
| L 26 | 16.669 | '____' |
| L 27 | 16.032 | '____' |
| L 28 | 15.234 | '____' |
| L 29 | 14.187 | '这个名字' |
| L 30 | 7.789 | '这个名字' |
| L 31 | 5.162 | '____' |
| L 32 | 0.816 | '____' |
| L 33 | 0.481 | '____' |
| L 34 | 0.595 | '____' |
| L 35 | 0.668 | '____' |
| **L 36** | 0.312 | 'Berlin' |
| L 37 | 0.906 | '____' |
| L 38 | 1.212 | '____' |
| L 39 | 0.952 | 'Berlin' |
| L 40 | 3.584 | 'Berlin' |

Control margin (JSON): first_control_margin_pos = 0, max_control_margin = 0.974 [L1656–L1659].

Ablation (no‑filler): L_copy_orig = null, L_sem_orig = 36; L_copy_nf = null, L_sem_nf = 36; ΔL_copy = null, ΔL_sem = 0 [L1633–L1640].

- ΔH = entropy(L_copy) − entropy(L_semantic): n.a. (strict copy null). Soft ΔHₖ: n.a. (soft k∈{1,2,3} null).
- Confidence milestones: p_top1 > 0.30 at L 31; p_top1 > 0.60 at L 32; final-layer p_top1 = 0.345 (CSV row L 40).
- Rank milestones (preferred lens=tuned; baseline in parentheses): rank ≤10: n/a (tuned) vs 32 (baseline) [L926–L935, L2090–L2137]; rank ≤5: n/a (tuned) vs 33 (baseline); rank ≤1: 39 (tuned) vs 36 (baseline).
- KL milestones: first_kl_below_1.0 at L 40; first_kl_below_0.5 at L 40; KL decreases with depth and ≈0 at final (last_layer_consistency.kl_to_final_bits=0.0) [L1080–L1087].
- Cosine milestones (norm): ge_0.2 at L 5; ge_0.4 at L 29; ge_0.6 at L 36; final cos_to_final ≈ 1.000 [L1026–L1031; CSV L 40].
- Normalized depths: L_semantic_frac = 0.90; first_rank_le_5_frac = 0.825 [L1033–L1040].
- Surface→meaning (norm): L_surface_to_meaning_norm = 36 with answer_mass=0.9530 and echo_mass≈0.0000044 [L1137–L1153]. Tuned: L_surface_to_meaning_tuned = 36 (first tuned summary) [L1211–L1231]; note tuned L_semantic=39.
- Geometry: L_geom_norm = 35 (cos_to_answer=0.081; cos_to_prompt_max=0.059) [L1154–L1161].
- Coverage: L_topk_decay_norm = 0; topk_prompt_mass@τ=0.33 at L = 0.0 [L1162–L1171].
- Norm temperature diagnostics: kl_to_final_bits_norm_temp@{25,50,75}% = {12.986, 13.204, 12.489} bits [L1099–L1110].

Copy‑robustness (threshold sweep): stability = “none”; earliest strict L_copy at τ=0.70 and 0.95 both null [L988–L1011]. No norm‑only strict copy flags in the window.

Prism Sidecar Analysis
- Presence: compatible=true; layers sampled = {embed,9,19,29} [L825–L836].
- Early-depth KL: baseline vs prism at L≈{0,9,19,29} → deltas ≈ {−0.21, −0.26, −0.25, −0.42} bits (prism higher; JSON p25/50/75 deltas negative) [L856–L871]. Final prism KL remains 14.84 bits (CSV L 40), while baseline final KL=0.
- Rank milestones: prism never reaches rank ≤10 (all null) [L845–L849].
- Top‑1 agreement: no prism fix-ups at sampled depths; prism top‑1 diverges and final is miscalibrated (KL≫0 at L 40).
- Cosine: prism cos_to_final lower than baseline at early/mid (no earlier stabilization; CSV L∈{0,9,19,29}).
- Copy flags: none fire spuriously under prism (no flip).
- Verdict: Regressive (higher KL, no rank improvements, misaligned final).

**4. Qualitative Patterns & Anomalies**

Negative control (Berlin→?): top‑5 shows no leakage of “Berlin”: “Germany 0.632 … which 0.247 … the 0.074 … what 0.009 … a 0.0048” [L14–L31].

Important‑word trajectory (NEXT). “Berlin” first enters top‑5 at L 33 and becomes top‑1 at L 36; it remains top‑1 by L 39–40 (CSV L 33/36/39/40). Early layers carry non‑semantic artifacts and fillers; geometry points to an early direction with late calibration: cos_to_final ≥0.2 by L 5 while KL remains ≈13 bits until very late, and rank milestones only cross at {32,33,36}.

Collapse‑layer instruction sensitivity: ablation shows L_sem unchanged (36→36; ΔL_sem=0) [L1633–L1640]. This suggests minimal dependency on the “simply” filler for semantic onset.

Rest‑mass sanity: rest_mass at L 36 is ≈0.00088 and rises to 0.236 at the final layer (CSV L 36 and L 40). Interpretation: top‑k coverage proxy only; not a fidelity measure.

Rotation vs. amplification: cosine rises early (ge_0.2 at L 5; ge_0.4 at L 29), while KL stays high until late (first_kl≤1 at L 40). This pattern—early direction, late calibration—is consistent with late decoder alignment. Final‑head calibration: clean (KL≈0 at final; temp_est=1.0) [L1080–L1087].

Lens sanity: raw_lens_check.sample → max_kl_norm_vs_raw_bits=17.67, lens_artifact_risk=“high” [L1618–L1623]; full scan → pct_layers_kl_ge_{1.0,0.5}={0.756,0.829}, max=98.58 bits, tier=high [L1066–L1076]. Caution: treat any pre‑final “early semantics” conservatively; rely on rank milestones and confirmed semantics. Strict‑copy norm‑only flags are not present.

Temperature robustness: at T=0.1, “Berlin” rank‑1 (p≈0.974; H≈0.173 bits) [L670–L676]; at T=2.0, “Berlin” remains rank‑1 with p≈0.036 (H≈13.161 bits) [L737–L743].

Checklist: RMS lens (✓) [L810–L817]; LayerNorm bias removed? (n.a., RMS) [L812]; Entropy rise at unembed (n.a.); FP32 un‑embed promoted (✓, casting before unembed) [L814–L817]; Punctuation/filler anchoring (early non‑semantic tokens visible; no consistent “is/the/a/of” top‑1 in L0–5); Copy‑reflex (✗); Preferred lens honored (tuned, with baseline parenthetical) [L2138–L2148]; Confirmed semantics reported (✓, L=36, source=raw) [L1449–L1452]; Full dual‑lens metrics cited (✓: pct≥1.0, n_norm_only, earliest, max KL, tier) [L1066–L1076]; Tuned‑lens attribution (✓: ΔKL_tuned, ΔKL_temp, ΔKL_rot at ~25/50/75%) [L2117–L2135].

Quotes (illustrative):
> “last_layer_consistency … kl_to_final_bits: 0.0 … temp_est: 1.0” [L1080–L1087]
> “raw_lens_full … max_kl_norm_vs_raw_bits: 98.58 … tier: high” [L1071–L1076]

**5. Limitations & Data Quirks**

- High raw‑vs‑norm divergence (max KL up to 98.58 bits; tier=high) indicates lens sensitivity; prefer rank milestones and confirmed semantics for onset reporting. Avoid absolute probability comparisons across models (measurement_guidance.suppress_abs_probs=true) [L1066–L1076, L2138–L2148].
- Rest_mass is top‑k coverage only; the rise at final does not imply calibration issues. Use last_layer_consistency for head calibration (here KL≈0) [L1080–L1087].
- Prism sidecar is diagnostic, not the model head. It is regressive for this run (higher KL, no rank gains); do not substitute its probabilities for the baseline head [L856–L876].
- Surface mass comparisons across families are confounded by tokenization; interpret within‑model trends only.

**6. Model Fingerprint**

“Qwen3‑14B: confirmed collapse at L 36 (tuned at L 39); final entropy 3.58 bits; ‘Berlin’ first enters top‑5 at L 33 and becomes rank‑1 by L 36.”

---
Produced by OpenAI GPT-5 

