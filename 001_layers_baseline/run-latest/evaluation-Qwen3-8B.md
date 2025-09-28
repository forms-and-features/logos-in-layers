**1. Overview**
Qwen/Qwen3-8B (8B) evaluated on 2025-09-28 20:02:40 CEST. The probe analyzes layer-by-layer next-token predictions under a norm lens, tracking entropy, KL-to-final, rank milestones, cosine-to-final, and surface mass while checking copy/filler vs semantic collapse.

**2. Method Sanity‑Check**
Diagnostics confirm RMSNorm and norm lens usage with rotary positions: “first_block_ln1_type: ‘RMSNorm’, layer0_position_info: ‘token_only_rotary_model’, use_norm_lens: true” [001_layers_baseline/run-latest/output-Qwen3-8B.json:810–817]. The context prompt matches and ends with “called simply” with no trailing space: “context_prompt: "Give the city name only, plain text. The capital of Germany is called simply"” [001_layers_baseline/run-latest/output-Qwen3-8B.json:817].

Copy detection configuration and flags are present: strict copy “copy_thresh: 0.95, copy_window_k: 1, copy_match_level: ‘id_subsequence’” with soft copy “copy_soft_config { threshold: 0.5, window_ks: [1,2,3] }” and `copy_flag_columns = ["copy_strict@0.95","copy_soft_k1@0.5","copy_soft_k2@0.5","copy_soft_k3@0.5"]` [001_layers_baseline/run-latest/output-Qwen3-8B.json:891–906,839–846,1377–1382]. Gold alignment is ok for both main and control prompts [001_layers_baseline/run-latest/output-Qwen3-8B.json:954,1404]. Negative control summary is present with margins [001_layers_baseline/run-latest/output-Qwen3-8B.json:1407–1408]. Ablation summary exists with both variants (“orig”/“no_filler”) [001_layers_baseline/run-latest/output-Qwen3-8B.json:1383–1390], and both variants appear in the pure CSV (e.g., `prompt_variant = no_filler`) [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:39].

Summary indices (bits/ranks in diagnostics): `first_kl_below_0.5 = 36`, `first_kl_below_1.0 = 36`, `first_rank_le_1 = 31`, `first_rank_le_5 = 29`, `first_rank_le_10 = 29` [001_layers_baseline/run-latest/output-Qwen3-8B.json:894–898]. Units for KL/entropy are bits (CSV headers include `entropy` and `kl_to_final_bits`). Last‑layer head calibration is clean: `kl_to_final_bits = 0.0`, `top1_agree = true`, `p_top1_lens = p_top1_model = 0.4334`, `temp_est = 1.0`, `kl_after_temp_bits = 0.0` [001_layers_baseline/run-latest/output-Qwen3-8B.json:955–963,973].

Lens sanity (raw vs norm): mode “sample”, `lens_artifact_risk: high`, `max_kl_norm_vs_raw_bits: 13.6049`, and no “norm‑only semantics” layer flagged [001_layers_baseline/run-latest/output-Qwen3-8B.json:1316–1375]. Given the high risk rating, pre‑final “early semantics” are interpreted using rank milestones rather than absolute probabilities.

Copy‑collapse flag check (L0–L3): No strict or soft copy flags fire in early layers (`copy_collapse = False`, `copy_soft_k1@0.5 = False` for layers 0–3) [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:2–5]. Earliest strict copy: none; earliest soft copy (k=1/2/3): none (strict and soft stay null in diagnostics) [001_layers_baseline/run-latest/output-Qwen3-8B.json:905–914,915–941]. ✓ rule did not fire (no evidence of copy‑reflex).

Main table uses only `prompt_id = pos`, `prompt_variant = orig` from the pure-next-token CSV.

**3. Quantitative Findings**
| Layer | Summary |
|---|---|
| L 0 | entropy 17.21 bits, top‑1 ‘CLICK’ |
| L 1 | entropy 17.21 bits, top‑1 ‘apr’ |
| L 2 | entropy 17.21 bits, top‑1 ‘财经’ |
| L 3 | entropy 17.21 bits, top‑1 ‘-looking’ |
| L 4 | entropy 17.21 bits, top‑1 ‘院子’ |
| L 5 | entropy 17.20 bits, top‑1 ‘(?)’ |
| L 6 | entropy 17.20 bits, top‑1 ‘ly’ |
| L 7 | entropy 17.15 bits, top‑1 ‘(?)’ |
| L 8 | entropy 17.13 bits, top‑1 ‘(?)’ |
| L 9 | entropy 17.12 bits, top‑1 ‘(?)’ |
| L 10 | entropy 17.02 bits, top‑1 ‘(?)’ |
| L 11 | entropy 17.13 bits, top‑1 ‘ifiable’ |
| L 12 | entropy 17.12 bits, top‑1 ‘ifiable’ |
| L 13 | entropy 17.13 bits, top‑1 ‘ifiable’ |
| L 14 | entropy 17.05 bits, top‑1 ‘""""’ |
| L 15 | entropy 17.04 bits, top‑1 ‘""""’ |
| L 16 | entropy 16.91 bits, top‑1 ‘-’ |
| L 17 | entropy 16.97 bits, top‑1 ‘-’ |
| L 18 | entropy 16.91 bits, top‑1 ‘-’ |
| L 19 | entropy 16.63 bits, top‑1 ‘ly’ |
| L 20 | entropy 16.70 bits, top‑1 ‘_’ |
| L 21 | entropy 16.41 bits, top‑1 ‘_’ |
| L 22 | entropy 15.22 bits, top‑1 ‘______’ |
| L 23 | entropy 15.22 bits, top‑1 ‘____’ |
| L 24 | entropy 10.89 bits, top‑1 ‘____’ |
| L 25 | entropy 13.45 bits, top‑1 ‘____’ |
| L 26 | entropy 5.56 bits, top‑1 ‘____’ |
| L 27 | entropy 4.34 bits, top‑1 ‘____’ |
| L 28 | entropy 4.79 bits, top‑1 ‘____’ |
| L 29 | entropy 1.78 bits, top‑1 ‘-minded’ |
| L 30 | entropy 2.20 bits, top‑1 ‘ Germany’ |
| **L 31** | **entropy 0.45 bits, top‑1 ‘Berlin’** |
| L 32 | entropy 1.04 bits, top‑1 ‘ German’ |
| L 33 | entropy 0.99 bits, top‑1 ‘Berlin’ |
| L 34 | entropy 0.67 bits, top‑1 ‘Berlin’ |
| L 35 | entropy 2.49 bits, top‑1 ‘Berlin’ |

Bold semantic layer (ID‑level gold): L 31 (`is_answer = True`) with ‘Berlin’ [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:33]. Gold answer string is “Berlin” [001_layers_baseline/run-latest/output-Qwen3-8B.json:1720–1728].

Control margin (negative control): `first_control_margin_pos = 1`; `max_control_margin = 0.9999977350` [001_layers_baseline/run-latest/output-Qwen3-8B.json:1407–1408].

Ablation (no‑filler): `L_copy_orig = null`, `L_sem_orig = 31`, `L_copy_nf = null`, `L_sem_nf = 31`, hence `ΔL_copy = null`, `ΔL_sem = 0` [001_layers_baseline/run-latest/output-Qwen3-8B.json:1383–1390].

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) not computed (L_copy = null). Soft ΔHₖ likewise not computed (all `L_copy_soft[k] = null`) [001_layers_baseline/run-latest/output-Qwen3-8B.json:905–914].

Confidence milestones (pure CSV): p_top1 > 0.30 at L 29 (0.7582); p_top1 > 0.60 at L 31 (0.9359); final‑layer p_top1 = 0.4334 [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:29–31,38; 001_layers_baseline/run-latest/output-Qwen3-8B.json:1225–1267].

Rank milestones (diagnostics): rank ≤ 10 at L 29; rank ≤ 5 at L 29; rank ≤ 1 at L 31 [001_layers_baseline/run-latest/output-Qwen3-8B.json:896–898].

KL milestones (diagnostics): first_kl_below_1.0 at L 36; first_kl_below_0.5 at L 36; KL decreases with depth and is ≈0 at final (clean head calibration) [001_layers_baseline/run-latest/output-Qwen3-8B.json:894–896,955–963].

Cosine milestones (pure CSV): no pre‑final layer reaches `cos_to_final ≥ {0.2, 0.4, 0.6}`; final `cos_to_final = 1.0` [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:2–37,38].

Prism Sidecar Analysis. Prism is present and compatible (`present: true, compatible: true`) [001_layers_baseline/run-latest/output-Qwen3-8B.json:825–833]. Early/mid‑depth KL(P_layer||P_final), baseline vs Prism at L≈{0,9,18,27}:
- L0: 12.79 → 12.94
- L9: 12.61 → 12.98
- L18: 12.41 → 13.00
- L27: 6.14 → 13.18
Prism shows higher KL across depths and does not achieve rank milestones (`first_rank_le_{10,5,1} = null`) in the stack, whereas baseline reaches rank‑1 at L 31 [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:33; 001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token-prism.csv]. Copy flags do not spuriously flip under Prism (none fire). Verdict: Regressive.

Tuned‑Lens Sidecar. Loaded with temperatures and summaries [001_layers_baseline/run-latest/output-Qwen3-8B.json:987–999,1628–1672]. ΔKL at depth percentiles (baseline − tuned, pure CSV, L≈{25,50,75}% = {9,18,27}): {+4.14, +4.00, +1.40} bits, indicating substantial early/mid‑depth calibration improvement. Tuned rank milestones: first_rank≤{10,5,1} = {30,31,34}, later than baseline for rank‑1 (34 vs 31). Entropy drift at mid‑depth (tuned L18): entropy 10.06 bits vs teacher 3.12 bits (Δ≈+6.94) [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token-tuned.csv]. Surface→meaning: `L_surface_to_meaning_{norm,tuned} = {31,31}` with (answer_mass_at_L, echo_mass_at_L) ≈ (0.936, 0.011) for norm and (0.101, 0.005) for tuned [001_layers_baseline/run-latest/output-Qwen3-8B.json:943–951,1201–1206]. Geometry: `L_geom_{norm,tuned} = {34,34}` [001_layers_baseline/run-latest/output-Qwen3-8B.json:946,1204]. Coverage: `L_topk_decay_{norm,tuned} = {0,1}` [001_layers_baseline/run-latest/output-Qwen3-8B.json:949,1207]. Norm temperature diagnostics present (`tau_norm_per_layer`, snapshots at 25/50/75%) [001_layers_baseline/run-latest/output-Qwen3-8B.json:848–885,975–986]. Skip‑layers sanity provided (advisory) [001_layers_baseline/run-latest/output-Qwen3-8B.json:1212–1218].

**4. Qualitative Patterns & Anomalies**
Negative control (“Berlin is the capital of”): top‑5 includes “ Germany, 0.7286; which, 0.2207; the, 0.0237; what, 0.0114; __, 0.0023 … Berlin, 0.00046” [001_layers_baseline/run-latest/output-Qwen3-8B.json:14–31,46–48]. Semantic leakage: Berlin rank ≈ 9 (p ≈ 0.00046).

Important‑word trajectory (records CSV; IMPORTANT_WORDS = [“Germany”, “Berlin”, “capital”, “Answer”, “word”, “simply”] [001_layers_baseline/run.py:349]). ‘Berlin’ begins to appear near NEXT late in the stack: at L28 (pos=15, ‘ simply’) it enters the top‑k (“… Berlin, 0.00356”) [001_layers_baseline/run-latest/output-Qwen3-8B-records.csv:562]; at L29 it strengthens (“… Berlin, 0.0265”) [001_layers_baseline/run-latest/output-Qwen3-8B-records.csv:578]; at L30 it is top‑2 at NEXT and present at other positions (e.g., pos=13/14) [001_layers_baseline/run-latest/output-Qwen3-8B-records.csv:591–594]; by L31, Berlin is top‑1 at NEXT with high mass (“ Berlin, 0.9359”) and dominates nearby positions by L32 (“pos=13 ‘is’: Berlin 0.9999; pos=14 ‘called’: Berlin 0.9960”) [001_layers_baseline/run-latest/output-Qwen3-8B-records.csv:610,628–629]. This reflects late consolidation around the answer token, with “Germany/German” active just before collapse (e.g., L30 NEXT top‑1 “ Germany”) [001_layers_baseline/run-latest/output-Qwen3-8B-records.csv:594].

Instruction ablation. Removing “simply” does not shift the collapse layer (`ΔL_sem = 0`; both 31) [001_layers_baseline/run-latest/output-Qwen3-8B.json:1383–1390], suggesting semantics are not anchored to this stylistic cue for this probe.

Rest‑mass sanity. Rest_mass falls after mid‑stack and remains modest post‑collapse; the maximum after L_semantic is ≈0.088 at L35, indicating stable top‑k coverage (no precision loss spike) [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:35].

Rotation vs amplification. KL_to_final decreases with depth (e.g., 12.79 at L0 → 1.06 at L31) while `p_answer` rises (≈6.7e‑06 at L0 → 0.936 at L31) and answer rank improves (14864 → 1) [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:2,31]. `cos_to_final` stays < 0 through most of the stack and never crosses 0.2 before the final row, indicating “late calibration” of the direction relative to the model head; rely on ranks rather than absolute cosine thresholds for earliness here.

Head calibration (final layer). Clean agreement and temperature 1.0: `kl_to_final_bits = 0`, `top1_agree = true`, `p_top1_lens = p_top1_model = 0.4334`, `kl_after_temp_bits = 0` [001_layers_baseline/run-latest/output-Qwen3-8B.json:955–963].

Lens sanity. Raw‑vs‑norm check reports `lens_artifact_risk: high` with `max_kl_norm_vs_raw_bits: 13.60` and no norm‑only semantic layer [001_layers_baseline/run-latest/output-Qwen3-8B.json:1371–1375]. Caution that any “early semantics” may be lens‑induced; rank milestones are preferred.

Temperature robustness. At T = 0.1, Berlin is rank‑1 with p ≈ 0.99915 (entropy ≈ 0.0099 bits); at T = 2.0, Berlin remains rank‑1 with p ≈ 0.04186 (entropy ≈ 13.40 bits) [001_layers_baseline/run-latest/output-Qwen3-8B.json:670–676,737–743].

Checklist:
– RMS lens? ✓ (RMSNorm; norm lens enabled) [001_layers_baseline/run-latest/output-Qwen3-8B.json:807–813].
– LayerNorm bias removed? ✓ “not_needed_rms_model” [001_layers_baseline/run-latest/output-Qwen3-8B.json:812].
– Entropy rise at unembed? ✓ final entropy 3.12 bits [001_layers_baseline/run-latest/output-Qwen3-8B.json:1222].
– FP32 un‑embed promoted? ✓ `unembed_dtype: torch.float32` (via analysis shadow) [001_layers_baseline/run-latest/output-Qwen3-8B.json:809].
– Punctuation / markup anchoring? ✓ early layers dominated by punctuation/garbage tokens (e.g., ‘CLICK’, ‘(?)’, quotes/underscores) [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:2–16,22–28].
– Copy‑reflex? ✗ (no strict/soft hits at L0–L3) [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:2–5].
– Grammatical filler anchoring? Partial: late pre‑collapse NEXT top‑1 is “ Germany” at L30 [001_layers_baseline/run-latest/output-Qwen3-8B-records.csv:594].

**5. Limitations & Data Quirks**
- Rest_mass remains < 0.3 after L_semantic (max ≈ 0.088 at L35), so no evidence of top‑k truncation artefacts in late layers. Still, Rest_mass is top‑k coverage only; fidelity is judged via KL/last‑layer checks.
- KL is lens‑sensitive; raw‑vs‑norm reports “high” artifact risk (max KL ≈ 13.60). Prefer within‑model trends and rank milestones for early‑layer claims.
- Final KL≈0 reflects head calibration; cross‑model probability comparisons should still rely on rank milestones and KL thresholds, not raw probabilities.
- Raw‑vs‑norm check used mode “sample”, so findings are sampled sanity rather than exhaustive [001_layers_baseline/run-latest/output-Qwen3-8B.json:1316].
- Surface mass depends on tokenizer; absolute levels are not compared across models here.

**6. Model Fingerprint**
“Qwen‑3‑8B: collapse at L 31; final entropy 3.12 bits; ‘Berlin’ stabilizes rank 1 late with ‘Germany’ briefly top‑1 at L 30.”

---
Produced by OpenAI GPT-5
