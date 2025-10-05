# Evaluation Report: google/gemma-2-27b

*Run executed on: 2025-10-04 18:04:23*
**Overview**
This evaluation covers google/gemma-2-27b (46 layers), run artifacts dated 2025-10-04, analyzing layer-by-layer behavior on “Germany → Berlin” with a norm lens plus sidecars (Prism, Tuned-Lens). The probe records copy/filler reflexes, semantic collapse (rank milestones), KL-to-final alignment, cosine trajectories, and calibration diagnostics.

**Method Sanity-Check**
The context prompt ends with “called simply” (no trailing space) in both script and run: "context_prompt = \"… called simply\""  (001_layers_baseline/run.py:254) and "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply"  (001_layers_baseline/run-latest/output-gemma-2-27b.json:4). Norm lens is enabled and used: "use_norm_lens": true (001_layers_baseline/run-latest/output-gemma-2-27b.json:808). Last-layer head calibration exists with non-zero KL and temperature fit: "kl_to_final_bits": 1.1352, "temp_est": 2.6102, "kl_after_temp_bits": 0.5665, "warn_high_last_layer_kl": true (001_layers_baseline/run-latest/output-gemma-2-27b.json:6462,6468-6470,6479). Measurement guidance: "prefer_ranks": true, "suppress_abs_probs": true; reasons include "warn_high_last_layer_kl", "norm_only_semantics_window", and "high_lens_artifact_risk" (001_layers_baseline/run-latest/output-gemma-2-27b.json:7566-7573). Copy detector configuration and flags are present and consistent across JSON/CSV: copy strict at τ ∈ {0.95,0.7,0.8,0.9} and soft k∈{1,2,3} at τ_soft=0.5 are listed under "copy_flag_columns" (001_layers_baseline/run-latest/output-gemma-2-27b.json:7033-7042) and appear in the pure CSV header (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:1). Summary indices exist: "first_rank_le_1": 46, "first_rank_le_5": 46, "first_rank_le_10": 46; "first_kl_below_1.0": null; "first_kl_below_0.5": null (001_layers_baseline/run-latest/output-gemma-2-27b.json:5645-5649). Raw-vs-norm lens window/full checks report high artifact risk and norm-only semantics at the final layer: "max_kl_norm_vs_raw_bits_window": 99.54, "norm_only_semantics_layers": [46] (001_layers_baseline/run-latest/output-gemma-2-27b.json:5769-5773) and full summary "lens_artifact_score": 0.9872, tier "high" (001_layers_baseline/run-latest/output-gemma-2-27b.json:5775-5784). Normalizer provenance shows pre-norm strategy with next ln1 and correct epsilon placement: "arch": "pre_norm", "strategy": "next_ln1" (001_layers_baseline/run-latest/output-gemma-2-27b.json:5788-5789), with layer-0 ln_source "blocks[0].ln1" and final "ln_final" (001_layers_baseline/run-latest/output-gemma-2-27b.json:5792-5793,6160-6161). Per-layer normalizer effect metrics are present and stable (e.g., layer 0: "resid_norm_ratio": 0.7865, "delta_resid_cos": 0.5722; final: 0.0527, 0.6617) (001_layers_baseline/run-latest/output-gemma-2-27b.json:5796,6164). Unembedding bias is absent: "present": false, l2_norm=0 (001_layers_baseline/run-latest/output-gemma-2-27b.json:826-830). Environment/determinism: torch 2.8.0, device=cpu, deterministic_algorithms=true, seed=316 (001_layers_baseline/run-latest/output-gemma-2-27b.json:7019-7033). Numeric health is clean (no NaN/Inf) (001_layers_baseline/run-latest/output-gemma-2-27b.json:6426-6432). Copy mask is present with plausible ignored tokens (whitespace runs), size 4668 (001_layers_baseline/run-latest/output-gemma-2-27b.json:5618-5636). Gold alignment is OK (ID-based): "gold_alignment": "ok" (001_layers_baseline/run-latest/output-gemma-2-27b.json:6460). Control summary present: "first_control_margin_pos": 0, "max_control_margin": 0.9911 (001_layers_baseline/run-latest/output-gemma-2-27b.json:7065-7069). Ablation summary present with both variants: L_copy_orig=0, L_sem_orig=46, L_copy_nf=3, L_sem_nf=46 (001_layers_baseline/run-latest/output-gemma-2-27b.json:7043-7047).

Copy-reflex evidence: in the pure CSV, layer 0 shows copy_collapse=True with strict flags at all τ and soft k1=True: "copy_collapse,copy_strict@0.95,copy_strict@0.7,copy_strict@0.8,copy_strict@0.9,copy_soft_k1@0.5=True" (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2). ✓ rule satisfied for early copy-reflex.

**Quantitative Findings**
Table (prompt_id=pos, prompt_variant=orig). For each layer L: entropy (bits) and top‑1 token at NEXT. Bold marks semantic collapse (rank≤1) under the norm lens.

- L 0 – entropy 0.0005 bits, top‑1 ' simply' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2)
- L 1 – entropy 8.7582 bits, top‑1 '' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:3)
- L 2 – entropy 8.7645 bits, top‑1 '' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:4)
- L 3 – entropy 0.8857 bits, top‑1 ' simply' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:5)
- L 4 – entropy 0.6183 bits, top‑1 ' simply' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:6)
- L 5 – entropy 8.5203 bits, top‑1 '๲' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:7)
- L 6 – entropy 8.5531 bits, top‑1 '' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:8)
- L 7 – entropy 8.5470 bits, top‑1 '' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:9)
- L 8 – entropy 8.5287 bits, top‑1 '' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:10)
- L 9 – entropy 8.5238 bits, top‑1 '𝆣' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:11)
- L 10 – entropy 8.3452 bits, top‑1 ' dieſem' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:12)
- L 11 – entropy 8.4928 bits, top‑1 '𝆣' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:13)
- L 12 – entropy 8.3244 bits, top‑1 '' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:14)
- L 13 – entropy 8.2225 bits, top‑1 '' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:15)
- L 14 – entropy 7.8766 bits, top‑1 '' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:16)
- L 15 – entropy 7.7925 bits, top‑1 '' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:17)
- L 16 – entropy 7.9748 bits, top‑1 ' dieſem' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:18)
- L 17 – entropy 7.7856 bits, top‑1 ' dieſem' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:19)
- L 18 – entropy 7.2999 bits, top‑1 'ſicht' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:20)
- L 19 – entropy 7.5278 bits, top‑1 ' dieſem' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:21)
- L 20 – entropy 6.2100 bits, top‑1 'ſicht' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:22)
- L 21 – entropy 6.4560 bits, top‑1 'ſicht' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:23)
- L 22 – entropy 6.3784 bits, top‑1 ' dieſem' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:24)
- L 23 – entropy 7.0104 bits, top‑1 ' dieſem' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:25)
- L 24 – entropy 6.4970 bits, top‑1 ' dieſem' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:26)
- L 25 – entropy 6.9949 bits, top‑1 ' dieſem' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:27)
- L 26 – entropy 6.2198 bits, top‑1 ' dieſem' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:28)
- L 27 – entropy 6.7007 bits, top‑1 ' dieſem' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:29)
- L 28 – entropy 7.1401 bits, top‑1 ' dieſem' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:30)
- L 29 – entropy 7.5741 bits, top‑1 ' dieſem' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:31)
- L 30 – entropy 7.3302 bits, top‑1 ' dieſem' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:32)
- L 31 – entropy 7.5652 bits, top‑1 ' dieſem' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:33)
- L 32 – entropy 8.8736 bits, top‑1 ' zuſammen' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:34)
- L 33 – entropy 6.9447 bits, top‑1 ' dieſem' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:35)
- L 34 – entropy 7.7383 bits, top‑1 ' dieſem' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:36)
- L 35 – entropy 7.6507 bits, top‑1 ' dieſem' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:37)
- L 36 – entropy 7.6577 bits, top‑1 ' dieſem' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:38)
- L 37 – entropy 7.5724 bits, top‑1 ' dieſem' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:39)
- L 38 – entropy 7.5536 bits, top‑1 'パンチラ' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:40)
- L 39 – entropy 7.2324 bits, top‑1 ' dieſem' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:41)
- L 40 – entropy 8.7105 bits, top‑1 '展板' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:42)
- L 41 – entropy 7.0817 bits, top‑1 ' dieſem' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:43)
- L 42 – entropy 7.0565 bits, top‑1 ' dieſem' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:44)
- L 43 – entropy 7.0889 bits, top‑1 ' dieſem' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:45)
- L 44 – entropy 7.5683 bits, top‑1 ' dieſem' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:46)
- L 45 – entropy 7.1406 bits, top‑1 ' Geſch' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:47)
- L 46 – entropy 0.1180 bits, top‑1 ' Berlin' (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48)

Bold collapse layer: L_semantic_norm = 46, confirmed (tuned) with use_confirmed_semantics=true (001_layers_baseline/run-latest/output-gemma-2-27b.json:6840-6847,7576-7578).

Control margin: first_control_margin_pos=0; max_control_margin=0.9911 (001_layers_baseline/run-latest/output-gemma-2-27b.json:7065-7069).

Ablation (no‑filler): L_copy_orig=0, L_sem_orig=46; L_copy_nf=3, L_sem_nf=46; ΔL_copy=3, ΔL_sem=0 (001_layers_baseline/run-latest/output-gemma-2-27b.json:7043-7047). Interpretation: removing “simply” delays strict copy detection but leaves semantic collapse unchanged.

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = 0.0005 − 0.1180 ≈ −0.1175 (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2,48). Soft ΔH_k (k=1 at τ_soft=0.5) is the same because L_copy_soft[k1]=0 here (001_layers_baseline/run-latest/output-gemma-2-27b.json:5656-5661).

Confidence milestones (p_top1, not necessarily the answer): p_top1 > 0.30 at L 0; p_top1 > 0.60 at L 0; final-layer p_top1 = 0.9841 (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2,48).

Rank milestones (norm): rank ≤ 10 at L 46; rank ≤ 5 at L 46; rank ≤ 1 at L 46 (001_layers_baseline/run-latest/output-gemma-2-27b.json:5647-5649). Preferred lens honored: measurement_guidance.preferred_lens_for_reporting="norm"; use_confirmed_semantics=true (001_layers_baseline/run-latest/output-gemma-2-27b.json:7576-7578).

KL milestones: first_kl_below_1.0 = null; first_kl_below_0.5 = null (001_layers_baseline/run-latest/output-gemma-2-27b.json:5645-5646). Final-layer KL is not ≈ 0: 1.1352 bits (001_layers_baseline/run-latest/output-gemma-2-27b.json:6462); last-layer calibration warns with temp_est=2.61 and kl_after_temp_bits=0.5665 (001_layers_baseline/run-latest/output-gemma-2-27b.json:6468-6469).

Cosine milestones (norm): ge_0.2 at L 1; ge_0.4 at L 46; ge_0.6 at L 46 (001_layers_baseline/run-latest/output-gemma-2-27b.json:5736-5740). Final cos_to_final ≈ 0.9994 at L 46 (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48).

Depth fractions: L_semantic_frac=1.0; first_rank_le_5_frac=1.0; L_copy_strict_frac=0.0 (001_layers_baseline/run-latest/output-gemma-2-27b.json:5744-5747).

Copy robustness (threshold sweep): stability="mixed"; earliest strict copy at τ=0.70 and τ=0.95 is L 0; norm_only_flags false at all τ (001_layers_baseline/run-latest/output-gemma-2-27b.json:5698-5724). Copy configuration: "copy_thresh": 0.95, "copy_window_k": 1, "copy_match_level": "id_subsequence" (001_layers_baseline/run-latest/output-gemma-2-27b.json:5642-5644). Soft-copy config: τ_soft=0.5, window_ks=[1,2,3] (001_layers_baseline/run-latest/output-gemma-2-27b.json:889-899).

Prism Sidecar Analysis. Presence/compatibility: present=true, compatible=true (001_layers_baseline/run-latest/output-gemma-2-27b.json:834-846). Early-depth KL: baseline p25/p50/p75 ≈ 42.01/43.15/42.51 bits vs Prism ≈ 19.43/19.42/19.43 bits (drops ≈ 22.6/23.7/23.1) (001_layers_baseline/run-latest/output-gemma-2-27b.json:862-888). Rank milestones under Prism remain null (no rank ≤ {10,5,1}) (001_layers_baseline/run-latest/output-gemma-2-27b.json:848-861). At the final layer, Prism’s answer_rank remains large (e.g., 165,699) with very low p_top1 (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token-prism.csv:48). Verdict: Regressive (KL decreases substantially but ranks do not improve to match norm lens semantics).

**Qualitative Patterns & Anomalies**
Negative control shows clean behavior: for “Berlin is the capital of”, top-5 are " Germany" (0.8676), " the" (0.0650), " and" (0.0065), " a" (0.0062), " Europe" (0.0056) (001_layers_baseline/run-latest/output-gemma-2-27b.json:14-36). No “Berlin” appears in that top-5, so no obvious semantic leakage under this test.

Important-word trajectory (records). The model strongly favors “Berlin” near the end of the context: at L46 pos=14 (“ is”), top‑1 is “ Berlin” (0.999998); at pos=15 (“ called”), “ Berlin” (0.999868); at NEXT (pos=16, “ simply”) “ Berlin” (0.9841) (001_layers_baseline/run-latest/output-gemma-2-27b-records.csv:804-806; 001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48). This suggests the lexical target is well formed pre‑final; the norm lens shows early directional alignment (cos≥0.2 by L1) but calibration remains high‑KL until the very end.

Copy/filler dynamics. A strict copy-collapsed top‑1 (“ simply”) occurs at L0-L4, meeting τ=0.95 and τ∈{0.7,0.8,0.9} (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2,5,6). Ablation removes the “simply” cue, shifting strict copy to L3 while leaving semantic collapse at L46 unchanged (001_layers_baseline/run-latest/output-gemma-2-27b.json:7043-7047). This points to stylistic anchoring influencing very early layers without affecting final semantics.

Rest-mass sanity. Rest_mass is tiny at the collapse layer (≈1.99e‑07) and does not spike post‑collapse (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48). No evidence of precision loss from top‑k truncation after semantics.

Rotation vs amplification. Cosine-to-final rises early (ge_0.2 at L1) while KL-to-final stays large across mid‑depths (p50≈43 bits), indicating “early direction, late calibration” (001_layers_baseline/run-latest/output-gemma-2-27b.json:5736-5740, 862-876). Final-layer calibration is imperfect: "kl_to_final_bits": 1.1352, with temperature fit reducing KL to 0.5665 (001_layers_baseline/run-latest/output-gemma-2-27b.json:6462,6469). Prefer rank-based milestones (per measurement_guidance) when comparing across families.

Lens sanity and artifacts. Raw‑vs‑norm checks show very large norm-vs-raw KL (up to ≈99.54 bits) and norm-only semantics at the final layer; risk tier "high" (001_layers_baseline/run-latest/output-gemma-2-27b.json:5769-5784). Treat any pre‑final semantics cautiously; here, rank‑1 only arrives at L46 in norm and is marked confirmed. Copy strict norm_only_flags are false across τ (001_layers_baseline/run-latest/output-gemma-2-27b.json:5717-5724).

Temperature robustness. At T=2.0 on the NEXT slot, “ Berlin” remains top but much flatter: 0.0492 (001_layers_baseline/run-latest/output-gemma-2-27b.json:1104-1120). Entropy rises accordingly (teacher entropy bits listed as 2.8856 for the final head).

Head calibration (final). Family-typical Gemma signature: top‑1 agrees but norm-lens vs final-head KL is high: "top1_agree": true, "p_top1_lens": 0.9841 vs "p_top1_model": 0.4226; "temp_est": 2.6102; "kl_after_temp_bits": 0.5665 (001_layers_baseline/run-latest/output-gemma-2-27b.json:6463-6469). Treat final probabilities as within-model only; rely on ranks for comparisons.

Checklist
- RMS lens? ✓ (001_layers_baseline/run-latest/output-gemma-2-27b.json:812-816)
- LayerNorm bias removed? ✓ (RMSNorm; no bias) (001_layers_baseline/run-latest/output-gemma-2-27b.json:812-816)
- Entropy rise at unembed? n.a.
- FP32 un-embed promoted? ✓ ("unembed_dtype": "torch.float32") (001_layers_baseline/run-latest/output-gemma-2-27b.json:810-812)
- Punctuation / markup anchoring? ✓ (early exotic tokens/punctuation; e.g., L5–L12 top‑1 are non‑semantic codepoints)
- Copy-reflex? ✓ (layer 0 strict) (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2)
- Grammatical filler anchoring? ✗ (L0–L5 top‑1 not in {is,the,a,of})
- Preferred lens honored? ✓ (preferred_lens_for_reporting="norm") (001_layers_baseline/run-latest/output-gemma-2-27b.json:7576)
- Confirmed semantics reported? ✓ (L_semantic_confirmed=46; source=tuned) (001_layers_baseline/run-latest/output-gemma-2-27b.json:6840-6847)
- Full dual‑lens metrics cited? ✓ (raw_lens_full tier=high; pct_ge_1.0≈0.979) (001_layers_baseline/run-latest/output-gemma-2-27b.json:5775-5784)
- Tuned‑lens attribution done? ✓ (ΔKL_tuned at percentiles small; prefer_tuned=false) (001_layers_baseline/run-latest/output-gemma-2-27b.json:7497-7562)
- normalization_provenance present? ✓ (layer 0 ln1; final ln_final) (001_layers_baseline/run-latest/output-gemma-2-27b.json:5792-5793,6160-6161)
- per-layer normalizer metrics present? ✓ (resid_norm_ratio, delta_resid_cos) (001_layers_baseline/run-latest/output-gemma-2-27b.json:5796,6164)
- unembed bias audited? ✓ (present=false) (001_layers_baseline/run-latest/output-gemma-2-27b.json:826-830)
- deterministic_algorithms = true? ✓ (001_layers_baseline/run-latest/output-gemma-2-27b.json:7027-7033)
- numeric_health clean? ✓ (001_layers_baseline/run-latest/output-gemma-2-27b.json:6426-6432)
- copy_mask plausible? ✓ (size=4668; whitespace runs) (001_layers_baseline/run-latest/output-gemma-2-27b.json:5618-5636)
- layer_map present? n.a.

**Limitations & Data Quirks**
Final‑head calibration is imperfect (warn_high_last_layer_kl=true; KL≈1.14 bits at final); follow measurement_guidance to prefer rank milestones and avoid absolute probability comparisons across families (001_layers_baseline/run-latest/output-gemma-2-27b.json:6462,6479,7566-7575). Raw‑vs‑norm checks show high lens‑artifact risk (max norm-vs-raw KL≈99.5 bits; tier=high), so treat any pre‑final “early semantics” as potentially lens‑induced; rely on confirmed semantics and ranks (001_layers_baseline/run-latest/output-gemma-2-27b.json:5769-5784). Surface‑mass and coverage metrics depend on tokenizer details; interpret within‑model trends rather than across‑family magnitudes. Soft-copy threshold here is τ_soft=0.5 (not the typical 0.33); treat “soft” hits accordingly (001_layers_baseline/run-latest/output-gemma-2-27b.json:889-899).

**Model Fingerprint**
Gemma‑2‑27B: collapse at L 46; final KL≈1.14 bits with temp≈2.61; “Berlin” is top‑1 at NEXT with p≈0.984.

---
Produced by OpenAI GPT-5 
