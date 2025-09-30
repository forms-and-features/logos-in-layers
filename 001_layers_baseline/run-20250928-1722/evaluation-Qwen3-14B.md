# Evaluation — Qwen3-14B

## 1. Overview
Qwen/Qwen3-14B (40 layers; pre-norm) evaluated in run-latest (timestamp-20250928-1722).
The probe captures per-layer entropy, KL-to-final, cosine geometry, answer rank, copy flags, and includes Prism and Tuned‑Lens sidecars.

## 2. Method sanity-check
> context_prompt: "Give the city name only, plain text. The capital of Germany is called simply" [001_layers_baseline/run-latest/output-Qwen3-14B.json:4]
> use_norm_lens=true; first_block_ln1_type=RMSNorm; layer0_position_info=token_only_rotary_model; mixed_precision_fix=casting_to_fp32_before_unembed [001_layers_baseline/run-latest/output-Qwen3-14B.json:807–816]
Positional encoding and the intended norm lens are active; the context ends with “called simply” (no trailing space). Model stats confirm a 40‑layer pre‑norm transformer [001_layers_baseline/run-latest/output-Qwen3-14B.json:1311–1317]. Last‑layer head calibration is aligned: kl_to_final_bits=0.0; top1_agree=true; p_top1_lens=p_top1_model=0.34514; temp_est=1.0; kl_after_temp_bits=0.0 [001_layers_baseline/run-latest/output-Qwen3-14B.json:959–977].

Copy detection config present: copy_thresh=0.95; copy_window_k=1; copy_match_level=id_subsequence; copy_soft_config={threshold: 0.5, window_ks: [1,2,3], extra_thresholds: []}; copy_flag_columns=[copy_strict@0.95, copy_soft_k1@0.5, copy_soft_k2@0.5, copy_soft_k3@0.5] [001_layers_baseline/run-latest/output-Qwen3-14B.json:895–906,839–847,1380–1384]. Gold‑token alignment is ok [001_layers_baseline/run-latest/output-Qwen3-14B.json:958]. Control prompt and summary exist (first_control_margin_pos=0; max_control_margin=0.9741542933185612) [001_layers_baseline/run-latest/output-Qwen3-14B.json:1394–1412]. Ablation summary exists (L_copy_orig=null, L_sem_orig=36; L_copy_nf=null, L_sem_nf=36; delta_L_sem=0) and both prompt variants appear in CSVs [001_layers_baseline/run-latest/output-Qwen3-14B.json:1386–1392; 001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:43–45].

Summary indices (bits and ranks): first_kl_below_0.5=40; first_kl_below_1.0=40; first_rank_le_1=36; first_rank_le_5=33; first_rank_le_10=32 [001_layers_baseline/run-latest/output-Qwen3-14B.json:898–902]. KL/entropy units are bits in the CSV (e.g., teacher_entropy_bits=3.5835) [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:42].

Lens sanity: raw_lens_check.mode=sample; summary: lens_artifact_risk=high; max_kl_norm_vs_raw_bits=17.6735; first_norm_only_semantic_layer=null [001_layers_baseline/run-latest/output-Qwen3-14B.json:1318–1378]. Treat any pre‑final “early semantics” cautiously and prefer rank milestones.

Copy‑collapse flag check (strict τ=0.95, δ=0.10): no rows with copy_collapse=True; earliest soft hits k1@τ=0.5: none (layers 0–3 are all False) [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:2–5].

## 3. Quantitative findings
Main prompt (prompt_id=pos, prompt_variant=orig). Table shows layer, entropy (bits), and generic top‑1 token.

| Layer | Entropy (bits) | Top-1 |
|---|---:|---|
| L 0 | 17.212854 | ' 梳' |
| L 1 | 17.212021 | ' 地处' |
| L 2 | 17.211170 | ' 是一部' |
| L 3 | 17.209875 | ' tics' |
| L 4 | 17.208380 | ' tics' |
| L 5 | 17.207327 | '-minded' |
| L 6 | 17.205141 | ' 过去的' |
| L 7 | 17.186316 | '�' |
| L 8 | 17.179604 | '-minded' |
| L 9 | 17.187605 | '-minded' |
| L 10 | 17.169565 | ' (?)' |
| L 11 | 17.151134 | ' 时代的' |
| L 12 | 17.165318 | ' といって' |
| L 13 | 17.115282 | ' nav' |
| L 14 | 17.140715 | ' nav' |
| L 15 | 17.148745 | ' 唿' |
| L 16 | 17.134632 | ' 闯' |
| L 17 | 17.137224 | ' 唿' |
| L 18 | 17.100914 | '____' |
| L 19 | 17.075287 | '____' |
| L 20 | 16.932322 | '____' |
| L 21 | 16.985991 | ' 年夜' |
| L 22 | 16.954144 | ' 年夜' |
| L 23 | 16.839663 | '____' |
| L 24 | 16.760223 | '____' |
| L 25 | 16.757845 | ' 年夜' |
| L 26 | 16.668522 | '____' |
| L 27 | 16.031609 | '____' |
| L 28 | 15.234417 | '____' |
| L 29 | 14.186926 | ' 这个名字' |
| L 30 | 7.789196 | ' 这个名字' |
| L 31 | 5.161718 | '____' |
| L 32 | 0.815953 | '____' |
| L 33 | 0.481331 | '____' |
| L 34 | 0.594809 | '____' |
| L 35 | 0.667881 | '____' |
| L 36 | **0.312212** | **' Berlin'** |
| L 37 | 0.905816 | ' ____' |
| L 38 | 1.212060 | ' ____' |
| L 39 | 0.952112 | ' Berlin' |
| L 40 | 3.583520 | ' Berlin' |

Control margin (JSON): first_control_margin_pos=0; max_control_margin=0.9741542933185612 [001_layers_baseline/run-latest/output-Qwen3-14B.json:1410–1411].

Ablation (no‑filler): L_copy_orig=null, L_sem_orig=36; L_copy_nf=null, L_sem_nf=36; ΔL_copy=null; ΔL_sem=0 [001_layers_baseline/run-latest/output-Qwen3-14B.json:1386–1392].

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n/a (L_copy=null). Soft ΔH_k (bits): k∈{1,2,3} all n/a (no soft copy layer).

Confidence milestones (pure CSV): p_top1>0.30 at L31; p_top1>0.60 at L32; final-layer p_top1=0.34514 [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:34–42].

Rank milestones (JSON): rank≤10 at L32; rank≤5 at L33; rank≤1 at L36 [001_layers_baseline/run-latest/output-Qwen3-14B.json:900–902].

KL milestones (JSON): first_kl_below_1.0 at L40; first_kl_below_0.5 at L40; KL decreases with depth and is ≈0 at final (kl_to_final_bits=0.0) [001_layers_baseline/run-latest/output-Qwen3-14B.json:898–901,959–961].

Cosine milestones (pure CSV): first cos_to_final≥0.2 at L5; ≥0.4 at L29; ≥0.6 at L36; final cos_to_final≈1.0000 [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:7,34,38,42].

Prism Sidecar Analysis
- Presence: compatible=true; layers=['embed', 9, 19, 29] [001_layers_baseline/run-latest/output-Qwen3-14B.json:825–836].
- Early-depth stability (KL_base vs KL_prism): L0 12.961→13.172; L10 12.909→13.166; L20 12.978→13.228; L30 12.730→13.438; L39 1.982→14.659; L40 0.000→14.840 (top‑1 often diverges, e.g., L0: '梳' vs '音响') [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:2; 001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token-prism.csv].
- Rank milestones (Prism pure): first_rank≤10=None; ≤5=None; ≤1=None (no improvement vs baseline).
- Cosine drift: cos_to_final markedly lower under Prism at all sampled depths (e.g., L30 0.464→0.058).
- Copy flags: no strict or soft copy flagged under Prism.
- Verdict: Regressive (higher KL across depths and later/no rank milestones).

Tuned‑Lens Sidecar (pure CSV)
- ΔKL at depth percentiles (baseline − tuned): L10 +4.682 bits; L20 +4.490 bits; L30 +3.900 bits (lower KL under tuned) [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv, 001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token-tuned.csv].
- Last‑layer agreement: kl_to_final_bits=0.0; top1_agree=true; temp_est=1.0; kl_after_temp_bits=0.0 [001_layers_baseline/run-latest/output-Qwen3-14B.json:959–977].
- Entropy drift (tuned, mid‑depth): L10 +7.766 bits; L20 +7.453 bits; L30 +1.207 bits (entropy − teacher_entropy_bits) [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token-tuned.csv].
- Rank earliness (tuned): rank≤10 at L33; rank≤5 at L34; rank≤1 at L39 (no improvement over baseline’s L32/L33/L36).
- Surface→meaning: L_surface_to_meaning_norm=36 (answer_mass=0.9530; echo_mass≈4.39e‑06); L_surface_to_meaning_tuned=36 (answer_mass=0.3632; echo_mass≈9.97e‑05) [001_layers_baseline/run-latest/output-Qwen3-14B.json:947–951,1121–1126].
- Geometry: L_geom_norm=35; L_geom_tuned=35 [001_layers_baseline/run-latest/output-Qwen3-14B.json:950,1125]. Coverage: L_topk_decay_norm=0; L_topk_decay_tuned=1 [001_layers_baseline/run-latest/output-Qwen3-14B.json:953,1128].
- Norm temperature snapshots present: kl_to_final_bits_norm_temp@25/50/75% = 12.986/13.204/12.489 at layers 10/20/30 [001_layers_baseline/run-latest/output-Qwen3-14B.json:979–989].

## 4. Qualitative patterns & anomalies
Negative control. “Berlin is the capital of” produces top‑5:  Germany (0.6320),  which (0.2468),  the (0.0737),  what (0.0094),  a (0.0048) — no “Berlin” in top‑5 [001_layers_baseline/run-latest/output-Qwen3-14B.json:10–31]. If “Give the country name only, plain text. Berlin is the capital of” is used, “ Germany” appears with p≈0.0282 among question‑shaped tokens [001_layers_baseline/run-latest/output-Qwen3-14B.json:640–664].

Important‑word trajectory (records CSV; IMPORTANT_WORDS=["Germany","Berlin","capital","Answer","word","simply"] [001_layers_baseline/run.py:349]). Around the NEXT position (pos=15), ‘Berlin’ strengthens late: at L33 pos=14 it is top‑3 (p≈0.106) and also appears at pos=15 (top‑5, p≈0.00253) [001_layers_baseline/run-latest/output-Qwen3-14B-records.csv:624–625]. At L34, ‘Berlin’ enters top‑5 at pos=13 (p≈0.0259) and pos=15 (top‑3, p≈0.0156) [001_layers_baseline/run-latest/output-Qwen3-14B-records.csv:639,641]. By L36, ‘Berlin’ is top‑1 across pos=13/14/15 with high confidence: “ Berlin, 0.9851 / 0.9933 / 0.9530” [001_layers_baseline/run-latest/output-Qwen3-14B-records.csv:708–710]. Final layer keeps ‘Berlin’ dominant across positions (e.g., pos=15: “ Berlin, 0.3451”) amid punctuation distractors [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:42].

Collapse‑layer instruction sensitivity. Test prompts without the exact “simply” phrasing still yield high immediate confidence (e.g., “Germany’s capital city is called” → “ Berlin, 0.7443”) but do not provide layer‑wise collapse indices; rely on the main run (L_semantic=36) [001_layers_baseline/run-latest/output-Qwen3-14B.json:200–232]. Stylistic ablation shows no shift (ΔL_sem=0) [001_layers_baseline/run-latest/output-Qwen3-14B.json:1386–1392].

Rest‑mass sanity. Rest_mass is low through L39 and spikes at final to 0.236 (top‑k coverage artifact, not fidelity) [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:36–42].

Rotation vs amplification. Cosine to final rises early (≥0.2 by L5; ≥0.4 by L29) while KL stays high until the very end (first<1.0 only at L40), and the answer turns on at L36 — early direction, late calibration [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:7,34; 001_layers_baseline/run-latest/output-Qwen3-14B.json:898–902]. Final‑head calibration is good (kl_to_final_bits=0.0; temp_est=1.0) [001_layers_baseline/run-latest/output-Qwen3-14B.json:959–967].

Lens sanity. Raw‑vs‑norm sampling flags lens_artifact_risk=high with max_kl_norm_vs_raw_bits≈17.67; first_norm_only_semantic_layer=null — treat any pre‑final “early semantics” cautiously; prefer rank‑based milestones [001_layers_baseline/run-latest/output-Qwen3-14B.json:1374–1378].

Temperature robustness. At T=0.1: “ Berlin, 0.9742” (entropy≈0.173 bits); at T=2.0: “ Berlin, 0.0363” (entropy≈13.161 bits) [001_layers_baseline/run-latest/output-Qwen3-14B.json:669–676,736–743].

- RMS lens? ✓ (RMSNorm model; use_norm_lens=true) [001_layers_baseline/run-latest/output-Qwen3-14B.json:807,810–811]
- LayerNorm bias removed? ✓ (layernorm_bias_fix=not_needed_rms_model) [001_layers_baseline/run-latest/output-Qwen3-14B.json:812]
- Entropy rise at unembed? ✓ (0.9521→3.5835 bits from L39→final) [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:41–42]
- FP32 un‑embed promoted? ✓ (unembed_dtype=torch.float32; mixed_precision_fix=casting_to_fp32_before_unembed) [001_layers_baseline/run-latest/output-Qwen3-14B.json:809,815]
- Punctuation / markup anchoring? ✓ ("____" dominates mid‑stack; punctuation in finals) [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:18–26,42]
- Copy‑reflex? ✗ (no strict or soft early hits) [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:2–5]
- Grammatical filler anchoring? ✗ (no early “is/the/a/of” top‑1 in L0–5) [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:2–7]

## 5. Limitations & data quirks
Rest_mass after L_semantic remains <0.3 (max 0.236 at final), indicating top‑k coverage rather than lens mis‑scale. KL is lens‑sensitive; despite final KL≈0 (good calibration), raw‑vs‑norm sampling shows lens_artifact_risk=high (max_kl_norm_vs_raw_bits≈17.67), so treat early “semantics” with caution and rely on rank milestones. Surface‑mass and rest_mass depend on tokenizer granularity; prefer within‑model trends.

## 6. Model fingerprint (one sentence)
Qwen‑3‑14B: semantics at L 36; final entropy 3.58 bits; early direction (cos≥0.2 at L5) with late calibration; Prism regressive.

---
Produced by OpenAI GPT-5 
