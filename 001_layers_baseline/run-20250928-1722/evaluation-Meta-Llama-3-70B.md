# Evaluation Report: meta-llama/Meta-Llama-3-70B
1. Overview

Meta‑Llama‑3‑70B; run timestamp 2025‑09‑28 (see `001_layers_baseline/run-latest/timestamp-20250928-1722`). This probe captures layer‑by‑layer next‑token behavior under a norm lens, tracking copy vs semantic collapse, calibration (KL in bits), geometry (cosine to final), and control/ablation effects.

2. Method sanity‑check

Norm lens and FP32 un‑embed are active with RMSNorm: `use_norm_lens=True`, `use_fp32_unembed=True`, `unembed_dtype=torch.float32`, `first_block_ln1_type=RMSNorm`, `final_ln_type=RMSNorm`, `layernorm_bias_fix=not_needed_rms_model`, `layer0_position_info=token_only_rotary_model`, `norm_alignment_fix=using_ln2_rmsnorm_for_post_block` (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json). The context prompt ends with “called simply” with no trailing space as defined in script (“Give the city name only, plain text. The capital of Germany is called simply”) (001_layers_baseline/run.py:232).

Copy detector configuration is present and mirrored in CSV/JSON: `copy_thresh=0.95`, `copy_window_k=1`, `copy_match_level=id_subsequence`, soft `threshold=0.5`, `window_ks=[1,2,3]`, `extra_thresholds=[]`; flags `['copy_strict@0.95','copy_soft_k1@0.5','copy_soft_k2@0.5','copy_soft_k3@0.5']` (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json). Diagnostics include `L_copy`, `L_copy_H`, `L_semantic`, `delta_layers`, `L_copy_soft`, `delta_layers_soft` (same JSON). Gold alignment is `ok` with ID‑level answer “Berlin” (`first_id=20437`, `variant=with_space`), and control prompt/summary are present: `first_control_margin_pos=0`, `max_control_margin=0.5168457566906` (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json).

Units are bits for both entropy and KL (`entropy` fields; `kl_to_final_bits`). Final‑head calibration is clean: `last_layer_consistency.kl_to_final_bits=0.000729`, `top1_agree=True`, `p_top1_lens=0.4783` vs `p_top1_model=0.4690`, `temp_est=1.0`, `warn_high_last_layer_kl=False` (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json). Lens sanity: `raw_lens_check.summary.lens_artifact_risk=low`, `max_kl_norm_vs_raw_bits=0.0429`, `first_norm_only_semantic_layer=None` (same JSON). Strict and soft copy flags do not fire in layers 0–3 (see Section 4). Ablation summary exists with both `orig` and `no_filler` variants recorded (see Section 3). Summary indices: `first_kl_below_0.5=80`, `first_kl_below_1.0=80`, `first_rank_le_1=40`, `first_rank_le_5=38`, `first_rank_le_10=38` (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json).

3. Quantitative findings

Table below uses only positive rows (`prompt_id=pos`, `prompt_variant=orig`) from `001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv`. Bold marks the first semantic layer (`is_answer=True`).

| Layer summary |
|---|
| L 0 — entropy 16.968 bits, top‑1 ' winding' |
| L 1 — entropy 16.960 bits, top‑1 'cepts' |
| L 2 — entropy 16.963 bits, top‑1 'улю' |
| L 3 — entropy 16.963 bits, top‑1 'zier' |
| L 4 — entropy 16.959 bits, top‑1 'alls' |
| L 5 — entropy 16.957 bits, top‑1 'alls' |
| L 6 — entropy 16.956 bits, top‑1 'alls' |
| L 7 — entropy 16.953 bits, top‑1 'NodeId' |
| L 8 — entropy 16.959 bits, top‑1 'inds' |
| L 9 — entropy 16.960 bits, top‑1 'NodeId' |
| L 10 — entropy 16.952 bits, top‑1 'inds' |
| L 11 — entropy 16.956 bits, top‑1 'inds' |
| L 12 — entropy 16.956 bits, top‑1 'lia' |
| L 13 — entropy 16.955 bits, top‑1 'eds' |
| L 14 — entropy 16.950 bits, top‑1 'idders' |
| L 15 — entropy 16.953 bits, top‑1 ' Kok' |
| L 16 — entropy 16.952 bits, top‑1 '/plain' |
| L 17 — entropy 16.948 bits, top‑1 ' nut' |
| L 18 — entropy 16.944 bits, top‑1 ' nut' |
| L 19 — entropy 16.948 bits, top‑1 ' nut' |
| L 20 — entropy 16.946 bits, top‑1 ' nut' |
| L 21 — entropy 16.938 bits, top‑1 ' burge' |
| L 22 — entropy 16.938 bits, top‑1 ' simply' |
| L 23 — entropy 16.936 bits, top‑1 ' bur' |
| L 24 — entropy 16.950 bits, top‑1 ' bur' |
| L 25 — entropy 16.937 bits, top‑1 '�' |
| L 26 — entropy 16.938 bits, top‑1 '�' |
| L 27 — entropy 16.937 bits, top‑1 'za' |
| L 28 — entropy 16.933 bits, top‑1 '/plain' |
| L 29 — entropy 16.933 bits, top‑1 ' plain' |
| L 30 — entropy 16.939 bits, top‑1 'zed' |
| L 31 — entropy 16.925 bits, top‑1 ' simply' |
| L 32 — entropy 16.941 bits, top‑1 ' simply' |
| L 33 — entropy 16.927 bits, top‑1 ' plain' |
| L 34 — entropy 16.932 bits, top‑1 ' simply' |
| L 35 — entropy 16.929 bits, top‑1 ' simply' |
| L 36 — entropy 16.940 bits, top‑1 ' simply' |
| L 37 — entropy 16.935 bits, top‑1 ' simply' |
| L 38 — entropy 16.934 bits, top‑1 ' simply' |
| L 39 — entropy 16.935 bits, top‑1 ' simply' |
| **L 40 — entropy 16.937 bits, top‑1 ' Berlin'** |
| L 41 — entropy 16.936 bits, top‑1 ' "' |
| L 42 — entropy 16.944 bits, top‑1 ' "' |
| L 43 — entropy 16.941 bits, top‑1 ' Berlin' |
| L 44 — entropy 16.926 bits, top‑1 ' Berlin' |
| L 45 — entropy 16.940 bits, top‑1 ' "' |
| L 46 — entropy 16.955 bits, top‑1 ' "' |
| L 47 — entropy 16.939 bits, top‑1 ' "' |
| L 48 — entropy 16.939 bits, top‑1 ' "' |
| L 49 — entropy 16.937 bits, top‑1 ' "' |
| L 50 — entropy 16.944 bits, top‑1 ' "' |
| L 51 — entropy 16.940 bits, top‑1 ' "' |
| L 52 — entropy 16.922 bits, top‑1 ' Berlin' |
| L 53 — entropy 16.933 bits, top‑1 ' Berlin' |
| L 54 — entropy 16.942 bits, top‑1 ' Berlin' |
| L 55 — entropy 16.942 bits, top‑1 ' Berlin' |
| L 56 — entropy 16.921 bits, top‑1 ' Berlin' |
| L 57 — entropy 16.934 bits, top‑1 ' Berlin' |
| L 58 — entropy 16.941 bits, top‑1 ' Berlin' |
| L 59 — entropy 16.944 bits, top‑1 ' Berlin' |
| L 60 — entropy 16.923 bits, top‑1 ' Berlin' |
| L 61 — entropy 16.940 bits, top‑1 ' Berlin' |
| L 62 — entropy 16.951 bits, top‑1 ' Berlin' |
| L 63 — entropy 16.946 bits, top‑1 ' Berlin' |
| L 64 — entropy 16.926 bits, top‑1 ' Berlin' |
| L 65 — entropy 16.933 bits, top‑1 ' "' |
| L 66 — entropy 16.941 bits, top‑1 ' Berlin' |
| L 67 — entropy 16.930 bits, top‑1 ' Berlin' |
| L 68 — entropy 16.924 bits, top‑1 ' Berlin' |
| L 69 — entropy 16.932 bits, top‑1 ' Berlin' |
| L 70 — entropy 16.926 bits, top‑1 ' Berlin' |
| L 71 — entropy 16.923 bits, top‑1 ' Berlin' |
| L 72 — entropy 16.922 bits, top‑1 ' Berlin' |
| L 73 — entropy 16.918 bits, top‑1 ' "' |
| L 74 — entropy 16.914 bits, top‑1 ' Berlin' |
| L 75 — entropy 16.913 bits, top‑1 ' Berlin' |
| L 76 — entropy 16.919 bits, top‑1 ' Berlin' |
| L 77 — entropy 16.910 bits, top‑1 ' Berlin' |
| L 78 — entropy 16.919 bits, top‑1 ' Berlin' |
| L 79 — entropy 16.942 bits, top‑1 ' Berlin' |
| L 80 — entropy 2.589 bits, top‑1 ' Berlin' |

Control margin (JSON `control_summary`): first_control_margin_pos = 0; max_control_margin = 0.516846.

Ablation (no‑filler) (JSON `ablation_summary`):
- L_copy_orig = None, L_sem_orig = 40
- L_copy_nf = None, L_sem_nf = 42
- ΔL_copy = None, ΔL_sem = 2

ΔH (bits) = n/a (strict copy layer not detected).
Soft ΔHₖ (bits) = n/a (no soft‑copy layers detected at k ∈ {1,2,3}).

Confidence milestones (pure CSV):
- p_top1 > 0.30 at layer 80; p_top1 > 0.60 at layer n/a; final‑layer p_top1 = 0.478.

Rank milestones (JSON diagnostics):
- rank ≤ 10 at layer 38; rank ≤ 5 at layer 38; rank ≤ 1 at layer 40.

KL milestones (JSON diagnostics and pure CSV):
- first_kl_below_1.0 at layer 80; first_kl_below_0.5 at layer 80; final `kl_to_final_bits` ≈ 0 (0.000729).

Cosine milestones (pure CSV):
- cos_to_final ≥ 0.2 at layer 80; ≥ 0.4 at layer 80; ≥ 0.6 at layer 80; final cos_to_final = 0.999989.

Prism Sidecar Analysis
- Presence: compatible=True (k=512; layers=['embed', 19, 39, 59]) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json).
- Early‑depth snapshots (baseline vs Prism): L0 KL 10.502 → 10.667; L20 KL 10.450 → 11.344; L40 KL 10.420 → 11.420; L60 KL 10.310 → 11.468 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv, 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token-prism.csv).
- Rank milestones (Prism pure CSV): first_rank≤10 = n/a; ≤5 = n/a; ≤1 = n/a (later than baseline’s 38/38/40).
- Top‑1 agreement at sampled depths: baseline rank is already 1 by L40; Prism remains far (rank 1723 at L40; 1227 at L60). Cosine to final is also lower under Prism at these depths (e.g., L40 0.097 → 0.003).
- Verdict: Regressive (higher KL at early/mid layers and later rank milestones).

4. Qualitative patterns & anomalies

Strict copy‑collapse and soft‑copy flags do not fire in L0–L3 (“copy‑reflex” not observed); `L_copy=None`, `L_copy_soft[k]=None` in diagnostics and all early per‑layer flags are False in the pure CSV. Negative control shows no semantic leakage: “Berlin is the capital of” → “ Germany” (0.8516), “ the” (0.0791), “ and” (0.0146), “ modern” (0.0048), “ Europe” (0.0031) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json). Temperature robustness: at T = 0.1, Berlin rank 1 (p = 0.993; entropy 0.058 bits); at T = 2.0, Berlin rank 1 (p = 0.036; entropy 14.464 bits) (same JSON).

Important‑word trajectory (records CSV). Berlin enters prompt‑adjacent top‑5 near the collapse onset: e.g., at layer 38 (token “ is”), “Berlin” appears with p ≈ 3.26e‑05 (> “… (‘Berlin’, 3.261e‑05)” 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-records.csv:815). At the NEXT position the first semantic hit is L40 with very low mass (top‑1 “Berlin” with p ≈ 2.392e‑05; “… (‘Berlin’, 2.39e‑05)” 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:42). “Germany” remains a frequent surface token in adjacent positions through L38–39 (e.g., records.csv:814), while punctuation quotes (“"”, “’”) dominate top‑1 mid‑stack before the final head calibrates strongly at L80.

Stylistic ablation shifts semantics later by ΔL_sem = +2 (40 → 42), consistent with weak dependence on the adverb “simply” rather than core knowledge change (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json). Rest‑mass sanity: Rest_mass falls late; max after L_semantic = 0.9999 (e.g., L60 = 0.9998), consistent with narrow top‑k coverage in mid‑stack rather than fidelity loss. Rotation vs amplification: KL to final decreases only at the very end (first below 1.0 at L80) while `cos_to_final` spikes only at L80; combined with rising `p_answer`/rank improvements around L38–40, this suggests “early direction, late calibration” concentrated in the final block. Final‑head calibration shows no anomaly (`warn_high_last_layer_kl=False`; `temp_est=1.0`). Lens sanity reports `lens_artifact_risk=low` and no “norm‑only semantics”.

Checklist
✓ RMS lens
✓ LayerNorm bias removed (RMS; not needed)
✓ Entropy rise at unembed
✓ FP32 un‑embed promoted
✓ Punctuation / markup anchoring
✗ Copy‑reflex
✓ Grammatical filler anchoring ("simply" often top‑1 pre‑collapse)

5. Limitations & data quirks

Rest_mass > 0.3 persists well past L_semantic (e.g., L60 ≈ 0.9998), indicating narrow top‑k coverage at mid‑depth; treat KL/entropy trends qualitatively. KL is lens‑sensitive; despite near‑zero final KL, mid‑depth KL remains high—prefer rank milestones for cross‑model claims. `raw_lens_check` uses sampled norms (mode reported in JSON), so findings are within‑model sanity rather than exhaustive. Surface‑mass relies on tokenizer granularity; avoid cross‑model comparisons of absolute masses.

6. Model fingerprint

“Meta‑Llama‑3‑70B: collapse at L 40; final entropy 2.6 bits; ‘Berlin’ top‑1 crystallizes late (final block).”

---
Produced by OpenAI GPT-5 
*Run executed on: 2025-09-28 17:22:48*
