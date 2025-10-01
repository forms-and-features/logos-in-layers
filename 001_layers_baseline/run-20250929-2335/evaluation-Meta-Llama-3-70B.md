**Overview**

- Model: meta-llama/Meta-Llama-3-70B (RMSNorm, rotary). Run date: 2025-09-30.
- Probe captures layer-by-layer NEXT-token behavior under a norm lens, with copy/collapse flags, ranks, KL-to-final, cosine trajectories, control prompt, and ablation.
- Final head alignment is excellent (KL≈0), and semantics emerge mid‑stack (L≈40) under the norm lens.

**Method Sanity‑Check**

Diagnostics confirm RMS norm lens usage, FP32 unembed, and rotary pos encodings: "use_norm_lens": true; "unembed_dtype": "torch.float32"; "first_block_ln1_type": "RMSNorm"; "layer0_position_info": "token_only_rotary_model" (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:807,809–811,816). Context ends with “called simply”: "context_prompt": "… is called simply" (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:816–819).
Copy detector config and metrics are present: "copy_thresh": 0.95, "copy_window_k": 1, "copy_match_level": "id_subsequence"; soft: "copy_soft_threshold": 0.5, "copy_soft_window_ks": [1,2,3]; L_copy = null, L_semantic = 40, delta_layers = null; L_copy_soft and delta_layers_soft per‑k present (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:975–986,990–995,1000–1021). Copy flag columns mirror labels: ["copy_strict@0.95","copy_strict@0.7","copy_strict@0.8","copy_strict@0.9","copy_soft_k1@0.5","copy_soft_k2@0.5","copy_soft_k3@0.5"] (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1310–1318; 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:1).
Gold alignment OK and ID‑based: "gold_alignment": "ok"; gold answer first_id = 20437 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1113,1349–1356). Control present with summary: first_control_margin_pos=0; max_control_margin=0.5168 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1342–1346).
Ablation present: L_sem_orig=40 → L_sem_nf=42 (ΔL_sem=2); L_copy_* all null (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1319–1330). Summary indices: first_kl_below_1.0=80; first_kl_below_0.5=80; first_rank_le_{10,5,1}={38,38,40} (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:979–983). Units are bits (teacher_entropy_bits and KL columns in pure CSV; 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:1).
Last‑layer head calibration: KL≈0 and agreement: "kl_to_final_bits": 0.0007293; "top1_agree": true; "p_top1_lens": 0.4783 vs "p_top1_model": 0.4690; warn_high_last_layer_kl=false (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1115–1118,1132). Measurement guidance: prefer ranks, suppress absolute probs: {"prefer_ranks": true, "suppress_abs_probs": true, reasons=["norm_only_semantics_window"]} (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1351–1356).
Raw‑vs‑Norm window: radius=4, centers=[38,40,80], norm_only_semantics_layers=[79,80], max_kl_norm_vs_raw_bits_window=1.2437 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1122–1131). Overall raw‑lens check: lens_artifact_risk="low", max_kl_norm_vs_raw_bits=0.0429, first_norm_only_semantic_layer=null (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1298–1308).
Strict copy flags: none in L0–L3; soft k1@0.5 also absent in L0–L3 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:2–5).

**Quantitative Findings**

Table (prompt_id=pos, prompt_variant=orig). Bold marks L_semantic (first is_answer=True).

| Layer | Summary |
|---|---|
| L 0 | entropy 16.968 bits, top - 1 ' winding' |
| L 1 | entropy 16.960 bits, top - 1 'cepts' |
| L 2 | entropy 16.963 bits, top - 1 'улю' |
| L 3 | entropy 16.963 bits, top - 1 'zier' |
| L 4 | entropy 16.959 bits, top - 1 'alls' |
| L 5 | entropy 16.957 bits, top - 1 'alls' |
| L 6 | entropy 16.956 bits, top - 1 'alls' |
| L 7 | entropy 16.953 bits, top - 1 'NodeId' |
| L 8 | entropy 16.959 bits, top - 1 'inds' |
| L 9 | entropy 16.960 bits, top - 1 'NodeId' |
| L 10 | entropy 16.952 bits, top - 1 'inds' |
| L 11 | entropy 16.956 bits, top - 1 'inds' |
| L 12 | entropy 16.956 bits, top - 1 'lia' |
| L 13 | entropy 16.955 bits, top - 1 'eds' |
| L 14 | entropy 16.950 bits, top - 1 'idders' |
| L 15 | entropy 16.953 bits, top - 1 ' Kok' |
| L 16 | entropy 16.952 bits, top - 1 '/plain' |
| L 17 | entropy 16.948 bits, top - 1 ' nut' |
| L 18 | entropy 16.944 bits, top - 1 ' nut' |
| L 19 | entropy 16.948 bits, top - 1 ' nut' |
| L 20 | entropy 16.946 bits, top - 1 ' nut' |
| L 21 | entropy 16.938 bits, top - 1 ' burge' |
| L 22 | entropy 16.938 bits, top - 1 ' simply' |
| L 23 | entropy 16.936 bits, top - 1 ' bur' |
| L 24 | entropy 16.950 bits, top - 1 ' bur' |
| L 25 | entropy 16.937 bits, top - 1 '�' |
| L 26 | entropy 16.938 bits, top - 1 '�' |
| L 27 | entropy 16.937 bits, top - 1 'za' |
| L 28 | entropy 16.933 bits, top - 1 '/plain' |
| L 29 | entropy 16.933 bits, top - 1 ' plain' |
| L 30 | entropy 16.939 bits, top - 1 'zed' |
| L 31 | entropy 16.925 bits, top - 1 ' simply' |
| L 32 | entropy 16.941 bits, top - 1 ' simply' |
| L 33 | entropy 16.927 bits, top - 1 ' plain' |
| L 34 | entropy 16.932 bits, top - 1 ' simply' |
| L 35 | entropy 16.929 bits, top - 1 ' simply' |
| L 36 | entropy 16.940 bits, top - 1 ' simply' |
| L 37 | entropy 16.935 bits, top - 1 ' simply' |
| L 38 | entropy 16.934 bits, top - 1 ' simply' |
| L 39 | entropy 16.935 bits, top - 1 ' simply' |
| **L 40 | entropy 16.937 bits, top - 1 ' Berlin'** |
| L 41 | entropy 16.936 bits, top - 1 ' "' |
| L 42 | entropy 16.944 bits, top - 1 ' "' |
| L 43 | entropy 16.941 bits, top - 1 ' Berlin' |
| L 44 | entropy 16.926 bits, top - 1 ' Berlin' |
| L 45 | entropy 16.940 bits, top - 1 ' "' |
| L 46 | entropy 16.955 bits, top - 1 ' "' |
| L 47 | entropy 16.939 bits, top - 1 ' "' |
| L 48 | entropy 16.939 bits, top - 1 ' "' |
| L 49 | entropy 16.937 bits, top - 1 ' "' |
| L 50 | entropy 16.944 bits, top - 1 ' "' |
| L 51 | entropy 16.940 bits, top - 1 ' "' |
| L 52 | entropy 16.922 bits, top - 1 ' Berlin' |
| L 53 | entropy 16.933 bits, top - 1 ' Berlin' |
| L 54 | entropy 16.942 bits, top - 1 ' Berlin' |
| L 55 | entropy 16.942 bits, top - 1 ' Berlin' |
| L 56 | entropy 16.921 bits, top - 1 ' Berlin' |
| L 57 | entropy 16.934 bits, top - 1 ' Berlin' |
| L 58 | entropy 16.941 bits, top - 1 ' Berlin' |
| L 59 | entropy 16.944 bits, top - 1 ' Berlin' |
| L 60 | entropy 16.923 bits, top - 1 ' Berlin' |
| L 61 | entropy 16.940 bits, top - 1 ' Berlin' |
| L 62 | entropy 16.951 bits, top - 1 ' Berlin' |
| L 63 | entropy 16.946 bits, top - 1 ' Berlin' |
| L 64 | entropy 16.926 bits, top - 1 ' Berlin' |
| L 65 | entropy 16.933 bits, top - 1 ' "' |
| L 66 | entropy 16.941 bits, top - 1 ' Berlin' |
| L 67 | entropy 16.930 bits, top - 1 ' Berlin' |
| L 68 | entropy 16.924 bits, top - 1 ' Berlin' |
| L 69 | entropy 16.932 bits, top - 1 ' Berlin' |
| L 70 | entropy 16.926 bits, top - 1 ' Berlin' |
| L 71 | entropy 16.923 bits, top - 1 ' Berlin' |
| L 72 | entropy 16.922 bits, top - 1 ' Berlin' |
| L 73 | entropy 16.918 bits, top - 1 ' "' |
| L 74 | entropy 16.914 bits, top - 1 ' Berlin' |
| L 75 | entropy 16.913 bits, top - 1 ' Berlin' |
| L 76 | entropy 16.919 bits, top - 1 ' Berlin' |
| L 77 | entropy 16.910 bits, top - 1 ' Berlin' |
| L 78 | entropy 16.919 bits, top - 1 ' Berlin' |
| L 79 | entropy 16.942 bits, top - 1 ' Berlin' |
| L 80 | entropy 2.589 bits, top - 1 ' Berlin' |

Control margin (JSON): first_control_margin_pos = 0; max_control_margin = 0.5168 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1342–1346).

Ablation (no‑filler): L_copy_orig = null; L_sem_orig = 40; L_copy_nf = null; L_sem_nf = 42; ΔL_copy = null; ΔL_sem = 2 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1319–1330). Interpretation: ΔL_sem=+2 (≈2.5% of depth) suggests mild sensitivity to the stylistic cue “simply”.

ΔH (bits) = n/a (L_copy_strict = null). Soft ΔHk (k∈{1,2,3}) = n/a (all L_copy_soft[k] = null).

Confidence milestones (pure CSV, pos/orig; note guidance prefers ranks): p_top1 > 0.30 at L 80; p_top1 > 0.60 not reached; final-layer p_top1 = 0.4783 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:82).

Rank milestones (JSON): rank ≤ 10 at L 38; rank ≤ 5 at L 38; rank ≤ 1 at L 40 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:981–983).

KL milestones (JSON): first_kl_below_1.0 at L 80; first_kl_below_0.5 at L 80; KL decreases toward ≈0 at final and is ≈0 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:979–980,1115).

Cosine milestones (JSON): first cos_to_final ≥0.2 at L 80; ≥0.4 at L 80; ≥0.6 at L 80; final cos_to_final ≈ 1.0000 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1054–1061; 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:82).

Copy robustness (threshold sweep): stability = "none"; earliest strict L_copy at τ∈{0.70,0.95} = null; norm_only_flags at τ∈{0.70,0.95} = null (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1028–1079).

Prism sidecar: compatible=true (k=512; layers=[embed,19,39,59]) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:867–878). Early‑depth KL is worse vs baseline (Δp25≈−0.89 bits; Δp50≈−1.00; Δp75≈−1.16); rank milestones do not appear (null) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:893–919). Copy flags do not spuriously flip (no strict nor soft hits in prism pure CSV; e.g., final row shows False; 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token-prism.csv:82). Verdict: Regressive.

**Qualitative Patterns & Anomalies**

Negative control shows no leakage of “Berlin”: top‑5 for “Berlin is the capital of” are [" Germany" (0.85), " the" (0.08), " and" (0.015), " modern" (0.0048), " Europe" (0.0031)] (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:10–32). No temperature exploration block was emitted in this run.
In the "records" stream at semantic onset, the NEXT distribution begins to place both the answer and semantically related tokens in the top‑k: at L 40, top‑k includes "Berlin", "Germany", and "city" among punctuation (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-records.csv:854). Earlier layers (L34–39) are dominated by prompt echo tokens like “ simply” before rank‑1 flips to "Berlin" (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token-rawlens-window.csv:38–44). The collapse‑layer index does not advance without the “one‑word” cue; ablation nudges semantics later by 2 layers (ΔL_sem=+2; 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1319–1330).
Rest‑mass sanity: top‑k coverage is tiny mid‑stack and only improves near the end; rest_mass peaks after L_semantic around 0.9999 (e.g., L 46) and drops to 0.107 at L 80 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:46,82). This reflects sharpening rather than a lens glitch; we rely on KL and ranks for fidelity.
Rotation vs amplification: KL(P_layer||P_final) decreases monotonically late and is ≈0 at L 80 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1115), while answer rank improves to 1 by L 40 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:981). Cos_to_final crosses thresholds only at the very end (ge_{0.2,0.4,0.6}=80; 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1054–1061), indicating “early direction, late calibration” is not pronounced here; alignment consolidates late.
Head calibration: final‑layer KL is low and warn_high_last_layer_kl=false, temp_est=1.0 (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1115–1132). We thus treat final probabilities as calibrated within‑model; cross‑family comparisons still prefer rank thresholds per guidance (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1351–1356).
Lens sanity: global raw‑vs‑norm risk is low (max_kl_norm_vs_raw_bits=0.0429; 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1304–1308). However, windowed checks show norm‑only semantics at L 79–80 with elevated KL windows (1.2437 bits), so we prefer rank milestones over absolute probabilities near the top (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1122–1131; 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token-rawlens-window.csv:76–80).
Prism sidecar is regressive: at L 80, prism decodes a peaked but unrelated token (p_top1≈1.0 for “oldt”) with huge KL to the final head (26.88 bits), while baseline lens aligns with the model head and answer (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token-prism.csv:82; 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:82). This supports treating Prism as a diagnostic, not a replacement head.

Checklist:
✓ RMS lens  (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:810–811)
✓ LayerNorm bias removed/not needed (RMS)  (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:812)
✓ Entropy fall to final head  (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:82)
✓ FP32 un‑embed promoted  (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:808–809)
✓ Punctuation / filler anchoring (early layers show fillers/punct.)  (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-records.csv:803–810)
✗ Copy‑reflex (no strict/soft hits in L0–3)  (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:2–5)
~ Grammatical filler anchoring (prompt echo “ simply” dominates L34–39)  (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token-rawlens-window.csv:34–39)

**Limitations & Data Quirks**

- Measurement guidance advises rank‑first and suppressing absolute probabilities due to norm‑only semantics near the top (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1351–1356,1126–1131).
- Raw‑lens sanity is sampled (mode="sample"); treat the raw‑vs‑norm summary as a sampled check, not exhaustive (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1248–1251).
- Rest_mass reflects top‑k coverage only and is high mid‑stack; rely on KL/rank for fidelity (001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:46,82).
- Prism behaves regressively here; use it as a diagnostic comparator, not a teacher for this model family (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:893–919; prism CSV final row: 82).

**Model Fingerprint**

“Llama‑3‑70B: collapse at L 40; final entropy 2.59 bits; Prism regressive; control margin peaks 0.52 mid‑stack.”

---
Produced by OpenAI GPT-5 
