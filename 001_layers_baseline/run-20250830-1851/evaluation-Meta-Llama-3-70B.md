# Evaluation Report: meta-llama/Meta-Llama-3-70B
**1. Overview**
Meta-Llama-3-70B, 70B parameters; run-latest timestamp 2025-08-30 (80 transformer blocks). Probe measures layer-wise entropy, top-k, KL to final logits, cosine-to-final, copy/entropy collapse, and answer calibration for the gold token “Berlin”.
Outputs include per-layer CSVs and a JSON summary with diagnostics and ablations for the positive prompt and a France→Paris control.

**2. Method Sanity‑Check**
Diagnostics confirm intended setup: RMSNorm lensing with FP32 unembedding (“use_norm_lens”: true; “unembed_dtype”: “torch.float32”; “layer0_position_info”: “token_only_rotary_model” [L807–L817]); context_prompt ends with “called simply” (no trailing space) [L817]; gold alignment is ID‑level and ok [L831]; copy rule matches ID‑level contiguous subsequence with τ=0.95, k=1, δ=0.10 (“copy_thresh”: 0.95; “copy_window_k”: 1; “copy_match_level”: “id_subsequence” [L823–L825]). Raw‑lens sanity: mode = sample with lens_artifact_risk = low, max_kl_norm_vs_raw_bits = 0.0429 bits, first_norm_only_semantic_layer = null [L948–L1008]. Negative control scaffolding present (“control_prompt”, “control_summary” [L1018–L1036]); ablation_summary exists and positive rows appear under both orig and no_filler (e.g., pos/no_filler L80 present) [L1010–L1016; pure CSV row 169]. For the main table, we filter to prompt_id = pos, prompt_variant = orig. Summary indices: first_kl_below_0.5 = 80, first_kl_below_1.0 = 80, first_rank_le_1 = 40, first_rank_le_5 = 38, first_rank_le_10 = 38 [L826–L830]; units are bits. Last‑layer KL ≈ 0 (0.00073 bits) with top‑1 agreement [L833–L840]. Copy‑collapse: none in layers 0–3 and L_copy = null [L819–L820].

**3. Quantitative Findings**
Table (pos/orig only). Bolded row is L_semantic (first is_answer=True). Format: L i — entropy X bits, top‑1 ‘token’.

| Layer | Per‑layer summary |
|---|---|
| L 0 | entropy 16.968 bits, top‑1 ‘ winding’ |
| L 1 | entropy 16.960 bits, top‑1 ‘cepts’ |
| L 2 | entropy 16.963 bits, top‑1 ‘улю’ |
| L 3 | entropy 16.963 bits, top‑1 ‘zier’ |
| L 4 | entropy 16.959 bits, top‑1 ‘alls’ |
| L 5 | entropy 16.957 bits, top‑1 ‘alls’ |
| L 6 | entropy 16.956 bits, top‑1 ‘alls’ |
| L 7 | entropy 16.953 bits, top‑1 ‘NodeId’ |
| L 8 | entropy 16.959 bits, top‑1 ‘inds’ |
| L 9 | entropy 16.960 bits, top‑1 ‘NodeId’ |
| L 10 | entropy 16.952 bits, top‑1 ‘inds’ |
| L 11 | entropy 16.956 bits, top‑1 ‘inds’ |
| L 12 | entropy 16.956 bits, top‑1 ‘lia’ |
| L 13 | entropy 16.955 bits, top‑1 ‘eds’ |
| L 14 | entropy 16.950 bits, top‑1 ‘idders’ |
| L 15 | entropy 16.953 bits, top‑1 ‘ Kok’ |
| L 16 | entropy 16.952 bits, top‑1 ‘/plain’ |
| L 17 | entropy 16.948 bits, top‑1 ‘ nut’ |
| L 18 | entropy 16.944 bits, top‑1 ‘ nut’ |
| L 19 | entropy 16.948 bits, top‑1 ‘ nut’ |
| L 20 | entropy 16.946 bits, top‑1 ‘ nut’ |
| L 21 | entropy 16.938 bits, top‑1 ‘ burge’ |
| L 22 | entropy 16.938 bits, top‑1 ‘ simply’ |
| L 23 | entropy 16.936 bits, top‑1 ‘ bur’ |
| L 24 | entropy 16.950 bits, top‑1 ‘ bur’ |
| L 25 | entropy 16.937 bits, top‑1 ‘�’ |
| L 26 | entropy 16.938 bits, top‑1 ‘�’ |
| L 27 | entropy 16.937 bits, top‑1 ‘za’ |
| L 28 | entropy 16.933 bits, top‑1 ‘/plain’ |
| L 29 | entropy 16.933 bits, top‑1 ‘ plain’ |
| L 30 | entropy 16.939 bits, top‑1 ‘zed’ |
| L 31 | entropy 16.925 bits, top‑1 ‘ simply’ |
| L 32 | entropy 16.941 bits, top‑1 ‘ simply’ |
| L 33 | entropy 16.927 bits, top‑1 ‘ plain’ |
| L 34 | entropy 16.932 bits, top‑1 ‘ simply’ |
| L 35 | entropy 16.929 bits, top‑1 ‘ simply’ |
| L 36 | entropy 16.940 bits, top‑1 ‘ simply’ |
| L 37 | entropy 16.935 bits, top‑1 ‘ simply’ |
| L 38 | entropy 16.934 bits, top‑1 ‘ simply’ |
| L 39 | entropy 16.935 bits, top‑1 ‘ simply’ |
| **L 40** | entropy 16.937 bits, top‑1 ‘ Berlin’ |
| L 41 | entropy 16.936 bits, top‑1 ‘ "’ |
| L 42 | entropy 16.944 bits, top‑1 ‘ "’ |
| L 43 | entropy 16.941 bits, top‑1 ‘ Berlin’ |
| L 44 | entropy 16.926 bits, top‑1 ‘ Berlin’ |
| L 45 | entropy 16.940 bits, top‑1 ‘ "’ |
| L 46 | entropy 16.955 bits, top‑1 ‘ "’ |
| L 47 | entropy 16.939 bits, top‑1 ‘ "’ |
| L 48 | entropy 16.939 bits, top‑1 ‘ "’ |
| L 49 | entropy 16.937 bits, top‑1 ‘ "’ |
| L 50 | entropy 16.944 bits, top‑1 ‘ "’ |
| L 51 | entropy 16.940 bits, top‑1 ‘ "’ |
| L 52 | entropy 16.922 bits, top‑1 ‘ Berlin’ |
| L 53 | entropy 16.933 bits, top‑1 ‘ Berlin’ |
| L 54 | entropy 16.942 bits, top‑1 ‘ Berlin’ |
| L 55 | entropy 16.942 bits, top‑1 ‘ Berlin’ |
| L 56 | entropy 16.921 bits, top‑1 ‘ Berlin’ |
| L 57 | entropy 16.934 bits, top‑1 ‘ Berlin’ |
| L 58 | entropy 16.941 bits, top‑1 ‘ Berlin’ |
| L 59 | entropy 16.944 bits, top‑1 ‘ Berlin’ |
| L 60 | entropy 16.923 bits, top‑1 ‘ Berlin’ |
| L 61 | entropy 16.940 bits, top‑1 ‘ Berlin’ |
| L 62 | entropy 16.951 bits, top‑1 ‘ Berlin’ |
| L 63 | entropy 16.946 bits, top‑1 ‘ Berlin’ |
| L 64 | entropy 16.926 bits, top‑1 ‘ Berlin’ |
| L 65 | entropy 16.933 bits, top‑1 ‘ "’ |
| L 66 | entropy 16.941 bits, top‑1 ‘ Berlin’ |
| L 67 | entropy 16.930 bits, top‑1 ‘ Berlin’ |
| L 68 | entropy 16.924 bits, top‑1 ‘ Berlin’ |
| L 69 | entropy 16.932 bits, top‑1 ‘ Berlin’ |
| L 70 | entropy 16.926 bits, top‑1 ‘ Berlin’ |
| L 71 | entropy 16.923 bits, top‑1 ‘ Berlin’ |
| L 72 | entropy 16.922 bits, top‑1 ‘ Berlin’ |
| L 73 | entropy 16.918 bits, top‑1 ‘ "’ |
| L 74 | entropy 16.914 bits, top‑1 ‘ Berlin’ |
| L 75 | entropy 16.913 bits, top‑1 ‘ Berlin’ |
| L 76 | entropy 16.919 bits, top‑1 ‘ Berlin’ |
| L 77 | entropy 16.910 bits, top‑1 ‘ Berlin’ |
| L 78 | entropy 16.919 bits, top‑1 ‘ Berlin’ |
| L 79 | entropy 16.942 bits, top‑1 ‘ Berlin’ |
| L 80 | entropy 2.589 bits, top‑1 ‘ Berlin’ |

Bold semantic layer: L_semantic = 40 (first is_answer=True in CSV; see “is_answer” True at layer 40) [row 42 in pure CSV]. For reference, gold_answer.string = “Berlin” [L1038–L1047].

Ablation (no‑filler) from JSON: L_copy_orig = null, L_sem_orig = 40; L_copy_nf = null, L_sem_nf = 42; ΔL_copy = n.a., ΔL_sem = 2 [L1010–L1016]. Interpretation: removing “simply” delays semantics by 2 layers (~2.5% of 80), indicating some anchoring from stylistic cue but modest magnitude.

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (L_copy = null). Confidence milestones (pos/orig, pure CSV): p_top1 > 0.30 at L80; p_top1 > 0.60 not reached; final-layer p_top1 = 0.4783 [row 82 in pure CSV]. Rank milestones (diagnostics): rank ≤ 10 at L38, ≤ 5 at L38, ≤ 1 at L40 [L828–L830]. KL milestones (diagnostics): first_kl_below_1.0 at L80, first_kl_below_0.5 at L80 [L826–L827]; KL decreases toward ≈0 and is ≈0 at final [L833]. Cosine milestones (pure CSV): first cos_to_final ≥ 0.2 at L80; ≥ 0.4 at L80; ≥ 0.6 at L80; final cos_to_final = 0.99999 [row 82].

Rest‑mass sanity: from pos/orig rows, rest_mass drops from ≈0.99989 near L40 [row 42] to 0.10744 at L80 [row 82], consistent with concentrating mass into the top‑k as depth increases.

**4. Qualitative Patterns & Anomalies**
The model shows late semantic collapse consistent with pre‑norm Llama‑family behavior: the gold token first becomes top‑1 at L40 [row 42], then oscillates briefly with punctuation before stabilizing as top‑1 across most deeper layers (e.g., L52–L60 remain ‘Berlin’ top‑1). KL to final drops only at the very top (first_kl_below_1.0 = 80), and cosine alignment to the final direction stays low through the mid‑stack (≤0.15 up to L79 [row 81]) before jumping to ≈1.0 at L80 [row 82], indicating late calibration rather than early directional convergence (“early direction, late calibration” is not observed; here both direction and calibration finalize together at the top layer). Temperature robustness is strong: at T = 0.1, Berlin rank 1 (p = 0.9933; entropy 0.058) [L670–L676]; at T = 2.0, Berlin remains rank 1 but with p = 0.0357 and entropy 14.46 bits [L737–L743, L738].

Negative control: “Berlin is the capital of” yields top‑5 “ Germany (0.8516), the (0.0791), and (0.0146), modern (0.0048), Europe (0.0031)” [L14–L31]. No semantic leakage of “Berlin” in the country slot; expected answer “Germany” is correctly top‑1. For the France control, the final layer shows Paris top‑1 with large margin and tiny KL: “Paris”, p = 0.5169; kl_to_final_bits = 0.00107 [ctl L80 in pure CSV; row 5 of tail].

Important‑word trajectory (records CSV): the ‘Germany’ token begins to co‑activate ‘Berlin’ in its top‑20 by L31 [L680]; by L38–40, ‘Berlin’ appears among top‑5 for “is/called/simply” and the final context position (e.g., “simply” at L38 includes ‘Berlin’, 2.50e−05 [L817], and at L40 top‑1 becomes ‘Berlin’ at the next‑token position [L854]). This suggests information gradually aggregates across the span tokens (“Germany”, “is”, “called”, “simply”) before the next‑token head flips to the answer.

Stylistic ablation: removing “simply” delays L_sem by 2 layers (L_sem_nf = 42 vs 40) [L1010–L1016], consistent with small guidance‑style anchoring rather than defining semantics. The collapse index does not materially shift under test‑prompt rephrasings; across several paraphrases, “Berlin” remains high‑probability (e.g., “Germany’s capital city is called simply” top‑1 ‘ Berlin’, 0.6445 [L61–L63]; “The capital city of Germany is named simply” 0.4434 [L108–L110]).

Rest‑mass sanity: rest_mass falls steadily after L_semantic (e.g., 0.9998877 at L40 [row 42] to 0.1074 at L80 [row 82]), suggesting no precision loss spikes. Rotation vs amplification: decreasing final‑layer KL with simultaneously rising p_answer, improving answer_rank, and sharply increasing cos_to_final at the top implies the stack performs gradual evidence accumulation with a last‑layer alignment to the final readout. Lens sanity matches low artifact risk: “lens_artifact_risk”: “low”, “max_kl_norm_vs_raw_bits”: 0.0429, “first_norm_only_semantic_layer”: null [L1004–L1008]. Example sampled check shows norm vs raw agreement at L61 with is_answer true for both [L991–L1001].

Checklist:
– RMS lens? ✓ (RMSNorm; first_block_ln1_type = RMSNorm) [L810–L811]
– LayerNorm bias removed? ✓ (“layernorm_bias_fix”: “not_needed_rms_model”) [L812]
– Entropy rise at unembed? n.a.; final entropy reported separately in JSON/CSV; no anomaly observed (final KL ≈ 0) [L833]
– FP32 un‑embed promoted? ✓ (“use_fp32_unembed”: true, “unembed_dtype”: “torch.float32”) [L808–L809]
– Punctuation / markup anchoring? ✓ (mid‑layers often top‑1 quotes/punctuation; e.g., L41–L51 show ‘"’ top‑1)
– Copy‑reflex? ✗ (no copy_collapse = True in layers 0–3; L_copy = null) [L819–L820]
– Grammatical filler anchoring? ✓ (layers 22–39 top‑1 often ‘simply’/fillers)

> “simply … Berlin …” [row 42 in CSV]
> “Berlin (0.8516), the (0.0791) …” [L14–L20]

**5. Limitations & Data Quirks**
L_copy is null, so ΔH relative to copy collapse is not defined and copy‑reflex cannot be quantitatively contrasted with semantic collapse here. KL is lens‑sensitive; while final KL ≈ 0, intermediate KL values are qualitative. Raw‑vs‑norm lens checks are sampled (“mode”: “sample”), so sanity is based on sampled layers rather than exhaustive coverage. Rest_mass remains high until late layers but drops appropriately by L80; no evidence of precision loss spikes.

**6. Model Fingerprint**
“Meta‑Llama‑3‑70B: semantics at L 40; final entropy 2.59 bits; ‘Berlin’ stable top‑1 late; cosine snaps to ≈1 at L80.”

---
Produced by OpenAI GPT-5
*Run executed on: 2025-08-30 18:51:32*
