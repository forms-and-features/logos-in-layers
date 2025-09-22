## 1. Overview
Mistral-Small-24B-Base-2501 (pre-norm; 40 transformer blocks) probed with the norm lens for layer-wise token predictions, entropy, KL-to-final and calibration. The run targets a single-position query: “Give the city name only… The capital of Germany is called simply …” and tracks when the gold token “Berlin” becomes rank-1.
Model identifier: "mistralai/Mistral-Small-24B-Base-2501" [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:805].

## 2. Method sanity-check
Intended lens and positions are confirmed: "use_norm_lens": true [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:807]; positional encoding noted as "layer0_position_info": "token_only_rotary_model" [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:816]. Context prompt ends with “called simply” (no trailing space) in both prompt and diagnostics: "context_prompt": "… is called simply" [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:4,817].

Implementation flags present: "first_block_ln1_type": "RMSNorm", "final_ln_type": "RMSNorm", "layernorm_bias_fix": "not_needed_rms_model", "unembed_dtype": "torch.float32" [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:810-812,809]. Copy-rule parameters are logged: "copy_thresh": 0.95, "copy_window_k": 1, "copy_match_level": "id_subsequence" [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:837-839]. Gold alignment is OK for both prompts: "gold_alignment": "ok" [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:845,1045]. Negative control is present: "control_prompt" and "control_summary" (first_control_margin_pos=1, max_control_margin=0.4679627253462968) [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1047-1049]. Ablation summary exists with both variants: {"L_sem_orig": 33, "L_sem_nf": 31} [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1026,1028]; corresponding rows appear in CSV for prompt_variant=orig and no_filler (e.g., lines starting with "pos,orig,…" and "pos,no_filler,…").

Summary indices: "first_kl_below_0.5": 40, "first_kl_below_1.0": 40, "first_rank_le_1": 33, "first_rank_le_5": 30, "first_rank_le_10": 30 (KL units: bits) [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:840-844]. Final-head calibration is clean: last-layer consistency shows "kl_to_final_bits": 0.0, "top1_agree": true, "temp_est": 1.0 [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:846-859].

Lens sanity (raw vs norm): mode "sample"; "lens_artifact_risk": "low"; "max_kl_norm_vs_raw_bits": 0.1792677499620318; "first_norm_only_semantic_layer": null [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1019-1022].

Copy-collapse flag check (τ=0.95, δ=0.10). No rows in layers 0–3 have copy_collapse=True in the pure CSV (search over pos,orig, layers 0–3 shows all False). Therefore no copy-reflex firing; rule not triggered.

## 3. Quantitative findings
Main position only (prompt_id=pos, prompt_variant=orig). Table shows top-1 token per layer and entropy in bits from the pure-next-token CSV.

| Layer | Entropy (bits) | Top‑1 token |
|---:|---:|:---|
| 0 | 16.9985 | ' Forbes' |
| 1 | 16.9745 | 随着时间的 |
| 2 | 16.9441 | 随着时间的 |
| 3 | 16.8120 | 随着时间的 |
| 4 | 16.8682 |  quelcon |
| 5 | 16.9027 | народ |
| 6 | 16.9087 | народ |
| 7 | 16.8978 | народ |
| 8 | 16.8955 |  quelcon |
| 9 | 16.8852 |  simply |
| 10 | 16.8359 |  hétérogènes |
| 11 | 16.8423 | 从那以后 |
| 12 | 16.8401 |  simply |
| 13 | 16.8709 |  simply |
| 14 | 16.8149 | стен |
| 15 | 16.8164 | luš |
| 16 | 16.8300 | luš |
| 17 | 16.7752 | luš |
| 18 | 16.7608 | luš |
| 19 | 16.7746 | luš |
| 20 | 16.7424 | luš |
| 21 | 16.7747 |  simply |
| 22 | 16.7644 |  simply |
| 23 | 16.7690 | -на |
| 24 | 16.7580 | -на |
| 25 | 16.7475 |  «** |
| 26 | 16.7692 |  «** |
| 27 | 16.7763 |  «** |
| 28 | 16.7407 |  «** |
| 29 | 16.7604 |  «** |
| 30 | 16.7426 | -на |
| 31 | 16.7931 | -на |
| 32 | 16.7888 | -на |
| 33 | 16.7740 | ' Berlin' |
| 34 | 16.7613 |  Berlin |
| 35 | 16.7339 |  Berlin |
| 36 | 16.6994 |  Berlin |
| 37 | 16.5133 | " """ |
| 38 | 15.8694 | " """ |
| 39 | 16.0050 |  Berlin |
| 40 | 3.1807 |  Berlin |

Semantic layer: L_semantic = 33 (first is_answer=True) [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:35]. Gold answer string: "Berlin" [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1052-1058].

Control margin (JSON): first_control_margin_pos = 1; max_control_margin = 0.4679627253462968 [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1048-1049].

Ablation (no‑filler): L_copy_orig = null, L_sem_orig = 33; L_copy_nf = null, L_sem_nf = 31; ΔL_copy = null; ΔL_sem = −2 [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1025-1030].

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (L_copy = null).

Confidence milestones (pure CSV):
- p_top1 > 0.30 at layer 40; final-layer p_top1 = 0.4555 [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:42]. No layer exceeds 0.60 before final.

Rank milestones (diagnostics): rank ≤ 10 at layer 30; rank ≤ 5 at layer 30; rank ≤ 1 at layer 33 [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:842-844].

KL milestones (diagnostics + CSV): first_kl_below_1.0 at layer 40; first_kl_below_0.5 at layer 40 [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:840-841]. KL decreases toward ≈0 by the final layer (final kl_to_final_bits = 0.0) [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:847].

Cosine milestones (pure CSV): first cos_to_final ≥ 0.2 at layer 35 [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:37]; ≥ 0.4 at layer 40; ≥ 0.6 at layer 40; final cos_to_final = 1.0000 [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:42].

Prism sidecar analysis. Presence/compatibility: present=true, compatible=true, k=512, layers=[embed, 9, 19, 29] [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:819-825]. Early-depth stability (KL bits, baseline → Prism): L0 10.52 → 10.68; L9 10.74 → 13.02; L19 10.68 → 16.19; L29 10.69 → 14.99 [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:2; 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token-prism.csv:1,10,19,29]. Rank milestones (Prism): no layer achieves rank ≤10 (answer_rank stays >40k across stack) [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token-prism.csv:1-40]. Top‑1 agreement at sampled depths is weak (e.g., L9 baseline top‑1 ' simply' vs Prism 'ingly'). Cosine drift: Prism cos_to_final remains small/negative at early/mid layers (e.g., L9 ≈ −0.0068) [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token-prism.csv:9]. Copy flags: no spurious copy_collapse flips under Prism. Verdict: Regressive (KL increases by several bits at early/mid layers; no earlier rank milestones).

## 4. Qualitative patterns & anomalies
The model exhibits a late semantic snap: the gold token first appears in the top‑k around L30 and becomes top‑1 at L33, with rising confidence thereafter. Records confirm emergence: “…, Berlin,0.000169…” at L30 [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-records.csv:584], strengthening by L31–35 and stabilizing by L40 (p ≈ 0.455) [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:42]. Prior to L30, predictions are dominated by multilingual filler/markup tokens (e.g., " simply", " «**", “-на”), indicating form-pattern anchoring before semantic consolidation.

Negative control (prompt: “Berlin is the capital of”): top‑5 are “ Germany (0.8021), which (0.0670), the (0.0448), _ (0.0124), what (0.0109)” [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:12-28]. Berlin still appears in the top‑10 at p=0.00481 → semantic leakage: Berlin rank ≈7 (p=0.0048) [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:28-48].

Important‑word trajectory (records): “Berlin” first enters any top‑5 by L30 and becomes top‑1 by L33 [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-records.csv:584,638]. Semantically related tokens (city/capital/Paris) co‑appear near the transition (e.g., L34 shows ' Paris' in the top‑k) [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-records.csv:655].

Stylistic ablation: removing “simply” leads to earlier semantics (ΔL_sem = −2; 33 → 31), suggesting the filler slightly delays consolidation rather than helping it [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1026-1030].

Rest‑mass sanity: rest_mass falls sharply only at the very last layer (e.g., L33 rest_mass ≈ 0.9988; final ≈ 0.1811) [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:35,42], consistent with limited top‑k coverage pre‑final; treat as coverage, not fidelity.

Rotation vs amplification: cos_to_final rises late (first ≥0.2 at L35) while KL remains >8 bits until L38–39, indicating early direction, late calibration. Final‑head calibration is clean (kl_to_final_bits=0.0; temp_est=1.0) [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:846-859], so final probabilities are comparable within‑model.

Lens sanity: raw‑vs‑norm summary flags low artifact risk with modest max KL (0.1793 bits) and no norm‑only semantic layer [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1019-1022].

Checklist:
- RMS lens? ✓ (RMSNorm) [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:810-811]
- LayerNorm bias removed? ✓ (“not_needed_rms_model”) [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:812]
- Entropy rise at unembed? ✗ (entropy drops to 3.18 bits) [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:870-888]
- FP32 un‑embed promoted? ✓ (unembed_dtype torch.float32) [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:809]
- Punctuation / markup anchoring? ✓ (quote/markup tokens dominate mid‑stack) [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:37-39]
- Copy‑reflex? ✗ (no copy_collapse=True in layers 0–3)
- Grammatical filler anchoring? ✓ (early top‑1 often ' simply'/' the' variants) [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv:9,21-22]

## 5. Limitations & data quirks
High rest_mass persists well past L_semantic (e.g., 0.9988 at L33), indicating limited top‑k coverage pre‑final; treat coverage as non‑diagnostic for lens fidelity. KL is lens‑sensitive; rely on rank milestones for cross‑model claims. Final‑head calibration is good here (warn_high_last_layer_kl=false), but KL trends remain qualitative. Raw‑vs‑norm sanity used "sample" mode, so findings are sampled rather than exhaustive.

## 6. Model fingerprint
Mistral‑Small‑24B‑Base‑2501: semantic collapse at L33; final entropy 3.18 bits; 'Berlin' stabilizes top‑1 late with clean final‑head calibration.

---
Produced by OpenAI GPT-5
