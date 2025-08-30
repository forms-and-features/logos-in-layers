# Evaluation Report: Qwen/Qwen2.5-72B

*Run executed on: 2025-08-24 15:49:44*
# Evaluation Report: Qwen/Qwen2.5-72B

## 1. Overview
Qwen‑2.5‑72B (80 layers) evaluated on 2025‑08‑24. The probe captures layer‑by‑layer next‑token distributions with a norm‑lens, entropy in bits, collapse flags, and compact diagnostics plus targeted prompt/temperature probes.

## 2. Method sanity‑check
Diagnostics confirm the intended RMS norm‑lens and positional handling: normalized residuals are used, FP32 unembedding is active, and layer‑0 interpretation treats rotary PEs as token‑only with real ln1 applied. Examples:

> "use_norm_lens": true  [output‑Qwen2.5‑72B.json L807]
>
> "layer0_position_info": "token_only_rotary_model"; "layer0_norm_fix": "using_real_ln1_on_embeddings"  [L816, L814]

Prompt integrity: context_prompt ends with “called simply” (no trailing space).

> "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply"  [L817]

Diagnostics fields present: L_copy, L_copy_H, L_semantic, delta_layers, and implementation flags (use_fp32_unembed, unembed_dtype, ln types).

> "unembed_dtype": "torch.float32"; "L_copy": null; "L_copy_H": null; "L_semantic": 80; "delta_layers": null  [L809, L819–L822]

Copy‑collapse flag check (copy_threshold=0.90, copy_margin=0.05): no rows with copy_collapse=True in the pure‑next‑token CSV (layers 0–80). First four layers are all False (e.g., layer 0 row below). Rule did not fire; no spurious triggers.

> "0,15,⟨NEXT⟩,…,rest_mass,…,False,False,False"  [output‑Qwen2.5‑72B‑pure‑next‑token.csv row 2]

## 3. Quantitative findings
Per‑layer next‑token probe (entropy bits, top‑1 token). Bold indicates first semantic layer (top‑1 = “Berlin”). Source: output‑Qwen2.5‑72B‑pure‑next‑token.csv.

| Layer | Entropy (bits) | Top‑1 |
|---:|---:|:---|
| 0 | 17.214241 | ‘s’ |
| 1 | 17.214149 | ‘下一篇’ |
| 2 | 17.142515 | ‘ولوج’ |
| 3 | 17.063072 | ‘شدد’ |
| 4 | 17.089064 | ‘.myapplication’ |
| 5 | 17.007233 | ‘ستحق’ |
| 6 | 17.031500 | ‘.myapplication’ |
| 7 | 16.937199 | ‘.myapplication’ |
| 8 | 16.798046 | ‘.myapplication’ |
| 9 | 16.120342 | ‘ستحق’ |
| 10 | 16.500811 | ‘.myapplication’ |
| 11 | 16.718040 | ‘.myapplication’ |
| 12 | 16.778097 | ‘かもしれ’ |
| 13 | 16.631441 | ‘かもしれ’ |
| 14 | 16.359472 | ‘かもしれ’ |
| 15 | 16.517048 | ‘のではない’ |
| 16 | 16.490759 | ‘iéndo’ |
| 17 | 16.212702 | ‘iéndo’ |
| 18 | 16.109280 | ‘有期徒’ |
| 19 | 15.757386 | ‘有期徒’ |
| 20 | 16.129038 | ‘有期徒’ |
| 21 | 16.155777 | ‘有期徒’ |
| 22 | 15.979851 | ‘有期徒’ |
| 23 | 16.401468 | ‘.myapplication’ |
| 24 | 15.998866 | ‘iéndo’ |
| 25 | 15.350595 | ‘hế’ |
| 26 | 15.943514 | ‘iéndo’ |
| 27 | 15.755867 | ‘iéndo’ |
| 28 | 15.749987 | ‘.myapplication’ |
| 29 | 15.884907 | ‘.myapplication’ |
| 30 | 16.122501 | ‘.myapplication’ |
| 31 | 16.169998 | ‘.myapplication’ |
| 32 | 16.170771 | ‘.myapplication’ |
| 33 | 16.419062 | ‘hế’ |
| 34 | 16.200062 | ‘iéndo’ |
| 35 | 16.455050 | ‘hế’ |
| 36 | 16.407827 | ‘iéndo’ |
| 37 | 16.210032 | ‘iéndo’ |
| 38 | 16.490444 | ‘hế’ |
| 39 | 16.417740 | ‘iéndo’ |
| 40 | 16.191633 | ‘iéndo’ |
| 41 | 16.465208 | ‘hế’ |
| 42 | 16.594849 | ‘hế’ |
| 43 | 16.497379 | ‘hế’ |
| 44 | 16.655331 | ‘続きを読む’ |
| 45 | 16.876968 | ‘国际在线’ |
| 46 | 17.002260 | ‘国际在线’ |
| 47 | 17.013271 | ‘主义思想’ |
| 48 | 17.021650 | ‘主义思想’ |
| 49 | 17.021749 | ‘reuseIdentifier’ |
| 50 | 16.967781 | ‘uckets’ |
| 51 | 16.972252 | ‘"’ |
| 52 | 17.008556 | ‘"’ |
| 53 | 16.926603 | ‘"’ |
| 54 | 16.908087 | ‘"’ |
| 55 | 16.942335 | ‘"’ |
| 56 | 16.938183 | ‘"’ |
| 57 | 16.840782 | ‘"’ |
| 58 | 16.914761 | ‘"’ |
| 59 | 16.920084 | ‘"’ |
| 60 | 16.886055 | ‘'’ |
| 61 | 16.903027 | ‘'’ |
| 62 | 16.833620 | ‘"’ |
| 63 | 16.890835 | ‘"’ |
| 64 | 16.894716 | ‘"’ |
| 65 | 16.868870 | ‘"’ |
| 66 | 16.899384 | ‘"’ |
| 67 | 16.893194 | ‘"’ |
| 68 | 16.778610 | ‘"’ |
| 69 | 16.875761 | ‘"’ |
| 70 | 16.786564 | ‘"’ |
| 71 | 16.504635 | ‘"’ |
| 72 | 16.649858 | ‘"’ |
| 73 | 15.786654 | ‘"’ |
| 74 | 16.080919 | ‘"’ |
| 75 | 13.349901 | ‘"’ |
| 76 | 14.742768 | ‘"’ |
| 77 | 10.847792 | ‘"’ |
| 78 | 15.397843 | ‘"’ |
| 79 | 16.665640 | ‘"’ |
| 80 | 4.115832 | **‘Berlin’** |

ΔH (bits) = n.a. (no copy‑collapse layer; “L_copy”: null).

Confidence milestones:
- p > 0.30 at layer 80 (p = 0.3395)  [pure‑next‑token.csv row 138]
- p > 0.60: n.a. (never exceeds 0.60 in layers 0–80)
- Final‑layer p = 0.3395; entropy = 4.1158 bits  [row 138]

## 4. Qualitative patterns & anomalies
The norm‑lens setup is consistent with best practice (Tuned‑Lens 2303.08112): normalized residuals with RMS at ln1 and FP32 unembed improve calibration without distorting logits. Diagnostics explicitly report RMS first/last norms and FP32 unembed ("first_block_ln1_type": "RMSNorm"; "final_ln_type": "RMSNorm"; "use_fp32_unembed": true) [L810–L811, L808]. The pure‑next‑token trajectory shows no early copy‑reflex; instead, the stack progresses from diffuse non‑English tokens and library identifiers to punctuation anchoring in the high‑middle layers (L51–L79), before semantic collapse at the final layer where “Berlin” becomes top‑1.

Negative control “Berlin is the capital of”: top‑5 are dominated by the country completion and fillers; Berlin does not appear. Quote: > “ Germany, 0.7695; the, 0.0864; which, 0.0491; a, 0.0125; what, 0.0075”  [output‑Qwen2.5‑72B.json L14–L31].

Records/important‑word trajectory (next‑token position). “Berlin” first enters any top‑5 at L78, then remains through L80: > “… Berlin, 0.0033” [pure‑next‑token.csv L136]; rises to rank‑2 at L79 (p = 0.0021) [L137], and top‑1 at L80 (p = 0.3395) [L138]. The semantically related word “capital” appears in the top‑5 earlier, from L67 through L76 (e.g., “… capital, 0.00696” [L134]), and then drops out by L77 [L135]; “Germany” does not appear in the top‑5 for the next‑token slot.

Instruction ablation. Several test prompts omit the “one‑word” instruction yet still put “ Berlin” at rank‑1 with moderate to high probability, e.g., “Germany’s capital city is called” → “ Berlin, 0.4473” [L249–L251]. Per‑layer collapse indices are not recorded for test prompts, so a shift in L_semantic cannot be measured from provided artifacts; however, the top‑1s indicate robust semantics even without the instruction.

Rest‑mass sanity. Rest_mass collapses from ~0.99 in early layers to 0.2977 at L80 (post‑semantic); no spikes >0.30 after L_semantic, suggesting adequate precision of the lens near collapse [pure‑next‑token.csv row 138].

Temperature robustness. At T = 0.1, “ Berlin” rank 1 (p = 0.9526), entropy = 0.2754 bits [output‑Qwen2.5‑72B.json L670–L676]. At T = 2.0, “ Berlin” still rank 1 (p = 0.0162), entropy = 15.0128 bits [L737–L743, L738].

Important‑word trajectory — Berlin first enters any top‑5 at layer 78 and stabilises by layer 78–80; “capital” enters by layer 67 and drops after layer 76; “Germany” does not feature in the next‑token top‑5. Examples: > “… capital, 0.00696” [pure‑next‑token.csv L134]; > “… Berlin, 0.3395” [L138].

Checklist:
- RMS lens?: ✓ (RMSNorm detected; use_norm_lens=true)  [L807, L810]
- LayerNorm bias removed?: ✓/n.a. (RMS model; "layernorm_bias_fix": "not_needed_rms_model")  [L812]
- Entropy rise at unembed?: ✓ slight (final model entropy 4.1356 vs. lens 4.1158)  [L826–L828; pure row 138]
- FP32 un‑embed promoted?: ✓ ("use_fp32_unembed": true)  [L808]
- Punctuation / markup anchoring?: ✓ (top‑1 often '"' across L51–79)  [pure‑next‑token.csv L130–L137]
- Copy‑reflex?: ✗ (no copy_collapse=True in layers 0–3)  [pure rows 2–5]
- Grammatical filler anchoring?: ✗ (top‑1 in L0–5 is not {is, the, a, of})  [pure rows 2–7]

> Final‑layer calibration cross‑check: “ Berlin, 0.3383 …” [final_prediction top‑k; output‑Qwen2.5‑72B.json L828–L836].

## 5. Limitations & data quirks
No copy‑collapse detected ("L_copy": null), so ΔH relative to a copy layer is undefined. Rest_mass remains very high through most of the stack (expected for diffuse early layers), with a sharp drop only at L80; nonetheless, Rest_mass ≤ 0.30 post‑semantic (0.2977 at L80), indicating no clear norm‑lens mis‑scale. Punctuation anchoring in late‑middle layers (quotes/commas) may reflect formatting priors in the tokenizer and training mix rather than semantic convergence.

## 6. Model fingerprint
Qwen‑2.5‑72B: collapse at L 80; final entropy 4.12 bits; “Berlin” enters top‑5 at L 78 then top‑1 at L 80; no copy‑reflex.

---
Produced by OpenAI GPT-5

