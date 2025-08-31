**Overview**

- **Model**: mistralai/Mistral-Small-24B-Base-2501 (24B). Run analyzes layer-wise entropy and next-token predictions via a norm-lens, plus prompt variants and temperature sweeps. The probe tracks copy-collapse, semantic collapse, and uncertainty dynamics.

**Method Sanity‑Check**

- JSON confirms RoPE-style positional encoding and norm-lens usage: > "use_norm_lens": true [L807] and > "layer0_position_info": "token_only_rotary_model" [L816] in 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json.
- Context prompt ends exactly with "called simply" (no trailing space): "Give the city name only, plain text. The capital of Germany is called simply" [L4]. Diagnostics include L_copy, L_copy_H, L_semantic, delta_layers, and flags such as "unembed_dtype": "torch.float32", "use_fp32_unembed": false, and LN types RMSNorm [L809–L813].
- Copy-collapse flag check: no row with copy_collapse=True was found (layers 0–3 and beyond). First row with copy_collapse=True: none → ✓ rule not spuriously fired.

**Quantitative Findings**

| Layer | Entropy (bits) | Top-1 |
|---|---:|---|
| L 0 | 17.00 | ' Forbes' |
| L 1 | 16.97 | '随着时间的' |
| L 2 | 16.94 | '随着时间的' |
| L 3 | 16.81 | '随着时间的' |
| L 4 | 16.87 | ' quelcon' |
| L 5 | 16.90 | 'народ' |
| L 6 | 16.91 | 'народ' |
| L 7 | 16.90 | 'народ' |
| L 8 | 16.90 | ' quelcon' |
| L 9 | 16.89 | ' simply' |
| L 10 | 16.84 | ' hétérogènes' |
| L 11 | 16.84 | '从那以后' |
| L 12 | 16.84 | ' simply' |
| L 13 | 16.87 | ' simply' |
| L 14 | 16.81 | 'стен' |
| L 15 | 16.82 | 'luš' |
| L 16 | 16.83 | 'luš' |
| L 17 | 16.78 | 'luš' |
| L 18 | 16.76 | 'luš' |
| L 19 | 16.77 | 'luš' |
| L 20 | 16.74 | 'luš' |
| L 21 | 16.77 | ' simply' |
| L 22 | 16.76 | ' simply' |
| L 23 | 16.77 | '-na' |
| L 24 | 16.76 | '-na' |
| L 25 | 16.75 | ' «**' |
| L 26 | 16.77 | ' «**' |
| L 27 | 16.78 | ' «**' |
| L 28 | 16.74 | ' «**' |
| L 29 | 16.76 | ' «**' |
| L 30 | 16.74 | '-na' |
| L 31 | 16.79 | '-na' |
| L 32 | 16.79 | '-na' |
| **L 33** | **16.77** | **' Berlin'** |
| L 34 | 16.76 | ' Berlin' |
| L 35 | 16.73 | ' Berlin' |
| L 36 | 16.70 | ' Berlin' |
| L 37 | 16.51 | ' "' |
| L 38 | 15.87 | ' "' |
| L 39 | 16.00 | ' Berlin' |
| L 40 | 3.18 | ' Berlin' |

- ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (L_copy = null; no copy-collapse detected in diagnostics).
- Confidence milestones: p > 0.30 at layer 40; p > 0.60 not reached; final-layer p = 0.4555 (" Berlin") [L826–L833].

**Qualitative Patterns & Anomalies**

The model shows classic pre-norm RMS lens behavior: uncertainty remains near-uniform through mid-stack, then answer probability emerges late. The semantic layer occurs at L33 where " Berlin" first becomes top-1 in the lens view (L 33, entropy 16.77 bits) and strengthens over layers 34–36 before punctuation competes at layers 37–38. This late rise is consistent with logit-lens observations (cf. A. Nanda et al., Tuned-Lens, arXiv:2303.08112) that meaning crystallizes after early token/format heuristics.

- Negative control: “Berlin is the capital of” top‑5 are “ Germany” (0.8021), “ which” (0.0670), “ the” (0.0448), “ _” (0.0124), “ what” (0.0109) — “ Berlin” appears later at rank 7 with p = 0.00481 → semantic leakage: Berlin rank 7 (p = 0.00481) [L14–L31, L38–L39].
- Important‑word trajectory: “ Berlin” starts to appear in top‑k around L30 (e.g., at L30 ‘is’: …, Berlin 0.00024) [row 582 in records CSV]; it becomes top‑1 by L33 across multiple positions (e.g., L33 ‘is’: Berlin 0.00087; L33 ‘called’: Berlin 0.00075; L33 ‘simply’: Berlin 0.00042) [rows 636–638]. Semantically close forms (ber/Ber/-Ber) surge by L35 (e.g., L35 ‘simply’: many ‘ber*’ variants alongside Berlin) [row 672].
- Collapse‑layer instruction shift: The JSON includes prompt variants without the one‑word instruction and shows strong final probabilities for “ Berlin” (e.g., “Germany’s capital city is called simply” p = 0.5113; “The capital city of Germany is named simply” p = 0.3062) [L56–L63, L104–L110]. Per‑layer collapse indices are not recorded for these tests, so L_semantic shift under instruction removal is n.a.
- Rest‑mass sanity: Rest_mass falls steadily after L33, with max after L_semantic = 0.9988 at L33 and a sharp drop by the final decode (0.1811) [pure CSV rows 35 and 42], consistent with concentration into the displayed top‑k.
- Temperature robustness: At T = 0.1, Berlin rank 1 (p = 0.9995; entropy 0.0061 bits) [L670–L676, L671]. At T = 2.0, Berlin rank 1 (p = 0.0299; entropy 14.36 bits), indicating heavy flattening with high temperature [L737–L743, L738].
- Punctuation/markup anchoring: Punctuation briefly dominates late (L37–L38 top‑1 is a quote), before the final decode reasserts “ Berlin” [pure CSV rows 39–40]. Grammatical fillers appear early but do not anchor top‑1 in layers 0–5.

Checklist:
- RMS lens? ✓ (“RMSNorm” types; “use_norm_lens”: true) [L807, L810–L811]
- LayerNorm bias removed? ✓ (“layernorm_bias_fix”: "not_needed_rms_model") [L812]
- Entropy rise at unembed? ✓ (entropy drops to 3.18 bits at L40; final decode) [pure CSV row 42]
- FP32 un‑embed promoted? ✓ (“unembed_dtype”: "torch.float32"; “mixed_precision_fix”: casting to fp32) [L809, L815]
- Punctuation / markup anchoring? ✓ (quotes at L37–L38) [pure CSV rows 39–40]
- Copy‑reflex? ✗ (no copy_collapse=True in layers 0–3; none anywhere)
- Grammatical filler anchoring? ✗ (no {is, the, a, of} as top‑1 in layers 0–5)

**Limitations & Data Quirks**

No copy-collapse detected (L_copy = null), so ΔH relative to L_copy is n.a. Rest_mass remains >0.99 across many mid/late layers but drops appropriately at the final decode, suggesting lens scaling is behaving; still, very high rest_mass after L_semantic (0.9988 at L33) indicates that much probability mass sits outside top‑20 until very late.

**Model Fingerprint**

Mistral‑Small‑24B‑Base‑2501: collapse at L 33; final entropy 3.18 bits; “Berlin” consolidates late and survives temperature sweeps.

---
Produced by OpenAI GPT-5

