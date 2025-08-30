# Evaluation Report: Qwen/Qwen3-14B

*Run executed on: 2025-08-24 15:49:44*
**Overview**

Qwen3‑14B (14B) evaluated on 2025‑08‑24 (timestamp‑20250824‑1549). The probe performs a layer‑by‑layer norm‑lens analysis of the pure next‑token distribution, tracking entropy and top‑k tokens, with copy/entropy collapse flags, diagnostic metadata, negative‑control prompts, and temperature sweeps.

**Method Sanity‑Check**

JSON diagnostics confirm RMSNorm with normalized residual lens and rotary positional handling for layer‑0: > "use_norm_lens": true [L807], "layer0_position_info": "token_only_rotary_model" [L816]. The `context_prompt` ends with “called simply” with no trailing space: > "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply" [L817]. Diagnostics include L_copy/L_copy_H/L_semantic/delta_layers and implementation flags: > "L_copy": 32, "L_copy_H": 32, "L_semantic": 36, "delta_layers": 4 [L819–L822]; > "unembed_dtype": "torch.float32" [L809].

Copy‑collapse in layers 0–3: none (all False in pure‑next‑token CSV rows 2–5). First row with `copy_collapse = True`: layer = 32, top‑1 = "____" (p1 = 0.8976), top‑2 = " ____" (p2 = 0.0583) [row 34 in CSV] → ✗ fired spuriously (triggered by entropy fallback; token not in prompt). Rule parameters: copy_threshold = 0.90, copy_margin = 0.05.

**Quantitative Findings**

| Layer | Entropy (bits) | Top‑1 |
|---|---:|---|
| L 0 | 17.213 | 梳 |
| L 1 | 17.212 | 地处 |
| L 2 | 17.211 | 是一部 |
| L 3 | 17.210 | tics |
| L 4 | 17.208 | tics |
| L 5 | 17.207 | -minded |
| L 6 | 17.205 | 过去的 |
| L 7 | 17.186 | � |
| L 8 | 17.180 | -minded |
| L 9 | 17.188 | -minded |
| L 10 | 17.170 |  (?) |
| L 11 | 17.151 | 时代的 |
| L 12 | 17.165 | といって |
| L 13 | 17.115 |  nav |
| L 14 | 17.141 |  nav |
| L 15 | 17.149 | 唿 |
| L 16 | 17.135 | 闯 |
| L 17 | 17.137 | 唿 |
| L 18 | 17.101 | ____ |
| L 19 | 17.075 | ____ |
| L 20 | 16.932 | ____ |
| L 21 | 16.986 | 年夜 |
| L 22 | 16.954 | 年夜 |
| L 23 | 16.840 | ____ |
| L 24 | 16.760 | ____ |
| L 25 | 16.758 | 年夜 |
| L 26 | 16.669 | ____ |
| L 27 | 16.032 | ____ |
| L 28 | 15.234 | ____ |
| L 29 | 14.187 | 这个名字 |
| L 30 | 7.789 | 这个名字 |
| L 31 | 5.162 | ____ |
| L 32 | 0.816 | ____ |
| L 33 | 0.481 | ____ |
| L 34 | 0.595 | ____ |
| L 35 | 0.668 | ____ |
| **L 36** | 0.312 | ** Berlin** |
| L 37 | 0.906 |  ____ |
| L 38 | 1.212 |  ____ |
| L 39 | 0.952 |  Berlin |
| L 40 | 3.584 |  Berlin |

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = 0.816 − 0.312 = 0.504

Confidence milestones (Berlin): p > 0.30 at layer 36, p > 0.60 at layer 36, final‑layer p = 0.345 [row 42 in CSV; JSON L829–L835].

**Qualitative Patterns & Anomalies**

Semantic emergence is late but decisive. Berlin first enters any top‑5 at L33 and becomes top‑1 with high confidence by L36: > “… ( Berlin, 0.00253)” [row 35 in CSV]; > “ Berlin, 0.9530 …” [row 38 in CSV]. This aligns with norm‑lens accounts where intermediate layers linearize features and later layers assemble task‑specific logits (Tuned‑Lens, arXiv:2303.08112).

Negative control (“Berlin is the capital of”): top‑5 are > “ Germany, 0.6320; which, 0.2468; the, 0.0737; what, 0.00943; a, 0.00475” [L14–L31 in JSON]. Berlin does not appear → no semantic leakage.

Important‑word trajectory (records.csv): Around L33–L36, the context tokens “Germany/is/called” align with Berlin: > “Germany … 柏林, 0.0711 …” [row 606]; > “called … 柏林, 0.2879; Berlin, 0.1062” [row 624]; > “is … Berlin, 0.9851 …” and “called … Berlin, 0.9933 …” [rows 708–709]. At the pure next‑token slot, Berlin becomes top‑1 at L36: > “ Berlin, 0.9530 …” [row 38 in CSV]. Later layers also surface subword forms (“BER”, “Ber”) around L39: > “… Berlin, 0.8123; BER, 0.1256; Ber, 0.0498 …” [row 41].

Instruction ablation (“one‑word” absent): Layer‑level collapse indices are not emitted for test prompts (n.a.). However, multiple formulations without the “one‑word” phrasing still predict Berlin strongly (e.g., “Germany has its capital at …” → “ Berlin”, 0.8926 [L343–L345]; “known as …” → “ Berlin”, 0.7338 [L390–L396]).

Rest‑mass sanity: Rest_mass declines sharply by mid‑stack (e.g., 0.000044 at L37 [row 39]), then increases at the final layer to 0.236 (row 42). This remains < 0.3, indicating no norm‑lens mis‑scale.

Temperature robustness: At T = 0.1, Berlin rank 1 (p = 0.9742; entropy 0.173 bits) [L670–L676, L671]; at T = 2.0, Berlin rank 1 (p = 0.0363; entropy 13.161 bits) [L737–L743, L738]. Entropy rises from 0.173 bits to 13.161 bits.

Punctuation/markup anchoring: Prior to semantic collapse, top‑1s are often markup/underscore tokens (e.g., “____”) and quotes even at very low entropy: > “… ____ , 0.8976;  ____ , 0.0583 …” [row 34]; > “… ____, 0.5587; ____, 0.3948;  Berlin, 0.0457 …” [row 40]. This is typical anchoring before content tokens dominate.

Checklist:
- RMS lens?: ✓ (RMSNorm; "use_norm_lens": true) [L807, L810–L811]
- LayerNorm bias removed?: n.a. (RMS model; "layernorm_bias_fix": "not_needed_rms_model") [L812]
- Entropy rise at unembed?: ✓ (L36 ≈ 0.31 → final 3.58 bits) [row 38; L826]
- FP32 un‑embed promoted?: ✗ ("use_fp32_unembed": false; dtype already fp32) [L808–L809]
- Punctuation / markup anchoring?: ✓ (underscores/quotes dominate pre‑collapse) [rows 34, 40]
- Copy‑reflex?: ✗ (layers 0–3 have copy_collapse = False) [rows 2–5]
- Grammatical filler anchoring?: ✗ (layers 0–5 top‑1 not in {“is”, “the”, “a”, “of”}) [rows 2–7]

**Limitations & Data Quirks**

- First `copy_collapse=True` is driven by entropy fallback at L32, not prompt‑token copying; top‑1 “____” is not in the prompt (rule check therefore spuriously fires for copy) [row 34; L817].
- Mixed‑language/markup tokens (“____”, CJK) dominate many early/mid layers; this inflates non‑semantic logits before collapse.
- Final‑layer rest_mass = 0.236 (< 0.3) suggests no mis‑scale; however, tail mass is non‑trivial at the last layer, reflecting broader generation options.

**Model Fingerprint**

Qwen3‑14B: collapse at L 36; final entropy 3.58 bits; “Berlin” enters top‑5 at L 33 and stabilizes by L 36; final p(“Berlin”) = 0.345.

---
Produced by OpenAI GPT-5
