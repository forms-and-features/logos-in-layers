**1. Overview**
Mistral‑7B‑v0.1 (7B) evaluated on 2025‑08‑24 15:49:44; probe runs a norm‑lens logit‑lens sweep, recording per‑layer entropy/top‑k and collapse flags for the next token after: “Give the city name only, plain text. The capital of Germany is called simply”.
Outputs include JSON diagnostics and per‑layer CSVs for pure next‑token and per‑prompt‑token records.

**2. Method Sanity‑Check**
Diagnostics confirm the intended norm lens and positional‑encoding handling: use_norm_lens is true and unembedding is decoded in FP32; positional info is rotary (“token‑only at layer 0”).
Quoted lines:
- “use_norm_lens": true … "unembed_dtype": "torch.float32” [rows 807–809 in JSON]
- “layer0_position_info": "token_only_rotary_model” [row 816 in JSON]

The `context_prompt` ends with “called simply” (no trailing space): “context_prompt": "… is called simply” [row 817 in JSON]. Diagnostics include “L_copy”, “L_copy_H”, “L_semantic”, and “delta_layers” [rows 819–822 in JSON]. In the pure‑next‑token CSV, no row has `copy_collapse = True` in layers 0–3 (or anywhere). Copy‑collapse flag check: n.a. (no row fired).

**3. Quantitative Findings**
Per‑layer pure next‑token results (entropy in bits, top‑1 token):

| Layer | Entropy | Top‑1 |
|---|---:|---|
| L 0 | 14.96 | ‘dabei’ |
| L 1 | 14.93 | ‘biologie’ |
| L 2 | 14.83 | ‘",\r"’ |
| L 3 | 14.88 | ‘[…]’ |
| L 4 | 14.85 | ‘[…]’ |
| L 5 | 14.83 | ‘[…]’ |
| L 6 | 14.84 | ‘[…]’ |
| L 7 | 14.80 | ‘[…]’ |
| L 8 | 14.82 | ‘[…]’ |
| L 9 | 14.78 | ‘[…]’ |
| L 10 | 14.78 | ‘[…]’ |
| L 11 | 14.74 | ‘[…]’ |
| L 12 | 14.64 | ‘[…]’ |
| L 13 | 14.73 | ‘[…]’ |
| L 14 | 14.65 | ‘[…]’ |
| L 15 | 14.45 | ‘[…]’ |
| L 16 | 14.60 | ‘[…]’ |
| L 17 | 14.63 | ‘[…]’ |
| L 18 | 14.52 | ‘[…]’ |
| L 19 | 14.51 | ‘[…]’ |
| L 20 | 14.42 | ‘simply’ |
| L 21 | 14.35 | ‘simply’ |
| L 22 | 14.39 | ‘“’ |
| L 23 | 14.40 | ‘simply’ |
| L 24 | 14.21 | ‘simply’ |
| **L 25** | 13.60 | ‘Berlin’ |
| L 26 | 13.54 | ‘Berlin’ |
| L 27 | 13.30 | ‘Berlin’ |
| L 28 | 13.30 | ‘Berlin’ |
| L 29 | 11.43 | ‘"""’ |
| L 30 | 10.80 | ‘“’ |
| L 31 | 10.99 | ‘"""’ |
| L 32 | 3.61 | ‘Berlin’ |

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (no copy‑collapse layer; L_semantic = 25 [row 821 in JSON]).

Confidence milestones:
- p > 0.30 at layer 32 (0.3822) [row 34 in pure‑next CSV]
- p > 0.60 at layer — n.a.
- final‑layer p = 0.3822 for ‘Berlin’ [row 34 in pure‑next CSV]

**4. Qualitative Patterns & Anomalies**
Negative control shows correct inversion: for “Berlin is the capital of”, the model predicts the country with high confidence and only weakly places “Berlin” in the list: “top‑5: Germany 0.8966, the 0.0539, both 0.00436, a 0.00380, Europe 0.00311 … Berlin 0.00284” [rows 13–36 in JSON]. Semantic leakage: Berlin rank 6 (p = 0.00284) [row 35 in JSON].

Important‑word trajectory (records.csv): “Berlin” first enters the top‑5 for the prompt token “Germany” by L24 (p = 0.00139) [row 423], strengthens by L25 where “called”→top‑1 ‘Berlin’ (0.0705) [row 442] and “Germany” includes ‘Berlin’ in top‑5 (0.00340) [row 440]. At the pure next‑token position, ‘Berlin’ becomes top‑1 at L25 (p = 0.0335) [row 27 in pure‑next CSV], dips under punctuation at L29–31, then recovers to rank‑1 at the final layer with p = 0.3822 [row 34 in pure‑next CSV]. Related tokens frequently co‑appear: ‘Germany’, ‘Deutschland’, ‘German’, ‘Frankfurt’ [rows 27–31 in pure‑next CSV]. “capital” briefly appears in top‑5 at L21 for the next token [row 23 in pure‑next CSV] and at the “Germany” position L21–24 [rows 372, 389, 406, 423], then fades.

Punctuation phase: layers 29–31 exhibit quotation‑mark anchoring at the next‑token head: L29 top‑1 ‘"""’ (0.0706), L30 top‑1 ‘“’ (0.1025), L31 top‑1 ‘"""’ (0.0935) [rows 31–33 in pure‑next CSV]. This is consistent with known lens behaviors where surface punctuation temporarily dominates before the answer consolidates (cf. Tuned‑Lens, arXiv:2303.08112).

Collapse‑layer and instructions: L_semantic = 25 for the main prompt [row 821 in JSON]. The test‑prompt blocks do not provide per‑layer collapses, so a precise shift without the “one‑word” instruction cannot be computed. However, across paraphrases without that instruction, the one‑step predictions remain robustly “Berlin” with high p (e.g., “Germany’s capital city is called simply” → ‘Berlin’, 0.5392 [rows 60–63 in JSON]).

Rest‑mass sanity: rest_mass falls substantially after L25, from 0.911 at L25 to 0.230 at the final layer [rows 27 and 34 in pure‑next CSV]. Max after L_semantic = 0.911 at layer 25; it steadily declines by L29–31 (≈0.746→0.739) and reaches 0.230 at L32, suggesting no precision loss.

Temperature robustness: at T = 0.1, Berlin rank 1 (p = 0.9996, entropy 0.005 bits) [rows 669–678 in JSON]; at T = 2.0, Berlin rank 1 (p = 0.0360, entropy 12.220 bits) [rows 740–744 and 762–764 in JSON]. Entropy rises accordingly from ~0.005 to ~12.220 bits.

Checklist:
- RMS lens? ✓ (“RMSNorm”, first/final) [rows 810–811 in JSON]
- LayerNorm bias removed? ✓ (not needed; RMS) [row 812 in JSON]
- Entropy rise at unembed? ✗ (final entropy 3.61 bits; no spike) [row 826 in JSON; row 34 in pure‑next CSV]
- FP32 un‑embed promoted? ✗ (“use_fp32_unembed”: false; “unembed_dtype”: torch.float32) [rows 808–809 in JSON]
- Punctuation / markup anchoring? ✓ (quotes dominate L29–31) [rows 31–33 in pure‑next CSV]
- Copy‑reflex? ✗ (no `copy_collapse = True` in layers 0–3) [rows 2–5 in pure‑next CSV]
- Grammatical filler anchoring? ✗ (no {is,the,a,of} as top‑1 in L0–5) [rows 2–7 in pure‑next CSV]

> Examples (inline quotes): “(layer 32, token = ‘Berlin’, p = 0.3822)” [row 34 in pure‑next CSV]; “(‘Germany’, 0.8966) … (‘Berlin’, 0.00284)” [rows 14–35 in JSON]; “(pos ‘called’, L25 → ‘Berlin’, 0.0705)” [row 442 in records CSV].

**5. Limitations & Data Quirks**
- No copy‑collapse detected (L_copy = null) [row 819 in JSON], so ΔH relative to a copy‑collapsed layer is not defined.
- Rest_mass remains > 0.3 for many layers after L_semantic (e.g., 0.91 at L25; 0.70–0.75 at L30±1) [rows 27, 31–33], though it ultimately falls to 0.23 at L32.
- Early‑layer top‑1 tokens are non‑linguistic/multilingual fragments (e.g., ‘[…]’), typical of raw lens views before consolidation; this does not affect the final consolidation.

**6. Model Fingerprint**
Mistral‑7B‑v0.1: collapse at L 25; final entropy 3.61 bits; ‘Berlin’ re‑emerges after punctuation phase and finishes rank 1 at L 32.

---
Produced by OpenAI GPT-5
