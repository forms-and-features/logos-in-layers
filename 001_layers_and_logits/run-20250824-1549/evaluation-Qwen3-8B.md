# Evaluation Report: Qwen/Qwen3-8B

*Run executed on: 2025-08-24 15:49:44*
**Overview**
- Model: Qwen/Qwen3-8B (8B), run 2025-08-24. The probe analyzes per-layer next-token predictions with a norm lens, tracking entropy and the emergence of the ground-truth token.
- Captures copy/filler behaviors vs. semantic collapse at the pure next-token position, with diagnostics and temperature sweeps.

**Method Sanity‑Check**
The JSON confirms the intended RMS norm lens and rotary positional handling, and the CSV includes collapse flags and rest mass. The context prompt ends exactly with “called simply” (no trailing space). Diagnostics include L_copy/L_semantic and implementation flags. Quotes:
> "use_norm_lens": true [output-Qwen3-8B.json L807]
> "layer0_position_info": "token_only_rotary_model" [output-Qwen3-8B.json L816]
Diagnostics present: "L_copy", "L_copy_H", "L_semantic", "delta_layers", "use_norm_lens", "unembed_dtype", "use_fp32_unembed". Copy-collapse early check (τ = 0.90, δ = 0.05): none in layers 0–3. First flagged row: layer = 31, token_1 = ' Berlin', p1 = 0.9359, token_2 = ' “', p2 = 0.0428 → ✗ fired spuriously (entropy fallback; top‑1 not a prompt copy).

**Quantitative Findings**
- L 0 – entropy 17.213 bits, top-1 'CLICK'
- L 1 – entropy 17.211 bits, top-1 'apr'
- L 2 – entropy 17.211 bits, top-1 '财经'
- L 3 – entropy 17.208 bits, top-1 '-looking'
- L 4 – entropy 17.206 bits, top-1 '院子'
- L 5 – entropy 17.204 bits, top-1 '(?)'
- L 6 – entropy 17.196 bits, top-1 'ly'
- L 7 – entropy 17.146 bits, top-1 '(?)'
- L 8 – entropy 17.132 bits, top-1 '(?)'
- L 9 – entropy 17.119 bits, top-1 '(?)'
- L 10 – entropy 17.020 bits, top-1 '(?)'
- L 11 – entropy 17.128 bits, top-1 'ifiable'
- L 12 – entropy 17.117 bits, top-1 'ifiable'
- L 13 – entropy 17.126 bits, top-1 'ifiable'
- L 14 – entropy 17.053 bits, top-1 '"'
- L 15 – entropy 17.036 bits, top-1 '"'
- L 16 – entropy 16.913 bits, top-1 '-'
- L 17 – entropy 16.972 bits, top-1 '-'
- L 18 – entropy 16.911 bits, top-1 '-'
- L 19 – entropy 16.629 bits, top-1 'ly'
- L 20 – entropy 16.696 bits, top-1 '_'
- L 21 – entropy 16.408 bits, top-1 '_'
- L 22 – entropy 15.219 bits, top-1 '______'
- L 23 – entropy 15.220 bits, top-1 '____'
- L 24 – entropy 10.893 bits, top-1 '____'
- L 25 – entropy 13.454 bits, top-1 '____'
- L 26 – entropy 5.558 bits, top-1 '____'
- L 27 – entropy 4.344 bits, top-1 '____'
- L 28 – entropy 4.786 bits, top-1 '____'
- L 29 – entropy 1.778 bits, top-1 '-minded'
- L 30 – entropy 2.203 bits, top-1 'Germany'
**L 31 – entropy 0.454 bits, top-1 'Berlin'**
- L 32 – entropy 1.037 bits, top-1 'German'
- L 33 – entropy 0.988 bits, top-1 'Berlin'
- L 34 – entropy 0.669 bits, top-1 'Berlin'
- L 35 – entropy 2.494 bits, top-1 'Berlin'
- L 36 – entropy 3.123 bits, top-1 'Berlin'

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = 0.454 − 0.454 = 0.000
Confidence milestones: p > 0.30 at layer 31, p > 0.60 at layer 31, final-layer p = 0.4334

**Qualitative Patterns & Anomalies**
The answer “Berlin” emerges sharply at L31 with very low entropy (0.454 bits) and stays top‑1 through the stack despite subsequent entropy increases, matching a classic late semantic collapse under a norm lens (cf. Tuned‑Lens 2303.08112). Before L31, the model cycles through punctuation/markup tokens (____, “, ______) and filler-like fragments; at L30 “Germany” briefly becomes top‑1 before yielding to “Berlin”, consistent with a country→city disambiguation step at the output head. Important-word evolution at the answer position: “Berlin” first enters the top‑5 at L29 (p = 0.026) and becomes top‑2 at L30 (p = 0.284), then top‑1 at L31 (p = 0.936) where it stabilizes; “Germany” is top‑1 at L30 then remains in the top‑5 through later layers; markup symbols re‑enter near the unembed and raise entropy. For example: > "… Berlin,0.0265 … Germany,0.0200 …" [output-Qwen3-8B-records.csv L578] and > "… (layer 31, token = ‘Berlin’, p = 0.936)" [L610].

Negative control — “Berlin is the capital of”: top‑5 = [“ Germany”, 0.7286], [“ which”, 0.2207], [“ the”, 0.0237], [“ what”, 0.0114], [“ __”, 0.0023]. “Berlin” still appears at rank 9 (p = 0.00046): semantic leakage: Berlin rank 9 (p = 0.00046). > "… ' Germany', 0.7286 … ' which', 0.2207 …" [output-Qwen3-8B.json L14–24] and > " ' Berlin', 0.000459…" [L46–48].

Temperature robustness: At T = 0.1, Berlin rank 1 (p = 0.9992; entropy 0.0099 bits); at T = 2.0, Berlin rank 1 (p = 0.0419; entropy 13.3983 bits). > "temperature": 0.1 … " Berlin", 0.99915 [L670–676] → vs. "temperature": 2.0 … " Berlin", 0.04186 [L737–744].

Important‑word trajectory: “Berlin” first enters top‑5 at L29 and stabilizes by L31 [records L578, L610]. “Germany” is top‑1 at L30 and stays in the top‑5 through L36 [records L594, L751]. “capital” appears only in earlier prompt positions and is absent from the answer position’s top‑5 after mid‑stack, consistent with instruction following.

Instruction variant: The JSON includes alternative phrasings (with/without “Give the city name only…”), but no per‑layer indices per variant are emitted; shift of collapse layer cannot be assessed from JSON alone (n.a.).

Rest‑mass sanity: Rest_mass is minimal around L31–L34 and increases near the final unembed; max after L_semantic = 0.175 (layer 36), well below 0.3.

Checklist
- RMS lens?: ✓ (RMSNorm modules; norm lens enabled)
- LayerNorm bias removed?: ✓ (RMS model; bias not applicable)
- Entropy rise at unembed?: ✓ (0.454 → 3.123 bits by L36)
- FP32 un-embed promoted?: ✗ (use_fp32_unembed = false; dtype still fp32)
- Punctuation / markup anchoring?: ✓ (____/“/______) mid‑stack and near output
- Copy‑reflex?: ✗ (no copy_collapse in L0–L3)
- Grammatical filler anchoring?: ✗ (no {is,the,a,of} as top‑1 in L0–5)

Quotes/examples
> “… (layer 31, token = ‘Berlin’, p = 0.936)” [output-Qwen3-8B-records.csv L610]  
> “… Berlin rank 1 at T=0.1 (0.99915) … rank 1 at T=2.0 (0.04186)” [output-Qwen3-8B.json L674–676, L741–744]

**Limitations & Data Quirks**
- Copy-collapse flag relies on an entropy fallback; first True at L31 is not a prompt echo (top‑1 not in prompt tokens). This can conflate low‑entropy semantic collapse with copy collapse.
- Final layers show rising rest_mass and punctuation tokens; while within acceptable bounds (<0.3), they can inflate entropy and obscure confidence comparisons across layers.
- No per‑variant per‑layer logs in JSON; cannot measure collapse‑layer shifts across prompt variants.

**Model Fingerprint**
“Qwen3‑8B: collapse at L 31; ΔH = 0.00 bits; final entropy 3.12 bits; ‘Berlin’ first enters top‑5 at L29 and dominates thereafter.”

---
Produced by OpenAI GPT-5
