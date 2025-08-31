# Evaluation Report: google/gemma-2-27b

*Run executed on: 2025-08-24 15:49:44*
**Overview**
- Model: google/gemma-2-27b (27B), run date 2025-08-24 15:49:44.
- Probe captures layer-wise next-token distributions via a norm lens, tracking entropy, copy/filler collapse, and emergence of the factual answer token.

**Method Sanity-Check**
The JSON confirms the intended norm lens and rotary positional handling are active: "use_norm_lens": true [L807], and "layer0_position_info": "token_only_rotary_model" [L816]. The context prompt ends exactly with “called simply” and no trailing space: "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply" [L4]. Diagnostics include required fields: "unembed_dtype": "torch.float32" [L809], "L_copy": 0 [L819], "L_copy_H": 0 [L820], "L_semantic": 46 [L821], "delta_layers": 46 [L822]. Copy-collapse was flagged early (τ=0.90, δ=0.05): first True row in pure CSV is layer 0: top‑1 ‘ simply’, p1 = 0.999976; top‑2 ‘ merely’, p2 = 7.52e‑06 (row 2 in CSV) — ✓ rule satisfied.

**Quantitative Findings**
- L 0 — entropy 0.000497 bits, top‑1 ‘ simply’
- L 1 — entropy 8.758229 bits, top‑1 ‘’
- L 2 — entropy 8.764487 bits, top‑1 ‘’
- L 3 — entropy 0.885666 bits, top‑1 ‘ simply’
- L 4 — entropy 0.618273 bits, top‑1 ‘ simply’
- L 5 — entropy 8.520256 bits, top‑1 ‘๲’
- L 6 — entropy 8.553085 bits, top‑1 ‘’
- L 7 — entropy 8.546973 bits, top‑1 ‘’
- L 8 — entropy 8.528743 bits, top‑1 ‘’
- L 9 — entropy 8.523797 bits, top‑1 ‘𝆣’
- L 10 — entropy 8.345239 bits, top‑1 ‘ dieſem’
- L 11 — entropy 8.492760 bits, top‑1 ‘𝆣’
- L 12 — entropy 8.324418 bits, top‑1 ‘’
- L 13 — entropy 8.222488 bits, top‑1 ‘’
- L 14 — entropy 7.876609 bits, top‑1 ‘’
- L 15 — entropy 7.792481 bits, top‑1 ‘’
- L 16 — entropy 7.974840 bits, top‑1 ‘ dieſem’
- L 17 — entropy 7.785551 bits, top‑1 ‘ dieſem’
- L 18 — entropy 7.299926 bits, top‑1 ‘ſicht’
- L 19 — entropy 7.527773 bits, top‑1 ‘ dieſem’
- L 20 — entropy 6.209991 bits, top‑1 ‘ſicht’
- L 21 — entropy 6.456000 bits, top‑1 ‘ſicht’
- L 22 — entropy 6.378438 bits, top‑1 ‘ dieſem’
- L 23 — entropy 7.010409 bits, top‑1 ‘ dieſem’
- L 24 — entropy 6.497042 bits, top‑1 ‘ dieſem’
- L 25 — entropy 6.994874 bits, top‑1 ‘ dieſem’
- L 26 — entropy 6.219814 bits, top‑1 ‘ dieſem’
- L 27 — entropy 6.700720 bits, top‑1 ‘ dieſem’
- L 28 — entropy 7.140120 bits, top‑1 ‘ dieſem’
- L 29 — entropy 7.574150 bits, top‑1 ‘ dieſem’
- L 30 — entropy 7.330207 bits, top‑1 ‘ dieſem’
- L 31 — entropy 7.565168 bits, top‑1 ‘ dieſem’
- L 32 — entropy 8.873556 bits, top‑1 ‘ zuſammen’
- L 33 — entropy 6.944745 bits, top‑1 ‘ dieſem’
- L 34 — entropy 7.738321 bits, top‑1 ‘ dieſem’
- L 35 — entropy 7.650662 bits, top‑1 ‘ dieſem’
- L 36 — entropy 7.657739 bits, top‑1 ‘ dieſem’
- L 37 — entropy 7.572387 bits, top‑1 ‘ dieſem’
- L 38 — entropy 7.553552 bits, top‑1 ‘ パンチラ’
- L 39 — entropy 7.232440 bits, top‑1 ‘ dieſem’
- L 40 — entropy 8.710523 bits, top‑1 ‘ 展板’
- L 41 — entropy 7.081689 bits, top‑1 ‘ dieſem’
- L 42 — entropy 7.056524 bits, top‑1 ‘ dieſem’
- L 43 — entropy 7.088928 bits, top‑1 ‘ dieſem’
- L 44 — entropy 7.568330 bits, top‑1 ‘ dieſem’
- L 45 — entropy 7.140568 bits, top‑1 ‘ Geſch’
- **L 46 — entropy 0.118048 bits, top‑1 ‘Berlin’**

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = 0.000497 − 0.118048 ≈ −0.118.

Confidence milestones:
- p > 0.30 at layer 46,  p > 0.60 at layer 46,  final‑layer p = 0.9841.

**Qualitative Patterns & Anomalies**
The early stack shows strong copy reflex on the final prompt token “simply” (layers 0, 3, 4 marked copy_collapse = True), then drifts into high‑entropy distributions dominated by non‑Latin artifacts and archaic German orthography (e.g., ‘ dieſem’, ‘ſicht’), consistent with lensing across partly formed features before consolidation (cf. tuned-lens, arXiv:2303.08112). By the final layer the answer consolidates sharply: “Berlin” becomes top‑1 with p = 0.984 (entropy 0.118 bits), and also lights up as top‑1 when lensing earlier positions near the answer span: “is” → “ Berlin” (0.999998) and “called” → “ Berlin” (0.999868) [rows 804–806 in records.csv].

Negative control shows no leakage: top‑5 for “Berlin is the capital of” are “ Germany” (0.8676), “ the” (0.0650), “ and” (0.0065), “ a” (0.0062), “ Europe” (0.0056) [L10–L31 in JSON], so no “Berlin” appears — no semantic leakage.

Temperature robustness: at T = 0.1 the model is extremely confident — “ Berlin” rank 1 (p = 0.9898; entropy 0.082 bits) [L670–L676]; at T = 2.0 “ Berlin” remains rank 1 but much flatter (p = 0.0492; entropy 12.631 bits) [L737–L743], consistent with the expected entropy inflation under high temperature.

Important-word trajectory: In the prompt positions, “capital” is already highly anchored at layer 0 (p ≈ 0.9994) [row 13 in records.csv], and “Germany” is salient (p ≈ 0.436) [row 15], while the NEXT-token prediction initially copies “simply” at L0 with overwhelming confidence (0.999976) [row 2 in pure CSV]. “Berlin” first enters any top‑5 only at the very end: at layer 46 it dominates across the tail of the prompt as well as NEXT (e.g., “… (‘Berlin’, 0.984)” [row 48 in pure CSV]; and “is”→“ Berlin”, 0.999998 [row 804 in records.csv]). This late emergence aligns with late‑stack semantic consolidation reported in tuned-lens analyses (arXiv:2303.08112).

Prompt variants show the expected behavior: removing the one‑word instruction and targeting the country instead yields “ Germany” top‑1 for “Give the country name only, plain text. Berlin is the capital of” (p = 0.449) [L437–L439]. For city‑targeted rephrasings without “simply”, e.g., “The capital city of Germany is named simply”, “ Berlin” remains top‑1 (p = 0.4316) [L483–L486]. The JSON does not report separate collapse‑layer indices for these variants, so any shift in L_semantic is n.a.

Rest‑mass sanity: Rest_mass falls steadily as the answer consolidates; final layer rest_mass = 1.99e‑07 (row 48 in pure CSV). No spikes after L_semantic (n.a. since L_semantic = final layer), suggesting no precision loss.

Quotes
> “context_prompt … called simply” [L4]; “use_norm_lens”: true; “layer0_position_info”: “token_only_rotary_model” [L807, L816].
> “Berlin is the capital of … (‘ Germany’, 0.8676, … ‘ Europe’, 0.0056)” [L10–L31].
> “T=0.1 … (‘ Berlin’, 0.9898) … T=2.0 … (‘ Berlin’, 0.0492)” [L670–L676; L737–L743].
> “… (‘Berlin’, 0.984)” [row 48 in pure CSV]; “is → ‘ Berlin’ (0.999998)” [row 804 in records.csv].

Checklist
- RMS lens?: ✓ (RMSNorm model; norm lens active) [L807, L810–L814]
- LayerNorm bias removed?: ✓/n.a. (“not_needed_rms_model”) [L812]
- Entropy rise at unembed?: ✓ (lens L46 0.118 bits vs final_prediction 2.886 bits) [row 48 in pure CSV; L826]
- FP32 un‑embed promoted?: ✓ (decoding in fp32; "unembed_dtype": "torch.float32") [L809]
- Punctuation / markup anchoring?: ✗ (NEXT-token early layers dominated by copy and orthographic artifacts, not punctuation)
- Copy‑reflex?: ✓ (copy_collapse True in layers 0–4; e.g., L0) [row 2 in pure CSV]
- Grammatical filler anchoring?: ✗ (no ‘is/the/a/of’ as top‑1 in L0–L5 of pure CSV)

**Limitations & Data Quirks**
- High mid‑stack rest_mass (~0.96–0.98), indicating heavy tail mass outside top‑20; lens readings are still coherent but sparse in top‑k coverage.
- Top‑1 tokens in mid‑layers include non‑Latin/orthographic artifacts (e.g., ‘ dieſem’, ‘ſicht’, ‘ パンチラ’), a common lens artifact rather than literal semantics.
- L_semantic coincides with the final layer; absence of post‑semantic layers prevents “max after L_semantic” rest‑mass checks.

**Model Fingerprint**
Gemma‑2‑27B: collapse at L 46; final lens entropy 0.118 bits; “Berlin” only appears as top‑1 at the last layer, with strong early copy reflex on “simply”.

---
Produced by OpenAI GPT-5
