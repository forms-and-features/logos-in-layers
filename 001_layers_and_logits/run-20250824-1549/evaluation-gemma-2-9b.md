# 1. Overview

Google Gemma‑2‑9B (9B params), evaluated on the prompt “Give the city name only, plain text. The capital of Germany is called simply”. The probe captures layer‑by‑layer next‑token distributions with a norm lens, tracking entropy, copy/filler collapse, and the semantic collapse where “Berlin” becomes top‑1.

# 2. Method sanity‑check

Diagnostics confirm the intended norm lens and positional handling: “use_norm_lens: true” [JSON L807]; layer‑0 position info indicates a rotary/token‑only model, “layer0_position_info: token_only_rotary_model” [JSON L816]. The context prompt ends with “called simply” [JSON L817], no trailing space. The diagnostics block includes L_copy, L_copy_H, L_semantic, delta_layers, and implementation flags: device, use_norm_lens, use_fp32_unembed, unembed_dtype, layernorm_bias_fix, norm_alignment_fix, and layer0_norm_fix [JSON L803–822].

Copy‑collapse flag check: first row with copy_collapse = True → layer = 0, top‑1 = “ simply”, p1 = 0.9999993; top‑2 = “simply”, p2 = 7.73e‑07 (pure‑next‑token CSV row 2). ✓ rule satisfied.

# 3. Quantitative findings

| Layer | Finding |
|---|---|
| L 0 | entropy 1.672130e-05 bits, top-1 ' simply' |
| L 1 | entropy 6.941615e-08 bits, top-1 ' simply' |
| L 2 | entropy 3.137544e-05 bits, top-1 ' simply' |
| L 3 | entropy 0.000430 bits, top-1 ' simply' |
| L 4 | entropy 0.002116 bits, top-1 ' simply' |
| L 5 | entropy 0.002333 bits, top-1 ' simply' |
| L 6 | entropy 0.127902 bits, top-1 ' simply' |
| L 7 | entropy 0.033569 bits, top-1 ' simply' |
| L 8 | entropy 0.098417 bits, top-1 ' simply' |
| L 9 | entropy 0.102087 bits, top-1 ' simply' |
| L 10 | entropy 0.281391 bits, top-1 ' simply' |
| L 11 | entropy 0.333046 bits, top-1 ' simply' |
| L 12 | entropy 0.109330 bits, top-1 ' simply' |
| L 13 | entropy 0.137400 bits, top-1 ' simply' |
| L 14 | entropy 0.165772 bits, top-1 ' simply' |
| L 15 | entropy 0.734873 bits, top-1 ' simply' |
| L 16 | entropy 3.568274 bits, top-1 ' simply' |
| L 17 | entropy 3.099445 bits, top-1 ' simply' |
| L 18 | entropy 3.336716 bits, top-1 ' simply' |
| L 19 | entropy 1.382336 bits, top-1 ' simply' |
| L 20 | entropy 3.163440 bits, top-1 ' simply' |
| L 21 | entropy 1.866495 bits, top-1 ' simply' |
| L 22 | entropy 2.190102 bits, top-1 ' simply' |
| L 23 | entropy 3.181111 bits, top-1 ' simply' |
| L 24 | entropy 1.107039 bits, top-1 ' simply' |
| L 25 | entropy 2.118879 bits, top-1 ' the' |
| L 26 | entropy 2.371327 bits, top-1 ' the' |
| L 27 | entropy 1.842460 bits, top-1 ' the' |
| L 28 | entropy 1.226664 bits, top-1 ' "' |
| L 29 | entropy 0.315988 bits, top-1 ' "' |
| L 30 | entropy 0.134063 bits, top-1 ' "' |
| L 31 | entropy 0.046090 bits, top-1 ' "' |
| L 32 | entropy 0.062538 bits, top-1 ' "' |
| L 33 | entropy 0.042715 bits, top-1 ' "' |
| L 34 | entropy 0.090030 bits, top-1 ' "' |
| L 35 | entropy 0.023370 bits, top-1 ' "' |
| L 36 | entropy 0.074091 bits, top-1 ' "' |
| L 37 | entropy 0.082534 bits, top-1 ' "' |
| L 38 | entropy 0.033455 bits, top-1 ' "' |
| L 39 | entropy 0.046899 bits, top-1 ' "' |
| L 40 | entropy 0.036154 bits, top-1 ' "' |
| L 41 | entropy 0.176738 bits, top-1 ' "' |
| **L 42 – entropy 0.370067 bits, top‑1 ' Berlin'** |

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = 1.6721e‑05 − 0.3700671 = −0.3700503.

Confidence milestones: p > 0.30 at layer 42, p > 0.60 at layer 42, final‑layer p = 0.9298 (pure‑next‑token CSV row 49).

# 4. Qualitative patterns & anomalies

Early copy‑reflex of the final prompt token “ simply” saturates layers 0–5 (e.g., “ simply, 1.0000” [records.csv L38] and “ simply, 0.9999993” [records.csv L18]). Mid‑stack, the model shifts to grammatical fillers and markup: “ the” becomes top‑1 at L25–27 (pure CSV rows 32–34), followed by a long quote‑mark regime where the top‑1 is '"' through L28–41 (e.g., '"', 0.9951 [pure CSV row 38]). This punctuation anchoring before semantic resolution is consistent with tuned‑lens observations on surface‑form circuits (cf. Tuned‑Lens 2303.08112).

Negative control (prompt: “Berlin is the capital of”): top‑5 = “ Germany, 0.8766; the, 0.0699; modern, 0.0077; a, 0.0053; ' ', 0.0034” [JSON L14–31]. Semantic leakage: Berlin rank 9 (p = 0.00187) [JSON L46–48].

Important‑word trajectory (records.csv) shows strong identity retention for context tokens: “capital” at pos=11 is top‑1 with p≈1.0 at L0–1 (e.g., “ capital, 1.0” [L13], [L33]); “Germany” at pos=13 remains near‑certain (e.g., 0.9999990 at L2 [L54], 0.9999380 at L3 [L71]). The next‑token position (the probe target) is dominated by the copied “ simply” until deep layers [L18, L38]. “Berlin” does not enter the next‑token top‑20 until the final layer, where it jumps to rank 1 with p=0.9298 (pure CSV row 49). Semantically close variants appear in test prompts (e.g., “Hauptstadt” in top‑10: 0.0153 [JSON L418–420]).

Collapse‑layer index with vs. without the “one‑word” instruction cannot be determined from test prompts alone (no per‑layer logs for those); n.a. The diagnostics show semantic collapse at L_semantic = 42 [JSON L821], aligning with the model’s total depth (num_layers = 42 [JSON L912]).

Rest‑mass sanity: rest_mass peaks mid‑stack at 0.2563 (L18) and shrinks to ~1.1e‑05 at L42 (pure CSV rows 26 and 49), indicating no precision loss at collapse.

Temperature robustness: At T = 0.1, Berlin rank 1 (p = 0.9809; entropy 0.137 bits) [JSON L669–676]. At T = 2.0, Berlin rank 1 (p = 0.0893; entropy 9.001 bits) [JSON L737–743].

Important‑word trajectory (next‑token position): Berlin first enters any top‑5 at layer 42 and stabilises by 42; “Germany” and “capital” never appear in the next‑token top‑5 at any layer (pure CSV scan).

Checklist:
- RMS lens?: ✓ “first_block_ln1_type”: “RMSNorm” [JSON L810]
- LayerNorm bias removed?: ✓ “layernorm_bias_fix”: “not_needed_rms_model” [JSON L812]
- Entropy rise at unembed?: ✓ L41 0.1767 → L42 0.3701 bits (pure CSV rows 48–49)
- FP32 un‑embed promoted?: ✗ “use_fp32_unembed”: false [JSON L808]
- Punctuation / markup anchoring?: ✓ long '"' regime L28–41 (pure CSV rows 36–45)
- Copy‑reflex?: ✓ copy_collapse = True in layers 0–3 (pure CSV rows 2–5)
- Grammatical filler anchoring?: ✗ (layers 0–5 top‑1 is “ simply”; “ the” emerges only at L25–27)

# 5. Limitations & data quirks

- L_semantic coincides with the final layer (42), so collapse timing beyond unembed cannot be disambiguated further. No per‑layer logs for alternate test prompts, so collapse‑layer shifts across instructions are n.a.
- Mid‑stack rest_mass reaches 0.2563 at L18 but is near‑zero at L42; values >0.3 would indicate mis‑scale, which is not observed here.

# 6. Model fingerprint

“Gemma‑2‑9B: collapse at L 42; final entropy 0.37 bits; ‘Berlin’ emerges only at unembed with p≈0.93.”

---
Produced by OpenAI GPT-5

