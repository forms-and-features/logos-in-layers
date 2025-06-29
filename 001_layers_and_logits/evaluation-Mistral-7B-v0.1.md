## 1. Overview

Mistral-7B-v0.1 (≈7 B parameters, 32 transformer blocks) was probed on 29 Jun 2025 with the layer-by-layer lens script.  The probe records the per-layer entropy and top-k predictions for the next token after the prompt "Give the city name only, plain text. The capital of Germany is called simply", together with diagnostics, auxiliary test prompts and a small temperature sweep.

## 2. Method sanity-check

The JSON diagnostics confirm that the probe applied the intended RMS norm-lens and used the model's real ln1 parameters on the embeddings:

> "first_block_ln1_type": "RMSNormPre", [L8]  
> "layer0_norm_fix": "using_real_ln1_on_embeddings", [L12]

The `context_prompt` string ends exactly with "called simply" and has no trailing space (L15).  `L_copy`, `L_semantic` and `delta_layers` fields are present (all `null`, 25, `null` respectively) confirming the script attempted both collapse metrics.  In the first four layers the top-1 token never becomes "called" or "simply", so the stricter copy-collapse rule did not trigger.

## 3. Quantitative findings

Layer | Entropy (bits) | Top-1 token
---|---|---
0  | 14.40 | ******
1  | 14.13 | ❶
2  | 13.96 | ❶
3  | 13.86 | eston
4  | 13.80 | simply
5  | 13.65 | simply
6  | 13.67 | simply
7  | 13.74 | plain
8  | 13.75 | olas
9  | 13.76 | anel
10 | 13.77 | anel
11 | 13.77 | inho
12 | 13.77 | ifi
13 | 13.81 | ív
14 | 13.74 | ív
15 | 13.66 | simply
16 | 13.41 | simply
17 | 13.15 | simply
18 | 12.17 | simply
19 |  4.18 | simply  
20 |  3.37 | simply  
21 |  5.39 | simply  
22 |  7.81 | simply  
23 |  4.65 | simply  
24 |  5.24 | simply  
**25** | **2.02** | **Berlin**  
26 |  1.63 | Berlin  
27 |  1.24 | Berlin  
28 |  1.44 | Berlin  
29 |  2.54 | Berlin  
30 |  2.38 | Berlin  
31 |  3.05 | Berlin  
32 |  3.61 | Berlin

No layer satisfied the copy-collapse threshold (p > 0.90 on a prompt token), so **L_copy = n/a**.  The semantic answer appears first at layer 25 (L_semantic = 25).  Δ cannot be computed.

## 4. Qualitative patterns & anomalies

From layer 4 onward the model locks onto the filler word "simply", and its probability climbs steadily to 0.77 at L20 (> 20: "simply", 0.77) [L20].  This long filler anchoring spans 21 layers before being supplanted by the correct answer.  Because p never exceeds 0.9 it evades the strict copy-collapse rule, yet it behaves like a classic copy-reflex in the Tuned-Lens sense (2303.08112).  The transition at L25 is abrupt: entropy plummets to 2 bits and "Berlin" overtakes with 45 % probability (L26).

Temperature exploration shows extreme confidence at τ = 0.1 (p≈0.999 on "Berlin") but a flat 12.2-bit distribution at τ = 2.0, indicating the unembedded logits are well-calibrated (L775-802).

The auxiliary prompt "Berlin is the capital of" yields p = 0.90 for "Germany" with entropy ≈ 0.95 bits [L118-126], showing that when the "one-word" instruction is absent the model's collapse layer can occur much earlier in the causal graph (outside the logged stack).

Checklist
- RMS lens? ✓  
- LayerNorm bias removed? ✓ (not applicable, RMS)  
- Punctuation anchoring? ✓ (quote marks dominate top-k after L25)  
- Entropy rise at unembed? ✓ (entropy rises from 1.24 → 3.61 between L27 and final)  
- Punctuation / markup anchoring? ✓  
- Copy reflex? ✗ (threshold not met)  
- Grammatical filler anchoring? ✓ ("simply" L4-L24)

## 5. Tentative implications for Realism ↔ Nominalism

1. Does the 21-layer persistence of a grammatical filler imply that early transformer blocks prioritise relational or discourse cues over named entities?
2. Could the sharp switch at L25 correspond to an attention head that first reads the subject token "Germany" and injects the capital-city relation?  Mapping heads in that band may clarify.
3. Why does entropy re-expand after the answer stabilises—does the unembed layer inject stylistic alternatives (quote marks) that re-open the distribution?
4. Would lowering the repetition-penalty or modifying the prompt to exclude "simply" shift the semantic collapse earlier, revealing a more nominalist pathway?

## 6. Limitations & data quirks

Tokenisation artefacts (e.g. "******", "❶") in the early layers complicate strict copy detection.  The missing `L_copy` may be a false negative given the prolonged "simply" plateau.  CSV quoting issues obscure some punctuation tokens.  All observations derive from a single prompt and temperature; generality is unknown.

## 7. Model fingerprint

"Mistral-7B-v0.1: no copy-collapse; semantics emerges at L 25; final entropy 3.6 bits; 'Berlin' stable from L25 onward."

---
Produced by OpenAI o3
