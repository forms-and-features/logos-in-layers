# Evaluation Report: meta-llama/Meta-Llama-3-8B

*Run executed on: 2025-06-29 21:02:18*

## 1. Overview
meta-llama/Meta-Llama-3-8B is an 8-billion-parameter base model inspected on 2025-06-29 by the layer-wise logit-lens probe.  The probe records the entropy and top-k logits for every layer's **pure next-token** distribution, flags "copy-collapse" (prompt echo) and the first semantic emergence of the ground-truth answer, and stores the results in structured JSON/CSV artefacts for later comparison.

## 2. Method sanity-check
The script prints that a normalised (RMS) lens is applied and that positional embeddings are handled correctly in rotary form:
```304:304:001_layers_baseline/run.py
print("Using NORMALIZED residual stream (RMS + learned scale)")
```
```335:336:001_layers_baseline/run.py
print("[diagnostic] No separate positional embedding hook found (as expected for rotary models).")
```
The JSON confirms the context prompt terminates precisely with "called simply" and that collapse metrics are emitted:
```15:20:001_layers_baseline/run-2025-06-29-21-02/output-Meta-Llama-3-8B.json
"context_prompt": "Give the city name only, plain text. The capital of Germany is called simply",
"L_copy": null,
"L_semantic": 25,
```
Layers 0-3 do **not** predict "called" or "simply" (see Section 3), so no early copy-reflex is detected.  All three diagnostic keys (`L_copy`, `L_semantic`, `delta_layers`) are present.

## 3. Quantitative findings
| Layer | Entropy (bits) | Top-1 token |
|-------|---------------|-------------|
| L 0 | 13.04 | 泛 |
| L 1 | 12.82 | mente |
| L 2 | 12.47 | tics |
| L 3 | 13.30 | tones |
| L 4 | 12.99 | tones |
| L 5 | 8.93 | |
| L 6 | 13.35 | rops |
| L 7 | 12.92 |  bul |
| L 8 | 12.96 | urement |
| L 9 | 12.99 |单 |
| L 10 | 13.28 | biased |
| L 11 | 13.11 |  Gott |
| L 12 | 13.40 | LEGAL |
| L 13 | 13.35 |  Freed |
| L 14 | 12.90 |  simply |
| L 15 | 11.49 |  simply |
| L 16 | 9.50 |  simply |
| L 17 | 11.92 |  simply |
| L 18 | 7.95 |  simply |
| L 19 | 11.79 |  simply |
| L 20 | 12.48 | ' |
| L 21 | 12.38 | ' |
| L 22 | 13.05 |  simply |
| L 23 | 13.09 |  simply |
| L 24 | 12.33 |  capital |
| **L 25** | **7.84** | **Berlin** |
| L 26 | 2.02 | Berlin |
| L 27 | 3.44 | Berlin |
| L 28 | 4.95 | Berlin |
| L 29 | 7.87 | Berlin |
| L 30 | 4.73 | Berlin |
| L 31 | 3.78 | Berlin |
| L 32 | 2.96 | Berlin |

`L_copy` not observed; `L_semantic` = 25.  Δ = n/a (no copy-collapse layer identified).

## 4. Qualitative patterns & anomalies
From layers 14-19 the model persistently predicts the filler token " simply", suggesting lexical anchoring before true semantic collapse.  The ground-truth answer "Berlin" appears abruptly at layer 25 and stabilises, with entropy plunging from 12.33 bits (L 24) to 7.84 bits (L 25) and further to 2.02 bits at L 26.  This delayed semantic emergence without an earlier high-confidence copy echo indicates a **long semantic search without copy-reflex**.

For the external probe "Berlin is the capital of" the model answers "Germany" with 0.90 probability and entropy ≈ 0.93 bits, confirming that removal of the "one-word only" instruction does not delay answer emergence > "... (" Germany", 0.895)" [L136].  Other paraphrases with "simply" also yield Berlin as rank-1 with moderate entropy (e.g. 0.37 at "called simply").  Temperature sweep shows extreme sharpening at τ = 0.1 (entropy < 0.001) and diffuse uncertainty at τ = 2.0 (entropy ≈ 13.9), matching logit-lens expectations.

Checklist:
- RMS lens? ✓  
- LayerNorm bias removed? n.a. (RMS)  
- Punctuation anchoring? ✗  
- Entropy rise at unembed? ✓ (drop from 13 → 2 bits)  
- Punctuation / markup anchoring? ✗  
- Copy reflex? ✗  
- Grammatical filler anchoring? ✓ (" simply" dominates L14-L19)

## 5. Tentative implications for Realism ↔ Nominalism
1. Does the absence of a copy-echo layer in Meta-Llama-3-8B indicate that lexical realism emerges only after deeper semantic consolidation rather than through surface pattern matching?
2. Could the six-layer gap between " simply" anchoring and the Berlin collapse suggest an internal curriculum from discourse-level scaffolding to entity grounding?
3. If rotary positional encoding delays early token identity information (no additive position hook), does this encourage semantic rather than copy-based shortcuts?
4. Does the sharp entropy cliff (L25-L26) hint at a phase-transition-like mechanism where a small subset of attention heads resolves the capital-city relation?

## 6. Limitations & data quirks
All inference ran on CPU float-32, so timing of collapse layers may shift under mixed-precision GPU weights.  CSV top-k truncation to 20 tokens hides lower-rank distribution mass.  Non-ASCII artefacts (e.g. "泛", "单") in early layers could reflect tokenizer quirks rather than meaningful activations.  The probe analyses a single prompt; generality across topics is untested.

## 7. Model fingerprint
"Meta-Llama-3-8B: first Berlin at L 25, no copy-collapse observed, final entropy ≈ 3 bits."

---
Produced by OpenAI o3

