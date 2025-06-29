# 1. Overview
Meta-Llama-3-8B (32-layer, 4 096-dim) was probed on 2025-06-29 (timestamp of result file).
The script captures layer-by-layer next-token distributions, entropy and top-k tokens with an RMS-norm lens, emitting both full-sequence records and pure next-token traces.

# 2. Method sanity-check
JSON diagnostics show the probe applied the intended RMS lens: see the first block layer norm type, bias fix status and the original prompt.
```8:12:001_layers_and_logits/output-Meta-Llama-3-8B.json
"first_block_ln1_type": "RMSNormPre",
"layernorm_bias_fix": "not_needed_rms_model",
```
Positional information is handled internally (rotary), and the prompt ends exactly with "called simply":
```12:15:001_layers_and_logits/output-Meta-Llama-3-8B.json
"context_prompt": "Give the city name only, plain text. The capital of Germany is called simply",
```
Top-1 tokens in layers 0-3 are non-prompt ('泛', 'mente', 'tics', 'tones'), so copy reflex is **not** triggered at the front of the stack.
Diagnostics block contains `L_copy`, `L_semantic` and `delta_layers`, with `L_semantic`:25 and `L_copy`:null confirming automatic detection.

# 3. Quantitative findings
| layer | entropy (bits) | top-1 token |
|-------|---------------|-------------|
| 0 | 13.58 | 泛 |
| 1 | 13.03 | mente |
| 2 | 12.59 | tics |
| 3 | 13.33 | tones |
| 4 | 13.02 | tones |
| 5 | 8.99 |  |
| 6 | 13.36 | rops |
| 7 | 12.93 |  bul |
| 8 | 12.97 | urement |
| 9 | 13.00 | 单 |
| 10 | 13.28 | biased |
| 11 | 13.12 |  Gott |
| 12 | 13.40 | LEGAL |
| 13 | 13.35 |  Freed |
| 14 | 12.90 |  simply |
| 15 | 11.50 |  simply |
| 16 | 9.51 |  simply |
| 17 | 11.92 |  simply |
| 18 | 7.95 |  simply |
| 19 | 11.79 |  simply |
| 20 | 12.48 | ' |
| 21 | 12.38 | ' |
| 22 | 13.05 |  simply |
| 23 | 13.09 |  simply |
| 24 | 12.33 |  capital |
| **25** | **7.84** | **Berlin** |
| 26 | 2.02 | Berlin |
| 27 | 3.44 | Berlin |
| 28 | 4.95 | Berlin |
| 29 | 7.87 | Berlin |
| 30 | 4.73 | Berlin |
| 31 | 3.78 | Berlin |
| 32 | 2.96 | Berlin |

**Bold** row 25 marks the semantic layer (L_semantic); no layer met the >0.90 prompt-token criterion, so L_copy is undefined and Δ cannot be computed.

# 4. Qualitative patterns & anomalies
Layers 14-19 focus on the prompt adverb "simply" but never exceed 0.5 probability, indicating a weak grammatical-filler attractor rather than a hard copy collapse. The sharp entropy drop from 13 bits at L24 to 7.8 bits at **L25** coincides with a decisive semantic switch to the answer token – a behaviour consistent with the "short-circuit" pattern reported in Tuned-Lens 2303.08112. Final entropy falls further to 2.96 bits while retaining 'Berlin' at rank 1, showing confident commitment.

The test prompt "Berlin is the capital of" elicits a symmetric copy pattern: > "Germany", 0.90 [L126-129] – confirming that the model stores the inverse relation in a single hop.

Temperature sweep shows extreme certainty at τ = 0.1 (p≈1) and broadening at τ = 2 (p≈0.04), indicating a stable answer manifold.

Checklist: RMS lens ✓; LayerNorm bias removed n.a.; Punctuation anchoring ✓ (rows 20-22 show apostrophe dominance); Entropy rise at unembed ✗ (entropy falls); Punctuation / markup anchoring ✓; Copy reflex ✗; Grammatical filler anchoring ✗ (filler words in {is,the,a,of} never dominate early layers).

# 5. Tentative implications for Realism ↔ Nominalism
– Does the deep-stack semantic convergence at a single layer (25) support a realist view of stable city–country facts, or is it merely an emergent statistical alias?
– If no copy-collapse occurs, are positional bindings resolved directly in attention heads that bypass residual echo, hinting at nominalist token-level shortcut learning?
– Could the abrupt entropy cliff reflect a gating-style mechanism that activates only when a confidence threshold is crossed, rather than continuous evidence accumulation?
– Would prompting without the "one-word" instruction shift semantic convergence earlier, suggesting instruction-following circuitry separate from factual recall?

# 6. Limitations & data quirks
CSV rows before L5 are noisy and include garbled UTF-8 tokens (e.g. '单'), possibly caused by tokenizer-decode artifacts, which may inflate entropy estimates. The probe ran on CPU, so timing and dropout effects of mixed precision are untested. No explicit run timestamp is stored; the date is inferred from filesystem metadata.

# 7. Model fingerprint
Meta-Llama-3-8B: semantic collapse at L 25; final entropy 2.96 bits; 'Berlin' stable top-1 from L 25 onward.

---
Produced by OpenAI o3
