# Evaluation Report: 01-ai/Yi-34B

*Run executed on: 2025-07-05 09:51:01*

# Overview  
01-ai/Yi-34B (≈34 B parameters) was probed on 2025-07-05.  The script performs a layer-by-layer logit-lens sweep, recording entropy and top-k tokens at every residual stream while applying the model's own RMSNorm-based lens.

# Method sanity-check  
The diagnostics block confirms the intended settings:  
> "use_norm_lens": true,  [L806]  
> "layer0_position_info": "token_only_rotary_model",  [L815]  
The context prompt line ends exactly with "called simply" and no trailing space:  
> "context_prompt": "… called simply",  [L3]  
No `copy_collapse` flag is set in layers 0-3 (see CSV lines 2-5), so no early copy reflex is detected.  The diagnostics object includes all expected keys – e.g. > "L_semantic": 44,  [L820] – confirming collection of L_copy/L_semantic and related flags.

# Quantitative findings  
| Layer | Entropy (bits) | Top-1 token |
|-------|---------------|-------------|
| L 0 | 15.96 | `Denote` |
| L 1 | 15.97 | `PREFIX` |
| L 2 | 15.96 | `ifiable` |
| L 3 | 15.96 | `.` |
| L 4 | 15.96 | `Marketing` |
| L 5 | 15.96 | `拐` |
| L 6 | 15.96 | `ostream` |
| L 7 | 15.95 | `tons` |
| L 8 | 15.95 | `tons` |
| L 9 | 15.95 | `诉讼代理人` |
| L 10 | 15.94 | `诉讼代理人` |
| L 11 | 15.95 | `诉讼代理人` |
| L 12 | 15.94 | `Marketing` |
| L 13 | 15.94 | `避` |
| L 14 | 15.93 | `避` |
| L 15 | 15.93 | `的人民` |
| L 16 | 15.93 | `诉讼代理人` |
| L 17 | 15.93 | `jo` |
| L 18 | 15.93 | `ERE` |
| L 19 | 15.93 | `ibo` |
| L 20 | 15.93 | `MDM` |
| L 21 | 15.91 | `MDM` |
| L 22 | 15.91 | `MDM` |
| L 23 | 15.88 | `.` |
| L 24 | 15.89 | `fing` |
| L 25 | 15.90 | `fing` |
| L 26 | 15.89 | `fing` |
| L 27 | 15.89 | `fing` |
| L 28 | 15.89 | `MDM` |
| L 29 | 15.88 | `MDM` |
| L 30 | 15.88 | `ANE` |
| L 31 | 15.87 | `MDM` |
| L 32 | 15.86 | `MDM` |
| L 33 | 15.85 | `MDM` |
| L 34 | 15.84 | `aday` |
| L 35 | 15.83 | `日` |
| L 36 | 15.82 | `geries` |
| L 37 | 15.80 | `踊` |
| L 38 | 15.78 | `什么呢` |
| L 39 | 15.74 | `simply` |
| L 40 | 15.74 | `capital` |
| L 41 | 15.71 | `"` |
| L 42 | 15.66 | `"` |
| L 43 | 15.59 | `"` |
| **L 44** | **15.34** | **`Berlin`** |
| L 45 | 15.23 | `Berlin` |
| L 46 | 14.68 | `Berlin` |
| L 47 | 14.62 | `Berlin` |
| L 48 | 14.95 | `Berlin` |
| L 49 | 14.58 | `Berlin` |
| L 50 | 14.54 | `Berlin` |
| L 51 | 14.60 | `Berlin` |
| L 52 | 14.86 | `Berlin` |
| L 53 | 14.88 | `Berlin` |
| L 54 | 14.91 | `Berlin` |
| L 55 | 14.93 | `Berlin` |
| L 56 | 15.05 | `Berlin` |
| L 57 | 14.69 | `Berlin` |
| L 58 | 14.85 | `Berlin` |
| L 59 | 14.09 | `[space]` |
| L 60 | 2.96 | `Berlin` |

# Qualitative patterns & anomalies  
Entropy remains near log₂|V| (~16 bits) for 40+ layers, indicating the model has not committed to any prediction.  Important prompt words start surfacing late: layer 39's top-1 is `simply`, layer 40 shows `capital`, and layer 41-43 are dominated by quotation-mark tokens—punctuation anchoring typical of RMS-norm lenses (> "… `"`", 0.00065) [L41].  The first semantic commitment appears at L 44 where 'Berlin' edges to 1.2 % probability (> "Berlin,0.0119" [L46]).  Probability climbs steadily, peaking above 7 % mid-stack and finally 56 % in the full model head (diagnostics final_prediction).  

Records CSV show "Germany" and "capital" entering the top-k around layers 39-43; "Germany" is already rank 3 at L 39 (> "Germany,0.00028" [L39]) and co-occurs with `simply`, suggesting converging context integration before semantic collapse.  Early layers are cluttered with placeholder or out-of-distribution fragments (`Denote`, `PREFIX`, Chinese legal jargon), consistent with observations in Tuned-Lens 2303.08112 that residual stream before attention heads is mostly lexical noise.  

Across the 14 auxiliary test prompts the model answers correctly: e.g. "Berlin is the capital of" → "Germany" with 0.85 probability [L14], confirming stored bidirectional relation and demonstrating direction-agnostic factual recall.  Removing the "one-word" instruction mildly delays certainty but not layer index—'Berlin' is already top-1 for "Germany's capital city is called simply ..." with 45 % at the model head (JSON block lines 25-34).  Temperature sweep shows near-determinism at τ = 0.1 (entropy ≈1.1 × 10-5, "Berlin" 99.999 %) and broadening to 12.5 bits at τ = 2.0 while still led by "Berlin", indicating a steep logit gap even after scaling.

Checklist  
✓ RMS lens  
n.a. LayerNorm bias removed  
✓ Punctuation anchoring  
✗ Entropy rise at unembed  
✗ FP32 un-embed promoted  
✓ Punctuation / markup anchoring  
✗ Copy reflex  
✗ Grammatical filler anchoring (layers 0-5 top-1 ∉ {is,the,a,of})

# Limitations & data quirks  
The probe stores only 20-way truncations; true entropy is inferred by adding `rest_mass` but small errors may persist.  Early-layer noise contains substantial non-Latin tokens, hinting tokenizer-specific artefacts.  Copy-collapse metric never fires, so the absence of L_copy may reflect threshold choice rather than genuine behaviour.

# Model fingerprint  
Yi-34B: semantic collapse at L 44; final entropy 2.96 bits; 'Berlin' probability rises monotonically to 56 %.

---
Produced by OpenAI o3

