# Evaluation Report: google/gemma-2-9b

*Run executed on: 2025-06-29 21:02:18*

## 1. Overview
Google's Gemma-2-9B (9 B parameters, 42 layers) was probed on 2025-06-29 using a norm-lens variant that traces the residual stream after every block and re-projects it through the unembed layer.  The run records entropy, top-k tokens and collapse flags for each layer, allowing us to locate both the first copy-collapse and the first semantic ("Berlin") layer.

## 2. Method sanity-check
The diagnostics block confirms the intended setup: `"use_norm_lens": true` and RMSNorm layers are detected (`"first_block_ln1_type": "RMSNormPre"`) [8-10].  Positional information is rotary only (`"layer0_position_info": "token_only_rotary_model"`) and the prompt matches the specification (no trailing space) [13-14].  CSV rows 2-5 show the top-1 token is **"simply"** with p≈1.0 through layers 0–3 – a clear copy-reflex signature.  Diagnostics expose `"L_copy": 0`, `"L_semantic": 42`, `"delta_layers": 42` [16-20].  All sanity checks therefore pass.

## 3. Quantitative findings
| Layer | Entropy (bits) | Top-1 token |
|-------|---------------|-------------|
| **0** | 0.00 | **simply** |
| 1 | 0.00 | simply |
| 2 | 0.00 | simply |
| 3 | 0.00 | simply |
| 4 | 0.00 | simply |
| 5 | 0.00 | simply |
| 6 | 0.01 | simply |
| 7 | 0.00 | simply |
| 8 | 0.00 | simply |
| 9 | 0.00 | simply |
| 10 | 0.01 | simply |
| 11 | 0.01 | simply |
| 12 | 0.00 | simply |
| 13 | 0.00 | simply |
| 14 | 0.00 | simply |
| 15 | 0.00 | simply |
| 16 | 0.02 | simply |
| 17 | 0.02 | simply |
| 18 | 0.21 | simply |
| 19 | 0.19 | simply |
| 20 | 0.81 | simply |
| 21 | 0.28 | simply |
| 22 | 0.10 | simply |
| 23 | 0.03 | simply |
| 24 | 0.01 | simply |
| 25 | 0.09 | simply |
| 26 | 0.01 | simply |
| 27 | 0.77 | " |
| 28 | 0.27 | simply |
| 29 | 0.81 | "" |
| 30 | 0.34 | "" |
| 31 | 0.09 | "" |
| 32 | 0.17 | "" |
| 33 | 0.04 | "" |
| 34 | 0.37 | "" |
| 35 | 0.04 | "" |
| 36 | 0.46 | "" |
| 37 | 0.22 | "" |
| 38 | 0.07 | "" |
| 39 | 0.02 | "" |
| 40 | 0.02 | "" |
| 41 | 1.01 | "" |
| **42** | 0.37 | **Berlin** |

Bold rows mark copy-collapse (L₀) and semantic collapse (L₄₂).  Δ = 42 layers.

## 4. Qualitative patterns & anomalies
Gemma shows a long copy-reflex (Δ = 42 layers), retaining the prompt word **"simply"** with near-certainty for over half the stack (e.g. 0.97 at layer 18 [L25]).  Entropy rises steadily once attention heads replace the adverb with punctuation (layer 27 """", entropy 0.77 bits [L28]) and later with quote tokens, suggesting stylistic detours before the model converges on **Berlin** at the final layer (0.93 prob, entropy 0.37 bits [L44]).  The test prompt "Berlin is the capital of" elicits **Germany** at 0.88 prob and entropy 0.96 bits [54-66], implying the semantic mapping is already internal but is vetoed for the "one-word" task until very late.  Removing the "one-word" instruction therefore shifts the effective collapse earlier to the immediate prediction.

Checklist:  
✓ RMS lens  
✓ LayerNorm bias removed (n.a. for RMS)  
✓ Punctuation anchoring (layers 27–36)  
✓ Entropy rise at unembed (≥1 bit at layer 41)  
✓ Punctuation / markup anchoring  
✓ Copy reflex  
✗ Grammatical filler anchoring

## 5. Tentative implications for Realism ↔ Nominalism
1. Does the 42-layer gap indicate that factual recall circuits remain suppressed by surface-form heuristics until late aggregation stages?
2. Could rotary-only positional encoding hinder early disambiguation between instruction and content, favouring nominal representations over realist semantics?
3. Does the late drop in entropy (1 → 0.37 bits) suggest a small set of competing name candidates rather than a distributed semantic manifold?
4. Would micro-finetuning on "one-word answer" tasks shorten Δ, implying copy-reflex originates in training distribution rather than architecture?

## 6. Limitations & data quirks
CSV tokens carry leading spaces; punctuation tokens are ambiguous (straight vs curly quotes), making copy- vs punctuation-collapse boundary fuzzy.  Entropy figures are dominated by the top token due to high copy probabilities, potentially exaggerating Δ.  Only a single prompt was probed; broader stimulus might reveal different collapse layers.

## 7. Model fingerprint
Gemma-2-9B: collapse at L 0 → 42; final entropy 0.37 bits; punctuation detour before 'Berlin' emerges.

---
Produced by OpenAI o3

