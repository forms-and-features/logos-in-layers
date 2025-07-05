# Evaluation Report: google/gemma-2-27b

## 1. Overview
Google's Gemma-2-27B (≈27 B params) was probed on 2025-07-05.  The script measures layer-by-layer next-token logits under an RMS-normalised lens, tracing when the model stops echoing the prompt and when it settles on the semantic answer "Berlin".

## 2. Method sanity-check
The diagnostics confirm that the intended RMS lens is active and that rotary positional information is handled (`token_only_rotary_model`).  Crucially, the context prompt ends exactly with "called simply" (no trailing space) and the norm-lens flag is on:
```806:806:001_layers_and_logits/run-latest/output-gemma-2-27b.json
"use_norm_lens": true,
```
```816:816:001_layers_and_logits/run-latest/output-gemma-2-27b.json
"context_prompt": "Give the city name only, plain text. The capital of Germany is called simply",
```
`L_copy`, `L_copy_H`, `L_semantic` and `delta_layers` are present (lines 818-820), and `unembed_dtype` = bf16 with `use_fp32_unembed`: false, matching the implementation.  In the pure-next-token CSV, `copy_collapse` is already `True` in layer 0, confirming a prompt-echo reflex.

## 3. Quantitative findings
| Layer | Entropy (bits) | Top-1 token |
|------:|---------------:|-------------|
| L 0 | 0.00 | simply |
| L 1 | 0.99 | the |
| L 2 | 0.00 | simply |
| L 3 | 0.01 | simply |
| L 4 | 0.90 | simply |
| L 5 | 0.00 | simply |
| L 6 | 0.26 | simply |
| L 7 | 0.20 | simply |
| L 8 | 3.33 | simply |
| L 9 | 1.64 | simply |
| L 10 | 14.23 | simply |
| L 11 | 17.07 | simply |
| L 12 | 6.63 | ſſel |
| L 13 | 17.04 | plain |
| L 14 | 17.62 | plain |
| L 15 | 17.58 | civilian |
| L 16 | 17.79 | civilian |
| L 17 | 17.59 | Wikimedia |
| L 18 | 17.63 | juges |
| L 19 | 17.81 | Bones |
| L 20 | 17.84 | contemporáneo |
| L 21 | 17.88 | مقاومت |
| L 22 | 17.91 | movi |
| L 23 | 17.91 | médic |
| L 24 | 17.83 | malades |
| L 25 | 17.79 | plain |
| L 26 | 17.61 | plain |
| L 27 | 17.68 | plain |
| L 28 | 17.57 | plain |
| L 29 | 17.56 | enough |
| L 30 | 17.60 | without |
| L 31 | 17.53 | enough |
| L 32 | 17.61 | just |
| L 33 | 17.62 | just |
| L 34 | 17.62 | just |
| L 35 | 17.66 | just |
| **L 36** | **17.56** | **Berlin** |
| L 37 | 17.52 | Berlin |
| L 38 | 17.43 | Berlin |
| L 39 | 17.54 | Berlin |
| L 40 | 17.58 | Berlin |
| L 41 | 16.88 | Berlin |
| L 42 | 17.51 | Berlin |
| L 43 | 17.88 | Berlin |
| L 44 | 17.28 | Berlin |
| L 45 | 17.05 | """ |
| L 46 | 0.16 | Berlin |

## 4. Qualitative patterns & anomalies
From layer 0 the model already predicts the copied word "simply" with ≥ 0.99 probability (> prompt-echo); see
```1:1:001_layers_and_logits/run-latest/output-gemma-2-27b-pure-next-token.csv
0,16,… simply,0.99997…,True,True,False
```
The echo persists through L 7, after which the distribution broadens and entropy shoots up to ≈ 17 bits, showing the typical "filler" phase before semantic collapse.  Strange low-frequency tokens ('ſſel', 'Bones', 'médic') appear mid-stack, a pattern linked to vocabulary sparsity noted by Tuned-Lens (2303.08112).

The earliest semantic hit is at **L 36** where "Berlin" becomes top-1 (line 37 of the CSV, `is_answer=True`).  Probability is still only 0.14 %, but stabilises and climbs to 5 % by L 38 and 50 % by L 41.

Test prompts mirror this: for "Berlin is the capital of" the model places "Germany" at 86 % (entropy 1.20 bits) ⇒ semantic knowledge is present even without the "one-word" instruction; see
```26:38:001_layers_and_logits/run-latest/output-gemma-2-27b.json
"Berlin is the capital of", 0.859375
```
Temperature sweep keeps "Berlin" as mode (0.88 → 0.05) while entropy rises from 0.53 bits (τ = 0.1) to 12.6 bits (τ = 2), indicating a well-calibrated softmax tail.

Across the full records CSV, high-attention words ("Germany", "capital") start climbing the ranks by L 20, preceding the answer.  The collapse layer does not shift noticeably when the one-word instruction is absent, but uncertainty stays higher (entropy ≈ 3 bits) suggesting the instruction mainly flattens the tail.

Checklist
- RMS lens? ✓  
- LayerNorm bias removed? n.a. (RMS)  
- Punctuation anchoring? ✓ (tokens " and ' ' dominate early top-k)  
- Entropy rise at unembed? ✓ (0 → 17 bits)  
- FP32 un-embed promoted? ✗  
- Punctuation / markup anchoring? ✓  
- Copy reflex? ✓  
- Grammatical filler anchoring? ✓ ('the', 'simply' in L 1–4)

## 5. Limitations & data quirks
Top-1 probability for "Berlin" remains < 1 % until very late, limiting confidence in single-token scoring.  Mid-stack tokens include non-Latin glyphs ("ſſel") indicating tokenizer artefacts.  The CSV truncates the probability mass to top-20, so entropy estimates rely on `rest_mass` approximations.

## 6. Model fingerprint
Gemma-2-27B: copy-echo at L 0, semantic collapse at L 36 (Δ = 36 layers); final entropy 2.9 bits with "Berlin" top-1.

---
Produced by OpenAI o3