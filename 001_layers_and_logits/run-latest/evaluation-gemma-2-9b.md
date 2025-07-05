# Evaluation Report: google/gemma-2-9b

## 1. Overview
Google Gemma-2-9B (42 transformer layers, 9 B parameters) was probed on 2025-07-05 with a layer-wise logit-lens sweep.  The probe captures per-layer entropy and top-k predictions for the next token after the prompt, flags copy-collapse vs semantic emergence, and records diagnostic metadata.

## 2. Method sanity-check
The diagnostics block confirms that the run uses RMS-norm lens with rotary positional encodings:
> "use_norm_lens": true  [line 802]  
> "layer0_position_info": "token_only_rotary_model"  [line 814]

`context_prompt` ends exactly with "called simply" (no trailing space)  [line 816].  Implementation flags `L_copy`, `L_copy_H`, `L_semantic`, `delta_layers`, `unembed_dtype` etc. are all present (lines 818-827).  
Pure-next-token CSV marks `copy_collapse = True` for layers 0–13 (row 2 → row 14), so copy-reflex ✓.  The first such row shows  
layer = 0, token₁ = "simply", p₁ ≈ 0.999999, token₂ = "simply", p₂ ≈ 8 × 10⁻⁷ — ✓ rule satisfied.

## 3. Quantitative findings
Layer-by-layer next-token predictions:

| Layer | Entropy (bits) | Top-1 token |
|-------|---------------|-------------|
| L0 | 0.00002 | 'simply' |
| L1 | 0.00000 | 'simply' |
| L2 | 0.00012 | 'simply' |
| L3 | 0.00029 | 'simply' |
| L4 | 0.00046 | 'simply' |
| L5 | 0.00115 | 'simply' |
| L6 | 0.04886 | 'simply' |
| L7 | 0.04580 | 'simply' |
| L8 | 0.05191 | 'simply' |
| L9 | 0.11989 | 'simply' |
| L10 | 0.17904 | 'simply' |
| L11 | 0.57932 | 'simply' |
| L12 | 0.73448 | 'simply' |
| L13 | 0.97499 | 'simply' |
| L14 | 2.03478 | 'simply' |
| L15 | 11.24841 | 'simply' |
| L16 | 15.04542 | 'simply' |
| L17 | 15.81262 | 'simply' |
| L18 | 15.84011 | 'simply' |
| L19 | 15.46481 | 'simply' |
| L20 | 16.20092 | 'simply' |
| L21 | 16.49104 | 'simply' |
| L22 | 15.95326 | 'simply' |
| L23 | 15.25627 | 'simply' |
| L24 | 13.82752 | 'simply' |
| L25 | 13.76875 | 'simply' |
| L26 | 10.05885 | 'simply' |
| L27 | 7.16313 | '"' |
| L28 | 3.33842 | 'simply' |
| L29 | 1.19789 | '"""' |
| L30 | 0.91896 | '"""' |
| L31 | 0.71662 | '"""' |
| L32 | 0.88265 | '"""' |
| L33 | 0.74182 | '"""' |
| L34 | 1.21587 | '"""' |
| L35 | 0.95264 | '"""' |
| L36 | 1.27162 | '"""' |
| L37 | 1.28947 | '"""' |
| L38 | 1.08376 | '"""' |
| L39 | 0.89198 | '"""' |
| L40 | 0.85639 | '"""' |
| L41 | 1.49753 | '"""' |
| **L42** | **0.33152** | **'Berlin'** |

ΔH (bits) = H(L_copy) − H(L_semantic) ≈ 0.00002 − 0.33152 = -0.331 bits (entropy rises after copy collapse).

Confidence milestones:  p > 0.30 at layer 42;  p > 0.60 at layer 42;  final-layer p = 0.94.

## 4. Qualitative patterns & anomalies
The model shows immediate copy-reflex: layers 0–13 echo "simply" from the prompt with p > 0.99, confirming a strong lexical copy path.  Entropy remains < 1 bit until layer 13, then spikes (11.2 bits at L15) as copy fades and the stack searches for content.  Mid-stack (L27–L34) punctuation and quotation marks dominate the top-1 slot — a classic filler-token anchoring reported in Tuned-Lens (2303.08112).  Berlin first enters the top-5 at layer 35 (p ≈ 0.0011) and steadily climbs (0.009 at L38) before winning the argmax at L42.

Negative control "Berlin is the capital of" shows semantic leakage: Berlin rank 9 (p ≈ 0.0017) while the model replies " Germany" rank 1 (p ≈ 0.88)  [JSON lines 730-745].  Thus knowledge of the answer is partially activated even when the answer word is in the prompt.

Important-word trajectory in records.csv mirrors this: "Germany" tokens saturate early positions across layers with negligible probability mass, while "capital" appears in mid-stack attention but never becomes top-1.  "Berlin" gains probability only after L30 and becomes dominant by L42, echoing the pure-next-token CSV pattern.

Removing the one-word instruction ("...is called simply") in test prompts accelerates collapse: for "Germany's capital city is called simply" the model already places Berlin at p = 0.56 (rank 1) without needing deep layers, suggesting instruction tokens slow convergence rather than knowledge retrieval.

Rest-mass sanity: spikes to 0.73 at L15 and 0.55 at L41 indicate precision loss during punctuation stages; after L_semantic rest_mass falls to 8 × 10⁻⁶, consistent with confident prediction.

Temperature robustness: at T = 0.1 Berlin rank 1 (p ≈ 0.98, entropy 0.13 bits); at T = 2 Berlin rank 1 (p ≈ 0.087, entropy 9.04 bits).  The model remains answer-focused but uncertainty broadens as expected.

Important-word timeline: "Berlin" first in top-5 at L35, stabilises by L38, dominates from L42.  "Germany" appears in top-5 of test prompts but never tops the next-token distribution here.  "capital" is salient mid-stack (records rows 22–28) but drops after L32.

Checklist:  
✓ RMS lens  (see first_block_ln1_type = RMSNorm)  
✓ LayerNorm bias removed (RMS model)  
✓ Entropy rise at unembed  
✗ FP32 un-embed promoted (flag false)  
✓ Punctuation / markup anchoring  
✓ Copy-reflex  
✗ Grammatical filler anchoring (few "is/the/a" top-1)

## 5. Limitations & data quirks
Large rest-mass (> 0.3) at L15 suggests the norm-lens scaling may under-represent low-probability tail during the copy-to-semantic transition.  Mid-stack punctuation dominance could be artefact of tokenisation quirks rather than genuine feature processing.  Entropy dip at final layer (0.33 bits) is still above theoretical minimum, indicating residual uncertainty.

## 6. Model fingerprint
Gemma-2-9B: copy collapse at L 0, semantic collapse at L 42; final entropy 0.33 bits; 'Berlin' emerges only in last layer.

---
Produced by OpenAI o3

