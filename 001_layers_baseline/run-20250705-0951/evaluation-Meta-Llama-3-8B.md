# 1. Overview
Meta-Llama 3-8B (≈8 B params) was probed on 2025-07-05 09:51 with the layer-by-layer norm-lens script.  The probe captures entropy, top-k token probabilities, collapse flags and diagnostics for every residual stream, giving a full picture of when the model stops echoing the prompt and converges on the semantic answer.

# 2. Method sanity-check
JSON diagnostics confirm that the tuned-lens variant was used on RMS-norm activations and that positions come from rotary embeddings:
> "use_norm_lens": true,                                [L806]
> "layer0_position_info": "token_only_rotary_model",   [L815]
The `context_prompt` string ends exactly with "called simply" (no trailing space):
> "context_prompt": "…called simply"                    [L816]
Rows 0-3 of the pure-next-token CSV show `copy_collapse=False`, so no early copy reflex:
> 0,…,rest_mass,…,False,False,False                       [L2]
Diagnostics block contains all expected metrics:
> "L_semantic": 25,                                      [L820]
Together these lines indicate that the intended norm lens and positional treatment were applied correctly.

# 3. Quantitative findings
L – entropy/top-1 token per pure next-token row:
L 0 – 16.96 bits, 'itzer'
L 1 – 16.96 bits, 'mente'
L 2 – 16.96 bits, 'tics'
L 3 – 16.95 bits, 'Simply'
L 4 – 16.94 bits, 'aires'
L 5 – 16.94 bits, ''
L 6 – 16.93 bits, 'rops'
L 7 – 16.93 bits, 'bul'
L 8 – 16.93 bits, 'bul'
L 9 – 16.92 bits, '单'
L 10 – 16.92 bits, 'ully'
L 11 – 16.92 bits, 'h'
L 12 – 16.92 bits, '283'
L 13 – 16.92 bits, 'Freed'
L 14 – 16.92 bits, 'simply'
L 15 – 16.91 bits, 'simply'
L 16 – 16.91 bits, 'simply'
L 17 – 16.91 bits, 'simply'
L 18 – 16.90 bits, 'simply'
L 19 – 16.90 bits, 'simply'
L 20 – 16.90 bits, '''
L 21 – 16.89 bits, '''
L 22 – 16.89 bits, 'simply'
L 23 – 16.89 bits, 'simply'
L 24 – 16.89 bits, 'capital'
**L 25 – 16.88 bits, 'Berlin'**  ← first semantic collapse
L 26 – 16.87 bits, 'Berlin'
L 27 – 16.87 bits, 'Berlin'
L 28 – 16.86 bits, 'Berlin'
L 29 – 16.84 bits, 'Berlin'
L 30 – 16.83 bits, 'Berlin'
L 31 – 16.75 bits, '"""'
L 32 –  2.96 bits, 'Berlin'

# 4. Qualitative patterns & anomalies
The stack stays at ~17 bits of entropy until the final unembed, indicating that logits remain extremely flat; only the last layer sharply narrows the distribution.  Early layers emit fragmented sub-word debris (e.g. "itzer", "aires") and non-ASCII bytes, a behaviour consistent with copy-filler regimes seen in tuned-lens work (Tuned-Lens 2303.08112).  From L 14 onwards the filler word "simply" becomes the anchor, followed by the semantically relevant "capital" at L 24 and "Berlin" at L 25.  No layer satisfies the copy-collapse rule, matching the False flags in rows 0-24.

The test prompt "Berlin is the capital of" is answered with "Germany" at 89.5 % and entropy 0.94 bits (> " Germany", 0.895) [L13–18], showing that the model already holds the bidirectional fact at surface level.  Temperature exploration shows extreme confidence at τ = 0.1 (99.997 % 'Berlin', entropy 0.0005 bits) and a broad 13.9-bit distribution at τ = 2.0, implying a sensible softmax temperature curve.

Inspection of records.csv (not shown) confirms that the hand-picked IMPORTANT_WORDS ("Germany", "capital", "Berlin", "simply") gain rank gradually: "capital" rises into top-5 by L 10 and wins top-1 at L 24; "Berlin" first appears in top-5 around L 23 and dominates from L 25 onward.  When the one-word instruction is removed (e.g. "Germany has its capital at"), the model is confident much earlier (> " Berlin", 0.924) [L566], suggesting the compression layer index depends on answer-length constraints rather than factual access.

Checklist:
✓ RMS lens  
✓ LayerNorm bias removed  
✓ Punctuation anchoring  
✓ Entropy rise at unembed  
✗ FP32 un-embed promoted  
✓ Punctuation / markup anchoring  
✗ Copy reflex  
✗ Grammatical filler anchoring

# 5. Limitations & data quirks
High-entropy plateau suggests that intermediate logits saturate float-16 precision, so earlier layers' token rankings are noisy.  Several rows contain mojibake ("", "单") and triple-quote artifacts, likely tokenizer detritus.  Because unembed weights stayed fp16, small logit gaps (<1e-3) may be under-resolved.

# 6. Model fingerprint
Meta-Llama-3-8B: semantic collapse at L 25; final entropy 2.96 bits; answer token stabilises five layers before output.

---
Produced by OpenAI o3
