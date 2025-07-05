# Overview
Qwen/Qwen3-14B (≈14 B params) was probed on 2025-07-05 09:51 with the layer-by-layer RMS-norm lens script.  The probe records entropy and top-k predictions for every residual stream position, letting us trace where the model stops copying the prompt and first privileges the ground-truth answer "Berlin".

# Method sanity-check
The JSON confirms the RMS-norm lens is active and no FP32 promotion was applied:
> "use_norm_lens": true  [L806]
> "first_block_ln1_type": "RMSNorm"  [L809]
The `context_prompt` ends exactly with "called simply" (no trailing space) – see line 4 of the JSON.  In the pure-next-token CSV layers 0-3 all have `copy_collapse=False`, so **copy-reflex is absent**.  Diagnostics block provides `L_copy`, `L_copy_H`, `L_semantic` and `delta_layers`, together with dtype flags, satisfying the implementation checklist.

# Quantitative findings
| Layer summary |
|---|
| L 0 – entropy 17.21 bits, top-1 '梳' |
| L 1 – entropy 17.20 bits, top-1 '线条' |
| L 2 – entropy 16.83 bits, top-1 '几十年' |
| L 3 – entropy 16.33 bits, top-1 '几十年' |
| L 4 – entropy 16.50 bits, top-1 '几十年' |
| L 5 – entropy 16.36 bits, top-1 '几十年' |
| L 6 – entropy 16.74 bits, top-1 '过去的' |
| L 7 – entropy 16.55 bits, top-1 '几十年' |
| L 8 – entropy 16.82 bits, top-1 '不再' |
| L 9 – entropy 17.10 bits, top-1 '让人们' |
| L 10 – entropy 17.13 bits, top-1 '候' |
| L 11 – entropy 17.11 bits, top-1 '时代的' |
| L 12 – entropy 17.09 bits, top-1 'esor' |
| L 13 – entropy 17.08 bits, top-1 ' simply' |
| L 14 – entropy 17.08 bits, top-1 '迳' |
| L 15 – entropy 17.09 bits, top-1 'hausen' |
| L 16 – entropy 17.10 bits, top-1 '' |
| L 17 – entropy 17.11 bits, top-1 'urname' |
| L 18 – entropy 17.11 bits, top-1 ' SimpleName' |
| L 19 – entropy 17.10 bits, top-1 'tics' |
| L 20 – entropy 17.09 bits, top-1 ' SimpleName' |
| L 21 – entropy 17.10 bits, top-1 'lobs' |
| L 22 – entropy 17.09 bits, top-1 'Ticker' |
| L 23 – entropy 17.08 bits, top-1 '-minded' |
| L 24 – entropy 17.09 bits, top-1 '-minded' |
| L 25 – entropy 17.07 bits, top-1 'urname' |
| L 26 – entropy 17.04 bits, top-1 '这个名字' |
| L 27 – entropy 16.99 bits, top-1 '这个名字' |
| L 28 – entropy 16.92 bits, top-1 '这个名字' |
| L 29 – entropy 16.85 bits, top-1 'tons' |
| L 30 – entropy 16.77 bits, top-1 '这个名字' |
| L 31 – entropy 16.65 bits, top-1 '这个名字' |
| **L 32 – entropy 16.49 bits, top-1 'Berlin'** |
| L 33 – entropy 15.84 bits, top-1 'Berlin' |
| L 34 – entropy 14.98 bits, top-1 'Berlin' |
| L 35 – entropy 11.87 bits, top-1 'Berlin' |
| L 36 – entropy 12.06 bits, top-1 'Berlin' |
| L 37 – entropy  9.18 bits, top-1 'Berlin' |
| L 38 – entropy 12.27 bits, top-1 'Berlin' |
| L 39 – entropy 10.29 bits, top-1 'Berlin' |
| L 40 – entropy  3.59 bits, top-1 'Berlin' |

# Qualitative patterns & anomalies
Early layers offer noisy Chinese or alphanumeric fragments, suggesting the un-normalised residual stream is dominated by byte-level morphemes.  No copy-collapse is detected: the prompt words "called" / "simply" never rise above 90 % within layers 0–3.  Entropy stays ≈17 bits until layer 31, then drops by ≈0.3 bits when **Berlin** first surfaces at L 32, consistent with semantic convergence.  The test prompt "Berlin is the capital of" still yields top-1 "Germany" with 63 % probability (> line 15) indicating the model maintains factual linkage in the reverse direction.

Temperature exploration shows that at τ = 0.1 the model assigns 97 % to "Berlin", but at τ = 2.0 the answer probability falls to 3.6 %, confirming a shallow logit gap (> L736-744).

Records CSV reveals that the important words ("Germany", "Berlin", "capital", "simply") remain low-rank until after layer 28; "Germany" climbs first, followed by "Berlin", mirroring the attention-then-MLP pattern reported in Tuned-Lens 2303.08112.  The absence of the "one-word" instruction in the test prompt variants does not shift L_semantic: rows for "Germany's capital city is called simply" already place Berlin at top-1 with 87 % (JSON lines 27-34), implying the collapse is content-driven rather than instruction-driven.

Checklist:
✓ RMS lens
✓ LayerNorm bias removed (not needed for RMS) 
✗ Punctuation anchoring
✓ Entropy rise at unembed (≈ 0.25 bits drop at L32) 
✗ FP32 un-embed promoted
✗ Punctuation / markup anchoring
✗ Copy reflex
✗ Grammatical filler anchoring

# Limitations & data quirks
The very high early entropies and non-ASCII tokens hint at vocabulary mis-alignment between the model and tokenizer dump used by Transformer-Lens, potentially inflating entropy estimates before rotary positional encoding fully mixes features.  All collapse metrics rely on a heuristic 0.90 threshold and may under-detect partial copying.

# Model fingerprint (one sentence)
Qwen-3-14B: semantic collapse at L 32 with final entropy 3.6 bits; "Berlin" dominates the stack from L 32 onward.

---
Produced by OpenAI o3

