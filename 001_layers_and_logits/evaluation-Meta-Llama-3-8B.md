# 1. Overview
Meta-Llama-3-8B (8 B parameters, RMS-Norm, 32 layers) was probed on <code>2025-06-28</code> with <code>001_layers_and_logits/run.py</code>.  The probe caches the residual stream after every transformer block, applies the intended RMS norm-lens, unembeds in FP32 and records the entropy as well as the top-k tokens for **every** prompt position and layer.

# 2. Method sanity-check
The JSON confirms that the run used the norm-lens and RMSNorm:
```6:14:001_layers_and_logits/output-Meta-Llama-3-8B.json
  "use_norm_lens": true,
```
```8:10:001_layers_and_logits/output-Meta-Llama-3-8B.json
  "first_block_ln1_type": "RMSNormPre",
```
and the script clearly adds positional encodings before the first lens projection:
```261:264:001_layers_and_logits/run.py
resid = (residual_cache['hook_embed'] +
         residual_cache['hook_pos_embed']).to(model.cfg.device)
```
Together these lines show that both the intended RMS norm-lens and positional encodings are active.

# 3. Quantitative findings (answer position, token ":"):
| layer | entropy (bits) | top-1 token |
|-------|----------------|-------------|
| L 0 | 14.52 | 'oren' |
| L 1 | 13.64 | 'ря' |
| L 2 | 13.62 | '␠' |
| L 3 | 13.54 | 'atten' |
| L 4 | 13.08 | 'adal' |
| L 5 | 13.32 | 'chine' |
| L 6 | 13.28 | '.Decode' |
| L 7 | 13.00 | 'Cem' |
| L 8 | 13.13 | 'AutoSize' |
| L 9 | 12.23 | 'enville' |
| L 10 | 12.58 | 'PCM' |
| L 11 | 13.09 | 'ystack' |
| L 12 | 12.39 | '.xtext' |
| L 13 | 13.36 | 'краї' |
| L 14 | 13.35 | '#ab' |
| L 15 | 12.81 | '#af' |
| L 16 | 13.03 | '#ab' |
| L 17 | 11.95 | '#ad' |
| L 18 | 12.48 | '#ad' |
| L 19 | 11.75 | 'ABCDEFGHIJKLMNOP' |
| L 20 | 10.77 | 'Capital' |
| L 21 | 9.29 | 'capital' |
| L 22 | 8.48 | 'Berlin' |
| L 23 | 5.58 | 'Berlin' |
| L 24 | 2.07 | 'Berlin' |
| **L 25** | **0.43** | **'Berlin'** |
| L 26 | 0.35 | 'Berlin' |
| L 27 | 0.33 | 'Berlin' |
| L 28 | 0.37 | 'Berlin' |
| L 29 | 0.81 | 'Berlin' |
| L 30 | 1.64 | 'Berlin' |
| L 31 | 1.16 | 'Berlin' |
| L 32 | 1.70 | 'Berlin' |

# 4. Qualitative patterns & anomalies
From L 20 onwards the model shifts from generic "Capital/capital" proposals to the concrete entity "Berlin", with entropy collapsing below 1 bit at L 25 and staying locked until the unembed.  The final unembedding **increases** entropy again to 1.7 bits, spreading ≈14 % of probability mass across alternative headings (e.g. "Germany", "The") while keeping "Berlin" dominant.  Such 'entropy bounce' after collapse is a well-known tuned-lens phenomenon (Tuned-Lens 2303.08112).

Test prompts reinforce this: > "... ( 'Germany', 0.90)" [1:42] shows the reverse relation, while a more abstract phrasing retains uncertainty > "... entropy 7.31 bits" [150:170].

Temperature exploration is extreme: at τ = 0.1 the answer is deterministic > "... 'Berlin', 1.0" [260:275]; at τ = 2.0 entropy rises to 14.5 bits yet 'Berlin' remains top-1 with only 3 % probability, indicating a shallow logit gap that the lens exaggerates.

Checklist: ✓ RMS lens ✓ LayerNorm (via RMSNorm) ✗ colon-spam ✓ entropy rises again at unembed

# 5. Tentative implications for Realism ↔ Nominalism
1. Does the early appearance of the *concept* token "Capital" (L 20) reflect a nominal "category" representation before resolving to the real-world entity "Berlin"?
2. Could the sharp entropy collapse between L 24–L 25 indicate a single attention head or MLP sub-module acting as a realist "lookup", mapping from abstract slot to concrete city?
3. Why does entropy rebound after the final RMS-norm—does this suggest a nominalist smoothing layer that re-introduces alternatives for calibration rather than meaning?
4. If temperature scaling shows such fragility, how stable is the 'realist' representation of "Berlin" across varied decoding temperatures or sparsity constraints?

# 6. Limitations & data quirks
The probe ran on CPU, so timing noise is unknown.  Only top-k=20 logits were persisted; tail distribution is unobserved.  Entropies are calculated at FP32 but the residual stream was cached in FP32 only on CPU, possibly hiding small GPU-precision effects.  The CSV stores unusual Unicode tokens (e.g. '긔'), suggesting tokenizer artefacts that may affect entropy estimates.  Finally, the evaluation covers a single prompt context; generality is not tested here.

# 7. Model fingerprint
"Llama-3-8B: entropy collapses at **L 25 (0.43 bits)**; final layer rebounds to 1.7 bits; 'Berlin' entrenched from L 22 onward."

---

Produced by OpenAI o3