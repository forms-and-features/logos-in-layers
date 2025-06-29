# 1. Overview
Mistralai **Mistral-7B-v0.1** (≈7 B parameters) was probed on 29 Jun 2025 using `run.py`.  
The script captures entropy and top-k token distributions for every residual stream – embeddings (L 0) through all 32 transformer blocks – plus diagnostic runs on alternative prompts and temperatures.

# 2. Method sanity-check
The JSON diagnostics confirm that the probe applies the intended RMS-norm lens and that positional information is rotary (token-only at L 0):
> "first_block_ln1_type": "RMSNormPre" [L9]  
> "layer0_position_info": "token_only_rotary_model" [L15]
No LayerNorm bias fix was needed and `use_norm_lens` is `true`; the CSV headers match the expected `layer,pos,token,entropy,…` format, indicating that residuals were normalised before unembedding.

# 3. Quantitative findings
| layer | entropy (bits) | top-1 token |
|------:|---------------:|-------------|
| 0 | 14.74 | acknow |
| 1 | 14.37 | dici |
| 2 | 14.11 | rapidly |
| 3 | 14.06 | bat |
| 4 | 13.72 | progressive |
| 5 | 13.69 | civilization |
| 6 | 13.64 | la |
| 7 | 13.48 | city |
| 8 | 13.55 | uty |
| 9 | 13.65 | city |
| 10 | 13.70 | surrounded |
| 11 | 13.72 | city |
| 12 | 13.78 | city |
| 13 | 13.61 | city |
| 14 | 13.62 | city |
| 15 | 13.69 | officially |
| 16 | 13.28 | city |
| 17 | 11.72 | city |
| 18 | 11.61 | city |
| 19 | 12.25 | city |
| 20 | 7.25 | Germany |
| 21 | 1.70 | Berlin |
| **22** | **0.99** | **Berlin** |
| 23 | 2.01 | Berlin |
| 24 | 2.71 | Berlin |
| 25 | 6.95 | Berlin |
| 26 | 10.68 | Berlin |
| 27 | 10.09 | Berlin |
| 28 | 10.84 | Berlin |
| 29 | 8.21 | 8 |
| 30 | 5.56 | 7 |
| 31 | 6.34 | 7 |
| 32 | 4.24 | 1 |

# 4. Qualitative patterns & anomalies
From L 0–20 entropy stays >7 bits with diffuse lexical guesses (mostly "city"-adjacent).  A sharp collapse begins at L 20, bottoms at L 22 (<1 bit) with "Berlin" 89 % probability, then entropy inflates again towards the unembed where numerals dominate.  Such mid-stack convergence followed by late divergence resembles the "Tuned-Lens" finding that knowledge crystallises mid-stack before being re-encoded downstream (2303.08112).

Numerical tokens ("1 2 7 8") overtake "Berlin" in the final layers and in the model's own top-20 prediction, hinting at a formatting bias or list-like prior in the head stack.  > "... top-1 '1' (0.19)" [L32]  This is absent at low temperature: > "... 'Berlin', 1.0" [L322].

Temperature exploration shows deterministic Berlin at τ = 0.1 (entropy ≈0) but a broad 12-bit spread at τ = 2.0 where Berlin still leads albeit at only 5.9 %.  The test prompts corroborate: the cloze "Berlin is the capital of ..." returns "Germany" with 64 % whereas paraphrases retain 15–32 % on Berlin.

Entropy rises again after L 24 despite Berlin staying top-1, implying that later layers add orthogonal information rather than reversing the answer.

Checklist:  
✓ RMS lens  
✓ LayerNorm n.a. (model uses RMSNorm)  
✗ Colon-spam  
✓ Entropy rise at unembed

# 5. Tentative implications for Realism ↔ Nominalism
1. Does the transient low-entropy plateau (L 21-24) correspond to a realist retrieval of factual content that is subsequently "nominalised" into format-agnostic tokens?  
2. Could the late numeric drift be an artefact of nominal formatting heads overriding realist semantic content when the prompt lacks an explicit answer style?  
3. If rotary position encoding only enters attention, does its absence at L 0 explain the need for several layers before factual collapse, suggesting spatial grounding precedes conceptual grounding?  
4. Would constraining softmax temperature during generation preserve the realism-aligned midpoint representation and mitigate nominal drift?

# 6. Limitations & data quirks
Only the top-20 probabilities are logged; unseen mass is aggregated, so entropy estimates after collapse (<1 bit) are upper bounds.  The run executed on CPU, potentially triggering different quantisation paths.  The prompt omits a trailing colon which might bias the final layers towards list tokens.  Finally, file timestamps—not explicit metadata—were used for run-date inference.

# 7. Model fingerprint
"Mistral-7B-v0.1: collapse at L 22; final entropy 4.2 bits; numerals outrank 'Berlin' at the unembed."

---
Produced by OpenAI o3
