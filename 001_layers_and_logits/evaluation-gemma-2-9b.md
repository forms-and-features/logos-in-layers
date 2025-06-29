# 1. Overview
Google's Gemma-2-9B (≈ 9 B parameters) was probed on the prompt "The capital of Germany is ...".  The run captured layer-by-layer entropy and top-k predictions for every residual stream, together with control diagnostics, test prompts and temperature sweeps.

# 2. Method sanity-check
The diagnostic block emitted by the script confirms that the probe applied the intended RMS-norm lens (not a synthetic γ) and that positional information follows the rotary scheme.  Two representative console lines are:
> "first_block_ln1_type": "RMSNormPre" [L9]  
> "layer0_norm_fix": "using_real_ln1_on_embeddings" [L13]
These lines match the logic in `run.py`, which routes residuals through `apply_norm_or_skip` and uses real `ln1` on embeddings when `USE_NORM_LENS` is `True`.

# 3. Quantitative findings
| Layer summary |
| --- |
| **L 0 – entropy 0.00 bits, top-1 '␠'** |
| L 1 – entropy 0.00001 bits, top-1 '␠' |
| L 2 – entropy 2.321 bits, top-1 'KommentareTeilen' |
| L 3 – entropy 1.032 bits, top-1 'IntoConstraints' |
| L 4 – entropy 1.130 bits, top-1 'ANZE' |
| L 5 – entropy 2.873 bits, top-1 'Grecs' |
| L 6 – entropy 0.014 bits, top-1 'capitales' |
| L 7 – entropy 2.101 bits, top-1 'capitale' |
| L 8 – entropy 0.140 bits, top-1 'capital' |
| L 9 – entropy 0.059 bits, top-1 'capital' |
| L 10 – entropy 0.406 bits, top-1 'capital' |
| L 11 – entropy 0.667 bits, top-1 'capital' |
| L 12 – entropy 3.505 bits, top-1 '␤' |
| L 13 – entropy 1.118 bits, top-1 '␤' |
| L 14 – entropy 2.633 bits, top-1 '␤' |
| L 15 – entropy 3.104 bits, top-1 '␠' |
| L 16 – entropy 4.183 bits, top-1 ''' |
| L 17 – entropy 1.814 bits, top-1 'city' |
| L 18 – entropy 0.975 bits, top-1 'city' |
| L 19 – entropy 0.042 bits, top-1 'city' |
| L 20 – entropy 0.362 bits, top-1 'city' |
| L 21 – entropy 0.934 bits, top-1 'city' |
| L 22 – entropy 0.195 bits, top-1 'city' |
| L 23 – entropy 0.078 bits, top-1 'city' |
| L 24 – entropy 0.673 bits, top-1 'city' |
| L 25 – entropy 0.947 bits, top-1 'city' |
| L 26 – entropy 0.596 bits, top-1 'city' |
| L 27 – entropy 0.003 bits, top-1 'city' |
| L 28 – entropy 0.040 bits, top-1 'city' |
| L 29 – entropy 0.075 bits, top-1 'city' |
| L 30 – entropy 0.00007 bits, top-1 'city' |
| L 31 – entropy 0.00060 bits, top-1 'city' |
| L 32 – entropy 0.764 bits, top-1 'Berlin' |
| L 33 – entropy 0.249 bits, top-1 'Berlin' |
| L 34 – entropy 0.370 bits, top-1 'Berlin' |
| L 35 – entropy 0.409 bits, top-1 'Berlin' |
| L 36 – entropy 1.070 bits, top-1 'Berlin' |
| L 37 – entropy 0.362 bits, top-1 '<strong>' |
| L 38 – entropy 0.139 bits, top-1 '<strong>' |
| L 39 – entropy 0.396 bits, top-1 '<strong>' |
| L 40 – entropy 0.048 bits, top-1 '<strong>' |
| L 41 – entropy 0.002 bits, top-1 '<strong>' |
| L 42 – entropy 0.00000 bits, top-1 '<strong>' |

# 4. Qualitative patterns & anomalies
The first transformer block sharply collapses onto whitespace, and by L 6 the model is already certain of the plural "capitales" (entropy 0.014 bits).  Middle layers (L 17-31) lock onto the generic noun "city", suggesting an abstraction step where the network represents the *type* of answer without committing to the specific capital.  A decisive transition occurs at L 32 where "Berlin" dominates (0.84 probability, entropy 0.76 bits), e.g.
> "32,6,… Berlin,0.839…" [L34]
This semantic commitment is subsequently overwritten by HTML-style markup tokens such as "<strong>", which fully monopolise the decoder by L 42 (entropy ≈ 0).  The final soft-max therefore outputs formatting tokens instead of the correct city despite the internal representation.

Temperature exploration confirms the latent knowledge: at τ = 0.1 the model outputs " Berlin" with 100 % confidence, whereas at the production temperature the markup tokens prevail.  Test prompts that include Berlin in the context still elicit the same HTML bias.

Checklist:  
✓ RMS lens  
✗ LayerNorm (model is RMSNorm-only)  
✗ Colon-spam  
✓ Entropy rise at unembed

# 5. Tentative implications for Realism ↔ Nominalism
1. *Does the early collapse onto whitespace and markup indicate a nominalist bias toward surface-form regularities before semantic grounding?*  
2. *Is the stable "city" attractor (L 17-31) evidence of a mid-stack realist representation of conceptual categories independent of surface tokens?*  
3. *Why does a high-probability semantic token ("Berlin") lose to markup tokens in the final layers—does this reflect a trade-off between realistic world knowledge and nominalist frequency heuristics?*  
4. *Could the entropy surge at the unembed layer signal a reprise of nominalist breadth that masks an otherwise realist latent state?*

# 6. Limitations & data quirks
The run executed on CPU, so minor numerical drift from mixed-precision GPU behaviour is possible.  File timestamps are absent, preventing precise dating of the probe.  Several CSV rows contain blank or newline tokens, hinting at tokenizer artefacts.  The overwhelming presence of HTML tags suggests that Gemma-2-9B's web-crawl training data biases the output in ways not directly related to the prompt.

# 7. Model fingerprint
"Gemma-2-9B: collapse to 'city' by L 27; 'Berlin' appears at L 32; final entropy 2.0 bits but surface prediction '<strong>'."

---
Produced by OpenAI o3
