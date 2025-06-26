## 1. Overview
Mistral-7B-v0.1 (≈7 B parameters) was inspected on 2025-06-26 using the custom **run.py** probe.  The script hooks every residual stream, applies a norm-lens, and logs the entropy and top-k predictions for the next-token distribution at each layer while answering “*Question: What is the capital of Germany?*”.

## 2. Method sanity-check
The console confirms that the probe used RMS norm-lens and that positional embeddings were available (though merged into the token embedding hook):

> Block normalization type: RMSNormPre – *"Using NORMALISED residual stream (RMS, no learnable scale)"* [L8-9]  
> [diagnostic] No separate positional embedding hook; using only token embeddings for layer 0 residual [L28]

This indicates (i) RMSNorm was detected and correctly normalised; (ii) the model uses rotary or fused positional information, so the embedding lens is appropriate.

## 3. Quantitative findings
| Layer | Entropy (bits) | Top-1 token |
|-------|---------------|-------------|
| 0 | 14.722 | 'laug' |
| 1 | 14.440 | 'zo' |
| 2 | 14.049 | 'ts' |
| 3 | 13.871 | 'richt' |
| 4 | 13.821 | 'amber' |
| 5 | 13.738 | 'aiser' |
| 6 | 13.732 | 'amber' |
| 7 | 13.636 | 'nab' |
| 8 | 13.616 | 'amber' |
| 9 | 13.582 | 'answer' |
| 10 | 13.322 | 'answer' |
| 11 | 13.479 | 'Answer' |
| 12 | 13.298 | '/******/' |
| 13 | 13.657 | 'ír' |
| 14 | 13.668 | 'Answer' |
| 15 | 13.553 | 'Answer' |
| 16 | 13.436 | 'Answer' |
| 17 | 12.921 | 'Answer' |
| 18 | 13.221 | 'cities' |
| 19 | 12.736 | 'cities' |
| 20 | 11.430 | 'cities' |
| 21 | 8.484 | 'capital' |
| 22 | 6.564 | 'Berlin' |
| 23 | 3.159 | 'Washington' |
| 24 | 2.882 | 'Berlin' |
| **25** | **0.558** | **'Berlin'** |
| 26 | 0.282 | 'Berlin' |
| 27 | 0.104 | 'Berlin' |
| 28 | 0.135 | 'Berlin' |
| 29 | 0.123 | 'Berlin' |
| 30 | 0.352 | 'Berlin' |
| 31 | 0.791 | 'Berlin' |
| 32 | 1.800 | 'Berlin' |

Entropy collapses (< 1 bit) for the first time at **layer 25**, dominated by the correct answer.

## 4. Qualitative patterns & anomalies
• Early layers are dominated by unrelated multilingual sub-word junk (e.g. > "Layer 4 … 'amber' (0.15)" [L55]).  
• Mid-layers fixate on the literal string *Answer/answer* without converging (> "Layer 10 … 'answer' (0.23)" [L236]).  
• Layer 12's top token is the obfuscated string `'/******/'`, suggesting filter-token artefacts (> "1. '/******/' (0.508)" [L294]).  
• A transient misprediction – **'Washington'** outranks *Berlin* at layer 23 (> "Layer 23 … 'Washington' (0.51)" [L536]).  
• Entropy "re-opens" after the collapse: 0.104 bits at layer 27 → 1.8 bits at final unembed (> "Layer 32 … entropy 1.800 bits" [L734]).

Checklist:  
• RMS used? ✓  
• LayerNorm? ✗ (only RMSNormPre detected)  
• Colon-spam? ✗  
• Re-opening of entropy at unembed? ✓

## 5. Implications & open questions for Realism ↔ Nominalism
• Does the sharp collapse at layer 25 indicate formation of a *concept token* (realist view) or just statistical shortcutting (nominalist)?  
• Why does a geopolitical distractor (*Washington*) appear late—noise or competing representation of "capital"?  
• Is entropy rebound (layers 30-32) evidence of representational diversification after a decision is made?  
• How does RMS normalisation shape when symbolic certainty emerges compared with LayerNorm models?

## 6. Limitations & data quirks
Small sample size (single prompt) limits generality; unusual tokens (`'/******/'`) hint at tokenizer quirks; line counts for entropy may vary with temperature; positional-embedding diagnostic shows only token embeddings, so rotary phase shifts are un-inspected.

## 7. Next probes with current artefact
1. Activation patching: replace layer-25 residual with layer-24 to test if collapse causally sets answer.  
2. Logit lens on alternative prompts (*"Germany's capital is …"*) already present—measure entropy per prompt without re-run.  
3. Compute KL divergence between layer 23 and 24 distributions to quantify step-change.

## 8. Model fingerprint
"Mistral-7B-v0.1: entropy collapses at layer 25; final entropy 1.8 bits; anomalous 'Washington' top-1 at layer 23."

---

Produced by OpenAI o3