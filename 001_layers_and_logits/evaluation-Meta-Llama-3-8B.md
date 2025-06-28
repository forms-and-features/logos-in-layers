# Evaluation of Meta-Llama-3-8B probe

## 1. Overview
Meta-Llama-3-8B (≈ 8 B parameters) was probed with the layer-wise RMS-lens script on a single prompt asking for Germany's capital. The run captures per-layer logit distributions, entropies and top-k tokens as well as auxiliary probes (alternative prompts, temperature sweeps). All results are saved in structured JSON/CSV artefacts and reproduced here.

## 2. Method sanity-check
The JSON diagnostics confirm that the run used the intended RMS-lens normalisation and detected RMSNorm layers at both the first transformer block and the final normalisation stage:
```14:22:001_layers_and_logits/output-Meta-Llama-3-8B.json
"use_norm_lens": true,
"first_block_ln1_type": "RMSNormPre",
```
Positional information was included via the script path that merges `hook_embed` and `hook_pos_embed` when available:
```281:283:001_layers_and_logits/run.py
if has_pos_embed:
    resid = (residual_cache['hook_embed'] +
             residual_cache['hook_pos_embed'])
```
Both checks indicate the probe operated as designed.

## 3. Quantitative findings
| Layer | Entropy (bits) | Top-1 token |
|-------|---------------|-------------|
| L 0   | 14.52 | 'oren' |
| L 1   | 13.64 | 'ря' |
| L 2   | 13.62 | ' ' |
| L 3   | 13.54 | 'atten' |
| L 4   | 13.08 | 'adal' |
| L 5   | 13.32 | 'chine' |
| L 6   | 13.28 | '.Decode' |
| L 7   | 13.00 | 'Cem' |
| L 8   | 13.13 | 'AutoSize' |
| L 9   | 12.23 | 'enville' |
| L 10  | 12.58 | 'PCM' |
| L 11  | 13.09 | 'ystack' |
| L 12  | 12.39 | '.xtext' |
| L 13  | 13.36 | 'краї' |
| L 14  | 13.35 | '#ab' |
| L 15  | 12.81 | '#af' |
| L 16  | 13.03 | '#ab' |
| L 17  | 11.95 | '#ad' |
| L 18  | 12.48 | '#ad' |
| L 19  | 11.75 | 'ABCDEFGHIJKLMNOP' |
| L 20  | 10.77 | 'Capital' |
| L 21  | 9.29  | 'capital' |
| L 22  | 8.48  | ' Berlin' |
| L 23  | 5.58  | ' Berlin' |
| L 24  | 2.07  | ' Berlin' |
| **L 25** | **0.43** | **' Berlin'** |
| L 26  | 0.35 | ' Berlin' |
| L 27  | 0.33 | ' Berlin' |
| L 28  | 0.37 | ' Berlin' |
| L 29  | 0.81 | ' Berlin' |
| L 30  | 1.64 | ' Berlin' |
| L 31  | 1.16 | ' Berlin' |
| L 32  | 1.70 | ' Berlin' |

## 4. Qualitative patterns & anomalies
After a long plateau of high entropy (> 10 bits) the distribution begins to specialise around layer 20 where the literal token "Capital/capital" dominates. A sharp semantic collapse occurs at L 25 where entropy drops below 1 bit and ' Berlin' holds ~96 % of probability. The system stays in this collapsed state until L 29, rises modestly in the final two layers and settles at 1.7 bits (final JSON, top-1 86 %). This rebound mirrors findings in Tuned-Lens (2303.08112) where unembedding reintroduces diversity.

Alternative prompts reveal asymmetry: "Berlin is the capital of ..." is answered with low entropy 0.93 bits and 89 % Germany, whereas the reversed form "Germany's capital is ..." stays diffuse at 6 bits (lines 46-66). Temperature sweeps show extreme sharpening (≈0 bits) at τ = 0.1 and full softening at τ = 2.0, yet Berlin remains highest-ranked throughout (lines 210-235).

> "...topk": [[ " Berlin", 0.8569 ...]" 16:22:001_layers_and_logits/output-Meta-Llama-3-8B.json

> "temperature": 0.1,..." Berlin", 1.0" 225:237:001_layers_and_logits/output-Meta-Llama-3-8B.json

Checklist:  
✓ RMS lens  
✗ LayerNorm  
✗ Colon-spam  
✓ Entropy rise at unembed

## 5. Tentative implications for Realism ↔ Nominalism
1. Does the entropy collapse at L 25 indicate a transition from distributed "capital" concept representation to a nominal token identity ('Berlin') that future layers merely refine?

2. Could the slight rebound after the unembed suggest a realism-style enlistment of latent lexical alternatives that preserve factual grounding while permitting stylistic variation?

3. Does the divergent behaviour on paraphrased prompts imply that nominal bindings are context-sensitive, challenging a naive realist view of stable factual circuits?

4. Might the persistence of 'Berlin' across all temperatures hint at a fundamentally nominal token anchoring that realism would need to reconcile with graded uncertainty elsewhere?

## 6. Limitations & data quirks
The probe uses a single geography prompt; findings may not generalise. Early-layer tokens are often garbled Unicode or code fragments, suggesting tokenizer artefacts rather than meaningful predictions. Device was Apple M-series (mps); numerical parity with CUDA is assumed but unverified. Only entropy and top-1 are inspected; deeper distributional shifts could be missed.

## 7. Model fingerprint
Meta-Llama-3-8B: collapse at L 25; final entropy ≈ 1.7 bits; 'Berlin' dominates from L 22 onward.

---
Produced by OpenAI o3
