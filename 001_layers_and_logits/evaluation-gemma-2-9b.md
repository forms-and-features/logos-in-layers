# google/gemma-2-9b – Layer-wise Interpretability Analysis

> Probe run: `run.py` (norm-lens disabled due to RMSNorm)  
> Artefact: `output-gemma-2-9b.txt`

## 1. Architectural context
* Model reports 42 transformer blocks, 3 584-dim residual width and 16 heads.
* Both block-internal and final normalisation layers are **RMSNormPre** rather than vanilla LayerNorm:
```25:32:001_layers_and_logits/output-gemma-2-9b.txt
Block LayerNorm type: RMSNormPre
⚠️  Non-vanilla norm detected (RMSNormPre) - norm-lens will be skipped to avoid distortion
```
  • Because RMSNorm rescales vectors differently from LayerNorm, the probe sensibly skipped normalisation to avoid mis-interpreting logits – a good safety step, but it also means early-layer entropy values should be read qualitatively rather than compared to vanilla-norm models.

## 2. Early-layer token expectations
Immediately after the prompt the model is **certain** the next token is a colon – even before any transformer block executes.  Colon certainty persists until ≈layer 9:
```28:48:001_layers_and_logits/output-gemma-2-9b.txt
Layer  0 (embeddings):
   1. ':' (1.000000)
...
Layer  1 …   1. ':' (1.000000)
...
Layer  8 …   1. ':' (1.000000)
```
Interpretation:
* The fixed embedding already "knows" the syntactic continuation "Answer: …"; transformer blocks 1-8 do not override that trivial prediction.
* Entropy ≈0 bits shows pathological over-confidence; such determinism this early suggests limited utilisation of long-range signal at these depths, consistent with findings that lower layers handle local syntax [nostalgebraist 2020](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens).

## 3. Mid-layer semantic drift
After layer 10 the model's top guess flips from punctuation to lexical words ("answer", then high-frequency function words "a / the"), signalling the network has switched to assembling a generic answer template but **has not yet retrieved factual content**:
```220:240:001_layers_and_logits/output-gemma-2-9b.txt
Layer 10 … 1. ' answer' (0.975)
Layer 14 … 1. 'a' (0.998)
Layer 20 … 1. ' the' (0.996)
```
This trajectory – from punctuation → generic token → content word – mirrors the "iterative refinement" pattern reported in tuned-lens work [Belrose 2023](https://arxiv.org/abs/2303.08112).

## 4. Emergence of factual answer
The correct city name **does not appear with appreciable probability until layer 31** and becomes singularly dominant by layer 34:
```700:730:001_layers_and_logits/output-gemma-2-9b.txt
Layer 31 … 17. ' Berlin' (≈0.0)
Layer 34 … 1. ' Berlin' (0.999)
Layer 35-42 … 1. ' Berlin' (1.000)
```
Observations:
* The switch from function words to the factual token is abrupt (≈3 layers), consistent with "feature pop-out" seen when a specific causal circuit completes [Elhage et al. 2021](https://transformercircuits.pub/2021/framework/index.html).
* Final layer over-confidence (prob 0.85 on first-space " Berlin", plus casing variants) suggests the model keeps multiple orthographic aliases active in logit space.

## 5. Final prediction quality
```1010:1035:001_layers_and_logits/output-gemma-2-9b.txt
Model's final prediction (entropy: 1.388):
   1. ' Berlin' (0.850)
   2. ' The'  (0.022)
   4. ' Bonn' (0.014)
```
* The correct answer is top-1 with large margin → good factual recall under this phrasing.
* Competing logits are semantically related (capital alternatives, article starters) – typical for next-token tasks.

## 6. Sensitivity to prompt wording
Additional probes highlight **robustness variance**:
```1045:1065:001_layers_and_logits/output-gemma-2-9b.txt
'Germany's capital is'  →  top-1 ' a' (0.34), no city in top-10
'Berlin is the capital of' → top-1 ' Germany' (0.90)
```
* When the answer token is earlier in the context ("Berlin … of"), the model comfortably predicts 'Germany'.  
* Abstract formulation ("Germany's capital is …") fails – indicates the knowledge is present but gated by surface pattern matching rather than truly relational abstraction.  This aligns with research showing LLM recall is highly prompt-sensitive [Brown et al. 2020](https://arxiv.org/abs/2005.14165).

### Implication for nominalism ⇄ realism debate
The sharp discrepancy depending on wording supports a **nominalist reading**: concepts (e.g. capital-relation) are not stored as stable ontic entities but as distributions over token n-grams.  The model "knows" the fact only when the literal surface form is evoked.

## 7. Temperature exploration
At T = 0.1 the distribution collapses entirely onto ' Berlin'; by T = 2.0 probability mass disperses yet still favours Berlin-variants (0.41).  This indicates a steep logit gap – a useful quantitative prior for future causal tracing.

## 8. Red flags & anomalies
| Observation | Evidence | Potential concern |
|-------------|----------|-------------------|
| Extreme early-layer certainty (entropy ≈0) | layers 0-8 colon fixation | May mask defects when using lenses that assume meaningful probabilities; caution interpreting early logits. |
| RMSNorm incompatible with norm-lens | normalisation warning | Comparison with models using vanilla LayerNorm must control for this difference. |
| Prompt-sensitivity of factual recall | probing section | Reliance on surface cues – philosophical analyses based on LLM "beliefs" must account for context fragility. |

## 9. Recommendations for this project
1. When using Gemma-2-9B to argue metaphysical positions, **benchmark multiple paraphrases** of each philosophical claim; treat variability as epistemic uncertainty rather than model "error".
2. For causal-circuit work, focus on layers 30-35 where the semantic feature 'Berlin' crystallises – interventions here are likeliest to reveal concept representations.
3. Consider fine-tuning a norm-lens (e.g. tuned-lens) that compensates for RMSNorm; current raw-lens may under-estimate mid-layer semantic content.

---

*Prepared by: OpenAI o3*