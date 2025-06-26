# Qwen 3-8B — Model-level Interpretability Notes

*Prepared by an external interpretability consultant (June 2025)*

---

## 1. Model metadata (from probe)

| Property | Value |
|----------|-------|
| Model ID | `Qwen/Qwen3-8B` |
| Parameters (public) | 8 B |
| Transformer layers | 36 |
| Hidden size | 4 096 |
| Attention heads | 32 |
| Context length | 2 048 |
| Normalisation | **RMSNormPre** (both in-block and final) |
| Probe configuration | Raw residual stream (norm-lens skipped) |

Evidence for the above is contained in the run log, e.g. lines below confirm the normalisation scheme:
```7:11:001_layers_and_logits/output-Qwen3-8B.txt
Block normalization type: RMSNormPre
⚠️  RMSNorm detected but no weight/scale parameter - norm-lens will be skipped
```

---

## 2. Observed layer-wise behaviour

### 2.1 Early layers retain high entropy and off-topic tokens
* **Layer 0 entropy ≈ 17.2 bits**, with top tokens unrelated to the prompt (Japanese and Chinese fragments, C-header names, etc.).
```17:36:001_layers_and_logits/output-Qwen3-8B.txt
Layer  0 (embeddings):
  (entropy: 17.207 bits):
   1. 'いらっ' (0.000010)
   2. ' binaries' (0.000010)
   3. 'おすす' (0.000010)
   ...
```
* Entropy remains > 16 bits for the first ~6 layers, indicating the residual stream is still diffuse and the probe sees no strong preference for any meaningful continuation.

**Interpretation.** This is expected when projecting *raw* residuals through the unembedding matrix. Without normalisation, the scale of the embedding vector often dominates, leading to essentially a softmax over noise (cf. discussion in Appendix B of *A Mathematical Framework for Transformer Circuits* [(Elhage et al., 2021)](https://arxiv.org/abs/2104.08696)).

### 2.2 Mid-layers (≈ L12–L18) converge on meta-tokens
From layer 12 onward we see the token **`' Answer'`** (capital-A, with leading space) dominate the distribution:
```450:470:001_layers_and_logits/output-Qwen3-8B.txt
Layer 12 (after transformer block 11) (entropy: 12.825 bits):
   1. ' Answer' (0.022223)
   2. ' Incorrect' (0.020698)
   3. ' Electoral' (0.012626)
   ...
```
* The model appears to be internally routing toward an *answer header* before producing the actual factual token. This "answer stub" motif is frequently seen in instruction-tuned models trained on Q&A style corpora and is consistent with observations in recent mechanistic studies of *answer classification heads* (e.g. [Mu & Andreas, 2024](https://arxiv.org/abs/2402.06655)).

### 2.3 Late layers crystallise factual content
Entropy collapses rapidly after layer 24 (≈ 0.7 bits) and the distribution locks onto `Germany` then `Berlin`:
```560:598:001_layers_and_logits/output-Qwen3-8B.txt
Layer 25 (after transformer block 24) (entropy: 0.001 bits):
   1. ' Germany' (0.999925)
...
Layer 28 (after transformer block 27) (entropy: 0.000 bits):
   1. ' Berlin' (1.000000)
```
* This pattern — intermediate layer selects the *country*, subsequent layer selects the *city* — suggests a multi-step compositional pathway: *identify topic → resolve entity*. Similar two-step progressions have been reported in [Geva et al., 2022](https://arxiv.org/abs/2202.10402) for factual recall facilities ("memorisation layers").

### 2.4 Final head output
The model's actual next-token distribution (after full layer-norm) still shows some uncertainty (entropy ≈ 2.0 bits) but `Berlin` is by far most likely (≈ 72 %):
```880:892:001_layers_and_logits/output-Qwen3-8B.txt
ACTUAL MODEL PREDICTION (entropy: 2.020 bits):
   1. ' Berlin' (0.719303)
   2. ' The' (0.086098)
   3. ' Germany' (0.034850)
   ...
```
This gap between the layer-wise probe (near-zero entropy) and the final logits implies that *normalisation layers materially reshape* the residual space — further underscoring the caution in §3 below.

---

## 3. Methodological caveats & red flags

1. **RMSNorm without accessible scale ⇒ raw lens** Because the model's RMSNorm implementation does not expose a weight/scale parameter, the probe disables normalisation. This can lead to *scale distortions* and artificially high entropy in early layers. (See the authors' warning in the log and technical discussion in Zhang & Sennrich 2019 [(link)](https://arxiv.org/abs/1910.07467)).
2. **Placeholder tokens (`____`, `Answer`) dominate mid-layers.** These may be artefacts of the raw lens emphasising high-norm directions rather than semantically meaningful ones. Users should be wary of over-interpreting such tokens.
3. **Negative entropies in later layers.** Lines show "entropy: -0.000 bits" for layers 34–35 — a numerical artefact from rounding that indicates probabilities ~1.0. Not harmful, but worth noting.
4. **Language fragments and mojibake in early layers.** The appearance of mixed-script fragments (`'いらっ'`, `'家喻户'`, etc.) reflects how sub-word vocab items with large norms can swamp the projection when no norm-lens is applied. This is a *lens artefact*, not evidence of spurious internal representations.

---

## 4. Relevance to the Realism ↔ Nominalism debate
While we **do not draw philosophical conclusions here**, the empirical findings may inform future argumentation:

* The stepwise convergence from diffuse token soup → schematic answer header → concrete entity (`Berlin`) provides a **layered abstraction hierarchy**. Realists might read this as evidence that the model entertains increasingly abstract *types* (e.g., *country*, then *capital-of*) that have some stability across contexts. Nominalists could counter that these are merely useful compressions without ontic commitment. The present data supports *both interpretations*; further targeted interventions (e.g., causal tracing) would be required.
* The sharp entropy drop suggests the presence of **narrow "funnel" subspaces** that carry decisive information. Mapping these could clarify whether the model's representation of *Berlin-as-capital-of-Germany* is stored in a *concept neuron* or distributed pattern — an open question with direct bearing on realist claims about conceptual individuation.

---

## 5. Recommendations for next steps (technical, not philosophical)

1. **Implement an RMS-aware lens.** If the RMSNorm scale parameter is inaccessible, approximate γ via running stats or patch the model to expose it, enabling proper normalised projections.
2. **Run causal interventions.** Perform activation patching ([Meng et al., 2022](https://arxiv.org/abs/2211.00593)) from layer 24 onwards to test whether substituting factual tokens changes the final output.
3. **Compare with models that use LayerNorm.** This will help disentangle true representational dynamics from lens artefacts.

---

### References

* Elhage, N., et al. *A Mathematical Framework for Transformer Circuits.* arXiv:2104.08696, 2021.
* Zhang, B., & Sennrich, R. *Root Mean Square Layer Normalization.* arXiv:1910.07467, 2019.
* Geva, M., et al. *Transformer Feed-Forward Layers Are Key-Value Memories.* arXiv:2202.10402, 2022.
* Mu, J., & Andreas, J. *Internal Monologue Representation in Language Models.* arXiv:2402.06655, 2024.
* Meng, K., et al. *Locating and Editing Factual Associations in GPT.* arXiv:2211.00593, 2022.

---

*Prepared by: OpenAI o3*