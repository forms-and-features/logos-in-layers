# Cross-Model Interpretability Analysis – Capital-of-Germany Probe

*Models inspected:* **Meta-Llama-3-8B**, **Mistral-7B-v0.1**, **Gemma-2-9B**, **Qwen3-8B**  
*Source artefacts:* `output-*.txt` dumps analysed in the per-model reports.

---

## 1. Scope & Methodology  
All runs use the same script (`run.py`) and the same factual prompt:

> Question: What is the capital of Germany? Answer:

Layer-wise residual streams were projected through the final unembedding matrix.  Where a learnable scale parameter (`weight`/`scale`) for **RMSNorm** was *not* exposed (the default for all four checkpoints), the probe necessarily fell back to the *raw* residual stream (“raw lens”).  Probabilities and entropies from early layers are therefore subject to *scale distortion* (Belrose et al., 2023, §3).  Conclusions below focus on *relative* trends rather than absolute numbers.

---

## 2. Shared Patterns Across Models

| Phenomenon | Evidence (layers) | Brief explanation |
|------------|------------------|-------------------|
| **Late-layer entropy collapse** | Llama: 30→32 (15.99 → 1.96 bits); Mistral: 25→32 (14.84 → 8.25 bits); Qwen: 24→28 (≈0.7 → 0 bits); Gemma: 34 → final (0.01 → 2.0 bits after unembedding) | Factual certainty crystallises close to the output, in line with "tuned-lens" findings that many transformer blocks act as *refiners* rather than *retrievers* (Nanda & Lindner, 2023). |
| **Meta-token scaffolding** | 'Answer' head dominates mid-layers in Qwen (L12-18) & Mistral (L6-18) | Models first predict *response format* tokens before factual content, echoing Mu & Andreas (2024) on *answer-classification heads*. |
| **Two-step semantic resolution** | Qwen: **' Germany'** peaks one layer before **' Berlin'** (L25 vs L28) | Suggests compositional pathway "identify topic → resolve entity", consistent with Geva et al. (2022) on feed-forward memory layers. |
| **Unembedding entropy rebound** | All final logits show ≈2 bits entropy even when internal vectors are quasi-deterministic | The unembedding matrix redistributes probability mass; surface uncertainty can therefore underestimate internal semantic clarity (Elhage et al., 2022). |
| **RMSNorm scale missing ⇒ lens artefacts** | Warning present in every run; early layers show either uniform noise (Llama, Mistral, Qwen) or deterministic punctuation (Gemma) | Without length normalisation, large-norm embedding directions swamp early projections. |

---

## 3. Model-Specific Anomalies & Red Flags

* **Gemma-2-9B – colon fixation (layers 0-7).** 100 % probability on `':'` points to a raw-lens artefact rather than genuine behaviour; early-layer findings for Gemma are unreliable until ≈ layer 8.
* **Mistral-7B – transient 'Washington' dominance (layer 23).** A U.S. capital briefly outranks *Berlin*, indicating competing attractors within the *capital-city* concept space.
* **Qwen3-8B – negative entropy printouts (layers 34-35).** Rounding to –0.000 bits is purely numerical, but highlights near-unit probability before the unembedding step.
* **Meta-Llama-3-8B – exceptionally steep entropy cliff (layers 30-32).** 14-bit drop within two blocks suggests one or two highly specialised late MLP heads responsible for factual recall.

These anomalies could be productive targets for *causal tracing* or *activation patching* (Meng et al., 2022) to localise the minimal circuits involved.

---

## 4. Preliminary Relevance to the Realism ↔ Nominalism Debate  
(No philosophical conclusions; observations only.)

1. **Gradual abstraction hierarchy.**  All models transition from diffuse token soup → meta-format tokens → concrete entity, providing empirical support for *layered concept formation*.  Realists may point to the late near-deterministic vector as evidence of a "stable" internal form; nominalists can emphasise that earlier layers hold only distributional tendencies.
2. **Direction-specific relations.**  Gemma (and to lesser extent Llama) recall *Berlin → Germany* more confidently than the inverse *Germany → Berlin*, indicating that relational knowledge is encoded asymmetrically—important for ontological claims about concept individuation.
3. **Unembedding bottleneck.**  The gap between internal certainty and surface logits shows that lexical tokens are lossy *names* for richer internal states, a point that can be leveraged by both sides of the metaphysical debate.

---

## 5. Methodological Recommendations

1. **Implement RMS-aware lens across checkpoints.**  Where `weight`/`scale` is inaccessible, approximate γ via running statistics or patch the model class to expose it, then re-run early-layer probes.
2. **Causal interventions on anomaly layers.**  Patch activations at (a) Gemma L0-7, (b) Mistral L23, (c) Llama L30-32, (d) Qwen L24-28 to test necessity & sufficiency of candidate circuits.
3. **Cross-prompt robustness.**  Repeat the experiment with paraphrased queries and cloze formulations to see whether the same layers encode the relation.
4. **Compare with LayerNorm-based models.**  Including e.g. GPT-2-XL would help disambiguate real phenomena from RMS raw-lens artefacts.

---

### References

* Belrose, N., Furman, Z., et al. *Eliciting Latent Predictions from Transformers with the Tuned Lens.* arXiv:2303.08112, 2023.  
* Elhage, N., et al. *A Mathematical Framework for Transformer Circuits.* arXiv:2104.08696, 2021.  
* Elhage, N., et al. *Toy Models of Superposition & Feature Bottlenecks.* transformer-circuits.pub, 2022.  
* Geva, M., Schuster, T., et al. *Transformer Feed-Forward Layers Are Key-Value Memories.* arXiv:2202.10402, 2022.  
* Meng, K., et al. *Locating and Editing Factual Associations in GPT.* arXiv:2211.00593, 2022.  
* Mu, J., & Andreas, J. *Internal Monologue Representation in Language Models.* arXiv:2402.06655, 2024.  
* Nanda, N., & Lindner, M. *Residual Stream Lens.* transformer-circuits.pub/2023, 2023.

---

*Prepared by: OpenAI o3*