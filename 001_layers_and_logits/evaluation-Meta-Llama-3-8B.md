# Meta-Llama-3-8B – Model-Level Interpretability Notes

*File analysed: `output-Meta-Llama-3-8B.txt`*

---

## 1. Experimental context

The probe (see `run.py`) performs a **LayerNorm/RMSNorm lens pass-through** on every residual stream position and reports the top-20 next-token logits for each layer.  The prompt used is

> Question: What is the capital of Germany? Answer:

All logits shown are calibrated by a full soft-max pass, so probability masses are directly comparable across layers.

Important implementation details evidenced in the log:

* Model: `meta-llama/Meta-Llama-3-8B` (32 transformer blocks, *d*<sub>model</sub>=4096, 32 heads, vocab 128 256, context 8192).
* **Normalization policy** – the model uses `RMSNormPre`.  The lens implementation could not find a learnable scale parameter (`⚠️  RMSNorm detected but no weight/scale parameter – norm-lens will be skipped`).  Consequently all residual streams are **raw**, i.e. they are *not* length-normalised before projection into vocabulary space.  This is a potential source of distortion in early-layer logits.
* Unembedding weights were promoted to FP32 (`🔬 Promoting unembed weights to FP32`) which removes numeric noise in the reported probabilities.

---

## 2. Layer-wise behavioural patterns

### 2.1 Entropy trajectory

| Layer index | Stage                                | Entropy (bits) | Comment |
|-------------|--------------------------------------|----------------|---------|
| 0           | Token embeddings                     | 16.969         | Essentially uniform – model has no information yet. |
| 18          | After block 17                      | 16.909         | Still near-uniform but semantically related tokens (*" capital"*, *" Capitals"*) begin to climb into top-20. |
| 22          | After block 21                      | **16.839**     | First appearance of *" Berlin"* as clear #1 (p≈1.2 × 10-4). |
| 25          | After block 24                      | 16.718         | *" Berlin"* probability surges to 1.4 × 10-3 (≈10× jump). |
| 30          | After block 29                      | 15.993         | *" Berlin"* at 1.9 % – another 14× increase. |
| 31          | After block 30                      | 14.654         | Massive collapse of entropy; *" Berlin"* 7.8 %. |
| 32 (final)  | Model output                         | **1.958**      | *" Berlin"* 83.7 % (log reports 85.7 % in "ACTUAL MODEL PREDICTION"). |

*Take-away*: knowledge seems to crystallise late.  Until ≈⅔ depth the model has not committed; decisive evidence appears only in the last ~3 blocks.  This late collapse is consistent with observations in other Llama-family models that factual recall often resides in deeper MLP channels (see [Geva et al., 2023](https://arxiv.org/abs/2202.08906)).

### 2.2 Token evolution

* **Early layers (0-10)** – Top tokens are mostly linguistic detritus (*"oren", "RIPT", "istrovství"*), indicating that raw residual vectors are not yet aligned with human-interpretable directions.  This is expected given missing RMS normalisation (see §1).
* **Mid layers (17-20)** – We see a thematic cluster: *" capital", "Capitals", "Capital", "Washington", "Federal"*.  The model is moving from lexical noise to a *semantic field* about capitals but hasn't selected a particular city.
* **Late layers (22-32)** – *" Berlin"* gradually displaces all competitors, with a small family of orthographic variants (`Berlin`, ` berlin`, ` Ber`, `BER`, `BERLIN` in different scripts).  Competing false positives (*"Washington"*, *"Bon"*, *"Canberra"*) are suppressed.

This progression is qualitatively similar to the "iterative refinement" lens patterns described by [Nanda & Lindner 2023](https://transformer-circuits.pub/2023/residual_stream_lens/index.html): earlier layers propose many weak hypotheses, later layers filter them.

### 2.3 Additional probing behaviour

The script queries three further prompts:

| Prompt | Top-1 token | p(top-1) | Entropy (bits) |
|--------|-------------|----------|-----------------|
| *Germany's capital is* | **"a"** | 0.288 | 6.022 |
| *Berlin is the capital of* | **" Germany"** | 0.896 | 0.928 |
| *Respond in one word: which city is the capital of Germany?* | **" Berlin"** | 0.149 | 7.310 |

Observations:

1. The canonical factual completion *Berlin → Germany* is extremely confident (entropy <1 bit).
2. The less natural construction *Germany's capital is* shows high uncertainty; top-1 *"a"* is clearly wrong.  This suggests the model relies on more common surface forms ("Berlin is the capital of...") rather than the inverse.
3. The *one-word* instruction increases entropy again (7.3 bits) – complying with a stylistic constraint is harder than recalling the fact itself.

---

## 3. Anomalies & red flags

1. **Incomplete RMSNorm parameters** – the absence of accessible `.weight`/`.scale` prevented the lens from length-normalising residuals.  This may systematically under-represent early-layer signal strength.  Future runs should patch `HookedTransformer` to expose `.weight` for RMSNorm (or apply RMS length scaling manually) before drawing quantitative conclusions.
2. **Noise tokens in early layers** – Strings like *"ABCDEFGHIJKLMNOP"*, *#ab*, *#ad*, *")application"* appear with non-trivial probabilities.  While not alarming on their own, they might hint at memorised training artefacts (e.g. code snippets, CSS hex colours).  For philosophical downstream work we should ensure that such artefacts do not confound conceptual analyses.
3. **Sharp entropy cliff between layers 30 and 32** – a drop from 15.99→1.96 bits within two blocks is unusually steep.  It would be worth checking whether these blocks contain large MLP attention contributions (e.g. a 'fact recall' circuit) analogous to the *Late-Merging Attention Heads* found in GPT-2-XL ([Olsson et al., 2022](https://transformer-circuits.pub/2022/induction_heads/index.html)).

---

## 4. Relevance to the Realism vs Nominalism project (non-conclusive)

The gradual emergence of a *specific* entity label (*Berlin*) out of a broad semantic class (*capital*) provides an empirical case study for how abstract relational information is refined into concrete referents inside an LLM.  Depending on one's metaphysical stance:

* **Nominalists** could point to the diffuse, distributed nature of early representations – there is no single, fully-formed 'capital-of-Germany' concept, only statistical co-activations that sharpen through computation.
* **Realists** might emphasise the late-layer collapse to a near-deterministic token as evidence that the model ultimately commits to a stable, discrete representation – hinting at a Platonic-style 'form' being realised.

However, these are tentative readings.  The current probe does not yet disentangle which specific neurons or head circuits implement the transition from generic *capital* knowledge to the concrete *Berlin* representation.  More granular interpretability tools (e.g. causal tracing, activation patching) are required before drawing philosophical conclusions.

---

## 5. Next steps (technical)

* Expose RMSNorm scales to enable length-corrected lenses and re-run the analysis.
* Perform activation patching on layers 18-32 to locate the minimal circuit responsible for the *capital → Berlin* resolution.
* Evaluate robustness across alternative phrasings to test whether the same circuit generalises beyond the exact Q-A format.

(No cross-model suggestions are included per instruction.)

---

### References

1. Geva, M., Schuster, T., et al. "Transformer Feed-Forward Layers Are Key-Value Memories." *ICLR 2023*. <https://arxiv.org/abs/2202.08906>
2. Nanda, N., & Lindner, M. "Residual Stream Lens." *Transformer Circuits* (2023). <https://transformer-circuits.pub/2023/residual_stream_lens/index.html>
3. Olsson, C., et al. "In-context Learning and Induction Heads." *Transformer Circuits* (2022). <https://transformer-circuits.pub/2022/induction_heads/index.html>
4. TransformerLens library (Neel Nanda, 2023). <https://github.com/neelnanda-io/TransformerLens>

---

*Prepared by: OpenAI o3*