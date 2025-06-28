# 1. Result synthesis

Across the four probed base models a consistent **late-stack entropy collapse** is observed.  The answer token becomes near-deterministic at roughly the same *relative* depth: Meta-Llama-3-8B collapses at L 25 / 32 ≈ 0.78 ([41-44](001_layers_and_logits/evaluation-Meta-Llama-3-8B.md)), Mistral-7B at L 25 / 32 ≈ 0.78 ([37](001_layers_and_logits/evaluation-Mistral-7B-v0.1.md)), Qwen-3-8B at L 28 / 36 ≈ 0.78 ([44](001_layers_and_logits/evaluation-Qwen3-8B.md)) and Gemma-2-9B at L 35 / 42 ≈ 0.83 ([34-35](001_layers_and_logits/evaluation-gemma-2-9b.md)).  This alignment across model families suggests that factual resolution is a depth-normalised phenomenon, echoing findings in tuned-lens work (arXiv:2303.08112).

A **concept-before-entity pattern** repeats: a generic "capital" token rises several layers *before* the first high-probability "Berlin".  Llama shows 'Capital' at L 20 ([41-42](001_layers_and_logits/evaluation-Meta-Llama-3-8B.md)); Mistral shows 'capital' at L 21 ([37 & 52](001_layers_and_logits/evaluation-Mistral-7B-v0.1.md)).  The same window precedes the collapse in Qwen (L 20–21, table) and Gemma (L 20–24, "the" plateau).  This supports the view that models first settle on a semantic category then retrieve the concrete entity.

All models exhibit an **entropy rebound after the final normalisation / unembed**.  Qwen jumps from 0.13 bits (L 35) to 2.02 bits in the final logits ([54](001_layers_and_logits/evaluation-Qwen3-8B.md)); Llama rises from 0.37 bits (L 28) to 1.7 bits post-unembed ([54-57](001_layers_and_logits/evaluation-Meta-Llama-3-8B.md)); Mistral shows 0.79 → 1.80 bits ([31-32](001_layers_and_logits/evaluation-Mistral-7B-v0.1.md)).  Gemma shows a smaller bump then decline (["entropy rise then fall" checklist](001_layers_and_logits/evaluation-gemma-2-9b.md)).  The rebound corroborates reports that unembedding sometimes de-sharpens internal states (arXiv:2303.08112).

Early-layer behaviour is heterogeneous.  Gemma is *over-confident on punctuation* – entropy <10⁻⁶ bits on ':' until L 9 ([63](001_layers_and_logits/evaluation-gemma-2-9b.md)) – while the other models emit high-entropy junk or multilingual tokens (e.g. 'ListViewItem', '␠', 'laugh') indicating that the norm lens surfaces noise when semantics are not yet integrated.

# 2. Misinterpretations in existing EVALS

- **Phase-change claim in Qwen** – The narrative of "a sharp phase change rather than gradual sharpening" ([54-57](001_layers_and_logits/evaluation-Qwen3-8B.md)) overlooks the steady entropy fall from 8.6 → 3.7 → 2.6 bits between L 19-21; the CSV shows a gradual slope, not a step.
- **Colon certainty in Gemma** – Line 63 of the Gemma report presents <10⁻⁶ bit entropy on ':' as model certainty, yet §6 later notes teacher-forcing artefacts.  The earlier phrasing over-attributes the zero entropy to the model rather than the probe set-up.
- **Attribution of rebound to LayerNorm in Mistral** – Line 52 links the entropy rise "after `ln_final`", but the script's final logits include both `ln_final` and the unembed weight; the evidence cannot isolate LayerNorm as the cause.
- **Unembed bounce in Llama** – Lines 54-57 ascribe the 0.37 → 1.7 bit rise to the unembed itself, yet rows L 29-30 already show entropy >0.8 bits before unembedding, suggesting the rebound begins earlier.

# 3. Usefulness for the Realism ↔ Nominalism project

The aligned depth-normalised collapse invites the hypothesis that a shared architectural milestone – perhaps the last two MLP blocks – houses a linear "fact lookup" subspace.  Testing whether targeted ablations in that window erase the answer across models could discriminate between a *realist* "stored fact" and a *nominalist* distributed voting mechanism.

The concept-before-entity pattern raises an open question: do models first converge on an abstract *slot* ("capital-city") then bind a concrete token?  Causal tracing of attention paths from the 'capital' layers into the collapse layers could reveal whether a discrete head executes the lookup (supporting realism) or whether many heads gradually sharpen logits (nominalism).

The entropy rebound suggests a calibration step that dilutes an over-confident intermediate state.  Intervening to bypass `ln_final` and directly sample from the pre-LN residual could test whether this rebound is functional (nominalist smoothing) or merely a by-product of normalisation (realist pass-through).

# 4. Limitations

The probe covers a *single prompt* and *single answer position*; generalisation to other factual relations is unverified.  Early-layer entropies are confounded by **teacher-forcing** – the lens evaluates the true next token, not the model's sample, inflating certainty on punctuation.  Only the top-20 logits are saved, obscuring tail mass and preventing exact entropy recovery.  All experiments ran on **CPU FP32**, so GPU-specific quantisation or fused-kernel effects are absent.  The RMS-based lens rescales residuals but may distort directionality, and positional embeddings are summed, blending syntactic and semantic features.  Finally, per-layer activations are cached in FP32 on host memory; rounding or truncation could mask subtle logit gaps.

---

Produced by OpenAI o3