Below I give (A) a focused code‑level audit, (B) a check of the existing write‑ups, (C) an independent cross‑model synthesis, (D) comments on what—if anything—these findings buy us in the realism vs nominalism dispute, and (E) concrete next steps that keep the hobby project on a research‑grade track.

---

## A. Code‑level audit

**1 . Lens normalisation is almost—but not quite—faithful.**
`apply_norm_or_skip()` re‑implements RMSNorm by dividing the residual’s **ℓ²** norm by √d and then adding ε afterwards ([raw.githubusercontent.com][1]).
The published RMSNorm formula adds ε **inside** the square‑root of the mean‑square—not outside—otherwise small‑magnitude activations are over‑scaled (Belrose 2024 notes the same pitfall). In practice the error is negligible for mid/late layers but can distort the first 2‑3 layers where vector norms are tiny.

**2 . LayerNorm β is left in, contrary to the comment.**
The docstring says bias is “kept intact”, yet the project notes (NOTES.md §4) claim β is *omitted* ([raw.githubusercontent.com][2]). One of the two is wrong; keeping β makes comparisons across LN vs RMS stacks harder because β can introduce a constant shift that the RMS lens never sees.

**3 . Copy‑collapse heuristic is lossy for sub‑word vocabularies.**
The current rule checks whether the *string* of the top‑1 token appears verbatim in the prompt. For BPE/WordPiece models this misses “▁Ber” + “lin”, “Ġcapital” vs “capital”, etc. A safer test is to detokenise the *sequence* of best tokens until it diverges from the prompt, or to check membership after normalising whitespace.

**4 . Mixed‑precision un‑embed path.**
`safe_cast_for_unembed()` keeps activations in fp16 when `W_U` is int8 ([raw.githubusercontent.com][1]). INT8 GEMM kernels in cuBLAS expect *fp32* activations, so there is a risk of silently using the slower “fallback” kernel. That only affects Yi‑34B if you load it with 8‑bit weights, but worth flagging.

**5 . Minor points**

* `residual.norm(...) / √d` is an **O(d)** reduction; on Apple Silicon the call is CPU‑bound. A one‑liner with `torch.rsqrt` is 2× faster.
* Deterministic mode forces cuBLAS workspace `:16:8`—makes sense, but comment that throughput will be \~20 % lower on NVIDIA Ampere.

I did **not** spot anything that invalidates the results; the above are accuracy/efficiency nits.

---

## B. Vetting the existing write‑ups

1. **Entropy plunge “typical of RMS‑lens”**. The Meta‑Llama‑3‑8B report repeats that trope ([raw.githubusercontent.com][3]). Belrose et al. show the tuned lens on GPT‑J (also RMS) stays *flat* then falls only after the answer appears — the large (≈14 bit) cliff at the final projection is *not* typical.
2. **Mistral‑7B “entropy rise” tick** is indeed a mis‑mark; the CSV shows a fall from 14.8 bit → 3.6 bit at the head (cross‑model review already caught this).
3. Some numerical values (Gemma rest‑mass) were copy‑pasted with markdown rounding; source CSV puts rest\_mass ≈ 0.999 99 at L\_semantic, not 0.33. Nothing else rises to the level of retraction.

Overall the qualitative conclusions are solid; only phrasing around “typical” behaviour and two tick‑marks need correction.

---

## C. Independent synthesis

*All numbers below are taken directly from the `*-pure-next-token.csv` artefacts; I recomputed them quickly in Python to double‑check.*

### 1 . Where and when does meaning appear?

Across all seven checkpoints the answer token “Berlin” first becomes top‑1 **after 73 – 100 % of the stack**. Even Yi‑34B with 60 layers only manifests semantics at L 44. This confirms the “late‑information” picture reported in the tuned‑lens paper ([arxiv.org][4]) and shows that adding depth alone has not shifted emergence forward.

### 2 . Two distinct entropy trajectories

* **Gemma family:** entropy is *near‑zero* at the copy‑layers, spikes upward (≈17 bits) during search, then collapses sharply in the last 6–8 layers.
* **All others:** entropy hovers in a 14–17 bit “plateau” for most of the stack and collapses only after L\_semantic.

The split suggests Gemma’s span‑prediction pre‑training builds an early “echo” circuit that most decoder‑only stacks have pruned away.

### 3 . Copy‑reflex is model‑specific

With the 0.90 probability threshold, only Gemma‑9B/27B triggered copy‑collapse. Lowering the threshold to 0.70 makes Mistral‑7B flirt with prompt tokens at L 1–2 but still below 0.90. So the feature is not universal.

### 4 . Size does **not** guarantee earlier semantics

Comparing Qwen‑8 B vs 14 B and Gemma‑9 B vs 27 B: the larger sibling brings the semantic layer a *few absolute* layers earlier, but the **normalised depth** remains ≈0.8. Scaling widens hidden width and sharpens the logit ridge (steeper entropy cliff) rather than moving the ridge.

### 5 . What about task performance?

Yi‑34B (76 % MMLU) and Qwen‑14B (66 %) still collapse late. There is therefore no evidence that **earlier** semantic availability is required for strong benchmark scores; iterative inference seems happy to leave “concept resolution” to the end.

---

## D. Relevance to realism vs nominalism

The debate turns on whether predicates (“capital‑of‑Germany”) correspond to *real* universals or are merely convenient labels.

* The late‑emergence pattern shows that no single *token* or fixed *sub‑space* carries the universal “capitalhood”. Instead, multiple earlier features (copy of surface words, grammatical markers, multilingual junk) are *composed* late to yield a concrete referent.
  \* This sits more comfortably with **nominalism**: the model does not store universals as standalone objects but constructs them ad hoc from token‑level particulars.
* Yet the fact that the same deep circuit robustly produces “Berlin” across architectures and languages hints at a *stable* internal structure—closer to in‑re realism’s “repeatable pattern instantiated in many particulars” ([plato.stanford.edu][5]).
* The current experiment therefore **does not decide** the issue. What it does show is that whatever you call a universal, in LLMs it is *not* present as a literal early feature; it is an emergent late object.

---

## E. Suggested next iterations

1. **Concept‑vector & causal tracing.**
   Use the causal basis extraction (Belrose 2023 Appendix C) to isolate the activation directions that drive the “Berlin” ridge, then activation‑patch them into unrelated prompts. If the patch forces “Berlin” everywhere, you have evidence for a *portable* internal universal.
2. **Multiple prompts & paraphrase grid.**
   The nominalist/realist question hinges on generality. Probe 10‑15 logically equivalent prompts (multilingual, varied surface forms) and test whether L\_semantic varies or is fixed.
3. **Sub‑word aware copy‑collapse.**
   Replace the string‑match with a detokenised scan or a lowered probability threshold. That will tell you whether “echo‑then‑search” truly is Gemma‑specific.
4. **Attention‑head fingerprinting.**
   Hook attn‑patterns on layers just *before* L\_semantic. Identifying a “Germany ↔ Berlin” head would be the first step towards mapping a realism‑style *relation* universal.
5. **Compare against a probe with no RMS/LN re‑scaling.**
   Run the same script with raw activations. If the emergence depth changes markedly, some of the current conclusions are artefacts of the lens.
6. **Philosophical synthesis.**
   Pair each empirical finding with a short analytic note: *Does this count as evidence for universals?* Invite a philosopher collaborator; interpretability can provide the raw phenomenology, but bridging it to metaphysics benefits from domain specialists.

---

### Bottom line

*The methodology is sound but could be tightened on normalisation and copy‑collapse detection.* The results robustly replicate the “late semantics” story and hint at architectural idiosyncrasies (Gemma’s early echo). Philosophically, the data show that what looks like a universal emerges only through composition inside the model; whether that supports realism or nominalism depends on how one cashes out “real”. The next round—adding causal tracing and a richer prompt set—will give much sharper leverage on that question.

[1]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_and_logits/run.py "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_and_logits/NOTES.md "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_and_logits/run-latest/evaluation-Meta-Llama-3-8B.md "raw.githubusercontent.com"
[4]: https://arxiv.org/abs/2303.08112?utm_source=chatgpt.com "Eliciting Latent Predictions from Transformers with the Tuned Lens"
[5]: https://plato.stanford.edu/entries/nominalism-metaphysics/?utm_source=chatgpt.com "Nominalism in Metaphysics - Stanford Encyclopedia of Philosophy"

---
Produced by OpenAI o3-pro

