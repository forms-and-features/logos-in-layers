# Cross-Model Evaluation of Δ-Collapse Patterns

## 1. Result synthesis
Across the four probed base checkpoints the distance between the first high-confidence echo of a **prompt token** (L_copy) and the first appearance of the **ground-truth answer** "Berlin" (L_semantic) varies widely.

* **Gemma-2-9B** shows an extreme lag. JSON exposes `"L_copy": 0, "L_semantic": 42` ( 16-20 ) giving **Δ = 42 layers** – the model parrots *"simply"* from the embedding stream right up to the final block.  Such a depth places Gemma far outside the cluster and suggests that its factual circuit only overrides surface heuristics after full stack aggregation.
* **Qwen-3-8B** produces a moderate gap. Diagnostics list `"L_copy": 25` and `"L_semantic": 31` ( 18-20 ) → **Δ = 6 layers**.  The copy reflex appears late and is quickly replaced by the city name; this aligns with its stronger public reasoning scores (MMLU 64 %, ARC-C 72 %).
* **Meta-Llama-3-8B** and **Mistral-7B-v0.1** report `L_copy = null` in their JSON (Llama 15-20; Mistral 18-21).  Neither model crosses the 0.9-probability echo threshold, although both linger on *"simply"* for many layers (Llama 14-19, Mistral 4-24).  We therefore treat them as **Δ ≈ 0** under the strict criterion, but note a "soft copy plateau" up to the mid-stack.

Taken together, only Gemma exhibits a Δ larger than ten layers; Qwen's six-layer gap is the next largest, while the two remaining models collapse semantically without meeting the hard copy criterion.  A coarse correlation emerges: the two higher-scoring models on MMLU/ARC (Qwen and Llama) escape surface echoes sooner, whereas Gemma, the lowest scorer, shows the deepest lag.

## 2. Misinterpretations in existing per-model write-ups
• Gemma report asserts the reflex "persists through **41** layers" (line 11) yet the JSON gives `L_semantic = 42`; the lag is therefore 42, not 41.
• Llama report states that entropy "plunges" from 12.33 to 7.84 bits (lines 37-40).  A 4.5-bit drop is moderate; CSV rows confirm entropy remains >7 bits for several layers, so the phrasing exaggerates the sharpness.
• Mistral write-up labels the prolonged *"simply"* phase as "no copy-reflex" (line 21) despite probabilities up to 0.77 (line 19).  While below the 0.9 threshold, this still signifies a partial echo and should be acknowledged as such.

## 3. Usefulness for the Realism ↔ Nominalism project
The spread in Δ suggests a potential signal for **where** semantic grounding overrides lexical copying.  Models with shorter or null Δ (Llama, Mistral) may bind entities earlier in the computation, hinting at more realist internal representations, whereas Gemma's long lag could indicate a stronger nominalist bias where surface tokens dominate until late aggregation.  Future work could examine whether fine-tuning that improves factual benchmarks shortens Δ, thereby tying realism to performance improvements.

## 4.; Limitations
The analysis relies on a single template prompt; collapse depths might be highly prompt-dependent.  The 0.9 probability threshold for L_copy is heuristic – lowering it would mark "soft" copy plateaus in Llama and Mistral.  All probes run in fp32 on CPU, so activation noise and timing may differ under mixed-precision GPU inference.  Finally, entropy values are computed from truncated top-k distributions, which can under-estimate uncertainty and distort layer-wise trends.

---
Produced by OpenAI o3
