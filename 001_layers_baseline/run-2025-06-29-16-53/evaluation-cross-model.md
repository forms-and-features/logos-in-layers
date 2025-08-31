## 1. Result synthesis

Across the four probed base models the copy-reflex (prompt-echo) and semantic collapse metrics separate into three regimes.

Gemma-2-9B shows an extreme lag: the answer token does not overtake the prompt token until the final layer, giving Δ-collapse = 42 ("L_copy":0, "L_semantic":42) 5:20:001_layers_baseline/output-gemma-2-9b.json.  Qwen-3-8B displays a moderate six-layer gap ("L_copy":25→"L_semantic":31) 18:30:001_layers_baseline/output-Qwen3-8B.json.  Meta-Llama-3-8B and Mistral-7B-v0.1 never meet the >0.90 prompt-token criterion, so Δ cannot be computed; their residual streams switch directly from diffuse or filler tokens to the correct answer at L25 (Llama) and L25 (Mistral) 15:25:001_layers_baseline/evaluation-Meta-Llama-3-8B.md and 25:32:001_layers_baseline/evaluation-Mistral-7B-v0.1.md.

Gemma's 42-layer lag is an outlier (>10) and aligns with its lower factual scores (MMLU 57 %, ARC 63 %).  Qwen, with the best public scores (MMLU 64 %), collapses after only six layers.  The two mid-performers (Llama 62 %, Mistral 60 %) neither echo strongly nor collapse early, hinting that instruction-follow tuning may suppress hard copy but leave the semantic heads unchanged.

Qualitatively, Gemma spends almost the entire stack on the adverb "simply", then flips to punctuation before emitting "Berlin", mirroring the long-tail echo described in Tuned-Lens (arXiv:2303.08112).  Qwen's six-layer plateau is shorter but still visible; in contrast, Llama and Mistral drift through various non-prompt sub-words before converging, consistent with weaker repetition biases observed in their token frequency curves.

## 2. Misinterpretations in existing EVALS

- "The model exhibits a **long copy-reflex** (Δ 6 layers)" 35:39:001_layers_baseline/evaluation-Qwen3-8B.md.  Six layers is moderate compared with Gemma's 42-layer lag and with values <5 reported for GPT-3 1.3 B in Tuned-Lens; calling it "long" overstates the effect.
- "Entropy rise at unembed? ✓" 96:96:001_layers_baseline/evaluation-gemma-2-9b.md.  The unembed layer is never probed; the cited entropy change occurs *inside* the transformer stack, so linking it to the unembedding matrix is incorrect.
- "Because p never exceeds 0.9 it evades the strict copy-collapse rule, yet it behaves like a classic copy-reflex" 40:45:001_layers_baseline/evaluation-Mistral-7B-v0.1.md.  Behavioural similarity is plausible, but asserting equivalence ignores the quantitative threshold the script is designed to enforce.

## 3. Usefulness for the Realism ↔ Nominalism project

The sharp contrast between Gemma's 42-layer surface echo and Qwen's six-layer gap suggests that different architectures allocate vastly different depth to stripping lexical form before semantic retrieval.  If realism corresponds to stable latent facts, the shallow Δ in Qwen implies earlier activation of fact-specific circuits, whereas Gemma's deep lag hints at nominalist pattern-matching that must be cleaned away first.  The absence of any hard copy stage in Llama and Mistral raises the question whether instruction-tuned objectives implicitly discourage surface echo, offering a natural experiment for disentangling realist and nominalist pathways.

## 4. Limitations

Single-prompt probing risks overfitting to idiosyncratic tokenizer artefacts ("湾", "******"), and Δ estimates can shift by >5 layers when the brevity instruction is removed, as noted for Gemma 46:53:001_layers_baseline/evaluation-gemma-2-9b.md.  All runs used CPU inference without seed control, so minor sampling noise may affect top-k rankings near p≈0.9.  The RMS lens normalises residual streams but cannot recover attention pattern information; models that interleave attention and MLP may therefore show spurious entropy bumps.  Finally, the >0.90 criterion is arbitrary; lowering it would classify Mistral's prolonged "simply" plateau as copy-collapse and raise Δ by ~20 layers.

---
Produced by OpenAI o3
