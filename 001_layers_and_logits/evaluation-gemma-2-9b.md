### 1. Overview

Google Gemma-2-9B (≈9 B params, base) was probed on 2025-06-26 with the layer-wise norm-lens script `run.py`.  The probe feeds a short Q→A prompt and, at every residual stream (embeddings + 42 transformer blocks), projects the residual through the model's unembedding to log the entropy and top-k token probabilities.

### 2. Method sanity-check

* Console reports "Using **NORMALISED residual stream (RMS, no learnable scale)" confirming that the RMS-norm lens path is used.
* The model exposes no separate positional embedding hook – the script notes  
  `No separate positional embedding hook; using only token embeddings for layer 0 residual.`  
  Together these show that the probe applied the intended RMS lens and handled positions correctly.

### 3. Quantitative findings

| Layer | Entropy (bits) | Top-1 token |
|-------|---------------|-------------|
| **0** | **0.000** | ':' |
| 1 | 0.000 | ':' |
| 2 | 0.000 | ':' |
| 3 | 0.000 | ':' |
| 4 | 0.000 | ':' |
| 5 | 0.000 | ':' |
| 6 | 0.000 | ':' |
| 7 | 0.000 | ':' |
| 8 | 0.003 | ':' |
| 9 | 0.079 | ':' |
| 10 | 0.738 | ' answer' |
| 11 | 0.326 | ':' |
| 12 | 0.279 | ':' |
| 13 | 0.968 | ':' |
| 14 | 1.081 | 'a' |
| 15 | 2.239 | 'a' |
| 16 | 1.679 | 'a' |
| 17 | 2.203 | 's' |
| 18 | 2.402 | ' ' |
| 19 | 2.453 | ' ' |
| 20 | 1.056 | ' the' |
| 21 | 0.399 | ' the' |
| 22 | 1.048 | ' the' |
| 23 | 0.954 | ' the' |
| 24 | 1.014 | ' the' |
| 25 | 1.539 | 'The' |
| 26 | 1.155 | 'The' |
| 27 | 0.492 | 'The' |
| 28 | 1.037 | 'The' |
| 29 | 0.773 | 'The' |
| 30 | 1.353 | ' The' |
| 31 | 1.277 | ' The' |
| 32 | 0.786 | ' The' |
| 33 | 1.136 | ' The' |
| 34 | 1.150 | ' Berlin' |
| 35 | 0.235 | ' Berlin' |
| 36 | 0.079 | ' Berlin' |
| 37 | 0.166 | ' Berlin' |
| 38 | 0.009 | ' Berlin' |
| 39 | 0.008 | ' Berlin' |
| 40 | 0.004 | ' Berlin' |
| 41 | 0.000 | ' Berlin' |
| 42 | 0.000 | ' Berlin' |

**Bold layer 0 is the first time entropy drops below 1 bit ("collapse layer").**

### 4. Qualitative patterns & anomalies

* Early **colon-spam** – eight consecutive layers output ":" with entropy ≈ 0  
  > Layer 3 … ':' (1.000) [L104]  
* Gradual entropy rise around block 10 before falling again  
  > Layer 10 entropy 0.738 bits, top-1 ' answer' [L180]  
* Second collapse centred on common article "the"  
  > Layer 21 entropy 0.399 bits, top-1 ' the' [L515]  
* Final semantic convergence on **"Berlin"** from layer 34 onward  
  > Layer 35 entropy 0.235 bits, top-1 ' Berlin' [L842]  
* **Re-opening at unembed** – although last hidden layers are near-deterministic, the unembedding step produces 2.0 bits of entropy (`Model's final prediction (entropy: 2.003 bits)`) [L1110].

Checklist: RMS used ✓ | LayerNorm ✗ | Colon-spam ✓ | Re-open at unembed ✓

### 5. Implications & open questions for Realism ↔ Nominalism

* Does the early colon-loop indicate nominal, surface-token matching before any conceptual grounding – a hint of nominalist processing?
* The sharp convergence on "Berlin" without intermediate competing capitals – realism that a stable abstract entity emerges by layer 34?
* Why does entropy re-inflate at the unembedding – does the semantic representation fan-out into lexical variants (realist "type" mapping to many "token" instantiations)?
* Are the alternating collapses (':' → 'the' → 'Berlin') evidence of multi-stage abstraction from syntax to semantics?

### 6. Limitations & data quirks

* Single prompt; conclusions may not generalise across prompts or languages.
* The probe stores only the last-token residual; interactions with earlier positions are unobserved.
* Layer-wise RMS lens for Gemma is heuristic; mis-scaling could distort entropies.
* File shows perfect zero entropy in many layers – may reflect numeric under-flow rather than strict determinism.

### 7. Next probes with current artefact

1. Activation patching: overwrite layer 0 vs layer 34 residuals with random noise and re-run unembedding to test robustness of collapse.
2. Compute attention pattern entropy in collapse vs semantic layers to see whether heads also collapse.
3. Re-scale the unembedding weights (FP32 vs FP16) within the existing dump to verify whether final 2 bits entropy is stable.

### 8. Model fingerprint

"Gemma-2-9B: entropy collapses from the outset; secondary collapse at layer 34; final unembed entropy 2.0 bits; early colon-spam."

---

Produced by OpenAI o3