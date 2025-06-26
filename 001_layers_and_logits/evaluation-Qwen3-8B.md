# 1. Overview

**Model:** Qwen/Qwen3-8B (8 B params)  
**Run date:** see console header – file generated on current run  
The probe feeds a short Q&A prompt into the model and, with TransformerLens hooks, prints the residual stream after every block, applies an RMS-norm lens, then unembeds to obtain layer-wise logits.

# 2. Method sanity-check
* "Block normalization type: RMSNormPre" [L8] confirms RMS rather than LN.  
* "Using NORMALISED residual stream (RMS, no learnable scale)" [L9] shows the correct norm-lens was applied.  
* "[diagnostic] No separate positional embedding hook; using only token embeddings for layer 0 residual." [L28] clarifies that positional embeddings are merged with token embeddings and therefore captured.

# 3. Quantitative findings
| Layer | Entropy (bits) | Top-1 token |
|-------|---------------|-------------|
| 0 | 5.671 | いらっ |
| 1 | 10.974 | ListViewItem |
| 2 | 10.000 | Buccane |
| 3 | 9.901 | Lauderdale |
| 4 | 11.345 | Buccane |
| 5 | 12.906 | 直接影响 |
| 6 | 14.145 | 我省 |
| 7 | 13.430 | portion |
| 8 | 12.072 | steller |
| 9 | 11.299 | Mus |
| 10 | 11.532 | 在游戏中 |
| 11 | 12.148 | 在游戏中 |
| 12 | 12.156 | Answer |
| 13 | 11.685 | Answer |
| 14 | 10.953 | Binary |
| 15 | 11.604 | Answer |
| 16 | 11.712 | Answer |
| 17 | 12.629 | Answer |
| 18 | 10.841 | Answer |
| 19 | 8.556 | Answer |
| 20 | 3.708 | Answer |
| 21 | 2.600 | Answer |
| 22 | 3.370 | Answer |
| 23 | 3.550 | Answer |
| 24 | 4.101 | ______ |
| 25 | 1.411 | Germany |
| 26 | 2.857 | ____ |
| 27 | 2.917 | Germany |
| **28** | **0.430** | **Berlin** |
| 29 | 0.194 | Berlin |
| 30 | 0.871 | Berlin |
| 31 | 0.011 | Berlin |
| 32 | 0.076 | Berlin |
| 33 | 0.011 | Berlin |
| 34 | 0.012 | Berlin |
| 35 | 0.130 | Berlin |
| 36 | 2.020 | Berlin |

The first entropy < 1 bit occurs at **layer 28** – the "collapse layer".

# 4. Qualitative patterns & anomalies
* **Sharp collapse on factual answer:**
  > Layer 28 … (entropy 0.430) 'Berlin' 0.93 [L662]
* **'Answer' obsession in mid-stack:** dominates layers 18-24 while entropy drifts 10.8→3.7 bits.  
  > Layer 20 top-1 'Answer' 0.85 [L470]
* **Underscore/blank-token ("colon-spam") phase** before collapse – high-prob '____' tokens at layer 26.  
  > Layer 26 top-1 '____' 0.30 [L613]
* **Entropy re-opens after unembed:** final layer rises to 2.02 bits, injecting distractors ('The', blank).  
  > Layer 36 entropy 2.020 bits [L838]

Checklist: RMS used ✓ | LayerNorm ✗ | Colon-spam ✓ | Re-opening at unembed ✓

# 5. Implications & open questions for Realism ↔ Nominalism
* Does the late collapse (layer 28) suggest that abstract category tokens ('Answer', '____') precede concrete entity instantiation ('Berlin')?
* Could the persistence of low-entropy 'Berlin' through layers 28-35 reflect a realist "object permanence" representation?  
* Does the re-opening at the final layer indicate nominalistic surface-form tailoring rather than changed belief?
* Are the underscore tokens a nominal placeholder for "nothing yet" rather than a realist commitment to a specific answer?

# 6. Limitations & data quirks
* Single prompt and single run; no statistics across inputs.  
* The probe stores only the last-token residual; earlier positions are not inspected.  
* Non-English vocabulary dominates early layers, possibly due to tokenizer idiosyncrasies.

# 7. Next probes with current artefact
1. Activation patching: replace layer 28 residual with layer 27 to test whether collapse is necessary for correct answer.  
2. Compare entropy trajectory on semantically similar but false questions (e.g., "capital of Australia → Berlin?") using same hooks.  
3. Compute KL divergence between layers 27-29 to quantify the sharpness of the collapse.

# 8. Model fingerprint
"Qwen3-8B: entropy collapses at layer 28; final entropy 2.0 bits; anomalous underscore spam before collapse, 'Berlin' locked in thereafter."

---

Produced by OpenAI o3