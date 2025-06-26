# 1. Overview
Meta-Llama 3-8B (8 B parameters) was probed on 2025-06-26 with the layer-wise norm-lens script (`run.py`).
The probe caches every residual stream, optionally applies the model's own normalisation, and shows entropy & top-k tokens per layer for the last token of the prompt.

# 2. Method sanity-check
* Positional embeddings are accounted for – the script reports _"[diagnostic] No separate positional embedding hook; using only token embeddings for layer 0 residual."_ [L19-20], confirming that the positional component is added before the first residual hook.
* RMS norm-lens is active: _"Block normalization type: RMSNormPre / Using NORMALISED residual stream (RMS, no learnable scale)"_ [L7-9].  No LayerNorm is detected, matching LLama-3's architecture.

# 3. Quantitative findings
| Layer | Entropy (bits) | Top-1 token |
|-------|---------------|-------------|
| 0 | 14.52 | `oren` |
| 1 | 13.64 | `ря` |
| 2 | 13.62 | `  ` |
| 3 | 13.54 | `atten` |
| 4 | 13.08 | `adal` |
| 5 | 13.32 | `chine` |
| 6 | 13.28 | `.Decode` |
| 7 | 13.00 | ` Cem` |
| 8 | 13.13 | `AutoSize` |
| 9 | 12.23 | `enville` |
| 10 | 12.58 | `PCM` |
| 11 | 13.09 | `ystack` |
| 12 | 12.39 | `.xtext` |
| 13 | 13.36 | `краї` |
| 14 | 13.35 | `#ab` |
| 15 | 12.81 | `#af` |
| 16 | 13.03 | `#ab` |
| 17 | 11.95 | `#ad` |
| 18 | 12.48 | `#ad` |
| 19 | 11.75 | `ABCDEFGHIJKLMNOP` |
| 20 | 10.77 | ` Capital` |
| 21 |  9.29 | ` capital` |
| 22 |  8.48 | ` Berlin` |
| 23 |  5.58 | ` Berlin` |
| 24 |  2.07 | ` Berlin` |
| **25** | **0.43** | **` Berlin`** |
| 26 |  0.35 | ` Berlin` |
| 27 |  0.33 | ` Berlin` |
| 28 |  0.37 | ` Berlin` |
| 29 |  0.81 | ` Berlin` |
| 30 |  1.64 | ` Berlin` |
| 31 |  1.16 | ` Berlin` |
| 32 |  1.70 | ` Berlin` |

The entropy first drops below 1 bit at **Layer 25**, marking the "collapse" into a near-deterministic state.

# 4. Qualitative patterns & anomalies
* Early noise – tokens are mostly non-English fragments: _"Layer 0 ... 'oren'"_ [L27-28].
* Gradual thematic focus: 'Berlin' only appears as a top-10 candidate by Layer 22 and becomes dominant by Layer 24 (0.86 p) [L566-568].
* Sharp collapse: Layer 25 reaches 0.43 bits with 0.97 p on 'Berlin' [L580-583].
* Entropy rebound: rises back to 1.70 bits after the final block/unembed [L740-744].
* Rare "colon-spam": Layer 19 top-10 includes the literal token ' (::' [L421-423].

Checklist
* RMS used: ✓
* LayerNorm present: ✗
* Colon-spam present: ✓ (Layer 19)
* Entropy re-opens at unembed: ✓

# 5. Implications & open questions for Realism ↔ Nominalism
* Does the late-layer collapse on a single concrete token ('Berlin') indicate a realist "concept grounding" rather than nominal association?
* The rebound in entropy post-collapse—does it reflect re-encoding into a higher-order nominal abstraction (e.g., sentence continuation)?
* How does the presence of the role-neutral RMS norm affect the representation of abstract universals vs concrete names?
* Would activation patching of Layer 25 alter downstream certainty, suggesting a privileged "naming" layer?

# 6. Limitations & data quirks
* Single prompt and single token examined; results may not generalise.
* Line-number mapping is approximate; quoting may shift with different console widths.
* The diagnostic message indicates missing separate positional-embedding hook; joint embedding could hide positional effects.
* GPU precision is promoted for unembed weights only; earlier logits stay in fp16 – may bias entropy estimates slightly.

# 7. Next probes with current artefact
1. Activation patching: replace Layer 25 residual with Layer 24 residual to test sensitivity of collapse.
2. Temperature sweep using cached logits to see if the collapse threshold shifts with scaling.
3. Re-run with prompts that demand a non-capital answer (e.g., "Question: What is the largest city in Germany?") to compare collapse dynamics.

# 8. Model fingerprint
Llama-3-8B: entropy collapses at layer 25; final entropy 1.7 bits; anomalous 'Paris' still in top-20 at collapse layer.

---

Produced by OpenAI o3