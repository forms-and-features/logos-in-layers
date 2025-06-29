# 1. Overview

Meta-Llama-3-8B (≈8 B parameters) was probed on 2025-06-29 using the `run.py` script, which captures layer-by-layer logit-lens views, per-layer entropy and top-k token probabilities, plus follow-up probes over alternative prompts and temperatures.  The artefacts analysed here are the structured JSON summary and two CSVs with detailed layer traces.

# 2. Method sanity-check

The JSON diagnostics report `"use_norm_lens": true` and `"layer0_position_info": "token_only_rotary_model"`, indicating that the intended RMS-norm lens was applied and that positional information is injected later via rotary attention rather than additive embeddings.  Console output corroborates this:

> "Using NORMALIZED residual stream (RMS + learned scale)" [L310]  
> "[diagnostic] Layer 0 contains TOKEN information only; positional info is injected inside attention layers." [L331]

Both CSVs list 33 rows (L 0–32) for the pure next-token position, matching the model's 32 transformer blocks and confirming hook placement after each block.

# 3. Quantitative findings

| Layer | Entropy (bits) | Top-1 token |
|------:|---------------:|-------------|
| L 0 | 14.30 | `톤` |
| L 1 | 13.60 | `Disposition` |
| L 2 | 13.04 | `.updateDynamic` |
| L 3 | 13.26 | `/lists` |
| L 4 | 11.64 | `"×␤"` |
| L 5 |  8.30 | `'gc` |
| L 6 | 12.56 | `.scalablytyped` |
| L 7 | 11.73 | `.scalablytyped` |
| L 8 | 12.54 | `.scalablytyped` |
| L 9 | 13.46 | `Radians` |
| L 10 | 13.37 | `Radians` |
| L 11 | 12.98 | `UGC` |
| L 12 | 12.70 | `engo` |
| L 13 | 13.42 | `ーデ` |
| L 14 | 13.60 | `#ab` |
| L 15 | 12.94 | `#ab` |
| L 16 | 13.19 | `.habbo` |
| L 17 | 12.31 | `ynom` |
| L 18 |  3.52 | `distance` |
| L 19 |  8.93 | `distance` |
| L 20 |  5.58 | `distance` |
| L 21 |  6.85 | `distance` |
| L 22 |  9.05 | `capital` |
| L 23 | 10.72 | `distance` |
| L 24 | 13.27 | `capital` |
| L 25 | 13.64 | `capital` |
| L 26 | 13.98 | `miles` |
| L 27 | 14.26 | `rome` |
| L 28 | 14.45 | `rome` |
| L 29 | 12.44 | `rome` |
| L 30 |  9.91 | `iber` |
| L 31 |  6.18 | `rome` |
| L 32 |  8.70 | `1` |

No layer crosses the <1 bit threshold, so no collapse point is bolded.

# 4. Qualitative patterns & anomalies

From L 0–17 the model oscillates between obscure code-style tokens (e.g. `.scalablytyped`, `UGC`) and byte-level artefacts, with entropy staying above 8 bits until L 5 where it briefly dips to 8.3 bits.  A sharp **entropy trough at L 18 (3.5 bits)** coincides with the token *distance*, suggesting a transient over-confidence on a semantically unrelated trajectory.  Subsequent layers partially recover entropy yet remain anchored to *distance/ capital* motifs up to L 24 where *Berlin* finally surfaces in rank 2 (prob ≈ 2.7 %).  Nevertheless the final unembedded distribution reverts to numeric tokens, with *1* top-ranked and entropy 8.7 bits, indicating the collapse did not propagate to the head.

Alternative prompts show mixed reliability: for "Germany's capital city is called " the answer *Berlin* is rank 2 with 9 % mass (> "… (‘ Berlin’, 0.09)" [L233]).  Temperature exploration reveals that low-temperature (τ = 0.1) rescales the logits so that *Berlin* captures the entire distribution (entropy ~0 bits), whereas high temperature (τ = 2.0) keeps *Berlin* but disperses mass widely (entropy 14.5 bits) (> "… temperature": 2.0, "entropy": 14.52" [L460]).

Checklist:
- RMS lens? ✓  
- LayerNorm? n.a. (RMSNorm model)  
- Colon-spam? ✗  
- Entropy rise at unembed? ✓

# 5. Tentative implications for Realism ↔ Nominalism

• Does the late appearance of *Berlin* (after L 24) imply that factual knowledge about capitals is stored in deep MLP sub-spaces rather than emerging compositionally earlier?  

• Might the transient over-confidence on *distance* at L 18 indicate that the model's intermediate representations latch onto surface-level lexical associations ("distance from Germany") rather than abstract relations, supporting a nominalist view of knowledge encoding?  

• How stable is the factual circuit across prompts given the strong dependence on temperature and phrasing observed here? Could realism-style latent facts be present yet obscured by nominalist lexical priors?  

• Would linking cross-layer attention patterns to the entropy oscillations clarify whether the 'distance' attractor stems from a specific head, and thereby illuminate whether abstract truth values or surface statistics dominate?  

# 6. Limitations & data quirks

The run executed on CPU, prolonging inference and potentially altering timing-dependent hooks but not logits.  Tokenisation emits many byte-level or code-artifact tokens, hinting at tokenizer domain mismatch.  The probe inspects only a single next-token position; multi-step generation dynamics remain untested.  CSV entropies are derived from the top-k slice plus rest-mass approximation; very low-probability tails are aggregated rather than explicit.

# 7. Model fingerprint

"Meta-Llama-3-8B: sharp entropy dip at L 18 without full collapse; 'Berlin' only reaches rank 2 by L 24; final head entropy 8.7 bits with numeric token '1' dominant."

---
Produced by OpenAI o3
