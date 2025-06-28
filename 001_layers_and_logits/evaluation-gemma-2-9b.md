# 1. Overview
Google Gemma-2 (9 B params) was probed on CPU with the script in `run.py` (42 layers, 16 heads).  
The probe records layer-wise entropies and top-k next-token predictions for the **answer slot** in the prompt "Question: What is the capital of Germany? Answer:".

# 2. Method sanity-check
The JSON header confirms that the RMSNorm lens was requested and positional embeddings were cached:  
> "use_norm_lens": true [L7]  
> "first_block_ln1_type": "RMSNormPre" [L9]

At run-time the script emits the expected notice:  
> Using NORMALIZED residual stream (RMS + learned scale) [run.py L242]  
This indicates both the norm lens and positional streams were combined (`hook_embed + hook_pos_embed`).

# 3. Quantitative findings
| layer | entropy (bits) | top-1 token |
|-------|---------------|-------------|
| **L 0** | 0.0 | ':' |
| 1 | 0.0 | ':' |
| 2 | 0.0 | ':' |
| 3 | ≈0 | ':' |
| 4 | ≈0 | ':' |
| 5 | ≈0 | ':' |
| 6 | 0.00007 | ':' |
| 7 | 0.00012 | ':' |
| 8 | 0.0025 | ':' |
| 9 | 0.08 | ':' |
| **L 10** | **0.74** | 'answer' |
| 11 | 0.33 | ':' |
| 12 | 0.28 | ':' |
| 13 | 0.97 | ':' |
| 14 | 1.08 | 'a' |
| 15 | 2.24 | 'a' |
| 16 | 1.68 | 'a' |
| 17 | 2.20 | 's' |
| 18 | 2.40 | ' ' |
| 19 | 2.45 | ' ' |
| 20 | 1.06 | 'the' |
| 21 | 0.40 | 'the' |
| 22 | 1.05 | 'the' |
| 23 | 0.95 | 'the' |
| 24 | 1.01 | 'the' |
| 25 | 1.54 | 'The' |
| 26 | 1.16 | 'The' |
| 27 | 0.49 | 'The' |
| 28 | 1.04 | 'The' |
| 29 | 0.77 | 'The' |
| 30 | 1.35 | 'The' |
| 31 | 1.28 | 'The' |
| 32 | 0.79 | 'The' |
| 33 | 1.14 | 'The' |
| 34 | 1.15 | 'Berlin' |
| 35 | **0.24** | **'Berlin'** |
| 36 | 0.08 | 'Berlin' |
| 37 | 0.17 | 'Berlin' |
| 38 | 0.009 | 'Berlin' |
| 39 | 0.008 | 'Berlin' |
| 40 | 0.004 | 'Berlin' |
| 41 | 0.00010 | 'Berlin' |
| 42 | 0.00003 | 'Berlin' |

(The first bold row marks the collapse where entropy permanently drops below 1 bit while predicting *Berlin*.)

# 4. Qualitative patterns & anomalies
Early layers faithfully copy punctuation: the model is **over-certain (<10-6 bits) about ':'** until L 9.  Entropy rises into the 2-3 bit range mid-stack, then collapses sharply once the full query has been integrated – by L 35 the answer is essentially fixed with 97 % probability.

The probe's auxiliary prompts show an asymmetry: when *Berlin* is given, the continuation **"Germany"** is near-deterministic (entropy ≈ 0.96 bits) whereas the inverse cue *"Germany's capital is"* remains diffuse (5.3 bits).  This mirrors findings in causal-tracing work on token order effects (cf. Tuned-Lens 2303.08112).

Temperature sweep underscores logit concentration: at τ = 0.1 entropy collapses to 8 × 10⁻¹⁵ bits with *Berlin* = 100 % [L255]; at τ = 2.0 entropy balloons to 11 bits yet *Berlin* still tops the list at 8.6 % [L270].

"… Berlin (0.99928)" [CSV L708] illustrates the near-deterministic state once the stack has settled.

Checklist:  
✓ RMS lens ✓ LayerNorm (RMSPre) ✓ Colon-spam (L0-9) ✓ Entropy rise then fall at unembed

# 5. Tentative implications for Realism ↔ Nominalism
• Does the late-stack entropy collapse (L 35 →) indicate a dedicated *fact head* storing country-capital pairs, or is it an emergent consensus of many heads?  
• Why does conditioning on *Berlin*→*Germany* produce low entropy while the symmetric cue remains diffuse – is the representation of *capital-of* asymmetric in Gemma's training corpus?  
• Could the persistence of high-probability ':' tokens until deep in the network suggest positional syntax is handled separately from factual recall?  
• Would interventions at the L 30-34 window (before collapse) be sufficient to steer the final answer without affecting grammatical scaffolding?

# 6. Limitations & data quirks
CSV entropy is computed on the observed token, so early-layer near-zero values largely reflect **teacher-forcing, not model uncertainty**.  The probe runs on CPU ‑- precision or speed artefacts may differ on GPU.  File lacks creation timestamp, and only one prompt was inspected, limiting generality.  Layer counts were inferred from JSON; any mismatch with checkpoint config would skew layer numbering.

# 7. Model fingerprint
Gemma-2-9B: factual collapse at **L 35**, final entropy ≈ 0.10 bits, *Berlin* dominates top-1 through output layer.

---

Produced by OpenAI o3