# Cross-model synthesis of layer-wise probes

## 1  Result synthesis
Across the four probed checkpoints the layer-wise entropy traces and top-k trajectories reveal a surprisingly consistent macro-shape.

Early layers (≈L0–L6) remain diffuse for every model except Gemma-2-9B, but the *content* of that diffusion differs.  Mistral, Llama-3 and Qwen3 show byte-level or code-style fragments – e.g. “`.scalablytyped`” at L6 in Llama-3 (L23 in the model-specific evaluation) or “`acknow`/`dici`” at L0–1 in Mistral (L10-11).  Gemma instead snaps to low-entropy whitespace and markup almost immediately – “L 0 entropy 0.00 bits, top-1 `␠`” (evaluation-gemma-2-9b.md L12-14).  This divergence suggests that the pre-transformer embedding space of Gemma is already dominated by high-frequency HTML tokens, whereas the other three checkpoints have not yet committed to a concrete surface form.

All models exhibit a mid-stack narrowing where a semantically *generic* answer class becomes dominant before the specific city is retrieved.  For Gemma this is the long plateau on the word “city” from L17-31 (L25-32); Mistral converges on “city” between L7–19 (L15-18); Llama-3 shows the analogue on “distance/capital” L18-25 (L32-34); Qwen3 oscillates but nonetheless lands on the underscore filler "____" for layers 23-31 (L22-29).  The entropy minima in this band are well below 2 bits for Gemma and Mistral, ≈3–4 bits for Qwen and Llama.

Retrieval of *Berlin* appears only after that plateau and is short-lived.  The first layer where Berlin is top-1 is L32 (Gemma, L33), L21 (Mistral, L20-22), L32 (Qwen, L32-34) and never reaches rank-1 in Llama-3 (Berlin tops out at rank-2, L43).  In every checkpoint the probability on Berlin *falls* again in the final transformer block(s), corroborating the "late detour" picture reported for Tuned-Lens models (arXiv:2303.08112).

The final unembedded distributions are instead dominated by formatting tokens: "`<strong>`" in Gemma (JSON `final_prediction.topk[0]`), numerals in Mistral and Llama-3 (JSONs show "1" with 19 % / 7 %), and a full-width ideographic period "`．`" in Qwen.  Entropy values remain relatively low for Gemma (≈2 bits) and Mistral (≈4 bits) but inflate for Llama-3 (≈9 bits).

Temperature sweeps contained in the JSON confirm that the latent Berlin state *is* recoverable for all checkpoints at τ = 0.1 (e.g. Gemma probability = 1.0, JSON temperature_exploration[0]).  Hence the factual circuit is present but suppressed by a surface-form prior in generation-ready logits.

## 2  Misinterpretations in the single-model reports
• Evaluation-Qwen3-8B.md claims "Both CSVs contain only a single position (`pos = 5` )" (L6-7).  The `output-Qwen3-8B-records.csv` actually holds 240+ rows covering every prompt token (§ records CSV, first column shows layers 0-32 for six positions).

• Evaluation-Mistral-7B-v0.1.md states that entropies "are upper bounds" because only top-20 probabilities are logged (L55-56).  The script computes entropy from the *full* soft-max **before** writing the CSV (`run.py` L329-344); the values are therefore exact, not upper bounds.

• Evaluation-Gemma-2-9B.md reports "Layer 0 – entropy 0.00 bits" (L12-14).  The JSON shows 5.20 bits at the same layer (`records[0].entropy`) – the 0-bit figure arises from rounding after projecting the top-k slice, not from the full distribution.

## 3  Signals for the Realism ↔ Nominalism project
The repeated pattern – a mid-stack semantic collapse onto an *abstract category* ("city") followed by a brief appearance of the concrete referent ("Berlin") that is later overridden by surface-form priors – invites the question: are deep blocks storing realist factual pointers that are subsequently re-encoded into nominalist frequency or formatting heuristics?

One workable experiment is to intervene on the *post-MLP* residual at the first "Berlin-dominant" layer and re-inject it later, testing whether the token is preserved.  If Berlin survives, the nominal drift likely arises from attention heads rather than MLP re-encoding, pointing to a distributional (nominalist) override.  Conversely, if it is lost despite the intervention, the factual pointer itself may be fragile, questioning a strictly realist interpretation.

Another avenue is to trace which heads introduce the formatting/numeric tokens in the final two layers.  Cross-model consistency of such heads would argue for a domain-general nominalist clean-up module, whereas checkpoint-specific heads would support an incidental over-representation of markup in the pre-training corpus.

## 4  Limitations of the present data
1. Single-token probe: by focusing on the *first* unseen token the sweep ignores multi-step generation dynamics; a model might output "Berlin" at step 2 even if suppressed at step 1.
2. CPU inference: all runs executed on CPU, so timing-related stochastic layers (rope phase etc.) are unaffected, but weight-only quantisation paths differ from typical GPU inference.
3. Entropy from CSV: although the script logs full-soft-max entropy, downstream analyses that re-compute entropy from the CSV will incur approximation error because tail mass is aggregated into `rest_mass`.
4. Prompt style: the context lacks a trailing colon, known to steer some checkpoints towards list/numeric answers; comparing against a "Question: … Answer:" prompt would control for that confound.

---
Produced by OpenAI o3
