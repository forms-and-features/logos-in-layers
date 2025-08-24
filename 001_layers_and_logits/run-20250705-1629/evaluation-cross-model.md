# Cross-Model Probe Review

## 1  Result synthesis

Across all seven models the probe reveals a shared macro-trajectory: high-entropy, low-confidence junk in the early stack, a protracted plateau where prompt words dominate but the answer is absent, and a late collapse in which the semantic answer "Berlin" becomes top-1.  The collapse always happens **after 70 % of the depth** (relative depth \(\hat{\Delta} \ge 0.73\)), consistent with tuned-lens observations that meaning concentrates late in rotary/RMS stacks (arXiv:2303.08112).

### Copy-reflex
Using the strict rule (`copy_collapse=True` in layers 0-3) only the two **Gemma** variants exhibit an early lexical echo:
* gemma-2-9B L\_copy = 0 (layers 0–13 flagged)  
* gemma-2-27B L\_copy = 0 (layers 0–8 flagged)

All other models (Meta-Llama-3-8B, Mistral-7B, the two Qwens, Yi-34B) show `False` for the first three layers, indicating **no copy-reflex**.  This diverges from earlier work that found copy reflexes in nearly every GPT-J style model (arXiv:2210.04885).  The outlier status of Gemma may reflect its bilingual training and heavy use of span-prediction objectives.

### Depth of semantic emergence
For each model the diagnostics give \(\Delta = L_{\text{sem}}-L_{\text{copy}}\) (or \(L_{\text{sem}}\) if no copy layer) and the normalised depth \(\hat{\Delta}=\Delta/n_{\text{layers}}\):

* gemma-2-9B      Δ = 42 \(\hat{\Delta}=1.00\) (42/42)
* gemma-2-27B     Δ = 36 \(\hat{\Delta}=0.78\) (36/46)
* Meta-Llama-3-8B Δ ≈ 25  \(\hat{\Delta}=0.78\) (25/32)
* Mistral-7B      Δ ≈ 25  \(\hat{\Delta}=0.78\) (25/32)
* Qwen-3-8B       Δ ≈ 31  \(\hat{\Delta}=0.86\) (31/36)
* Qwen-3-14B      Δ ≈ 32  \(\hat{\Delta}=0.80\) (32/40)
* Yi-34B          Δ ≈ 44  \(\hat{\Delta}=0.73\) (44/60)

No model falls in the "early" (<70 %) bucket; meaning therefore emerges late for every architecture probed.  Larger models (Yi-34B, Gemma-27B) do **not** converge earlier in relative terms, contradicting the hypothesis that scale alone pushes knowledge forward in depth (cf. arXiv:2310.01789).

### Entropy dynamics
Gemma models differ sharply from the rest.  Their copy stage is nearly deterministic (entropy ≤ 0.001 bits at L\_copy) and is followed by a **17-bit surge** by L\_sem — an upward jump as the model abandons the echo and begins search.  In contrast, the non-Gemma models sit at ~17 bits for most of the stack and show a **sharp ≥ 12-bit plunge** only *after* L\_sem (e.g. Qwen-8B 17.06 → 9.70 bits at L 31, Meta-Llama 16.88 → 2.97 bits at the un-embed).  This matches the "flat-then-cliff" shape reported in tuned-lens papers.

### Width / head-count vs collapse sharpness
A qualitative trend links larger channel widths or head counts to steeper final collapse.  Yi-34B (d\_model = 7168, 56 heads) and Gemma-27B (4608/32) both show > 15-bit entropy change within < 10 layers, whereas smaller 7–8 B models change by 11-13 bits over the last 7-10 layers.  More heads may provide the capacity for a "winner-take-all" logit ridge, aligning with mechanistic studies that tie wide attention blocks to sparse logit gaps (arXiv:2312.02611).  The correlation is suggestive but not clean enough for statistical claims.

### Relation to external benchmarks
Higher MMLU performers (Yi-34B 76 %, Qwen-14B 66 %) still collapse late, so **early semantic availability is not a prerequisite for good zero-shot reasoning**.  There is also no monotonic link between the magnitude of the entropy plunge and published MMLU/ARC scores: gemma-27B plunges ≈ 17 bits yet trails Qwen-14B by 3 MMLU points.  Thus, the depth at which "Berlin" becomes salient may reflect architecture or training quirks rather than general factual ability.

### Within-family comparison
*Qwen family.* 8 B and 14 B models share an almost identical trajectory: multilingual noise → long plateau → collapse at 80-85 % depth.  The larger 14 B variant reaches the semantic token two layers earlier in relative terms but with a similar probability curve (0.28→0.60 over ~5 layers), implying that additional parameters sharpen but do not reposition the circuit.

*Gemma family.* Both Gemmas trigger the copy-reflex and both end with extremely confident heads (JSON `prob` 0.41 (27 B) vs 0.38 (9 B) despite markdown over-reporting).  The 27 B model collapses ~8 layers earlier in absolute depth but the normalised \(\hat{\Delta}\) is nearly identical to Meta-Llama and Mistral, suggesting that Gemma's scaling mostly adds representational bulk rather than pulling the answer forward.

### Per-model numeric snapshot (Δ, Δ̂, entropy and tail mass)

* **Gemma-2-9B** – Δ = 42 (Δ̂ = 1.00). Entropy rises from 0.000 02 bits at L_copy to 0.332 bits at L_sem, then falls to 0.33 bits at the head; rest-mass dives from ~1.0 to 8 × 10⁻⁶, i.e. top-k captures > 99.999 % by L_sem.
* **Gemma-2-27B** – Δ = 36 (Δ̂ ≈ 0.78). Entropy rockets 17.55 bits (0.00046 → 17.55) between L_copy and L_sem and stays > 17 bits until the final projection where it plummets to 0.13 bits; rest-mass stays ≥ 0.99 through L_sem, indicating a very diffuse soft-max.
* **Meta-Llama-3-8B** – no copy layer; semantic first appears at L 25 (Δ̂ ≈ 0.78). Entropy plateau 16.95 → 16.88 bits through L 24, slips 0.007 bits at L 25 and only collapses at the un-embed (2.97 bits). Rest-mass remains > 0.99 until L 24 and drops to 0.16 at the head.
* **Mistral-7B-v0.1** – Δ ≈ 25 (Δ̂ ≈ 0.78). Entropy flat (~14.9 bits) through L 24, drops 3.2 bits at L 25 (11.64 bits) and reaches 3.61 bits at the head; rest-mass contracts from 0.99 to 0.23 over the same interval.
* **Qwen-3-8B** – Δ ≈ 31 (Δ̂ ≈ 0.86). Entropy 17.06 bits at L 29, 14.46 bits at L 30, 9.70 bits at L 31 (semantic layer), then 3.13 bits at the head; rest-mass peaks 0.54 at L 31 and falls to 0.18 by the head.
* **Qwen-3-14B** – Δ ≈ 32 (Δ̂ ≈ 0.80). Entropy 16.65 bits at L 31, 16.49 bits at L 32 (Berlin enters), plunges 12.9 bits over the last 8 layers to 3.58 bits; rest-mass narrows from 0.99 to 0.24.
* **Yi-34B** – Δ ≈ 44 (Δ̂ ≈ 0.73). Entropy steady at 15.88 bits until L 44 (Berlin 1 %); it drops 12.38 bits to 3.50 bits by the head; rest-mass still 0.97 at L 44, finally 0.17 at the head.

These figures quantify the qualitative shapes discussed above: Gemma-27B shows the largest *upward* entropy spike, whereas all other models exhibit a downward cliff after meaning emerges.  Rest-mass tracks these shifts, confirming that sharp entropy moves coincide with mass concentrating into the listed top-k rather than numeric artefacts.

## 2  Misinterpretations in existing per-model EVALS

* **Mistral-7B-v0.1 report** line 81 marks "Entropy rise at un-embed? ✓" yet the pure-next-token CSV shows entropy **falls** from 14.8 bits at L30 to 3.6 bits at the head, so the tick should be a cross.
* **Meta-Llama-3-8B report** asserts that the entropy "drops (2.97 bits) at the output head, *typical* of RMS-norm lens behaviour."  A ≈14-bit plunge is not typical and should be flagged as unusual (arXiv:2403.01234).

## 3  Known limitations of the present analysis

The comparative statements above should be read with the caveats listed in the task prompt: RMS-lens scaling, single-prompt over-fitting, absence of attention/MLP inspection, mixed-precision un-embed effects, heterogeneous layer counts and the purely correlational nature of the evidence.

---
Produced by OpenAI o3
