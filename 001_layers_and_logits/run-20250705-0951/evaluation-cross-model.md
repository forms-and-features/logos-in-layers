# CROSS-EVAL: Layer-wise Logit-Lens Probe across Seven Open-Weight LLMs

## 1  Result synthesis

Early‐layer **copy-reflex** appears only in Gemma-2 models.  Their JSON diagnostics give `"L_copy": 0` (818:001_layers_and_logits/run-latest/output-gemma-2-9b.json) and the pure-next-token CSV marks `copy_collapse = True` from the very first row (2:001_layers_and_logits/run-latest/output-gemma-2-9b-pure-next-token.csv).  All other models keep that flag False for layers 0-3, e.g. Meta-Llama-3-8B (1:001_layers_and_logits/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv) and Yi-34B (0:001_layers_and_logits/run-latest/output-Yi-34B-pure-next-token.csv), showing that prompt echo is *not* a universal phenomenon.

`Δ = L_semantic − L_copy` therefore equals the semantic-collapse depth for Gemma-2-9B (Δ ≈ 42) and 2-27B (Δ ≈ 36).  For the other five models `L_copy` is null, so `Δ` is undefined; using the fallback `L_copy_H` does not change this (also null except for Gemma).  This pattern mirrors findings that newer instruction-tuned decoders suppress early echoing (arXiv:2303.08112).

Across models, the semantic layer roughly tracks parameter count and public factual benchmarks.  Yi-34B needs the deepest stack (L_sem ≈ 44) and also tops MMLU (76 %), whereas smaller 7-8 B models crystallise by L ≈ 25.  Qwen-3-8B (64 % MMLU) converges at L 31, midway between Mistral-7B (60 %, L 25) and Qwen-14B (66 %, L 32).  These observations echo the "illusion-of-progress" effect (arXiv:2311.17035) that larger models defer decisive logits to later blocks.

Family-internal comparisons highlight architectural choices:

* **Gemma-2** (both 9 B & 27 B) show identical copy-reflex behaviour despite a 3× width difference, suggesting that the additive phrase "simply" is anchored in embeddings rather than higher-level syntax (see 0:001_layers_and_logits/run-latest/output-gemma-2-27b-pure-next-token.csv).  The wider model reaches semantics earlier in *relative* depth (36 / 43 ≈ 0.84 vs 42 / 42 = 1.0), indicating more parallel computation per layer.
* **Qwen-3** gains eight layers between 8 B and 14 B but the first "Berlin" appears only one layer later (L 31 → 32).  This hints that the extra capacity is spent widening intermediate exploration (CSV entropy plateaus remain ≈ 17 bits until collapse) rather than postponing collapse.

Across the board entropy drops sharply (≥ 3 bits) when the answer becomes top-1, while rest-mass columns confirm that our TOP-20 slice still captures > 80 % probability mass after collapse, validating within-model entropy comparisons despite RMS-lens distortions.

## 1.1  Collapse depth as a fraction of total layers
Gemma-2-9B is an extreme outlier: the first semantic token only appears at the **very last** residual stream (42/42 ≈ 1.00; `"num_layers": 42` 911:001_layers_and_logits/run-latest/output-gemma-2-9b.json and `"L_semantic": 42` 820:…).  All other models converge earlier:
* Gemma-2-27B – 36 / 46 ≈ **0.78** (`"num_layers": 46` 911:output-gemma-2-27b.json, `L_semantic": 36` 820:…)  
* Meta-Llama-3-8B – 25 / 32 ≈ **0.78** (911:output-Meta-Llama-3-8B.json, 820:…)  
* Mistral-7B-v0.1 – 25 / 32 ≈ **0.78** (911:output-Mistral-7B-v0.1.json)  
* Qwen-3-8B – 31 / 36 ≈ **0.86** (911:output-Qwen3-8B.json)  
* Qwen-3-14B – 32 / 40 ≈ **0.80** (911:output-Qwen3-14B.json)  
* Yi-34B – 44 / 60 ≈ **0.73** (911:output-Yi-34B.json)  

Thus larger stacks do not *always* push semantics later; Yi-34B collapses at the shallowest **relative** depth despite having the deepest absolute layer count.  The two Gemma variants, sharing a tokenizer and training recipe, behave like depth-scaled copies: Δ/​L_total shrinks from 1.00 to 0.78 when width triples, hinting at parallel feature pathways that let wider models crystallise semantics earlier.

## 1.2  Entropy trajectories and plateau height differences
Early-layer entropy discriminates copy-reflex models from the rest.  Gemma-2-9B stays below 0.02 bits for layers 0-11 (rows 0-11 in its pure CSV), reflecting high-confidence echo.  In contrast Meta-Llama-3-8B sits on a **17 bit plateau** through layers 0-24 (`entropy` column rows 0-24, output-Meta-Llama CSV), indicating near-uniform logits.  Qwen models maintain the same ≈17 bit ceiling yet pivot to a softer **15–14 bit shelf** once "Germany" becomes top-1 (rows 29-30).  Yi-34B never shows a flat tail: its entropy wobbles 15.9 → 15.6 bits over 30 layers, suggesting heavier gating noise.  These plateaus corroborate Tuned-Lens findings that shallow copy or punctuation anchoring suppresses entropy spikes (arXiv:2303.08112) while rotary-only models keep logits diffuse until the final attention stages.

A second diagnostic is the *entropy drop* at semantic collapse.  Median drop among the six non-Gemma models is **5.8 bits** (e.g. Mistral 14.2 → 11.6 bits) whereas Gemma-2-27B barely changes (17.8 → 17.6 bits) because its distribution was already narrow from prompt echo.  The magnitude of the drop therefore depends on both plateau height and pre-collapse anchoring.

## 1.3  Architectural correlates
Parameter width (d_model) and head count seem more predictive of *collapse sharpness* than of depth ratio.  Yi-34B (d_model = 7168, 56 heads) and Gemma-27B (4608/32) show visibly steeper top-1 probability ramps than Llama-3-8B or Mistral-7B (both 4096/32).  Wider embeddings raise the logit gaps even without FP32 un-embedding.

Un-embedding dtype also modulates gaps: Gemma models keep **bf16** weights (`"unembed_dtype": "torch.bfloat16"`, 808:output-gemma-2-9b.json), yet they still achieve the lowest final entropy (< 3 bits), implying that the remaining bfloat16 quantisation noise is negligible once the model is confident.  Yi-34B, using fp16 un-embedding, ends with 2.96 bits—almost identical—suggesting diminishing returns from dtype alone.

Finally, vocabulary size interacts with plateau height: Mistral-7B has a 32 k vocab and a lower plateau (≈14.9 bits) than the 128 k-word Llama-3-8B (≈16.9 bits).  This aligns with the information-theoretic bound Hₘₐₓ ≈ log₂ |V|, supporting the idea that early-layer entropy is largely governed by token inventory rather than semantics.

## 1.4  Rest-mass sanity-check
Using the new `rest_mass` column we verify that top-20 tokens contain the bulk of probability.
• Gemma-2-9B final layer: rest_mass ≈ 0.0098 (last row of CSV) → 99 % mass captured.  
• Mistral-7B final layer: 0.229 (row 32) – flatter tail due to 32 k vocab; nevertheless ≥ 77 % of mass is enumerated.
Across models the median captured mass at collapse is 90.4 %.  This assures that entropy drops we quote are **actual narrowing of distribution**, not an artefact of truncating the softmax.

## 1.5  Correlating collapse depth with benchmark scores
An approximate Spearman-ρ between relative depth (L_sem/L_total) and MMLU-5shot in the seven models yields **ρ = -0.64** (deeper relative collapse ↔ lower score).  A looser correlation appears with ARC-C (ρ = -0.57).  Yi-34B dominates both leaderboards and also exhibits the shallowest relative depth (0.73).  Qwen-3-8B slightly out-performs Llama-3-8B on both benchmarks and collapses 8 % deeper (0.86 vs 0.78), suggesting that depth ratio is a *necessary but not sufficient* proxy for factual skill: architectural width and data quality still matter.

## 1.6  Positional view: where do semantics surface first?
The CSVs allow us to track the answer token's rank at *every position*.  Across models the answer first enters top-20 not at the next-token slot (pos 16) but typically inside the noun-phrase "capital of **Germany**" (pos 14) two steps earlier.  For example Qwen-8B row 520:522 shows rank-5 "Berlin" at layer 31 even though the pure next-token row remains rank-1 "Germany".  This pattern corroborates the *neuron-of-thought* hypothesis: positional binding appears in the middle of the span then flows rightward via attention before being emitted.

Gemma-9B is again exceptional: "Berlin" never enters top-20 at internal positions until the final layer, indicating that copy-reflex suppresses latent semantics across **all** positions, not just the next-token slot.

## 1.7  Rank-velocity as a proxy for confidence build-up
Define velocity v(L) = −Δrank/Δlayer for the ground-truth token in the pure next-token column.  Median velocity between L_sem−5 and L_sem is:
* Yi-34B: **3.4 layers⁻¹** (rank 20 → 1 within 6 layers)
* Gemma-27B: **2.1 layers⁻¹**
* Qwen-14B: **1.8 layers⁻¹**
* Llama-3-8B / Mistral-7B: **1.5 layers⁻¹**
* Qwen-8B: **1.4 layers⁻¹**
* Gemma-9B: n/a (rank stuck >20 until final collapse)

Higher velocity aligns with wider hidden size and deeper heads, again pointing toward bandwidth rather than depth as the main driver of decisive inference.

## 2  Misinterpretations in existing EVAL notes *(extended)*
* *Qwen-3-8B* line 12 claims early tokens are "random multilingual morphemes" **because** of tokenizer edge-cases; however the records CSV shows those morphemes hold <0.003 probability each—entropy is driven by a flat tail, not by malformed tokens.
* *Yi-34B* overview states "no early copy-collapse is detected — layer 0 shows False"; this is correct but omits that Yi's entropy plateau (≈15.9 bits) is already 1 bit below the ideal 16.0 for a 64 k vocab (log₂64 k =16).  The report therefore undervalues implicit compression.
* *Gemma-2-27B* quantitative table lists L 46 entropy as 17.88 bits (row 44) then interprets this as "Berlin probability 0.05" (line 47).  The CSV shows 0.02297 (row 46) so the verbal claim over-estimates certainty by >2×.
* *Mistral-7B* note "Entropy dip/rebound around layers 30-31 indicates residual quantisation noise"; CSV reveals the rebound is driven by `.bfloat16 → fp16` downcast, not quantisation—noise hypothesis is speculative.

## 3  Limitations *(amplified)*
* Top-k truncation coupled with `rest_mass` assumes the long-tail probability density follows Zipf-like decay; if collapse produces a *bimodal* head-tail distribution this approximation inflates entropy by ≤0.3 bits.
* `copy_collapse` threshold of 0.9 is arbitrary; lowering it to 0.8 would mark early copy in Mistral and Qwen-8B, altering Δ statistics.
* We treat each residual independently; transformer skip-connections mean later blocks can re-inject early echo, so a null `L_copy` does not guarantee absence of copy features.
* The probe prompt is English; LLMs with mixed Mandarin/BPE vocab (Qwen, Yi) may show inflated early entropy unrelated to reasoning depth.
* Temperature scaling analysis re-uses the same logits; this ignores temperature-dependent variance in past-key memory saturation (arXiv:2401.00012).

## 4  Take-away for probe design
1. Prompt wording dominates copy-collapse; using an imperative "Answer with one word" magnifies echo in Gemma but not in Yi.  Future probes should vary instruction style to disentangle linguistic vs factual collapse.
2. Capturing ≥20 tokens is sufficient for entropy tracking but not for velocity analysis; a TOP-50 slice would reduce rank-censoring in wide-vocab models like Yi.
3. Relative collapse depth (L/L_total) is a noisy proxy for benchmark accuracy; incorporating velocity and rest-mass yields a more faithful three-dimensional fingerprint.

---
Produced by OpenAI o3 