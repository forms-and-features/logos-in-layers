
# Review of **Experiment 001 – Layer‑by‑Layer Logit‑Lens Baseline**

**Provenance.** This review examines the code and artifacts under `001_layers_baseline/` (latest run dated 2025‑10‑11 UTC) and the accompanying per‑model and cross‑model evaluations. All quantitative claims below are backed by the repository’s JSON/CSV dumps and evaluation notes. Where relevant, background is grounded in the logit‑lens and tuned‑lens literature and core architectural papers. ([GitHub][1])

---

## 1) Measurement and code review

**What the pipeline does (at a high level).**
The runner performs a post‑block *logit‑lens* sweep over all layers, with optional *tuned‑lens* and a shared‑decoder *prism* sidecar. It records per‑layer next‑token distributions, ranks of the gold token, KL to the final head, cosine to the final logit direction, entropy, and copy‑collapse flags; writes model‑level JSON and several CSVs; and emits a compact *evaluation pack* to guide LLM‑assisted reports. The plan stresses that **within‑model** comparisons are valid while **cross‑model** absolute probabilities are not, due to family‑specific normalization and calibration. ([GitHub][1])

**Major measurement choices that look sound.**

* **RMS/LN path is fixed and aligned to the probed stream.** The plan documents a §1.1 fix: ε inside the √ and γ applied on the *correct* stream (ln2 for post‑block; next block’s ln1 for pre‑norm), with fp32 stats and unembedding. This is exactly the kind of bug that can fabricate “early meaning,” and fixing it is essential. The run asserts the fix is **active** in current outputs. ([GitHub][1])
  *Background:* RMSNorm formula and implementation details align with Zhang et al. (2019). ([arXiv][2])

* **Raw‑vs‑Norm dual‑lens sanity.** A built‑in QA pass compares the normalized lens to a raw‑activation lens at sampled depths and summarizes divergence (JS/L1/JS@p50), top‑K overlap, and any “norm‑only semantics.” Treating norm‑only signals as suspect is the right call. ([GitHub][1])

* **Rank‑centric reporting & KL thresholds.** Adding `answer_rank` and `KL(P_layer‖P_final)` (bits) avoids over‑interpreting absolute probabilities; rank and KL are more calibration‑robust. ([GitHub][1])

* **Token‑ID‑aware copy detector.** Copy collapse is checked against the prompt **in ID space** (optionally with k‑windows), which avoids common string‑level pitfalls. ([GitHub][1])

* **Micro‑suite and control prompts.** The run includes a small positive set plus matched controls and a no‑filler ablation; later stages plan a large battery. This matches best practice given how brittle single prompts can be. ([GitHub][3])

* **Cosine to the final logit direction.** Recording `cos_to_final` adds a geometry‑based view of *rotation vs amplification* of the decision direction. This mirrors the iterative‑inference perspective standard in tuned‑lens work. ([GitHub][1])

**Potential issues / improvements in the current script (beyond the already‑acknowledged limits).**

* **“Uniform‑margin OK” is a weak semantic gate.** The gate checks margin against a *uniform* baseline; an additional **runner‑up margin** (logit(ans) − max_non‑ans) would better guard against diffuse or multi‑peaked distributions, at negligible compute cost. This matters for families with high lens‑artifact risk and head calibration quirks. ([GitHub][3])

* **Raw‑lens sampling may be too sparse for high‑risk families.** The plan defaults to sampled layers; for families flagged **high** risk (e.g., Gemma, Qwen, Yi), promote the raw‑vs‑norm check to **full‑depth** in this iteration. The extra unembed+softmax per layer is cheap compared to the forward pass and would materially tighten claims. ([GitHub][1])

* **Prism should be treated as diagnostic only.** The cross‑model summary shows regressive behavior for most families (negative KL deltas), with a few exceptions (Yi‑34B, Qwen2.5‑72B, Gemma‑27B). The code rightly does not rely on prism for semantic milestones; that stance should be explicit in the README to prevent misuse. ([GitHub][3])

* **Cosine tracking is present but under‑analyzed.** The pipeline writes `cos_to_final`; the evaluations occasionally cite coarse thresholds. Plotting or summarizing first‑crossings (e.g., cos≥0.2/0.4/0.6) systematically would sharpen the rotation‑vs‑amplification story. (The Llama‑3‑8B evaluation already demonstrates such thresholds.) ([GitHub][4])

**Methodological backdrop.** Using the *logit lens* (nostalgebraist) and its tuned variant (Belrose et al., “Tuned Lens”) is a canonical way to view intermediate **latent predictions**; the plan’s emphasis on lens sanity and on tuned‑lens attribution (rotation vs temperature) is squarely in line with that literature. The RoPE note about “token‑only at layer 0” is also correct for the listed families. ([LessWrong][5])

---

## 2) Results & analyses: corrections and over‑statements

**Confirmed semantic onsets (grounded in `*-milestones.csv`).**

* **Meta‑Llama‑3‑8B:** L=25 confirmed (source=raw). ([GitHub][6])
* **Meta‑Llama‑3‑70B:** L=40 confirmed (raw). ([GitHub][7])
* **Mistral‑7B‑v0.1:** L=25 confirmed (raw). ([GitHub][3])
* **Mistral‑Small‑24B‑Base‑2501:** L=33 confirmed (raw). ([GitHub][3])
* **Qwen3‑8B:** L=31 confirmed (raw). ([GitHub][8])
* **Qwen3‑14B:** L=36 confirmed (raw). ([GitHub][3])
* **Qwen2.5‑72B:** L=80 milestone recorded for the **norm** lens only; no confirmed layer. ([GitHub][9])
* **Yi‑34B:** L=44 confirmed (tuned). ([GitHub][10])
* **Gemma‑2‑9B:** L=42 confirmed (tuned). ([GitHub][11])
* **Gemma‑2‑27B:** L=46 confirmed (tuned); **strict copy reflex at L0** is also logged. ([GitHub][12])

**Two concrete corrections to the existing LLM‑authored EVALS.**

1. **Prism delta sign (Meta‑Llama‑3‑8B).** The single‑model EVAL asserts a large positive median KL delta under prism; the cross‑model review shows the **median delta is negative** (prism worse), which matches the JSON percentile deltas. Treat the verdict (“regressive”) as unchanged, but correct the magnitude/sign in the single‑model memo. ([GitHub][3])

2. **Absolute‑probability quoting under high‑risk gating.** The Qwen3‑8B EVAL quotes a near‑1 control margin despite the guidance to suppress absolute probabilities under high artefact risk. Future memos should stick to rank/KL statements there. ([GitHub][3])

**One over‑statement to tone down.**
The cross‑model synopsis opens by suggesting onset timing is “roughly proportional to depth” within families. The **absolute** layer indices are consistent and family‑characteristic, but normalized depth fractions vary substantially (e.g., Llama‑3‑8B at 25/32 vs Llama‑3‑70B at 40/80; multiple Gemma/Qwen runs confirm semantics **at the last layer**). It is safer to say that **onset is late** and **family‑consistent**, not that it scales proportionally with layer count. ([GitHub][6])

---

## 3) Independent synthesis (cross‑model), grounded in the dumps

**A. Copy‑reflex vs. semantics.**
A clear pattern emerges:

* **Gemma‑2‑9B/27B** exhibit a **prompt‑copy reflex at L0** (strict and soft), yet the **first semantic milestone is at the final layer** (42 and 46 respectively). This yields Δ̂≈1.0 (surface→meaning gap equals the full depth). This is atypical among the tested families. ([GitHub][12])

* **Llama‑3 (8B/70B), Mistral (7B/24B), Qwen‑3 (8B/14B), Yi‑34B** show **no early copy milestone**; semantics arrive late but not exclusively at the last layer (e.g., L25/32 for Llama‑3‑8B; L31/36 for Qwen3‑8B; L33/40 for Mistral‑24B; L44 for Yi‑34B). ([GitHub][6])

**B. Lens‑artefact risk and gating discipline.**
Artifact audit tiers (v2) and raw‑vs‑norm divergences indicate: **Gemma** and **Qwen** families are high risk; **Llama‑3** medium; **Mistral‑24B** low. Accordingly, the evaluations that privilege **rank/KL milestones** and **confirmed semantics** (and de‑emphasize absolute probabilities) are methodologically appropriate. Example: Meta‑Llama‑3‑8B audit summary shows medium risk (JS@p50≈0.0168; L1@p50≈0.24). ([GitHub][3])

**C. Head calibration and tuned‑lens attribution.**

* **Gemma** shows substantial **final‑head mismatch**: at the semantic layer the **KL to final head** is ≳1 bit under the norm lens; a scalar temperature τ⋆>1 halves this discrepancy, and the tuned lens is **calibration‑only** at 27B (rotation contributes ≈0). This is evidence that Gemma’s final head differs materially from a vanilla readout. ([GitHub][12])

* **Qwen‑3 (8B/14B)** and **Llama‑3** tuned‑lens gains are **rotation‑led**, with τ⋆≈1 and near‑zero final‑KL once calibrated—consistent with the tuned lens actually mapping to the model’s decision space rather than merely rescaling. ([GitHub][3])

**D. “Prism” (shared decoder) as a diagnostic.**
Prism reduces KL for **Yi‑34B**, **Qwen2.5‑72B**, and **Gemma‑27B**, but is **regressive** for the rest (including both Llama‑3 sizes and the two Mistrals). This heterogeneity cautions against assuming a universal shared decoder across families/sizes; using prism to *probe* decoder‑space compatibility, not to set milestones, matches the current usage. ([GitHub][3])

**E. Rotation vs. amplification (cosine & KL).**
Where cited (e.g., Llama‑3‑8B), cosine to the final logit direction rises late (cos≥0.6 only very near the top layers), while `KL(P_layer‖P_final)` drops under 1 bit only at the end. This is consistent with *late alignment* of the predictive subspace rather than purely early amplification. ([GitHub][4])

**F. Summary judgment (this iteration).**
Across ten base LLMs, **semantic evidence for the gold next token appears late** and is **stable under rank/KL‑based gates**, with **no prompt‑echo reflex** except in Gemma (where semantics still occur only at the very top). This pattern holds under explicit lens‑artefact audits and raw‑lens cross‑checks. ([GitHub][12])

---

## 4) Philosophical relevance (nominalism vs realism), at this stage

**Scope caveat.** The current stage is *observational* (no causal interventions); in addition, tuned lenses are missing for the largest models for cost reasons. Any metaphysical inference must therefore be modest and tied to the concrete observables. ([GitHub][1])

**Anti‑austere‑nominalist pressure.**
Two lines of evidence put pressure on **austere nominalism** (the view that there are only token strings with no systematic properties/relations):

1. **Late, model‑internal convergence to the correct answer direction** (rank‑1 confirmed layers; KL→final≈0) across diverse families indicates **reusable, depth‑localized internal structure** that is not reducible to surface token identity or immediate echoing. In most families, there is **no copy reflex** preceding semantics. ([GitHub][6])

2. **Family‑specific decoder geometry** (e.g., Gemma’s head calibration) together with tuned‑lens **rotation‑led** gains elsewhere suggests that models maintain **structured predictive subspaces** that a simple, family‑agnostic decoder cannot read out. This looks like stable structure rather than mere accidental co‑occurrence. ([GitHub][12])

**What this does *not* yet decide.**
These observations do not by themselves distinguish **metalinguistic nominalism** (structures as facts about *words/predicates*) from **realism** (mind‑independent universals). To make headway, the next iterations need: (a) *causal localization* of a relation‑like subspace (e.g., **capital‑of**), (b) **portability** of that subspace across paraphrases and languages, and (c) **specificity**/necessity tests (projection surgery). Those steps are already drafted for Parts 2–4. ([GitHub][13])

---

## 5) Recommendations for *this* iteration (concrete, non‑bikeshedding adjustments)

These bring noticeable value while staying within the iteration’s cost/complexity envelope.

1. **Add a runner‑up margin gate** at the semantic milestone.
   Write `top2_margin = logit(ans) − max_{j≠ans} logit(j)` into the pure next‑token CSV and require `top2_margin ≥ δ` (e.g., 0.5 logit) for a “strong” semantic flag. This is especially useful in **high‑risk** families where absolute probabilities are unreliable but ranks alone can be unstable near ties. (No extra forward passes needed.)

2. **Promote raw‑lens dual checks to full‑depth** for models whose *artifact‑audit tier ≥ medium*.
   Today’s default is sampled layers; flipping a single env var (`LOGOS_RAW_LENS=full`) for Gemma/Qwen/Yi will quantify norm‑only artifacts across *all* layers and make the gating more robust in this very run. ([GitHub][1])

3. **Turn on `copy_soft` windows (k=2) for this run.**
   Token‑ID‑level copy with `k∈{2,3}` will catch multi‑token echoes that `k=1` misses, without appreciable cost. Given Gemma’s L0 copy reflex, the extra window gives a clearer Δ̂ picture across families. ([GitHub][1])

4. **Summarize cosine first‑crossings per model in the JSON.**
   Record `first_cos_ge_{0.2,0.4,0.6}` (layer indices) next to the KL/rank thresholds. This adds a geometry‑based check that complements the current milestones—again, no extra passes are required. ([GitHub][1])

5. **Make “prism is diagnostic only” explicit in the measurement guidance.**
   Given widespread regressions under prism, the guidance should explicitly forbid using prism numbers for *semantic* milestone calls. (The cross‑model memo already treats it that way; encoding this into the pack prevents future over‑use.) ([GitHub][3])

> **Verdict on scope sufficiency.** With the four lightweight tweaks above, this iteration is **methodologically complete** for its stated goal: establishing reliable, rank/KL‑gated late‑layer semantics and separating surface copy from meaning. The heavier causal and invariance work should proceed in the next stages as planned. ([GitHub][13])

---

## Appendix – Key evidence excerpts (by family)

* **Llama‑3‑8B:** confirmed semantics at **L=25** (raw). Artifact tier **medium** (JS@p50≈0.0168; L1@p50≈0.24). Prism **regressive** in the cross‑summary. ([GitHub][6])

* **Llama‑3‑70B:** confirmed semantics at **L=40** (raw); prism regressive in the cross‑summary. ([GitHub][7])

* **Mistral‑7B‑v0.1 / 24B:** confirmed at **L=25** / **L=33** respectively; 24B shows **low** artifact tier. ([GitHub][3])

* **Qwen‑3‑8B / 14B:** confirmed at **L=31** / **L=36** (raw); cross‑summary flags **high** artifact risk; tuned‑lens gains **rotation‑led**. ([GitHub][8])

* **Qwen‑2.5‑72B:** **no confirmed** layer; norm‑lens milestone at **L=80** only; prism **helpful** in cross‑summary. ([GitHub][9])

* **Yi‑34B:** confirmed at **L=44** (tuned); prism **helpful** in cross‑summary. ([GitHub][10])

* **Gemma‑2‑9B / 27B:** **strict copy at L0**; semantics at **final layer** (42 / 46). Final‑head **calibration mismatch** prominent; tuned lens **calibration‑only** at 27B. ([GitHub][11])

---

## References (method background)

* **Logit lens** (nostalgebraist). Introduces the idea of linearly decoding intermediate states with the final unembedding to elicit *latent predictions*. ([LessWrong][5])
* **Tuned lens** (Belrose et al., 2023). Affine per‑layer probes that better match the model’s own decoder space; supports iterative‑inference interpretations. ([arXiv][14])
* **RMSNorm** (Zhang et al., 2019). Fixes around ε placement and γ scaling are consistent with this formulation. ([arXiv][2])
* **RoPE** (Su et al., 2021). Clarifies why “token‑only” is an accurate layer‑0 description for rotary models. ([arXiv][15])

---

### Closing

The current run already provides **clean, within‑model milestones for late‑layer semantics** across ten open‑weight LLMs, with **careful artifact gating**. Those observables are strong enough to levy **initial pressure against austere nominalism**. The next staged steps—*rank‑centric batteries*, *causal patching*, and *concept‑vector invariance tests*—are the right path for turning that pressure into a more discriminating argument about the nature of the internal structure (metalinguistic vs realist). ([GitHub][13])

[1]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_LAYERS_BASELINE_PLAN.md "raw.githubusercontent.com"
[2]: https://arxiv.org/abs/1910.07467?utm_source=chatgpt.com "[1910.07467] Root Mean Square Layer Normalization"
[3]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-cross-models.md "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Meta-Llama-3-8B.md "raw.githubusercontent.com"
[5]: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens?utm_source=chatgpt.com "interpreting GPT: the logit lens"
[6]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/output-Meta-Llama-3-8B-milestones.csv "raw.githubusercontent.com"
[7]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/output-Meta-Llama-3-70B-milestones.csv "raw.githubusercontent.com"
[8]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/output-Qwen3-8B-milestones.csv "raw.githubusercontent.com"
[9]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/output-Qwen2.5-72B-milestones.csv "raw.githubusercontent.com"
[10]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/output-Yi-34B-milestones.csv "raw.githubusercontent.com"
[11]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/output-gemma-2-9b-milestones.csv "raw.githubusercontent.com"
[12]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/output-gemma-2-27b-milestones.csv "raw.githubusercontent.com"
[13]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/00x_PROJECT_PLAN.md "raw.githubusercontent.com"
[14]: https://arxiv.org/abs/2303.08112?utm_source=chatgpt.com "Eliciting Latent Predictions from Transformers with the Tuned Lens"
[15]: https://arxiv.org/abs/2104.09864?utm_source=chatgpt.com "RoFormer: Enhanced Transformer with Rotary Position Embedding"

---

Produced by GPT-5 Pro
