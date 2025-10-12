# Technical review of “logos‑in‑layers”: layer‑wise probes over open‑weight LLMs and their philosophical use

## Executive summary

The project implements a careful, modern variant of logit‑lens analysis across ten open‑weight base models, with multiple guardrails that reduce classic lens artefacts (dual raw‑vs‑norm decoding, a *tuned lens*, a *shared‑decoder* “Prism”, temperature controls, copy‑collapse audits, and several surface‑vs‑meaning diagnostics). The code path appears numerically sane and reproducible; the outputs are consistent across families and mostly interpreted with appropriate caution. The largest empirical effects are (a) a **clear surface‑form “copy reflex” at L0 in Gemma‑2** and **no such reflex in Llama/Mistral/Qwen/Yi**, (b) **late semantic resolution in Qwen** (≈0.86–1.0 of depth), **mid‑stack onset in Llama/Mistral/Yi**, and **terminal‑layer semantics in Gemma‑2**, and (c) **high raw‑vs‑norm discrepancy** (lens‑artefact risk) in Qwen and Gemma families. These patterns agree with current interpretability literature that treats layer‑wise decoding as a *proxy* for iterative inference, not a ground‑truth read‑out. ([GitHub][1])

As a first iteration, the findings are useful **evidence about representational organization** rather than decisive data for metaphysical theses. Still, they can *inform* nominalism‑vs‑realism debates by operationalizing “surface names” vs “abstract role” signals: echo‑mass/top‑K coverage decay, geometric crossovers, and confirmed semantics provide an empirical handle on when the state ceases to be dominated by string tokens and aligns to an answer‑direction. With a few small additions (below), this iteration looks complete and well‑scoped.

---

## 1) Method & code review

**What is done well**

* **Determinism, health, and provenance.** Runs are deterministic (seeded) with numeric‑health checks; unembedding is promoted to fp32; LN bias is absent on RMS models; and the probe records normalizer provenance per layer. This is surfaced in the per‑model JSON and model‑level sanity sections in the evaluations. ([GitHub][2])

* **Norm‑lens with real architecture alignment.** The run path decodes post‑block residuals using a consistent *RMS‑norm lens* strategy and *also* collects the **pre‑norm raw residual** to audit lens‑induced effects (windowed and full sweeps). The plan documents the windowed (§1.19) and **full dual‑lens sweep (§1.24)** now *on by default*, with per‑layer raw vs norm probabilities, KL, and a “norm‑only semantics” flag. This is the right antidote to “early semantics from normalization.” ([GitHub][3])

* **Tuned lens integration.** A *tuned lens* translator (affine probe in‑d) is trained offline and loaded at run‑time, with per‑layer temperatures, last‑layer identity, skip‑layers sanity, and **rotation‑vs‑temperature attribution** so improvements are not mis‑credited to mere calibration (§§1.12, 1.16–1.17, 1.26). This matches the method introduced in *Tuned Lens* (Belrose et al., 2023). ([GitHub][3])

* **Shared‑decoder “Prism”.** A family‑agnostic, orthogonally‑constrained shared decoder (whiten + rotation to the unembedding) is fit and run as a sidecar, enabling cross‑layer comparability and catching per‑layer‑head overfitting (§1.10). Treating Prism as **diagnostic‑only** (not a head replacement) is appropriate. ([GitHub][3])

* **Surface‑vs‑meaning diagnostics.** Echo‑mass vs answer‑mass (§1.13), geometric crossover via decoder cosines (§1.14), and Top‑K prompt‑coverage decay (§1.15) are excellent, low‑cost checks that capture surface pull without relying on strict contiguous echo. ([GitHub][3])

* **Copy detection done conservatively.** The project keeps **strict copy** (k=1, τ=0.95), adds **soft/windowed** (k∈{1,2,3}, τ=0.33), and performs a **threshold sweep** (τ∈{0.70,0.80,0.90,0.95}) plus cross‑validation under raw vs norm (§1.23). This avoids over‑reading chance echoes from normalization or temperature. ([GitHub][3])

**Potential issues or places to tighten**

* **Rank‑only semantics can still be fragile.** The milestone CSVs show cases where **answer_rank==1 occurs with very low probability**, e.g., Meta‑Llama‑3‑70B at L=40 has rank‑1 but p≈2.4e‑5 with near‑uniform entropy (~16.94 bits). The project already sets *prefer_ranks* and advises suppressing absolute probabilities, but adding a lightweight **“above‑uniform margin”** gate (e.g., require (p_{\text{answer}} \geq 1/|V| + \delta)) would harden the definition of semantic onset without changing the reporting. ([GitHub][4])

* **Single concept, multiple paraphrases.** The run smartly varies prompt phrasings and includes negative controls, but it is still a **single conceptual relation** (“Germany → Berlin”). A *very* small battery of additional, equally elementary facts (e.g., “France→Paris”, “Italy→Rome”, 3–5 items) would materially improve robustness **within this iteration** without drifting scope. (This addresses tokenization quirks and idiosyncratic memorization while keeping the experiment single‑step.)

* **Family‑level head calibration.** As correctly noted in the analyses, **Gemma‑2** exhibits last‑layer KL to the model head and large temperature needed for model calibration; therefore absolute finals are incomparable to families with τ⋆≈1.0. The project already emits a *measurement_guidance* block to caution evaluators; continuing to *enforce* those hints in write‑ups (e.g., avoid quoting final‑layer probabilities for Gemma) will prevent accidental over‑reach. ([GitHub][5])

* **Repeatability skipped.** Determinism is on, so the repeatability module is intentionally skipped; this is fine for a baseline but leaves no within‑run variance quantification. A cheap “dummy nondeterministic pass” on a tiny shard (e.g., 32 positions) would give a flip‑rate number with negligible cost—useful when comparing families with high raw‑vs‑norm divergence. ([GitHub][5])

*(Background literature: tuned lens refines logit‑lens probing by training per‑layer translators; RMSNorm modifies LayerNorm by removing re‑centering; superposition/phase‑change phenomena warn against over‑interpreting single‑neuron or single‑view read‑outs; induction heads and other circuits can produce sharp phase‑like changes in behavior. These motivate the project’s multiple corroborating diagnostics.)* ([arXiv][6])

---

## 2) Results & existing analyses: what looks right, what’s overstated, what’s missing

**Core timing (semantic onset) is consistent and well‑supported.** Confirmed semantics (rank‑1 corroborated by raw or tuned within a small window) cluster as follows:

* **Llama family.** Meta‑Llama‑3‑8B: **L=25/32** (confirmed by raw). Meta‑Llama‑3‑70B: **L=40/80** (raw). These are clean mid‑stack onsets. ([GitHub][1])
* **Mistral family.** 7B: **L=25/32** (raw). 24B: **L=33/40** (raw). ([GitHub][5])
* **Qwen family.** 3‑8B: **L=31/36** (raw). 3‑14B: **L=36/40** (raw). **2.5‑72B: L=80/80** with *no confirmation*. Late/terminal behavior is the salient family trait. ([GitHub][7])
* **Yi‑34B.** **L=44/60** (confirmed by tuned). Mid‑late onset. ([GitHub][5])
* **Gemma‑2.** 9B: **L=42/42** (confirmed by tuned). 27B: **L=46/46** (tuned; calibration‑only lens). Semantics at the terminal layer in both sizes. ([GitHub][8])

**Copy‑reflex vs semantics.** Only **Gemma‑2** hits **strict copy at L0**; the other families do not register early copy under the project’s strict or soft detectors in the first few layers. This yields a maximal **surface→meaning gap** in Gemma (Δ̂≈1.0 of depth). This is documented in the milestone CSVs and audits and was correctly called out in the cross‑model evaluation. ([GitHub][8])

**Lens‑artefact risk.** The *artifact‑audit* (full dual‑lens sweep) shows **high raw‑vs‑norm divergence** in **Qwen** and **Gemma** families (elevated JS/L1 and high pct of layers with KL≥1.0), **medium** in Llama‑3, and **low** in Mistral‑24B. This validates the project’s policy to **prefer ranks** and to **suppress absolute probabilities** in high‑risk settings. ([GitHub][9])

**Tuned lens attribution.** The project’s rotation‑vs‑temperature breakdown shows **rotation dominates** the KL reduction in Llama/Qwen/Yi, whereas **Gemma‑2‑27B** is **calibration‑only** (temperature accounts for the gains, with little or negative rotation contribution). This is the correct way to interpret tuned‑lens improvements. ([GitHub][5])

**Prism (shared decoder).** As expected for a diagnostic, Prism is **helpful on some families/sizes** (e.g., Gemma‑2‑27B) and **regressive on others** (e.g., Llama‑3‑8B), with little impact on rank milestones. The evaluation text mostly handles this properly, though one per‑model write‑up labels Qwen‑3‑14B Prism “Neutral” despite a small negative KL delta in the JSON; calling that “slightly regressive” would be more precise. ([GitHub][5])

**Two spots to adjust wording in the existing EVALs**

* A per‑model EVAL cites **final‑layer p(answer)** for Gemma in spite of explicit *measurement_guidance* to suppress absolute probabilities there (family‑level head calibration issue). This should be avoided or restated using ranks/KL only. ([GitHub][5])
* The cross‑model EVAL’s use of “late semantics” for Qwen is correct, but it should emphasize that **Qwen‑2.5‑72B lacks confirmation** at L=80, so onset is **norm‑only**—a caution already present in the JSON and worth foregrounding. ([GitHub][5])

---

## 3) Independent deep‑dive (cross‑family synthesis)

**Normalized depth of semantic onset (approximate).** Llama‑3‑8B resolves near **0.78** (25/32), Llama‑3‑70B near **0.50** (40/80), Mistral‑7B near **0.78** (25/32), Mistral‑24B near **0.83** (33/40), Qwen‑3‑8B near **0.86** (31/36), Qwen‑3‑14B **0.90** (36/40), Qwen‑2.5‑72B **1.0** (80/80, unconfirmed), Yi‑34B **0.73** (44/60), Gemma‑2‑9B **1.0** (42/42), Gemma‑2‑27B **1.0** (46/46). The family picture that emerges is **mid‑stack in Llama/Mistral/Yi** vs **late/terminal in Qwen/Gemma**. ([GitHub][1])

**Surface→meaning transition.** In Gemma‑2, strict copy at L0 plus confirmed semantics only at the final layer yields the project’s maximal Δ̂, while in Llama/Mistral **no copy flags** trip early and semantics appears once KL to final drops and cosine thresholds are crossed mid‑stack. That qualitative difference is robust to prompt variants and matches the surface‑diagnostic metrics (echo‑mass, top‑K prompt mass) the plan added. ([GitHub][8])

**Raw‑vs‑norm discrepancies cluster by family.** Qwen and Gemma show **large** per‑layer JS/L1 and many layers with KL≥1.0 between raw and norm distributions, indicating that **normalization can substantially reshape decoded probabilities**—a classical failure mode of naive logit‑lens. The project’s full dual‑lens sweep mitigates over‑reading such layers by (a) down‑weighting absolute probability claims and (b) using **confirmed semantics** (norm corroborated by raw or tuned). This is exactly the practice recommended by the literature when using layer‑wise decoders. ([GitHub][9])

**Entropy drift.** Entropy remains high and near‑uniform until late layers in Llama/Mistral/Qwen2.5/Yi and only collapses near or at the final head; Gemma behaves differently due to head calibration. Reporting *entropy gaps* relative to the model head (“teacher”) is a good stability measure. ([GitHub][5])

**Context from recent interpretability work.** These observations align with **iterative‑inference** views of transformers where intermediate states get progressively “closer” to the final distribution (tuned‑lens framing), with **phase‑like changes** known from **induction‑head** formation and **superposition** phenomena suggesting that single‑view probes can be brittle without corroboration. The project’s design choices (dual lenses, confirmation windows, rotation‑vs‑temperature attribution) are consistent with those cautions. ([arXiv][6])

---

## 4) Usefulness for the realism vs nominalism debate

Philosophically, **nominalism** about universals denies mind‑independent universals, while **realism** affirms them (with immanent vs transcendent variants); trope theories are a common nominalist option. The project’s diagnostics offer a way to **operationalize** this abstract debate for LLMs:

* **Surface‑form dominance** (prompt echo‑mass, soft/strict copy flags, prompt Top‑K mass) is a natural proxy for **nominalist‑like “name matching.”** If a model “solves” the task by latching onto context strings, not meanings, these metrics will fire early. ([Stanford Encyclopedia of Philosophy][10])
* **Answer‑direction alignment and early, paraphrase‑robust rank‑1** (confirmed by raw or tuned) are better proxies for **realist‑like “type” structure**—representations that cut across verbal forms and track an abstract role (e.g., *capital‑of(·)*). The project’s **geometric crossover** and **confirmed semantics** are good first‑pass indicators of this shift.

On the present stimulus (a single fact with variants), the empirical signal is **mixed**:

* **Gemma‑2** looks **nominalist‑leaning** at shallow depth (L0 copy‑reflex) with **no abstraction until the last layer** (Δ̂≈1.0). This says more about **how Gemma exposes internal states to a lens** than about the ontology of universals—but **on these probes** the internal state behaves like a *name matcher* for most of the stack. ([GitHub][8])
* **Llama/Mistral/Yi** look **realist‑leaning** insofar as **surface cues decay** and **semantic rank‑1 appears mid‑stack** in a way that is corroborated by raw/tuned and by KL drops/cosine milestones. Within this operational frame, the model’s internal state aligns to a *role* not just the prompt string. ([GitHub][1])
* **Qwen** is **late/terminal**: internal states do not confidently represent the abstract role until very late; whether this is “more nominalist” or simply “conservative internalization” is under‑determined by this single fact.

**Bottom line.** The current iteration **does not adjudicate metaphysics**; it **does** give a *measurable, falsifiable* yardstick for when a model’s internal state shifts from **surface‑word pull** toward **answer‑role alignment**. That is a useful empirical bridge from NLP measurements to metaphysical vocabulary. For a stronger philosophical claim, future iterations need **multi‑concept suites** and **invariance checks across paraphrase, language, and negative evidence**.

*(Background refs for the debate.)* ([Stanford Encyclopedia of Philosophy][11])

---

## 5) Plan review and targeted adjustments (only where value is non‑negligible)

The plan changes in §§1.11–1.28 are already implemented and materially improved this run (Prism sidecars, soft copy, echo/geometry/Top‑K diagnostics, full raw‑vs‑norm sweep, confirmed‑semantics, norm‑temperature controls, rotation‑vs‑temperature attribution, evaluator guidance). No bikeshedding is needed. A few **small, immediately valuable** additions are recommended:

1. **Gate rank‑1 by an above‑uniform margin (no extra forward).**
   Keep the *reported* milestone as is, but compute an auxiliary flag `semantics_margin_ok = p_answer ≥ 1/|V| + δ` (e.g., δ=0.002, tune by vocab) and surface it in JSON. This prevents “rank‑only semantics” with near‑uniform distributions (observed in large‑vocab models) from being over‑interpreted. Minimal code, no extra cost. ([GitHub][4])

2. **Add a 3–5 item micro‑suite of equally trivial facts** (same format).
   Examples: *France→Paris*, *Italy→Rome*, *Japan→Tokyo*, *Canada→Ottawa*. This preserves single‑step probing but **averages out tokenizer idiosyncrasies** and allows reporting **within‑model variance** of milestones and copy indices. It also enables a **concept‑invariance sanity** (do L_sem/echo‑mass/top‑K decay behave similarly across trivially isomorphic facts?). No training needed; adds minutes to runtime.

3. **Publish the *surface→meaning* crossover depths in JSON** (already computed ad‑hoc in CSVs).
   The plan already adds echo/geometry/Top‑K metrics; ensure the **first‑crossing depths** for each are mirrored in `summary` so evaluators don’t recompute them from CSVs. The plan’s §1.20/§1.18 sketch covers this; verify consistency across models. ([GitHub][3])

4. **Make the evaluator honor *measurement_guidance* mechanically.**
   Add a trivial check in evaluation prompts: if `suppress_abs_probs=true` or `prefer_ranks=true`, **hide** final‑layer probabilities and **default** to confirmed‑semantics depths. This prevents slips like the Gemma probability read‑out in a high‑risk family. ([GitHub][5])

Everything else in the current plan looks necessary and proportionate; this iteration can be considered **complete** once the above small patches land.

---

## 6) Known limitations and alignment with current literature

The project’s own scoping notes (RMS‑lens comparability, single‑prompt brittleness, not inspecting attention/MLP activations, tuned‑lens budget constraints) are appropriate and consistent with what has been learned from the literature: logit‑lens is a **useful but brittle** proxy; tuned lenses mitigate some brittleness by learning a translator; RMSNorm changes the calibration landscape; superposition and circuit phase‑changes caution against single‑view over‑reads. ([arXiv][6])

---

## 7) Specific citations to the run artefacts that support the main claims

* **Semantic onset (confirmed).**
  Meta‑Llama‑3‑8B **L=25/32**; Meta‑Llama‑3‑70B **L=40/80**; Qwen‑3‑8B **L=31/36**; Gemma‑2‑9B **L=42/42**; Gemma‑2‑27B **L=46/46**; Yi‑34B **L=44/60**. ([GitHub][1])

* **Copy‑reflex in Gemma‑2 at L0; none early elsewhere.** ([GitHub][8])

* **Rank‑only at high entropies (example).** Meta‑Llama‑3‑70B: L=40 has rank‑1 with p≈2.4e‑5, entropy≈16.94 bits. ([GitHub][4])

* **Raw‑vs‑norm artefact risk (family clustering).** Llama‑3‑70B medium; Qwen and Gemma high; Mistral‑24B low. ([GitHub][9])

* **Tuned‑lens attribution (rotation vs temperature).** Rotation dominates in Llama/Qwen/Yi; Gemma‑2‑27B calibration‑only. ([GitHub][5])

---

## 8) Closing perspective

Layer‑wise decoding will never settle metaphysics, but it can **discipline the conversation**: if “realist‑like” structure means *token‑invariant alignment to an abstract role*, then the project’s **confirmed semantics**, **echo/geometry/top‑K** crossovers, and **late vs mid‑stack** differences deliver exactly the kind of empirical signals that philosophers of universals can engage with (carefully). For a second iteration aimed at the philosophy question, the minimal additions above (micro‑suite of facts; a margin gate on rank‑only semantics) would make the bridge stronger without expanding scope.

---

### References (method background)

* **Tuned Lens.** Belrose, N. et al. *Eliciting Latent Predictions from Transformers with the Tuned Lens* (2023). ([arXiv][6])
* **RMSNorm.** Zhang, B., Sennrich, R. *Root Mean Square Layer Normalization* (2019). ([arXiv][12])
* **Superposition.** Elhage, N. et al. *Toy Models of Superposition* (2022). ([arXiv][13])
* **Induction Heads.** Olsson, C. et al. *In‑context Learning and Induction Heads* (2022). ([arXiv][14])
* **Philosophical context.** *Nominalism in Metaphysics*; *Properties* (SEP). ([Stanford Encyclopedia of Philosophy][10])

*(All other empirical claims are supported directly by the run artefacts cited above.)*

[1]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/output-Meta-Llama-3-8B-milestones.csv "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Meta-Llama-3-8B.md "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_LAYERS_BASELINE_PLAN.md "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/output-Meta-Llama-3-70B-milestones.csv "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-cross-models.md "raw.githubusercontent.com"
[6]: https://arxiv.org/abs/2303.08112?utm_source=chatgpt.com "[2303.08112] Eliciting Latent Predictions from Transformers ..."
[7]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/output-Qwen3-8B-milestones.csv "raw.githubusercontent.com"
[8]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/output-gemma-2-9b-milestones.csv "raw.githubusercontent.com"
[9]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/output-Meta-Llama-3-70B-artifact-audit.csv "raw.githubusercontent.com"
[10]: https://plato.stanford.edu/entries/nominalism-metaphysics/?utm_source=chatgpt.com "Nominalism in Metaphysics"
[11]: https://plato.stanford.edu/entries/properties/?utm_source=chatgpt.com "Properties - Stanford Encyclopedia of Philosophy"
[12]: https://arxiv.org/abs/1910.07467?utm_source=chatgpt.com "[1910.07467] Root Mean Square Layer Normalization - arXiv"
[13]: https://arxiv.org/abs/2209.10652?utm_source=chatgpt.com "Toy Models of Superposition"
[14]: https://arxiv.org/abs/2209.11895?utm_source=chatgpt.com "In-context Learning and Induction Heads"

