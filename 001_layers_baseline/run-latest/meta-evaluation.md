# Review of “Logos‑in‑Layers: 001 Layers Baseline”

## Executive summary

The baseline succeeds at turning a simple logit‑lens sweep into a measurement suite with provenance, numeric sanity checks, and per‑model and cross‑model readouts. The approach is largely sound: normalization is handled with care, deterministic runs are enforced, raw‑vs‑norm artefact checks are built in, and the reports avoid over‑claiming where calibration or decoding‑point sensitivity is high. A few issues deserve correction or tightening:

* **Citation accuracy.** The plan cites the wrong RMSNorm paper ID and a non‑existent Anthropic URL about “LN2 as the right hook.” RMSNorm is **arXiv:1910.07467**; the LN2 guidance should be grounded instead in widely used practice documented in TransformerLens (hooking the post‑block residual / `hook_resid_post`). ([arXiv][1])
* **Cross‑run provenance.** The plan itself warns not to cite `run-latest/`. The current reports nonetheless anchor most citations there. Without a stable `run_id` and `code_commit_sha` in the review, reproducibility is weaker than the code enables. ([GitHub][2])
* **Interpretation edges.** Some layers marked as “early semantics” are view‑dependent under the raw‑vs‑norm audit; claims in those regions should be phrased as **lens‑contingent**. The cross‑model write‑up usually does this correctly, but a few per‑model summaries could tighten phrasing. ([GitHub][3])

Within those caveats, the measurements support a careful, **anti‑austere‑nominalist** conclusion: across families, token‑ID‑level copy reflexes fade and **rank/ KL milestones for true answers consolidate late**, often at consistent depth fractions, with cosine alignment to the final decision direction rising near semantic onset. Methodologically, this is exactly the kind of **reusable internal structure** that austere nominalism denies. Whether those structures speak for **real universals** or **linguistic regularities** remains open; the current iteration appropriately does not attempt that heavier philosophical inference. ([GitHub][2])

---

## 1) Code & approach review

**What’s right**

* **Normalization provenance is explicit and (now) correct.** The plan documents a **post‑block vs next‑block choice** and the ε‑inside‑√ fix for RMSNorm; the JSON diagnostics for each model record which norm module fed the lens, including special handling at layer 0. This addresses a major source of spurious “early meaning.” ([GitHub][2])
* **Determinism and numeric health.** Runs set deterministic algorithms and log `any_nan`, `any_inf`, and flagged layers; per‑model evaluations show clean numeric health. ([GitHub][4])
* **Probe geometry.** The new `cos_to_final` metric (cosine between each layer’s **logit direction** and the final head’s logits) is the right quantity for detecting **rotation vs amplification**; it is scale‑free and interpretable under both norm and tuned lenses. ([GitHub][2])
* **Raw‑vs‑norm artefact check.** The suite computes KL(P_norm‖P_raw), top‑k overlaps, and marks **norm‑only semantics**; this sharply reduces over‑interpretation from the lens itself. ([GitHub][2])
* **Copy detection is token‑aware.** Moving to **ID‑level contiguous subsequence** with a tunable window and threshold is a meaningful upgrade over string heuristics. ([GitHub][2])

**What needs fixing or sharpening**

1. **Citation corrections.**

   * RMSNorm should cite **Zhang & Sennrich (2019)** `arXiv:1910.07467`, not `1910.04751`. ([arXiv][1])
   * The plan references an Anthropic post “Why LN2 is the Right Hook…” at `transformer-circuits.pub/ln2-analysis`, which cannot be located. The **defensible reference** is the TransformerLens practice of hooking at `hook_resid_post` (post‑block residual), which aligns with using LN2 for post‑block snapshots. Adjust the text to cite TransformerLens docs. ([transformerlensorg.github.io][5])

2. **Provenance discipline.** The plan itself requires citing a **stable run folder** and commit hash rather than `run-latest/`. The public reports should follow that rule; otherwise, results become moving targets. ([GitHub][2])

3. **Last‑layer mismatch in Gemma is framed as “likely temperature/softcap.”** That’s a reasonable hypothesis, and the diagnostic tries a scalar τ; just ensure the wording stays explicitly **hypothetical** unless there is model documentation. (The reports mostly do this, but keep the hedge.) ([GitHub][3])

4. **Raw/Norm sampling strategy.** The baked‑in raw‑vs‑norm check samples a handful of layers by default. Where **early semantics** are claimed, consider promoting to `full` mode for that model to guarantee the risk tier reflects all layers, not a sample. ([GitHub][2])

5. **Gate naming clarity.** The **“uniform‑margin”** and **“top‑2 gap”** gates are sensible, but users may read them as probability claims. The reports already suppress absolute probabilities when `suppress_abs_probs=true`; that policy is correct and should be emphasized wherever those gates are used as **calibration checks**, not semantic proof. ([GitHub][3])

---

## 2) Review of results & analyses

**Cross‑model patterns are mostly stated with appropriate caution.** Highlights that are both informative and defensible:

* **Late consolidation is common, with family‑specific timing.** E.g., Qwen‑2.5‑72B reaches rank‑1 only at the **final layer**, whereas Llama‑3‑70B consolidates around mid‑stack (normalized by depth). The report expresses these as **within‑model** depth fractions and cites KL/rank milestones rather than absolute probabilities—a good practice. ([GitHub][3])
* **Lens‑artefact risk stratification** (JS/L1/Jaccard, prevalence of `KL≥1.0`, presence of norm‑only semantics) properly downgrades early‑layer interpretations where raw vs norm diverge heavily (e.g., Gemma‑2‑27B high risk; Mistral‑24B low). ([GitHub][3])
* **Copy‑reflex vs semantics** is measured at the **token ID** level and reported as **Δ‑gaps across depth** rather than only point claims. Early copy on Gemma, absent on Llama/Qwen/Yi at layers ≤3, is consistent with the CSV evidence. ([GitHub][3])
* **Prism (shared‑decoder) behavior** is carefully framed as **diagnostic**: it reduces KL early for some families but does **not** always improve rank milestones, so it is **not** treated as the primary semantics lens. This avoids overstating what Prism shows. ([GitHub][3])

**Where phrasing could be tightened**

* In per‑model reports, any claim that “semantics appear at L = k” should **always** carry the lens qualifier if the **norm‑only semantics** flag is active at or before that layer (e.g., Yi‑34B and Qwen models with high risk tiers). Most reports do this; ensure there are no stray unqualified sentences. ([GitHub][3])
* When **gate stability under small rescalings** is **0.0** at the target layer (e.g., Llama‑3‑8B, Mistral‑24B, Qwen‑2.5‑72B), treat the semantic milestone as **calibration‑sensitive** in the bullet summary, not just in the diagnostics pane. ([GitHub][3])

---

## 3) Independent deep‑dive (synthesis across families)

This synthesis follows the cross‑model structure but emphasizes only **rank/KL and geometry** (cosine), and marks lens contingencies where warranted.

### 3.1 Timing of semantic consolidation

* **Gemma‑2 (9B, 27B)**: Both show **final‑layer consolidation** under the baseline lens; copy reflex is visible in early layers; tuned‑lens audits indicate **calibration‑only** for 27B (τ⋆≈3), mixed for 9B. Given high lens‑artefact scores and **decoding‑point sensitivity** at the semantic layer, treat all **early** semantics as lens‑dependent; the **confirmed** semantics is late. ([GitHub][3])
* **Llama‑3 (8B, 70B)**: Consolidation earlier relative to depth than Gemma/Qwen; **medium** artefact tier. Llama‑3‑8B shows consistent decoding‑point behavior at the marked layer, but the **small‑rescaling gate** is fragile—again a calibration, not a semantics, issue. ([GitHub][3])
* **Mistral (7B, 24B)**: Late‑mid consolidation; **artefact tier splits**—7B high, 24B low—suggest family‑/scale‑specific normalization idiosyncrasies. Under the raw‑vs‑norm audit, 24B looks robust enough that layer timings can be read more strongly than for 7B. ([GitHub][3])
* **Qwen (3‑8B, 3‑14B, 2.5‑72B)**: Timing shifts **later with scale**, culminating in **rank‑1 only at the final layer** for 72B; the tuned lens is missing for 72B, so all claims rely on the norm lens and are marked **decoding‑point** and **calibration** sensitive. ([GitHub][3])
* **Yi‑34B**: Mid‑late consolidation with **many norm‑only semantics layers** and low raw/norm top‑k overlaps at targets; tuned preference is justified; treat early semantics as **view‑dependent**. ([GitHub][3])

**Interpretation:** In line with **tuned‑lens** findings, many models look like they **rotate** into the final decision basis and/or **amplify** a proto‑direction late, not that “full meaning” is present and stable from the start. Where tuned lenses are available, they often reduce perplexity and bias and better reflect the model’s final distribution—precisely their advertised behavior. ([arXiv][6])

### 3.2 Geometry and entropy

* **Cosine to final** generally **rises** near the semantic milestone; where it rises early but ranks lag, that is consistent with **amplification after rotation** or with normalization‑induced calibration artefacts. The reports show this pattern for multiple families (see cosine milestones in per‑model JSON). ([GitHub][4])
* **Entropy** typically **stays high** through the mid‑stack and **collapses late** in families that consolidate late (Qwen‑2.5‑72B, Gemma‑2‑27B), whereas Llama tends to collapse earlier relative to depth. This matches an iterative‑inference picture, not brittle prompt echo. ([GitHub][3])

### 3.3 Raw‑vs‑norm and Prism

* **Raw‑vs‑norm** divergence and **norm‑only semantics** are concentrated in high‑risk families (Gemma, some Qwen, Yi). For **low‑risk** models (e.g., Mistral‑24B), raw and norm agree more and early semantics are rarer, supporting stronger claims. ([GitHub][3])
* **Prism** (shared decoder) reduces KL for some families (Gemma‑2‑27B, Qwen‑2.5‑72B) but does not systematically improve **earliness of rank‑1**, so it is appropriately treated as a **diagnostic calibration lens**, not evidence for earlier semantics. (The method family resembles “logit prism” style decompositions used in interpretability.) ([GitHub][3])

---

## 4) Relevance to nominalism vs realism

**What the data speak to now**

* The current iteration was explicitly scoped to **get the measurements right**. Even so, the **repeatable** emergence of semantic rank milestones late in the stack, the **token‑aware** dismantling of copy‑reflex, and the **geometry** (cosine to final) seen across unrelated architectures are **precisely the kind of structured, reusable internal patterns** that undermine **austere nominalism** (the view that there are only tokens and any talk of properties can be paraphrased away). A purely nominalist story that treats all regularities as surface wordplay struggles with depth‑indexed, model‑internal regularities that persist under alternative wordings and across models. ([GitHub][2])

**What the data do *not* yet speak to**

* Whether those structures evidence **mind‑independent universals** (realism) or **metalinguistic regularities** (conceptualism / metalinguistic nominalism) is not decided by a logit‑lens baseline. Distinguishing these requires interventions that tie internal variables to **world‑structure** rather than **linguistic structure** alone. The SEP entries give the relevant fault lines: nominalism rejects universals; some forms allow **predicates or tropes** to do the classificatory work instead. The present results are consistent with both a **linguistic‑structure** reading and a **property‑structure** reading. ([Stanford Encyclopedia of Philosophy][7])

**Prudent philosophical upshot for this iteration**

* The baseline **adds empirical pressure against austere nominalism** by showing reproducible, depth‑indexed structure that is not reducible to prompt echo. It **does not** yet adjudicate **realism vs metalinguistic nominalism**; that is correctly deferred to later iterations that add causal/grounding tests.

---

## 5) Is “getting the measurements right” complete? Targeted improvements that add real value now

The iteration is close to complete. The following changes are **non‑trivial** yet **squarely within scope** and would materially strengthen measurement reliability:

1. **Fix the citations and references in the plan.**

   * Replace RMSNorm citation with **arXiv:1910.07467**; remove/replace the non‑existent Anthropic link with a citation to **TransformerLens** documentation for post‑block hooks. This avoids mis‑teaching a subtle but crucial norm provenance issue. ([arXiv][1])

2. **Enforce stable provenance in reports.**

   * Bake the stable `run_id` and `code_commit_sha` into every report header and footnote each citation to `run-YYYYMMDD‑HHMM/…` instead of `run-latest/`. The plan already mandates this; applying it in the published markdown will prevent readers from re‑anchoring against a moving target. ([GitHub][2])

3. **Auto‑escalate the raw‑vs‑norm audit for risky cases.**

   * When `risk_tier ∈ {high}` **or** `n_norm_only_semantics_layers>0`, automatically switch `LOGOS_RAW_LENS=full` for that model so the artefact tiers are computed over **all** layers, not a sample. This costs little and removes a sampling confound at exactly the layers readers care about. ([GitHub][2])

4. **Pin the semantics claim to a *lens tag* everywhere.**

   * The cross‑model report already labels the **source lens** for the semantics layer (raw/tuned). Require the same in every **per‑model bullet summary** (e.g., “L=25, **source=raw**”) and prepend a short clause if **calibration‑sensitive** or **decoding‑point‑sensitive** at that layer. This reduces the risk that casual readers mistake a **lens‑dependent** effect for a **model‑intrinsic** one. ([GitHub][3])

5. **Tighten the copy detector’s default stability.**

   * The new **ID‑contiguous** detector is excellent. For baseline stability, set `copy_window_k=2` for models whose tokenizers tend to split named entities (e.g., German city names), and record that per‑model in the JSON. This remains in scope and reduces false negatives in early copy checks on spiky tokenizations. (The plan already exposes the knob; the suggestion is to default‑raise it where tokenization warrants.) ([GitHub][2])

6. **Cosine‑to‑final plot + single‑number summary.**

   * The `cos_to_final` column is present; add a small analysis snippet that computes the **first layer with `cos_to_final ≥ 0.6`** and **Δcos** over the last 5 layers, and print those alongside rank/KL milestones in the JSON. This triangulates rotation vs amplification without adding any new probes. ([GitHub][2])

7. **Keep absolute probabilities suppressed by default.**

   * The reports already respect `suppress_abs_probs=true` where RMS/LN lenses distort levels. Make the suppression **hard‑default** for cross‑model sections so readers don’t inadvertently compare across families with different normalization idiosyncrasies. ([GitHub][3])

These are meaningful, non‑bikeshedding upgrades that further **de‑risk measurement artefacts** while keeping the iteration within its “measurement only” charter.

---

## Closing context and links

* **Tuned Lens** paper and code underpin much of the calibrated reporting here; referencing them explicitly in the write‑ups helps readers understand why tuned vs raw behave differently. ([arXiv][6])
* **RMSNorm** details (ε inside √, γ after rescale) matter; the plan’s fix is aligned with the paper and should be the authoritative reference in the text. ([arXiv][8])
* **TransformerLens** docs remain the operational source of truth for correct hook points (e.g., `hook_resid_post`/“LN2”), which the plan can cite instead of an unavailable Anthropic URL. ([transformerlensorg.github.io][5])
* **Nominalism/realism** background for readers coming from ML: SEP entries on **Nominalism in Metaphysics** and **Properties** are the right neutral anchors for the philosophical framing the project adopts. ([Stanford Encyclopedia of Philosophy][7])

---

### Appendix: Spot‑checks against the project artefacts cited above

* **Plan items implemented.** The plan logs completion of normalization, copy detector, rank/KL metrics, raw‑vs‑norm QA, cosine metric, and last‑layer consistency checks; the model JSONs and cross‑model report reflect these additions. ([GitHub][2])
* **Per‑model examples.**

  * **Llama‑3‑8B:** semantics at **L=25 (raw)**; medium artefact tier; calibration‑sensitive under small rescalings; position‑fragile. ([GitHub][4])
  * **Gemma‑2‑27B:** semantics at **L=46 (tuned‑confirmed)**; high artefact tier; decoding‑point sensitive; strong last‑layer mismatch reduced by τ⋆≈3.0. ([GitHub][9])
  * **Qwen‑2.5‑72B:** semantics only at **L=80**; no tuned lens; high artefact tier; calibration‑ and decoding‑point sensitive. ([GitHub][10])

Overall, the measurements are careful and—once the small fixes above land—sufficiently robust to carry the limited but important philosophical claim of **anti‑austere nominalism** pressure. Deeper metaphysical conclusions will rightly wait for the planned follow‑ups that move beyond projection lenses to **causal** and **grounding** tests.

[1]: https://arxiv.org/abs/1910.07467?utm_source=chatgpt.com "Root Mean Square Layer Normalization"
[2]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_LAYERS_BASELINE_PLAN.md "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-cross-models.md "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Meta-Llama-3-8B.md "raw.githubusercontent.com"
[5]: https://transformerlensorg.github.io/TransformerLens/generated/demos/Main_Demo.html?utm_source=chatgpt.com "Transformer Lens Main Demo Notebook - GitHub Pages"
[6]: https://arxiv.org/abs/2303.08112?utm_source=chatgpt.com "Eliciting Latent Predictions from Transformers with the Tuned Lens"
[7]: https://plato.stanford.edu/entries/nominalism-metaphysics/?utm_source=chatgpt.com "Nominalism in Metaphysics"
[8]: https://arxiv.org/pdf/1910.07467?utm_source=chatgpt.com "Root Mean Square Layer Normalization"
[9]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-gemma-2-27b.md "raw.githubusercontent.com"
[10]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Qwen2.5-72B.md "raw.githubusercontent.com"
