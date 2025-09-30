# Review of *Logos‑in‑Layers / Experiment 001*

*(layer‑by‑layer logit‑lens sweeps across open‑weight base LLMs; probe: “The capital of Germany is called simply …”) *

---

## 1) Executive summary

The experiment is thoughtfully engineered for a first iteration and already implements a number of hard‑won logit‑lens hygiene practices that many published analyses omit (RMSNorm‑aware normalization, fp32 statistics, last‑layer calibration checks, raw‑vs‑norm sanity windows, ID‑level copy detection, negative controls, and ablations). The per‑model reports and the cross‑model synthesis mostly avoid overclaiming and repeatedly warn against cross‑family comparisons when calibration differs. The strongest quantitative pattern that survives these guardrails is **late semantic collapse in most bases**, with **Gemma‑2** showing a pronounced **early copy reflex** plus **final‑layer‑only** semantic rank‑1, while **Llama‑3‑70B** shows **mid‑stack** semantics under the norm lens. These patterns are supported by the run artefacts. ([GitHub][1])

That said, there are a handful of correctness and measurement issues that should be tightened before using these results as evidence in the nominalism‑vs‑realism program:

* The **norm lens vs raw lens** divergences are sometimes very large in the sampled windows (flagged as “norm‑only semantics” in several models), so **rank‑based claims should remain strictly within‑model** and be corroborated with a learned translator (Tuned‑Lens) or a shared decoder (Prism) when available. ([GitHub][2])
* **Gemma‑2** families show **~1 bit KL at the last post‑block layer** to the model head (warn_high_last_layer_kl=true), which means **absolute probabilities from the norm lens are mis‑calibrated** for this family; stick to **ranks and within‑model milestones** for Gemma. ([GitHub][3])
* **Single‑prompt dependence** remains a major scoping limitation; the cross‑model conclusions should be labeled *probe‑specific*. This is acknowledged in the project notes but deserves stronger procedural enforcement (multi‑prompt suites). ([GitHub][1])

On philosophical relevance: as an initial step, the consistent **surface→meaning transition** observed **within** each model (even if late) provides *some* pressure against **austere nominalism** (the view that there are only particular word tokens and nothing resembling stable, reusable structure). However, it is not yet discriminative between **metalinguistic nominalism** and **realism**: the present probe demonstrates structured internal regularities, but does not show that these regularities track mind‑independent universals rather than merely linguistic usage. Strengthening this case will require generalization across tasks, prompts, languages, and interventions that causally verify the reuse and portability of the putative “forms”. ([GitHub][1])

---

## 2) Methods & code: what is solid vs. what needs tightening

### What is solid

1. **Architecture‑aware normalization & precision.**
   The run applies the **next block’s ln1 for pre‑norm** / **current ln2 for post‑norm**, computes LN/RMS statistics in **fp32**, and promotes the **unembedding to fp32** for decoding. This is the right direction: naive re‑application of LayerNorm/RMSNorm can otherwise inject large calibration artefacts; for RMSNorm the ε must be **inside** the square root. The notes document this explicitly and place the fix at the 2025‑08‑24 cut‑over. ([GitHub][1])
   External context: RMSNorm’s definition and motivation are consistent with Zhang & Sennrich (2019). ([arXiv][4])

2. **Last‑layer consistency checks.**
   Every run logs `KL(P_last || P_final)` and estimates a scalar temperature; **Llama/Qwen/Mistral/Yi** runs show near‑zero last‑layer KL (good), while **Gemma‑2‑9B/27B** show ≈1 bit (warn). This justifies the internal rule “prefer ranks, suppress absolute probs” for families with last‑layer mismatch. ([GitHub][5])

3. **Raw‑vs‑norm lens sanity windows.**
   Sampled windows (±4 around centers) record `KL(P_norm || P_raw)` and flag **norm‑only semantics**. Several models (e.g., **Llama‑3‑8B**, **Yi‑34B**) show norm‑only semantic layers; **Llama‑3‑70B** shows *low* global artefact risk but still has norm‑only semantics in the very top layers. This is a crucial guardrail against “lens‑induced early semantics”. ([GitHub][2])

4. **ID‑level gold & copy detection.**
   The **gold** token is handled at **ID level** (first answer piece), and **copy** detection uses **ID‑level contiguous subsequences** with explicit thresholds and a soft/windowed variant; whitespace/punctuation are masked. This avoids many classic string‑match pitfalls. ([GitHub][1])

5. **Sidecars for calibration/geometry.**
   **Tuned‑Lens** sidecars (where present) and **Prism** sidecars provide independent calibration views. The project literature correctly notes that Tuned‑Lens generally **reduces the bias and perplexity** of vanilla logit lens. ([arXiv][6])

### What needs tightening or fixing

* **Scope of the raw‑lens check.** The windowed sanity is sampled (low cost), but in models with high risk (e.g., **Gemma‑2**, **Yi‑34B**) a **full per‑layer raw‑vs‑norm** sweep for the pure NEXT token should be emitted at least once per family for auditability; otherwise, early “bumps” could still be artefacts. (The JSON’s window KLs of **~5–90 bits** in some runs are too large to rely on norm‑lens probabilities—even ranks can shift when the dominating logit changes under raw.) ([GitHub][7])

* **Interpreting cosine to final (`cos_to_final`).** This is a *logit‑direction* cosine; because the final row is itself the reference, late‑layer cosines approach 1 by construction. Analyses should avoid implying cross‑family comparability and keep cosine statements strictly descriptive *within* a model (“alignment consolidates late/early”). The reports generally do this, but some phrasings (e.g., “direction alignment rises early”) are easy to misread as family comparisons. ([GitHub][8])

* **Gemma family calibration.** For **Gemma‑2‑9B/27B** the last‑layer KL≈1 bit persists even after a scalar‑temperature fit, so **absolute masses (entropy, answer probability) from the norm lens should be de‑emphasized** in prose and plots; stick to rank milestones and within‑model depth fractions. The per‑model reports do caution this, but the cross‑model text should repeat the caveat at each Gemma claim. ([GitHub][3])

* **Single‑prompt bias.** The project notes acknowledge this. Given how sensitive copy‑flags and early ranks are to tokenization quirks, *at least* a small, counterbalanced prompt suite (countries with multi‑piece capitals, languages with leading‑space variants, alternative wordings) should be considered part of “measurement” rather than “next iteration”. ([GitHub][1])

* **Prism provenance and objectives.** The cross‑model report treats Prism correctly as a diagnostic, but given the mixed/regressive behavior in many families, **Prism’s whitening/rank, fit data, and held‑out validation should be summarized in a dedicated paragraph per run** (not just JSON). This will prevent misinterpretation of negative KL deltas as “failures” when Prism was trained for a different operating point. ([GitHub][8])

External references that justify this measurement posture include the **Tuned‑Lens** paper’s empirical demonstration that vanilla logit lens is biased and brittle—and that a learned translator meaningfully reduces that bias—plus multiple write‑ups on LayerNorm/RMSNorm subtleties and LN‑folding biases that can leak into logits. ([arXiv][6])

---

## 3) Results & analyses: what’s correctly interpreted vs. overstated or missed

### Findings that are well‑supported by the artefacts

* **Gemma‑2‑9B/27B**

  * **Strict copy at layer 0; semantics only at the last layer.** The gemma‑2‑9B run shows `L_copy=0` and `L_semantic=42`; gemma‑2‑27B shows `L_copy=0`, `L_semantic=46`. Ablating “simply” does **not** move these indices. The JSON also flags warn_high_last_layer_kl≈1 bit, consistent with mis‑calibration of the norm lens for Gemma. ([GitHub][3])
  * Cross‑model summary reproduces this pattern and correctly warns not to compare absolute probabilities across families when last‑layer KL isn’t ≈0. ([GitHub][8])

* **Llama‑3‑70B**

  * **Mid‑stack semantics under norm lens (≈40/80) with clean last‑layer agreement** (KL≈0), and **no early copy flags**. Raw‑vs‑norm shows **low global artefact risk** while still flagging **norm‑only semantics in the top window**, so the report’s choice to **prefer ranks over absolute probabilities near the top** is sound. Ablation nudges semantics by **+2 layers**, small but detectable. ([GitHub][5])

* **Mistral‑7B‑v0.1 & Mistral‑Small‑24B‑Base‑2501**

  * **Mistral‑7B:** rank‑1 at **L≈25**; raw‑vs‑norm window shows norm‑only semantics at **L=32** → correct to lead with ranks. **Ablation** moves semantics by −1 layer (noise‑scale). ([GitHub][9])
  * **Mistral‑Small‑24B‑Base:** late collapse at **L≈33** with **low** lens‑artefact risk and clean last‑layer KL≈0. ([GitHub][10])

* **Qwen family (3‑8B / 3‑14B / 2.5‑72B)**

  * **8B:** rank‑1 at **L=31**; final KL≈0; high lens‑artefact risk flagged → use ranks. ([GitHub][11])
  * **14B:** rank‑1 at **L=36**; final KL≈0; no early copy; control shows robust France→Paris margin. ([GitHub][12])
  * **72B:** **very late** rank‑1 at **L=80/80**; final KL≈0; ablation leaves L_sem unchanged (0 shift). ([GitHub][13])

* **Yi‑34B**

  * Rank‑1 at **L=44/60**; last‑layer KL≈0. Raw‑vs‑norm window flags extensive **norm‑only semantics** layers ({44–48,56,60}) → strictly prefer rank milestones. ([GitHub][7])

* **Prism and Tuned‑Lens sidecars (where present).**

  * **Prism** is **regressive** in most families here (KL percentiles worse than baseline; rank milestones not earlier), with a **notable mid‑depth KL reduction only in Gemma‑2‑27B**. The cross‑model report correctly treats Prism as a shared‑decoder diagnostic rather than a replacement head. ([GitHub][8])
  * **Tuned‑Lens** shows **KL reductions** (e.g., Qwen‑8B, Mistral‑7B) but **does not consistently move rank‑1 earlier** on this single probe—again appropriately reported as a sidecar, not a headline. This matches public findings that Tuned‑Lens reduces logit‑lens bias and improves calibration. ([GitHub][11])

### Overstatements or misses

* **Cross‑family comparisons of absolute quantities.** The reports mostly refrain, but a few sentences drift toward comparing cosine or entropy levels across families; with different normalization families and last‑layer mismatches (Gemma), this should be systematically avoided or reframed as within‑model milestones only. ([GitHub][8])

* **Under‑emphasis of raw‑vs‑norm flags in Yi and Gemma.** The per‑model texts do mention “high” risk, but given **very large window KLs** and **norm‑only semantics** at exactly the reported collapse layers, each semantic claim should be explicitly restated as **rank‑only under norm lens; raw lens does not corroborate**. ([GitHub][7])

* **Prism provenance in prose.** The JSON contains `prism_provenance` and summary metrics; a short, human‑readable paragraph per model would make it harder to misread Prism’s negative deltas as a “failure” rather than as **evidence that a shared decoder trained under one distribution does not transfer to this probe**. ([GitHub][8])

---

## 4) Independent deep‑dive synthesis (within the probe’s limits)

**A. Surface→meaning transition is universal but usually late**
Across all nine analysed bases, the **answer’s first token reaches rank‑1 only after mid‑depth**, often very late (≥70% of layers): Qwen‑3‑8B (L=31/36), Qwen‑3‑14B (36/40), Qwen‑2.5‑72B (80/80), Mistral‑7B (25/32), Mistral‑24B (33/40), Yi‑34B (44/60), Llama‑3‑8B (≈25/32, but with norm‑only flags in the window), **Llama‑3‑70B (≈40/80, notably earlier than others)**, **Gemma‑2** (only at final). Each of these statements is documented in the per‑model JSON summaries. ([GitHub][11])

**B. “Early copy reflex” is family‑specific**
The **strict ID‑level copy rule (τ=0.95)** and **soft k=1** trigger **at layer 0** only in **Gemma‑2**; all other families show **no early copy** under the same rule. This argues that the striking layer‑0 “echo” in Gemma is a **family artifact**, not a universal phenomenon. ([GitHub][8])

**C. Calibration posture matters**
Where **last‑layer KL≈0** (most families), **KL(P_layer||P_final)** declines late alongside rank improvements, supporting a narrative of **late sharpening**. Where **last‑layer KL≈1 bit** (Gemma‑2), **absolute probabilities are mis‑calibrated** even at the top layer, so **rank milestones** are the only safe semantics indicators. This exactly matches the Tuned‑Lens literature’s warning that vanilla logit lens is biased relative to the model’s true head. ([GitHub][8])

**D. Geometry vs. calibration**
Within models, **cosine to the final logit direction** increases toward 1.0 by construction; the more informative piece is **where cosine crosses low thresholds relative to where rank‑1 appears**. In **Llama‑3‑70B**, cosine thresholds are only reached at the end (L=80), yet rank‑1 appears at L=40—suggesting **direction comes into place by ~40 but calibration/alignment consolidates only near the top**. In contrast, **Gemma‑2** keeps the surface form as top‑1 through almost the entire stack and flips only at the end, i.e., **extreme “late meaning”**. ([GitHub][5])

**E. Sidecars**
**Prism** improves mid‑depth KL only in **Gemma‑2‑27B** (large reductions), is mixed in **Yi/Qwen‑72B**, and regressive elsewhere; **Tuned‑Lens** reduces KL but does not reliably make rank‑1 earlier on this probe. Conclusion: **use ranks for semantics**, use **Tuned‑Lens KL deltas** as calibration diagnostics, and treat **Prism** as a shared‑decoder sanity check rather than as a semantics detector. ([GitHub][8])

---

## 5) Philosophical read‑through: nominalism vs. realism (what this run supports and what it doesn’t)

**What the run supports (against austere nominalism).**
Even with a single, simple prompt, every base model exhibits a **consistent surface→meaning trajectory**: token mass and rank migrate from prompt tokens toward the **answer token** as depth increases, with **repeatable** milestones under multiple families and calibrations. This **within‑model structural regularity** is hard to square with the strongest form of **austere nominalism** that denies any substantive internal structure beyond token transitions: the trajectories are not arbitrary, and many models exhibit **similar qualitative phases** (“echo‑dominated early… sharpening late”). The project’s plan explicitly frames this as the initial target. ([GitHub][1])

**What the run does *not* yet establish (metalinguistic nominalism vs. realism).**
The present evidence is consistent with **metalinguistic nominalism**—that the observed structures encode **linguistic regularities** (e.g., distributional facts about “capital of X is Y”), without committing to **mind‑independent universals**. To push toward realism, one would need **stronger invariance** (task‑ and language‑general features, robust under paraphrase and adversarial controls) and **causal transfer** (e.g., manipulating a “capital‑of” representation discovered in one model or context causing the right behavior in others). Current LLM interpretability literature suggests such directions (e.g., **learned translators / Tuned‑Lens** for calibration; **dictionary learning / sparse autoencoders** for feature‑level invariances), but this run has not yet executed them on this task. ([arXiv][6])

---

## 6) Concrete, high‑value next steps (edits to `PROJECT_NOTES.md` that would materially improve the evidence)

The items below are *non‑bikesheddy* and directly increase the philosophical informativeness while staying within the project’s current toolset.

### A. Expand the probe set (small but decisive lift)

* **Balanced prompt suite.** Add 12–20 *matched* prompts covering:
  – *Single‑piece vs multi‑piece answers* (“Bern”, “New Delhi”, “Côte d’Ivoire”).
  – *Paraphrase templates* (“The capital city of Germany is”, “Germany’s capital is”).
  – *Cross‑lingual variants* (DE/FR/ES prompts).
  – *Non‑geographic universals*: hypernymy (“A robin is a kind of …”), synonyms/antonyms, definitional prompts.
  For each, log the same summary indices (`first_rank_le_{10,5,1}`, `L_surface_to_meaning`, raw‑vs‑norm window). The aim is to demonstrate *stability of the surface→meaning trajectory under controlled stimulus changes*—a precondition for any anti‑austere argument. (This concretely mitigates the single‑prompt caveat already noted in the repo.) ([GitHub][1])

### B. Strengthen measurement where artefact risk is high

* **Full raw‑lens sweep sidecar for high‑risk families.** For **Gemma‑2** and **Yi‑34B**, add a **raw‑lens pure‑next‑token CSV** mirroring the norm CSV (one row per layer). Promote the raw‑vs‑norm comparison from a sampled window to a **layer‑by‑layer plot** once per family. This directly addresses the very large window KLs currently reported. ([GitHub][7])

* **Gate “semantic collapse” by corroboration.** Define **semantic onset** as rank‑1 under **both**: (a) norm lens *and* (b) either **Tuned‑Lens** or **raw lens** (no requirement to agree on the same layer, but within ±n layers). This makes early semantics claims robust to lens artefacts and aligns with the Tuned‑Lens literature showing reduced bias vs logit lens. ([arXiv][6])

### C. Feature‑level analysis to test reuse/portability (directly relevant to realism)

* **Dictionary‑learning / SAEs on the residual stream at the NEXT position.** Train **sparse autoencoders** on activations around the NEXT token across the balanced prompt suite; label discovered features by **logit attributions** toward the answer token and away from prompt token sets. Then:
  – **Causal tests**: activate/deactivate the top “capital‑of” or “country→city” features and measure changes in the NEXT distribution;
  – **Cross‑prompt reuse**: do the same features light up for *France→Paris*, *Japan→Tokyo*, etc.?
  – **Cross‑family transfer** (optional): map top features across two bases by correlation of **tuned‑lens** logits or by **whitened Procrustes** between subspaces.
  This gives a **feature‑level** invariance/causality story—precisely the sort of evidence needed to move beyond metalinguistic nominalism. (Anthropic and OpenAI have demonstrated that SAEs can recover monosemantic features at scale.) ([transformer-circuits.pub][14])

### D. Make Prism/Tuned‑Lens evaluation more decision‑useful

* **Per‑model Prism paragraph** (not just JSON): report training depths, whitening kind, rank `k`, and KL deltas at {25,50,75}% with a one‑line verdict (improves mid‑depth KL / regressive / neutral). Keep stating that Prism is **shared‑decoder QA**, not a semantics oracle. ([GitHub][8])

* **Norm‑temperature baseline plots.** The JSON already includes `kl_to_final_bits_norm_temp`; add a small per‑model chart (or textual bullet) that decomposes **tuned vs norm** into **rotation** (translator) vs **calibration** (temperature). This implements exactly the plan’s §1.16 and avoids over‑crediting the translator for wins that are purely temperature. ([GitHub][1])

### E. Minimal causal interventions with existing tooling

* **Ablate “echo mass” tokens**: at mid‑depths before semantic collapse, **zero** the logits (or residual projections) corresponding to the **prompt token set** and observe whether rank‑1 onset advances. If it does across prompts, this supports a **general surface→meaning pressure** rather than idiosyncratic token quirks. (This can be done post‑hoc in the analysis path without re‑running the model forward.) ([GitHub][1])

* **Activation patching across prompts**: patch the residual stream (around the inferred onset layer) from a *donor* prompt where the answer is already rank‑1 into a *recipient* prompt just before collapse; measure whether the recipient flips earlier. This is a standard circuit‑tracing move and directly probes reuse. ([Neel Nanda][15])

---

## 7) Relation to the broader interpretability literature (why these adjustments matter)

* **Vanilla logit lens vs learned translators.** The **Tuned‑Lens** literature shows that vanilla logit lens is biased (systematically mis‑weights parts of the vocabulary) and that learned translators **substantially reduce bias and perplexity**; using Tuned‑Lens to corroborate rank milestones is therefore appropriate. ([arXiv][6])

* **Normalization details matter.** RMSNorm’s formula and LN‑folding effects can visibly change apparent “early semantics”; using correct epsilon placement and architecture‑aware norms is essential. The repo already addresses this; keeping raw‑lens corroboration makes it solid. ([arXiv][4])

* **Dictionary learning / SAEs as the next lever.** Recent work shows that **SAEs** recover interpretable features at non‑toy scales (including production models), enabling **causal** and **cross‑dataset** investigations. That is exactly the type of evidence needed to separate “mere distributional regularities” from “reusable forms” in the philosophical sense. ([transformer-circuits.pub][14])

---

## 8) Specific call‑outs on the current reports

* **Cross‑model evaluation** rightly emphasizes **within‑model rank milestones** (≤10/5/1) and flags **lens‑artifact risk** per family. Keep that framing and explicitly **tag every early‑semantics claim with the raw‑lens status** (present/absent) to prevent drift in future iterations. ([GitHub][8])

* **Per‑model Gemma‑2 reports** are careful about last‑layer KL≈1 and mixed Prism results. Maintain the rule “no absolute‑probability claims for Gemma under the norm lens”; prefer ranks and highlight **Tuned‑Lens** corroboration if/when added. ([GitHub][3])

---

## 9) Bottom line for the philosophy goal—what this iteration buys

This iteration **does** provide evidence **against austere nominalism**: even very different base models share a **repeatable** within‑model **surface→meaning trajectory** on a concrete factual task, measurable by **rank milestones**, **echo mass decay**, and **geometric crossovers**. However, the evidence **does not yet** discriminate between **metalinguistic nominalism** and **realism**. The proposed next steps—especially the **multi‑prompt suite**, **raw‑lens corroboration**, and **feature‑level causal reuse**—are the shortest path to raising the evidential bar without abandoning the project’s current tooling. ([GitHub][1])

---

### Pinned references

* **Experiment artefacts & notes.** Run‑level per‑model evaluations (e.g., gemma‑2‑9B, Llama‑3‑70B, Qwen‑3‑8B, Mistral‑7B, Yi‑34B), cross‑model evaluation, and project notes describing normalization, copy rules, raw‑vs‑norm windows, Prism/Tuned‑Lens, and mass/geometry diagnostics. ([GitHub][3])

* **Tuned‑Lens** (reduces logit‑lens bias; better calibration/perplexity). ([arXiv][6])

* **RMSNorm** (definition; implementation cautions). ([arXiv][4])

* **TransformerLens** (analysis infrastructure; LN‑bias context). ([GitHub][16])

* **Anthropic / Transformer‑Circuits SAE work** (monosemantic features; scaling). ([transformer-circuits.pub][14])

* **Activation/attribution patching** (for causal tests). ([Neel Nanda][15])

---

**Verdict:** As a first iteration, the engineering and the reporting discipline are unusually strong for a hobby‑scale study. With a modest set of measurement upgrades and a shift toward **feature‑level** and **multi‑prompt causal** evidence, the next round can produce materially stronger, philosophy‑relevant claims while staying squarely within mainstream interpretability practice.

[1]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/PROJECT_NOTES.md "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Meta-Llama-3-8B.md "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-gemma-2-9b.md "raw.githubusercontent.com"
[4]: https://arxiv.org/abs/1910.07467?utm_source=chatgpt.com "Root Mean Square Layer Normalization"
[5]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Meta-Llama-3-70B.md "raw.githubusercontent.com"
[6]: https://arxiv.org/pdf/2303.08112?utm_source=chatgpt.com "Eliciting Latent Predictions from Transformers with the ..."
[7]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Yi-34B.md "raw.githubusercontent.com"
[8]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-cross-models.md "raw.githubusercontent.com"
[9]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Mistral-7B-v0.1.md "raw.githubusercontent.com"
[10]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Mistral-Small-24B-Base-2501.md "raw.githubusercontent.com"
[11]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Qwen3-8B.md "raw.githubusercontent.com"
[12]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Qwen3-14B.md "raw.githubusercontent.com"
[13]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Qwen2.5-72B.md "raw.githubusercontent.com"
[14]: https://transformer-circuits.pub/2023/monosemantic-features?utm_source=chatgpt.com "Decomposing Language Models With Dictionary Learning"
[15]: https://www.neelnanda.io/mechanistic-interpretability/attribution-patching?utm_source=chatgpt.com "Attribution Patching: Activation Patching At Industrial Scale"
[16]: https://github.com/TransformerLensOrg/TransformerLens?utm_source=chatgpt.com "TransformerLensOrg/TransformerLens: A library for ..."
