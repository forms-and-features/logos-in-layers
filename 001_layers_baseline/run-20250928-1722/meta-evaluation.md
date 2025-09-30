
# Review of “Logos‑in‑Layers” — code, results, cross‑model synthesis, and philosophical takeaways

---

## Executive summary

The pipeline is solid and unusually careful for a first sweep: it applies an architecture‑aware normalization lens, sticks to ID‑level answer alignment, logs rank/KL milestones, includes a negative control, and ships both Prism and Tuned‑Lens sidecars. The biggest empirical patterns supported by the artifacts are:

* **Late semantic collapse in most models** (rank‑1 for *Berlin* appears in the last 10–30% of layers), with **clean final‑head calibration** for most families (KL≈0 at the last layer), and **no strict copy‑collapse** in early layers—**except the Gemma‑2 family**, which shows an extreme early copy reflex on the filler token “simply” and a systematic final‑layer calibration mismatch under the lens. ([GitHub][1])
* **“Early direction, late calibration.”** Cosine to the final logit direction is often substantial very early (sometimes from layer 0), while KL to the final head remains high until late; Qwen2.5‑72B is a clean example. This is precisely the scenario for which Tuned‑Lens is expected to help. ([GitHub][2])
* **Prism sidecars rarely improve early/mid‑depth calibration** on this task; they frequently increase KL and fail to hit rank milestones earlier than the baseline lens. Tuned‑Lens typically **reduces early/mid‑depth KL** but may **slightly delay rank‑1** vs the norm‑lens baseline—exactly as the Tuned‑Lens paper would predict when calibration dominates rotation at mid‑depths. ([GitHub][1])

These findings are **compatible with (but do not yet entail)** a realist reading that models maintain robust, reusable structure not reducible to token‑list regularities; the present run is still correlational. The **next tranche of work should be causal** (activation/attribution patching, head fingerprinting, portable concept vectors) and **cross‑prompt**, as already sketched in the project plan; a few targeted adjustments below will materially increase philosophical and scientific value. ([GitHub][3])

---

## Code & methodology review

### What is correct and strong

* **Architecture‑aware normalization and precision policy.** The run detects LN/RMS modules, selects the correct post‑block normalizer (ln2 for post‑norm models), logs that choice, and promotes the unembedding path to fp32 for analysis. This avoids the common “ε‑outside‑sqrt” and “wrong‑stream γ” pitfalls that produce spurious early semantics. The relevant decisions (e.g., `norm_alignment_fix`, `unembed_dtype`, `layer0_position_info`, temperature snapshots) are persisted in JSON. ([GitHub][4])
  These design choices align with well‑established concerns in the literature about logit‑lens brittleness and the value of calibrated/learned decoders like Tuned‑Lens. ([arXiv][5])
* **ID‑level gold alignment and copy detection.** The script resolves the *first answer token ID* from the model’s tokenizer and uses **ID‑level** checks both for `is_answer` and for detecting copy‑collapse as a **contiguous subsequence** of prompt IDs. This avoids string‑level false positives and multi‑piece misalignment. ([GitHub][4])
* **Negative control & ablation.** The France→Paris control is included and summarized; a **no‑filler** variant is run to test stylistic sensitivity (important for diagnosing Gemma’s early copy reflex). ([GitHub][4])
* **Calibration, geometry, and surface metrics.** The pure next‑token CSVs include entropy, KL‑to‑final, rank, answer mass, prompt‑echo mass, top‑K prompt mass coverage, cosine to final, and norm‑temperature snapshots; JSON aggregates first‑hit thresholds (`first_rank_le_{10,5,1}`, `first_kl_below_{1.0,0.5}`) and surface/geometry crossovers. This is exactly the right set for within‑model interpretation. ([GitHub][3])
* **Raw‑vs‑norm “lens sanity”.** The run records sampled dual‑decode divergences (`kl_norm_vs_raw_bits`, norm‑only semantics, risk tier). This correctly encourages **rank milestones** over raw probabilities when risk is high. ([GitHub][3])

### Fragilities and actionable issues

1. **Final‑layer calibration mismatch in Gemma.** The JSON repeatedly flags high last‑layer KL under the lens (≈1 bit) and large norm‑vs‑raw divergences; this is *not* an error in the script but a family‑specific head calibration issue that makes **absolute probabilities** suspect—even at the last layer. Evaluations sometimes discuss Gemma’s final probabilities without immediately foregrounding this fact; those lines should consistently push readers toward **rank thresholds** for Gemma. ([GitHub][6])
2. **Raw‑lens sampling may under‑detect artifacts.** The default sampled mode can miss a narrow early window of norm‑only semantics. For families flagged as **high risk** (e.g., Qwen3‑14B, Gemma‑2‑27B), add an *automatic* escalation to a denser check around candidate collapse layers (±4 layers) to harden conclusions based on early rank drops. ([GitHub][1])
3. **Copy‑collapse strictness vs usefulness.** The strict rule (`p_top1>0.95`, k=1) is good for precision but under‑fires, leaving Δ‑collapse undefined in most models. Soft/windowed detectors (already present) should be promoted to a **first‑class summary** in the markdown, not just JSON side fields, to stabilize cross‑model comparisons. ([GitHub][3])
4. **Prism interpretation.** The sidecars often worsen KL/ranks at most depths here; that is not necessarily a failure—Prism is a **shared linear decoder** used as a calibration/robustness baseline, not a tuned translator. Reports should make this status explicit to avoid implying regression relative to the model head. Point to an explanatory reference when you call a Prism run “regressive.” ([Neural Blogs][7])

---

## Results and existing analyses — correctness, over‑statements, omissions

### Per‑family highlights (selected)

* **Gemma‑2‑9B.**
  Early **copy reflex** on “simply”: strict and soft copy fire at **layer 0**; **semantic collapse at L=42**; **warn_high_last_layer_kl=true** with temp_est≈2.61; Prism worsens KL substantially; Tuned‑Lens shows mid‑stack ΔKL improvements but “rank‑earliness” unchanged and a regression warning in skip‑sanity. These points are well supported in the artifacts. ([GitHub][6])

* **Qwen3‑8B.**
  **No early copy**, **rank‑1 at L=31**, clean final calibration (KL≈0), Tuned‑Lens reduces early/mid KL but slightly **delays rank‑1** vs baseline; Prism is **regressive** on KL and ranks. The evaluation correctly emphasizes rank milestones over probabilities given high lens‑artifact risk in the raw‑vs‑norm check. ([GitHub][8])

* **Qwen2.5‑72B.**
  **Rank‑1 only at the final head (L=80)**, **clean last‑layer KL≈0**, and a textbook **“early direction, late calibration”** pattern: cos_to_final ≈0.59 at L0 while KL is ≈9.39 bits, dropping to ≈0 only at the end. Prism is regressive at the final head; tuned reduces mid‑depth KL but does not advance rank milestones. The evaluation reads the artifacts accurately. ([GitHub][2])

* **Cross‑model synthesis.**
  The cross‑model write‑up captures the main pattern—**late collapse in most models, earliest in Llama‑3‑70B (≈50% depth)**—and **flags Gemma’s family‑level calibration issue**. It also correctly recommends **ranks over probabilities** whenever last‑layer KL≠0 or raw‑vs‑norm risk is high. One place to tighten: when discussing **rest‑mass** and **top‑K coverage**, explicitly pair those with `topk_prompt_mass@50` and remind readers these are **coverage proxies**, not fidelity checks. ([GitHub][1])

### A few specific notes to amend or foreground

* When a model’s **last‑layer KL>0** under the lens (e.g., Gemma), **do not** summarize final probabilities without immediately reminding readers of the calibration mismatch; lead with **ranks** and **rank thresholds**. ([GitHub][6])
* Where a report highlights mid‑depth KL spikes/drops, **quote the raw‑vs‑norm risk tier and `max_kl_norm_vs_raw_bits`** alongside the observation (several single‑model evals allude to this but do not always include the numbers). ([GitHub][1])
* The Prism sidecars are **diagnostics**, not expected improvements; avoid language that would read as comparing them to the **model head** rather than to the **norm lens**. ([Neural Blogs][7])

---

## Independent cross‑model deep‑dive (within‑model comparisons)

This section summarizes only *within‑model* facts supported by the run artifacts; absolute probabilities are avoided when final‑layer KL≠0.

* **Copy reflex (L0–L3).** Among the ten models, **only Gemma‑2‑9B and Gemma‑2‑27B** trigger strict/soft copy at the start; all others show **no early copy flags**. This difference is robust across strict and soft detectors at default thresholds. ([GitHub][1])
* **Rank milestones.** Reported `first_rank_le_{10,5,1}` (diagnostics) are:
  – Llama‑3‑70B: **{10,5,1} at {38,38,40}**; **earliest** rank‑1 relative to depth (~50%).
  – Qwen3‑8B: **{29,29,31}**; Qwen3‑14B: **{32,33,36}**; Qwen2.5‑72B: **{74,78,80}**.
  – Llama‑3‑8B and Mistral‑7B: rank‑1 at **L=25** (32‑layer stacks).
  – Yi‑34B: **rank‑1 at L=44**.
  – Gemma collapses only at the final head in both sizes. These are all quoted directly from diagnostics. ([GitHub][1])
* **Geometry vs calibration.** In several families, **cos_to_final rises early** (≥0.2 by very early layers) while **KL to the final** stays high until late; the Qwen 72B case makes this especially clear. This is the qualitative signature that a **rotation/translator** (Tuned‑Lens) can meaningfully reduce KL **without changing ranks much**. ([GitHub][2])
* **Surface→meaning transitions (mass/geometry).**
  Qwen3‑14B: `L_surface_to_meaning_norm ≈ 36` with answer_mass ≈ 0.953 vs echo_mass ≈ 4.4e‑06; `L_geom_norm ≈ 35`. Comparable within‑model patterns for Qwen3‑8B, Llama‑3‑8B, Yi‑34B are present. These support a **consistent late crossover** where answer mass overtakes diffuse prompt‑token mass. Avoid cross‑family comparisons of *absolute* masses. ([GitHub][1])
* **Sidecars.**
  **Prism** frequently increases KL at early/mid depths and **does not** improve rank milestones (e.g., Qwen3‑8B and Gemma‑2‑9B summaries). **Tuned‑Lens** generally **reduces** KL by 3–8 bits at early/mid depths but **leaves rank‑1 the same or slightly later**; last‑layer agreement remains good in both baseline and tuned runs for the well‑calibrated families. ([GitHub][8])

### Context from current literature

* **Logit‑Lens vs Tuned‑Lens.** Tuned‑Lens explicitly **learns affine translators** per layer to align intermediate representations to the model’s head; it is known to be more predictive and less biased than raw logit‑lens, and its causal‑basis results show overlap with model features. The behavior observed here—**large early/mid‑depth KL reductions with small changes to rank earliness**—is in line with those results. ([arXiv][5])
* **RMSNorm details.** Correct ε placement **inside** the square‑root and attention to γ‑scaling on the right residual stream matter for lens calibration. The code path and notes reflect this, which is essential when probing RMS‑normed families like Qwen and Gemma. ([arXiv][9])
* **Attribution/activation patching.** For causal follow‑ups, **activation patching** and its gradient‑based approximation **attribution patching** are the current workhorses; they scale causal mapping and are well suited to the “capital‑of” setting. The plan’s Group 3 is well aligned with these methods. ([Neel Nanda][10])

---

## Philosophical relevance: what the present sweep does and does not show

* The current evidence **pressures austere (extreme) nominalism**. If early layers already align in direction with the ultimate answer while top‑1 remains elsewhere, and if rank‑1 appears reliably at a consistent late layer across families—and especially if such patterns persist across paraphrases and languages—then the best explanation involves **reusable internal structure** rather than a mere list of token‑token co‑occurrences. The present data **suggests** such structure (e.g., **early direction** + **surface→meaning crossovers**) but is **not yet causal**. ([GitHub][2])
* These results **do not yet distinguish** *metalinguistic nominalism* from *realism*: a metalinguistic nominalist can still claim the structures are sophisticated facts about linguistic predicates. Distinguishing these requires **invariance** (e.g., across languages) and **manipulative evidence** (patching/ablation) that shows **the same internal objects** do the work under non‑trivial relabelings and modalities. The project’s Group 2–4 plans target precisely this gap. ([GitHub][3])

---

## Recommendations that add **non‑negligible value** (no bikeshedding)

### 1) Harden measurement where it materially changes interpretation

* **Escalate raw‑vs‑norm checks around candidate collapse layers** for any model whose sampled check reports `lens_artifact_risk="high"` or `max_kl_norm_vs_raw_bits ≥ 1`. Re‑run the dual decode on **±4 layers** around `first_rank_le_5`. This costs almost nothing and prevents over‑reading early rank improvements. ([GitHub][1])
* **Promote soft/windowed copy to the summary.** Keep strict copy as the headline conservative metric, but **always** report `L_copy_soft[k]` (k∈{1,2,3}) and compute Δ‑collapse with the soft index when strict is null; this stabilizes cross‑model claims. ([GitHub][3])
* **Make Prism’s role explicit** in the write‑ups: a **shared decoder** for robustness/comparability, **not** expected to match the final head; label regressions as “relative to norm lens,” not “relative to the model head.” Link a short Prism explainer. ([Neural Blogs][7])

### 2) Move to **small but decisive causal tests** before scaling breadth

* **Activation/attribution patching at L≈(L_sem−3…L_sem+3).** Start with one well‑calibrated model (e.g., Llama‑3‑70B or Mistral‑Small‑24B). Record **causal L_sem** and split by sublayer (attn‑only vs MLP‑only) to separate retrieval vs construction. This immediately shifts the interpretation from correlation to **necessity** claims. ([Neel Nanda][10])
* **Head fingerprinting near L_sem.** Catalog heads with both high attention to the subject→answer span and ≥0.5‑bit causal effect via zero‑ablation. Persist a `relation_heads.json`. This yields **named internal causes** that can be stress‑tested under permutations. (This is already in the plan; elevate it in priority.) ([GitHub][3])
* **Portable concept vectors (CBE‑style).** Extract a low‑rank vector at L_sem that raises log‑prob of the correct capital across unseen country prompts. Even modest portability would be strong anti‑austere evidence; it also enables later permutation and multilingual tests. (Again, already planned; prioritize.) ([arXiv][11])

### 3) Add two **low‑lift** controls that punch above their weight

* **Predicate‑permutation control (Quine guard).** Apply a fixed bijection to country tokens across a prompt battery and **re‑use the same heads/vectors** found on clean data; record the **drop** in control margin, vector effect, and head consistency. This is a crisp test of whether internal objects track **relations** rather than **labels**. ([GitHub][3])
* **Compact multilingual pass (5 languages).** Use ID‑level gold alignment per language and compare **rank thresholds** and **tuned KL crossings**; flag deviations >2 layers in `first_rank_le_5` or >0.1 in normalized depth. This helps shift the debate toward language‑independent structure. ([GitHub][3])

### 4) Reporting changes that materially de‑risk interpretation

* When last‑layer KL≠0 (e.g., **Gemma**), **lead with ranks**; **demote absolute probabilities** in the narrative and footnote the temperature fit (`temp_est`, `kl_after_temp_bits`). ([GitHub][6])
* In every per‑model write‑up, **quote the raw‑vs‑norm summary** line (risk tier + `max_kl_norm_vs_raw_bits`) in the “Method sanity” block so readers automatically calibrate how much to trust early‑depth probabilities. ([GitHub][1])

---

## Appraisal of the current plan (PROJECT_NOTES.md)

The plan is unusually thoughtful and already aims at exactly the evidence the philosophical project needs:

* **Group 1 (measurement)** is complete and well designed; nothing to add.
* **Group 2 (variations)**: prioritize **permutation control** and the **multilingual pass**; both directly bear on token‑label vs relational structure and are cheaply compatible with the existing harness. ([GitHub][3])
* **Group 3 (causal)**: move **activation/attribution patching** and **head fingerprinting** ahead of high‑lift SAE work. They are cheap, decisive, and directly connect to “necessity/sufficiency” claims. **SAE work** is valuable but should follow once a single model has named heads and a portable vector to target. ([Neel Nanda][10])
* **Group 4 (philosophical consolidation)**: the proposed matrix of causal L_sem under style/paraphrase/language is exactly what’s needed to push past **austere nominalism**. For **metalinguistic‑nominalism vs realism**, the decisive additions will be **permutation robustness failures** and **vision→text bridge** results (even very modest positive results increase pressure on purely metalinguistic stories). ([GitHub][3])

---

## Closing: what this run has *securely* established vs what remains

* **Securely established (within‑model, correlational):**

  * Most models show **late rank‑1 collapse** for *Berlin*; **Gemma** is an outlier with **early copy reflex** and **calibration mismatch** under the lens. ([GitHub][6])
  * **Early directional alignment** with late calibration is common; Tuned‑Lens reduces mid‑depth KL accordingly. ([GitHub][2])
  * **Prism** acts as a robustness baseline but is **not** a calibrated head; regressions vs norm lens are expected and should be read as such. ([Neural Blogs][7])

* **Next to establish (causal/invariance):**

  * **Necessity** of specific layers/heads for the capital relation (activation/attribution patching). ([Neel Nanda][10])
  * **Portability** of a low‑rank **capital‑direction** across prompts (and, later, languages). ([arXiv][11])
  * **Failure under permutation controls**, indicating that internal objects latch onto relations, not labels. ([GitHub][3])

With these additions, the project will move from **suggestive depth curves** to **named, manipulable internal causes**—the kind of evidence that seriously constrains nominalist paraphrases while setting the stage for a sharper realism‑vs‑metalinguistic‑nominalism test.

---

### Key sources for methods and context

* *Tuned‑Lens* (Belrose et al., 2023): motivation, improved calibration of intermediate layers, causal basis extraction. ([arXiv][5])
* *RMSNorm* (Zhang & Sennrich, 2019): normalization details relevant to lens calibration. ([arXiv][9])
* *Logit Prisms* explainer/materials: shared/whitened decoders as robustness baselines. ([Neural Blogs][7])
* *Attribution/Activation patching*: scalable causal mapping for LLMs. ([Neel Nanda][10])
* SEP entry on **Nominalism** (for the philosophical frame). ([Stanford Encyclopedia of Philosophy][12])

---

### Pointers to the artifacts cited in this review

* **Script** and diagnostics fields (normalizer selection, temperature, copy detector config, raw‑vs‑norm sanity, control/ablation): `run.py`. ([GitHub][4])
* **Per‑model evaluations** and **JSON/CSV** for **Gemma‑2‑9B**, **Qwen3‑8B**, **Qwen2.5‑72B** (examples used above). ([GitHub][6])
* **Cross‑model synthesis** summarizing rank milestones, calibration status, and sidecar behavior across all ten models. ([GitHub][1])
* **Project plan** laying out measurement, causal passes, and philosophical roadmap. ([GitHub][3])

*All numeric claims in this review are drawn from the linked JSON/CSV/evaluation artifacts and are used strictly within‑model unless otherwise noted.*

[1]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-cross-models.md "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Qwen2.5-72B.md "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/PROJECT_NOTES.md "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run.py "raw.githubusercontent.com"
[5]: https://arxiv.org/abs/2303.08112?utm_source=chatgpt.com "Eliciting Latent Predictions from Transformers with the Tuned Lens"
[6]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-gemma-2-9b.md "raw.githubusercontent.com"
[7]: https://neuralblog.github.io/logit-prisms/?utm_source=chatgpt.com "Logit Prisms: Decomposing Transformer Outputs for ..."
[8]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Qwen3-8B.md "raw.githubusercontent.com"
[9]: https://arxiv.org/abs/1910.07467?utm_source=chatgpt.com "Root Mean Square Layer Normalization"
[10]: https://www.neelnanda.io/mechanistic-interpretability/attribution-patching?utm_source=chatgpt.com "Attribution Patching: Activation Patching At Industrial Scale"
[11]: https://arxiv.org/pdf/2303.08112?utm_source=chatgpt.com "Eliciting Latent Predictions from Transformers with the ..."
[12]: https://plato.stanford.edu/entries/nominalism-metaphysics/?utm_source=chatgpt.com "Nominalism in Metaphysics"
