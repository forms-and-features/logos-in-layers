# Interpretability Project – Development Notes for AI Assistant

## Philosophical Project Context

**Goal.** Bring concrete interpretability data to the centuries‑old dispute between **nominalism** and **realism** about universals. The first iterations target the low bar: **austere (extreme) nominalism**, which claims that only particular tokens exist and any talk of properties or relations can be paraphrased away.

By showing that LLMs contain robust, reusable internal structures, detected through logit‑lens baselines and causal patches, we aim to gather empirical pressure against that austere view. Once those methods are sound and the anti‑austere evidence is in hand, the project will move to the harder task of discriminating between **metalinguistic nominalism** (which treats those structures as facts about words and predicates) and **realism** (which treats them as evidence of mind‑independent universals).

## Part 2 — Observational Battery & Lightweight Controls

> **Goal.** Establish robust, multi‑item empirical baselines for copy vs. semantics, collapse depth, and lens reliability **within each model**, under strict measurement gating from Part 1. No causal interventions here—only observational probing across many prompts, paraphrases, languages, and lightweight controls.

---

### 2.0. Scope & Gating (applies to all of Part 2)

**Why.** Ensures all observational results are comparable across items and models, and prevents over‑interpretation in high‑artefact regimes.

**What.**

* **Lens selection per model**: use `measurement_guidance.preferred_lens_for_reporting` (from Part 1); if `tuned_is_calibration_only=true`, use **norm lens** for semantic milestones and treat tuned as calibration‑only.
* **Artefact tier gating**: if `risk_tier="high"` or `n_norm_only_semantics_layers>0` near candidate layers, restrict to **rank milestones**, **KL thresholds**, and **confirmed semantics**; avoid absolute probabilities.
* **Confirmed semantics**: when available, prefer `L_semantic_confirmed` (with `confirmed_source`), else report `L_semantic_norm` with an artefact caution.
* **Consistency across items**: for each model, probe exactly the same item set, paraphrases, and controls; record any tokenization‑induced exclusions.

**How.**

* Runner reads gating flags from Part 1 JSON; applies per‑model lens selection and reporting restrictions automatically.
* All metrics reported as **within‑model** normalized depths: `frac = L / n_layers`.

---

### 2.1. Rank‑Centric Prompt Battery (primary observational set)

**Why.** Single‑prompt brittleness is the main risk; a sizable, diverse battery enables robust distributions for collapse depth and copy reflex.

**What.**

* **Relations**: `capital_of`, `currency_of`, `official_language_of`.
* **Entities**: for each relation, sample **200** subject entities (stratified by global frequency and length) yielding ~600 items.
* **Canonical prompts** (EN):

  * Positive: “Give the [answer‑type] only, plain text. The capital of {COUNTRY} is called simply” (or relation‑specific template).
  * Control: “Give the [answer‑type] only … The capital of **France** is called simply” (when subject ≠ France).
  * **No‑filler** ablation: drop “simply”.
* **Outputs per item**: `L_copy_strict` (if any), earliest `L_copy_soft[k]`, `L_semantic_norm`, `L_semantic_confirmed` (if present), `Δ̂ = (L_sem − L_copy_variant)/n_layers`, `first_rank_le_{10,5,1}`, `first_kl_below_{1.0,0.5}`, entropy gap percentiles, and artefact tier.

**How.**

* Generate items from a static CSV of `(relation, subject, gold_answer_string)`; pre‑compute `answer_ids` with gold‑token alignment.
* Run with cached residuals; produce per‑item pure CSVs and a **battery roll‑up** (see 2.7).

---

### 2.2. Multilingual (preliminary observational pass)

**Why.** Tests linguistic invariance and tokenization confounds with minimal lift.

**What.**

* Languages: **English, German, French, Spanish, Chinese (simplified)**.
* For each relation/entity, generate **1 prompt per language** using literal templates (no idioms).
* Record `gold_alignment_rate` per language; drop items with unresolved alignment.

**How.**

* Reuse 2.1 item list; plug language‑specific templates.
* Add language tag in CSVs; summarize per‑language distributions of `L_semantic_frac`, `Δ̂`, and artefact tier.

---

### 2.3. Paraphrase Robustness (observational)

**Why.** Ensures that collapse depth and rank milestones are not template‑specific.

**What.**

* For each item (EN only), generate **10 paraphrases** (systematic rephrasings: passive voice, alternate slot syntax, synonym of “capital”, punctuation variants).
* Compute per‑item dispersion: `IQR(L_semantic_frac)`, `max−min` over 10 paraphrases; same for `Δ̂`.

**How.**

* Use a deterministic paraphrase generator (template permutations, not LLM‑generated text); record a `paraphrase_id`.
* Report **per‑model** median dispersion and the share of items with `IQR(L_semantic_frac) ≤ 0.05`.

---

### 2.4. Predicate‑Permutation Control (Quine guard; observational)

**Why.** Checks that the model’s behavior is truly predicate‑sensitive and not just co‑occurrence.

**What.**

* Build a **permuted mapping** of answers within the item set (e.g., shuffle capitals among countries).
* Prompts remain unchanged; evaluate whether the **permuted gold** ranks rise spuriously.

**How.**

* For each batch (e.g., 50 items), construct a derangement; compute `perm_rank@L_semantic` and `perm_topk@L_semantic`.
* Report per‑model: fraction of items where `perm_rank ≤ 10` (should be near chance).

---

### 2.5. Negation Control (observational)

**Why.** Distinguishes propositional content tracking from surface co‑occurrence.

**What.**

* Add: “The capital of {COUNTRY} is **not** called simply …”
* Track `negation_margin_layer[l] = p_ans(nonneg,l) − p_ans(neg,l)`; summarize `first_negation_respected_pos` and `max_negation_margin`.

**How.**

* Compute on the same item set; report distributions across items/models.

---

### 2.6. Position & Filler Perturbations (light OOD; observational)

**Why.** Tests positional generalization and filler sensitivity without leaving the language domain.

**What.**

* Insert neutral tokens before the answer slot to shift **next‑token position** by +{1,3,7}.
* Add filler variants: remove/add “simply”, add comma or colon, vary whitespace.
* Track changes in `L_semantic`, `Δ̂`, and `pos_ood_gap` (reusing §1.43 positional audit logic per prompt).

**How.**

* Deterministic variants; one pass per variant; aggregate per‑model median deltas.

---

### 2.7. Battery Reporting & Outputs

**Why.** Standardized roll‑ups enable model‑level comparisons and LLM‑assisted evaluations.

**What.**

* **Per‑item CSV**: `battery-<MODEL>-<REL>-<LANG>-records.csv` (one row per layer per item).
* **Per‑item summary CSV**: `battery-<MODEL>-items.csv` with columns:
  `relation,subject,lang,paraphrase_id,L_copy_strict,L_copy_soft_k,copy_variant,L_semantic_norm,L_semantic_confirmed,confirmed_source,delta_hat,first_rank_le_10,5,1,first_kl_below_1.0,0.5,negation_first_pos,negation_max_margin,perm_rank,perm_topk,artefact_tier,js_p50,l1_p50,jaccard_p50,entropy_gap_p50,repeatability_flag`.
* **Model roll‑up JSON**: `battery-<MODEL>-summary.json` with:

  * Distributions (median, IQR, p10/p90) of `L_semantic_frac`, `Δ̂`, `first_rank_le_{1,5,10}/n_layers`.
  * Paraphrase dispersion stats (% items with `IQR ≤ 0.05`).
  * Language‑wise stats (means/medians per language).
  * Negation & permutation success rates.
  * Positional perturbation deltas.
  * Counts by artefact tier and by `confirmed_source`.
  * A `battery_evaluation_pack` (Part 1.44‑style) to support LLM evaluators.

**How.**

* Implement a batch runner that iterates (relation, subject, lang, paraphrase_id, variant) with caching; export the above CSV/JSON artifacts per model.

---

## Part 3 — Causal Localization & Mechanistic Tests

> **Goal.** Identify **where** and **how** the relation is computed: which layers, sublayers, and heads are **sufficient** (and, when possible, **necessary**) for the behavior, and whether a compact **concept direction** can causally control the output.

---

### 3.0. Scope & Gating

**Why.** Ensures causal conclusions build on reliable observables.

**What.**

* Apply Part 2 artefact gating: prioritize items with **confirmed semantics** or low artefact tier; skip items where Part 2 shows unstable ranks (`repeatability_flag` or high paraphrase dispersion).
* Use `preferred_lens_for_reporting` for decoding when needed (for diagnostics only; causal metrics are lens‑agnostic).

**How.**

* Pre‑select **50 items per relation** per model (stratified from Part 2) for causal tests.

---

### 3.1. Activation Patching (clean↔corrupted)

**Why.** Standard causal probe for **sufficiency** of layer/sublayer activations.

**What.**

* **Clean**: “The capital of **Germany** is called simply …”
* **Corrupted**: swap subject with a confounder (e.g., **France**).
* **Patch**: copy activations from clean → corrupted at layer ℓ for **(a) post‑attention**, **(b) post‑MLP**, and **(c) post‑block**.
* **Metrics per ℓ**:

  * **Logit difference** `Δ_logit = logit_ans(patched) − logit_ans(corrupted)`
  * **Δ log‑prob (bits)** for the gold answer
  * **Rank flip** (does answer become rank‑1?)
  * **Causal L_sem**: earliest ℓ where (b) or (c) yields **rank‑1** and `Δ log‑prob ≥ 0.5` bits.
* **Success criteria per item**: causal `L_sem` within **±1 layer** of observational `L_semantic_confirmed` (or `L_semantic_norm` if not confirmed).

**How.**

* Implement token‑aligned patching for the **answer‑slot step**; verify shape/position.
* Record per‑ℓ results to `patch-<MODEL>-<REL>-items.csv` and a model‑level JSON summary.

---

### 3.2. Sublayer Decomposition & Onset (ATTN vs MLP)

**Why.** Distinguishes factual retrieval vs. composition pathways.

**What.**

* From 3.1, extract:

  * `causal_L_sem_mlp` = earliest ℓ where **post‑MLP** patch suffices.
  * `causal_L_sem_attn` = earliest ℓ where **post‑Attention** patch suffices.
* Compute normalized depths and compare with observational `L_sem`.

**How.**

* Persist to JSON: `{ causal_L_sem_mlp, causal_L_sem_attn, causal_L_sem_postblock }` and normalized fractions; aggregate by relation and model.

---

### 3.3. Head Fingerprinting (retrieval & routing)

**Why.** Localizes which attention heads enable the relation.

**What.**

* For a window around `L_sem` (e.g., ℓ ∈ [L_sem−2, L_sem+2]):

  * **Ablate** heads one‑by‑one (zero output or replace with corrupted).
  * **Patch** heads (copy from clean).
* **Metrics**: Δ log‑prob bits on the gold answer, and rank flips.
* **Outputs**: per‑head impact scores, top‑k critical heads per model/relation.

**How.**

* Save per‑head CSVs: `heads-<MODEL>-<REL>-ℓ.csv`; model summary with ranked head IDs and cumulative ablation curves.

---

### 3.4. Concept Vector Extraction (linear direction)

**Why.** Tests whether a **low‑rank direction** captures the relation and can be used for causal control.

**What.**

* Learn a linear **concept direction** `v_rel` at a chosen **anchor layer** (e.g., `median(causal_L_sem_mlp)`):

  * From training pairs (e.g., 150 (country, capital)), fit a logistic or linear classifier to predict the first answer token’s logit difference; L2‑regularized.
* **Portability tests**:

  * **Held‑out items**: add/remove α·`v_rel` (projection add/remove) at the answer‑slot step; measure causal control.
  * **Paraphrases** & **languages**: reuse the same `v_rel` to test invariance.

**How.**

* Derive `v_rel` only from items passing gating; store to disk with provenance.
* CSV: `concept-<MODEL>-<REL>-alpha_sweep.csv`; JSON: success rates vs. α, off‑target effects (see 3.7).

---

### 3.5. Projection Surgery (necessity‑leaning test)

**Why.** Tests whether removing the concept subspace disrupts behavior.

**What.**

* Build a projector `P = I − v_rel v_relᵀ` (or small subspace from top‑k directions).
* Apply `P` to residuals at the anchor layer; measure drop in gold‑answer logit vs. clean.

**How.**

* Report per‑item Δ log‑prob bits and rank flips; aggregate necessity‑leaning evidence (acknowledging imperfect necessity).

---

### 3.6. Causal Mediation (optional pilot)

**Why.** Explores whether identified subspaces mediate the effect of earlier components.

**What.**

* Fit a simple causal mediation using **patched** vs. **projected** runs to estimate indirect effects via `v_rel`.

**How.**

* Run on a 20‑item pilot; include in an **optional** appendix.

---

### 3.7. Side‑Effects & Specificity (steering safety)

**Why.** Ensure concept control does not cause broad misbehavior.

**What.**

* Measure **off‑target** token mass changes (top‑50) and **perplexity** drift at the answer position.
* Define **Specificity Index** = (Δ log‑prob on gold) / (L1 mass change on non‑gold top‑50). High is better.

**How.**

* Compute during α‑sweeps in 3.4/3.5; log to `concept-…-alpha_sweep.csv`.

---

### 3.8. Outputs & Success Criteria

**What.**

* JSON `causal-<MODEL>-summary.json`: medians/IQRs of `causal_L_sem_*` (normalized), head impact distributions, concept vector control rates, specificity, projection surgery effects.
* Success if:

  * `median(|causal_L_sem_postblock − L_semantic_confirmed|) ≤ 1 layer` (or vs. `L_semantic_norm` if unconfirmed).
  * ≥70% items show **rank‑1** after **post‑MLP** patch at or before `L_semantic_confirmed`.
  * Concept vector yields ≥0.5 bit median **Δ log‑prob** on held‑out items with **Specificity Index ≥ 1.5**.

---

## Part 4 — Invariance & Philosophical Stress‑Tests (Language‑Internal)

> **Goal.** Test whether the identified internal structure is **predicate‑like** and **invariant** across paraphrases, languages, roles, and controls—evidence against austere nominalism and toward structured internal representations.

---

### 4.1. Paraphrase Invariance (application of Part 2.3)

**Why.** Predicate identity should not depend on wording.

**What.**

* On the 10 paraphrases per item (Part 2.3), compute:

  * **Invariance Index** = 1 − normalized IQR of `L_semantic_frac` (0=no invariance; 1=perfect).
  * Percent of items with **confirmed semantics** unchanged (±1 layer) across paraphrases.
* Optional causal check: does **post‑MLP** patch at the anchor layer generalize across paraphrases?

**How.**

* Merge Part 2.3 summaries; for causal checks, patch on a subset and report rates.

---

### 4.2. Multilingual Consistency (application of Part 2.2)

**Why.** If the relation is language‑independent, onset and control should be stable across languages.

**What.**

* Compare per‑language distributions of `L_semantic_frac`, `Δ̂`.
* Cross‑lingual **patching**: patch the **post‑MLP** activation from EN clean into other languages’ corrupted prompts; measure recovery.

**How.**

* Use consistent token‑aligned positions; restrict to items with `gold_alignment_rate=1.0` in both languages.

---

### 4.3. Role‑Asymmetry & Directionality

**Why.** Tests if the model encodes **capital_of** direction, not a symmetric association.

**What.**

* Prompts reversing roles: “{CAPITAL} is the capital of …” and “{COUNTRY} is the capital of …” (incorrect role).
* Track whether semantics onset occurs for the **correct** direction only; compute **Asymmetry Score** = onset(correct) − onset(incorrect).

**How.**

* Run on a 100‑item subset per model; aggregate Asymmetry Scores.

---

### 4.4. Predicate‑Permutation (Quine guard; application of Part 2.4)

**Why.** Ensures behavior isn’t driven by simple co‑occurrence.

**What.**

* Report the **Permutation Leakage Rate**: fraction of items where permuted answers enter top‑10 at `L_semantic_confirmed`.

**How.**

* Reuse Part 2.4 results; emphasize confirmed semantics only.

---

### 4.5. Frequency‑Controlled Analysis

**Why.** Ensure effects are not explained by lexical frequency.

**What.**

* Bin items by subject and answer frequency (e.g., corpus percentiles).
* Within bins, compare `L_semantic_frac` and causal success rates; regress out frequency to estimate residual variance attributed to structure.

**How.**

* Use a static frequency table (e.g., Wikipedia counts or training corpus metadata); record bin per item.

---

### 4.6. Novelty & Fictional Entities

**Why.** Test extrapolation beyond memorized co‑occurrences.

**What.**

* Construct **fictional countries** with assigned fictional capitals; ensure names are unseen or rare.
* Probes: measure whether concept direction `v_rel` (from 3.4) can steer the model toward the assigned capital when the base prompt gives no evidence.

**How.**

* Limit to observational plus concept‑vector steering; do not treat success as proof of understanding—report as **portability** evidence.

---

### 4.7. Pressure Tally & Reporting

**What.**

* Per model, produce `invariance-<MODEL>-summary.json` with:

  * Paraphrase Invariance Index distribution; % unchanged confirmed semantics (±1 layer).
  * Multilingual onset alignment (median absolute difference across languages).
  * Role‑Asymmetry distribution; Permutation Leakage Rate; Negation respect rate.
  * Frequency‑controlled residuals; Novelty steering outcomes.
* A model‑level **Austere‑Nominalism Pressure Score** (composite of above, weights documented), used **only** for qualitative ranking.

**How.**

* Combine Part 2 and Part 3 artifacts; compute indices and export.

---

## Part 5 — Cross‑Modal & Cross‑Model Universality

> **Goal.** Test whether the learned relation is **language‑independent** and **model‑portable**—moving beyond metalinguistic nominalism toward realism‑leaning evidence—while keeping compute low.

---

### 5.1. Vision→LM Linear Bridge (primary cross‑modal probe; no fine‑tuning)

**Why.** Tests if a text‑learned concept direction transfers to non‑text inputs without training the LLM.

**What.**

* Build a frozen **image encoder** (e.g., CLIP‑like) to embed **flag** or **map** images for countries in the battery.
* Learn a **linear mapping** (ridge regression) from image embeddings into the LLM’s **anchor layer residual space** (same anchor as 3.4).
* At the answer‑slot step, **inject** the mapped vector (scaled α) and test whether it increases the correct answer’s log‑prob on **image→text** trials.

**How.**

* Train the linear map on **seen** items; evaluate on **held‑out** countries; record Δ log‑prob, rank flips, and specificity (3.7).
* Outputs: `vision_bridge-<MODEL>-summary.json` and `…-alpha_sweep.csv`.
* Success threshold: median **Δ ≥ 0.5** bits and Specificity ≥ 1.5 on held‑out items.

---

### 5.2. Cross‑Model Vector Portability (optional, low‑lift variant)

**Why.** Tests whether concept directions align across models without heavy training.

**What.**

* For two models A,B of similar size/family:

  * Learn `v_rel^A` at A’s anchor layer; using a small **Procrustes** alignment (computed on neutral corpus activations), map A’s residual subspace onto B’s.
  * Inject mapped `v_rel^A` into B; evaluate control on B’s held‑out items.

**How.**

* Limit to **one** pair per family due to compute; export `cross_model-<A>-<B>-summary.json`.

---

### 5.3. Synthetic Mini‑Language Swap (deferred, heavier)

**Why.** Stronger pressure on metalinguistic nominalism via symbol remapping.

**What.**

* Create a **synthetic lexicon** for countries/capitals; remap tokens; evaluate whether `v_rel` and patching continue to work under remapping.

**How.**

* Run as a small **pilot** on one 7–9B model; defer large‑scale runs.

---

### 5.4. Cross‑Modal/Model Reporting

**What.**

* For each model, export `universality-<MODEL>-summary.json` with:

  * Vision bridge outcomes (held‑out Δ bits, specificity).
  * Cross‑model portability (if attempted).
  * Overall **Universality Index** (documented weighting).
* Aggregate report across models ranking universality evidence qualitatively.

**How.**

* Collate from 5.1–5.3 artifacts; include gating notes (e.g., which items/layers were used).

---

### 5.5. Compute & Sequencing Notes (practical)

**What.**

* Primary compute is in Part 2 battery and Part 3 patching; keep 70B runs **sampling‑only** (e.g., 25% of items) for Parts 2–3; full battery on 7–27B models.
* Part 5 runs only on **1–2 models** initially (best‑behaved by artefact tier and invariance score).
* Maintain deterministic seeds and cache residuals wherever possible.

**How.**

* Provide run manifests (`manifest-*.json`) that specify item subsets per model/phase; store to repo for reproducibility.

---

### References

* SEP‑Nominalism — **“Nominalism in Metaphysics,”** *Stanford Encyclopedia of Philosophy* (2023).
* SEP‑Tropes — **“Tropes,”** *Stanford Encyclopedia of Philosophy* (2023).
* SEP‑Fictionalism — **“Fictionalism,”** *Stanford Encyclopedia of Philosophy* (2021).
* Loux‑2023 — Michael J. Loux, *Metaphysics*, 4th ed., Routledge (2023).
* Brandom‑2000 — Robert B. Brandom, *Articulating Reasons: An Introduction to Inferentialism*, Harvard UP (2000).
* RMSNorm — Zhang & Sennrich, “Root Mean Square Layer Normalization,” arXiv:1910.07467 (2019).
* RoFormer — Su et al., “RoFormer: Enhanced Transformer with Rotary Position Embedding,” arXiv:2104.09864 (2021).
* Superposition — Elhage et al., “Toy Models of Superposition,” arXiv:2209.10652 (2022).
* FFN as KV — Geva et al., “Transformer Feed‑Forward Layers Are Key‑Value Memories,” arXiv:2012.14913 (2020).
* Tuned Lens — Belrose et al., “Eliciting Latent Predictions from Transformers with the Tuned Lens,” arXiv:2303.08112 (2023).
* Logit Prisms — “Logit Prisms: Decomposing Transformer Outputs for Mechanistic …” (implementation blog), https://neuralblog.github.io/logit-prisms/.
* SAE Absorption — Chanin et al., “A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders,” arXiv:2409.14507 (2024).
* Dictionary Learning (Othello‑GPT) — He et al., “Dictionary Learning Improves Patch‑Free Circuit Discovery in Mechanistic Interpretability: A Case Study on Othello‑GPT,” arXiv:2402.12201 (2024).
* Monet — Park et al., “Monet: Mixture of Monosemantic Experts for Transformers,” arXiv:2412.04139 (2024).

---

# Audience

* Software engineer, growing ML interpretability knowledge
* No formal ML background but learning through implementation
* Prefers systematic, reproducible experiments with clear documentation
