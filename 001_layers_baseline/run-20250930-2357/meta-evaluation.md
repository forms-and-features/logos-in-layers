# Review of “Logos‑in‑Layers — Experiment 001”

## Executive summary

The experiment is thoughtfully designed and generally well‑executed. It standardizes a **norm‑aware logit‑lens** across multiple foundation models (Gemma‑2, Qwen, Llama‑3, Mistral, Yi), logs rich per‑layer metrics (probabilities, ranks, KL‑to‑final, cosine to final direction, coverage), and includes **sanity checks** (raw‑vs‑norm dual lens, last‑layer head calibration, negative controls, ablations). The team also explores **Tuned‑Lens** and a shared‑decoder **Prism** sidecar. Across models, the consistent picture is **late semantic collapse** with minimal early prompt‑copy reflex outside the Gemma family; **ranks improve gradually while KL to the final head stays large until late**, matching an “early direction, late calibration” story. These observations are supported throughout the run artifacts and per‑model reports. ([GitHub][1])

There are a few **correctable issues** and **interpretation risks**:

* A mis‑citation of **Tuned‑Lens** (wrong arXiv id in notes); the correct paper is Belrose et al., *Eliciting Latent Predictions from Transformers with the Tuned Lens* (arXiv:2303.08112). ([GitHub][2])
* The notes include a **PyTorch `kl_div` usage pattern** that—if copied literally—computes KL in the *reverse* direction by default; it should be double‑checked against the helper actually used. (Details below.) ([GitHub][1])
* **Prism** is consistently **regressive** in these runs (higher KL, no earlier rank milestones); it should be treated as diagnostic only, not as an alternative to model heads. ([GitHub][3])

Within the scope defined by the project’s guardrails (single prompt family, RMS‑lens caveats, residual‑only probes), the **findings are largely sound** and **well‑hedged** by the internal QA signals. The next iteration can raise evidential strength—especially for the realism/nominalism program—by adding **causal tests**, **prompt diversity**, **feature‑level analyses**, and **robustness checks** described at the end of this review.

---

## Methodology & code review

### What’s solid

* **Architecture‑aware normalization.** The run applies the *correct normalizer for the stream being decoded* (RMSNorm with ε inside √; γ applied after; post‑block snapshots choose LN2 for post‑norm or the next block’s LN1 for pre‑norm; `ln_final` at the end). This addresses a classic source of spurious “early semantics”. The details and provenance are laid out in the project notes and run diagnostics. ([GitHub][1])

  * RMSNorm/LayerNorm references are appropriate: Zhang & Sennrich (RMSNorm) and Pre‑LN vs Post‑LN behavior (Xiong et al.). ([arXiv][4])
* **Precision and head calibration.** Unembedding is promoted to fp32; last‑layer consistency compares the lens distribution to the model’s head and fits a scalar temperature (e.g., Gemma shows ∼1 bit residual KL that drops after a temperature fit, a known family‑specific calibration issue). ([GitHub][5])
* **Raw‑vs‑norm dual lens.** The baked‑in comparison (sampled windows and full summaries) is exactly the kind of QA that prevents over‑claiming. Qwen2.5‑72B and Gemma‑2 show high **norm‑only semantics** risk (extreme KL between norm and raw), while Llama‑3‑70B and Mistral‑Small‑24B look low/medium risk—this appropriately gates interpretation. ([GitHub][6])
* **Rank‑first reporting & within‑model comparisons.** The reports consistently prefer **ranks and KL thresholds** over absolute probabilities and limit cross‑family claims—exactly right given lens and head differences. ([GitHub][7])

### What needs correction or clarification

1. **Tuned‑Lens citation.** Notes attribute Tuned‑Lens to “Cai et al., 2023, arXiv:2303.17564” which is not the Tuned‑Lens paper; the correct reference is **Belrose et al. 2023**, arXiv:2303.08112 (with FAR AI page as secondary). Please fix the citation in `NOTES.md` / `001_LAYERS_BASELINE_PLAN.md` and any generated prompts. ([GitHub][2])

2. **KL orientation in the notes.** `001_LAYERS_BASELINE_PLAN.md §1.3` shows a snippet using:

```python
torch.kl_div(probs.log(), final_probs, reduction="sum")
```

By default (`log_target=False`), `torch.kl_div(input=logP, target=Q)` computes **KL(Q‖P)**, not **KL(P‖Q)**. The diagnostics repeatedly state the metric is **KL(P_layer‖P_final)**. If the production helper `layers_core.numerics.kl_bits` already implements the correct direction, great; otherwise, update it to:

```python
torch.sum(P_layer * (P_layer.log() - P_final.log()))
```

(or set `log_target=True` carefully). This won’t change the “goes to 0 at the final layer” property but can shift numeric values and threshold crossings. The doc should guarantee the intended direction. ([GitHub][1])

3. **Where to normalize post‑block streams (pre‑norm architectures).** The notes say “pre‑norm models use the next block’s LN1 for post‑block probes” (alternatively LN2 for post‑norm). This is a defensible convention, but it is a *modeling choice*, not a ground truth. Given known roles of LN for attention expressivity and gradient scaling, it would be helpful to **explicitly log** which normalizer was applied at each layer and provide a toggle to run **(a) no re‑normalization, (b) next‑block LN1, (c) tuned head**—then compare raw‑vs‑norm KL and rank milestones for all three. (Research context: LN placement materially changes dynamics and interpretability. ) ([Proceedings of Machine Learning Research][8])

4. **Cosine to final head: bias handling.** The notes say cosine is computed on **logit directions**. If any evaluated family uses a non‑zero unembed bias or soft‑cap, the doc should clarify whether cosine uses **(resid·W_U)** or **(resid·W_U + b)**. Consistent with Tuned‑Lens best practice, prefer bias‑free comparisons to avoid bias terms dominating direction. ([arXiv][9])

5. **Prism sidecar framing.** The reports already treat Prism as **diagnostic**. Given the consistent regressions (higher KL, null rank milestones) across families, explicitly mark Prism as **“for visualization/QA only”** in the README and suppress summary deltas unless they beat baseline under pre‑registered metrics. ([GitHub][7])

---

## Results & analysis review

### Claims that are well‑supported

* **Late semantic collapse is the norm.**
  – **Qwen2.5‑72B** reaches rank‑1 only at the **final layer (L80)** with last‑layer agreement near‑perfect but extreme **norm‑only semantics** risk (max KL(norm vs raw) ≈ 83 bits; tier=high). The report correctly hedges: rely on ranks; treat any pre‑final semantics as lens‑induced. ([GitHub][6])
  – **Gemma‑2‑9B** shows collapse at **L42/42** with strong head calibration mismatch (∼1 bit final KL, improved by temperature), and **strict copy at L0** (“simply”)—the one family with a clear bottom‑layer copy reflex under the current rule. ([GitHub][5])
  – **Mistral‑Small‑24B** collapses at **L33/40** with low lens‑artifact risk and clean final calibration. ([GitHub][3])
  – **Llama‑3‑70B** collapses unusually **early (L40/80)** with medium lens‑artifact risk near the top; the report uses rank milestones and is careful about norm‑only late layers. ([GitHub][10])
  – **Mistral‑7B‑v0.1**: **L25/32**, tuned lens shifts rank milestones later and improves KL within model; raw‑vs‑norm risk high enough to gate early semantics claims. ([GitHub][11])

* **Negative controls and ablations behave sensibly.** The “Berlin→country” control is clean (Germany on top) with the right mass order; removing “simply” tends to **leave L_sem unchanged or move it slightly earlier**, not later—consistent with the filler acting as surface anchor rather than supplying content. ([GitHub][3])

* **Prism regressions.** For Gemma, Qwen2.5‑72B, Llama‑3‑70B, and Mistral variants, Prism dilates KL at percentiles and fails to produce earlier rank milestones. The documents state this and do not over‑sell Prism. ([GitHub][3])

* **Measurement guidance is applied.** Where raw‑vs‑norm risk is high, the write‑ups consistently **de‑emphasize absolute probabilities** and **prefer ranks** or **confirmed semantics** sources (raw/tuned). ([GitHub][7])

### Places to tighten wording or avoid over‑reach

* **Cross‑family generalizations.** The cross‑model memo is careful, but a few passages could be read as implying weak correlations between **earliness** and **capability**. The text already tempers this (“use within‑family trends only”); keep that language prominent and avoid any suggestion of cross‑family monotonicity (Qwen2.5‑72B is a counterexample: highest capability yet latest collapse). ([GitHub][7])
* **Copy reflex conclusions.** The result “copy shows only in Gemma” is **true under the current strict rule** (k=1, τ=0.95, ignoring whitespace/punct). It should be explicitly caveated as **operational**: modest changes in the prompt or thresholds can change counts, per the accepted limitations. (The write‑ups mostly do this; keep the caveat in the top‑level summary.) ([GitHub][1])

---

## Independent deep‑dive synthesis (across the provided runs)

**Pattern 1 — Early direction, late calibration.** In Llama‑3‑70B and Mistral‑24B, the cosine between intermediate and final *logit directions* rises well before KL‑to‑final drops, and **rank milestones** improve earlier than probabilities become confident. This fits with a picture where the model finds the **right direction** mid‑stack and **amplifies** it late; Tuned‑Lens improves this picture by learning per‑layer linear maps. This is aligned with the Tuned‑Lens literature, which shows tuned probes are more predictive than raw logit‑lens and better match the model’s own features. ([GitHub][3])

**Pattern 2 — Surface→format→meaning.** Gemma’s trajectory—bottom‑layer copy of the filler “simply”, then quotes/punctuation in the late mid‑stack, then a sharp final collapse to “Berlin”—nicely illustrates the **surface anchoring and formatting tokens** preceding semantics. Similar but weaker formatting dominance occurs in Qwen2.5‑72B (quotes dominating late layers) before the very late semantic flip. This is consistent with Pre‑LN dynamics and the known roles of normalization in attention expressivity and representation scaling. ([GitHub][5])

**Pattern 3 — Family idiosyncrasies matter.**

* **Gemma**: non‑zero last‑layer KL to the head and bottom‑copy reflex; treat absolute probs cautiously.
* **Qwen**: extreme norm‑vs‑raw divergence (especially at the top), making norm‑only early semantics suspicious.
* **Llama‑3**: relatively early collapse at 70B (mid‑depth), late at 8B.
* **Mistral**: clean final calibration; tuned lens can *delay* le_1 slightly while improving KL/rotation.
  The reports generally hew to these family traits without over‑generalizing. ([GitHub][5])

**What this *doesn’t* show (yet):** With a **single prompt family**, these runs cannot resolve whether the observed phenomena are **tokenizer‑ and phrasing‑specific** or robust across semantic domains. The project acknowledges this limitation; the next iteration should widen the probe set to test stability of L_copy/L_sem and the ∆ dynamics. ([GitHub][1])

---

## Relevance to the realism vs nominalism program

The project’s stated near‑term target is **austere nominalism**—the claim that there are only word tokens and no robust internal structures to speak of. Even within the narrow initial scope, these runs provide **incipient pressure against the austere view**:

* The **repeatable layer‑wise trajectories**—surface anchoring → formatting → semantic collapse—with consistent **rank milestones** and **cosine‑to‑final trends** across models indicate **structured, reusable intermediate representations**, not mere parroting of surface n‑grams. ([GitHub][7])
* The **tuned‑lens** improvements and raw‑vs‑norm checks together argue that **intermediate residuals encode semantic directionality** that can be **linearly decoded** with appropriate calibration—again suggesting latent structure beyond surface strings. ([arXiv][9])

At the same time, **lens artifacts** (Qwen/Gemma norm‑only semantics) warn against over‑interpreting “early meaning”. Until causal tests (e.g., path patching) show that **perturbing a specific intermediate feature causally changes the downstream answer**, the evidence remains **suggestive**, not decisive, for the broader **realism vs. metalinguistic nominalism** question. The measured discipline around ranks/KL and the plan to add causal probes are the right way to sharpen this. ([GitHub][6])

---

## Plan review and concrete next steps

The current plan already emphasizes **measurement hygiene** (§1.* of the project notes). The following **additions/adjustments** would add non‑negligible value for the project’s goals:

1. **Lock down KL direction and document it.**
   *Action.* Ensure `layers_core.numerics.kl_bits` computes **KL(P_layer‖P_final)**; add a unit test comparing against a manual sum. Update docs to eliminate any ambiguity created by the illustrative snippet in §1.3.
   *Value.* Removes a subtle source of interpretive drift in threshold‑based milestones. ([GitHub][1])

2. **Broaden prompts with minimal variance axes.**
   *Action.* Keep the same answer (“Berlin”) but vary phrasing along **three orthogonal axes**: syntactic template (declarative vs interrogative), filler tokens (remove/replace “simply” with synonyms), and **language** (e.g., German prompt with English answer). Report **stability of L_copy, L_sem, ∆̂**, and **rank milestones** under these controlled perturbations.
   *Value.* Directly tests whether trajectories reflect **meaning**, not **tokenizer quirks**—a key step toward the realism program. (The plan already flags over‑fitting risk; this implements the fix.) ([GitHub][1])

3. **Causal interventions (minimum viable).**
   *Action.* Implement **path patching** on the residual stream at layers just before/after L_semantic, swapping activation slices between “Germany→Berlin” and a matched control (“France→Paris”), measuring the **causal effect** on the NEXT token’s distribution.
   *Value.* Converts correlational collapse timing into **causal evidence** that specific intermediate features **carry semantic content**. This connects directly to the realism question and leverages Transformer‑Circuits style methodology. ([Transformer Circuits][12])

4. **Feature probing beyond the answer token.**
   *Action.* Add **feature‑space probes** for attributes orthogonal to the target (e.g., named‑entity class, country‑capital relation) using **linear probes** on the residual stream with **frozen heads**; report where such features peak relative to L_sem.
   *Value.* If abstract features **peak earlier** than answer selection, that strengthens a **realist** reading (latent universals) over mere surface anchoring.

5. **Normalize normalization choices.**
   *Action.* For pre‑norm decoders, run **three variants** side‑by‑side on a subset: *(a)* raw residuals, *(b)* next‑block LN1, *(c)* tuned head. Log **raw‑vs‑norm KL** and **rank milestones** deltas.
   *Value.* Separates **model structure** from **lens convention**, reducing measurement dependence on the chosen normalizer. (Relevant literature underscores that LN placement/roles matter deeply.) ([Proceedings of Machine Learning Research][8])

6. **Prism gating.**
   *Action.* Given consistent regressions, mark Prism as **experimental**; require **pre‑registered criteria** (e.g., earlier le_5 with ≤ KL inflation) before including Prism numbers in summaries.
   *Value.* Prevents accidental over‑weighting of a shared decoder that currently under‑calibrates. ([GitHub][7])

7. **Document family‑specific head transforms.**
   *Action.* Where `warn_high_last_layer_kl` is true (Gemma), keep the **temperature fit** in the JSON but **suppress absolute probability talk** in the markdown; foreground **ranks** and **KL thresholds**.
   *Value.* Avoids cross‑family misinterpretations caused by head calibration differences. ([GitHub][5])

8. **Add a small panel of relation prompts.**
   *Action.* Re‑run the minimal panel (e.g., 8–12 prompts) across the same models: capitals (x3), superlatives (“largest country by area”), simple facts (“2+2”), and one **negated** variant.
   *Value.* Tests whether **∆̂** and **rank milestone shapes** generalize beyond a single semantic relation, moving the evidence closer to **reusable structure** rather than a one‑off pattern.

---

## Alignment with current literature

The measurement choices (rank‑first, KL‑to‑final, cosine to final direction) and the decision to integrate **Tuned‑Lens** are strongly aligned with the state of the art: Tuned‑Lens improves reliability over a brittle raw logit‑lens by learning per‑layer linear probes; LayerNorm/RMSNorm placement and calibration are known to affect interpretability; and the **Transformer‑Circuits** style focus on causal tests is the right next step to move from correlation to mechanism. ([arXiv][9])

---

## Bottom line

*Methodologically,* the experiment is on the right track, with commendable QA and careful reporting. *Empirically,* the runs support a picture of **structured internal computation**: surface/form signals dominate early, **semantic direction emerges mid‑stack**, and **decisive calibration happens late**—with family‑specific twists. That is **evidence against austere nominalism** and a foundation for stronger, **causal** evidence in the next iteration. The concrete fixes and additions above would tighten measurement, reduce lens artifacts, and materially advance the realism/nominalism inquiry. ([GitHub][7])

---

### Sources

* Project notes and diagnostics (methods, QA, guardrails). ([GitHub][1])
* Cross‑model synthesis. ([GitHub][7])
* Per‑model reports: Mistral‑24B, Qwen2.5‑72B, Llama‑3‑70B, Mistral‑7B‑v0.1, Gemma‑2‑9B. ([GitHub][3])
* Tuned‑Lens (Belrose et al., 2023). ([arXiv][13])
* RMSNorm (Zhang & Sennrich, 2019). ([arXiv][4])
* LayerNorm placement and roles (Xiong et al., 2020; Brody et al., 2023). ([Proceedings of Machine Learning Research][8])
* Transformer‑Circuits program. ([Transformer Circuits][14])

[1]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_LAYERS_BASELINE_PLAN.md "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/NOTES.md "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Mistral-Small-24B-Base-2501.md "raw.githubusercontent.com"
[4]: https://arxiv.org/pdf/1910.07467?utm_source=chatgpt.com "Root Mean Square Layer Normalization"
[5]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-gemma-2-9b.md "raw.githubusercontent.com"
[6]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Qwen2.5-72B.md "raw.githubusercontent.com"
[7]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-cross-models.md "raw.githubusercontent.com"
[8]: https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf?utm_source=chatgpt.com "On Layer Normalization in the Transformer Architecture"
[9]: https://arxiv.org/pdf/2303.08112?utm_source=chatgpt.com "Eliciting Latent Predictions from Transformers with the ..."
[10]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Meta-Llama-3-70B.md "raw.githubusercontent.com"
[11]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Mistral-7B-v0.1.md "raw.githubusercontent.com"
[12]: https://transformer-circuits.pub/2021/framework/index.html?utm_source=chatgpt.com "A Mathematical Framework for Transformer Circuits"
[13]: https://arxiv.org/abs/2303.08112?utm_source=chatgpt.com "Eliciting Latent Predictions from Transformers with the Tuned Lens"
[14]: https://transformer-circuits.pub/?utm_source=chatgpt.com "Transformer Circuits Thread"
