
# Review of “Logos‑in‑Layers / 001_layers_baseline”

## Executive summary

* **Overall**: The project implements a careful, architecture‑aware logit‑lens baseline across ten open‑weight base LLMs, with RMS‑lens normalization, depth‑by‑depth metrics (ranks/KL/entropy/cosine), calibrated “semantic onset” gates (uniform‑baseline, top‑2 margin, and run‑of‑two), and sidecars for **tuned‑lens** and **logit‑prism** checks. The measurement guidance embedded in each run correctly warns against over‑interpreting absolute probabilities when last‑layer KL is non‑zero or raw‑vs‑norm divergence is high. ([GitHub][1])

* **Findings (cross‑model)**: Emergence of rank‑1 semantics occurs **mid‑depth** for Llama‑3‑70B (~50% of layers), **late** for Mistral‑7B/24B, Llama‑3‑8B, Qwen3‑8B/14B, **very late / last layer** for Gemma‑2‑9B/27B and Qwen2.5‑72B. Copy‑reflex (early top‑1 prompt echo) appears in **Gemma** but not the other families. Raw‑vs‑norm lens artifact risk is **low** for Llama‑3‑70B and **high** for several Qwen3 and Yi runs; “norm‑only semantics” flags cluster at the very end for Gemma and Qwen2.5‑72B. Prism is **helpful** mid‑depth for Gemma‑2‑27B, **mixed** for Qwen2.5‑72B (early helpful, late regressive), and **regressive** for Qwen3/Llama‑3‑8B/Mistral‑7B. ([GitHub][2])

* **Interpretation quality**: Per‑model and cross‑model evaluation files mostly follow the project’s own guidance—favoring **ranks/KL** over absolute probabilities and flagging weak/near‑uniform onsets—but the prism verdicts should be nuanced for Qwen2.5‑72B (early/mid KL reductions) rather than labeled simply “regressive.” ([GitHub][2])

* **Philosophical use**: As a first iteration, the results give **empirical pressure** against an *austere* nominalism about internal representations (there are robust, layer‑localized invariants for a simple relation across diverse models), but they **do not** adjudicate the metaphysical existence of mind‑independent universals. Claims should be framed as evidence of **shared representational structure** in these LMs, not as proof of universals in the world. ([Stanford Encyclopedia of Philosophy][3])

---

## 1) Code & measurement pipeline: what looks right, and what merits adjustment

**Architecture‑aware normalization is a major strength.** All evaluated checkpoints are pre‑RMSNorm models; the runner applies a **true RMS lens** (ε inside √, fp32 stats, γ handling), uses **next block ln1** for post‑block normalization in pre‑norm architectures, and resorts to `ln_final` at the end. This matches best practice for decoding the residual stream and avoids common sources of logit‑lens brittleness. ([GitHub][4])

**Gating and guidance are well designed.** The pipeline adds a **uniform‑baseline margin** (δ_abs=0.002), a **top‑2 (runner‑up) logit gap** (δ_top2=0.5), plus a **run‑of‑two** stability gate; it also emits machine‑readable **measurement guidance** that flips on rank‑first reporting when raw‑vs‑norm risk is high, norm‑only semantics appear, or last‑layer KL is non‑zero. These gates are exactly the kind of practical guardrails a logit‑lens baseline needs. ([GitHub][1])

**Copy detection is token‑ID and sub‑word aware.** The updated rule matches contiguous subsequences at the ID level with a strict threshold sweep, explicitly ignoring trivial whitespace/punctuation—all appropriate given tokenizer quirks. ([GitHub][4])

**Tuned‑lens and prism are treated as diagnostics**, not truth oracles—good. Tuned‑lens is correctly cited as improving robustness relative to the raw logit lens (and the audit distinguishes rotation vs temperature contributions). Prism is framed as a “shared‑decoder” diagnostic with neutral/helpful/regressive verdicts based on **rank milestones** and **KL deltas**, which is reasonable for this iteration. ([arXiv][5])

**Potential adjustments / pitfalls**

* **Head‑calibration detection**: Several families (notably Gemma) show **non‑zero last‑layer KL** that can drop after a temperature re‑scale (τ⋆). The pipeline surfaces this (`tau_star_modelcal`) but evaluators occasionally still discuss absolute probabilities around the final layer. Keep the current guidance (“suppress_abs_probs=true”), and ensure all evaluations consistently avoid absolute probability narratives when last‑layer KL ≠ 0. ([GitHub][2])

* **Prism comparability**: The prism sidecar uses a fixed setting (e.g., k≈512 and sampled layers). Because prism assumes a linearized decoder and freezes non‑linearities, results can **depend on the sampling plan**; calling a model “regressive” on a few samples risks over‑reach. The cross‑model file already flags this nuance for Qwen2.5‑72B; consider echoing that nuance in the per‑model markdown. ([GitHub][2])

* **Repeatability**: The runs are fully deterministic; the “repeatability test” is therefore correctly skipped. If the project later wants a repeatability number, consider adding a **synthetic jitter** (e.g., fp16 rounding or tiny Gaussian on the residual **outside** the model) to quantify near‑threshold rank flips without changing the core scope. (No change needed for this iteration.)

Overall, the approach is methodologically sound for a baseline, and the **known limitations** section in the brief is appropriate and consistent with the implementation choices.

---

## 2) Results & analyses: what was overstated, missed, or misinterpreted

**Strengths in the existing write‑ups**

* The **cross‑model synthesis** gets the big picture right: Llama‑3‑70B confirms semantics early (~40/80), Qwen3 variants are late (31/36, 36/40), Gemma confirms only at the **last** layer (42/42; 46/46), and Qwen2.5‑72B also only at the last layer (80/80). It correctly distinguishes **confirmed** vs “weak rank‑1” onsets using the gates. ([GitHub][2])

* **Copy‑reflex** is attributed to the **Gemma** family only; other families do not show early strict/soft copy hits under the defined thresholds—a fair claim given the CSVs and diagnostics. ([GitHub][2])

* The artifact‑risk section properly calls out **high raw‑vs‑norm divergence** for Qwen3/ Yi and **low tier** for Llama‑3‑70B. ([GitHub][2])

**Items to tune down or clarify**

* **Prism on Qwen2.5‑72B**: The single‑model evaluation labels prism “regressive” at the final layer, which is correct for the last layer—but **early/mid‑depth** prism shows **lower KL** than the norm lens. The cross‑model file already corrects this to “mixed.” Per‑model text should mirror that nuance. ([GitHub][2])

* **Absolute probabilities**: In a few places the evaluations quote layerwise entropies and margins right next to “last‑layer KL non‑zero” warnings (e.g., Gemma). The measurement guidance says to **prefer ranks** when these flags are on; keep all narratives probability‑agnostic in those cases (the current files are mostly compliant). ([GitHub][6])

* **Lens‑induced early onsets**: For models with “norm‑only semantics” near the end (Gemma, Qwen2.5‑72B), the write‑ups correctly restrict claims. Extend the same caution for **“weak” onsets** (rank‑1 without uniform margin) in Llama‑3‑70B and Llama‑3‑8B—some passages already do this; make it universal. ([GitHub][7])

No major misinterpretations were found beyond these nuances.

---

## 3) Independent deep‑dive read of the outputs (by family)

**Gemma‑2 (9B / 27B)**

* **Semantics** only at the **final layer** (L=42/46). Both models exhibit **early copy‑reflex** (L0–L3). Raw‑vs‑norm artifact risk is **high**, and **last‑layer KL is non‑zero** with **τ⋆≈2.85–3.01** indicating calibrated final heads; tuned‑lens acts mostly as **calibration** rather than rotation. Prism is **helpful** mid‑depth for 27B (large negative KL deltas), but diverges near the very end where norm‑only semantics appears. These patterns argue for **rank/KL‑only** onset claims. ([GitHub][6])

**Qwen‑3 (8B / 14B)**

* **Late** semantic confirmation (31/36, 36/40). **High** raw‑vs‑norm divergence; tuned‑lens **rotation** contributes substantially (ΔKL_rot p50≈0.92 for 8B), and tuned sometimes **delays** rank‑1 relative to the baseline—underscoring that tuned‑lens is not merely temperature calibration. Prism is **regressive** at sampled depths. Control margins become strong by ~L30–36. ([GitHub][8])

**Qwen‑2.5‑72B**

* **Final‑layer only** semantics (80/80). Mixed raw‑vs‑norm overlap and **no strong** control position reported. Prism is **helpful early/mid** (KL reductions vs norm) but **fails** at the final layer (rank ≫ 1 under prism), hence **mixed** overall. ([GitHub][2])

**Llama‑3 (8B / 70B)**

* **70B** confirms at **L40/80** with **low** artifact tier; the uniform‑margin gate at L40 **fails** (weak onset), with margin passing only later. Prism mildly regresses; no tuned‑lens sidecar was trained (cost). **8B** is **late** (~25/32), artifact tier **medium**, tuned‑lens rotation is sizable and **delays** the rank‑1 milestone relative to norm; margin passes at the end. Neither shows early copy‑reflex. ([GitHub][7])

**Mistral (7B / Small‑24B)**

* Both **late** (25/32; 33/40). Artifact risk **high** for 7B, **low** for 24B. Prism regresses or is neutral; control margins become strong by ~L31 in 24B. ([GitHub][2])

**Yi‑34B**

* **Mid‑late** (~44/60), high raw‑vs‑norm risk, with **tuned‑confirmed** onset; prism often increases KL relative to norm. ([GitHub][2])

**Cross‑cutting metric patterns**

* **Entropy drift** shrinks as rank improves and KL falls; p50 drifts are large (≈12–14 bits) for Llama‑3 and Mistral‑24B, **negative** for Gemma‑2‑9B (consistent with calibrated heads + very late semantics), **modest positive** for 27B. These match the family‑level calibration stories. ([GitHub][2])

**Context with the literature**

* The pipeline’s preference for **tuned‑lens** over raw lens aligns with results showing tuned‑lens to be more predictive and less biased than the classic logit lens; the decomposition into **rotation vs temperature** mirrors established analyses. ([arXiv][5])
* The **late consolidation** of semantics is consistent with work showing upper layers house more semantic, value‑like “memories” (MLP key‑value views), while earlier layers track shallow or positional signals. ([arXiv][9])
* The **copy‑reflex** in Gemma connects to copy/induction‑style circuitry emerging at specific training phases; while the present study does not isolate heads, the behavioral echo is compatible with that line of evidence. ([Transformer Circuits][10])

---

## 4) Usefulness for the nominalism vs realism debate

**What these results do support**

* There are **robust, layer‑localized regularities** in how diverse LLMs converge from weak, near‑uniform distributions to confident, fact‑selective distributions for the same simple relation (“capital of”). The depth fractions for semantic onset are **family‑specific yet consistent within family**, and they persist across several paraphrase/control variants and lenses when guarded by margins and run‑of‑two. As such, the study offers **empirical evidence of shared internal structure** that is not exhausted by mere name‑matching. This weighs against an **austere nominalism** (on which “similarity” in behavior would be entirely a matter of labels or predicates) **about LLM internals**. ([GitHub][2])

**What they do *not* establish**

* The existence of **mind‑independent universals** is a metaphysical thesis about the world, not the model. These experiments demonstrate **model‑internal invariants** (plausibly relation‑like features or subspaces), but that is compatible with **nominalist** accounts (e.g., trope or resemblance‑based) of how a system can organize particulars without positing universals. Any claim that these layerwise invariants *are* universals would overreach the evidence in this iteration. ([Stanford Encyclopedia of Philosophy][3])

**Bridge to stronger philosophical claims (future work pointers, not scope for this iteration)**

* Causal localization (activation patching) of **relation‑specific features** that generalize across many country–capital pairs would strengthen the case for **type‑like** internal structure. Lightweight residual‑stream patching (as in standard transformer MI workflows) can provide this without changing training. ([alignmentforum.org][11])

---

## 5) Targeted additions/adjustments for **this** iteration (low lift, non‑negligible value)

1. **Causal patch sanity‑check at the semantic layer (two‑prompt patching).**
   Patch the **residual stream** at `L_semantic_confirmed` from a run on pair A (e.g., *France→Paris*) into a run on pair B (*Germany→?*) at the token immediately before the answer. If the patched run shifts B’s answer toward **Paris** (or otherwise degrades the correct **Berlin**), that is direct causal evidence of **relation‑level features** at that layer. This uses existing caching hooks; no new training or datasets are required, and it ties the semantics claim to **causal structure** rather than readout alone. ([alignmentforum.org][11])

2. **Micro‑suite to N≈10 with “near neighbors.”**
   Add a handful of **hard negatives / near neighbors** (e.g., *Germany→Bonn*, *Australia→Sydney*) to complement the existing France control. This sharpened control reduces the risk that a single well‑known pair drives depth estimates and provides a clearer **margin** picture without changing the conceptual scope or compute profile. (The current five‑fact micro‑suite is a great start; doubling it slightly improves within‑model medians/IQRs.) ([GitHub][1])

3. **Report the “strong & stable” semantic onset (already in JSON) in the per‑model markdown.**
   The JSON already computes `L_semantic_strong` and `L_semantic_strong_run2`; surface these milestones in each evaluation file alongside `L_semantic_confirmed` so readers consistently see the effect of **uniform margin + top‑2 gap + run‑of‑two**. This is editorial, not computational. ([GitHub][1])

4. **Prism nuance line in each model evaluation.**
   Add one sentence to each per‑model report clarifying whether prism helps early/mid but fails late (as in Qwen2.5‑72B) vs uniformly regresses (as in Qwen3‑8B). This preserves the diagnostic’s value while avoiding over‑generalization. ([GitHub][2])

These adjustments keep the iteration **self‑contained**, improve interpretive reliability, and slightly extend causal evidence without new training or costly lens fits.

---

## 6) Closing context against current interpretability literature

* **Tuned‑lens** is the right comparison point and is known to be more predictive than the raw logit lens; the project’s tuned audits (rotation vs temp) reflect state‑of‑the‑art practice. ([arXiv][5])
* **RMSNorm‑aware** lensing is essential for modern pre‑norm models and is correctly implemented here. ([arXiv][12])
* The observed **late semantic consolidation** aligns with views of MLPs as **key‑value memories** and with **induction‑style** mechanisms supporting copying/completion—consistent with the Gemma copy‑reflex and late consolidation in other families. ([arXiv][9])
* If later iterations pursue **feature‑level** claims (e.g., “capital‑of features”), sparse autoencoder work and superposition caveats should guide expectations about **non‑canonical** feature sets. (This is beyond scope for 001, but relevant background.) ([Transformer Circuits][13])

---

## Bottom line

The **experimental design is solid** for a first‑pass, multi‑model logit‑lens baseline, and the **results are interpreted with appropriate caution** in most places. The **family‑specific depth profiles**, **copy‑reflex in Gemma**, **artifact‑risk diagnostics**, and **prism/tuned sidecars** are all meaningful contributions. With **three small adjustments** (a minimal causal patch, slightly larger micro‑suite, and consistent reporting of the “strong & stable” onset), this iteration can stand as a **reliable empirical foundation** for any subsequent philosophical or mechanistic work on how LLMs represent simple relational structure—without overreaching into metaphysics. ([GitHub][2])

---

### Sources (selection)

* Project plan & implementation notes, including normalization, gates, micro‑suite design. ([GitHub][1])
* Cross‑model and per‑model evaluation files (Oct 12, 2025 runs). ([GitHub][2])
* Tuned‑lens (Belrose et al., 2023). ([arXiv][5])
* RMSNorm (Zhang & Sennrich, 2019). ([arXiv][12])
* Key‑value memory view of FFNs (Geva et al., 2020/2021). ([arXiv][9])
* Induction heads (Olsson et al., 2022). ([Transformer Circuits][10])
* SEP entries on nominalism/realism and universals. ([Stanford Encyclopedia of Philosophy][3])

*Methodological note*: All claims about specific layers, margins, and artifact risk were taken **directly** from the JSON/CSV‑derived evaluation markdowns for each model and the cross‑model synthesis. When absolute probability narratives risked misinterpretation (e.g., family‑specific head calibration), the project’s own **rank‑first** guidance was followed. ([GitHub][6])

[1]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_LAYERS_BASELINE_PLAN.md "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-cross-models.md "raw.githubusercontent.com"
[3]: https://plato.stanford.edu/entries/nominalism-metaphysics/?utm_source=chatgpt.com "Nominalism in Metaphysics"
[4]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/NOTES.md "raw.githubusercontent.com"
[5]: https://arxiv.org/pdf/2303.08112?utm_source=chatgpt.com "Eliciting Latent Predictions from Transformers with the ..."
[6]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-gemma-2-9b.md "raw.githubusercontent.com"
[7]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Meta-Llama-3-70B.md "raw.githubusercontent.com"
[8]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Qwen3-8B.md "raw.githubusercontent.com"
[9]: https://arxiv.org/abs/2012.14913?utm_source=chatgpt.com "Transformer Feed-Forward Layers Are Key-Value Memories"
[10]: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html?utm_source=chatgpt.com "In-context Learning and Induction Heads"
[11]: https://www.alignmentforum.org/posts/hnzHrdqn3nrjveayv/how-to-transformer-mechanistic-interpretability-in-50-lines?utm_source=chatgpt.com "How-to Transformer Mechanistic Interpretability—in 50 ..."
[12]: https://arxiv.org/abs/1910.07467?utm_source=chatgpt.com "Root Mean Square Layer Normalization"
[13]: https://transformer-circuits.pub/2024/scaling-monosemanticity/?utm_source=chatgpt.com "Extracting Interpretable Features from Claude 3 Sonnet"

---

Produced by GPT-5 Pro
