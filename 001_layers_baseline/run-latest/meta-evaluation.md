
# Review of “Logos‑in‑Layers: copy‑ vs semantic‑collapse with a logit‑lens sweep across 10 base LLMs”

## Executive summary

* **What the run robustly shows (within the stated scope):**
  On a simple factual task (“Germany→Berlin”), most models only reach **rank‑1 for the gold token late in depth**; several families show **no prompt‑echo “copy reflex”** by the strict detector; Gemma‑2 is the notable outlier with an **L0 echo of the prompt word “simply”** while semantic rank‑1 appears only at the **final layer**. These claims are directly visible in the JSON diagnostics and pure‑next‑token traces. ([GitHub][1])

* **Calibration caveats are real and material:**
  Several models (notably Gemma‑2, and intermittently Qwen/Yi) show **high KL between the lens readout and the model’s final head** and “norm‑only semantics” flags in the raw‑vs‑norm check, indicating **lens artifacts**. This weakens any probability‑level cross‑model claims and motivates using **learned calibration readouts (Tuned Lens)** in the next iteration. ([GitHub][2])

* **For the philosophy goal (realism vs nominalism):**
  The present sweep is a good start for **operationalizing “surface‑form vs concept”** as **L\_copy vs L\_semantic**, but it remains **insufficient to adjudicate between nominalism and realism**. To make progress, the next round should add **invariance/causality tests (activation patching), paraphrase/control batteries**, and **feature‑level analyses (sparse autoencoders)** to distinguish string‑matching from a **conceptual “capital‑of” relation** implemented by a circuit. Prior literature supports these improvements (Tuned Lens; induction‑head style copying; monosemantic features). ([arXiv][3])

---

## Method & code: what looks solid, and what to change

**Strengths of the approach**

* **Architecture‑aware normalization** for a norm‑lens (pre‑norm: next block LN1 / final LN; post‑norm: LN2) with FP32 stats before unembed. This is the right family of choices for RMSNorm models and avoids some classic “raw lens” pitfalls. The diagnostics confirm the choices per model. ([GitHub][1])
* Clear, reproducible **definitions**:
  – **L\_copy** (strict and soft variants) by prompt‑echo with hard probability and margin thresholds;
  – **L\_semantic** as first layer with gold‑token rank==1;
  – **Δ = L\_semantic − L\_copy** as the “cling‑to‑surface vs retrieve‑meaning” gap.
  The JSON “copy\_detector” and “ablation\_summary” blocks make the decisions inspectable. ([GitHub][4])
* **Controls and ablations** exist (France→Paris; “no‑filler” variant), and Prism overlays were wired in. ([GitHub][1])

**Issues & actionable fixes**

1. **Lens calibration is the main weak point.**
   Multiple models show **high last‑layer KL** between the lens readout and the model head (e.g., Gemma‑2‑27B `kl_to_final_bits≈1.14`, `warn_high_last_layer_kl=true`). Raw‑vs‑norm samples mark **“norm‑only semantics”** in some families (Meta‑Llama‑3‑8B, Yi‑34B). This indicates that the current normalization is **not sufficient as a probability lens**; rank‑only milestones are safer than probability/entropy comparisons. **Recommendation:** add **Tuned Lens** (train affine probes per block; standard in 2023‑25 work) and report **both** tuned‑ and norm‑lens metrics side‑by‑side. ([GitHub][2])

   *Why:* Tuned Lens explicitly corrects layerwise basis/scale drift and has been shown to be **more predictive, reliable, and less biased** than the raw/norm logit lens. ([arXiv][3])

2. **Single‑prompt dependence is too brittle for philosophy claims.**
   The team already flags this, but the current summaries still lean on one phrasing. The CSVs show frequent punctuation/quote tokens in top‑k, which can interact with the copy detector and entropy. **Recommendation:** run a **paraphrase suite** (dozens of templates, including different clause orders and synonyms), and aggregate **per‑model distributions of L\_copy/L\_semantic** rather than single points.

3. **Copy detector may misinterpret “prompt closures”.**
   The positive prompt ends with “simply”, and early layers in Gemma predict “simply” again (L0). That’s a true prompt echo by the rule, but it’s **trivially explainable by last‑token recency**, not necessarily by an induction‑style copy circuit. **Recommendation:** add **prompt variants that do not end with a high‑frequency closure token**, and examine **k>1** phrase echoes more centrally (the code computes them; the analysis should foreground them). Induction‑style copying is well‑studied and is more visible as **\[A]\[B] … \[A]→\[B]** patterns than single‑token repeats. ([Transformer Circuits][5])

4. **Report temperature‑adjusted last‑layer alignment.**
   The diagnostics already estimate a temperature (e.g., Gemma‑2‑27B `temp_est≈2.61`). Add a per‑model **monotone temperature calibration** for the last layer and re‑report **KL\_after\_temp\_bits** as the “final‑head alignment” metric. This will separate “head mismatch” from “true semantic uncertainty.” ([GitHub][2])

5. **Avoid cross‑family probability/entropy comparisons.**
   Given different normalizers, vocabularies, and head calibrations, **only within‑model** trends are reliable. Keep cross‑family claims to **ranks and depths** unless tuned probes are used.

---

## Results & analysis audit

**What clearly checks out in the dumps**

* **Meta‑Llama‑3‑8B**: `num_layers=32`, `L_semantic=25` (≈78% depth), last‑layer lens agrees with the model (`kl_to_final_bits=0.0`). ([GitHub][1])
* **Mistral‑7B‑v0.1**: `num_layers=32`, `L_semantic=25` (≈78%), KL≈0 at the last layer. ([GitHub][6])
* **Qwen‑3‑8B**: `num_layers=36`, `L_semantic=31`, `L_copy_H=31` (Δ≈0). Raw‑vs‑norm samples show large `max_kl_norm_vs_raw_bits≈13.6` → **lens artifact risk**. ([GitHub][4])
* **Qwen‑3‑14B**: `num_layers=40`, `L_semantic=36`, `L_copy_H=32` (Δ≈4). Raw‑vs‑norm `max_kl_norm_vs_raw_bits≈17.7` → **high lens artifact risk** before the last layer. ([GitHub][7])
* **Yi‑34B**: `num_layers=60`, `L_semantic=44` (≈73%). Raw‑vs‑norm flags **“first\_norm\_only\_semantic\_layer=46”** (norm‑only semantics) even though final KL is tiny (`≈0.00028`). ([GitHub][8])
* **Gemma‑2‑9B**: `num_layers=42`, `L_copy=0`, `L_semantic=42` (semantic only at the end). ([GitHub][9])
* **Gemma‑2‑27B**: `num_layers=46`, `L_copy=0`, `L_semantic=46`, **last‑layer KL high** (`≈1.14`, `warn_high_last_layer_kl=true`). ([GitHub][2])

**Cross‑model evaluation file** (already in the repo) is consistent with the above and adds coverage for models not inspected here (Llama‑3‑70B, Mistral‑Small‑24B, Qwen‑2.5‑72B):
– **Llama‑3‑70B**: `first_rank_le_1=40/80` (≈50% → relatively early).
– **Mistral‑Small‑24B**: `33/40` (≈83%).
– **Qwen‑2.5‑72B**: `80/80` (latest).
It also documents the Gemma family’s copy‑reflex at L0, and the **Prism** overlays (k=512) being present and compatible. ([GitHub][10])

**Where the analysis slightly over‑reaches**

* **Entropy‑level comparisons across families.** The cross‑model markdown quotes entropies at L\_sem for several models; given the KL/normalization divergences, the conservative reading is **“rank collapses late, but the distribution may remain broad”** rather than “model X is more/less certain than model Y”. This is noted in places, but the summary occasionally slides toward cross‑family entropy contrasts despite the calibration warnings. ([GitHub][10])

* **Interpreting Gemma’s L0 echo as a meaningful copy circuit.** The observed L0 top‑1=“simply” is very plausibly a **recency/closure effect**; calling it a “copy reflex” is correct by the rule but should not be treated as **induction‑head‑like behavior** without attention‑pattern evidence. ([GitHub][10])

**Good practice already present**

* The cross‑model page **downgrades** confidence where **raw‑vs‑norm** flags “high risk” and where **`warn_high_last_layer_kl=true`** (Gemma‑2). This is the right stance for a first‑pass lens sweep. ([GitHub][10])

---

## Independent deep dive: what else is visible in the dumps

1. **“Late semantics” is the default; “mid semantics” is rare and size‑dependent.**
   Small/mid models (Llama‑3‑8B, Mistral‑7B) and large models with strong downstream scores (Qwen‑2.5‑72B) **all** reach rank‑1 late (≈75–100%), with **Llama‑3‑70B** an exception (≈50%). This suggests that **semantic collapse depth is not a monotonic proxy for capability**—it looks more like **calibration + architecture/training idiosyncrasy**. ([GitHub][1])

2. **Copy vs semantic deltas are mostly undefined or ≈0.**
   Except Gemma‑2 (copy at L0; semantics at the end) and Qwen‑3‑14B (a small Δ≈4 using `L_copy_H`), **most models do not trigger the strict or soft detectors before semantics**. That is, **surface‑form echo is not a prominent transitional state** under this task—consistent with the idea that models “**compute**” the answer more than they “**parrot**” it here. ([GitHub][2])

3. **Raw‑vs‑norm sanity checks matter.**
   Norm‑only semantics (e.g., Yi‑34B at 46) warns that **normalization can make a hidden state look more “answer‑like” than the model’s own head thinks it is**. This is exactly the failure mode Tuned Lens aims to correct by learning a per‑layer linear readout that *maps into the final unembedding basis*. ([GitHub][8])

4. **Prism overlays are present but do not change the story qualitatively.**
   In the cross‑model summary, Prism sometimes **improves KL** at sampled depths (Gemma‑2‑27B) and sometimes **worsens it** (Gemma‑2‑9B), **without** causing earlier rank milestones. Treat Prism as a useful **calibration overlay** rather than a substitute for tuned probes. ([GitHub][10])

---

## Context from the literature (why the above recommendations)

* **Tuned Lens** trains an **affine probe per block** to decode hidden states and has empirical evidence of being **more predictive, reliable, and less biased** than logit‑lens variants; it aligns the per‑layer basis to the final unembedding and is designed to **reduce lens artifacts**. This is the canonical upgrade for analyses like this. ([arXiv][3])

* **Induction‑style copying** is a specific **two‑head circuit** that completes `[A][B] … [A]→[B]`. When diagnosing “copy reflex,” looking for this pattern (via attention heads and activation patching) is more informative than single‑token repeats. ([Transformer Circuits][5])

* **Sparse Autoencoders & monosemantic features:** recent work shows that **feature dictionaries** can be extracted at scale and used as an analysis substrate, including for **factual/relational features**. This is the right tool to probe whether a **“capital‑of”** feature exists and where it lives. ([Transformer Circuits][11])

* **TransformerLens** remains the standard library for **activation patching** and lens experiments across many open models; it also links the **Tuned Lens** paper and related tooling. ([GitHub][12])

---

## On usefulness for the **realism vs nominalism** debate

**What the current sweep establishes:**
A **behavioral proxy for “surface vs concept”** can be implemented operationally: **L\_copy** (prompt‑echo) vs **L\_semantic** (gold‑token rank‑1). The fact that **most models do not pass through a detectable echo phase** before semantics on “Germany→Berlin” suggests that, *for this prompt*, they are **not merely copying** the string “Germany” or other tokens; instead, the lens sees a late **collapse toward the semantic answer**. Within the narrow scope, that is **weak evidence** for **concept retrieval over string echo**.

**Why this does *not yet* move the philosophical needle:**
Nominalism vs realism turns on whether models use **general concepts (universals)** or merely **names/associations**. The present test uses **one concrete instance** and a **single language/wording**; **late rank‑1** could arise from **memorized name–pair association** or from a **relational concept** (“capital‑of”). Distinguishing these requires **invariance and causality**:

* **Invariance:** does the collapse remain early/late and strong across **many paraphrases**, **other country–capital pairs (seen/unseen)**, **other languages**, **tokenizations**, and **synthetic counterfactuals** (“In *Freedonia*, the capital is …”)?
* **Causality:** patch the **subject token (“Germany”)** and **relation tokens (“capital”, “called”, “named”)** between prompts to see whether **specific layers/heads** carry a **general “capital‑of” feature** that propagates to the answer, as opposed to string echoes.

Without these, it is not possible to argue that the models encode a **universal concept** rather than a **cluster of learned associations**.

---

## Recommendations: concrete, high‑value next steps

1. **Add Tuned Lens as a first‑class readout** (keep norm‑lens for comparability).

   * Report **L\_semantic (tuned)** vs **L\_semantic (norm)**, **KL\_to\_final**, and **entropy** per layer for both.
   * Keep **rank milestones** as the primary cross‑model comparator until calibrations converge. ([arXiv][3])

2. **Build a paraphrase & control battery** (≥50 templates).

   * Vary tense, voice, and clause order: “Germany’s capital is…”, “The city that serves as Germany’s capital is…”, etc.
   * **Remove closure tokens** at the end (“simply”) and test **no‑filler** and **“give only the city”** instructions separately.
   * Summarize models by **distributions** of L\_copy/L\_semantic/Δ across the suite, not single numbers.

3. **Activation patching to localize the “capital‑of” computation.**

   * Patch **residual stream** activations for the **subject token** between “Germany→Berlin” and “France→Paris”; identify **heads/MLPs** whose outputs causally swing the answer.
   * This is a decisive test of **concept‑carrying pathways** vs **name‑matching**. (Standard tools exist in TransformerLens for this workflow.) ([GitHub][12])

4. **Sparse Autoencoder (SAE) pass on mid‑layers.**

   * Train SAEs on a modest corpus for each model (or reuse public dictionaries where available) and **probe for features that activate on “capital‑of” contexts** across multiple countries and languages.
   * If a **stable, interpretable feature** appears and **drives** the answer under patching, that is **non‑trivial evidence** for **realist‑style conceptual structure**. ([Transformer Circuits][11])

5. **Cross‑lingual & OOD tests for invariance.**

   * Run German (“Die Hauptstadt von Deutschland ist …”), French, and mixed‑language prompts; also **invented countries/cities** with **held‑out pairs** to distinguish **memorized associations** from **relational reasoning**.
   * Track whether **L\_semantic depth** and **the same circuit** reappears.

6. **Refine the copy detector.**

   * Prioritize **k>1 windows** and **span‑aware detectors** (e.g., longest‑common‑subsequence on decoded prefixes) to better capture true in‑context copy rather than last‑token closure.
   * Consider a **quote/punctuation mask** learned from frequency, not only a fixed ignore‑list.

7. **Strengthen calibration reporting.**

   * Always include **`kl_to_final_bits`, `kl_after_temp_bits`**, and a **“lens artifact risk”** tag (low/med/high) *per model*, with short rationales drawn from raw‑vs‑norm checks.
   * When **Prism** is active, summarize **ΔKL vs baseline** at fixed depth percentiles (25/50/75%) and note whether **rank milestones** shift (so far they do not). ([GitHub][10])

8. **Extend beyond single‑token answers.**

   * Repeat the battery on **multi‑token capitals** to ensure the conclusions aren’t an artifact of single‑token decodability.

---

## Comments on the existing plan

The project notes already emphasize careful normalization, single‑prompt scoping, and calibration overlays (Prism). The **highest‑leverage additions** for iteration 2 are:

* **Tuned Lens integration** (core);
* **Paraphrase/control suite** at scale;
* **Activation patching** and **SAEs** specifically targeted at the **“capital‑of”** relation;
* **Cross‑lingual/fictional controls** to stress conceptual invariance.

These directly advance the **nominalism vs realism** objective by moving from **rank‑timing observations** to **feature/circuit evidence** about whether the model uses a **general relation** vs **specific name‑pairs**. The literature and the current dumps both justify these priorities. ([arXiv][3])

---

## Selected evidence excerpts (by source)

* **Model JSONs (examples):**
  – *Meta‑Llama‑3‑8B*: `L_semantic=25/32`, `kl_to_final_bits=0.0`. ([GitHub][1])
  – *Mistral‑7B‑v0.1*: `L_semantic=25/32`, `kl_to_final_bits=0.0`. ([GitHub][6])
  – *Qwen‑3‑8B*: `L_semantic=31/36`, `L_copy_H=31`, raw‑vs‑norm `max_kl≈13.6` (risk high). ([GitHub][4])
  – *Qwen‑3‑14B*: `L_semantic=36/40`, `L_copy_H=32`, raw‑vs‑norm `max_kl≈17.7` (risk high). ([GitHub][7])
  – *Yi‑34B*: `L_semantic=44/60`, `first_norm_only_semantic_layer=46`, `kl_to_final≈0.00028`. ([GitHub][8])
  – *Gemma‑2‑9B*: `L_copy=0`, `L_semantic=42/42`. ([GitHub][9])
  – *Gemma‑2‑27B*: `L_copy=0`, `L_semantic=46/46`, `kl_to_final≈1.14`, `warn_high_last_layer_kl=true`. ([GitHub][2])

* **Cross‑model evaluation** (repo): early/late timing across all ten models; Gemma’s L0 copy; Prism presence; KL cautions. ([GitHub][10])

* **Background references:**
  – *Tuned Lens* (affine probes; reduces lens brittleness). ([arXiv][3])
  – *Induction Heads* (copy circuit signature to look for). ([Transformer Circuits][5])
  – *Scaling Monosemanticity* (SAE‑based feature dictionaries at scale). ([Transformer Circuits][11])
  – *TransformerLens* (standard toolkit used for patching/lens work; links to Tuned Lens). ([GitHub][12])

---

### Bottom line

As a **first iteration**, this sweep is **well‑designed and candidly caveated**. The core empirical picture—**semantic collapse late; copy reflex rare; Gemma a copy‑at‑L0 outlier; calibration issues visible**—is supported by the dumps. The next step is to **trade breadth for depth**: add **tuned calibration, invariance batteries, and causal/feature probes** to say something meaningful about whether the models are exercising a **general “capital‑of” concept** (realist‑leaning) rather than **memorized name‑pairs** (nominalist‑leaning).

[1]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/output-gemma-2-27b.json "raw.githubusercontent.com"
[3]: https://arxiv.org/abs/2303.08112?utm_source=chatgpt.com "Eliciting Latent Predictions from Transformers with the Tuned Lens"
[4]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/output-Qwen3-8B.json "raw.githubusercontent.com"
[5]: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html?utm_source=chatgpt.com "In-context Learning and Induction Heads"
[6]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json "raw.githubusercontent.com"
[7]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/output-Qwen3-14B.json "raw.githubusercontent.com"
[8]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/output-Yi-34B.json "raw.githubusercontent.com"
[9]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/output-gemma-2-9b.json "raw.githubusercontent.com"
[10]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-cross-models.md "raw.githubusercontent.com"
[11]: https://transformer-circuits.pub/2024/scaling-monosemanticity/?utm_source=chatgpt.com "Extracting Interpretable Features from Claude 3 Sonnet"
[12]: https://github.com/neelnanda-io/TransformerLens "GitHub - TransformerLensOrg/TransformerLens: A library for mechanistic interpretability of GPT-style language models"
