# Review of **Experiment 001 – Layers & Logits**

*Logit‑lens sweeps across ten open‑weight base LLMs; copy‑ vs semantic‑collapse; implications for nominalism vs realism.*

---

## Executive summary

**What is solid**

* The script correctly performs a **per‑layer logit‑lens** sweep with **model‑appropriate normalization** (RMSNorm/LayerNorm) applied at decode time, and it records per‑layer **entropy and top‑k** for the **pure next token** rather than averaging over the prompt—avoiding a common pitfall. The implementation places **ε inside the √** for RMSNorm and probes **post‑block residuals with `ln2`** (or the correct pre/post norm), which are the right choices for this setup. ([GitHub][1])
* The run artifacts expose enough metadata to reason about numerical hygiene (device, dtype, whether FP32 unembed was used, etc.), and the JSON schema already includes **`L_copy`**, **`L_semantic`**, and model stats, enabling disciplined within‑model comparisons. ([GitHub][2])
* The accompanying cross‑model write‑up is careful about **RMS‑lens comparability limits** and makes mostly conservative claims; it frames the results as suggestive rather than definitive. ([GitHub][3])

**What needs correction or tightening**

* **Copy‑collapse** is currently detected by string membership of the **top‑1 token** in the prompt with a probability threshold; this **misses multi‑piece tokens** and occasionally fires spuriously when the fallback entropy rule triggers. This inflates or nulls **`L_copy`** in places and should be replaced by a **detokenised, ID‑aware** comparison (and, if kept, the entropy fallback should be reported separately). ([GitHub][3])
* **Cross‑model probability** comparisons are still tempting in the narrative. Those should be replaced by **rank‑based** or **KL‑to‑final** metrics, or by using a **Tuned Lens**/**Logit Prism** so that geometry is better aligned across depth and models. The **tuned lens** paper documents precisely why raw logit‑lens can be **biased and brittle** across models. ([arXiv][4])
* The current single‑prompt design is adequate for a first pass but **over‑fits** to tokenizer quirks and instruction‑following cues (e.g., the adverb “simply”). This is noted in the plan and should be addressed before drawing philosophical conclusions. ([GitHub][5])

**High‑level takeaways from the data (first iteration)**

* **Early “copy” vs late “meaning”.** Several models show a delay between first prompt‑echo (`L_copy`) and first gold‑token top‑1 (`L_semantic`), consistent with **surface retrieval preceding semantic consolidation**. For example, **Qwen3‑14B** shows `L_copy = 32`, `L_semantic = 36` (Δ = 4), while **Gemma‑2‑9B** shows **no copy collapse** and reaches `L_semantic` at the **final layer** (`42/42`). **Qwen2.5‑72B** reaches `L_semantic` only at **layer 80/80** with `L_copy = null`. **Llama‑3‑8B** reaches `L_semantic` at **25/32** with `L_copy = null`. These are **within‑model** observations and should not (yet) be turned into size‑is‑later generalizations. ([GitHub][6])

---

## Method & code review

### What the script does right

1. **Normalization lens is applied correctly.**

   * For RMS models, the code computes RMS with **ε inside √** and applies the learned scale (γ) when present; for post‑block snapshots it uses **`ln2`**, which is the correct hook for pre‑norm architectures. This addresses a frequent source of “early semantics” artefacts. ([GitHub][1])
   * The code explicitly **casts to FP32 before unembedding** when needed, reducing small‑gap under‑resolution. Diagnostics record the **effective unembed dtype**. ([GitHub][1])

2. **Pure‑next‑token logging.**
   The script writes a **“pure next token”** row per layer (with the token labeled `⟨NEXT⟩`), preventing entropy deflation from previous prompt tokens. This is visible in both the JSON structure and the console/record logic. ([GitHub][1])

3. **Determinism & provenance.**
   A **fixed seed**, deterministic PyTorch algos, and diagnostic fields (device, dtypes, norm types) are present in the JSON outputs, enabling reproducibility and post‑hoc audits. ([GitHub][1])

4. **Within‑model (not cross‑model) framing.**
   The **NOTES**/plan already warns that RMS/LN lenses distort absolute probabilities in model‑specific ways; the code and analyses largely respect this constraint. ([GitHub][5])

### Fragilities or pitfalls in the current approach

1. **Copy‑collapse detection is too brittle.**

   * It checks whether the **decoded top‑1 token string** is a substring of the prompt and `p_top1 > 0.90`. This misses **multi‑piece tokens** (e.g., `"▁Berlin"` versus `" Berlin"`) and can miscount in the presence of punctuation or case variation. The cross‑model write‑up notes odd `L_copy` behavior (e.g., a spurious copy flag for **Qwen3‑8B** at layer 31) consistent with this failure mode. A **detokenised, ID‑aware** check is needed; the plan already proposes this fix. ([GitHub][3])

2. **Residual probe ≠ model’s computation.**
   Even when normalization is correct, a frozen **logit lens** is a **biased/brittle read‑out** relative to the model’s final head. The **Tuned Lens** paper shows the logit lens often over‑weights particular vocab items and frequently predicts **input tokens** in early layers for models like BLOOM/OPT; it recommends replacing logit lens with per‑layer **affine probes** fitted to reduce **KL to final**. Use tuned lens (and/or **Logit Prism**) before making cross‑model or absolute probabilistic claims. ([arXiv][4], [Neural Blogs][7])

3. **Single‑prompt scope.**
   One English prompt with an instruction‑style filler (“simply”) is sufficient for scoping, but it **cannot** distinguish instruction‑following circuits from geography knowledge. The plan correctly proposes a **negative‑control** prompt (“France → Paris”) and **style ablation** (drop “simply”). Prioritize those before further interpretation. ([GitHub][5])

4. **No causal/representational checks yet.**
   Without **activation patching**, **feature‑level ablations**, or **representation‑drift** curves (cosine to final logits), depth‑based claims remain correlational. The plan includes these upgrades; they are necessary to move beyond suggestive patterns. ([GitHub][5])

---

## Results & analysis check

### Model‑level spot checks (from JSON)

* **Gemma‑2‑9B (42 layers)**: `L_copy = 0`, `L_semantic = 42`, meaning the **gold token** first becomes top‑1 only at the **final** unembedding. Diagnostics confirm RMSNorm, post‑block probing, and FP32 paths. **Interpretation:** strong late consolidation; no early prompt echo at p>0.9 under the current detector. ([GitHub][2])
* **Qwen3‑14B (40 layers)**: `L_copy = 32`, `L_semantic = 36` (Δ = 4). **Interpretation:** brief window where **prompt‑echo** precedes semantic top‑1; plausible surface‑→meaning progression. ([GitHub][6])
* **Qwen2.5‑72B (80 layers)**: `L_copy = null`, `L_semantic = 80`. **Interpretation:** semantic top‑1 only at the final head, with **no high‑confidence prompt copy** under the current test; may reflect stricter decoding or different instruction‑following pathways. ([GitHub][8])
* **Meta‑Llama‑3‑8B (32 layers)**: `L_copy = null`, `L_semantic = 25`. **Interpretation:** comparatively **earlier** semantic top‑1 (relative to its depth) without a copy‑collapse event. ([GitHub][9])

### Cross‑model narrative (consistent with the repo’s summary)

* The **relative delay** between prompt echo and semantic top‑1 (Δ) is **non‑zero** in several RMS models and **varies across families**, supporting the descriptive claim that some models **cling to surface form** briefly before converging on the gold token. This is a **within‑model** characterization. ([GitHub][3])
* The tendency of some large models (**Qwen2.5‑72B**, **Gemma‑2‑9B**) to show **final‑head semantic collapse** suggests either **later constructive computation** or **lens misalignment** at intermediate layers (the tuned‑lens literature warns about both). This strengthens the recommendation to add **KL‑to‑final** curves and/or **tuned lens**. ([GitHub][8], [arXiv][4])

### Where the current write‑ups over‑reach or miss nuance

* Claims that a family is **“later or earlier”** should avoid implying **parameter‑count causality**. The examples above show family‑specific effects that are likely **architecture/training‑regime** contingent; the tuned‑lens paper documents that **raw logit‑lens trajectories** vary idiosyncratically across model families. Prefer **rank/KL** metrics and **within‑model** language. ([arXiv][4])
* **Copy‑collapse** is treated as a single event, but **subword** and **multi‑token** echoes can precede the first single‑piece match. This likely under‑counts early copy in tokenizers that split “ Berlin”. Implement the **detokenised window** variant in the plan (§1.2). ([GitHub][5])

---

## Independent deep dive (layer‑fraction view + realism/nominalism reading)

A simple, robust normalization for cross‑model depth is **layer fraction** `L_semantic / n_layers`:

* **Llama‑3‑8B:** 25/32 ≈ **0.78** (semantic top‑1 before the final head). ([GitHub][9])
* **Qwen3‑14B:** 36/40 = **0.90**. ([GitHub][6])
* **Gemma‑2‑9B:** 42/42 = **1.00**. ([GitHub][2])
* **Qwen2.5‑72B:** 80/80 = **1.00**. ([GitHub][8])

**Interpretation:** On this prompt, some families show **semantic consolidation strictly at the final head**, others earlier. Two readings are live:

1. **Construct‑then‑select:** earlier layers build partial evidence; the final head selects.
2. **Rotate‑into‑final‑basis:** the answer direction exists earlier but **basis misalignment** hides it from a raw logit lens; tuned lens or **KL‑to‑final** would show earlier convergence even if top‑1 differs. The tuned‑lens results explicitly show that tuned probes **reduce KL** to final by **\~an order of magnitude**, revealing earlier “meaning” than raw logit lens. ([arXiv][4])

A **representation‑drift** curve (**cosine(logits\_ℓ, logits\_final)**) would adjudicate between (1) and (2): a **monotone cosine rise** with smooth KL drop suggests amplification of a pre‑existing direction; a **late rotation** suggests truly late construction. The plan proposes logging `cos_to_final`; implementing that will materially strengthen the analysis. ([GitHub][5])

---

## Context from the literature (why the upgrades matter)

* **Logit lens vs tuned lens.** Raw logit‑lens trajectories are **biased and brittle**; tuned lens trains an **affine probe per layer** and shows substantially **lower KL to final**, more consistent early semantics, and causal alignment with the model’s features. This is the right next step for depth‑trajectory claims. ([arXiv][4])
* **Induction & copy behavior.** Induction heads and related copy circuits are well‑documented; they often arise at specific depths and can explain **prompt‑echo windows** before semantic consolidation. This supports interpreting non‑zero Δ as surface‑→meaning transitions in some models. ([arXiv][10], [Transformer Circuits][11])
* **Feature‑level realism tests.** **Sparse autoencoders (SAEs)** recover **monosemantic features** that can be **causally manipulated**, increasingly at scale (Anthropic 2023 → 2024). Demonstrating that a small set of **stable features** governs “capital‑of” behaviour across prompts/languages would be much stronger evidence against **austere nominalism** than depth alone. ([Transformer Circuits][12])
* **Shared decoder geometry.** **Logit Prism** (a shared whitening/rotation) offers a **single decoder across layers**, enabling **comparable** per‑layer contributions and reducing probe freedom—useful for cross‑layer/cross‑model claims. ([Neural Blogs][7])

---

## Philosophical usefulness (nominalism vs realism), given this first iteration

**Austere nominalism** says only particular tokens exist and all facts reduce to token occurrences. Even this first pass pressures that view slightly:

* The presence of a **structured, layer‑consistent transition** from prompt‑echo to semantic token dominance (Δ in several models) indicates **systematic internal processing**, not a mere list of token co‑occurrences. Within models where Δ>0 (e.g., Qwen3‑14B), the data suggest **distinct stages** in a reusable pipeline. ([GitHub][6])

**However**, these are **correlational** signals using a brittle probe. To move beyond suggestive pressure on austere nominalism and toward distinguishing **metalinguistic nominalism** from **realism**, the project should:

* Add **negative controls** (e.g., “France → Paris”) ensuring that the curve truly tracks **world knowledge** rather than **lexical co‑occurrence**;
* Demonstrate **cross‑prompt** and **cross‑language** stability of the **same causal layer(s)/features**;
* Identify **features** (SAE or tuned‑lens‑aligned vectors) that **causally steer** the answer (up/down) **across contexts**.
  If stable, **language‑independent** features that causally govern “capital‑of” emerge, the weight shifts further away from nominalism. ([GitHub][5], [Transformer Circuits][13])

---

## Specific corrections and additions to the plan (prioritised; high value only)

### 1) Make `L_copy` sound (low lift, high value)

* **Implement the detokenised copy detector** already sketched in §1.2 of the plan (decode both sides, normalise whitespace/case), and optionally keep a **rolling window** for multi‑piece echoes. **Expose the threshold** (`--copy-thresh`) and write it to JSON for provenance; **log `L_copy@{0.70,0.80,0.90}`** to enable sensitivity analyses. This removes the largest measurement artefact. ([GitHub][5])

### 2) Add **KL‑to‑final** and **cosine‑to‑final** curves (low–medium lift, very high value)

* Per layer, compute **KL(p\_ℓ ∥ p\_final)** (in bits) and **cos(logits\_ℓ, logits\_final)** and write both to the **pure‑next‑token CSV** and JSON summary (**first\_kl\_below\_{0.5,1.0}**). This cleanly distinguishes **amplification** vs **rotation** narratives and makes tuned‑lens adoption measurable. ([GitHub][5])

### 3) Add a **negative‑control prompt** (very low lift, high value)

* Run the same sweep on an **adversarial control** (“Give the country name only… Berlin is the capital of …”). Flag layers where the control elevates the wrong answer (e.g., **Berlin** in a **France** prompt). Report this alongside Δ. ([GitHub][5])

### 4) **Tuned Lens** integration (medium lift, high value)

* Train a tuned lens per model (50k tokens suffice) and **re‑emit** the trajectories with and without tuned lens. Expect significantly **earlier KL convergence** even if top‑1 differs; this will likely revise some “final‑only” semantic collapses into **earlier representational convergence**. Record lens version/sha in JSON. ([arXiv][4])

### 5) **Representation‑drift** and **threshold sweeps** (low lift)

* Log `cos_to_final` as in §1.5; sweep **copy thresholds** `{0.7, 0.8, 0.9}` to show robustness (output `L_copy(τ)` for each τ). These are cheap but informative upgrades. ([GitHub][5])

### 6) Minimal **causal checks** (medium lift, very high value)

* Perform **activation patching** on the top 2–3 layers identified by KL/cosine as decisive: patch those residuals from a **Berlin‑correct** run into **distracting** contexts (or the France control) and measure gold‑token gain. Keep it layer‑ and sublayer‑local to start (attention vs MLP). This moves depth‑based correlations into **causal evidence**. (Canonical technique; any of the TL docs plus prior circuit posts suffice as implementation references.) ([transformerlensorg.github.io][14])

### 7) **Feature‑level** pass (medium–high lift, transformational value)

* Train a small **SAE** (e.g., 16× expansion on 1–2 layers flagged by KL/cosine) and look for **features** whose activation **causally** increases/decreases `p(" Berlin")` **across prompts/languages**. Even one stable, steerable feature is powerful evidence **against austere nominalism** and sets up the MN vs realism phase. ([Transformer Circuits][12])

### 8) Optional: **Logit Prism** (shared decoder) for cross‑layer comparability

* Fit a **whiten+rotation** shared decoder (**Logit Prism**) and re‑plot trajectories. This reduces probe freedom and enables **component‑wise** layer contributions that are more directly comparable. Use it as a complement to tuned lens, not a replacement. ([Neural Blogs][7])

---

## Concrete, near‑term edits to the codebase

* **Copy detector** (`layers_core/collapse_rules.py`):
  Replace single‑piece string membership with **detokenised ID‑aware** check and add a **rolling `k`‑piece window** (configurable). Emit `copy_thresh` and `copy_window_k` in JSON. (Plan §1.2 provides exact snippets.) ([GitHub][5])
* **CSV schema**:
  Add columns to `*-pure-next-token.csv`: `p_top1`, `p_top5`, `p_answer`, `kl_to_final_bits`, `cos_to_final`. Add JSON run‑summary keys `first_kl_below_{0.5,1.0}`. (Plan §1.3–§1.5.) ([GitHub][5])
* **Flags**:
  Add `--raw-lens`, `--use-tuned-lens`, `--use-logit-prism`, `--copy-thresh`, `--dual-lens N`. Record all flags in JSON under a `lens` block for provenance. (Plan §1.4, §1.10–§1.11.) ([GitHub][5])
* **Negative control**:
  Add a 2‑prompt run mode with `prompt_id` in CSV/JSON. (Plan §1.8.) ([GitHub][5])

---

## Bottom line for the nominalism vs realism debate (status after iteration 1)

* **Status now:** The present results, with correct normalization and a single English prompt, show **structured depth trajectories** suggestive of **surface‑→meaning** transitions in some models, mildly **pressuring austere nominalism**. But claims remain **correlational** and **probe‑dependent**. ([GitHub][6])
* **After the proposed upgrades:**
  If **tuned‑lens KL** and **cosine** show **early convergence** while **copy** remains distinct; if **negative controls** rule out lexical leakage; if **activation patching** and **SAE features** reveal **stable, language‑robust causal features** governing “capital‑of”, the project would possess **non‑negligible empirical pressure** not only against austere nominalism but also **toward** views positing **reusable structure** (metalinguistic nominalism or realism). Distinguishing MN vs realism would then hinge on **cross‑language invariance** and **modality‑independent causality**—reasonable goals for later phases. ([GitHub][5], [Transformer Circuits][13])

---

## Citations

* **Code & artifacts**: experiment script and outputs. ([GitHub][1])
* **Logit/Tuned lens**: Belrose et al., *Eliciting Latent Predictions from Transformers with the Tuned Lens* (2023). Documents logit‑lens **bias/brittleness** and tuned‑lens **KL improvements**. ([arXiv][4])
* **RMSNorm**: Zhang & Sennrich (2019). Correct ε placement and scaling details. ([arXiv][15])
* **Induction heads** / copy mechanisms: Olsson et al. (2022). ([arXiv][10], [Transformer Circuits][11])
* **Sparse Autoencoders / monosemantic features**: Anthropic 2023; 2024 scaling update. ([Transformer Circuits][12])
* **Logit Prism** shared decoder: Nguyen (2024). ([Neural Blogs][7])

---

### Final assessment

The engineering is **competent and careful** on normalization and numerics; the **measurement is now trustworthy** *within* models. The main blockers to stronger claims are **copy‑collapse detection** and **probe brittleness**. Implement the plan’s low‑lift fixes (copy detector, KL/cosine, negative control), then integrate **tuned lens** and **minimal causal checks**. Those steps will materially upgrade the scientific value of the findings and make them relevant to the **nominalism vs realism** question rather than merely illustrative.

[1]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run.py "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/output-gemma-2-9b.json "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-cross-models.md "raw.githubusercontent.com"
[4]: https://arxiv.org/pdf/2303.08112?utm_source=chatgpt.com "[PDF] Eliciting Latent Predictions from Transformers with the Tuned Lens"
[5]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/PROJECT_NOTES.md "raw.githubusercontent.com"
[6]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/output-Qwen3-14B.json "raw.githubusercontent.com"
[7]: https://neuralblog.github.io/logit-prisms/?utm_source=chatgpt.com "Logit Prisms: Decomposing Transformer Outputs for ..."
[8]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/output-Qwen2.5-72B.json "raw.githubusercontent.com"
[9]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json "raw.githubusercontent.com"
[10]: https://arxiv.org/abs/2209.11895?utm_source=chatgpt.com "In-context Learning and Induction Heads"
[11]: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html?utm_source=chatgpt.com "In-context Learning and Induction Heads"
[12]: https://transformer-circuits.pub/2023/monosemantic-features?utm_source=chatgpt.com "Decomposing Language Models With Dictionary Learning"
[13]: https://transformer-circuits.pub/2024/scaling-monosemanticity/?utm_source=chatgpt.com "Extracting Interpretable Features from Claude 3 Sonnet"
[14]: https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.ActivationCache.html?utm_source=chatgpt.com "transformer_lens.ActivationCache - TransformerLens Documentation"
[15]: https://arxiv.org/abs/1910.07467?utm_source=chatgpt.com "Root Mean Square Layer Normalization"

---

Produced by OpenAI GPT-5 Pro
