# Review of “Logos‑in‑Layers” (Experiment 001)

## Executive summary

The experiment is a careful, mostly well‑engineered application of a logit‑lens–style probe (with RMS/RMSNorm‑aware normalization) to measure **where** ten open‑weight base LLMs stop echoing the prompt (“copy‑collapse”) and start **resolving the intended meaning** (“semantic‑collapse”) on a simple *Germany → Berlin* fact. The code choices around normalization, dtype, and diagnostics are sound and align with best practice for RMS‑normalized, pre‑norm transformer families. The analysis correctly emphasizes **within‑model** (not cross‑model) probability interpretations and adds rank/KL milestones to reduce calibration artifacts. A few interpretations in the per‑model write‑ups are overstated (notably around “rest\_mass” and one now‑stale cross‑doc mismatch), and several high‑value additions would materially raise the evidential bar for the project’s philosophical aims.

Across models, **semantic collapse is late** (≈ top 30% of layers) in eight of nine families examined; **Meta‑Llama‑3‑70B** is the notable early outlier (L\_sem ≈ 40/80), and **Gemma‑2** shows **very early copy reflex** but **only head‑level semantics** (L\_sem at the final row). These patterns—and the documented spikes in cosine alignment and KL→final only at the head—are consistent with the **Tuned Lens** literature: representations rotate toward the final readout gradually, while **probabilistic calibration** is consolidated at the very top. ([GitHub][1]) ([arXiv][2])

From a philosophy‑of‑universals perspective, the present run provides **initial**, observational evidence against **austere nominalism**: the measured gap Δ = L\_semantic − L\_copy (when defined) and the KL/cosine trajectories indicate a structured progression from surface tokens to relation‑level content, not mere token re‑use. However, nothing here yet forces a decision between **metalinguistic nominalism** (predicate‑level routines) and **realism** (robust universals). That discrimination requires **causal** and **portability** evidence (activation/attribution patching, concept‑vector portability, head fingerprinting, SAEs)—several of which are already sketched in the project plan and should be prioritized next. ([GitHub][3])

---

## 1) Code & methodology review

### What is done right

* **Architecture‑aware normalization (RMS lens)**
  The script applies the **correct normalizer to the correct stream**: for pre‑norm families it uses the *next block’s* `ln1` (or `ln_final` at the end) and for post‑norm it uses the current block’s `ln2`. RMS statistics are computed in fp32 with ε **inside** √ and only then cast back, preventing well‑known scale distortions that plague raw logit lens on RMS models. This is aligned with RMSNorm’s formula and avoids spurious “early semantics.” ([GitHub][4]) ([arXiv][5])

* **Final‑head consistency checks**
  The code compares the last post‑block lens distribution to the model’s true final head, reports KL in bits, top‑1 agreement, and temperature‑like rescales. This caught **Gemma‑2**’s family‑specific head calibration (final‑layer KL ≈ 1 bit), which the analyses correctly flagged as a reason to prefer **rank** over **probability** comparisons for that family. ([GitHub][1])

* **Gold‑token ID alignment and ID‑level copy detection**
  Matching the gold **first token id** (not strings) and detecting copy at the **token‑ID contiguous‑subsequence** level (with τ=0.95 and p1−p2 margin) avoids tokenizer and whitespace artifacts. This is an appropriate guard for subword vocabularies. ([GitHub][3])

* **Deterministic seeding and fp32 unembed**
  The run promotes the unembedding to fp32 when compute is bf16/fp16 and seeds deterministically; this reduces numerical noise in small logit gaps without inviting training‑time stochastics. ([GitHub][4])

* **Raw‑vs‑norm sanity sampling**
  The addition of a “norm‑only semantics” detector and `max_kl_norm_vs_raw_bits` is valuable. Where it reports **high risk** (e.g., Yi‑34B, Qwen‑3‑14B), the write‑ups are right to lean on **rank milestones** and **within‑model** trends instead of absolute p’s. ([GitHub][6])

### Gaps and risks (actionable)

* **Lens induced artifacts remain a central caveat**
  Even with RMS‑aware scaling, a non‑learned lens can miscalibrate probabilities. The analyses already cite Tuned Lens (Belrose et al.), which empirically reduces these biases. Incorporating a **Tuned Lens side‑by‑side decode (or a Logit Prism baseline)** would make early‑layer statements more trustworthy and standardize cross‑layer comparisons. ([arXiv][2], [Neural Blogs][7])

* **Copy‑collapse threshold sensitivity**
  Copy is currently k=1 with τ=0.95 and δ=0.10, and no entropy fallback. That is fine for strictness but brittle. A small, automated **threshold sweep** (τ∈{0.70,0.80,0.90})—already proposed in the plan—should be implemented now and summarized per model to quantify robustness of L\_copy. ([GitHub][3])

* **Sampled raw‑vs‑norm checks**
  The “mode=sample” sanity is useful but can miss pathologies. A light‑weight **stratified sampling** (early/mid/late layers; punctuation vs content tokens) would reduce false reassurance at modest cost. (No new framework needed—just expand the sample.)

* **LN bias & folding caveat**
  For LN models (less relevant here but future‑proofing), **LayerNorm bias folding** can skew logit‑lens readouts; a guard is to re‑apply the exact LN (γ, β) from the probed stream. The documentation in TransformerLens highlights this bias; the current code already preserves LN β when it applies LayerNorm, which is correct. Keep this in place as models vary. ([transformerlensorg.github.io][8])

* **Minor repo hygiene**
  Some `*-pure-next-token.csv` files appear truncated when fetched via raw links (one‑line responses), while the markdown evaluations clearly used full CSVs. Ensure serialization/hosting is consistent; otherwise readers cannot reproduce claims. (This is a publishing/hosting issue, not a methodological one.)

---

## 2) Review of results & analyses

### What the artifacts show (and what they do **not**)

* **Late semantic collapse in most families; early in Llama‑3‑70B**
  The cross‑model synthesis and per‑model reports concur: *Gemma‑2‑9B (42/42), Gemma‑2‑27B (46/46), Qwen‑3‑8B (31/36), Qwen‑3‑14B (36/40), Qwen‑2.5‑72B (80/80), Yi‑34B (44/60), Mistral‑7B (25/32), Mistral‑Small‑24B (33/40)* all collapse at ≥ \~70% depth; **Meta‑Llama‑3‑70B** collapses earlier at **40/80**. This is consistent with cosine‑to‑final and KL→final milestones that only “snap” near the head. ([GitHub][1])

* **Gemma’s “copy‑reflex + head‑only semantics”**
  Under the strict ID‑level copy rule, Gemma‑2 shows early copy flags but **no** pre‑final rank‑1 for the gold answer; semantics only appear at the head, with a **non‑zero last‑layer KL** to the model head—hence probability comparisons should be avoided across families and ranks preferred. ([GitHub][1])

* **Punctuation/markup anchoring mid‑stack**
  Multiple reports show quotes/underscores/non‑English fragments dominating top‑1 across many middle layers, while “Berlin” rises in top‑k but becomes top‑1 only late (e.g., Qwen‑2.5‑72B L51–L79 are often quotes; top‑1 = **Berlin** only at L80). This pattern supports **“early direction, late calibration.”** ([GitHub][9])

* **Ablation on “simply”**
  Removing the stylistic adverb shifts **copy** slightly in Gemma‑2‑27B but leaves **L\_semantic** largely unchanged elsewhere (e.g., **Qwen‑3‑14B** ΔL\_sem = 0; **Llama‑3‑70B** small ±2 shift). This indicates **filler sensitivity is modest** relative to semantic resolution. ([GitHub][10])

* **Negative control behaves**
  “Berlin is the capital of” → “ Germany” top‑1 with large margins, no leakage of “Berlin” into the country slot in the per‑model reports examined. This is a minimal but important probe‑leakage sanity. ([GitHub][11])

### Over‑statements or misses in the markdown evaluations

* **“Rest\_mass ⇒ lens fidelity”**
  One per‑model write‑up treats falling `rest_mass` (top‑k coverage) as evidence of no precision loss from the norm lens; this is **not** a fidelity metric. Lens fidelity is better diagnosed by **last‑layer KL** (≈0 for well‑calibrated families) and **raw‑vs‑norm KL** samples. Recommend rephrasing any such claims to avoid over‑reach. ([GitHub][1])

* **Cross‑doc mismatch (now likely corrected)**
  The cross‑model synthesis flags that some **Qwen‑2.5‑72B** cosine milestones matched **Yi‑34B** numbers; the current Qwen‑2.5 write‑up does **not** contain those values, and Yi‑34B does. Treat the cross‑doc note as an earlier revision warning; keep an automated check (see “Recommendations”). ([GitHub][1])

* **Be explicit when “top‑1” is not the answer**
  A phrasing in the Gemma‑2‑27B evaluation (“p\_top1 > 0.30 at L0; … > 0.60 at L0”) can be misread as **answer** confidence. Clarify that this refers to **generic** top‑1 (copy reflex on “simply”), not `p_answer`. ([GitHub][1])

* **Lean more on **rank milestones** where last‑layer KL ≠ 0**
  The reports already do this for Gemma; extend the same caution consistently wherever `raw_lens_check` reports **high** artifact risk (e.g., Qwen‑3‑14B, Yi‑34B). ([GitHub][10])

---

## 3) Independent cross‑model reading (contextualized by current literature)

The depth‑wise trajectories documented here reproduce two robust themes from the **Tuned Lens** literature:

1. **Rotation first, calibration last.** Cosine alignment to the final (or tuned) head tends to **increase** before probabilities become well‑calibrated; **KL to final remains high until late**, then drops sharply near the head. The Qwen‑3‑14B and Llama‑3‑70B reports show this pattern cleanly. ([GitHub][10], [arXiv][2])

2. **Head‑level consolidation.** Many families only cross KL < 1 bit at (or extremely near) the final row, making **absolute probabilities** unreliable mid‑stack unless a learned translator (tuned lens) or a **shared decoder** (Logit Prism) is used. The project is correct to stick to **within‑model** probability comparisons and to emphasize **rank thresholds** across models until a tuned lens is added. ([GitHub][1], [Neural Blogs][7])

Interpretively, the results match standard mechanistic expectations: **FF layers act as value readouts / “key‑value memory,”** while **attention heads** route subject information, with relation‑specific content consolidating in the late stack—precisely where this probe sees `is_answer=True` emerging. This invites follow‑up that splits *attn‑vs‑MLP* causal roles around L\_sem. ([arXiv][12], [ACL Anthology][13])

---

## 4) Philosophical usefulness (nominalism vs realism)

**What the current sweep can support:**

* The **Δ between copy and semantics**, plus late **KL/cosine inflections**, shows that models **do more** than re‑emit present tokens; they **compute** relation‑level content that is **not** present as a contiguous prompt subsequence. This challenges **austere nominalism**, which would paraphrase model behavior as lists of token‑token co‑occurrences. (The Gemma copy‑reflex vs head‑only semantics is a vivid example.) ([GitHub][1])

**What it cannot yet decide:**

* Whether the *structure* responsible is best understood as **predicate‑level routines** (metalinguistic nominalism) or as **instantiated universals** (realism) requires **manipulative evidence** and **portability** across prompts, styles, and languages:

  * **Activation/attribution patching** to find **causal L\_sem** and partition **attn vs MLP** responsibility. ([Neel Nanda][14], [Neel Nanda][15])
  * **Concept vectors / low‑rank causal bases** whose injection **reliably** boosts the correct capital across *unseen* prompts. ([arXiv][2])
  * **Sparse autoencoders** to isolate **monosemantic features** that *predict and causally steer* `p_answer`—and that **generalize across paraphrases/languages**. ([OpenAI][16], [OpenAI CDN][17], [OpenReview][18])
  * **Permutation controls** (rename all countries) to ensure effects track the **relation**, not labels.

Until at least some of the above succeed, the realist reading is **under‑determined**; conversely, their success (especially under permutation and cross‑language perturbations) would significantly strengthen the anti‑nominalist case. ([GitHub][3])

---

## 5) Review of the project plan & concrete adjustments

The plan in `001_LAYERS_BASELINE_PLAN.md` already sketches most of the right upgrades. The following adjustments would add **non‑negligible** value quickly:

1. **Add a Logit‑Prism sidecar decode now (before tuned lens).**
   A single shared whitening+rotation decoder gives a stronger, lower‑variance baseline than raw norm‑lens for early layers and sharpens “rotation vs amplification” narratives—at trivial runtime cost (one extra matmul per layer). Keep norm‑lens as the primary; write Prism sidecar CSVs to compare **KL(P\_prism || P\_final)** at 25/50/75% depths. ([Neural Blogs][7])

2. **Implement the copy‑threshold sweep and report `L_copy(0.70/0.80/0.90)`.**
   This quantifies copy‑reflex robustness, crucial for claims about “surface vs semantics.” The plan already outlines this; shipping it will immediately improve the cross‑model synthesis. ([GitHub][3])

3. **Ship minimal attribution‑patching (3‑pass) around L\_sem for one model.**
   Start with the cleanest family (e.g., **Mistral‑Small‑24B** or **Llama‑3‑70B**, both with near‑zero final KL), and compute token×layer causal heat‑maps to *separate attention from MLP* around L\_sem. This is a **high‑leverage** step toward the philosophical pay‑off. ([GitHub][19], [Neel Nanda][14])

4. **One‑week sprint: narrow, reliability‑gated SAE pass (one model, 3 layers).**
   Follow the plan’s reliability gates (cross‑seed consensus; unitary vs bundle), and require **Δ log‑prob** improvements with **permutation** and **France** controls. This produces the first **feature‑level** causal evidence. ([GitHub][3], [OpenAI CDN][17])

5. **Automated fact‑checks in evaluations.**
   The cross‑doc mismatch noted in the synthesis suggests adding a tiny script that:

   * Recomputes and prints **first\_rank\_≤{1,5,10}**, **first\_KL\_<{1.0,0.5}**, and the first **cosine ≥ {0.2,0.4,0.6}** directly from the CSVs,
   * Then inlines those numbers into the markdown via templating.
     This will prevent stale or swapped values in write‑ups. ([GitHub][1])

6. **Broaden the prompt battery with *rank‑centric* metrics**
   Before multilingual and permutation passes, scale to a 100–1,000 prompt **country→capital** battery and report **rank milestones** (≤10, ≤5, =1) distributions *by model*. This is cheap, robust to lens calibration, and immediately strengthens claims about **where** semantics resolves.

7. **Tuned Lens integration (per model, seeded)**
   Once Prism is in, add tuned lens training checkpoints (frozen models, seed & corpus recorded). Use tuned logits for early‑layer statements; keep **last‑layer KL** and **raw‑vs‑norm** for sanity. ([arXiv][2])

---

## 6) Model‑by‑model highlights worth keeping (with cautions)

* **Meta‑Llama‑3‑70B**: Early semantic collapse (L\_sem = 40/80), very low raw‑vs‑norm risk, and a dramatic **cosine jump only at the head**—an informative counterexample to “late only.” Use this as a *baseline* for causal patching. ([GitHub][11])

* **Qwen‑3‑14B**: Late and sharp collapse (L\_sem = 36/40), **high** lens‑artifact risk mid‑stack ⇒ rely on **ranks** and within‑model KL, not absolute p’s. ([GitHub][10])

* **Qwen‑2.5‑72B**: Extreme head‑only consolidation (rank‑1 only at L80), punctuation dominance mid‑stack; a good candidate to demonstrate **“early direction, late calibration.”** ([GitHub][9])

* **Yi‑34B**: L\_sem ≈ 44/60 with “norm‑only semantics” flagged in raw‑vs‑norm sanity ⇒ treat early p’s cautiously; emphasize rank milestones. ([GitHub][6])

* **Gemma‑2 (9B/27B)**: **Copy‑reflex at layer 0–3** under strict rule; **semantics only at the head** with final‑layer KL ≈ 1 bit ⇒ stick to ranks. Also a great target for the **copy‑threshold sweep** and **permutation controls**. ([GitHub][1])

---

## 7) How these findings connect to current interpretability research

* The observed late consolidation and the need for a **translator** align with **Tuned Lens** (learned affine translators improve calibration/robustness). Adding TL strengthens early‑layer claims. ([arXiv][2])

* Splitting causal roles near L\_sem maps onto the **“KV‑memory”** view of FF layers and **routing** by attention; use **(attn‑only vs MLP‑only) patching** to confirm in‑model. ([ACL Anthology][13])

* **Attribution Patching** (gradient‑based) lets the project scale causal heat‑maps cheaply; combine with explicit patching for spot checks. ([Neel Nanda][14])

* **Sparse Autoencoders** can expose **portable, monosemantic features**; the OpenAI/ICLR work provides scalable recipes and evaluation criteria the project can adopt. ([OpenAI][16], [OpenAI CDN][17], [OpenReview][18])

* A **Logit Prism** baseline standardizes cross‑layer decoding without overfitting per layer and has already been field‑tested in the community. ([Neural Blogs][7])

---

## 8) Prioritized next actions (one page, do‑able now)

1. **Ship Logit Prism sidecar decode** (dual CSVs), re‑run one model per family, update cross‑model with **Prism vs norm‑lens** KL curves. ([Neural Blogs][7])
2. **Implement and publish copy‑threshold sweep**; add `L_copy(0.70/0.80/0.90)` to cross‑model synthesis. ([GitHub][3])
3. **Run attribution‑patching (3‑pass) around L\_sem** for **Llama‑3‑70B** and **Mistral‑Small‑24B**; report `causal_L_sem`, `causal_L_sem_attn`, `causal_L_sem_mlp`. ([GitHub][11], [Neel Nanda][14])
4. **Scale to 100–1,000 fact prompts** (rank milestones only); publish distributions of `first_rank_≤{10,5,1}` per model.
5. **Start one focused SAE pass** on a single model (3 layers, cross‑seed gating, permutation & France controls) and release **feature‑level** effects (Δ log‑prob). ([GitHub][3], [OpenAI CDN][17])
6. **Automate markdown numbers** from CSV (cosine/KL/rank milestones) to eliminate cross‑doc mismatches. ([GitHub][1])
7. **Then** integrate **Tuned Lens** to stabilize early‑layer probabilities; keep rank/KL as the primary cross‑model comparators regardless. ([arXiv][2])

---

### Citations to project artifacts

* Code & technical notes; run structure and normalization details. ([GitHub][4])
* Cross‑model synthesis. ([GitHub][1])
* Per‑model evaluations cited in text (examples): **Llama‑3‑70B** (early L\_sem), **Qwen‑3‑14B** (late, sharp L\_sem), **Qwen‑2.5‑72B** (head‑only), **Yi‑34B** (norm‑only semantics risk). ([GitHub][11])

### External references (selected)

* **Tuned Lens** (improves early‑layer decoding, causally aligned). ([arXiv][2])
* **RMSNorm** (ε inside √; rationale for RMS lens). ([arXiv][5])
* **Logit Prism** (shared decoder baseline). ([Neural Blogs][7])
* **FF layers as KV memory** (expect MLP causal role near L\_sem). ([ACL Anthology][13])
* **Attribution Patching** (scalable causal approximation). ([Neel Nanda][14])
* **Sparse Autoencoders** (interpretable monosemantic features at scale). ([OpenAI][16], [OpenAI CDN][17], [OpenReview][18])

---

## Bottom line

The current sweep is a **solid first iteration** that already contains multiple self‑checks and conservative interpretations. With a **Logit Prism** or **Tuned Lens** decode, a **copy‑threshold sweep**, and **one concrete causal pass** (attribution patching, then a focused SAE), the project will move from observational depth curves to **manipulative, portable structure**—the kind of evidence that materially advances the nominalism‑versus‑realism debate beyond mere token‑statistics.

[1]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-cross-models.md "raw.githubusercontent.com"
[2]: https://arxiv.org/abs/2303.08112?utm_source=chatgpt.com "Eliciting Latent Predictions from Transformers with the Tuned Lens"
[3]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_LAYERS_BASELINE_PLAN.md "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/NOTES.md "raw.githubusercontent.com"
[5]: https://arxiv.org/abs/1910.07467?utm_source=chatgpt.com "Root Mean Square Layer Normalization"
[6]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Yi-34B.md "raw.githubusercontent.com"
[7]: https://neuralblog.github.io/logit-prisms/?utm_source=chatgpt.com "Logit Prisms: Decomposing Transformer Outputs for ..."
[8]: https://transformerlensorg.github.io/TransformerLens/generated/demos/Main_Demo.html?utm_source=chatgpt.com "Transformer Lens Main Demo Notebook - GitHub Pages"
[9]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Qwen2.5-72B.md "raw.githubusercontent.com"
[10]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Qwen3-14B.md "raw.githubusercontent.com"
[11]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Meta-Llama-3-70B.md "raw.githubusercontent.com"
[12]: https://arxiv.org/abs/2012.14913?utm_source=chatgpt.com "Transformer Feed-Forward Layers Are Key-Value Memories"
[13]: https://aclanthology.org/2021.emnlp-main.446.pdf?utm_source=chatgpt.com "Transformer Feed-Forward Layers Are Key-Value Memories"
[14]: https://www.neelnanda.io/mechanistic-interpretability/attribution-patching?utm_source=chatgpt.com "Attribution Patching: Activation Patching At Industrial Scale"
[15]: https://neelnanda.io/attribution-patching-demo?utm_source=chatgpt.com "Attribution Patching Demo - Colab"
[16]: https://openai.com/index/extracting-concepts-from-gpt-4/?utm_source=chatgpt.com "Extracting Concepts from GPT-4"
[17]: https://cdn.openai.com/papers/sparse-autoencoders.pdf?utm_source=chatgpt.com "Scaling and evaluating sparse autoencoders"
[18]: https://openreview.net/forum?id=F76bwRSLeK&utm_source=chatgpt.com "Sparse Autoencoders Find Highly Interpretable Features in ..."
[19]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Mistral-Small-24B-Base-2501.md "raw.githubusercontent.com"

