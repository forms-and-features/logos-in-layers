# Review of “Logos‑in‑Layers” (Experiment 001)

**Scope.** This review audits the code path and measurement choices, checks the per‑model and cross‑model write‑ups against the artifacts, performs an independent synthesis of patterns visible in the runs, and proposes concrete next steps—especially those that bear on the philosophical target (nominalism vs realism).

---

## Executive summary

* **Measurement is mostly sound and well‑instrumented.** The run records *rank milestones*, KL to the final head, cosine‑to‑final, and (where applicable) copy collapse; it also logs norm‑vs‑raw lens divergence and final‑layer calibration. These are exactly the right guardrails for a logit‑lens baseline. The code claims (and diagnostics corroborate) that RMS/LN are applied in fp32 to the correct post‑block residual, with family‑aware hooks and fp32 unembed for decoding. ([GitHub][1])
* **Empirical pattern:** In nine of ten models here, **semantic collapse arrives late** (≥70% of depth); *Llama‑3‑70B* is a clear early outlier (≈50%). *Gemma‑2* uniquely fires the strict copy rule in the very first layers while postponing semantics to the very top. Prism sidecars are consistently **regressive or neutral** in these sweeps. ([GitHub][2])
* **Over‑statements avoided, but two corrections help.** (a) Treat Gemma probabilities as **mis‑calibrated** relative to the final head and prefer *ranks* there; (b) treat early “semantics” with caution anywhere the **norm‑only** flag is raised (e.g., Llama‑3‑8B, Yi‑34B). Both are already noted; the review reinforces them and suggests tuned‑lens checks to strengthen claims. ([GitHub][3])
* **Philosophical read:** The data already exerts pressure against *austere nominalism*. The same gold token (“Berlin”) emerges reliably from very different token soups across families—*and* the direction to the final head often forms **before** amplitudes calibrate—suggesting reusable internal structure rather than mere surface copying. That is not yet a knock‑down case for **realism** over **metalinguistic nominalism**, but it sets up decisive tests (feature reuse across prompts/languages; causal interventions on concept features). ([GitHub][2])

---

## Code & measurement review

**What’s right (and important):**

* **Architecture‑aware normalization on the right residual.** The notes specify *post‑block* decoding with the right norm (LN2 for post‑norm, next‑block LN1 for pre‑norm), ε *inside* √ for RMS, and all LN/RMS statistics in fp32. The run metadata for multiple models shows these flags in place (e.g., `use_norm_lens: true`, RMS types, positional info). This addresses the classic source of spurious “early meaning.” ([GitHub][1])
* **Final‑head consistency check.** Each run compares the last post‑block lens distribution to the model’s actual output head; Gemma is correctly flagged as divergent (≈1 bit KL), others near zero. This is critical for trusting within‑model probabilities. ([GitHub][3])
* **Raw‑vs‑norm lens sanity.** The JSON captures `max_kl_norm_vs_raw_bits` and flags the first *norm‑only* semantic layer where applicable, correctly warning that early “semantics” may be lens‑induced (e.g., Llama‑3‑8B, Yi‑34B). Good practice. ([GitHub][4])
* **Rank‑first reporting.** Adding `answer_rank` and `first_rank_le_{10,5,1}` is the right way to stabilize across families; this is used throughout the evals. ([GitHub][5])
* **Copy detection at ID level with provenance.** The strict subsequence + p‑margin rule is logged and, in these runs, fires early only on Gemma. That behavior is plausible and consistent across variants. ([GitHub][3])

**Issues / risks and targeted fixes:**

1. **Copy rule too strict to be informative outside Gemma.** With `τ=0.95` and `k=1` (plus the margin), most families never set `L_copy`. That loses a useful alignment baseline for Δ‑collapse. Keep the strict rule for “hard copy,” but also log a **soft copy index** (e.g., first layer with *any* k∈{1,2,3} token‑ID window copied at p>0.5). This allows Δ comparisons everywhere without compromising the main metric. (Implementation: second set of fields `L_copy_soft`, `k_copy_soft`.) ([GitHub][2])
2. **Cross‑model entropies are still lens‑sensitive.** The notes already caution this; add **optional tuned‑lens translators** to check whether *direction vs calibration* narratives persist when per‑layer projections are trained. Tuned‑lens has repeatedly out‑performed raw logit lens on fidelity to final distributions. ([arXiv][6])
3. **Prism sidecar unclear.** The sidecar consistently fails to match final‑head distributions and never improves rank milestones. Unless Prism is vital for some later step, gate it behind a `--prism` flag and quarantine in the write‑ups (“regressive/neutral in these runs; excluded from claims”). ([GitHub][2])
4. **Single‑prompt sensitivity.** The run uses one positive prompt (with a no‑filler ablation) and one control. That is fine for a baseline, but the copy statistic and collapse depth are known to be tokenization‑dependent. The plan should explicitly schedule a small **prompt sweep** (synonyms, different word orders, quote/no‑quote, multilingual variants) and report **per‑variant** `L_sem` distributions. (The current ablation results—mostly ΔL\_sem≈0, small ±2 effects—are a helpful first step.) ([GitHub][7])
5. **Cosine metric: document exactly what’s compared.** The pipeline computes cosine between *logit* directions (per layer vs final). That’s good; it should be highlighted more prominently in reports because it supports the “rotation vs amplification” discussion without leaning on lens calibration. ([GitHub][1])

---

## Spot‑checks on the per‑model reports

These checks confirm the reports’ main claims against the artifacts; they also highlight any over‑reach.

* **Llama‑3‑8B.** `L_semantic = 25/32`; early “semantics” flagged as *norm‑only*; final KL≈0. All shown in the JSON and CSV. The report appropriately leans on rank milestones. ([GitHub][4])
* **Mistral‑7B‑v0.1.** `L_semantic = 25/32`; no copy hits; KL→0 only at final; cosine crosses ≥0.4 near 25–26. The report’s “early direction, late calibration” summary is supported. ([GitHub][8])
* **Mistral‑Small‑24B.** `L_semantic = 33/40`; no copy; ablation shifts semantics earlier by 2 layers (−2). Matches JSON/CSV. ([GitHub][7])
* **Llama‑3‑70B.** Early relative collapse `L_semantic = 40/80`; ablation delays semantics by +2 layers; final KL≈0. Correct. ([GitHub][9])
* **Qwen‑3‑8B.** `L_semantic = 31/36`; no copy; final KL≈0; Prism regresses badly. Correct and appropriately cautious about *lens\_artifact\_risk*. ([GitHub][10])
* **Qwen‑3‑14B.** `L_semantic = 36/40`; no copy; ΔH≈+0.50 bits around collapse; Prism regresses. The “late snap” narrative is supported. ([GitHub][11])
* **Qwen‑2.5‑72B.** Latest collapse `L_semantic = 80/80`; clean final calibration; cosine aligns earlier while KL stays high—textbook rotation→late calibration. ([GitHub][12])
* **Yi‑34B.** `L_semantic = 44/60`; *norm‑only* semantic flag in diagnostics; report handles this with rank‑first language. ([GitHub][13])
* **Gemma‑2‑27B.** **Copy at L0** with `p≈0.99998` for “simply”; **semantics only at L46/46**; last‑layer KL≈1.14 bits with temperature probe—report correctly urges rank‑first claims. ([GitHub][3])
* **Gemma‑2‑9B.** Same family signature (copy at bottom, semantics at top; non‑zero final KL). The individual report also notes this; the cross‑model synthesis summarizes it as a Gemma‑specific trait. ([GitHub][14])

**Verdict:** The per‑model and cross‑model write‑ups are well‑aligned with the artifacts. Where the reports call out lens artifacts or calibration issues, those flags are justified by diagnostics. The only systematic messaging improvement would be to push *rank‑first* language even harder anywhere `lens_artifact_risk = high`.

---

## Independent cross‑model synthesis

**Late semantics is the default; early collapse is family‑specific.** Using `L_semantic / num_layers` as a relative depth index, all families collapse late except *Llama‑3‑70B* (\~50%). *Qwen‑2.5‑72B* and *Gemma‑2* collapse last (100%); *Mistral* and *Llama‑3‑8B* sit around \~78–83%; *Yi‑34B* around \~73%. ([GitHub][2])

**Copy vs semantics vary by family.** The strict copy detector fires at the very bottom only for *Gemma‑2*; others show no hard copy reflex but still exhibit form/punctuation anchoring mid‑stack (e.g., “simply”, quotes). This family contrast is crisp in CSVs. ([GitHub][3])

**Direction vs calibration.** Cosine‑to‑final tends to rise before KL collapses, indicating **rotation first, then amplification/calibration** (e.g., Mistral‑7B, Qwen‑3‑14B), with *Llama‑3‑70B* a “late‑direction, late‑calibration” outlier (cosine jumps very late). These signatures match tuned‑lens findings that per‑layer translators reduce brittleness and recover smoother trajectories. ([GitHub][8])

**Prism is non‑helpful here.** Across models, Prism increases KL at mid‑depth and never improves rank milestones; in some cases it fails even at the final layer. In these data, it should be excluded from conclusions. ([GitHub][2])

---

## Relation to the interpretability literature

* The **tuned lens** replaces the raw logit lens with per‑layer affine translators and is known to produce distributions that are **closer to the final head** and more faithful to internal features. This directly underwrites the plan to check rotation vs calibration with a stronger lens. ([arXiv][6])
* **Induction heads** and related copy circuits explain early template/pattern effects (e.g., quote marks, “simply”) before factual content consolidates—consistent with mid‑stack filler anchoring observed here. ([Transformer Circuits][15])
* **Causal scrubbing** (behavior‑preserving resampling ablations) is an appropriate method to *validate* any hypothesized “capital‑of” circuit discovered via activation patching or SAEs. It should be introduced once specific features/heads are implicated. ([LessWrong][16])
* **Sparse autoencoders / monosemantic features** provide a practical way to search for reusable *concept* directions (e.g., “capital city”, “country”) and then causally test them. This is the most promising bridge from measurement to the realism/nominalism question. ([Anthropic][17])

---

## Philosophical relevance (nominalism vs realism)

**What these runs already show.** For the same prompt family, ten architectures traverse very different early‑layer token mixtures yet nonetheless converge to the *same answer token* at predictable depths. In several families, *direction* towards the final solution emerges before *calibration*. That pattern strongly suggests **internal, reusable structure** beyond mere token‑level copying—evidence that strains **austere nominalism** (which denies the need for properties/relations over and above word tokens). ([GitHub][2])

**Why this is not yet decisive.** A metalinguistic nominalist can still claim the structures are about **words** (e.g., distributional facts about “capital” and “Germany”) rather than mind‑independent universals. To distinguish **realism** from **metalinguistic nominalism**, the project must: (a) show that *the same* internal features mediate many distinct linguistic realizations (paraphrases, languages, scripts), and (b) show causal necessity/sufficiency of those features for correct behavior across contexts *without* lexical cues.

**Testable predictions for the next iteration:**

* If there is a **universal‑like “capital‑of” feature**, then a sparse set of features (or a small head‑MLP subcircuit) should causally mediate many `(country → capital)` prompts, in multiple languages, and should fire even when *surface* cues are removed or replaced with adversarial templates.

---

## Actionable next steps (high value)

**A. Strengthen the measurement**

1. **Add tuned‑lens translators** (per‑layer affine probes trained on held‑out text) and re‑plot rank/KL/cosine curves; keep raw lens for comparability but treat tuned‑lens as primary when `lens_artifact_risk = high`. (Small engineering lift; large payoff in robustness.) ([arXiv][6])
2. **Dual copy metrics.** Keep strict `L_copy` (τ=0.95, k=1) but add a *soft* copy index and a windowed variant (k∈{1,2,3}). Report Δ between copy and semantics under both to avoid “null everywhere” in non‑Gemma families. ([GitHub][2])
3. **Prompt family sweep (10–20 variants).** Swap “simply”, alter word order, add/remove quotes, translate to DE/FR/ES/AR/JP, use clozes (`Germany → [ ]`). Report distributions of `L_sem` and early cosine across variants. (This directly addresses tokenization sensitivity noted in the plan.) ([GitHub][1])
4. **Retire Prism from conclusions** (or document it as “diagnostic only”). It is regressive in these runs and confuses calibration stories. ([GitHub][2])

**B. Move from correlational to causal evidence**

5. **Activation patching + causal scrubbing.** Use automated *position‑localized* activation patching to identify heads/MLP layers that move `answer_rank` and `p_answer`; then run **causal scrubbing** to test whether the hypothesized circuit truly mediates the behavior. (E.g., resample non‑causal parts while preserving the causal path.) ([LessWrong][16])
6. **SAE feature mining at the emergence layers.** Train sparse autoencoders on residuals near `L_sem` (±4 layers) to discover features that correlate with `(country, capital)` and related semantic fields (e.g., “is‑a‑capital”, “is‑a‑country”, “capital‑of” relation). Track a small set of candidate features across prompts. ([Anthropic][17])
7. **Causal mediation tests of features.** Once features are identified, (i) *knock‑out* (zero or noise) and measure `answer_rank` degradation; (ii) *knock‑in* (add scaled feature) in harder prompts (paraphrases; other languages) and test `answer_rank` improvements. Use scrubbing to ensure the mediation path matches the hypothesis. ([LessWrong][16])

**C. Experiments tailored to realism vs nominalism**

8. **Cross‑lingual invariance.** Construct a multilingual benchmark of `(country → capital)` covering different scripts and tokenization regimes. Success criterion: the *same* features/heads mediate answers across languages (feature reuse), not just per‑language lexical cues.
9. **Paraphrase & anti‑lexical controls.** Use prompts that avoid the word *capital* (e.g., “Germany’s seat of government is the city of …”) and adversarially *replace* nearby lexical scaffolds (“simply”, quotes, punctuation). Recovering semantics with the same features under these manipulations undercuts metalinguistic accounts.
10. **Relational generalization.** Add other binary relations (“river‑in‑country”, “author‑of‑book”) and test whether relation‑features discovered with SAEs are re‑used across domains—evidence of **universals** vs mere lexical co‑occurrence.

---

## Specific feedback on the current plan (PROJECT\_NOTES.md)

* The **“Get the measurement right”** checklist is excellent and (per diagnostics) implemented: correct RMS/LN math, fp32 stats, post‑block lensing, raw‑vs‑norm checks, cosine, last‑layer calibration. Keep this as non‑negotiable QA. ([GitHub][1])
* Add two line‑items to “Next steps”: (a) **Tuned‑lens integration** (with a clear on/off provenance flag) and (b) **soft copy index** as above. Both materially improve interpretability without bikeshedding. ([arXiv][6])
* In analysis notebooks, prioritize **rank milestones and cosine** for cross‑model claims; relegate KL and entropies to within‑model narrative, especially in families with non‑zero final‑layer KL (Gemma). ([GitHub][3])
* Explicitly **demote Prism to exploratory** until a variant is shown to *improve* rank milestones or match the final head in at least one family. ([GitHub][2])

---

## Appendix: representative evidence snippets

* *Mistral‑7B‑v0.1:* `L_sem = 25/32` (rank‑1 “Berlin” at L25); KL→0 only at final; cosine ≥0.4 at L25/L26—rotation precedes calibration. ([GitHub][8])
* *Llama‑3‑8B:* `L_sem = 25/32`; norm‑only flag at L25; final KL≈0; negative control clean. ([GitHub][4])
* *Llama‑3‑70B:* `L_sem = 40/80` (early among families); ablation +2 layers; final KL≈0. ([GitHub][9])
* *Mistral‑Small‑24B:* `L_sem = 33/40`; ablation −2 layers (earlier semantics); final KL≈0. ([GitHub][7])
* *Qwen‑3‑14B:* `L_sem = 36/40`; late snap with ΔH≈+0.50 bits; Prism regressive. ([GitHub][11])
* *Qwen‑2.5‑72B:* `L_sem = 80/80`; cosine aligns early while KL collapses only at final—late calibration. ([GitHub][12])
* *Yi‑34B:* `L_sem = 44/60`; norm‑only semantics flag at 46; rank‑first reporting. ([GitHub][13])
* *Gemma‑2‑27B:* `L_copy = 0`; `L_sem = 46/46`; final KL≈1.14 bits with temperature probe; copy reflex at layer 0 (“simply”). ([GitHub][3])

---

### References (selected)

* **Tuned Lens**: affine per‑layer translators improving fidelity vs logit lens; used here as a recommended upgrade. ([arXiv][6])
* **Induction Heads**: canonical copy/completion circuits, explaining form anchoring before content. ([Transformer Circuits][15])
* **Causal Scrubbing**: behavior‑preserving resampling testbed for validating circuit hypotheses. ([LessWrong][16])
* **Monosemantic features / SAEs**: dictionary‑learning approach to identify reusable concept features. ([Anthropic][17])

---

## Bottom line

As a **baseline**, this run is in strong shape: the right metrics are logged, the right sanity checks are in place, and the write‑ups mostly lean on rank‑ and cosine‑based claims. The next iteration should (i) add tuned‑lens translators, (ii) broaden copy detection slightly, (iii) retire Prism from conclusions, and (iv) pivot from descriptive to **causal** analyses (activation patching + scrubbing; SAE features). Those changes will transform a plausible anti‑*austere nominalism* gesture into a rigorous, falsifiable argument about **reusable internal structure**—and put the project in a position to probe **realism vs metalinguistic nominalism** with genuine teeth. ([GitHub][1])

[1]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/PROJECT_NOTES.md "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-cross-models.md "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-gemma-2-27b.md "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Meta-Llama-3-8B.md "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/NOTES.md "raw.githubusercontent.com"
[6]: https://arxiv.org/abs/2303.08112?utm_source=chatgpt.com "Eliciting Latent Predictions from Transformers with the Tuned Lens"
[7]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Mistral-Small-24B-Base-2501.md "raw.githubusercontent.com"
[8]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Mistral-7B-v0.1.md "raw.githubusercontent.com"
[9]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Meta-Llama-3-70B.md "raw.githubusercontent.com"
[10]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Qwen3-8B.md "raw.githubusercontent.com"
[11]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Qwen3-14B.md "raw.githubusercontent.com"
[12]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Qwen2.5-72B.md "raw.githubusercontent.com"
[13]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Yi-34B.md "raw.githubusercontent.com"
[14]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/output-gemma-2-9b-records.csv "raw.githubusercontent.com"
[15]: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html?utm_source=chatgpt.com "In-context Learning and Induction Heads"
[16]: https://www.lesswrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing?utm_source=chatgpt.com "Causal Scrubbing: a method for rigorously testing ..."
[17]: https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning?utm_source=chatgpt.com "Decomposing Language Models With Dictionary Learning"

---

Produced by OpenAI GPT-5 Pro
