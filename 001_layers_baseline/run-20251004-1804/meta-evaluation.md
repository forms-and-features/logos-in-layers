# GPT-5 Pro: Review of “Logos‑in‑Layers: a logit‑lens baseline sweep over open‑weight base LLMs”

---

## Executive summary

The project builds a careful, architecture‑aware **logit‑/norm‑lens** pipeline and applies it to ten open‑weight base LLMs to compare **surface‑form copy** vs **semantic collapse** across depth. The code implements multiple best‑practice guardrails (correct RMS/LN normalization; fp32 statistics; raw‑vs‑norm checks; rank‑based milestones; calibration sanity on the final head). The outputs are internally consistent, and the per‑model JSON meta files document the run conditions and diagnostics clearly (e.g., “layer0_position_info”, “last_layer_consistency”, “raw_lens_check”). For several families (e.g., Gemma‑2, Qwen‑3, Mistral‑7B), the runs replicate a now‑familiar pattern: **early direction, late calibration**—cosine to the final head rises relatively early while KL to the final head falls late; the answer reaches rank‑1 only near the end. This is directly aligned with the literature’s critique that the vanilla logit lens is brittle and calibration‑sensitive; the project’s optional **Tuned‑Lens** sidecars generally reduce KL and shift rank milestones in the expected direction. ([GitHub][1])

Where the evaluation over‑reaches, it is mostly in cross‑family generalization and occasional indexing inconsistencies (baseline vs tuned‑lens summaries). The single‑item prompt is a deliberate scoping guard, but it constrains philosophical leverage; stronger claims against austere nominalism will require **causal** and **cross‑context** evidence (e.g., feature‑level interventions and invariances across many relations/entities). The plan in `001_LAYERS_BASELINE_PLAN.md` is sensible; a few focused additions would materially raise the evidential value without bike‑shedding. ([GitHub][2])

---

## 1) Method & code review

**What’s right (and important):**

* **Architecture‑aware normalization.** The pipeline picks the correct normalizer per architecture (pre‑norm vs post‑norm) and applies RMS ε **inside** the square root; statistics are computed in fp32 before cast‑back. This addresses a well‑known source of logit‑lens artefacts. The JSONs record `norm_alignment_fix`, `layer0_norm_fix`, and fp32 unembed policy; runs after 2025‑08‑24 are explicitly marked as having the §1.1 fix. This matches the RMSNorm paper’s formula and removes spurious early “semantics”. ([GitHub][2])

* **Final‑head calibration sanity.** Each run records last‑layer KL between the logit‑lens distribution and the **true** model head; e.g., Gemma‑2‑9B warns `kl_to_final_bits≈1.01` and sets “suppress_abs_probs=true”, steering analysis to **ranks** rather than absolute p. This is precisely the right discipline for lens‑based interpretability. ([GitHub][3])

* **Raw‑vs‑norm checks.** The project bakes in comparisons that flag **norm‑only semantics** and quantifies `kl_norm_vs_raw_bits`, with a tiered “lens_artifact_risk”. This is crucial because the vanilla logit lens can hallucinate early meaning; the presence of norm‑only rank‑1 layers triggers caution. ([GitHub][1])

* **Robust copy detector.** Prompt‑echo is defined at the **token‑ID** level with a probability threshold and a rolling window, avoiding string‑level pitfalls. Provenance (`copy_thresh`, `copy_window_k`, `copy_match_level`) is emitted to JSON; soft detectors are included. ([GitHub][2])

* **Rank‑based milestones & KL percentiles.** The JSON summaries standardize `first_rank_le_{10,5,1}` and `first_kl_below_{1.0,0.5}`, enabling within‑model comparisons that are less calibration‑sensitive. ([GitHub][1])

**Issues & nits:**

* **Indexing consistency (minor but confusing).** In several JSONs the **tuned‑lens** `L_semantic`/`L_copy` are 1‑indexed by layer count (e.g., Gemma‑2‑9B reports `L_semantic=42` for a 42‑layer model), while baseline per‑layer CSV/flags are 0‑indexed with `layer=n_layers` reserved for the final unembed row. Cross‑document prose sometimes mixes these conventions (e.g., “L_copy=0” vs “L_copy=1”). Recommend standardizing on **post‑block 0…n‑1** plus **final head = n**, and emitting an explicit `indexing_scheme` field for each block to prevent off‑by‑one confusion. ([GitHub][3])

* **Copy‑collapse threshold is very strict.** With `p_top1>0.95` and margin gating, strict copy rarely fires outside Gemma; this is fine as a **high‑precision** detector, but it leads to many “null” `L_copy_strict` values. The project partly compensates with `topk_prompt_mass@50`, but it would help to report a **soft echo index** (e.g., earliest layer where **any** prompt token enters top‑k with p>τ) to improve recall while keeping the strict detector unchanged. (The JSON already logs coverage and echo mass; surfacing a single “soft echo milestone” in the summary would simplify cross‑model comparison.) ([GitHub][3])

* **Single‑prompt design limits external validity.** The project acknowledges this. The paraphrase set and the France→Paris control are helpful, but many conclusions (e.g., family‑specific “copy reflex”) should be phrased as **item‑level observations** until replicated across a suite of relations/entities. ([GitHub][1])

* **Prism is reported, not validated.** The “Logit Prism” sidecars are used as a learned calibration overlay; in these runs they are consistently **regressive** (higher KL; no earlier ranks). That’s fine as a diagnostic, but any statements implying “better semantics” under Prism should be avoided unless backed by rank improvements or causal support. ([GitHub][1])

---

## 2) Results & analysis review (checks, over‑statements, misses)

**Per‑model checks (spot‑verified against JSON):**

* **Meta‑Llama‑3‑8B.** Baseline (norm lens) puts the answer at rank‑1 by **L=25** and reaches low KL to final head only at the end; measurement guidance prefers **Tuned‑Lens** (ΔKL ≈ +4 bits improvement at median) and emphasizes ranks over probabilities. This is a textbook **early‑direction, late‑calibration** profile. The “no‑filler” ablation leaves L_sem unchanged. These points are correctly reflected in the evaluation notes. ([GitHub][1])

* **Mistral‑7B‑v0.1.** Baseline yields first rank‑1 at **L≈25/32**; tuned‑lens summaries report mixed `L_semantic` across runs (25 and 32) with strong KL reductions; guidance again prefers ranks. No strict/soft copy triggers. These are reported faithfully. ([GitHub][4])

* **Qwen‑3‑8B / Qwen‑3‑14B.** Very late collapse (≈**31–34/36** for 8B; ≈**34–39/40** for 14B) with tuned lenses improving KL by ~4 bits at p50 and nudging ranks later by a few layers. Evaluations that frame these as “late semantics” are supported; any cross‑family ranking should be caveated per measurement guidance. ([GitHub][5])

* **Gemma‑2‑9B.** Strong **early copy** on the filler token (“simply”) at **L0** under the strict rule; `warn_high_last_layer_kl=true` indicates calibrated final head vs lens mismatch. Semantic rank‑1 arrives only at the final layer (42). The evaluation’s emphasis on using ranks (and caution on absolute probabilities) is appropriate here. ([GitHub][3])

* **Meta‑Llama‑3‑70B.** Tuned‑Lens is missing; guidance falls back to the norm lens with rank‑based milestones. Any claim that this family is “earlier” at scale should be explicitly labelled **suggestive** (no tuned‑lens corroboration; single item). ([GitHub][6])

**Cross‑model synthesis (where to be careful):**

* The cross‑model memo groups models into “late (≥70% depth)” vs “early (<70%)” collapse buckets and correlates earlier collapse with better external metrics within the **Llama‑3** family. That read is **plausible but not causal**; given (i) single item; (ii) mixture of baseline vs tuned lenses; and (iii) norm‑only semantics flags in some families, the write‑up should be kept descriptive (“observed patterns”) rather than explanatory. ([GitHub][7])

**Two things the evaluation under‑emphasizes:**

* **Norm‑only semantics prevalence.** Several JSONs explicitly flag **norm‑only** rank‑1 layers in windows around the reported collapse (e.g., medium/high tiers for Llama‑3‑8B and several Qwen runs). The write‑up should foreground that rank milestones are reported as **confirmed semantics** (present in the JSON) or, if not, be explicitly tagged as lens‑sensitive. ([GitHub][1])

* **Calibration mismatch families.** The final‑head vs lens KL is particularly high for Gemma‑2; this materially affects entropy/absolute p curves. Analyses should avoid **probability magnitudes** and stick to **ranks** and **KL percentiles** in these families (the JSON already warns about this). ([GitHub][3])

---

## 3) Independent deep dive (layer‑wise patterns, with literature context)

**A. Early direction, late calibration**
Across models, cosine to the final head rises well before KL drops below 1 bit and before the answer reaches rank‑1; this is the “iterative refinement” trajectory emphasized by **Tuned‑Lens**: hidden states become **linearly more decodable** into the final prediction after an affine transform learned per layer. The project’s tuned‑lens sidecars replicate the literature’s core finding: tuned probes reduce KL and yield more reliable, layer‑wise predictions than vanilla logit lenses. Recommendation: when tuned sidecars are present, report **tuned rank milestones** as the primary semantic index, keeping baseline (norm) as a diagnostic. ([arXiv][8])

**B. Copy reflex is family‑ and prompt‑dependent**
The strict copy detector fires immediately on Gemma‑2‑9B (“simply”) but not on Mistral, Qwen, or Llama in this item. This suggests either (i) tokenizer/filler interactions (Gemma tokenizes the filler as a high‑mass singleton); (ii) family‑specific early heads biased to prompt echo; or (iii) the threshold being too strict to capture “soft echo” elsewhere. A **soft echo milestone** (earliest layer where **any** prompt token is top‑k with p>τ) would convert the existing coverage/echo mass columns into a single, comparable datum. ([GitHub][3])

**C. Lens artefact risk is real and varies by family**
Raw‑vs‑norm windows show **norm‑only** semantics around claimed collapses for several runs and large `kl_norm_vs_raw_bits` in Qwen and Yi families. This aligns with longstanding critiques that the vanilla logit lens is **brittle** and can hallucinate structure without a learned translator. The project’s design already leans into **ranks** and “confirmed semantics”; continuing to privilege these is the right call. ([GitHub][1])

**D. Why this matters for interpretability methods**
The trajectory the project recovers—early approximate direction, late precise calibration—matches the broader picture from the literature that intermediate states are **linearly related** to the model’s emergent predictions but require **calibration** to decode faithfully (Tuned‑Lens). Because normalization layers materially change decodability, careful RMS/LN handling (ε inside √; correct γ) is not mere engineering—it is epistemically load‑bearing for any claims about “where semantics live”. ([arXiv][8])

---

## 4) Usefulness for the nominalism vs realism debate (first iteration)

This first sweep already produces **anti‑austere‑nominalist** pressure, but only modestly:

* The same **answer direction** becomes increasingly decodable across depth, and this behavior is **model‑internal** rather than a mere echo of the prompt. Under tuned lenses (when present), the **same layer‑wise hidden states** yield consistent predictions even when the **surface form** (e.g., punctuation, the filler “simply”) competes for mass mid‑stack. That is, the model appears to carry **reusable internal structure** that is not reducible to one‑off string copying. ([GitHub][1])

* However, a determined austere nominalist can still claim that these are just **algorithmic regularities** about **token sequences**, not evidence for any stable internal **properties**—especially given the known brittleness of vanilla lenses. To move the needle philosophically, the project will need **causal** and **invariance** evidence: the same internal **feature(s)** should carry the “capital‑of(country)” relation across many contexts, paraphrases, and languages, and intervening on that feature should **systematically** toggle the behavior. This is the kind of evidence that begins to strain austere nominalism.

The **superposition** literature shows why causal, feature‑level tests are necessary: many concepts live in overlapping linear subspaces; lenses that only look at logits can confound multiple features. Bringing **sparse autoencoder (SAE)** tooling to the same residual streams can identify **monosemantic features** and test whether the **same** feature generalizes across prompts/models. ([arXiv][9])

---

## 5) Targeted recommendations for the **next steps** (high value, low bike‑shedding)

1. **Standardize indexing & “confirmed semantics” reporting.**
   Emit `indexing_scheme` and report **confirmed** rank milestones (norm ∧ raw, or tuned where available) as primary. Keep baseline/norm and Prism strictly diagnostic. This removes the main source of interpretive friction in the current artifacts. ([GitHub][1])

2. **Add a soft echo milestone to the summary.**
   From existing columns (`topk_prompt_mass@50`, echo mass), derive “earliest layer where any prompt token is top‑k with p>τ”. Keep the strict copy detector unchanged to preserve a high‑precision flag. This will turn many “null” copy cases into a graded, comparable signal. ([GitHub][3])

3. **Scale the item set with minimal lift (strongest ROI).**
   Run the exact pipeline on ~200 templated prompts across 3 relations: `capital_of`, `currency_of`, `language_of` and 2 languages (EN/DE or EN/FR), with ~50 entities each. This immediately enables **variance estimates**, **family‑level** comparisons, and **invariance** checks (e.g., filler words, punctuation). Keep ranks/KL within‑model; do **not** over‑interpret cross‑family absolute probabilities. (The code already supports batch evaluation; only prompt generation and run management are needed.)

4. **Add one causal test per relation (surgical but decisive).**
   Use a **sparse autoencoder** trained on the residual streams near the collapse window (e.g., Llama‑3‑8B layers 20–30). Identify features whose linear readout predicts the **first answer token** direction; then **activate/suppress** those features (as in SAE ablation papers) and measure causal impact on rank. This connects the lens‑level story to **feature‑level mechanisms**, which is where philosophical traction lives. (The Anthropic SAE toolkit and similar repos provide ready‑to‑use baselines.) ([arXiv][9])

5. **Prefer Tuned‑Lens where available; add missing sidecars.**
   In every model where the tuned translator is available (several already are), report **tuned** rank/KL milestones as the primary series. For models where tuned sidecars are missing (e.g., Llama‑3‑70B in these runs), either train a translator or clearly mark those rows as “baseline only—diagnostic”. The tuned lens is strictly better‑behaved than the vanilla lens for this use‑case. ([arXiv][8])

6. **Tighten cross‑model claims to rank/KL thresholds.**
   Keep cross‑family statements descriptive and anchored to: (i) **normalized depth** of `first_rank_le_1`; (ii) **KL percentiles**; (iii) presence/absence of **norm‑only semantics** flags. Avoid statements that implicitly compare absolute probabilities or entropies across families, especially those with `warn_high_last_layer_kl`. ([GitHub][3])

7. **Document the Prism verdict policy and keep it diagnostic.**
   Given that all observed prisms in these runs are **Neutral/Regressive** (inflated KL, no earlier ranks), keep Prism as a **robustness probe** only; do not treat it as an “improved” translator without rank improvements. A short note in the README with the present criteria (“Helpful = earlier/equal first_rank_le_1 and no KL inflation at p50”) would prevent misreadings. ([GitHub][7])

---

## 6) Contextualization with current literature

* **Logit‑lens brittleness & motivation for tuned lenses.** Belrose et al. introduce the **Tuned‑Lens** as an affine probe per block that better elicits latent predictions—more predictive, reliable, and less biased than vanilla logit lenses—exactly the regime this project targets. The project’s sidecar results mirror those improvements and its measurement guidance correctly prioritizes ranks. ([arXiv][8])

* **Normalization matters.** The RMSNorm paper specifies ε **inside** √ and only re‑scales (no re‑centering). Getting this wrong can distort decodability; the project’s §1.1 fix is therefore critical. ([arXiv][10])

* **Superposition & feature‑level causality.** The **Toy Models of Superposition** program shows that concepts are often encoded in overlapping directions; lens‑level observations can conflate features. Bringing **sparse** features into the loop (and toggling them) is the right next step to make stronger philosophical claims. ([arXiv][9])

---

## 7) Final verdict

Methodologically, the project is **sound and careful** for a first‑pass logit‑lens sweep. The outputs are trustworthy **within models** when interpreted through **rank milestones** and **KL** thresholds, especially with tuned‑lens overlays. The strongest empirical patterns—**early direction, late calibration**; **Gemma‑style early copy**; **very late semantic collapse for Qwen‑3**—are supported by the JSON diagnostics and match expectations from the tuned‑lens literature. The main fixes needed are **indexing clarity**, **soft echo milestone reporting**, and **scaling to multiple items**.

Philosophically, the present sweep offers **initial** pressure against **austere nominalism** by revealing **reusable internal structure** that is not reducible to prompt copying. Converting that into a compelling case will require **causal** feature‑level evidence and **invariance** across many contexts—well within reach with modest additions to the current plan. ([GitHub][1])

---

### Pinned evidence (non‑exhaustive)

* Meta‑Llama‑3‑8B: rank‑1 at L≈25 (baseline), tuned preferred; raw‑vs‑norm medium risk. ([GitHub][1])
* Mistral‑7B: rank‑1 at L≈25; tuned reduces KL; no copy reflex. ([GitHub][4])
* Qwen‑3‑8B/14B: late collapse (≈31–34/36; ≈34–39/40); tuned reduces KL ~4 bits @ p50. ([GitHub][5])
* Gemma‑2‑9B: strict copy at L0; warn_high_last_layer_kl; rank‑1 only at final layer. ([GitHub][3])
* Llama‑3‑70B: tuned lens missing; baseline only; use ranks. ([GitHub][6])
* Project plan & normalization policy (§1.1) captured in `001_LAYERS_BASELINE_PLAN.md`. ([GitHub][2])
* Tuned‑Lens paper and motivation. ([arXiv][8])
* RMSNorm formula (ε in √). ([arXiv][10])
* Superposition motivation for feature‑level tests. ([arXiv][9])

---

[1]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-Meta-Llama-3-8B.md "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_LAYERS_BASELINE_PLAN.md "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-gemma-2-9b.md "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/output-Qwen3-8B.json "raw.githubusercontent.com"
[6]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json "raw.githubusercontent.com"
[7]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-latest/evaluation-cross-models.md "raw.githubusercontent.com"
[8]: https://arxiv.org/abs/2303.08112?utm_source=chatgpt.com "Eliciting Latent Predictions from Transformers with the Tuned Lens"
[9]: https://arxiv.org/abs/2209.10652?utm_source=chatgpt.com "Toy Models of Superposition"
[10]: https://arxiv.org/abs/1910.07467?utm_source=chatgpt.com "Root Mean Square Layer Normalization"

---

# Claude Sonnet 4.5 Extended Thinking

## 1. Code Quality and Methodological Soundness

### 1.1 Architecture Detection and Normalization

The implementation demonstrates sophisticated awareness of transformer architecture variations. The codebase correctly distinguishes between pre-norm and post-norm architectures, selecting the appropriate normalization module (ln2 for post-block residuals in post-norm models, or next block's ln1 in pre-norm models). The epsilon placement inside the square root for RMSNorm conforms to the original specification (Zhang et al., 2019, arXiv:1910.07467), and the fp32 computation of normalization statistics with subsequent dtype casting addresses numerical stability concerns appropriately.

However, a critical issue emerges in the normalization provenance logging. The script records substantial residual norm changes at early layers (e.g., Mistral-7B showing `resid_norm_ratio` values exceeding 100× at layers 0-2, with `delta_resid_cos` dropping to 0.31 at layer 0). While the diagnostics correctly flag `normalization_spike: true` when these occur before L_semantic, the interpretation remains ambiguous. These extreme ratios could indicate either (a) correct scaling behavior in models with unnormalized embeddings, or (b) a mismatch between the chosen normalization module and the actual residual being probed. The cross-model evaluation acknowledges this as a caution flag but does not conclusively resolve whether such spikes represent genuine model behavior or probe artifacts.

**Recommendation:** Add a validation check comparing the chosen norm module's parameters (γ, ε) against the residual stream's actual statistics at a known-good layer (e.g., the penultimate layer where behavior should be stable). If γ varies dramatically across blocks, document whether the script correctly tracks which γ applies to which residual.

### 1.2 Copy-Collapse Detection

The ID-level subsequence matching with rolling windows (k∈{1,2,3}) and multiple thresholds (strict τ∈{0.70,0.80,0.90,0.95}, soft τ=0.33/0.50) represents a methodologically sound approach. The ignore mask for whitespace and punctuation (1,179 tokens for Mistral-7B) prevents spurious matches, though the mask generation logic is not fully transparent—it would be valuable to verify that the mask covers all tokenizer-specific formatting tokens (e.g., BOS, EOS, special instruction markers).

The strict copy detector's null results across most models (only Gemma-2 shows strict copy at L0) contrast sharply with the project's philosophical motivation. If strict copy represents "austere nominalism's token-listing," its near-universal absence in the baseline sweep actually *weakens* rather than strengthens the anti-nominalist case. The soft detectors also fire rarely (mostly null across models). This suggests either (a) the capital-of-Germany task is too demanding for pure surface recombination, or (b) the prompt design inadvertently selects against copy-prone behavior. The ablation summary showing negligible ΔL_copy between "simply" and no-filler variants further supports interpretation (a).

**Recommendation:** The project plan proposes a rank-centric battery (§2.3) with 100-1,000 country-capital pairs. Prioritize this extension, as single-prompt brittleness is now empirically confirmed. Additionally, consider prompts explicitly designed to *elicit* copy behavior (e.g., "Repeat after me: Berlin") as positive controls.

### 1.3 Gold Token Alignment

The fallback chain (tokenizer → TL tokens → unresolved) with explicit variant tracking (with-space preference, no-space fallback) is robust. The persistence of `gold_answer` fields including `pieces`, `answer_ids`, and `variant` enables full auditability. However, the unresolved rate is not reported in the cross-model evaluation. Given that tokenization quirks could produce multi-token answers or leading-space mismatches, the script should emit a summary statistic: `gold_alignment_failures / total_prompts`.

**Recommendation:** Add to JSON diagnostics: `"gold_alignment_rate": 1.0` (or fraction) and flag models where alignment fails on primary or control prompts.

### 1.4 Raw-vs-Norm Dual-Lens Checks

The three-tier sampling strategy (sampled at ~4 depths by default, windowed at ±R layers around collapse candidates, full per-layer) is well-calibrated. The windowed escalation (R=8 when initial risk is high) prevents missed artifacts. The `lens_artifact_score` formula (60% weight on KL≥1.0 prevalence, 30% on norm-only semantics, 10% on max KL scaled) provides a principled aggregate, though the weights appear ad-hoc. The tier thresholds (low<0.2, medium 0.2-0.5, high>0.5) align with the observed distributions but lack external validation.

The full raw-vs-norm sidecar (§1.24) reveals substantial divergences: Mistral-7B shows `max_kl_norm_vs_raw_bits=8.56` with earliest `norm_only_semantic=32`; Qwen3-14B and Qwen2.5-72B exceed 13 bits in their window checks; Yi-34B reaches 90.47 bits with 14 norm-only layers. These KL magnitudes are **extreme** (90 bits corresponds to a 10²⁷ fold probability distortion) and suggest either (a) the raw lens produces highly diffuse distributions that normalization dramatically sharpens, or (b) a numerical artifact in the KL computation when raw probabilities approach machine epsilon.

**Critical Issue:** The `kl_norm_vs_raw_bits` is computed as KL(P_norm || P_raw). When P_raw has near-zero entries that P_norm does not, the KL diverges. The script should report complementary metrics: (i) reverse KL(P_raw || P_norm), (ii) Jensen-Shannon divergence (symmetric), and (iii) the fraction of the vocabulary where `|P_norm - P_raw| > 0.01`. This would clarify whether the large KLs arise from a few pathological tokens or a global calibration gap.

**Recommendation:** Augment raw_lens_full with per-layer JS divergence and report the entropy of P_raw vs P_norm. If P_raw has entropy ~15 bits (nearly uniform over 32k vocab) while P_norm has entropy ~3 bits (focused), the KL is uninformative about lens reliability; the models simply haven't committed to an answer yet when decoded raw. Use the entropy gap as a more interpretable substitute for KL in the artifact score.

### 1.5 Tuned-Lens Integration

The implementation closely follows Belrose et al. (2023, arXiv:2303.08112) with per-layer affine translators and learned temperatures. The training budget (32M tokens, width-scaled rank k≤256, multi-layer updates) is reasonable for the tested models. The last-layer identity constraint and near-zero final-layer KL checks confirm the translators correctly map to the model's actual output space.

The rotation-vs-temperature attribution (§1.26) decomposes ΔKL into calibration (temp-only) and semantic shift (rotation) components. For Llama-3-8B, the ΔKL_rot values (~4.5 bits at mid-depth) dominate ΔKL_temp (~0 bits, since norm-temp already near-optimal), suggesting genuine representational refinement rather than mere scaling. However, the attribution assumes linear additivity of effects, which may not hold if rotation and temperature interact nonlinearly. The `prefer_tuned` gate (ΔKL_rot≥0.2 at p50 or rank advances ≥2 layers) is reasonable but arbitrary—0.2 bits is ~15% change on a base of 1.5 bits, which is noticeable but not dramatic.

**Recommendation:** Report the interaction term ΔKL_interaction = ΔKL_tuned - (ΔKL_rot + ΔKL_temp) in the JSON. If this exceeds 0.5 bits, note that the decomposition is approximate. For models where `prefer_tuned=true`, include a brief explanation of *why* rotation helps (e.g., "translators realign feature axes that drift across depth").

### 1.6 Prism (Logit Prism Shared Decoder)

The Prism implementation (whitening + orthogonal rotation, k=512 for Mistral-7B, sampled at 4 depths over ~100k tokens) is conceptually sound as a cross-layer calibration baseline. However, the cross-model evaluation summary classifies all compatible Prism results as "Neutral" or "Regressive," with KL inflation (Δ<0, worse calibration) at mid-depth for most models. Llama-3-8B shows Δ≈-8.29 bits at p50, Mistral-7B Δ≈-17.54 bits—these are *large* regressions, not subtle shifts.

This pattern suggests the Prism fitting procedure may be over-fitting to the sampled layers or under-estimating the distributional shift between shallow and deep representations. The provenance block records `layers: ["embed", 7, 15, 23]` for Mistral-7B, sampling at 7/32≈22%, 15/32≈47%, 23/32≈72% depth. If early layers (≤3) were not included in the fit, the whitening transform may mis-scale their residuals.

**Recommendation:** Expand Prism sampling to include at least one very early layer (≤10% depth) and one very late layer (≥90% depth). Re-run fits with k∈{256, 512, 1024} to test sensitivity to rank. If regressions persist, document Prism as a "diagnostic-only" mode (as the JSON already notes) and use it primarily to flag families where shared-decoder assumptions fail (e.g., Gemma with its calibrated final head).

### 1.7 Numeric Health and Determinism

The numeric health checks (NaN/Inf detection, max_abs_logit, min_prob) are present but underutilized. The diagnostics report `numeric_health.layers_flagged` arrays, but these are empty or sparsely populated in the current run. The environment capture (torch version, CUDA, deterministic_algorithms flag) is thorough.

The `nondeterministic` flag is set when deterministic algorithms cannot be enabled (e.g., some CUDA ops lack deterministic implementations). This is correctly propagated to `measurement_guidance.reasons`, advising evaluators to prefer rank-based comparisons. However, the script does not re-run prompts to measure empirical reproducibility. Adding a "repeatability check" (run the same prompt twice with different seeds, report max rank deviation) would quantify nondeterminism's actual impact.

**Recommendation:** Add a lightweight repeatability test in self-test mode: run 10 prompts twice each, compute `max(|rank_run1 - rank_run2|)` per model. If this exceeds 5 ranks, flag high nondeterminism; otherwise, note that rank-based metrics remain stable despite kernel nondeterminism.

---

## 2. Results Interpretation and Overstatements

### 2.1 Cross-Model Semantic Collapse Timing

The cross-model evaluation reports normalized collapse depths spanning 0.50 (Llama-3-70B) to 1.0 (Gemma-2-9B/27B, Qwen2.5-72B). The narrative correctly notes that Llama-3-70B's early collapse (40/80=0.50) aligns with strong MMLU performance (79.5%), while the late-collapsing Qwen2.5-72B also achieves high MMLU. This **contradicts** a simple "earlier collapse → better reasoning" hypothesis, undermining any straightforward causal claim.

However, the evaluation then pivots to "suggestive, not causal" framing, which is appropriate. The issue is that the project's philosophical framing (in 001_LAYERS_BASELINE_PLAN.md §1 "Linking collapse depth to public scores") invites readers to interpret depth as a semantic milestone, yet the empirical spread is so wide (0.50 to 1.0, a factor of 2) that depth alone provides little predictive power. Within-family comparisons (Llama 70B vs 8B) do show earlier collapse at scale, but this could reflect better head calibration or longer context integration, not necessarily semantic sophistication.

**Critique:** The evaluation correctly applies the "correlational, not causal" disclaimer but does not emphasize that the *variance* in collapse depths across equal-performing models (e.g., Qwen2.5-72B at 1.0 vs Mistral-Small-24B at 0.825, both strong) suggests collapse depth is confounded by architectural choices (final head transforms, normalization schemes, training recipes) orthogonal to semantic competence. The write-up should state explicitly: "Collapse depth reflects a model's internal computation path, not its endpoint capability."

### 2.2 Lens Artifact Risk and Reporting Guidance

The cross-model evaluation flags high artifact risk for Qwen3-8B, Qwen3-14B, Qwen2.5-72B, Yi-34B, and Mistral-7B based on window KL checks and `lens_artifact_score`. The measurement guidance correctly advises "prefer ranks" and "suppress absolute probabilities" for these models. However, the evaluation then proceeds to discuss `p_answer` and `echo_mass` values (e.g., "Llama-3-8B L_surface=32 with answer_mass≈0.520") for models with high-risk warnings.

This creates interpretive tension: if absolute probabilities are unreliable under the norm lens, citing them—even with caveats—risks anchoring readers to a specific numeric scale. The evaluation should consistently redirect to rank-based claims or KL trends when discussing high-risk models.

**Example of safer framing:** Instead of "Llama-3-8B L_surface=32 with answer_mass≈0.520," write "Llama-3-8B crosses the surface-to-meaning boundary by layer 32 (answer rank improves while prompt token ranks decline)."

**Recommendation:** The cross-model evaluation should partition models into "low-artifact" (Mistral-Small-24B) vs "high-artifact" (Qwen, Yi, Mistral-7B) tiers and explicitly state that numeric probability comparisons are **only valid** within the low-artifact tier. For high-artifact models, report rank milestones and KL trends exclusively.

### 2.3 Confirmed Semantics and Multi-Lens Corroboration

The `L_semantic_confirmed` metric (§1.25) requires rank-1 under the norm lens to be corroborated by raw or tuned within a ±2 layer window. This is a valuable safeguard against norm-induced illusions. However, the implementation detail that matters here is not mentioned in the cross-model evaluation: **which models achieved confirmed semantics, and where did corroboration come from (raw vs tuned)?**

Checking the Mistral-7B JSON: `diagnostics.confirmed_semantics` is present, showing `L_semantic_norm=25`, but the corroboration source is not reported (the field exists but is not surfaced in the summary). This is a missed opportunity—if tuned consistently corroborates where raw does not, that would support the interpretation that normalization introduces genuine refinement (not artifact), whereas if both raw and tuned corroborate equally, it suggests the norm lens is already faithful.

**Recommendation:** Elevate `confirmed_semantics.confirmed_source` to the cross-model summary table. Report for each model: `L_semantic_norm`, `L_semantic_confirmed`, `Δ_confirm` (difference), and `source` (raw/tuned/both/none). Discuss the pattern: "Models where tuned corroborates but raw does not (e.g., Llama-3-8B) show semantic representations that benefit from learned projections, consistent with iterative refinement. Models where raw corroborates (e.g., Mistral-Small-24B low-artifact tier) exhibit more direct readouts."

### 2.4 Tuned-Lens Gains: Rotation vs Temperature

The tuned-lens summary reports substantial rotation gains (ΔKL_rot>4 bits) for Llama-3-8B, Qwen3-8B, Qwen3-14B, Mistral-7B, Mistral-Small-24B, and Yi-34B at mid-depth (p50). The evaluation interprets these as "genuine representational refinement." This is plausible given that Belrose et al. (2023) demonstrated tuned lenses use features similar to the model's own computation.

However, the rotation gains are *averages across all prompts in the calibration stream*, which includes the training data for the tuned lens. The project uses a single country-capital prompt for each model's main sweep, which may or may not have been part of the tuned lens training set (the script draws calibration from "the first 256k tokens consumed by the TL fitter"—unclear if this overlaps with the target prompt). If the tuned lens was trained on text containing "capital of Germany," its strong performance on that exact prompt could reflect memorization rather than generalization.

**Critical Question:** Was the target prompt ("The capital of Germany is called simply") withheld from the tuned-lens training set? The provenance block (`tuned_lens.provenance`) does not clarify this. If not withheld, the rotation gains might partially reflect overfitting.

**Recommendation:** Document the tuned-lens training procedure in the JSON: `"provenance.train_prompts_overlap": "target_prompt_excluded"` (or `"included"`). If included, re-run tuned-lens fitting on a held-out stream that excludes the target prompt and compare ΔKL_rot on the held-out prompt. If rotation gains persist on unseen prompts, the generalization claim is secured; if they drop, acknowledge that tuned lenses require careful train-test splits.

### 2.5 Prism Regression and Misalignment

As noted in §1.6, Prism shows consistent KL regressions (negative Δ) across models. The cross-model evaluation correctly labels these as "Regressive" but does not explore *why* Prism fails where tuned lens succeeds. Both methods learn a projection from residuals to logits, but Prism enforces orthogonality and uses a shared decoder (one matrix for all layers), whereas tuned lens allows per-layer affine freedom.

A plausible hypothesis: The shared-decoder constraint *over-regularizes* for early layers, where residuals have not yet converged to the representational subspace that the final head decodes. The orthogonality requirement (Procrustes alignment) may also force the projection to preserve irrelevant variance (e.g., style or position information) that the final head ignores.

**Recommendation:** Test a "late-only Prism" variant: fit Prism using only layers ≥70% depth, where representations are closer to the final output space. If KL regressions disappear, this confirms the hypothesis that Prism's shared decoder cannot accommodate the shallow-to-deep representational shift. Alternatively, relax orthogonality to allow low-rank updates (Prism + learned rank-k adjustments per layer), effectively interpolating toward tuned lens.

---

## 3. Independent Analysis of Results

### 3.1 Collapse Depth Patterns and Architectural Confounds

Plotting the reported `L_semantic_frac` values against model parameters and architecture type reveals a clustering pattern:

- **Post-norm RMSNorm models** (Mistral, Qwen, Yi, Llama-3-8B): Late collapse (≥0.73), often to final layer
- **Pre-norm RMSNorm models** (Llama-3-70B): Mixed (early 0.50 for 70B)
- **Post-norm with calibrated head** (Gemma-2): Final layer (1.0) *by design*—the final head applies additional temperature/softcap

The Gemma family's 1.0 collapse is an **artifact of architectural choice**, not semantic latency. The `last_layer_consistency` block confirms this: Gemma-2-9B shows KL(lens_final || model_final)≈1.01 bits, meaning the lens's penultimate-layer distribution differs from the model's true output by ~1 bit. The model's head applies a learned calibration (temperature or softcap) that the norm lens cannot replicate without per-layer probes (i.e., tuned lens).

Importantly, the tuned-lens results for Gemma show **negative ΔKL_rot** (<0 at all percentiles), indicating the rotation actually *worsens* calibration relative to norm+temperature. This is consistent with the head doing the "final rotation" itself. The `measurement_guidance.preferred_lens_for_reporting="norm"` correctly directs evaluators away from the regressive tuned lens for Gemma.

**Implication for Philosophical Claims:** If a model's "semantic collapse" is determined by whether it applies final-layer transforms, then collapse depth is a *design choice*, not an emergent property of semantic competence. This weakens any claim that "late collapse" reflects a model storing abstractions in a specific way—it may simply reflect where the architect placed the calibration step.

**Recommendation:** The cross-model evaluation should explicitly separate models with learned final-head transforms (Gemma, likely others with `head_scale_cfg` or `head_softcap_cfg` detected) from those with identity heads. Compare collapse depths *within each group* to remove this confound. State: "Collapse depth is architectural prior as much as representational signature."

### 3.2 Surface-to-Meaning Transition: Mass vs Geometry

The surface diagnostics (§§1.13-1.15) provide three operationalizations of the surface-to-meaning transition:

1. **Mass crossover** (`L_surface_to_meaning`): First layer where `AnsMass ≥ EchoMass + 0.05`
2. **Geometric crossover** (`L_geom`): First layer where `cos_to_answer ≥ cos_to_prompt_max + 0.02`
3. **Top-K decay** (`L_topk_decay`): First layer where `TopKPromptMass ≤ 0.33`

For Mistral-7B:
- `L_surface_norm=32` (mass crossover)
- `L_geom_norm=null` (geometry never crosses)
- `L_topk_decay_norm=0` (Top-K immediately low)
- `L_semantic=25` (rank-1)

The **discordance** between these measures is striking. The Top-K metric fires immediately (L0), suggesting the model never concentrates probability mass on prompt tokens in the decoded space—yet mass crossover occurs at L32, seven layers *after* semantic collapse (L25). Meanwhile, geometry never achieves a clean crossover (cos_to_answer remains below cos_to_prompt_max + γ throughout).

This suggests the three metrics capture orthogonal aspects:
- **Top-K decay** measures *diffuseness*—how spread out the distribution is
- **Mass crossover** measures *dominance*—whether the answer token surpasses the sum of prompt tokens
- **Geometry** measures *direction alignment*—whether the residual points more toward the answer decoder column than toward prompt columns

For Mistral-7B, the pattern is: (i) the distribution starts diffuse (Top-K low), (ii) the answer becomes rank-1 by L25, (iii) but it takes until L32 for the answer to accumulate *mass* exceeding the scattered prompt mass, (iv) and the residual geometry remains ambiguous throughout (high cosine to both answer and prompt).

**Interpretation:** This is **consistent with a sparse steering vector** (few dimensions encode the answer) embedded in a high-dimensional residual that still carries prompt information in orthogonal subspaces. The model can decode rank-1 correctly from a sparse signal without fully suppressing the geometric signature of the prompt.

**Philosophical Implication:** The sparse-signal interpretation aligns with **feature-based accounts** of representation (Olah et al., Anthropic; Elhage et al., Anthropic, 2021-2023), where concepts live in low-rank subspaces. This is harder to reconcile with austere nominalism (which would predict dense, token-specific patterns) but is consistent with *either* metalinguistic nominalism (features encode predicates like "capital-of") *or* realism (features encode universals like "capital-of" instantiated across tokens). The current baseline sweep does not distinguish these.

### 3.3 Control Margin and Lexical Co-Occurrence Leakage

The control summary reports `first_control_margin_pos` and `max_control_margin` for the France→Paris control. For Mistral-7B, these fields are not surfaced in the cross-model evaluation excerpt, but the JSON should contain them. Checking the structure: `control_summary.first_control_margin_pos` would indicate the first layer where `p(Paris) > p(Berlin)` in the control prompt.

The philosophical significance is that **late or null control margin** would indicate the model leaks lexical co-occurrence (Germany and Berlin co-occur frequently, so "Germany" activates "Berlin" even in unrelated contexts). **Early control margin** (e.g., L<10) suggests the model disambiguates subject-predicate relations early, suppressing spurious co-occurrences.

**Recommendation:** The cross-model evaluation should report control margin timing alongside collapse depth: "Mistral-7B: `L_semantic=25`, `first_control_margin_pos=X`, `max_control_margin=Y`. The model [distinguishes France from Germany reliably by layer X / shows late disambiguation, hinting at residual lexical leakage]."

If control margin appears late or remains negative (Paris never exceeds Berlin in the control prompt), this is **evidence for a nominalist interpretation**—the model's "capital-of" behavior could be a sophisticated conditional frequency tracker rather than a relational abstraction. Conversely, early clean separation would support the claim that the model encodes a predicate distinct from raw co-occurrence.

### 3.4 Entropy Drift and Teacher Alignment

The cross-model evaluation mentions mid-depth entropy drift (e.g., Llama-3-8B L≈16 shows `entropy - teacher ≈ 13.89 bits`). This is the entropy of the layer's decoded distribution minus the entropy of the model's final output ("teacher"). A drift of ~14 bits over a 128k-vocab model (log2(128k)≈17 bits maximum entropy) indicates the mid-layer distribution is nearly **uniform**, while the final output is concentrated (~3-4 bits, corresponding to ~10-16 plausible tokens).

This pattern—extreme diffuseness at shallow layers, gradual sharpening—is well-documented in iterative inference models (Jastrzębski et al., 2017) and aligns with the "early direction, late calibration" framing from nostalgebraist's original logit lens work. However, the tuned-lens results show that learned projections can **skip the diffuse phase**: ΔKL drops by ~4 bits at p25, suggesting the tuned lens extracts a focused distribution even from shallow residuals.

**Implication:** The entropy drift is partly a **decoder mismatch**, not inherent representational fuzziness. The residuals at shallow layers may encode focused information in a basis that the final-layer decoder does not align with. The tuned lens realigns the basis, revealing earlier commitment.

**Recommendation:** Add a "decision sharpness" metric: For each layer, compute the entropy gap `H(P_layer) - H(P_final)` under the *same* lens (norm or tuned). If the gap shrinks monotonically, the model is refining iteratively. If it fluctuates (increases mid-stack, then decreases), the model may be composing subcomputations in a non-monotonic way. This would inform whether iterative inference is the right conceptual frame or whether transformers perform parallel, non-sequential computations (as suggested by recent circuit-tracing work, e.g., Wang et al., ICLR 2023, "Interpretability in the Wild").

---

## 4. Usefulness for the Nominalism vs Realism Debate

### 4.1 Evidence Against Austere Nominalism (Strong)

The baseline sweep provides **robust evidence against austere nominalism**, which claims all structure reduces to lists of concrete token occurrences. Key findings:

1. **Cross-prompt stability** (implicit): The tuned lens learns a single set of translators from a training stream distinct from the target prompt, yet achieves ΔKL_rot>4 bits on the held-out prompt (assuming proper train-test split). This demonstrates **transfer** of learned projections across prompts, inconsistent with prompt-specific token-matching.

2. **Rank robustness** (strong): Even under high lens artifact risk (KL divergences >8 bits between raw and norm), **rank orderings remain stable**. The measurement guidance correctly notes that ranks are less sensitive to calibration distortions. Austere nominalism predicts fragile, token-specific patterns; robust rank orderings across lens distortions suggest underlying structure that transcends surface forms.

3. **Control margin** (conditional): If early control margins are confirmed across models (data not fully reported in cross-model evaluation), this would show that models distinguish Germany→Berlin from France→Paris **despite shared co-occurrence patterns** ("capital of..."), supporting predicates/relations over pure token lists.

**Assessment:** The evidence against austere nominalism is **provisionally strong** but requires the following confirmations:
- Control margin timing across all models
- Tuned-lens train-test splits verified
- Replicate findings on the planned rank-centric battery (100-1,000 prompts) to rule out single-prompt overfitting

### 4.2 Evidence on Metalinguistic Nominalism vs Realism (Weak)

The current baseline sweep does **not** yet adjudicate between metalinguistic nominalism (MN) and realism. MN treats internal structure as sophisticated facts about linguistic predicates (e.g., the model encodes "capital-of" as a predicate relation over word-tokens). Realism posits that internal structure tracks mind-independent universals (e.g., the capital-of *relation itself*).

The planned probes in 001_LAYERS_BASELINE_PLAN §§4-5 (vector portability across modalities §4.7, synthetic language swap §5.2, paraphrase robustness §4.2, multilingual consistency §4.3) are designed to stress this distinction. None of these have been executed yet.

**Critical Observation:** The mass vs geometry discordance (§3.2) and entropy drift patterns (§3.4) are **equally consistent** with MN and realism. Both frameworks predict:
- Iterative refinement (compositional predicates in MN; convergence to universal structures in realism)
- Sparse representations (predicate features in MN; instantiated universals in realism)
- Cross-layer consistency (predicate stability in MN; universal stability in realism)

To distinguish MN from realism, the project must demonstrate phenomena that **resist linguistic paraphrase**. The planned multimodal probe (§4.7 vision→LM bridge) is the strongest current candidate: if a vector learned from text transfers to decode country names from flag images (non-linguistic input), this strains MN's linguistic-predicate story and supports realism's claim of language-independent structure.

**Recommendation:** Prioritize §4.7 (vision→LM bridge, now marked "minimal" scope) over §3.7 (SAE feature-level analysis, marked "ambitious"). The vision bridge is lower-lift (linear projection from CLIP to LLM, no LLM training) and has higher philosophical payoff. If it succeeds, the project can claim **cross-modal universals**; if it fails, this documents a limit case where structure remains language-bound (supporting MN).

---

## 5. Project Plan Review and Recommendations

### 5.1 Measurement Right (Group 1) - Status

The project has successfully executed §§1.1-1.36 (normalization fixes, copy detection, gold alignment, probability/rank metrics, raw-vs-norm checks, tuned lens, surface diagnostics, Prism, threshold sweeps, full raw-lens, confirmed semantics, rotation-vs-temperature attribution, etc.). The implementation quality is **production-grade** for a research prototype.

**Remaining Group 1 gaps:**
- **§1.7 Gold token alignment rate:** Not reported as a summary statistic (see §1.3 above)
- **§1.19/1.24 Raw-lens interpretation:** High KL values (>80 bits) suggest potential numerical artifacts; see §1.4 above

**Recommendation:** Before advancing to Group 2, add the two missing pieces (alignment rate, JS divergence in raw-lens checks). Otherwise, Group 1 is complete.

### 5.2 Experimental Variations (Group 2) - Priority Order

The project plan proposes:
- §2.1 Multilingual prompts (5 languages)
- §2.2 Predicate-permutation control (Quine guard)
- §2.3 Rank-centric battery (100-1,000 prompts)

**Revised Priority:**
1. **§2.3 Rank-centric battery** (HIGH PRIORITY): Single-prompt brittleness is now confirmed. Running 100-1,000 prompts will establish whether collapse depth, control margin, and tuned-lens gains are **robust generalizations** or **tokenization artifacts**. This is the most urgent next step.

2. **§2.1 Multilingual prompts** (MEDIUM PRIORITY): Valuable for testing linguistic invariance, but less critical than scale-up. Language-specific tokenization could introduce confounds (e.g., multi-token answers in non-English prompts).

3. **§2.2 Predicate-permutation** (LOW PRIORITY FOR NOW): Philosophically interesting (tests Quinean inscrutability) but operationally complex (requires maintaining permutation mappings, re-running circuits). Defer until Groups 3-4 are underway and circuit-level interventions are validated.

**New Addition:** Before §2.3, run a **single-prompt repeatability check** across 10 models × 10 seeds to quantify collapse depth variance from nondeterminism and initialization. If variance is <3 layers, collapse depth is a stable measure; if variance is >10 layers, it is too noisy to serve as a philosophical anchor.

### 5.3 Advanced Interventions (Group 3) - Sequencing

The project plan proposes:
- §3.1 Activation patching (causal tracing)
- §3.2 Attention head fingerprinting
- §3.3 Concept vector extraction (Causal Basis)
- §3.7 Targeted SAEs
- §3.9 Cross-model SAE universality

**Critical Issue:** §3.7 (SAE feature-level analysis) is marked "ambitious, replaces current §3.7" and includes extensive reliability checks (seed replication, decomposition tests). While methodologically laudable, this is a **major engineering lift** (train 3× SAE models per layer, cross-seed matching, meta-SAE fitting) and may not be necessary for the philosophical goals.

**Recommendation:** Re-scope §3.7 as **optional exploration** rather than baseline requirement. The core philosophical claims (Group 4 evidence against MN) can be established via:
1. **§3.1 Activation patching** (causal tracing): Demonstrates sufficiency of specific layers for answer generation
2. **§3.2 Attention head fingerprinting**: Localizes factual retrieval to specific heads
3. **§3.3 Concept vector extraction**: Tests portability of learned directions

These three are **lighter-lift** than SAEs (no multi-seed training, no meta-SAE fitting) and provide strong causal evidence. SAE analysis (§3.7) should be reframed as a **future direction** once the core causal probes are validated.

**Sequencing for Group 3:**
1. §3.1 Activation patching on a 10-prompt sample (quick feasibility check)
2. §3.2 Head fingerprinting at `L_semantic ± 2` for the capital prompt
3. §3.3 Concept vector extraction (learn "capital-of" direction from 50 country pairs)
4. Test vector portability on held-out 50 pairs
5. If all succeed, consider §3.7 SAE analysis as a deep-dive extension

### 5.4 Consolidating Case Against Austere Nominalism (Group 4)

The project plan proposes:
- §4.1 Instruction words vs capital-relation (causal check)
- §4.2 Paraphrase robustness
- §4.3 Multilingual consistency
- §4.4 Large WikiData battery with causal L_sem
- §4.5 Lexical ambiguity stress test
- §4.6 Instruction-style grid
- §4.7 Vision→LM bridge (marked "minimal")
- §4.8 Feature-steering side-effects

**Critical Observation:** Many Group 4 items overlap with Group 2 (e.g., §4.3 duplicates §2.1 multilingual prompts; §4.4 WikiData battery duplicates §2.3 rank-centric battery). This suggests the plan evolved organically without consolidation.

**Recommendation:** **Merge Groups 2 and 4** into a single "Experimental Battery" phase:
- **Phase 2A (Observational Scale-Up):**
  - §2.3 / §4.4: WikiData battery (100-1,000 prompts) with full metrics (collapse depth, control margin, ablation deltas)
  - §2.1 / §4.3: Multilingual subset (50 prompts × 5 languages)
  - §4.2: Paraphrase robustness (10 paraphrases of capital prompt)
  - §4.5: Lexical ambiguity (50 ambiguous names)

- **Phase 2B (Causal Validation):**
  - §3.1: Activation patching on battery sample
  - §3.2: Head fingerprinting on high-signal prompts
  - §3.3: Concept vector extraction and portability tests
  - §4.1: Instruction ablation (part of paraphrase battery)
  - §4.7: Vision→LM bridge (lightweight probe)

- **Phase 2C (Reporting):**
  - Aggregate battery statistics (collapse depth distributions, control margin prevalence, causal L_sem vs observational L_sem correlations)
  - Write integrated report: "Evidence Against Austere Nominalism from Large-Scale Probing"

This consolidation avoids duplication, sequences work logically (scale-up → causal validation → aggregation), and delivers a coherent philosophical claim.

### 5.5 Probing MN vs Realism (Group 5)

The project plan proposes:
- §5.1 Vector portability across modalities (VLM fine-tuning)
- §5.2 Synthetic mini-language swap

**Critical Issue:** §5.1 as originally scoped (fine-tune LLaVA) is **too heavy**—fine-tuning a VLM is resource-intensive and introduces confounds (the fine-tuning process could *create* cross-modal structure that wasn't there initially).

**Recommendation:** Replace §5.1 with the lightweight vision→LM bridge now in §4.7. This avoids fine-tuning, preserves the frozen LLM, and tests whether text-learned vectors generalize to non-linguistic inputs. If successful, this is **strong evidence for language-independent structure** (supporting realism over MN).

§5.2 (synthetic language swap) remains valuable but is lower priority than the vision bridge. Synthetic languages require construction of a mini-lexicon, re-tokenization, and controlled training—this is a **multi-month project**. Defer until the vision bridge is validated.

**Revised Group 5:**
- **§5.1 Vision→LM bridge** (lifted from §4.7, now primary probe)
- **§5.2 Synthetic language** (future work, scope TBD after §5.1 results)

---

## 6. Specific Additions to Next Steps

### 6.1 Missing Ablation: Attention-Only vs MLP-Only Collapse

The project plan includes sublayer patching (§3.1 attn vs MLP) but does not extend this to the logit-lens sweep. Given that Geva et al. (2020, arXiv:2012.14913) demonstrated MLPs perform factual retrieval while attention mediates composition, it would be valuable to decompose `L_semantic` into:
- `L_semantic_attn_only`: First layer where decoding attention output alone yields rank-1
- `L_semantic_mlp_only`: First layer where decoding MLP output alone yields rank-1
- `L_semantic_residual`: The standard post-block residual (current `L_semantic`)

**Why it matters:** If `L_semantic_mlp_only < L_semantic_attn_only`, this supports the claim that MLPs store factual knowledge (supporting realism's "knowledge-as-structure" reading). If they coincide, composition and retrieval are entangled.

**Implementation:** In `run.py`, add hooks to capture attention and MLP outputs separately at each layer. Decode both through the lens (norm or tuned) and compute ranks. Add to JSON: `"L_semantic_attn_only"`, `"L_semantic_mlp_only"`.

**Lift:** Low (minor extension of existing hooks). **Philosophical payoff:** High (localizes factual knowledge to sublayers).

### 6.2 Missing Analysis: Collapse Depth vs Layer Width

The cross-model evaluation notes that `L_semantic` does not correlate simply with model scale (e.g., Qwen2.5-72B collapses at 1.0 despite 72B params). However, it does not test whether collapse depth correlates with **layer width** (d_model). A plausible hypothesis from representation-learning theory: wider residuals provide more "workspace" for semantic representations to emerge without interference, enabling earlier collapse.

**Recommendation:** Plot `L_semantic_frac` against `d_model` across all models. If a negative correlation emerges (wider → earlier), this would support the hypothesis that collapse is **capacity-limited** in narrow models. If no correlation, collapse is determined by training or architectural choices orthogonal to width.

**Data available:** `model_stats.d_model` is already logged in the JSON for each model.

### 6.3 Missing Control: Negated Prompt

The project includes a France→Paris control (§1.8) to test lexical leakage but does not include a **semantic negation control**. Example: "The capital of Germany is **not** called simply ___". A robust semantic model should rank Berlin **lower** (or at least not higher) under negation, while a surface-form co-occurrence model might still rank Berlin high (because "Germany" and "Berlin" co-occur regardless of negation).

**Implementation:** Add a third prompt variant (`prompt_id="neg", prompt_variant="orig"`) with "not" inserted. Compute `negation_margin = p(Berlin|nonneg) - p(Berlin|neg)` per layer. If `negation_margin` is positive throughout depth, the model respects negation; if negative or zero, it ignores negation (surface leakage).

**Lift:** Low (reuse existing pipeline). **Philosophical payoff:** High (tests whether the model encodes *propositional content* beyond lexical co-occurrence, a key distinction between MN and realism).

### 6.4 Missing Diagnostic: Residual Norm Growth Rate

The normalization provenance logs `resid_norm_ratio` per layer, but the cross-model evaluation does not analyze its trajectory. A model with **monotonic norm growth** (residuals grow steadily across depth) may be accumulating information iteratively, whereas a model with **non-monotonic norms** (spikes, dips) may be composing discrete subcomputations.

**Recommendation:** Add to JSON diagnostics: `"resid_norm_trajectory_shape"` ∈ {"monotonic_growth", "plateau", "non_monotonic", "spike"}. Classify based on linear regression slope and variance.

**Why it matters:** Non-monotonic norms could indicate **modular processing** (attention blocks "erase" MLP contributions or vice versa), which is harder to reconcile with iterative refinement models and may support a more compositional (realist) reading.

---

## 7. Contextualization via Latest Research

### 7.1 Logit Lens Limitations (Belrose et al. 2023, 2024)

The tuned-lens paper (Belrose et al., 2023, arXiv:2303.08112) documents that the original logit lens "fails to elicit plausible predictions for models like BLOOM and GPT-Neo" and produces "representational drift" where features are encoded differently across layers. The current project correctly adopts tuned lenses and warns users via `measurement_guidance` when absolute probabilities are unreliable.

Recent work (Paulo et al., 2024, arXiv:2404.05971) extends tuned lenses to RNNs, confirming that per-layer affine transformations improve calibration across architectures. However, this same work notes that "the method only conceptually depends on pre-norm residual blocks"—suggesting that post-norm architectures (like many models in this sweep) may not be ideal candidates for lens-based analysis.

**Implication:** The project should explicitly test whether post-norm models (Mistral, most Qwens) show worse lens calibration than pre-norm models (Llama-3-70B). The `arch` field in `normalization_provenance` provides this data. If post-norm consistently shows higher artifact risk, future work should prioritize pre-norm models or develop post-norm-specific lens methods.

### 7.2 Activation Patching Best Practices (Zhang & Nanda 2024)

Zhang & Nanda (2024, arXiv:2309.16042, ICLR 2024) systematically examine activation patching hyperparameters and recommend:
- **Metric choice:** Use KL divergence or logit difference for continuous outputs; avoid accuracy for rank-sensitive tasks (collapse detection is rank-sensitive).
- **Corruption method:** Prefer prompt-based corruption (clean → corrupted) over noise-based methods (ROME-style Gaussian noise) for precise causal claims.
- **Interpretation:** Activation patching identifies **sufficiency** (this activation is enough to restore behavior) but not **necessity** (removing it breaks behavior). Necessity requires ablation or causal scrubbing.

The project plan's §3.1 (activation patching) does not specify which metric or corruption method will be used. Given that the philosophical goal is to establish **sufficiency of specific layers for capital retrieval**, the recommended approach is:
1. Clean prompt: "The capital of Germany is called simply ___"
2. Corrupted prompt: "The capital of **France** is called simply ___"
3. Patch: Replace layer ℓ's residual from corrupted with clean
4. Metric: Logit difference `logit(Berlin|patched) - logit(Berlin|corrupted)`
5. Claim: Layer ℓ is **sufficient** if patching raises Berlin's logit above threshold

**Recommendation:** Specify this design in §3.1 and note the sufficiency/necessity distinction explicitly in the philosophical interpretation.

### 7.3 Sparse Autoencoders and Feature Superposition (Templeton et al. 2024, Anthropic)

Recent work from Anthropic (Templeton et al., 2024; Sharkey et al., 2022) demonstrates that SAEs can disentangle **polysemantic neurons** (neurons that activate for multiple unrelated concepts) into **monosemantic features** (sparse latents that activate for single concepts). This is directly relevant to §3.7's SAE proposal.

However, Anthropic's work also reveals a critical limitation: **feature splitting under capacity constraints**. When SAEs are under-provisioned (expansion factor <8×), they split genuine monosemantic features into multiple correlated latents, creating illusory interpretability. The project plan's rank-scaled width (k≤256 for d_model≈4096) corresponds to expansion factor ≈0.06×, which is **far below** the recommended 8-16×.

**Critical Recommendation:** If §3.7 proceeds, increase SAE expansion factor to at least **4×** (k≈16384 for d_model=4096) and budget for the resulting memory costs. Alternatively, acknowledge that the current k budget targets **coarse-grained circuits** (collections of features) rather than individual monosemantic features, and adjust claims accordingly.

### 7.4 Multilingual Concept Representations (recent work)

A recent arXiv paper (2024) on multilingual LLM representations via activation patching (Chen et al., 2024; Ghandeharioun et al., 2024) demonstrates that models encode concepts in a **language-agnostic subspace**—you can patch the concept representation from one language into another and generate correct translations without changing the language token.

This finding is **directly supportive** of the project's realism hypothesis: if concepts exist independently of linguistic labels, this strains MN (which ties structure to linguistic predicates). However, the cited work was on instruction-tuned models (Llama-2-chat variants), not base models. The project's use of base models (Mistral-7B-v0.1, Llama-3-8B-base) may produce different results.

**Recommendation:** When running §4.3 multilingual consistency, check whether cross-lingual concept patching works (patch German "Deutschland" residual into French "France" prompt; does it generate "Paris"?). If yes, this is **strong evidence for language-independent structure** (realist reading). If no, document this as a base vs instruct-tuned model difference.

---

## 8. Summary of Critical Recommendations

### Immediate (Before Group 2)
1. **Add gold alignment rate to JSON diagnostics** (§1.3)
2. **Replace KL with JS divergence in raw-lens artifact score** (§1.4)
3. **Document tuned-lens train-test split** (§2.4)
4. **Report control margin timing in cross-model evaluation** (§3.3)
5. **Run repeatability check (10 models × 10 seeds)** (§5.2, new addition)

### Group 2 Revisions (Scale-Up Phase)
6. **Prioritize §2.3 rank-centric battery (100-1,000 prompts) over multilingual** (§5.2)
7. **Defer predicate-permutation to post-Group 3** (§5.2)
8. **Add negated-prompt control** (§6.3, new addition)

### Group 3 Revisions (Causal Phase)
9. **De-scope SAE analysis (§3.7) to optional future work** (§5.3)
10. **Specify activation patching design (clean/corrupted prompts, KL metric)** (§7.2)
11. **Add attention-only vs MLP-only collapse metrics** (§6.1, new addition)

### Group 4/5 Consolidation
12. **Merge Groups 2 and 4 into unified Experimental Battery** (§5.4)
13. **Elevate vision→LM bridge from §4.7 to primary Group 5 probe** (§5.5)
14. **Defer synthetic language to future work** (§5.5)

### Analysis Enhancements
15. **Partition cross-model evaluation by artifact risk tier (low/high)** (§2.2)
16. **Report confirmed_semantics.confirmed_source in summary** (§2.3)
17. **Plot collapse depth vs d_model (layer width)** (§6.2, new addition)
18. **Classify residual norm trajectory shape** (§6.4, new addition)

---

## 9. Conclusion

The Logos-in-Layers baseline sweep represents a **methodologically sophisticated** first iteration, correctly implementing normalization fixes, gold token alignment, multi-lens checks, and tuned-lens integration. The code quality is production-grade for a research prototype. The results provide **strong evidence against austere nominalism** through rank robustness, tuned-lens transferability, and (pending confirmation) control margin separation.

However, three critical gaps limit immediate philosophical claims:

1. **Single-prompt brittleness:** The current sweep's reliance on one prompt per model (plus ablation/control) leaves open the possibility of tokenization artifacts. The planned 100-1,000 prompt battery must be prioritized.

2. **Lens artifact ambiguity:** High KL divergences (>80 bits) in raw-vs-norm checks raise concerns about numerical artifacts. Switching to JS divergence and entropy gaps would clarify whether large KLs reflect genuine distributional shifts or edge-case probability ratios.

3. **MN vs realism underdetermination:** The current baselines do not yet distinguish metalinguistic nominalism from realism. The vision→LM bridge (§4.7/new §5.1) is the most tractable probe to stress this distinction and should be elevated to primary status.

With these adjustments, the project is well-positioned to deliver on its philosophical goals: **demonstrating that LLM internal structure resists austere nominalism and providing initial tests of whether that structure is linguistic-predicate-bound (MN) or language-independent (realism)**.
