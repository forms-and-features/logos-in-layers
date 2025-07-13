# Interpretability Project - Development Notes for AI assistant

# Philosophical Project Context
**Goal**: Use interpretability to inform nominalism vs realism debate.

# Next steps

Items are ordered by the approximate engineering lift required.

---

## 1. Get the measurement right

### 1.1. Fix the RMS/LN scaling path (γ + ε placement)

**Why**
If you normalise a *different* residual stream (post‑block) with γ that was trained for the *pre‑block* stream, logits are systematically mis‑scaled; early‑layer activations can be inflated by >10 ×.  An incorrect ε outside the square‑root likewise shifts all norms upward.  These distortions then propagate through the logit lens, giving spurious “early meaning” or hiding true signal.  RMSNorm’s official formula places ε **inside** the √ and multiplies by γ afterwards ([arxiv.org][1]).

**What**
*Apply the correct normaliser that is contemporaneous with each residual stream, and follow the RMSNorm formula exactly.*

**How**

1. Inside `apply_norm_or_skip`:

   ```python
   # compute rms with eps *inside* the sqrt
   rms = torch.sqrt(residual.pow(2).mean(-1, keepdim=True) + norm_module.eps)
   return residual / rms * scale
   ```
2. Instead of always pulling `block[i].ln1`, fetch the *next* norm that actually generated the residual being probed:

   ```python
   norm_module = model.blocks[i].ln2 if probe_after_block else model.blocks[i].ln1
   ```

   (Wrap this in a helper; autodetect whether the architecture is Pre‑Norm or Post‑Norm.)
3. Add a unit test that decodes layer 0 twice: once with γ=1, once with learned γ.  The KL between them should match KL between *raw* hidden states with and without γ, proving that scaling now matches semantics.

---

### 1.2. Sub‑word‑aware copy‑collapse detector

**Why**
For BPE/WordPiece vocabularies the answer “Berlin” may surface as two tokens “Ber ▁lin”.  The current string‑match misses that, under‑counting copy events and making Gemma look unique when it may not be.

**What**
*Detect prompt echo at the string level after detokenisation, regardless of how many word‑pieces the tokenisation used.*

**How**

1. Replace:

   ```python
   candidate = tokenizer.decode(top1_id)  # returns single piece
   collapse = (candidate in prompt) and (top1_p > THRESH)
   ```

   with:

   ```python
   cand_detok = tokenizer.decode([top1_id]).strip()
   collapse = cand_detok in prompt and top1_p > THRESH
   ```

2. Optional: allow contiguous multi‑token matches by keeping a rolling window of the last *k* best‑tokens, detokenising them jointly, and checking membership.

3. Parameterise the probability threshold; expose `--copy-thresh` CLI option so reviewers can run sensitivity analyses.

---

### 1.3. Record top‑1 p, top‑5 p\_cumulative, and KL‑to‑final

**Why**
Entropy alone conflates “one spike” vs “five near‑ties”.  KL(ℓ ∥ final) is the metric used in the tuned‑lens paper to show convergence ([arxiv.org][2]).  These curves tell you whether the model is already *near* its final answer direction (supporting realism) or still exploring (supporting nominalism).

**What**
*Three new floating‑point columns in every `*-pure-next-token.csv`: `p_top1`, `p_top5`, `kl_to_final_bits`.*

**How**

1. During the logit sweep cache `final_logits = logits[-1]` once.
2. Per layer:

   ```python
   probs = logits.softmax(dim=-1, dtype=torch.float32)
   p_top1 = probs[0, top1_id]
   p_top5 = probs[0, torch.topk(probs, 5).indices].sum()
   kl = torch.kl_div(probs.log(), final_probs, reduction="sum") / math.log(2)  # bits
   ```
3. Append to the CSV writer.
4. Store kl_to_final_bits even when the tuned lens is active (it then measures residual mismatch, not absolute).

---

### 1.4. Raw‑activation lens toggle

**Why**
If “early meaning” disappears when you skip normalisation, that meaning was an artefact of the lens, not the model.

**What**
*A boolean CLI flag `--raw-lens` that bypasses `apply_norm_or_skip`.*

**How**

1. Add `parser.add_argument("--raw-lens", action="store_true")`.
2. Wrap the call:

   ```python
   resid_to_decode = residual if args.raw_lens else apply_norm_or_skip(residual, norm_mod)
   ```
3. Output the flag value into the JSON meta file so that result artefacts are traceable.

---

### 1.5. Representation‑drift cosine curve

**Why**
A realist reading predicts an answer‑token direction that exists early and merely grows in magnitude; a nominalist picture predicts the direction rotates into place late.  Cosine similarity across depth quantifies which is true.

**What**
*A per‑layer scalar `cos_to_final` written alongside entropy metrics.*

**How**

1. Compute once:

   ```python
   final_dir = final_logits / final_logits.norm()
   ```
2. At each layer:

   ```python
   curr_dir = logits / logits.norm()
   cos = torch.dot(curr_dir, final_dir).item()
   ```
3. Write `cos_to_final` column; include in any diagnostic plots.

---

### 1.6. Negative‑control prompt

**Why**
If Berlin outranks Paris in “The capital of *France* is …”, your probe is leaking string co‑occurrence, invalidating any metaphysical claim.

**What**
*Run every model on one additional control prompt (“Give the city name only … France …”) and log the same metrics.*

**How**

1. `PROMPTS = [positive_prompt, control_prompt]`.
2. Loop over prompts; append a `prompt_id` column to CSV and JSON.
3. In the analysis notebook auto‑flag any layer where `p_paris < p_berlin` in the control row.

---

### 1.7. Ablate stylistic filler (`simply`)

**Why**
Gemma’s early copy‑collapse may be triggered by instruction‑style cues, not semantics.  Removing “simply” tests that hypothesis.

**What**
*Rerun the base prompt with and without filler and compare `L_copy`, `L_semantic`.*

**How**

1. Duplicate prompt, drop the adverb.
2. Record both runs with `prompt_variant` metadata.
3. Plot `Δ-collapse` for the two variants; a large shift confirms the stylistic‑cue explanation.

---

### 1.8. Lightweight CI / regression harness

**Why**
As soon as you integrate tuned lens or refactor, you need guard‑rails ensuring that numbers do not silently drift.

**What**
*A GitHub Actions workflow that executes `python run.py --models meta-llama/Meta-Llama-3-8B --cpu --dry-run` and checks that:*

* JSON meta contains the new schema keys
* `L_semantic` remains within ±1 layer of an expected value stored in `expected.json`.

**How**

1. In `run.py` add a `--dry-run` flag that loads the model and decodes only the first 5 layers.
2. Commit `expected_meta.json` with the reference values.
3. GH Actions job matrix: `{python-version: [3.10], torch: [2.3]}`.

---

### 1.9. Integrate a Tuned Lens

**Why**
Tuned Lens learns an affine probe per layer that automatically compensates for scaling and basis rotation, reducing KL by an order of magnitude and eliminating garbled early‑layer strings ([arxiv.org][2]).

**What**

*Train a lens on \~50 k tokens once per model, save to `model_id/tuned_lens.pt`, and have `run.py` optionally load it with `--use-tuned-lens`.*

**How**

1. `pip install tuned-lens==0.6.*` (RMSNorm support landed in 0.5).
2. Training snippet:

   ```python
   from tuned_lens import TunedLens, train_all_layers
   model.eval(), model.requires_grad_(False)
   lens = TunedLens(model)
   train_all_layers(lens, tokenizer, text_corpus, max_steps=500, lr=1e-4)
   lens.save_pretrained(save_dir)
   ```
3. In the sweep:

   ```python
   if args.use_tuned_lens:
       lens = TunedLens.from_pretrained(save_dir).to(model.device)
       logits = lens(hidden_states, layer_idx=ℓ)
   else:
       logits = (unembed @ resid)  # existing path
   ```
4. Store lens SHA/hash in the run metadata for provenance.

---

### 1.10. *(Optional)* Logit Prism shared decoder

**Why**
A single whitening + rotation matrix (`W_prism`) that works for *all* layers makes cross‑layer geometry directly comparable and reduces probe freedom ([neuralblog.github.io][3]).

**What**
*Fit `W_prism` on a slice of hidden states (e.g. 100 k tokens, every other layer), save to disk, and expose `--use-logit-prism` which bypasses per‑layer lenses.*

**How**

1. Sample hidden states `H` size `[N, d_model]`, compute per‑component variance and mean; whiten.
2. Solve for `R` that minimises reconstruction `‖W_U · R · H_i − logits_final‖`.  `torch.linalg.lstsq` is fine at 8–9 B scale.
3. Save `{mean, var, R}` to a `*.npz`; load at runtime and apply:

   ```python
   logits = W_U @ (R @ (resid - mean) / sqrt(var))
   ```
4. Add a CLI flag parallel to `--use-tuned-lens`; mutual‑exclusion logic ensures the user picks exactly one decoding scheme.

---

#### Wrap‑up

Executing these ten items upgrades the measurement pipeline from an informative prototype to a rigour‑grade toolchain.  Only after this foundation is secure should we move on to the broader prompt battery and causal‑intervention work.

[1]: https://arxiv.org/abs/1910.07467?utm_source=chatgpt.com "Root Mean Square Layer Normalization"
[2]: https://arxiv.org/abs/2303.08112?utm_source=chatgpt.com "Eliciting Latent Predictions from Transformers with the Tuned Lens"
[3]: https://neuralblog.github.io/logit-prisms/?utm_source=chatgpt.com "Logit Prisms: Decomposing Transformer Outputs for Mechanistic ..."

---

## 2.  Straight‑forward experimental variations on the current design

> **Philosophical background referenced**
>
> * Realists hold that universals exist mind‑independently; immanent realists say they exist “in” particulars, transcendent realists say they can exist uninstantiated ([plato.stanford.edu][4]).
> * Nominalists reject universals, often replacing them with classes, predicates, or resemblance networks of particulars ([plato.stanford.edu][5]).
> * Trope theorists accept only *particularised* properties (tropes) and treat cross‑object similarity as exact resemblance between tropes ([plato.stanford.edu][6]).

Keeping those distinctions in view, each variation below probes whether an LLM’s internal processing looks more like a single stable entity (universal) or a patchwork of particular‑tied cues (nominalist or trope‑like).

---

### 2.1. Threshold sweep for copy‑collapse

**Why**
Copy‑collapse is meant to record when the network *re‑uses a token already present in the prompt*—i.e. when it relies on **particular** lexical material.  If that layer changes drastically when the probability threshold moves from 0.90 to 0.70, the phenomenon is fragile and more consonant with nominalist “name matching” than with an entrenched universal.

**What**
Add columns `copy@0.70`, `copy@0.80`, `copy@0.90` to every pure‑next‑token CSV and emit a one‑page summary listing `L_copy(threshold)` for every model.

**How**
Compute `p_top1` per layer, evaluate the three inequalities, write booleans, and post an auto‑generated Markdown diff.

---

### 2.2. Multilingual prompt – preliminary pass

**Why**
Language‑independent behaviour is *compatible* with realism (a universal instantiated across linguistic frameworks) but *not mandated by it*.  Conversely, if depth systematically depends on language, that is prima facie evidence that the model’s “relation” is tied to particular linguistic encodings—more in line with class‑nominalism, where each language’s term picks out its own class of particulars ([plato.stanford.edu][5]).

A full causal/representational variant appears in Group 4; use this preliminary pass only as a quick consistency check until Group 3 tools are ready.

**What**
Translate the prompt into five major languages with equivalent subject–predicate order.  Record normalised `L_sem / n_layers` and visualise variance.

**How**
Maintain a YAML file of prompts keyed by ISO codes; run sweeps; bar‑plot and highlight deviations > 0.05.

---

### Closing note on epistemic modesty

None of these experiments *conclusively* vindicates realism or nominalism.  What they can do is chart **which kinds of linguistic variation the network treats as superficial and which provoke deeper representational work**.  Mapping that pattern against the philosophical taxonomy of universals, properties, and relations tells us *where* realist or nominalist readings gain empirical traction.

### Caution on metrics

Raw “semantic‑collapse depth” (the layer where the gold token first becomes top‑1) is a correlational signal. Before drawing philosophical conclusions, validate any depth‑based claim with at least one causal or representational check (activation patching, tuned‑lens KL, concept‑vector alignment). See Group 3 & 4 tasks.

[4]: https://plato.stanford.edu/entries/properties/?utm_source=chatgpt.com "Properties - Stanford Encyclopedia of Philosophy"
[5]: https://plato.stanford.edu/entries/nominalism-metaphysics/?utm_source=chatgpt.com "Nominalism in Metaphysics - Stanford Encyclopedia of Philosophy"
[6]: https://plato.stanford.edu/entries/tropes/?utm_source=chatgpt.com "Tropes - Stanford Encyclopedia of Philosophy"

---

## 3.  Advanced interpretability interventions


### 3.1. Layer‑wise activation patching (“causal tracing”)

**Why**
Correlation‑based probes can be fooled by coincidental features.  Activation patching — copying hidden state ℓ from a *corrupted* prompt (e.g. “The capital of Germany is Paris”) into the *clean* run — tests whether that layer *causally* fixes the prediction.  If a *single late layer* is decisive across many (subject, object) pairs, that looks like a reusable internal relation (realist‑friendly).  If influence is diffuse or depends on token idiosyncrasies, it fits resemblance‑ or class‑nominalism ([arxiv.org][8]).

**What**
*Given a prompt pair (clean, corrupted), produce a CSV of “causal Δ log‑prob” per layer and record `causal_L_sem` = first layer whose patch flips the top‑1 token.*

**How**

1. Implement `patch_layer(hidden_clean, hidden_corr, ℓ)` inside `run.py`.
2. For each ℓ, run the forward pass with the patched residual, decode with the tuned lens, and log Δ p(Berlin).
3. Stop when the answer flips; save `causal_L_sem` to JSON meta.
4. Add CLI flags `--patching` and `--corrupted-answer "Paris"` to ease experimentation.

---

### 3.2. Attention‑head fingerprinting near L sem

**Why**
If the binary relation *capital‑of* corresponds to a *specialised head* that consistently attends from the subject token to the object token, that is evidence of a discrete internal mechanism (akin to a realist universal).  If instead attention routes vary per prompt, the relation may be an emergent resemblance class ([arxiv.org][9], [neelnanda.io][10]).

**What**
*Catalogue all heads in layers L\_sem − 2 … L\_sem for which:*

* `attn_weight ≥ top‑k(0.8 quantile)` across heads for that layer and
* Zero‑ablation of the head drops answer log‑prob by ≥ 0.5 bits.

Store a JSON manifest `relation_heads.json` listing `(layer, head)` tuples for every model.

**How**

1. Hook attention weights in the forward pass; identify subject and candidate answer positions.
2. Compute head‑specific importance by zeroing its output vector and re‑running the remainder of the model.
3. Save heads meeting both criteria; visualise with a simple heat map.
4. Optional: run CHG (Causal Head Gating) to refine head attribution ([arxiv.org][9]).

---

### 3.3. Concept‑vector extraction via Causal Basis (CBE)

**Why**
Belrose et al. show a low‑rank subspace can *causally* steer the model’s logits ([arxiv.org][11]).  Extracting a “Berlin direction” and transplanting it into prompts about Poland probes whether the *capital‑of* universal is carried by a portable vector (strong realist evidence) or whether it is context‑bound.

**What**

*Deliver a PyTorch module `CapitalDirection` with weights `{U, Σ}` such that adding `α · U Σ v` (for a learned v) to the residual stream at layer L sem reliably increases the log‑prob of the correct capital across ≥ 80 % of country prompts, while minimally disrupting unrelated outputs.*

**How**

1. Sample 1 000 (country, capital) prompts.
2. Use the tuned lens to get layer‑ℓ logits; fit CBE on those activations to identify vectors that maximise Δ p(answer).
3. Freeze the top‑k singular directions; test generalisation on held‑out prompts.
4. Implement `apply_patch(resid, strength)` to inject the vector in new contexts.

---

### 3.4. Attribution patching for scalable causal maps

**Why**
Full activation‑patch grids scale O(L²) runs; attribution patching (gradient‑based approximation) gets the entire layer×token causal heat‑map from *three* passes ([neelnanda.io][12]).  This enables causal tracing over the entire WikiData battery without prohibitive compute.  More data gives better evidence on whether causal responsibility clusters in reusable sub‑modules (realist) or is diffuse (nominalist).

**What**
*A script `attribution_patch.py` that, for a batch of prompts, outputs an HDF5 tensor `attr[L, T]` of estimated causal contributions for every layer L and token position T, plus a notebook that plots token‑level heat‑maps.*

**How**

1. Implement the three‑pass protocol: clean forward, corrupted forward, backward pass on KL divergence.
2. Cache residuals and gradients; compute attribution scores per layer/token.
3. Validate against explicit patching on a 10‑prompt subset (correlation > 0.9).
4. Integrate into CI to run nightly on a 100‑prompt sample.

---

### 3.5. Cross‑model concept alignment (CCA / Procrustes)

**Why**
If *capital‑of‑Germany* evokes **the same activation geometry across independently trained models**, that strongly suggests an architecture‑internal universal rather than model‑specific trope.  Conversely, divergent sub‑spaces reinforce a nominalist picture of idiosyncratic classes ([arxiv.org][13]).

**What**

*Produce an analysis notebook `concept_alignment.ipynb` that:*

1. Collects layer‑L sem activations for the token “Berlin” in the same prompt across all seven models.
2. Performs CCA or orthogonal Procrustes alignment to a shared 128‑D space.
3. Reports average inter‑model cosine similarity before vs after alignment and visualises clusters.

**How**

1. Dump 10 k activation vectors per model to disk.
2. Use `sklearn.cross_decomposition.CCA` (or `emalign` for Procrustes) to learn mappings.
3. Evaluate: if mean pairwise cosine ≥ 0.6 pre‑alignment, geometry is already convergent; if it jumps only post‑alignment, differences are mostly rotational.  Interpret results in the accompanying markdown narrative.

---

### 3.6. (Optional) Causal scrubbing of candidate circuits

**Why**
Causal scrubbing replaces multiple intermediate signals at once to test entire hypothesised *circuits* for necessity and sufficiency.  It can falsify the “single relation head” story by showing the answer still emerges when that head’s output is replaced with noise.

**What**
Encode a circuit hypothesis (subject‑head→MLP→answer) in a Python spec and automatically test all 2ᴺ subsets of components, outputting a table of accuracy drops.

**How**

1. Adopt the Open‑source `causal‑scrubbing` library.
2. Write a spec file mapping nodes to model components.
3. Run exhaustive subset ablations on a 100‑prompt subset; visualise results as a lattice diagram.

---

## Philosophical pay‑off

* **Causal patching** distinguishes *where* the model irrevocably instantiates the capital‑of relation, countering the nominalist claim that apparent universals are artefacts of shallow token overlap.
* **Head fingerprinting** and **concept vectors** probe whether that relation is localised and portable—the hallmarks of a realist universal—versus being context‑specific.
* **Cross‑model alignment** asks whether the same entity recurs across distinct training histories, a requirement for *trans‑instance* universality stressed in SEP’s discussion of immanent realism ([arxiv.org][11]).
* **Attribution patching** and **causal scrubbing** broaden the evidence base from one prompt to thousands, mitigating cherry‑picking and allowing statistical arguments.

Together, these interventions push the project from **descriptive** lens diagnostics to **manipulative** evidence about the inner ontology of LLMs—crucial ground for any serious engagement with the realism‑versus‑nominalism debate.

[8]: https://arxiv.org/abs/2202.05262?utm_source=chatgpt.com "Locating and Editing Factual Associations in GPT"
[9]: https://www.arxiv.org/pdf/2505.13737?utm_source=chatgpt.com "[PDF] A Framework for Interpreting Roles of Attention Heads in Transformers"
[10]: https://www.neelnanda.io/mechanistic-interpretability/glossary?utm_source=chatgpt.com "A Comprehensive Mechanistic Interpretability Explainer & Glossary"
[11]: https://arxiv.org/abs/2303.08112?utm_source=chatgpt.com "Eliciting Latent Predictions from Transformers with the Tuned Lens"
[12]: https://www.neelnanda.io/mechanistic-interpretability/attribution-patching?utm_source=chatgpt.com "Attribution Patching: Activation Patching At Industrial Scale"
[13]: https://arxiv.org/html/2310.12794v2?utm_source=chatgpt.com "Are Structural Concepts Universal in Transformer Language Models ..."

## 4. Ontology‑Focused Evaluations Using Causal & Representational Metrics

These studies go beyond first‑pass collapse‑depth checks. They require the advanced techniques from Group 3 (tuned lens, layer‑wise activation patching, attention‑head fingerprinting, concept‑vector extraction). Each experiment asks whether the network’s internal machinery behaves more like a stable universal or like an assemblage of particulars once those richer metrics are in hand.

---

### 4.1.  Instruction‑Language Ablation with Causal Metrics

**Why**
Pragmatic words (“please”, “simply”) are speech‑act particulars. We need to know whether they merely nudge surface probabilities or actually change *where* the model fixes the **capital‑of** relation.

**What**
Run the original prompt and a “plain” prompt; record
\* (a) the tuned‑lens KL‑curve inflection,
\* (b) **causal L\_sem** obtained by single‑layer activation patching, and
\* (c) Δ log‑prob when the top two “style heads” (found via head fingerprinting) are zeroed.

**How**

1. Generate both prompts; tag `variant=instruct/plain`.
2. For each, sweep layers, patch the corrupted prompt at ℓ until the answer flips; store causal L\_sem.
3. During the clean run, zero candidate style heads and measure answer log‑prob drop.
4. Summarise:
   `Gemma‑9B — causal L_sem unchanged (45→45); style‑head ablation −0.1 bits ⇒ semantics robust to pragmatics.`

---

### 4.2.  English Paraphrase Robustness via Causal L\_sem and Vector Alignment

**Why**
Nominalists would expect relation processing to hinge on close lexical resemblance; realists expect stability across paraphrases.

**What**
Ten English paraphrases. For each:
\* (a) causal L\_sem,
\* (b) cosine similarity of the answer‑logit direction to the canonical Berlin vector after whitening,
\* (c) tuned‑lens KL divergence at L\_sem.

Visualise variance; compute coefficient of variation (CV) of causal L\_sem.

**How**

1. Store paraphrases in YAML.
2. Batch‑run; cache residuals for concept‑vector whitening.
3. Use the concept‑vector module to obtain Berlin direction per paraphrase; compute cosines.
4. Plot violin of causal L\_sem; print `CV = 0.06` (low) or `CV = 0.32` (high).

---

### 4.3.  Multilingual Prompt Study with Concept‑Vector Consistency

**Why**
If an LLM hosts an *immanent* universal for **capital‑of**, the Berlin direction in German, Spanish, Arabic, etc. should align up to rotation.

**What**
Five language versions of the prompt. Measure:
\* (a) tuned‑lens KL inflection layer,
\* (b) cosine between each language’s Berlin vector and the English one *after the language‑specific whitening transforms*,
\* (c) causal L\_sem.

**How**

1. Verify translations keep subject–predicate order.
2. Extract concept vectors; apply whitening per language.
3. Compute pairwise cosines; output a short Markdown table of `⟨cos⟩ = 0.71 ± 0.05` or similar.
4. Flag languages whose causal L\_sem deviates > 10 % of total layer count.

---

### 4.4.  Large WikiData “Capital‑of” Battery with Causal Statistics

**Why**
A universal should work across hundreds of particulars. We want the distribution of **causal L\_sem** and whether token‑level features predict it.

**What**
1 000–5 000 (country, capital) prompts. For each: causal L\_sem, answer token length, frequency.
Output:
\* Histogram of causal L\_sem,
\* OLS regression `L_sem ∼ len + log_freq`.
Interpret whether β‑coefficients are near zero (token features irrelevant) or large (token features matter → nominalist reading).

**How**

1. Use activation patching in batched mode (two passes per prompt: clean & patch grid).
2. Compute causal L\_sem for each.
3. Fit regression; print `R²`.
4. Store results in `battery_capital.csv`.

---

### 4.5.  Lexical‑Ambiguity Stress Test with Entropy Plateau & Head Timing

**Why**
Ambiguous names instantiate multiple resemblance classes. If the model delays commitment (maintains high entropy) or fires relation heads later, that suggests on‑the‑fly disambiguation.

**What**
50 ambiguous vs 50 control prompts. Metrics:
\* (a) entropy plateau height (mean entropy over layers before causal L\_sem),
\* (b) first‑firing layer of the dominant relation head (from fingerprinting).

Statistical test: Wilcoxon on each metric.

**How**

1. Curate ambiguous list (“Georgia”, “Jordan”).
2. Run sweeps with attention recording.
3. Detect dominant head per prompt (`attn_weight > 0.2`).
4. Compute layer index; perform paired non‑parametric test; print p‑values.

---

### 4.6.  Instruction‑Style Grid with Causal Metrics

**Why**
Separates predicate content from pragmatics on a large scale.

**What**
12 prompt styles (4 modifiers × 3 moods) run over the WikiData battery. For each cell:
\* mean causal L\_sem,
\* mean log‑prob drop when style heads are ablated,
\* mean tuned‑lens KL at L\_sem.

Heat‑map the three statistics.

**How**

1. Auto‑generate prompt grid.
2. Batch activation patching; reuse style‑head list.
3. Aggregate per cell; render three matplotlib heat‑maps.

---

### 4.7.  Property vs Kind Probe Using Vector Rank & Separability

**Why**
The SEP notes that properties and kinds are different metaphysical categories; do LLMs reflect that in their internal geometry?

**What**
200 property sentences (`The cat is black.`) and 200 kind sentences (`Berlin is a city.`). Metrics:
\* (a) rank of correct answer in tuned‑lens logits at layer ℓ = 0.5 · n\_layers,
\* (b) linear‑probe AUC that separates property vs kind token directions in early layers,
\* (c) causal L\_sem per prompt.

**How**

1. Collect adjective and noun lists.
2. Extract hidden states at mid‑stack; train logistic classifier.
3. Record rank and AUC; compare distributions property vs kind.

---

### 4.8.  Symmetric vs Asymmetric Relations via Head Structure and Causality

**Why**
A symmetric universal (“distance‑between”) might be encoded by a single bidirectional head; an asymmetric one (“west‑of”) may need polarity encoding.

**What**
100 city pairs with distance & bearing. For each relation class:
\* dominant head layer index,
\* head‑polarity test: swap query and key positions and see if attention persists.
Measure causal log‑prob drop when that head is ablated.

**How**

1. Build prompts: `Berlin is 878 km from` ⟨NEXT⟩ vs `Berlin is east of` ⟨NEXT⟩.
2. Record attention maps; detect polarity.
3. Ablate head and recompute logits; log Δ.

---

### 4.9.  (Optional) Trope‑Sensitivity Probe with Context‑Specific Vectors

**Why**
Trope theory holds that each instantiation of a property is particularised. If injecting a “blackness” vector learned on one object raises log‑prob for *another* black object, that undercuts trope accounts.

**What**
50 noun pairs with property adjective (“black pawn”, “black asphalt”).
Metrics:
\* (a) cosine similarity between black‑vectors extracted in each context,
\* (b) causal increase in p(“black”) when vector from A is patched into B.

**How**

1. Use concept‑vector extraction per sentence.
2. Compute cross‑context cosines; perform activation patching.
3. Report mean Δ log‑prob and compare to intra‑context baseline.

---

### Implementation Dependency Notice

All experiments above **presuppose** that the following Group‑3 capabilities are available and validated:

\* **Tuned / Prism lens** for reliable logits and KL curves.
\* **Layer‑wise activation patching** to determine causal L\_sem and to test vector transplant effects.
\* **Attention‑head fingerprinting** for relation‑ and style‑head discovery.
\* **Concept‑vector extraction** (CBE) for measuring vector similarity and portability.

Without those, the metrics would revert to raw collapse depth and lose their interpretive bite.


# Audience
- Software engineer, growing ML interpretability knowledge
- No formal ML background but learning through implementation
- Prefers systematic, reproducible experiments with clear documentation

