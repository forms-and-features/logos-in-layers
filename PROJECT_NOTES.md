# Interpretability Project - Development Notes for AI assistant

# Philosophical Project Context
**Goal**: Use interpretability to inform nominalism vs realism debate.

# Next steps

Items are ordered by the approximate engineering lift required.

---

### 1. Fix the RMS/LN scaling path (γ + ε placement)

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

### 2. Sub‑word‑aware copy‑collapse detector

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

### 3. Record top‑1 p, top‑5 p\_cumulative, and KL‑to‑final

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
4. Update `output-gemma-2-9b-pure-next-token.csv` schema version number so downstream notebooks fail loudly if they expect old columns.

---

### 4. Raw‑activation lens toggle

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

### 5. Representation‑drift cosine curve

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

### 6. Negative‑control prompt

**Why**
If Berlin outranks Paris in “The capital of *France* is …”, your probe is leaking string co‑occurrence, invalidating any metaphysical claim.

**What**
*Run every model on one additional control prompt (“Give the city name only … France …”) and log the same metrics.*

**How**

1. `PROMPTS = [positive_prompt, control_prompt]`.
2. Loop over prompts; append a `prompt_id` column to CSV and JSON.
3. In the analysis notebook auto‑flag any layer where `p_paris < p_berlin` in the control row.

---

### 7. Ablate stylistic filler (`simply`)

**Why**
Gemma’s early copy‑collapse may be triggered by instruction‑style cues, not semantics.  Removing “simply” tests that hypothesis.

**What**
*Rerun the base prompt with and without filler and compare `L_copy`, `L_semantic`.*

**How**

1. Duplicate prompt, drop the adverb.
2. Record both runs with `prompt_variant` metadata.
3. Plot `Δ-collapse` for the two variants; a large shift confirms the stylistic‑cue explanation.

---

### 8. Lightweight CI / regression harness

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

### 9. Integrate a Tuned Lens

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

### 10. *(Optional)* Logit Prism shared decoder

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

## 2 .  Straight‑forward experimental variations on the current design

> **Philosophical background referenced**
>
> * Realists hold that universals exist mind‑independently; immanent realists say they exist “in” particulars, transcendent realists say they can exist uninstantiated ([plato.stanford.edu][1]).
> * Nominalists reject universals, often replacing them with classes, predicates, or resemblance networks of particulars ([plato.stanford.edu][2]).
> * Trope theorists accept only *particularised* properties (tropes) and treat cross‑object similarity as exact resemblance between tropes ([plato.stanford.edu][3]).

Keeping those distinctions in view, each variation below probes whether an LLM’s internal processing looks more like a single stable entity (universal) or a patchwork of particular‑tied cues (nominalist or trope‑like).

---

### 1. Threshold sweep for copy‑collapse

**Why**
Copy‑collapse is meant to record when the network *re‑uses a token already present in the prompt*—i.e. when it relies on **particular** lexical material.  If that layer changes drastically when the probability threshold moves from 0.90 to 0.70, the phenomenon is fragile and more consonant with nominalist “name matching” than with an entrenched universal.

**What**
Add columns `copy@0.70`, `copy@0.80`, `copy@0.90` to every pure‑next‑token CSV and emit a one‑page summary listing `L_copy(threshold)` for every model.

**How**
Compute `p_top1` per layer, evaluate the three inequalities, write booleans, and post an auto‑generated Markdown diff.

---

### 2. Remove instruction framing

**Why**
Realist arguments apply to *predicate content* (“capital‑of”) not to metalinguistic directives (“give the answer plainly”).  Instruction tokens are closer to speech‑act particulars; their removal tests whether collapse depths depend on such particulars.

**What**
Use prompt variant:
`The capital of Germany is`
Run the full sweep; report differences in `L_copy`, `L_sem`, `Δ-collapse`.

**How**
Tag with `prompt_variant="plain"`, rerun, and diff against baseline.

---

### 3. Paraphrase set in one language

**Why**
Resemblance‑nominalists say that general terms apply because their instances resemble one another; if small wording changes push the semantic‑collapse layer wildly, that would indicate the model tracks *surface resemblance* rather than an invariant internal relation.

**What**
Ten human‑written paraphrases (changing mood, syntax, synonyms); produce distribution plots of `L_sem`.

**How**
Store paraphrases in YAML; sweeping loop iterates; analysis notebook aggregates and visualises.

---

### 4. Multilingual prompt study

**Why**
Language‑independent behaviour is *compatible* with realism (a universal instantiated across linguistic frameworks) but *not mandated by it*.  Conversely, if depth systematically depends on language, that is prima facie evidence that the model’s “relation” is tied to particular linguistic encodings—more in line with class‑nominalism, where each language’s term picks out its own class of particulars ([plato.stanford.edu][2]).

**What**
Translate the prompt into five major languages with equivalent subject–predicate order.  Record normalised `L_sem / n_layers` and visualise variance.

**How**
Maintain a YAML file of prompts keyed by ISO codes; run sweeps; bar‑plot and highlight deviations > 0.05.

---

### 5. WikiData capital‑of battery

**Why**
Treating **(capital, country)** pairs en masse operationalises the *binary relation* without privileging any single particular.  If the distribution of `L_sem` is narrow and unrelated to token idiosyncrasies, that suggests the model employs an internal pattern applicable across particulars—closer to an immanent universal.  If depth is predicted by token length or frequency, that supports a nominalist or trope‑style account where processing hinges on the micro‑features of each instantiation.

**What**
SPARQL‑extract up to 5 000 capital pairs, run sweeps, and regress `L_sem` on answer‑token length, log frequency, rarity, etc.  Include histograms of `Δ-collapse`.

**How**
Batch prompts through the existing pipeline; analytics in pandas; output Markdown with R² and plots.

---

### 6. Lexical‑ambiguity stress test

**Why**
Ambiguous names (“Georgia”) instantiate multiple resemblance‑classes of particulars.  Nominalists predict the model must delay commitment when resemblance is insufficient to disambiguate.  Realists could still say the *universal* capital‑of‑Georgia exists, but why the network reaches it later becomes an empirical question.

**What**
50 ambiguous and 50 control prompts.  Compute median `L_sem` per set; Wilcoxon signed‑rank test for shift.

**How**
Curate list manually or by regex; tag prompts; run.  A notebook prints p‑value and effect size.

---

### 7. Instruction‑style ablation grid

**Why**
Speech‑act features (“please”, “in one word”) are neither universals nor particulars of the capital‑relation—they are pragmatic cues.  If they systematically move `L_sem`, the probe is partly reading pragmatics, not ontology.

**What**
12 prompt variants (4 style modifiers × 3 sentence moods).  Produce a heat map of `L_sem` averaged over WikiData battery.

**How**
Generate prompts programmatically; tag with `style`, `mood`; sweep; pivot‑table to heat map.

---

### 8. Property‑vs‑Kind probe (monadic universals)

**Why**
Universals divide into **properties** (adjectival “is‑black”) and **kinds** (substantial “is‑a‑city”).  SEP’s entry on *properties* notes that some nominalists treat adjectival cases via resemblance but prefer class constructions for kind membership ([plato.stanford.edu][1]).  Comparing collapse depth across these two monadic types asks whether the network treats them alike or differently.

**What**
Create prompt families:
*Property:* `The cat is` ⟨NEXT⟩ → “black”.
*Kind:* `Berlin is a` ⟨NEXT⟩ → “city”.
Run 200 instances of each; compare median depths.

**How**
Hand‑collect adjectives and noun kinds or mine WikiData.  Tag `univ_type = property/kind`; run sweeps; simple stats.

---

### 9. Symmetric vs asymmetric relations

**Why**
The SEP’s account of universals notes that relations can be symmetric (being 878 km‑from) or asymmetric (being west‑of) ([plato.stanford.edu][4]).  If the network encodes polarity as an additional feature layered on top of an otherwise shared relation representation, asymmetric prompts may collapse later than symmetric ones.

**What**
Prompt pairs:
`Berlin is 878 km from` ⟨NEXT⟩ → “Paris” (symmetric)
`Berlin is east of` ⟨NEXT⟩ → “Paris” (asymmetric)
Measure `L_sem` across 100 such pairs, plot depth distributions.

**How**
Use OpenStreetMap API for city distances/bearings; generate prompts; tag `relation_symmetry`; run sweeps; plot violin.

---

### 10. (Optional) Trope‑sensitivity probe

**Why**
If the model’s representation is *trope‑like*, each instance of “black‑ness” in a sentence should be tied to context.  Replacing one black object with another visually different black object should therefore increase semantic collapse depth—because the resemblance net needs recalibrating.

**What**
Craft prompts pairwise:
`The chess pawn is black.`
`The asphalt is black.`
Remove the object and ask for the adjective.  Compare `L_sem` when the surrounding context is varied.

**How**
Select 50 noun pairs differing in visual form; run sweeps; compute per‑pair depth deltas; perform paired t‑test.

---

### Closing note on epistemic modesty

None of these experiments *conclusively* vindicates realism or nominalism.  What they can do is chart **which kinds of linguistic variation the network treats as superficial and which provoke deeper representational work**.  Mapping that pattern against the philosophical taxonomy of universals, properties, and relations tells us *where* realist or nominalist readings gain empirical traction.

[1]: https://plato.stanford.edu/entries/properties/?utm_source=chatgpt.com "Properties - Stanford Encyclopedia of Philosophy"
[2]: https://plato.stanford.edu/entries/nominalism-metaphysics/?utm_source=chatgpt.com "Nominalism in Metaphysics - Stanford Encyclopedia of Philosophy"
[3]: https://plato.stanford.edu/entries/tropes/?utm_source=chatgpt.com "Tropes - Stanford Encyclopedia of Philosophy"
[4]: https://plato.stanford.edu/entries/universals-medieval/?utm_source=chatgpt.com "The Medieval Problem of Universals"


---

## 3 .  Advanced interpretability interventions


### 1. Layer‑wise activation patching (“causal tracing”)

**Why**
Correlation‑based probes can be fooled by coincidental features.  Activation patching — copying hidden state ℓ from a *corrupted* prompt (e.g. “The capital of Germany is Paris”) into the *clean* run — tests whether that layer *causally* fixes the prediction.  If a *single late layer* is decisive across many (subject, object) pairs, that looks like a reusable internal relation (realist‑friendly).  If influence is diffuse or depends on token idiosyncrasies, it fits resemblance‑ or class‑nominalism ([arxiv.org][1]).

**What**
*Given a prompt pair (clean, corrupted), produce a CSV of “causal Δ log‑prob” per layer and record `causal_L_sem` = first layer whose patch flips the top‑1 token.*

**How**

1. Implement `patch_layer(hidden_clean, hidden_corr, ℓ)` inside `run.py`.
2. For each ℓ, run the forward pass with the patched residual, decode with the tuned lens, and log Δ p(Berlin).
3. Stop when the answer flips; save `causal_L_sem` to JSON meta.
4. Add CLI flags `--patching` and `--corrupted-answer "Paris"` to ease experimentation.

---

### 2. Attention‑head fingerprinting near L sem

**Why**
If the binary relation *capital‑of* corresponds to a *specialised head* that consistently attends from the subject token to the object token, that is evidence of a discrete internal mechanism (akin to a realist universal).  If instead attention routes vary per prompt, the relation may be an emergent resemblance class ([arxiv.org][2], [neelnanda.io][3]).

**What**
*Catalogue all heads in layers L\_sem − 2 … L\_sem for which:*

* `attn_weight(subject→answer) > 0.20` and
* Zero‑ablation of the head drops answer log‑prob by ≥ 0.5 bits.

Store a JSON manifest `relation_heads.json` listing `(layer, head)` tuples for every model.

**How**

1. Hook attention weights in the forward pass; identify subject and candidate answer positions.
2. Compute head‑specific importance by zeroing its output vector and re‑running the remainder of the model.
3. Save heads meeting both criteria; visualise with a simple heat map.
4. Optional: run CHG (Causal Head Gating) to refine head attribution ([arxiv.org][2]).

---

### 3. Concept‑vector extraction via Causal Basis (CBE)

**Why**
Belrose et al. show a low‑rank subspace can *causally* steer the model’s logits ([arxiv.org][4]).  Extracting a “Berlin direction” and transplanting it into prompts about Poland probes whether the *capital‑of* universal is carried by a portable vector (strong realist evidence) or whether it is context‑bound.

**What**

*Deliver a PyTorch module `CapitalDirection` with weights `{U, Σ}` such that adding `α · U Σ v` (for a learned v) to the residual stream at layer L sem reliably increases the log‑prob of the correct capital across ≥ 80 % of country prompts, while minimally disrupting unrelated outputs.*

**How**

1. Sample 1 000 (country, capital) prompts.
2. Use the tuned lens to get layer‑ℓ logits; fit CBE on those activations to identify vectors that maximise Δ p(answer).
3. Freeze the top‑k singular directions; test generalisation on held‑out prompts.
4. Implement `apply_patch(resid, strength)` to inject the vector in new contexts.

---

### 4. Attribution patching for scalable causal maps

**Why**
Full activation‑patch grids scale O(L²) runs; attribution patching (gradient‑based approximation) gets the entire layer×token causal heat‑map from *three* passes ([neelnanda.io][5]).  This enables causal tracing over the entire WikiData battery without prohibitive compute.  More data gives better evidence on whether causal responsibility clusters in reusable sub‑modules (realist) or is diffuse (nominalist).

**What**
*A script `attribution_patch.py` that, for a batch of prompts, outputs an HDF5 tensor `attr[L, T]` of estimated causal contributions for every layer L and token position T, plus a notebook that plots token‑level heat‑maps.*

**How**

1. Implement the three‑pass protocol: clean forward, corrupted forward, backward pass on KL divergence.
2. Cache residuals and gradients; compute attribution scores per layer/token.
3. Validate against explicit patching on a 10‑prompt subset (correlation > 0.9).
4. Integrate into CI to run nightly on a 100‑prompt sample.

---

### 5. Cross‑model concept alignment (CCA / Procrustes)

**Why**
If *capital‑of‑Germany* evokes **the same activation geometry across independently trained models**, that strongly suggests an architecture‑internal universal rather than model‑specific trope.  Conversely, divergent sub‑spaces reinforce a nominalist picture of idiosyncratic classes ([arxiv.org][6]).

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

### 6. (Optional) Causal scrubbing of candidate circuits

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
* **Cross‑model alignment** asks whether the same entity recurs across distinct training histories, a requirement for *trans‑instance* universality stressed in SEP’s discussion of immanent realism ([arxiv.org][4]).
* **Attribution patching** and **causal scrubbing** broaden the evidence base from one prompt to thousands, mitigating cherry‑picking and allowing statistical arguments.

Together, these interventions push the project from **descriptive** lens diagnostics to **manipulative** evidence about the inner ontology of LLMs—crucial ground for any serious engagement with the realism‑versus‑nominalism debate.

[1]: https://arxiv.org/abs/2202.05262?utm_source=chatgpt.com "Locating and Editing Factual Associations in GPT"
[2]: https://www.arxiv.org/pdf/2505.13737?utm_source=chatgpt.com "[PDF] A Framework for Interpreting Roles of Attention Heads in Transformers"
[3]: https://www.neelnanda.io/mechanistic-interpretability/glossary?utm_source=chatgpt.com "A Comprehensive Mechanistic Interpretability Explainer & Glossary"
[4]: https://arxiv.org/abs/2303.08112?utm_source=chatgpt.com "Eliciting Latent Predictions from Transformers with the Tuned Lens"
[5]: https://www.neelnanda.io/mechanistic-interpretability/attribution-patching?utm_source=chatgpt.com "Attribution Patching: Activation Patching At Industrial Scale"
[6]: https://arxiv.org/html/2310.12794v2?utm_source=chatgpt.com "Are Structural Concepts Universal in Transformer Language Models ..."

# User Context
- Software engineer, growing ML interpretability knowledge
- No formal ML background but learning through implementation
- Prefers systematic, reproducible experiments with clear documentation

