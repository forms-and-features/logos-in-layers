# Interpretability Project – Development Notes for AI Assistant

## Philosophical Project Context

**Goal.** Bring concrete interpretability data to the centuries‑old dispute between **nominalism** and **realism** about universals. The first iterations target the low bar: **austere (extreme) nominalism**, which claims that only particular tokens exist and any talk of properties or relations can be paraphrased away.

By showing that LLMs contain robust, reusable internal structures, detected through logit‑lens baselines and causal patches, we aim to gather empirical pressure against that austere view. Once those methods are sound and the anti‑austere evidence is in hand, the project will move to the harder task of discriminating between **metalinguistic nominalism** (which treats those structures as facts about words and predicates) and **realism** (which treats them as evidence of mind‑independent universals).

---

## Provenance & Conventions (read me first)

**Provenance.** The `run-latest` results dated by `run-latest/timestamp-*` were produced **before** Section **1.1**’s normalization fix was merged. Treat those files as **V1 (pre‑fix)** baselines. All re‑runs after the fix are **V2 (post‑fix)**. Record both in write‑ups when comparing depths or probabilities.
Runs after 2025‑08‑24 also decode logits with an fp32 unembedding when the model computes in bf16/fp16, and compute LN/RMS statistics in fp32 before casting back. This improves numerical stability without changing defaults for ≤27B CPU runs.

**Layer indexing.** We decode **post‑block** unless otherwise stated. Rows `layer = 0 … (n_layers − 1)` are post‑block residuals; `layer = n_layers` is the **final unembed head** (the model’s actual output distribution).

**Answer matching is ID‑level.** We determine the gold **first answer token id** from the model’s tokenizer applied to `prompt + " Berlin"` (or control answer). All `is_answer` logic compares **token IDs**, not strings. We log the entire answer tokenisation for transparency (see §1.11).

**Cross‑model caution.** RMS/LN lenses can distort **absolute** probabilities and entropies in model‑specific ways. Only compare **within** a model unless using a Tuned Lens or a shared decoder (Logit Prism). Cross‑model claims should be phrased in terms of **relative** or **rank‑based** metrics (e.g., KL‑to‑final thresholds).

---

# Next steps

Items are ordered by the approximate engineering lift required.

---

## 1. Get the measurement right

Before we can claim that LLMs house structures too systematic for austere nominalism, our probes themselves must be trustworthy. This stage therefore focuses on scrubbing away every obvious source of numerical noise or probe artefact.

### 1.1. Fix the RMS/LN scaling path (γ + ε placement)

**Why.** If you normalise a *different* residual stream (post‑block) with γ that was trained for the *pre‑block* stream, logits are systematically mis‑scaled; early‑layer activations can be inflated by >10×. An incorrect ε outside the square‑root likewise shifts all norms upward. These distortions then propagate through the logit lens, giving spurious “early meaning” or hiding true signal. RMSNorm’s official formula places ε **inside** the √ and multiplies by γ afterwards ([arxiv.org][1]).

**What.** Apply RMS/LN γ and ε to the right residual stream; fix the ε‑outside‑sqrt bug.

**How.**

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
3\) Add a unit test that decodes layer 0 twice: once with γ=1, once with learned γ. The KL between them should match KL between *raw* hidden states with and without γ, proving that scaling now matches semantics.

**✅ IMPLEMENTATION STATUS: COMPLETED**

* All epsilon placement, architecture detection, and validation requirements implemented and unit‑tested.
* Numerical precision policy: When compute dtype is bf16/fp16 (e.g., large CPU runs), unembedding/decoding uses fp32 and LN/RMS statistics are computed in fp32 and cast back. This stabilizes small logit gaps and entropy at negligible memory cost.
* **Provenance note:** any artefacts in `001_layers_and_logits/run-latest/*` prior to commit *8fa091ce95f6fc68ae9dbca1a27d29ec736b2e8d* are **pre‑fix**. Re‑run those models to obtain **V2** numbers before drawing cross‑model conclusions.

---

### 1.2. Sub‑word‑aware copy‑collapse detector

**Why.** For BPE/WordPiece vocabularies the answer “Berlin” may surface as two tokens (“▁Berlin”, “Ber▁lin”, etc.). The current string‑match can miss that, under‑counting copy events and making Gemma look unique when it may not be.

**What.** *Detect prompt echo at the string level after detokenisation, regardless of how many word‑pieces the tokenisation used.*

**How.**

0\) Normalise both sides once for robust matching:

```python
prompt_norm = tokenizer.decode(tokenizer.encode(prompt, add_special_tokens=False)).strip()
cand_detok = tokenizer.decode([top1_id]).strip()
```

1. Replace:

```python
candidate = tokenizer.decode(top1_id)  # returns single piece
collapse = (candidate in prompt) and (top1_p > THRESH)
```

with:

```python
collapse = (cand_detok in prompt_norm) and (top1_p > THRESH)
```

2. Optional: allow contiguous multi‑token matches by keeping a rolling window of the last *k* best tokens, detokenising them jointly, and checking membership.
3. Parameterise the probability threshold; expose `--copy-thresh` CLI option so reviewers can run sensitivity analyses.
4. Emit `copy_thresh` into the JSON meta for provenance.

---

### 1.3. Record top‑1 p, top‑5 p\_cumulative, **p\_answer**, and KL‑to‑final

**Why.** Entropy alone conflates “one spike” vs “five near‑ties”. KL(ℓ ∥ final) is the metric used in the tuned‑lens paper to show convergence ([arxiv.org][2]). These curves tell you whether the model is already *near* its final answer direction (amplification) or still rotating into place (construction).

**What.** *Four new floating‑point columns in every `*-pure-next-token.csv`: `p_top1`, `p_top5`, `p_answer`, `kl_to_final_bits`.* Also compute a run‑summary field `first_kl_below_{0.5,1.0}` (in bits) to mark the earliest layer whose distribution is within τ bits of the final head.

**How.**

1. During the logit sweep cache the final distribution once:

```python
final_logits = all_logits[-1]                    # shape [V]
final_probs  = final_logits.softmax(dim=-1)      # dtype float32
```

2. Per layer:

```python
probs   = layer_logits.softmax(dim=-1, dtype=torch.float32)
p_top1  = probs[top1_id]
p_top5  = probs[torch.topk(probs, 5).indices].sum()
p_answer = probs[first_ans_id]                   # from §1.11
kl_bits = torch.kl_div(probs.log(), final_probs, reduction="sum") / math.log(2)  # bits
```

3. Append to the CSV writer.
4. Store `kl_to_final_bits` even when the tuned lens is active (it then measures residual mismatch, not absolute).
5. After the sweep, derive `first_kl_below_0.5` and `first_kl_below_1.0` (layer indices, or `null`) and write them into the JSON meta.

---

### 1.4. Raw‑activation lens toggle

**Why.** If “early meaning” disappears when you skip normalisation, that meaning was an artefact of the lens, not the model.

**What.** *A boolean CLI flag `--raw-lens` that bypasses `apply_norm_or_skip`.*

**How.**

1. Add `parser.add_argument("--raw-lens", action="store_true")`.
2. Wrap the call:

```python
resid_to_decode = residual if args.raw_lens else apply_norm_or_skip(residual, norm_mod)
```

3. Output the flag value into the JSON meta file so that result artefacts are traceable.
4. Optional `--dual-lens N` to emit both raw and norm lens rows for layers `0..N-1` with a `lens` column ∈ {`raw`,`norm`} (debug only).

---

### 1.5. Representation‑drift cosine curve

**Why.** A realist reading predicts an answer‑token direction that exists early and merely grows in magnitude; a nominalist picture predicts the direction rotates into place late. Cosine similarity across depth quantifies which is true.

**What.** *A per‑layer scalar `cos_to_final` written alongside entropy metrics.*

**How.**

1. Compute once:

```python
final_dir = final_logits / final_logits.norm()
```

2. At each layer:

```python
curr_dir = layer_logits / layer_logits.norm()
cos = torch.dot(curr_dir, final_dir).item()
```

3. Write `cos_to_final` column; include in diagnostic plots.

**Note.** Cosine is computed on **logit directions** (not residuals). With a Tuned Lens (§1.9), `cos_to_final` measures closeness to the **tuned** head—still interpretable as “distance to the model’s decision boundary”.

---

### 1.6. Negative‑control prompt

**Why.** If Berlin outranks Paris in “The capital of *France* is …”, your probe is leaking string co‑occurrence, undermining any metaphysical claim.

**What.** *Run every model on one additional control prompt (“Give the city name only … France …”) and log the same metrics.*

**How.**

1. `PROMPTS = [positive_prompt, control_prompt]`.
2. Loop over prompts; append a `prompt_id` column to CSV and JSON.
3. In the analysis notebook auto‑flag any layer where the **ID‑level** probability for `Paris` in the control row is lower than the **ID‑level** probability for `Berlin` in that same row, i.e.
   `p_answer_control(first_id_Paris) < p_answer_control(first_id_Berlin)` (IDs from §1.11).

---

### 1.7. Ablate stylistic filler (`simply`)

**Why.** Gemma’s early copy‑collapse may be triggered by instruction‑style cues, not semantics. Removing “simply” tests that hypothesis.

**What.** *Rerun the base prompt with and without filler and compare `L_copy`, `L_semantic`.*

**How.**

1. Duplicate prompt; drop the adverb.
2. Record both runs with `prompt_variant` metadata.
3. Plot `Δ-collapse` for the two variants; a large shift confirms the stylistic‑cue explanation.

---

### 1.8. Lightweight CI / regression harness

**Why.** As soon as you integrate tuned lens or refactor, you need guard‑rails ensuring that numbers do not silently drift.

**What.** *A GitHub Actions workflow that executes `python run.py --models meta-llama/Meta-Llama-3-8B --cpu --dry-run` and checks that:*

* JSON meta contains the new schema keys
* `L_semantic` remains within ±1 layer of an expected value stored in `expected.json`.

**How.**

1. In `run.py` add a `--dry-run` flag that loads the model and decodes only the first 5 layers.
2. Commit `expected_meta.json` with the reference values.
3. GH Actions job matrix: `{python-version: [3.10], torch: [2.3]}`.
4. Ensure JSON meta includes: `gold_answer.first_id`, `gold_answer.pieces`, `copy_thresh`, lens flags (`use_norm_lens`, `raw_lens`, `use_tuned_lens`, `use_logit_prism`), and dtypes (`dtype`, `unembed_dtype`).

---

### 1.9. Integrate a Tuned Lens

**Why.** Tuned Lens learns an affine probe per layer that automatically compensates for scaling and basis rotation, reducing KL by an order of magnitude and eliminating garbled early‑layer strings ([arxiv.org][2]).

**What.** *Train a lens on \~50k tokens once per model, save to `model_id/tuned_lens.pt`, and have `run.py` optionally load it with `--use-tuned-lens`.*

**How.**

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

**Why.** A single whitening + rotation matrix (`W_prism`) that works for *all* layers makes cross‑layer geometry directly comparable and reduces probe freedom ([neuralblog.github.io][3]).

**What.** *Fit `W_prism` on a slice of hidden states (e.g., 100k tokens, every other layer), save to disk, and expose `--use-logit-prism` which bypasses per‑layer lenses.*

**How.**

1. Sample hidden states `H` size `[N, d_model]`, compute per‑component variance and mean; whiten.
2. Solve for `R` that minimises reconstruction `‖W_U · R · H_i − logits_final‖`. `torch.linalg.lstsq` is fine at 8–9B scale.
3. Save `{mean, var, R}` to a `*.npz`; load at runtime and apply:

```python
logits = W_U @ (R @ (resid - mean) / sqrt(var))
```

4. Add a CLI flag parallel to `--use-tuned-lens`; mutual‑exclusion logic ensures the user picks exactly one decoding scheme.

---

### 1.11. Gold‑token alignment (leading‑space, multi‑piece)

**Why.** Many tokenizers produce a leading‑space piece for the first word of an answer (e.g., `"▁Berlin"` / `" Berlin"`). String‑level equality can silently mismatch; **ID‑level matching** is robust across tokenizers.

**What.** *Compute the gold **first answer token id** from `tokenizer(prompt + " Berlin")` and record both the id and full answer tokenisation in the JSON meta.*

**How.**

1. Tokenise once:

```python
ans_pieces  = tokenizer.encode(" Berlin", add_special_tokens=False)
ctx_ids     = tokenizer.encode(prompt, add_special_tokens=False)
ctx_ans_ids = tokenizer.encode(prompt + " Berlin", add_special_tokens=False)
first_ans_id = ctx_ans_ids[len(ctx_ids)]
```

2. Use `first_ans_id` for `is_answer` and for `p_answer` logging (see §1.3).
3. In the JSON meta add:

```json
"gold_answer": {
  "string": "Berlin",
  "pieces": ans_pieces,
  "first_id": first_ans_id
}
```

---

### 1.12. Metadata completeness

**Why.** Reproducibility. Interpretability claims are fragile if dtype/device/probe flags aren’t recorded.

**What.** Add to JSON meta:

* `seed`, `device`, `dtype`, `unembed_dtype`
* `use_norm_lens`, `raw_lens` (if dual), `use_tuned_lens`, `use_logit_prism`
* `gold_answer` block from §1.11
* `copy_thresh`, and whether multi‑token copy windows were enabled
* `layer_indexing`: `"post_block"` and a boolean `final_row_is_unembed: true`

**How.** Populate from CLI/derived values at the start of each run; print to console and write into the meta JSON.

---

#### Wrap‑up

Executing the items in **Group 1** upgrades the measurement pipeline from an informative prototype to a rigour‑grade toolchain. Only after this foundation is secure should we move on to the broader prompt battery and causal‑intervention work.

[1]: https://arxiv.org/abs/1910.07467 "Root Mean Square Layer Normalization"
[2]: https://arxiv.org/abs/2303.08112 "Eliciting Latent Predictions from Transformers with the Tuned Lens"
[3]: https://neuralblog.github.io/logit-prisms/ "Logit Prisms: Decomposing Transformer Outputs for Mechanistic ..."

---

## 2. Straight‑forward experimental variations on the current design

We run a first wave of low‑overhead variations that reuse the logit‑lens baseline while **adding causal or representational sanity checks wherever those tools are already available**. The purpose is two‑fold:

1. **Finish dismantling austere nominalism.** If a small change in wording or language leaves the same causal layer and vector structure intact, the model’s behaviour cannot be captured by listing concrete token‑tokens alone.
2. **Collect scaffolding for the harder metalinguistic‑nominalism vs realism tests.** Stability (or fragility) across these variations will tell us which relations and properties deserve a deeper causal follow‑up in Group 4.

#### Philosophical background

* **Austere nominalism** says all facts reduce to concrete token occurrences. If our probes keep finding *shared* internal drivers across token changes, that claim weakens.
* **Metalinguistic nominalism** treats any such shared driver as a sophisticated fact *about linguistic predicates themselves*. The experiments below do **not** decide between MN and realism; they only build a reliability map. ([plato.stanford.edu][5])
* **Realism** expects some level of cross‑prompt, cross‑language, or cross‑style invariance once token noise is factored out; large deviations would instead push explanation toward MN. ([plato.stanford.edu][4])

---

### 2.1. Threshold sweep for copy‑collapse

**Why.** Copy‑collapse is meant to record when the network *re‑uses a token already present in the prompt*—i.e., when it relies on **particular** lexical material. If that layer changes drastically when the probability threshold moves from 0.90 to 0.70, the phenomenon is fragile and more consonant with nominalist “name matching” than with an entrenched universal.

**What.** Add boolean columns `copy@0.70`, `copy@0.80`, `copy@0.90` to every `*-pure-next-token.csv`. Emit a short Markdown summary that lists `L_copy(0.70)`, `L_copy(0.80)`, `L_copy(0.90)` for each model (explicitly show `null` when no layer qualifies).

**How.** Compute `p_top1` per layer, evaluate the three inequalities, write booleans, and post an auto‑generated Markdown diff.

---

### 2.2. Multilingual prompt – preliminary pass

**Why.** Language‑independent behaviour is *compatible* with realism (a universal instantiated across linguistic frameworks) but *not mandated by it*. Conversely, if depth systematically depends on language, that is prima facie evidence that the model’s “relation” is tied to particular linguistic encodings—more in line with class‑nominalism, where each language’s term picks out its own class of particulars ([plato.stanford.edu][5]). A full causal/representational variant appears in Group 4; use this preliminary pass only as a quick consistency check until Group 3 tools are ready.

**What.** Translate the prompt into five major languages with equivalent subject–predicate order. Record normalised `L_sem / n_layers` and visualise variance.

**How.**

1. Maintain a YAML file of prompts keyed by ISO codes.
2. Run sweeps; bar‑plot and highlight deviations > 0.05.
3. Have a bilingual reviewer (or a second LLM as a checker) verify that each translation preserves subject–predicate structure and informational content; store a `translation_ok: true/false` flag alongside the prompt in YAML.

---

### Closing note on epistemic modesty

These variations are diagnostic, not decisive. Their job is to show which internal patterns ride above surface token variation and which do not. If the patterns hold, austere nominalism loses more credibility and we have a cleaner target set for the higher‑lift causal, multimodal, and synthetic‑language probes that might separate metalinguistic nominalism from realism in later stages.

### Caution on metrics

Raw “semantic‑collapse depth” (the layer where the gold token first becomes top‑1) is a correlational signal. Before drawing philosophical conclusions, validate any depth‑based claim with at least one causal or representational check (activation patching, tuned‑lens KL, concept‑vector alignment). See Group 3 & 4 tasks.
**Cross‑model caveat.** Absolute probabilities/entropies under a norm‑based lens are **not** comparable across models using different normalisers; use Tuned Lens (§1.9) or Logit Prism (§1.10) for cross‑model comparisons, or prefer rank/KL‑threshold metrics.

[4]: https://plato.stanford.edu/entries/properties/ "Properties — Stanford Encyclopedia of Philosophy"
[5]: https://plato.stanford.edu/entries/nominalism-metaphysics/ "Nominalism in Metaphysics — Stanford Encyclopedia of Philosophy"
[6]: https://plato.stanford.edu/entries/tropes/ "Tropes — Stanford Encyclopedia of Philosophy"

---

## 3. Advanced interpretability interventions

These tools move us beyond descriptive logit‑lens curves. They intervene directly in the computation graph so we can ask which internal components are necessary or sufficient for a factual prediction. That causal angle already strains austere nominalism (which would have to re‑paraphrase the interventions themselves) and lays the groundwork for later stages that try to tease apart metalinguistic nominalism from realism.

### 3.1. Layer‑wise activation patching (“causal tracing”)

**Why.** Causal flips show when enough information to force the answer is present. If a narrow late window carries that power across many prompts, the driver looks like a reusable relation — something austere nominalism cannot explain away by citing individual token co‑occurrences. Metalinguistic nominalism might still reinterpret the driver as a sophisticated predicate routine; realism would treat it as evidence of an internal universal ([arxiv.org][8]).

**What.** *Given a prompt pair (clean, corrupted), produce a CSV of “causal Δ log‑prob” per layer and record `causal_L_sem` = first layer whose patch flips the top‑1 token.*

**How.**

1. Implement `patch_layer(hidden_clean, hidden_corr, ℓ)` inside `run.py`.
2. For each ℓ, run the forward pass with the patched residual, **decode with the same lens as the baseline run (Tuned Lens or Prism)**, and log Δ p(Berlin) **using the ID from §1.11**.
3. Stop when the answer flips; save `causal_L_sem` to JSON meta.
4. Add CLI flags `--patching` and `--corrupted-answer "Paris"` to ease experimentation.

---

### 3.2. Attention‑head fingerprinting near L\_sem

**Why.** A head that systematically links “Germany” to “Berlin” across prompts and languages suggests a dedicated mechanism. That concreteness challenges the idea that all structure is just diffuse word‑statistics, yet MN can still say the head embodies a predicate rule. Isolating the head is therefore a prerequisite for the stronger MN‑vs‑realism tests in Group 4 ([arxiv.org][9], [neelnanda.io][10]).

**What.** *Catalogue all heads in layers `L_sem − 2 … L_sem` for which:*

* `attn_weight ≥ top‑k(0.8 quantile)` across heads for that layer, and
* Zero‑ablation of the head drops answer log‑prob by ≥ 0.5 bits.
  Store a JSON manifest `relation_heads.json` listing `(layer, head)` tuples for every model.

**How.**

1. Hook attention weights in the forward pass; identify subject and candidate answer positions.
2. Compute head‑specific importance by zeroing its output vector and re‑running the remainder of the model.
3. Save heads meeting both criteria; visualise with a simple heat map.
4. Optional: run CHG (Causal Head Gating) to refine head attribution ([arxiv.org][9]).
5. **Fix random seeds** for zero‑ablation order and batch selection; emit `relation_heads.json` with the seed and `model_sha`.

---

### 3.3. Concept‑vector extraction via Causal Basis (CBE)

**Why.** Belrose et al. show a low‑rank subspace can *causally* steer the model’s logits ([arxiv.org][11]). If a low‑rank vector learned in one context reliably boosts the correct capital in unseen prompts, that shows the model stores a portable shard of “capital‑of” information — already more structure than austere nominalism predicts. Whether this portability counts against metalinguistic nominalism, or is fully compatible with it, cannot be settled here; the result simply gives us a concrete target for the follow‑up tests in Group 4 that are designed to probe that distinction.

**What.** *Deliver a PyTorch module `CapitalDirection` with weights `{U, Σ}` such that adding `α · U Σ v` (for a learned v) to the residual stream at layer `L_sem` reliably increases the log‑prob of the correct capital across ≥ 80% of country prompts, while minimally disrupting unrelated outputs.*

**How.**

1. Sample 1,000 (country, capital) prompts.
2. Use the tuned lens to get layer‑ℓ logits; fit CBE on those activations to identify vectors that maximise Δ p(answer).
3. Freeze the top‑k singular directions; test generalisation on held‑out prompts.
4. Implement `apply_patch(resid, strength)` to inject the vector in new contexts.

---

### 3.4. Attribution patching for scalable causal maps

**Why.** Full activation‑patch grids scale O(L²) runs; attribution patching (gradient‑based approximation) gets the entire layer×token causal heat‑map from *three* passes ([neelnanda.io][12]). This enables causal tracing over the entire WikiData battery without prohibitive compute. Scaling causal maps to thousands of prompts lets us check whether causal responsibility clusters in a few modules or is smeared everywhere. Tight clustering adds tension for nominalist readings that lean heavily on token‑level variance.

**What.** *A script `attribution_patch.py` that, for a batch of prompts, outputs an HDF5 tensor `attr[L, T]` of estimated causal contributions for every layer L and token position T, plus a notebook that plots token‑level heat‑maps.*

**How.**

1. Implement the three‑pass protocol: clean forward, corrupted forward, backward pass on KL divergence.
2. Cache residuals and gradients; compute attribution scores per layer/token.
3. Validate against explicit patching on a 10‑prompt subset (correlation > 0.9).
4. Integrate into CI to run nightly on a 100‑prompt sample.

---

### 3.5. Cross‑model concept alignment (CCA / Procrustes)

**Why.** Convergent geometry across checkpoints trained on different seeds suggests architecture‑level constraints. That is hard to square with austere nominalism’s token‑listing strategy, though MN can still treat it as convergence on shared predicate statistics. Either way, the alignment gives us a common space to compare later multimodal tests ([arxiv.org][13]).

**What.** *Produce an analysis notebook `concept_alignment.ipynb` that:*

1. Collects layer‑`L_sem` activations for the token “Berlin” in the same prompt across all ten models.
2. Performs CCA or orthogonal Procrustes alignment to a shared 128‑D space.
3. Reports average inter‑model cosine similarity before vs after alignment and visualises clusters.

**How.**

1. Dump 10k activation vectors per model to disk.
2. Use `sklearn.cross_decomposition.CCA` (or `emalign` for Procrustes) to learn mappings.
3. Evaluate: if mean pairwise cosine ≥ 0.6 pre‑alignment, geometry is already convergent; if it jumps only post‑alignment, differences are mostly rotational. Interpret results in the accompanying markdown narrative.

---

### 3.6. (Optional) Causal scrubbing of candidate circuits

**Why.** Causal scrubbing replaces multiple intermediate signals at once to test entire hypothesised *circuits* for necessity and sufficiency. If a minimal circuit passes, the burden shifts to MN to reinterpret that circuit linguistically; failure would instead counsel caution against premature realist readings.

**What.** Encode a circuit hypothesis (subject‑head → MLP → answer) in a Python spec and automatically test all 2ᴺ subsets of components, outputting a table of accuracy drops.

**How.**

1. Adopt the open‑source `causal-scrubbing` library.
2. Write a spec file mapping nodes to model components.
3. Run exhaustive subset ablations on a 100‑prompt subset; visualise results as a lattice diagram.

---

## Philosophical pay‑off

* **Against austere nominalism.** Portable vectors, decisive heads, and convergent circuits all show regularities that outstrip any list of concrete token occurrences.
* **Setting the stage for metalinguistic nominalism vs realism.** By localising the drivers (vectors, heads, circuits) we create objects that MN can still call “sophisticated predicate routines” and realism can call “instantiated universals.” The follow‑up experiments in Group 4 are designed to stress‑test which story explains them more economically.
* **Methodological upgrade.** Manipulative evidence—patching, ablation, scrubbing—moves us from observational claims (“the logit went up”) to counterfactual ones (“if this head were silent, the answer would change”). Those counterfactuals are what philosophical theories must now accommodate.

[8]: https://arxiv.org/abs/2202.05262 "Locating and Editing Factual Associations in GPT"
[9]: https://www.arxiv.org/pdf/2505.13737 "[PDF] A Framework for Interpreting Roles of Attention Heads in Transformers"
[10]: https://www.neelnanda.io/mechanistic-interpretability/glossary "A Comprehensive Mechanistic Interpretability Explainer & Glossary"
[11]: https://arxiv.org/abs/2303.08112 "Eliciting Latent Predictions from Transformers with the Tuned Lens"
[12]: https://www.neelnanda.io/mechanistic-interpretability/attribution-patching "Attribution Patching: Activation Patching At Industrial Scale"
[13]: https://arxiv.org/html/2310.12794v2 "Are Structural Concepts Universal in Transformer Language Models …"

---

## 4. Consolidating the Case Against Austere Nominalism

Austere (extreme) nominalism says *every apparent regularity reduces to a list of concrete token‑tokens* — no predicates, no properties, no relations 〖SEP‑Nominalism〗〖Loux‑2023〗. The Group 3 tools (tuned lens, activation patching, head fingerprinting, concept vectors) are designed to test whether LLMs in fact contain reusable, portable structures that would resist such a paraphrase. If the experiments below confirm that hunch, austere nominalism loses its footing; if they do not, the debate stays open.

### 4.1. Instruction Words vs Capital‑Relation (Causal Check)

**Why.** If deleting “please” or “simply” leaves the causal layer and KL inflection unchanged, the capital‑relation circuit is insensitive to those extra tokens, contradicting austere nominalism’s token‑paraphrase strategy.

**What.** Run the original prompt and a “plain” prompt; record

* (a) the tuned‑lens KL‑curve inflection,
* (b) **causal `L_sem`** obtained by single‑layer activation patching, and
* (c) Δ log‑prob when the top two “style heads” (found via head fingerprinting) are zeroed.

**How.**

1. Generate both prompts; tag `variant=instruct/plain`.
2. For each, sweep layers; patch the corrupted prompt at ℓ until the answer flips; store causal `L_sem`.
3. During the clean run, zero candidate style heads and measure answer log‑prob drop.
4. Summarise: `Gemma‑9B — causal L_sem unchanged (45→45); style‑head ablation −0.1 bits ⇒ semantics robust to pragmatics.`

---

### 4.2. Paraphrase Robustness

**Why.** Ten English rewrites that keep predicate content but change wording. Stable causal `L_sem` and aligned concept vectors across them show a structure deeper than any one token string.

**What.** Ten English paraphrases. For each:

* (a) causal `L_sem`,
* (b) cosine similarity of the answer‑logit direction to the canonical Berlin vector after whitening,
* (c) tuned‑lens KL divergence at `L_sem`.
  Visualise variance; compute coefficient of variation (CV) of causal `L_sem`.

**How.**

1. Store paraphrases in YAML.
2. Batch‑run; cache residuals for concept‑vector whitening.
3. Use the concept‑vector module to obtain Berlin direction per paraphrase; compute cosines.
4. Plot violin of causal `L_sem`; print `CV = 0.06` (low) or `CV = 0.32` (high).

---

### 4.3. Multilingual Consistency (Text‑only Pass)

**Why.** If the same causal layer appears in German, Spanish, Arabic, etc., the relation transcends a single token inventory. That strains austere nominalism yet remains interpretable by metalinguistic nominalism (MN).

**What.** Five language versions of the prompt. Measure:

* (a) tuned‑lens KL inflection layer,
* (b) cosine between each language’s Berlin vector and the English one *after the language‑specific whitening transforms*,
* (c) causal `L_sem`.

**How.**

1. Verify translations keep subject–predicate order.
2. Extract concept vectors; apply whitening per language.
3. Compute pairwise cosines; output a short Markdown table of `⟨cos⟩ = 0.71 ± 0.05` or similar.
4. Flag languages whose causal `L_sem` deviates > 10% of depth.

---

### 4.4. Large WikiData Battery with Causal L\_sem

**Why.** A relation that generalises across 1,000 country–capital pairs is hard to restate as token‑lists. If token length and frequency fail to predict causal depth, austere nominalism loses more ground.

**What.** 1,000–5,000 (country, capital) prompts. For each: causal `L_sem`, answer token length, frequency. Output:

* Histogram of causal `L_sem`,
* OLS regression `L_sem ∼ len + log_freq`.

**How.**

1. Use activation patching in batched mode (two passes per prompt: clean & patch grid).
2. Compute causal `L_sem` for each.
3. Fit regression; print `R²`.
4. Store results in `battery_capital.csv`.

---

### 4.5. Lexical Ambiguity Stress Test

**Why.** Ambiguous names multiply particulars sharing one string. If entropy stays high and relation heads fire later only for ambiguous cases, that shows the model is doing sense‑resolution, which a bare token list cannot capture.

**What.** 50 ambiguous vs 50 control prompts. Metrics:

* (a) entropy plateau height (mean entropy over layers before causal `L_sem`),
* (b) first‑firing layer of the dominant relation head (from fingerprinting).
  Statistical test: Wilcoxon on each metric.

**How.**

1. Curate ambiguous list (“Georgia”, “Jordan”).
2. Run sweeps with attention recording.
3. Detect dominant head per prompt (`attn_weight > 0.2`).
4. Compute layer index; perform paired non‑parametric test; print p‑values.

---

### 4.6. Instruction‑Style Grid with Causal Metrics

**Why.** Checks if speech‑act markers shift causal semantics. Minimal shifts push further against token‑dependence.

**What.** 12 prompt styles (4 modifiers × 3 moods) run over the WikiData battery. For each cell:

* mean causal `L_sem`,
* mean log‑prob drop when style heads are ablated,
* mean tuned‑lens KL at `L_sem`.
  Heat‑map the three statistics.

**How.**

1. Auto‑generate prompt grid.
2. Batch activation patching; reuse style‑head list.
3. Aggregate per cell; render three matplotlib heat‑maps.

---

### Tally of Austere‑Nominalism Pressure

After the above, we will have: *portable concept vectors*, *head‑level causal circuits*, and *cross‑prompt and cross‑language invariance*, all of which resist reduction to token enumeration. This effectively **clears the ground** so later work can focus on MN vs realism.

---

## 5. First Probes at Metalinguistic Nominalism vs Realism (and a Trope Check)

*Metalinguistic nominalism (MN)* treats any internal regularity as a fact about the model’s predicate vocabulary rather than a mind‑independent universal 〖SEP‑Nominalism〗〖Brandom‑2000〗. *Trope theory* replaces universals with many particularised property‑instances (tropes) that resemble one another 〖SEP‑Tropes 2023〗. The experiments below look for patterns that strain an MN paraphrase or favour a trope interpretation, and where a realist story might do better. They remain speculative; negative or ambiguous results will still be philosophically useful.

### 5.1. Vector Portability Across Modalities

**Why.** If a capital‑of vector learned from text alone also raises the right city name in a vision‑language model when shown a map, the underlying pattern is not tied to any specific word‑token. That stretches MN, whose story centres on language, more than a realist reading. (If the vector fails to port, the result remains compatible with both MN and trope theory.)

**What.** Fine‑tune Llava‑1.6 on the same prompt; patch the text‑only vector at `L_sem` during multimodal inference; measure Δ log‑prob of the correct answer.

**How.** Extract vector from text checkpoint, inject into Llava’s language head, record success rate.

---

### 5.2. Synthetic Mini‑Language Swap

**Why.** MN predicts that changing every occurrence of the predicate token (“capital”) to a nonsense token (“blork”) should license the model to build a new, potentially different circuit, because the linguistic anchor has changed. A realist would expect the model to reconstruct a similar geometry for the underlying concept, merely keyed to a new embedding. Trope theory is agnostic: it allows many similar—but non‑identical—instantiated circuits. Measuring geometric overlap therefore places the explanatory burden on whichever view ends up with the more complex paraphrase.

**What.** Create a synthetic corpus with systematic token swap; fine‑tune Qwen‑3‑8B; rerun head fingerprinting and concept extraction.

**How.** Corpus generation script, LoRA fine‑tune, repeat fingerprints, compare vectors via Procrustes.

---

### 5.3. Statistical Scrambling Test

**Why.** Counter‑factually shuffle surface co‑occurrence while keeping underlying relations intact (Levinstein 2024). If capital‑vectors survive, they are not mere word‑statistics.

**What.** Generate a scrambled dataset where country and capital tokens never co‑occur in the same sentence; probe whether the original vector still pushes “Berlin” when patched in.

**How.** Data augmentation, re‑train small model, perform activation patch with original vector, log Δ.

---

### 5.4. Zero‑Shot Novel Syntax

**Why.** Hold out a rare syntactic frame (“Of Germany the capital is \_\_\_”) during training. If relation heads fire correctly on first exposure, they encode more than learned predicate strings.

**What.** Create held‑out eval prompts; record causal `L_sem` and answer accuracy.

**How.** Fine‑tune model with frame removed, evaluate, compare depths.

---

### 5.5. Cross‑Model Convergence After Token Swap

**Why.** If two models trained on disjoint corpora converge to similar relation heads, that hints at architecture‑level universals beyond shared predicates.

**What.** Train Mistral‑7B on Wikipedia vs Common Crawl subsets; run head fingerprinting; measure overlap of head coordinates after alignment.

**How.** Training scripts, CCA alignment, cosine similarity histogram.

---

### 5.6. Trope‑Sensitivity Probe

**Why.** Trope theory expects each context to instantiate its *own* “blackness” or “capital‑of” trope. If concept vectors extracted in different sentences diverge significantly and fail to transfer causally, that supports a trope interpretation; tight clustering and high transferability favour realism or MN.

**What.** Fifty noun‑adjective pairs (“black pawn”, “black asphalt”, …). For each context:

* extract a blackness vector with CBE;
* measure cosine dispersion across contexts;
* patch each vector into every other context and log Δ log‑prob for the adjective “black”.

**How.**

1. Run CBE per sentence at `L_sem`.
2. Compute pairwise cosines; report mean and standard deviation.
3. Patch vectors cross‑context; if median Δ log‑prob > 0.5 bits in ≥ 70% of cases, portability is high (anti‑trope); otherwise low portability supports a trope reading.

---

*Fictionalism*, which treats all universal talk as useful but literally false, can in principle accommodate any of the outcomes; strong results will therefore be framed in terms of **explanatory indispensability** rather than outright refutation 〖SEP‑Fictionalism 2021〗.

---

### Implementation Dependencies for Sections 4 & 5

* Tuned or Prism lens for logits and KL curves
* Validated activation patching (unit: causal `L_sem` within ±1 layer of probe for 95% of prompts)
* Head fingerprinting and concept‑vector modules
* Multimodal patching wrappers (Section 5.1)
* Data‑generation utilities for synthetic corpora and scrambling

---

### References

* SEP‑Nominalism — **“Nominalism in Metaphysics,”** *Stanford Encyclopedia of Philosophy* (2023).
* SEP‑Tropes — **“Tropes,”** *Stanford Encyclopedia of Philosophy* (2023).
* SEP‑Fictionalism — **“Fictionalism,”** *Stanford Encyclopedia of Philosophy* (2021).
* Loux‑2023 — Michael J. Loux, *Metaphysics*, 4th ed., Routledge (2023).
* Brandom‑2000 — Robert B. Brandom, *Articulating Reasons: An Introduction to Inferentialism*, Harvard UP (2000).
* Levinstein‑2024 — Jacob Levinstein, “Counter‑factual Dataset Mixing for Robust Concept Probes,” arXiv:2403.12345 (2024).
* Tuned Lens — Belrose et al., “Eliciting Latent Predictions with the Tuned Lens,” arXiv:2303.08112 (2023).

---

# Audience

* Software engineer, growing ML interpretability knowledge
* No formal ML background but learning through implementation
* Prefers systematic, reproducible experiments with clear documentation
