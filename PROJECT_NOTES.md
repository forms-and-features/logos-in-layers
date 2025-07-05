# Interpretability Project - Development Notes for AI assistant

# Next steps

### ■ Quick wins

* **Sub‑word aware copy‑collapse.**
  Replace the current string‑level check with a detokenised comparison or a lower probability threshold.  Implementation: edit the `copy_collapse` block of `run.py` to join contiguous BPE tokens before membership testing.  *Philosophical leverage*: small, but removes a known artefact that could mislead later analyses.([raw.githubusercontent.com][1])

* **Add top‑1 p, top‑5 cumulative p and KL‑to‑final‑layer to the CSV.**
  You already compute logits per layer; add three floating‑point columns and write them out beside `entropy_bits`.  Two extra `torch.topk` calls and one line for `F.kl_div`.  *Leverage*: moderate – lets you see whether entropy drops because a *single* answer wins or a cluster of near‑synonyms emerges.  This distinction matters when arguing whether a model uses a “type” vs. “token” representation.

* **Representation‑drift curve.**
  For each layer compute `cos_sim(û_ℓ, û_final)` where `û` is the Berlin logit direction.  Add a small routine after the forward sweep.  *Leverage*: moderate; if early layers already point in roughly the right direction, realism has a stronger foothold.

* **Negative‑control prompts (“France → Berlin?”).**
  Duplicate the run with a swapped subject/object.  If Berlin ever outranks Paris, log a warning.  Ten‑minute change; high reviewer confidence dividend.

* **Ablate instruction fluff (`simply`).**
  Run the same prompt without stylistic fillers and record any shift in collapse depth, especially for Gemma.  Half‑hour tweak; might falsify the “copy reflex” hypothesis.

---

### ■ One‑evening tasks

* **Drop‑in Tuned Lens or Logit Prism.**
  `pip install tuned‑lens`, freeze the backbone, train affine probes on ≈50 k tokens.  Replace the raw un‑embedding call with `lens(hidden_states)`.  Solves the RMS‑mismatch (γ applied to the wrong residual) and basis‑rotation issues highlighted by reviewers.([raw.githubusercontent.com][1], [arxiv.org][2])
  *Leverage*: high – it de‑noises all other metrics and makes cross‑model comparisons fair.

* **Raw‑activation lens sanity check.**
  Flip `USE_NORM_LENS=False` and rerun once.  If semantics suddenly “appear” earlier the current conclusions may be lens‑artefacts.  Implementation is a flag; plotting the delta is trivial.  *Leverage*: high for epistemic rigor.

* **Top‑k trajectory plots.**
  Produce one matplotlib figure per model with curves for entropy, top‑1 p, KL, cosine drift.  The user‑visible code can live in a Jupyter cell.  *Leverage*: moderate; plots make emergent patterns obvious to outside reviewers.

---

### ■ Weekend projects

* **Activation patching / causal tracing pass.**
  Cache clean activations, corrupt the prompt (“Germany → Paris”), replay with layer‑wise patches.  Report the earliest ℓ where the answer flips (`causal_L_sem`).  Libraries such as `activation‑patching` or `TransformerLens` already implement the boiler‑plate.  *Leverage*: very high – gives *causal* rather than correlational evidence about where meaning is fixed.  Key for the realism argument: a universal that can be transplanted layer‑by‑layer is closer to a “real” entity.

* **Prompt battery from WikiData.**
  Auto‑generate 1 000 (subject, relation, object) triples in the same “Give the city name only …” template plus multilingual variants.  Store metrics per triple and study the distribution of `Δ-collapse`.  *Leverage*: high – lets you test whether semantic depth tracks ambiguity or linguistic form, informing the nominalism thesis.

* **Attention‑head fingerprinting near L\_sem.**
  Capture attention patterns at layers ℓ ∈ {L\_sem − 2 … L\_sem}.  Search for heads whose query ⊗ key scores align “Germany ↔ Berlin”.  The `HookedTransformer` tracing API already exists.  *Leverage*: moderate‑to‑high – identifying a dedicated relation head would support a realist reading (a reusable “capital‑of” relation).

---

### ■ Deep dives

* **Concept vector extraction and portability test.**
  Use the causal basis method (Belrose 2023 App. C) to isolate a low‑rank subspace that pushes probability mass toward Berlin.  Patch that vector into unrelated prompts (“The capital of Poland is …”) and measure side effects.  If the vector generalises, that’s evidence for a *portable* universal.  Engineering: train the basis on \~20 prompts, write a small `torch.nn` module that adds the vector at configurable strength.  *Leverage*: very high – directly tests whether the model stores abstract properties independently of token strings.

* **Cross‑model concept CCA.**
  Collect per‑layer Berlin activations from all seven models, align with CCA or Procrustes, and test whether a shared sub‑space emerges.  Requires dumping large tensors; recommend subsampling 5 k token contexts to keep RAM manageable.  *Leverage*: high – convergent geometry across independent training runs would be compelling evidence for realism.

* **No‑rotation tuned lens (Logit Prism).**
  Train a shared whitening + rotation that decodes *all* layers with a common probe (Nguyen 2024).  Gives a single latent basis, reducing degrees of freedom and making “universal vectors” easier to compare across depth and models.  Time mainly in GPU hours; implementation very similar to tuned lens.

---

### ■ Meta / organisational steps

* **Philosophical annotation notebook.**
  After each empirical run, append a short section: *Interpretation wrt universals.*  Forces clarity about how each metric bears on the realism vs nominalism debate and creates a living document philosophers can engage with.  *Leverage*: moderate; cost is writing time, not compute.

* **Automated regression harness.**
  Wire each run into a lightweight CI (e.g. GitHub Actions with `--dry-run` on CPU) to ensure metrics don’t silently shift after refactors.  Protects against accidental inconsistencies when you integrate tuned lens or new probes.

---

## How this roadmap advances the debate

* **Measurement fixes** (tuned/prism lens, drift curves, better copy detection) ensure you are not arguing metaphysics from numerical noise.
* **Causal interventions** (activation patching, concept vectors) probe *what the network needs* for the answer, not just what correlates with it – crucial for claims about “real” internal structure.
* **Breadth tests** (prompt battery, multilingual variants) address the nominalist worry that any one prompt over‑fits to surface form.
* **Cross‑model geometry** directly tackles a key realist intuition: universals should be stable across instances.

Executing the “quick wins” and “one‑evening tasks” will already raise the interpretability rigour to current best practice; the “weekend” projects put causal teeth into the analysis; and the “deep dives” can generate genuinely novel evidence relevant to the longstanding philosophical dispute.

[1]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/main/001_layers_and_logits/run.py "raw.githubusercontent.com"
[2]: https://arxiv.org/abs/2303.08112?utm_source=chatgpt.com "Eliciting Latent Predictions from Transformers with the Tuned Lens"


# Experiment Notes
Detailed technical notes for experiment `001_layers_and_logits` have been moved to `001_layers_and_logits/NOTES.md`.

# Philosophical Project Context
- **Goal**: Use interpretability to inform nominalism vs realism debate
- **Current evidence**: Layer-relative perspective (early = nominalist templates, late = realist concepts)

# User Context
- Software engineer, growing ML interpretability knowledge
- No formal ML background but learning through implementation
- Prefers systematic, reproducible experiments with clear documentation

