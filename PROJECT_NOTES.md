# Interpretability Project - Development Notes for AI assistant

## Next steps

### Track the shape of the logit distribution, not just entropy.

Entropy-in-bits alone can hide whether probability mass is flowing into a single answer token or into a small cluster of semantically-related tokens. Many groups now plot:
- top-1 prob,
- cumulative top-5 prob,

### KL divergence to the final-layer distribution (the “convergence curve”, § 4 in Belrose et al. 2023, arXiv 2303.08112).
Add two extra columns to the pure-next-token CSV (top-1 p, KL) and comment on where each curve bends.

### Add a causal-tracing sanity pass.

With the same residuals already cached you can do “activation patching” (Meng et al. 2022, 2205.14135): replace layer ℓ’s residual with that from a corrupted prompt (“The capital of Germany is Paris”) and measure how soon the output flips. Report the earliest causal L_semantic next to the diagnostic L_semantic. Reviewers can then note mismatches (often ±1-2 layers).

### Check representation drift across depth.
Logit-lens probes occasionally mis-report “early meaning” because the token’s direction already exists but then rotates away. Computing the cosine similarity between the un-embedded Berlin logit vector at layer ℓ and at the final layer reveals whether the concept is stabilising or drifting. A simple line-plot inserted in Section 3 lets reviewers call out unstable models.

### Probe negative controls.

Include one extra row in the test-prompt set that mentions France or Paris. If the answer token Berlin ever outranks Paris in that control, the reviewer should flag “semantic leakage”. This is a lightweight check for over-reliance on string cues.

### Switch to a Tuned‑Lens baseline

Train affine probes on the frozen weights (10‑min per model) to remove norm‑lens bias and to quantify representation drift (Belrose et al. 2023).

### Extend the prompt battery

Pull 1 000 – 5 000 (subject, relation, object) triples from WikiData; include synonyms and homographs to see if collapse depth tracks lexical ambiguity.

### Causal tracing

Use activation patching: overwrite the residual stream at layer ℓ with that from a corrupted prompt (“Paris is…”) and measure the KL divergence at the output. The earliest layer where patching flips the answer gives a causal L_sem that is immune to rank‑1 accidents.

### Concept geometry tests

Apply CCA or probing heads across models to check if the latent Berlin vectors align in a shared sub‑space. Strong cross‑model alignment would bolster a realist account.

### Ablate filler tokens

Drop the word simply and examine whether Gemma still collapses at L 42. If not, this supports the hypothesis that copy‑reflex hinges on the “instruction‑style” flanker.

### Philosophical leverage

Formulate a concrete argument map: If concepts = clusters that (i) are decodable across prompts, (ii) survive translation between tokenisations, and (iii) causally drive correct behaviour, then evidence (ii) & (iii) would undermine nominalism. The next experimental cycle should focus on (ii) and (iii).

## Experiment Notes
Detailed technical notes for experiment `001_layers_and_logits` have been moved to `001_layers_and_logits/NOTES.md`.

## Philosophical Project Context
- **Goal**: Use interpretability to inform nominalism vs realism debate
- **Current evidence**: Layer-relative perspective (early = nominalist templates, late = realist concepts)

## User Context
- Software engineer, growing ML interpretability knowledge
- No formal ML background but learning through implementation
- Prefers systematic, reproducible experiments with clear documentation

