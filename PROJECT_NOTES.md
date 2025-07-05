# Interpretability Project - Development Notes for AI assistant

# Next steps


---

## 1 .  Get the measurement right – eliminate numerical noise first

* **Correct the RMS/LN γ scaling path** (≈ 15 min).
  Read the *post‑block* scaling parameter, or skip γ altogether and let the lens absorb it; prevents systematic norm error in early layers.

* **Sub‑word‑aware copy‑collapse detector** (≈ 15 min).
  Detokenise the top‑1 string before the prompt‑membership check so that “▁Ber lin” is caught; removes a known false negative that could masquerade as “no copy reflex”.

* **Add top‑1 p, cumulative top‑5 p, and KL‑to‑final to the CSV** (≈ 30 min).
  These curves separate “single winner” convergence from “cluster convergence”, clarifying whether a universal is token‑like or type‑like.

* **Raw‑activation lens toggle** (≈ 30 min).
  Run once with the *norm lens removed*; if semantics shift greatly you know the lens, not the model, created the effect.

* **Representation‑drift cosine curve** (≈ 30 min).
  Checking ‖Berlin⃗ \_ℓ · Berlin⃗ \_final‖ shows whether the concept direction already exists and only needs strengthening (realism‑friendly) or rotates into place late (nominalism‑friendly).

* **Negative‑control prompt row (“The capital of France is …”)** (≈ 30 min).
  Ensures Berlin never outranks Paris when the prompt demands Paris; guards against surface‑string leakage that would fake an early “universal”.

* **Ablate stylistic filler (`simply`)** (≈ 45 min).
  If Gemma’s copy collapse disappears, you have evidence that shallow style tokens, not semantic content, triggered the reflex.

* **Lightweight CI / regression harness** (≈ 1–2 h).
  Runs `--dry‑run` on CPU for every commit so that future refactors cannot silently skew metrics.

* **Integrate a Tuned Lens** (≈ 4–6 h, one evening).
  `pip install tuned‑lens`, train affine probes on \~50 k tokens, and swap `lens(hidden)` for the raw `unembed`; fixes both scaling‑mismatch and basis‑rotation problems and yields an order‑of‑magnitude drop in KL to the model’s own answers ([arxiv.org][1], [arxiv.org][2]).

* **Optional: Logit Prism (shared whitening + rotation)** (≈ 1 day).
  Gives a *single* decoder that works for all layers, simplifying cross‑model geometry searches ([neuralblog.github.io][3]).

These steps make sure later philosophical claims rest on solid numerical ground rather than artefacts of the probing setup.

---

## 2 .  Straight‑forward experimental variations on the current design

* **Scan the copy‑collapse probability threshold** (≈ 10 min).
  Reveals whether “copy reflex” is a continuum rather than a binary event.

* **Prompt without command words** (≈ 30 min).
  Checks if instruction style, not factual content, sets collapse depth.

* **Small paraphrase set (5–10 re‑wordings, same language)** (≈ 1 h).
  Verifies whether L\_sem shifts with purely syntactic variation; nominalism would predict larger variance.

* **Multilingual versions of the same fact** (≈ 2 h, needs translations).
  If the collapse depth stays constant across languages, that points toward language‑independent universals.

* **Prompt battery from WikiData (≈ 1 day to script + run).**
  Auto‑generate 1 000 (subject, relation, object) triples; lets you correlate Δ‑collapse with ambiguity of the object, number of syllables, etc.

* **Lexical‑ambiguity stress test** (≈ 1 day).
  Include homographs and near‑synonyms (“Georgia” state vs country); tracks whether the model delays semantics when meaning is under‑determined.

* **Instruction‑style ablation grid** (≈ 1 day).
  Systematically vary terseness/politeness modifiers to see if surface instruction cues shift copy or semantic collapse.

These variations broaden the empirical base: do the observed layer depths generalise across wording, language, and factual domain?  Those answers directly inform whether LLM “universals” are robust entities (realism) or thin regularities (nominalism).

---

## 3 .  Advanced interpretability interventions

* **Layer‑wise activation patching / causal tracing** (≈ 0.5–1 day).
  Replace hidden state ℓ from a corrupted prompt (“Germany → Paris”) into the clean run and find the earliest ℓ that flips the answer; gives *causal* L\_sem immune to probe errors ([arxiv.org][4]).

* **Attention‑head fingerprinting near L\_sem** (≈ 1–2 days).
  Hook queries/keys and identify heads specialising in “Germany ↔ Berlin”; the presence of a dedicated relation head would support a realist “relation as internal module”.

* **Concept‑vector extraction with causal basis (CBE)** (≈ 2–3 days).
  Use the method in Appendix C of the tuned‑lens paper to isolate a low‑rank Berlin sub‑space; transplant it into unrelated contexts and measure portability ([arxiv.org][2]).

* **Cross‑model concept alignment (CCA / Procrustes)** (≈ 3 days).
  Align Berlin vectors from all seven checkpoints; convergent geometry across independent models would be striking evidence for realism.

* **Attribution‑patching for scale‑out causal maps** (≈ 3 days).
  Caches three runs (clean, corrupted, grads) and reconstructs the full patching heat‑map in one shot ([neelnanda.io][5]); allows large prompt batteries without cubic run‑time.

These advanced methods move beyond correlation to intervention and cross‑model generality—the strongest kinds of evidence we can marshal.  If a single low‑rank direction, or the same attention head pattern, *causally controls* “capital‑of” across prompts and models, that is the kind of structural stability realists expect of universals.  Conversely, if every model solves each prompt with idiosyncratic heads or widely different sub‑spaces, the case for nominalism strengthens.

---

### Why this structure matters

1. **First, make the measurement faithful.**  Without tuned/prism lenses and basic sanity checks we risk mistaking probe artefacts for philosophical insight.
2. **Second, ask whether the phenomena replicate once wording, language, and domain vary.** Robust cross‑prompt behaviour is a necessary condition for treating an internal feature as a “universal”.
3. **Finally, perform causal and geometric probes that can reveal *re‑usable* internal structures.** Those are the decisive pieces that can either anchor a realist interpretation or undercut it in favour of nominalism.

Executing the first group upgrades rigour; the second tests generality; the third can generate genuinely new, philosophically relevant evidence.

[1]: https://arxiv.org/abs/2303.08112?utm_source=chatgpt.com "Eliciting Latent Predictions from Transformers with the Tuned Lens"
[2]: https://arxiv.org/pdf/2303.08112?utm_source=chatgpt.com "[PDF] Eliciting Latent Predictions from Transformers with the Tuned Lens"
[3]: https://neuralblog.github.io/logit-prisms/?utm_source=chatgpt.com "Logit Prisms: Decomposing Transformer Outputs for Mechanistic ..."
[4]: https://arxiv.org/abs/2202.05262?utm_source=chatgpt.com "Locating and Editing Factual Associations in GPT"
[5]: https://www.neelnanda.io/mechanistic-interpretability/attribution-patching?utm_source=chatgpt.com "Attribution Patching: Activation Patching At Industrial Scale"


# Experiment Notes
Detailed technical notes for experiment `001_layers_and_logits` have been moved to `001_layers_and_logits/NOTES.md`.

# Philosophical Project Context
- **Goal**: Use interpretability to inform nominalism vs realism debate
- **Current evidence**: Layer-relative perspective (early = nominalist templates, late = realist concepts)

# User Context
- Software engineer, growing ML interpretability knowledge
- No formal ML background but learning through implementation
- Prefers systematic, reproducible experiments with clear documentation

