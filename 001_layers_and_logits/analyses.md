# Layer-by-Layer Probe Analysis of 4 Base Models

*Experiment context*: Using the LayerNorm-lens probe implemented in `run.py`, we asked each model a short factual question:

```text
Question: What is the capital of Germany? Answer:
```

For every layer we un-embedded the (optionally normalised) residual stream at the last token position and printed the top 20 next-token predictions together with entropy.  Below is an interpretability-oriented write-up of the resulting traces.

---

## 1. Per-Model Findings

### 1.1 google/gemma-2-9b  (42 layers)

**When does the answer crystallise?**  
`Berlin` first appears at *layer 33* and becomes >99 % by layer 40.  This is ≈ 80 % through the network, slightly later (depth-wise) than Meta-Llama-3 and Mistral.

**Early-layer behaviour**
* Layers 0-7 are dominated by the punctuation token `:` with *zero entropy*—the model is extremely certain it should keep emitting a colon.  This indicates a strong positional / n-gram prior that "`Answer:`" is usually followed by ":"", a spurious correlation that the LayerNorm lens exposes very clearly.
* Entropy stays *artificially low* (≤ 0.06) up to layer 8, then rises sharply once semantic tokens (" answer", "Answer", etc.) compete, before falling again once `Berlin` takes over.

**Middle layers (9-25)**
* Transition sequence: `:` → " answer"/"Answer" → articles (`the`, `The`, `a`) → finally a *city* distribution.
* Noticeable stepwise entropy drops coincide with the head that predicts `Berlin` becoming active.

**Anomalies / red flags**
* The over-confidence on punctuation (entropy 0) is unusual and may signal that the embedding-norm is very small for `:` relative to other tokens, causing the softmax to collapse when projected directly from early residual streams.

### 1.2 meta-llama/Meta-Llama-3-8B  (32 layers)

**Answer emergence**  
`Berlin` surfaces already at *layer 22* (≈ 69 % depth) and locks in by layer 24 (>96 %).  This is the earliest of the four models relative to total depth.

**Noise in shallow layers**
* Layers 0-4 show an eclectic list of low-probability sub-word fragments (e.g. `"oren"`, `"賀"`, `"Ｌ"`).  Entropy ≈ 10 bits—the model's projections are nearly uniform.  This is typical for Llama-family models whose first block's residual stream is essentially just the word-piece embedding.
* Entropy gradually decreases as syntactic structure is integrated; by layer 14 lexical prior (`capital`, `Capital`) appears.

**Mid-layer dynamics**
* Layers 16-20 heavily feature the *concept word* `capital`/`Capital(s)` with ~17 % mass, indicating the model has mapped the question type but not yet selected the entity.
* Sharp drop in entropy (5.9 → 1.4) once `Berlin` emerges, suggesting a discrete representation "slot" being filled by an entity head.

**Observations**
* No strong distractor cities—probability mass concentrates directly onto `Berlin` without detours (`Washington` never leads the list).
* The model's final entropy (1.18) is the lowest of the cohort, reflecting a *highly peaked* answer distribution.

### 1.3 mistralai/Mistral-7B-v0.1  (32 layers)

**Answer timeline**  
`Berlin` enters the top-k around *layer 18* but fights with `Washington` until layer 24.  Only from layer 26 onward (entropy < 0.1) does `Berlin` fully dominate.

**Key patterns**
* Pronounced *"Answer" token attractor*: layers 5-10 repeatedly rank `Answer` or `answer` #1.  This suggests Mistral's pre-training distribution contains many QA patterns with the literal word `Answer` after the colon.
* **Geographical confusion**: layer 22 ranks `Washington` at 47 %.  This may stem from the frequent bigram "capital—Washington" in US-centric corpora and shows up before the model conditions on the last few prompt tokens.

**Entropy curve**
* Starts very high (10.2), drops below 6 only at layer 20, then plummets to < 0.1 by layer 26.  The comparatively *late* certainty and the intermediate `Washington` spike signal that Mistral's knowledge is accessed via a multi-step deliberation chain.

**Red flags**
* The strong `Washington` prior could leak into generation under *temperature sampling* with slightly mis-specified prompts, a potential safety issue.

### 1.4 Qwen/Qwen3-8B  (36 layers)

**Answer emergence**  
The model oscillates between `Answer` scaffolding tokens and entity tokens much longer than the others: `Berlin` only wins decisively at *layer 27* (> 75 % depth).

**Unusual token distribution**
* Shallow layers (0-10) output a mix of CJK characters, programming identifiers (`ValidationResult`, `ListViewItem`), and random English nouns—an unusually *heterogeneous* early lattice with entropy ≈ 8–10.
* Repetitive `Answer` / `_ANS` / `____` tokens dominate layers 18-24.  This looks like an artefact of prompt-format training data where blanks (underscores) follow "Answer".
* Only at layer 24 does `Germany` top the list; the city appears three layers later.

**Entropy profile**
* Entropy crashes from ~6 to <0.15 within three layers (24→27) once the country token cues the city head.

**Notable quirks**
* Heavy use of leading spaces in tokens (`' Berlin'` not `Berlin`) is consistent with Qwen's tokenizer but leads to very low entropy once a leading-space token dominates.
* The model shows higher final entropy (1.40) than others, indicating residual uncertainty among rewrite variants (`' The'`, `' Germany'`).

---

## 2. Cross-Model Insights

### 2.1 Convergence depth

| Model | Layers tot | Earliest layer where **Berlin > 50 %** | % depth |
|-------|-----------|----------------------------------------|---------|
| Meta-Llama-3-8B | 32 | 22 | 69 % |
| Mistral-7B-v0.1 | 32 | 24 | 75 % |
| Qwen-3-8B | 36 | 27 | 75 % |
| Gemma-2-9B | 42 | 33 | 79 % |

*Interpretation*: Llama's architecture (with SwiGLU, grouped-query attention, etc.) seems to retrieve factual entities earlier given its depth.  Gemma, despite more layers, relies on deeper circuitry before locking the answer.

### 2.2 Common processing pipeline

All four traces share a **three-stage pattern**:

1. **Prompt scaffolding** – punctuation or the literal word `Answer` dominates (layers 0-10).  
2. **Type cue phase** – words like `answer`, `capital`, articles (`the`, `The`) become top candidates (mid-layers).  
3. **Entity resolution** – probability mass collapses onto `Berlin` with a rapid entropy drop (final quarter).

This supports the hypothesis that transformer layers form a *progressive information-refinement pipeline*: syntax → semantics → entity.

### 2.3 Red-flag divergences

* **Mistral's `Washington` spike** – suggests a US-centric bias that could surface in generation if the prompt is ambiguous.
* **Qwen's blank/underscore tokens** – large probability on templates like `____` may lead to degenerate or template-style outputs when sampling with temperature.
* **Gemma's zero-entropy colon** – over-confidence on punctuation tokens hints at softmax saturation; mitigation might require rescaling unembedding weights in interpretability lenses.

### 2.4 Entropy as confidence signal

Across models, entropy < 0.3 almost always coincides with semantically correct prediction.  The steep drop happens within 2-4 layers, suggesting a *critical factual head* whose output is amplified by subsequent MLP blocks.  This offers a concrete target for causal tracing or patching.

### 2.5 Implications for interpretability tools

* The LayerNorm-lens appears to align well with final predictions once entropy < 1; before that point, raw residuals can be misleading (cf. Gemma's fixation on `:`).
* Monitoring **entropy gradients** across layers is a cheap heuristic to locate factual-information retrieval circuits without scanning logits.

---

*Prepared by OpenAI o3*
