# Meta-Llama-3-8B – Layer-wise Probe Analysis

*Experiment artefact analysed: `output-Meta-Llama-3-8B.txt` (full console dump produced by `run.py`)*

---

## 1  Experiment recap & model context

| Item | Details |
|------|---------|
| Model ID | `meta-llama/Meta-Llama-3-8B` |
| Parameters | 8 billion (32 transformer blocks) |
| Normalisation | **Pre-RMSNorm** (both block and final) – see probe log below |
| Probe method | Logit-lens style sweep over *embedding plus every residual stream* (after each block), hooking only the **last-token** position. |
| Prompt used | `Question: What is the capital of Germany? Answer:` |
| Hardware | Not reported; run on single process via TransformerLens |

> "Block LayerNorm type: **RMSNormPre** … non-vanilla norm detected – norm-lens will be **skipped**"

*(`output-Meta-Llama-3-8B.txt`, Normalisation Analysis section)*

Because the model employs RMSNorm [Zhang & Sennrich 2019](https://arxiv.org/abs/1910.07467) instead of vanilla LayerNorm, the "norm-lens" correction was disabled. All residual snapshots are therefore **raw**, an important caveat when comparing entropy magnitudes across layers.

---

## 2  High-level trajectory of model predictions

The probe prints the top-20 next-token predictions for the last position after every layer together with *per-layer entropy*.

### 2.1 Entropy trend

* Early layers (Embedding → Block 5) sit around **11.76 bits**.
* Between Blocks 24 → 30 entropy declines steadily to **≈11.08 bits**.
* A sharp collapse occurs only after the **final block (32 / lninal)** – entropy plunges to **1.36 bits**, almost matching the model's true head (**1.18 bits**).

This mirrors the pattern first highlighted by the *Logit Lens* technique [Nostalgebraist 2020](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens): earlier layers explore a broad hypothesis set; final layers sharpen a single answer.

### 2.2 Token evolution

Layer excerpts (top-5 shown; probabilities in brackets):

| Layer | Most probable candidate | Notes |
|-------|-------------------------|-------|
| 0 (embedding) | `oren` (0.050) | Essentially noise; no semantic signal of *capital*. |
| 18 | `capital` (0.0505) | First appearance of the semantic concept we want. |
| 22 | ` Berlin` (0.095) | Correct city enters top spot but with modest margin. |
| 26 | ` Berlin` (0.405) | Dominant (>40 %). |
| 32 (after lninal) | ` Berlin` (0.912) | Model decisively commits. |

> ```
> Layer 22 … 1. ' Berlin' (0.0951)
> Layer 26 … 1. ' Berlin' (0.4045)
> Layer 32 … 1. ' Berlin' (0.9116)
> ```

The *concept* "Berlin" is therefore not retrieved in one jump but **accreted** over ~10 layers, consistent with iterative‐refinement views of decoder-only transformers.

---

## 3  Salient patterns & anomalies

### 3.1 Early-layer noise characterised by rare sub-word units

Layers 0-4 are dominated by tokens such as `׳rya`, `mler`, `賀`, `энэ` – extremely low-frequency BPE fragments. Similar observations have been reported as an artefact of applying the logit-lens without normalisation (Belrose et al., 2023). They are not cause for alarm but do mean *per-token* visual inspection of very early layers is rarely informative.

### 3.2 Persistent high-probability junk token "ABCDEFGHIJKLMNOP"

From Blocks 17 → 21 the nonsense token `ABCDEFGHIJKLMNOP` appears repeatedly with probabilities up to **7 %**:

> ```
> Layer 19 … 1. 'ABCDEFGHIJKLMNOP' (0.0736)
> Layer 20 … 2. 'ABCDEFGHIJKLMNOP' (0.0672)
> ```

Such uniformly‐capitalised 16-character strings are almost absent from natural corpora. Their presence at mid-layers suggests either:

1. A memorised artefact from code/documentation training data.
2. An internal *placeholder direction* that "soaks up" probability mass when evidence is weak.

Either way, it is **worth flagging** for follow-up mechanistic analysis (e.g. SAE or feature search) because it may correspond to a *spurious feature channel*.

### 3.3 Conflation of 'capital' vs proper noun

Before Block 22 the model strongly backs generic continuations such as ` capital`, ` Capitals`, ` Capital`. Only later does it pivot to the specific answer. This is a concrete illustration of the *type–token transition* hypothesis: models first decide **semantic category** then refine to **instance**.

### 3.4 RMSNorm and absence of norm-lens

Because the probe skipped normalisation each residual vector was passed directly to the unembedding matrix. **Magnitude drift** across layers therefore affects softmax entropy directly. While the qualitative story (monotonic refinement) still holds, entropy values are not comparable to vanilla-LayerNorm models where norm-lens is applied. Future probes should consider *RMSNorm-aware* rescaling (see Jiang et al., 2023).

---

## 4  Behaviour under auxiliary queries

| Test prompt | Top-1 prediction | Comment |
|-------------|-----------------|---------|
| `Germany's capital is` | ` a` (0.48) | Fails. Pattern suggests the model treats this as **definition sentence** expecting an adjective/noun continuation, not the factual answer. |
| `Berlin is the capital of` | ` Germany` (0.92) | Correct and confident – retrieval succeeds when *city → country* direction is invoked. |
| `Respond in one word: which city is the capital of Germany?` | ` Berlin` (0.34) | Answer present but only 34 %; instruction following likely competes with factual head. |

These results reinforce a known phenomenon: *forward* relation (city→country) is easier than *reverse* unless the prompt explicitly tears down the grammatical frame.

---

## 5  Temperature sweep

The single-pass rescaling experiment shows that at **T = 0.1** probability on ` Berlin` is pushed to ≈1.0, while at **T = 2.0** it declines to 0.51 yet remains dominant. This indicates the logits are not *fragile*: the answer is encoded with a comfortable margin.

---

## 6  Implications for the nominalism ⇄ realism project

1. **Gradual concretisation**: the city name emerges progressively from an initially amorphous distribution, supporting the *realist* view that stable ontic representations (here: *Berlin*) are constructed inside the network rather than merely projected by human labelling.  
2. **Category before instance**: the network's early focus on the token ` capital` echoes nominalist concerns—language categories appear first, individuals later. Mechanistic tracing of when and where the transition occurs could illuminate how abstract categories are grounded.  
3. **Spurious feature channels** (e.g. `ABCDEFGHIJKLMNOP`) are reminders that not every internal direction has external semantics, tempering realist claims of one-to-one concept mapping.

---

## 7  Red flags & recommendations for further study

| Issue | Why it matters | Suggested follow-up |
|-------|---------------|----------------------|
| Mid-layer junk token with high prob. | Possible memorisation / overshoot feature. | Activation patching / SAE on layers 17-21. |
| Probe lacks RMSNorm correction | Entropy numbers inflated / deflated unpredictably. | Implement RMSNorm-aware lens (multiply by learned scale then divide by running RMS; cf. Zhang & Sennrich 2019). |
| Early-layer noise tokens | Not harmful but clutter visualisations. | Filter top-k by frequency or use tuned lens (Belrose et al., 2023). |

---

## 8  Key take-aways

* **Correct factual retrieval** is only locked in by the final block; intermediate blocks show partial evidence and high entropy.
* The model cleanly separates *category* (`capital`) from *instance* (`Berlin`) over ~10 layers.
* Presence of RMSNorm necessitates customised interpretability tooling; vanilla norm-lens assumptions do not hold.
* Several anomalous high-probability tokens (e.g. `ABCDEFGHIJKLMNOP`) merit mechanistic investigation.

---

### References

* Belrose, N. *et al.* (2023). *Eliciting Latent Predictions from Transformers with the Tuned Lens.* arXiv:2303.08112.
* Nostalgebraist (2020). *Interpreting GPT: The Logit Lens.* LessWrong. https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
* Zhang, B. & Sennrich, R. (2019). *Root Mean Square Layer Normalization.* arXiv:1910.07467.
* Jiang, Z. *et al.* (2023). *Pre-RMSNorm and Pre-CRMSNorm Transformers: Equivalent and Efficient Pre-LN Transformers.* arXiv:2305.14858. 