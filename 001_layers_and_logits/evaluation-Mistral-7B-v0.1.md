# Mistral-7B-v0.1 – Layer-wise Logit-Lens Inspection  

*File analysed:* `output-Mistral-7B-v0.1.txt` (full console dump produced by `run.py`).  
*Model:* `mistralai/Mistral-7B-v0.1` – 32-layer decoder-only Transformer, *d*ₘₒ𝖽𝖾𝖑 = 4096, 32 heads, vocab ≈ 32 k, context = 2 048. The architecture uses **RMSNorm-pre** at every block and before the final unembedding.  

---

## 1. Methodological notes & caveats

| Item | Evidence | Implications |
|------|----------|--------------|
| RMSNorm without exposed scale parameter | "⚠️ RMSNorm detected but no weight/scale parameter – norm-lens will be skipped" (normalisation analysis section) | Intermediate predictions are read **from the raw residual stream** rather than a normalised version. Prior work (e.g. [Belrose et al., 2023](https://arxiv.org/abs/2303.08112)) shows that lack of calibration can distort early-layer distributions. Magnitudes and entropies should therefore be treated **qualitatively**, not quantitatively. |
| Unembed weights promoted to FP32 | "🔬 Promoting unembed weights to FP32…" | Numerical stability for probability/entropy estimates is adequate despite raw lens. |
| Prompt | `Question: What is the capital of Germany? Answer:` | Simple factual-retrieval setup; permits inspection of last-token logits across layers. |

> **Take-away:** Findings about *which* tokens climb or fall in rank are trustworthy, but exact probabilities, especially in the first ~10 layers, may shift when using a tuned or normed lens.

---

## 2. Entropy trajectory

The last-token entropy (in bits) is reported per layer:

| Stage | Layers | Entropy range | Comment |
|-------|--------|---------------|---------|
| Embedding & very early | 0–5 | **14.966–14.965** | Almost maximally flat (uniform over 32 k ≈ 15 bits). Model has not specialised yet. |
| Mid stack | 6–18 | 14.965 → 14.951 | Slow, almost linear decrease; distribution still effectively noise. |
| Fact-emergence band | 19–24 | 14.945 → 14.891 | Noticeable information gain; first appearance of factual tokens (`Berlin`, `Germany`, `capital`). |
| Late convergence | 25–32 | 14.842 → **8.254** | Rapid sharpening; final layers perform bulk of logit focusing (≈ 6.6 bits drop across last seven layers). |
| Final logits | — | **1.800** | Extremely peaked: `Berlin` 0.83 prob. |

*Pattern echoes observations in tuned-lens work that many factual models defer decisive computation to late layers.*

---

## 3. Layer-wise token dynamics

### 3.1 Early layers (0–5)
* Top-20 lists are filled with morphologically rare sub-tokens (`"laug"`, `"avax"`, `"ueto"`).
* No prompt-related strings appear. This is typical when the unnormalised residual is dominated by embedding noise.

### 3.2 Transitional middle (6–18)
* Repetitive meta-tokens surge: `'Answer'`, `'answer'`, `'swer'`, `'cities'`.
* Indicates the network is **first parsing the prompt format** (Q → A) before grounding factual content.
* Minor but rising semantic hints – by layer 18 the words **`capital`** and **`Germany`** enter top-20.

### 3.3 Fact emergence (19–24)
* Layer 19: `cities` still rank #1, but **`Germany` (0.00008)** and **`Berlin` (0.00007)** appear.
* Layer 21: ranking flips – `capital` then **`Berlin` #2** (0.00019).
* **Anomaly:** Layer 23 list is topped by **`Washington`** (0.00054) *ahead* of `Berlin` (0.00050). Possible U.S. capital interference suggests overlapping representation for the concept *capital-city* without strong country disambiguation yet.
* Layer 24 reverses: `Berlin` overtakes (`0.00071`) while `Washington` slips.

### 3.4 Late convergence (25–32)
* Probabilities for `Berlin` grow super-linearly:
  * L 25 → 0.0029
  * L 26 → 0.0047
  * L 27 → 0.0090
  * L 28 → 0.0259
  * L 29 → 0.0514
  * L 30 → 0.0785
  * L 31 → 0.1416
  * L 32 → 0.296 (intermediate) – before softmax temperature rescaling.
* Final unembedded logits yield **0.83** probability for `Berlin`.
* Entropy drops accordingly, with the largest single decrease between layers 31→32 (~3.5 bits), highlighting **decisive computation in the final block**.

---

## 4. Noteworthy anomalies & red flags

1. **Spurious U.S. bias:** The temporary dominance of `Washington` at layer 23 is striking. Although later corrected, it implies a residual association "capital → Washington" that competes strongly with the country cue.
2. **Generic noun confusion:** Prolonged high rank of `cities` (layers 18–21) suggests the model sometimes treats *capital* as a generic plural class before narrowing to a specific entity.
3. **Prompt-echo artefacts:** Tokens like `'Answer'`, `'answer'`, `'swer'` persist through layer 18 despite being invalid next-word completions, reflecting over-representation of prompt templates in pre-training data.
4. **Flat early entropy:** Almost perfect 15-bit entropy until layer 5 indicates that without normalisation the embedding vector alone offers negligible discriminative signal, in line with findings that raw logit lens overestimates early-layer uncertainty ([nostalgebraist 2020](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)).

---

## 5. Interpretability take-aways

* The **capital-city fact** is *not* encoded in embeddings; it materialises gradually and is only decisively represented after ~80 % of the network depth.
* Competitive alternatives (`Washington`, `Frankfurt`, `Paris`) rise and fall, hinting at an internal **selection circuit** that weighs multiple query-matching candidates before committing.
* The sharp late-layer logit jump lends support to the hypothesis that transformer blocks closer to the output act as **refiners/validators** rather than knowledge retrievers (cf. "tuned lens" observations on Llama-2).
* The raw-lens limitation under RMSNorm underscores the importance of **norm-aware lenses**; without them, early-layer semantics are likely under-estimated (see discussion in Belrose et al., 2023, §3).

---

## 6. Relevance to the Realism vs Nominalism debate (tentative)

| Observation | Nominalist reading | Realist reading |
|-------------|-------------------|-----------------|
| Early stages dominated by surface tokens (`Answer`, `cities`) | Model begins with *names* and syntactic forms, supporting view that categories are constructed labels. | These surface forms are scaffolding; true *concepts* (e.g. a particular city) emerge later, suggesting underlying reality. |
| Gradual sharpening towards a single city | Conceptual representation appears to *track an external fact* (`Berlin`), hinting that the model holds a reality-anchored entity. | Could still be an internal convention derived from statistical co-occurrence; no necessary commitment to ontological realism. |
| Competing `Washington` candidate | Shows category overlap and absence of innate essences – concepts are distributional clusters. | The eventual suppression of `Washington` demonstrates a constraint consistent with the real-world mapping "Germany → Berlin". |

> **Caution:** These traces offer raw empirical glimpses into the model's internal states; they do **not** by themselves settle metaphysical questions. They might, however, inform *testable predictions* for future philosophical inquiry (e.g. whether concept individuation corresponds to unique attractor states inside the residual stream).

---

## 7. Summary of key points

1. **Entropy profile** is almost flat until layer 18, then collapses – factual information is consolidated late.
2. **Layer 23 anomaly** (`Washington` > `Berlin`) highlights residual geographic ambiguity.
3. **Prompt-format tokens** clutter the mid-stack, reflecting dataset biases.
4. Raw logit lens under RMSNorm deserves calibration; results are directionally informative but numerically noisy.
5. Findings provide concrete hooks for philosophical analysis without yet favouring realism or nominalism.

---

### References

* Belrose, N., Furman, Z., et al. (2023). *Eliciting Latent Predictions from Transformers with the Tuned Lens.* arXiv:2303.08112.
* nostalgebraist (2020). *Interpreting GPT: The Logit Lens.* LessWrong post. <https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens>.

---

*Prepared by: OpenAI o3*