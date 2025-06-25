# Cross-Model Layer-wise Probe Analysis 

**Models inspected**: Qwen/Qwen3-8B · Google/Gemma-2-9B · Meta-Llama-3-8B · Mistral-7B-v0.1  
**Probe script**: `run.py` (raw residual lens; norm-lens skipped for all runs due to RMSNormPre)  
**Prompt**: `Question: What is the capital of Germany? Answer:`  

---

## 1 Executive summary

1. Across *all four* architectures the factual token *Berlin* only becomes linearly readable in the **final quarter of network depth** (≈60-80 % of layers). This supports the iterative-refinement picture in decoder-only transformers first reported by Nostalgebraist-Logit-Lens and later formalised by the Tuned-Lens work ([Nostalgebraist 2020](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens); [Belrose et al. 2023](https://arxiv.org/abs/2303.08112)).
2. All checkpoints employ **Pre-RMSNorm**, forcing the probe to skip LayerNorm-lens. Consequently, entropy magnitudes are *not* directly comparable with papers that assume vanilla LayerNorm; nevertheless, relative trends are internally consistent.
3. A common **type → instance trajectory** is observed: models first converge on the *category* token "capital/answer" and only later on the *instance* "Berlin". The timing of this transition is a fertile locus for mechanistic tracing of factual circuits.
4. Each model shows idiosyncratic **format or junk-token attractors** in mid-depth, hinting at spurious features that could confound philosophical interpretation (e.g. literal token "Answer" in Qwen, `ABCDEFGHIJKLMNOP` in Llama-3, repeated colon in Gemma).
5. Prompt paraphrase tests reveal **asymmetric recall**: all models answer the reverse relation ("Berlin → Germany") more reliably than the forward ("Germany's capital is → ?"). This surface-form sensitivity is a substantive data-point for nominalism vs. realism debates.

---

## 2 Depth at which *Berlin* emerges

| Model | Layers (total) | First appearance ≥5 % | Dominant (p > 0.6) | Depth fraction (dominant) |
|-------|---------------|------------------------|--------------------|---------------------------|
| Qwen-3-8B | 36 | Block 25 | Block 28 | **0.78** |
| Gemma-2-9B | 42 | Block 31 | Block 34 | **0.81** |
| Llama-3-8B | 32 | Block 22 | Block 26 | **0.81** |
| Mistral-7B | 32 | Block 22 | Block 30 | **0.94** |

*Evidence*: layer excerpts quoted in individual evaluation files (`evaluation-*.md`, e.g. Qwen lines 650-669 and Gemma lines 700-730).  The ratios confirm a *late-binding* pattern consistent with the "knowledge funnel" reported by Elhage et al. 2021 (<https://transformercircuits.pub/2021/framework/index.html>).

---

## 3 Shared phenomena

### 3.1 Early-layer entropy plateau & sub-word noise

All runs start with high entropy (≈10–12 bits) and top-k tokens comprising low-frequency BPE fragments (e.g. `՛rya`, `賀`, code identifiers). This replicates prior observations that without norm-lens, early residual directions correlate with rare embedding axes, producing seemingly random high-probability junk ([Belrose et al. 2023], §3.2).

### 3.2 Category-before-instance dynamic

In every model, an abstract category token ("capital", "Answer", or a generic article "the/a") gains probability *before* the city name. Such staging suggests that networks compute relational templates prior to binding them to entities—empirical scaffolding for the realist view that abstract relations have internal instantiation vectors later fused with entity vectors.

### 3.3 Norm-type confounder

Because Pre-RMSNorm rescales by RMS only, applying a vanilla LayerNorm lens would distort directionality ([Zhang & Sennrich 2019](https://arxiv.org/abs/1910.07467)). The probe prudently skips norm-lens, but this means entropy curves cannot be compared numerically with vanilla-LN models in the literature. A dedicated **RMS-aware lens** (scale-only or tuned-lens retrained) is recommended for future runs.

---

## 4 Model-specific anomalies & red flags

| Model | Anomaly | Layer span | Why it matters |
|-------|---------|------------|----------------|
| Qwen-3-8B | Literal token **" Answer"** becomes near-delta (p ≈ 0.74) mid-stack; placeholder tokens "____" spike later | 17–24 | Indicates a strong *template prior* that temporarily overrides factual circuits; could mislead causal tracing if not controlled. |
| Gemma-2-9B | Embedding & first nine layers are *deterministic* on `':'` (entropy ≈ 0) | 0–8 | Early over-confidence masks signal; suggests limited long-range integration until ≥ layer 10. |
| Llama-3-8B | Junk token **`ABCDEFGHIJKLMNOP`** obtains 7 % probability repeatedly | 17–21 | Likely memorised artefact or placeholder feature; warrants activation patching to assess circuit function. |
| Mistral-7B | Newline token dominates instruction prompt ("Respond in one word: …"); oscillation between *Berlin* and **Washington** | 23–31 | Shows formatting bias and semantic interference between capital-city features—critical if precise output control is required. |

---

## 5 Prompt robustness analysis

The probe's auxiliary prompts expose a shared weakness: the forward relation ("Germany's capital is …") often fails, while the inverse ("Berlin is the capital of …") is answered confidently (≥90 % for Llama-3 and Mistral).  This asymmetry corroborates findings from Brown et al. 2020 (<https://arxiv.org/abs/2005.14165>) that factual recall is heavily prompt-dependent.  For the philosophical project, such surface sensitivity provides ammunition for a **nominalist** interpretation: factual "knowledge" may be stored more as conditional distributions over strings than as abstract, context-independent propositions.

---

## 6 Implications for the nominalism ↔ realism debate

1. **Support for Realism**: The consistent late-layer crystallisation of a single factual token across diverse architectures implies the presence of robust internal features corresponding to the concept *Berlin-as-capital*. Realists can argue these features are discoverable, stable representations.
2. **Support for Nominalism**: The strong mid-layer attraction to format tokens ("Answer", colon, newline) and the high dependency on prompt wording signal that what the network "knows" is tightly coupled to surface statistics. Concepts may therefore be linguistic constructs rather than ontic entities.
3. **Synthesis**: The data favour a two-stage story—early nominalist template fitting followed by realist instance binding—suggesting the debate may benefit from a *layer-relative* framing rather than a binary.

---

## 7 Recommendations for next steps

1. **Implement a scale-only or trained RMS-Tuned-Lens** to enable meaningful cross-model entropy comparisons.  See Jiang et al. 2023 (<https://arxiv.org/abs/2305.14858>) for equivalence proofs between Pre-LN and Pre-RMSNorm.
2. **Causal tracing of the category→instance hand-off**: patch attention heads where *Berlin* probability first spikes (Qwen block 25, Gemma 31, etc.) to isolate the circuit that injects entity-specific information.
3. **Ablate template attractor features** (e.g. Qwen "Answer", Llama-3 junk token) to test whether they are necessary for correct factual recall.
4. **Systematically vary prompt framing** to quantify how representation stability varies with surface form; treat variance as epistemic uncertainty in philosophical arguments.

---

### Reference list

* Belrose, N., Furman, Z., Smith, L. *et al.* (2023). **Eliciting Latent Predictions from Transformers with the Tuned Lens**. *arXiv:2303.08112*.
* Brown, T. *et al.* (2020). **Language Models are Few-Shot Learners**. *arXiv:2005.14165* (prompt-sensitivity observations, §6.2).
* Elhage, N., Nanda, N., Henighan, T. & Olsson, C. (2021). **A Transformer Circuit Framework for Mechanistic Interpretability**. <https://transformercircuits.pub/2021/framework/index.html>.
* Jiang, Z. *et al.* (2023). **Pre-RMSNorm and Pre-CRMSNorm Transformers: Equivalent and Efficient Pre-LN Transformers**. *arXiv:2305.14858*.
* Nostalgebraist (2020). **Interpreting GPT: The Logit Lens**. <https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens>.
* Zhang, B. & Sennrich, R. (2019). **Root Mean Square Layer Normalization**. *arXiv:1910.07467*. 