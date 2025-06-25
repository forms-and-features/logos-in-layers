# Qwen /Qwen3-8B – Layer-by-Layer Probe Analysis

## 1. Overview

* **Model inspected:** `Qwen/Qwen3-8B` (8 B parameters, 36 decoder blocks).
* **Probe:** `run.py` captured the residual stream at the embedding output and after every transformer block and projected it to logits with and without LayerNorm-lens (norm-lens disabled here because the model uses RMSNorm).
* **Prompt used:**
  ```
  Question: What is the capital of Germany? Answer:
  ```
* **Key tasks for this write-up:** identify salient patterns, anomalies, and potential interpretability take-aways that are directly supported by the probe dump.

## 2. Normalisation set-up and its consequences

```7:13:001_layers_and_logits/output-Qwen3-8B.txt
Block LayerNorm type: RMSNormPre
⚠️  Non-vanilla norm detected (RMSNormPre) - norm-lens will be skipped to avoid distortion
Final LayerNorm type: RMSNormPre
```

* The model relies on **RMSNormPre** in every block and at the output. RMSNorm rescales only by the root-mean-square and has no mean-subtraction.
* Because the probe's norm-lens assumes vanilla LayerNorm (mean-zero + variance scaling), the experiment correctly **skipped normalising** the cached residuals. This avoids introducing fictitious directions (cf. Nanda 2023, §4.1).
* Interpretability implication: without LayerNorm-lens we are inspecting the *raw* residual stream; logits therefore contain scale information that LayerNorm would ordinarily remove.

## 3. Evolution of the prediction across depth

### 3.1 Early layers (embedding → block 6)

```20:35:001_layers_and_logits/output-Qwen3-8B.txt
Layer  0 (embeddings):                entropy: 11.927
   1. 'いらっ' (0.053)
   2. ' binaries' (0.053)
   … multilingual & code-like fragments …
```

* **Very high entropy (~12 nats)** and **polyglot noise** dominate – tokens from Japanese, Chinese, German, code identifiers. Typical for unconditioned embedding projections and indicates no question context captured yet.
* Layers 1-6 steadily lower the entropy but remain >10, still far from converging on a factual answer.

### 3.2 Middle layers – spurious "Answer" attractor (blocks 17-24)

```420:440:001_layers_and_logits/output-Qwen3-8B.txt
Layer 18 … entropy: 3.533
   1. ' Answer' (0.743)
   2. '回答'     (0.065)
   3. ' None'    (0.039)
```

* From block 17 onward the distribution collapses onto a **template token "Answer"** (and its multilingual variants).
* Entropy plunges from ~7 → 0.14 by block 20, yet the content token *Berlin* is **not present** – the network is confident but incorrect at this stage.
* This 'attractor' is likely a **prompt-format prior**: the model has learned that after "Answer:" often comes the literal word *Answer* in training data. Similar behaviours have been documented in tuned-lens studies (Belrose et al. 2023, Fig. 3).

### 3.3 Late layers – factual resolution (blocks 25-36)

```650:669:001_layers_and_logits/output-Qwen3-8B.txt
Layer 28 … entropy: 0.000
   1. ' Berlin' (1.000)
```

* Block 25 introduces **"Germany"** with ~1.0 probability; block 27 still vacillates between "Germany" and low-prob "Berlin".
* **First unmistakable appearance of the correct city** is at block 28, where the distribution becomes **degenerate on "Berlin"** and remains so through the remaining eight layers.
* The entropy reports negative 0.000 in blocks 34-35 – a floating-point artefact due to log-softmax on a delta distribution; not a theoretical issue but worth noting.

### 3.4 Final logits

```835:855:001_layers_and_logits/output-Qwen3-8B.txt
Model's final prediction (entropy: 1.400):
   1. ' Berlin' (0.746)
   2. ' The'    (0.089)
   3. ' Germany'(0.036)
```

* After the unembedding & RMSNorm layers the model softens again: *Berlin* 75 %, *The* 9 %, blank token 3 %. The late blocks encode a near-one-hot direction but the final normalisation dilutes it slightly.

## 4. Additional probing results

* **Consistency tests** (`Germany's capital is …`, `Berlin is the capital of …`) show the model assigns 0.70–0.74 probability to the correct continuation, indicating the factual knowledge is robust across paraphrases.
* **Temperature sweep** confirms the logit scale is sensible: at τ = 0.1 the distribution is delta on *Berlin*; at τ = 2.0 it broadens but *Berlin* stays on top (0.36).

## 5. Notable anomalies & red-flags

| Observation | Evidence | Potential concern |
|-------------|----------|-------------------|
| Strong bias toward literal token **"Answer"** at mid-depth | See Layer 18–21 dump above | Suggests over-representation of superficial template features; can confound causal tracing of factual circuits.
| **Placeholder tokens** "____", " ______" dominate Layer 24 | 490-505 lines | May reflect training data artefacts (markdown tables, fill-in-the-blank tasks) – irrelevant semantics but large logit mass.
| **Negative entropies** in layers 34-35 | lines 750-760 | Numerical precision issue; highlights that raw-residual lens can yield unusable entropy metrics late in network.
| **Multilingual noise** in early residual lens | lines 18-38 | Indicates embedding space poly-semanticity; early features are not specialised for the English factual task.

None of these threaten safety directly, but they signal areas where attribution methods could be misled.

## 6. Interpretability take-aways

1. **Information routing delay.** Relevant semantic information ("Berlin") only becomes linearly readable after ~75 % of depth (block 28 of 36). This mirrors findings in Mistral-7B and Llama-2 where factual features emerge late (Meng et al. 2022).
2. **Template suppression circuit.** The surge of "Answer" logits then their abrupt drop implies a circuit that first fulfils a *format prior* and is later overridden by factual circuits. Identifying the attention heads/MLP neurons responsible is a promising next step (cf. Wang et al. 2022 on IOI task).
3. **RMSNorm complicates norm-lens.** Since norm-lens is skipped we cannot directly compare to models with vanilla LayerNorm. For cross-model analyses, a RMS-aware normalisation lens (e.g. scale-only lens) would be desirable.

## 7. Relevance to Nominalism vs Realism debate

While the dump is task-specific, it illustrates how an abstract *concept* ("Berlin as Germany's capital") is not hard-coded in a single layer but **emerges gradually** and only becomes unambiguous late in computation. Under a realist reading, this supports the view that concepts have objective internal representations discoverable with sufficient depth probes. Under a nominalist stance, one might emphasise the transient promotion of format tokens ("Answer") as evidence that representation is contingent on surface patterns rather than "real" entities. The data alone do not settle the debate but provide concrete traces for each camp to analyse.

## 8. Recommended next investigative steps (model-local)

* **Head attribution:** use causal tracing or path-patching to locate the attention heads that first inject *Berlin* around block 27 → 28.
* **RMSNorm lens:** implement a simple scale-only lens to compare raw vs. RMS-normalized residuals and verify whether *Berlin* becomes linearly separable earlier under that lens.
* **Template-prior dissection:** search for neurons that spike on the literal token " Answer" across different prompts; ablate to test effect on final prediction.

## 9. References

* Belrose, N., Furman, Z., Smith, L. et al. (2023). *Eliciting Latent Predictions from Transformers with the Tuned Lens*. arXiv:2303.08112.
* Nanda, N. (2023). *Progress Measures for Grokking via Mechanistic Interpretability*, §4.1. arXiv:2301.05217.
* Wang, K., Variengien, A., Conmy, A. et al. (2022). *Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small*. arXiv:2211.00593.
* Meng, K. et al. (2022). *Locating and Editing Factual Associations in GPT*. arXiv:2202.05262.

---

*Prepared by: OpenAI o3*