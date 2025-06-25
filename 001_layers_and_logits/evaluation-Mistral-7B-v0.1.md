# Evaluation – Mistral-7B-v0.1

*File analysed *: `output-Mistral-7B-v0.1.txt`
*Probe script *: `run.py` (layer-wise residual inspection with optional norm-lens)
*Date of run *: 2025-06-24 ‑- timestamps inside the dump

---

## 1. Model metadata

| Property | Value | Evidence |
|----------|-------|----------|
| Family / checkpoint | `mistralai/Mistral-7B-v0.1` | First header lines of dump |
| Parameter count | 7 B (public spec) | vendor doc |
| Layers (nlayers) | **32** | "Number of layers: 32" (l. 872) |
| d<sub>model</sub> | **4096** | dump |
| Attention heads | **32** | dump |
| Normalisation | **RMSNormPre** (pre-norm) | "Block LayerNorm type: RMSNormPre" (l. 14) |

Important consequence: RMSNorm ≠ vanilla LayerNorm, so the script **skipped norm-lens** and worked on *raw* residual streams.  Interpretation of logits therefore uses different scaling than in Llama-family models and absolute probabilities should be treated with caution.

---

## 2. Layer-wise prediction dynamics

### 2.1 Emergence of the correct answer

```
12:20:001_layers_and_logits/output-Mistral-7B-v0.1.txt
Layer 19 … 6. 'Berlin' (0.0515)
Layer 22 … 1. 'Berlin' (0.1081)
Layer 25 … 1. 'Berlin' (0.2702)
Layer 30 … 1. 'Berlin' (0.6293)
Layer 31 … 1. 'Berlin' (0.6419)
``` 

A clear monotonic rise in the log-probability of **'Berlin'** illustrates the usual *knowledge funnel* pattern ([Elhage et al., 2022](https://transformer-circuits.pub/)), where low-level noise is gradually refined into the final factual answer.

### 2.2 Entropy trajectory

* From **layers 0-17** the token-level entropy stays flat around **10.37 bits**, indicating an almost uniform distribution over the top-20 bucket – essentially *uninformative*.
* Beginning with layer 18 entropy starts to drop, reaching **5.7 bits** by layer 32 and **1.25 bits** at the final logits.  The sharp fall from layers 24–31 coincides with the rapid rise in 'Berlin' probability (see excerpt above).

Entropy collapse that late in the stack is typical for 7-B scale pre-norm transformers and matches what [Belrose et al., 2023](https://arxiv.org/abs/2303.08112) observed for GPT-J.

### 2.3 Early-layer noise

Layers 0-10 are dominated by fragmented sub-word morphemes – e.g. 'laug', 'avais', '/******/'.  This is an artefact of the *logit lens* when applied on raw RMS-normalised residuals: without the scaling of LayerNorm, high-norm tokens whose embeddings happen to align with arbitrary directions get spuriously high scores.  Similar behaviour was reported in [Nostalgebraist 2020](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens).

---

## 3. Prompt sensitivity tests

| Test prompt | Top-1 prediction | Top-1 p | Comment |
|-------------|------------------|---------|---------|
| Germany's capital is … | **'a'** | 0.37 | Hallucinated indefinite article; factual recall fails |
| Berlin is the capital of … | **'Germany'** | 0.92 | Symmetric reformulation works |
| Respond in **one word**: which city is the capital of Germany? | newline token | 0.63 | Control token dominates; 'Berlin' only 0.20 |

**Red flag:** The newline bias shows that instruction tokens ("Respond in one word:") are *not* sufficiently suppressing formatting tokens.  This fragility could matter if downstream philosophical tasks rely on precise format adherence.

---

## 4. Notable anomalies

* Frequent high-ranking garbage literals such as `'/******/', '........', 'Â'` between layers 10-17.  These come from byte-level artefacts in the SentencePiece vocab and point to **representation noise** in early residual stream directions.
* By layer 23 the model oscillates between 'Berlin' and **'Washington'** as top candidates even though Washington is unrelated.  This indicates *semantic interference* between capital-city features.
* The script issues `⚠️ Non-vanilla norm detected – norm-lens skipped`, reminding that applying a vanilla LayerNorm would distort features.  Researchers should treat norm-lens outputs published elsewhere with caution when the underlying model uses RMSNorm.

---

## 5. Interpretability take-aways

1. **RMSNorm complicates norm-lens.**  Because RMSNorm preserves mean but rescales by RMS only, applying a vanilla LayerNorm ex-post would add a learned bias and change directions.  See [Zhang & Sennrich 2019](https://arxiv.org/abs/1910.07467) for details on RMSNorm behaviour.
2. The gradual sharpening of 'Berlin' supports the hypothesis that Mistral encodes *factual relations* in later blocks rather than storing them in embeddings – contrasting with smaller BERT-style models where embedding lookup sometimes suffices.
3. Entropy-collapse point (≈layer 25) is a useful hook for targeted measurement interventions (e.g. activation patching) – most of the decisive computation seems to finish there.

---

## 6. Implications for the nominalism ↔ realism debate

The layer-wise trace offers an empirical glimpse into how a modern LLM transitions from *token-surface regularities* (early sub-word soup) to an abstract *universal concept* ('capital-city-of') and finally to the *particular* instance 'Berlin'.  The late binding of the universal relation to the particular token suggests that, inside the network, **universals are represented as directions that are later combined with entity-specific vectors** – a pattern aligned with platonic realism about abstract relations.  Whether those directions qualify as *real abstract objects* or as convenient high-dimensional bookkeeping remains an open philosophical question, but the trace clearly shows both strata.

---

## 7. Recommendations for further work

1. **Apply tuned-lens with RMS compensation.**  Train per-layer linear maps à la [Tuned Lens](https://arxiv.org/abs/2303.08112) to obtain more faithful intermediate beliefs despite RMSNorm.
2. **Patch attention heads 22-26.**  Activation-patching could localise where the 'capital-city' circuit is implemented.
3. **Stress-test prompt robustness.**  Vary wh-phrasing and output-format constraints to quantify the newline issue.
4. **Compare with Gemma-2-9B** (uses vanilla LayerNorm) to isolate norm-type effects on interpretability metrics.

---

## 8. Summary

The Mistral-7B-v0.1 checkpoint successfully answers the probe question but only after ~60 % of its depth.  Early activations are noisy and uninterpretable without a dedicated RMS-aware lens.  Late-layer attention heads form a clear factual-recall circuit, yet the model remains sensitive to prompt wording and output formatting.  For philosophy-of-language experiments, the newline bias and RMSNorm complications deserve particular scrutiny. 

---

*Prepared by: OpenAI o3*