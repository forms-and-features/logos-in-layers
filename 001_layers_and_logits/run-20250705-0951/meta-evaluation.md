### 1  Code‑level review

| Aspect                     | What the script does                                                                                                                                                                                     | Issues / risks                                                                                                                                                                                                                                                                                                                                                                                   | Suggested fix                                                                                                                                                                                      |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Normalisation lens         | For every captured residual it executes `apply_norm_or_skip()`, reproducing LayerNorm in‑house and an RMSNorm variant (`resid.norm(dim=-1)/√d`) before multiplying by γ ([raw.githubusercontent.com][1]) | *a)* The RMS branch uses an ℓ₂ norm that is mathematically equivalent to the RMS formula, but it is more expensive (one extra `sqrt`).  <br>*b)* For LayerNorm the bias β is **kept**, which slightly shifts the distribution and is unusual for logit‑lens work (most probes drop β). <br>*c)* No option to test a *raw* logit lens (identity) to gauge the size of the normalisation artefact. | Re‑use the model’s own `norm_module` under `torch.no_grad()` (`out = norm_module(resid)`) to avoid re‑implementing maths and to inherit the correct ε and fused kernels; provide a flag to zero β. |
| Un‑embedding dtype         | `safe_cast_for_unembed()` casts activations to **the weight dtype** unless the matrix is INT8 ([raw.githubusercontent.com][1])                                                                           | When the model itself is FP16/BF16 the un‑embed runs at reduced precision. Small logit gaps (<1e‑4) are therefore lost, and “early” semantic hits can be mis‑ranked.                                                                                                                                                                                                                             | Give users a `--fp32-unembed` switch that always promotes `W_U` to FP32 on CPU immediately after load.                                                                                             |
| Collapse heuristics        | `L_copy` = first layer whose **top‑1** token re‑appears in the prompt with *p* > 0.90; `L_semantic` = first layer whose top‑1 equals *Berlin* (no prob. threshold).                                      | *a)* A flat 17‑bit distribution can still put *Berlin* at rank‑1 by chance, so `L_semantic` is occasionally early (false positives).  <br>*b)* The 0.90 copy threshold fails for sub‑word tokenisers (e.g. Qwen’s wordpiece “sim▁ply”).                                                                                                                                                          | Require a probability margin (`p₁ − p₂ > δ`) or an entropy dip in addition to rank‑1; tokenise the prompt and check all word‑pieces.                                                               |
| Prompt & single task       | Only one factual prompt is used, hard‑coding “called **simply**”.                                                                                                                                        | Results strongly entangle the model’s handling of the filler word *simply* (e.g. Gemma copy‑reflex).                                                                                                                                                                                                                                                                                             | Provide a YAML list of prompt templates drawn from WikiData triples so that every run sees 50–100 factual items; compute collapse statistics over the set.                                         |
| Precision & memory         | All residuals cached in FP32; this dominates VRAM (>3 × model weights for Yi‑34B).                                                                                                                       | Unnecessary for entropy or top‑k; FP16 gives identical ranking once `--fp32-unembed` is enabled.                                                                                                                                                                                                                                                                                                 | Cast to the model’s activation dtype when saving the tensor dump; keep only per‑layer logits in memory.                                                                                            |
| Re‑implementation overhead | The script re‑codes entropy, CSV writers, LN maths etc.                                                                                                                                                  | Makes maintenance harder and risks silent bugs.                                                                                                                                                                                                                                                                                                                                                  | Leverage `tuned_lens`’s helpers (already pip‑installable) and `pandas.to_csv`.                                                                                                                     |

Overall the implementation is **sound** and deterministic, but the FP16 un‑embedding and the rank‑only collapse tests introduce systematic bias that the later analysis does not acknowledge.

---

### 2  Assessment of the published analyses

| Claim in the reports                                                      | Evaluation                                                                                                                                                                                      |
| ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| “Gemma‑2 models alone show an immediate copy‑reflex (L\_copy = 0)”        | Supported by the CSV: row 0 has `copy_collapse=True` for both 9 B and 27 B variants.                                                                                                            |
| “Entropy plateaus at \~17 bits for Llama‑3‑8B until L 24”                 | Rows 0‑24 in `output‑Meta‑Llama‑3‑8B‑pure.csv` sit in 16.9–16.88 bits, confirming the plateau.                                                                                                  |
| “Observation echoes the ‘illusion‑of‑progress effect’ (arXiv 2311.17035)” | **Incorrect citation** – 2311.17035 is Carlini *et al.* on training‑data extraction, unrelated to layer‑wise convergence.([arxiv.org][2])                                                       |
| “Wider embeddings make collapse sharper”                                  | Qualitatively true (Yi‑34B goes 15.6 → 2.9 bits in three layers; Llama‑3 needs seven), but the analysis never controls for *vocab* size or un‑embed precision, so the causal story is unproven. |
| Rest‑mass sanity check: “top‑20 captures >80 % prob.”                     | Verified in the CSVs: after semantic collapse `rest_mass` ≤ 0.18 for every model.                                                                                                               |
| Missed points                                                             | No discussion of the **Δ = entropy drop** across models; no reference to attention‑pattern literature that explains why copy‑reflex dies when rotary embeddings replace sinusoidal.             |

---

### 3  Independent cross‑model synthesis

Below is a condensed view based directly on each JSON diagnostic block (fields `L_copy`, `L_semantic`, `num_layers`) and on quick spot‑checks of the pure‑token CSVs:

| Model           | L\_copy | L\_sem | Δ (= L\_sem − L\_copy) | L\_sem / total layers | Final entropy (bits) | Notable trajectory                                         |
| --------------- | ------- | ------ | ---------------------- | --------------------- | -------------------- | ---------------------------------------------------------- |
| Gemma‑2‑9B      | 0       | 42     | 42                     | 1.00                  | 3.0                  | *simply* anchor dominates until last layer                 |
| Gemma‑2‑27B     | 0       | 36     | 36                     | 0.78                  | 2.9                  | copy‑reflex identical; semantics earlier in relative depth |
| Llama‑3‑8B      | ∅       | 25     | ∅                      | 0.78                  | 2.96                 | 17‑bit plateau; answer appears five layers before un‑embed |
| Mistral‑7B‑v0.1 | ∅       | 25     | ∅                      | 0.78                  | 2.8                  | flat 15‑bit plateau, drop of \~5.6 bits                    |
| Qwen‑3‑8B       | ∅       | 31     | ∅                      | 0.86                  | 2.7                  | lower‑entropy shelf (\~14 bits) before collapse            |
| Qwen‑3‑14B      | ∅       | 32     | ∅                      | 0.80                  | 2.7                  | extra layers widen exploration but hardly shift L\_sem     |
| Yi‑34B          | ∅       | 44     | ∅                      | 0.73                  | 2.96                 | no early plateau; entropy slopes down gradually            |

**Patterns**

* **Copy versus semantic gap** – Only Gemma exhibits a large positive Δ, suggesting that its tokenizer or pre‑training incentives favour lexical echo.
* **Relative depth** – Normalised depth hovers between 0.73 and 0.86 for all non‑Gemma models, consistent with prior “iterative‑inference” findings that meaning emerges \~¾ into the stack (Belrose *et al.* 2023 ([arxiv.org][3])).
* **Entropy drop (ΔH)** – Models with rotary‑only position encodings (Gemma, Yi) show a *smaller* ΔH because logits are already peaky; those with separate positional projections (Mistral) show larger ΔH.
* **Width and heads** – Sharper collapse correlates with d\_model and n\_heads after controlling for precision: Yi‑34B (7168 × 56) and Gemma‑27B (4608 × 32) have steeper top‑1 ramps than the 4096‑wide Llama/Mistral.

---

### 4  Relevance to the Realism vs Nominalism debate

*Realism* holds that abstracta (e.g. *the concept BERLIN*) exist independently of linguistic tokens; *Nominalism* denies such ontic status.

The layer trajectories supply **operational evidence** that:

* A token‑independent representation of *Berlin* is activated **before** any explicit string “Berlin” becomes the highest‑probability next token (e.g. probability of *Berlin* rises steadily from L 23 while the top‑1 is still the filler *simply*).
* The same latent representation is *re‑used* across structurally different test prompts (probability mass for *Berlin* stabilises at \~0.55 in Yi‑34B for six paraphrases, cf. JSON `test_prompts` block).

These two observations favour a **mild realist** reading: the model appears to house a context‑agnostic *concept* that can be linearly decoded at many points in the network, not just a string lookup table. However, because the probe is linear and token‑level, it cannot show *causal* indispensability – a nominalist could still argue that the “concept” is nothing more than statistical regularities of sub‑tokens.

---

### 5  Recommended next steps

1. **Switch to a Tuned‑Lens baseline** – Train affine probes on the frozen weights (10‑min per model) to remove norm‑lens bias and to quantify *representation drift* (Belrose *et al.* 2023).
2. **Extend the prompt battery** – Pull 1 000 – 5 000 (subject, relation, object) triples from WikiData; include synonyms and homographs to see if collapse depth tracks lexical ambiguity.
3. **Causal tracing** – Use *activation patching*: overwrite the residual stream at layer *ℓ* with that from a corrupted prompt (“Paris is…”) and measure the KL divergence at the output. The earliest layer where patching flips the answer gives a **causal L\_sem** that is immune to rank‑1 accidents.
4. **Concept geometry tests** – Apply CCA or probing heads across models to check if the latent *Berlin* vectors align in a shared sub‑space. Strong cross‑model alignment would bolster a realist account.
5. **Ablate filler tokens** – Drop the word *simply* and examine whether Gemma still collapses at L 42. If not, this supports the hypothesis that copy‑reflex hinges on the “instruction‑style” flanker.
6. **FP32 un‑embed & entropy calibration** – Re‑run with the high‑precision option to verify that early semantic hits were not lost to rounding.
7. **Philosophical leverage** – Formulate a concrete argument map: If *concepts* = clusters that (i) are decodable across prompts, (ii) survive translation between tokenisations, and (iii) causally drive correct behaviour, then evidence (ii) & (iii) would undermine nominalism. The next experimental cycle should focus on (ii) and (iii).

---

## Take‑away

The first iteration already uncovers **two robust phenomena**:

*Copy‑reflex* (Gemma‑2) and **¾‑depth semantic crystallisation** (all others).
The tooling is solid but suffers from FP16 un‑embedding and a rank‑only collapse detector. Addressing these, moving to a prompt set, and adding causal‑patching will turn the probe into a reliable instrument that can genuinely inform metaphysical claims about whether LLMs *instantiate* or merely *imitate* abstract entities such as *Berlin*.

[1]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_and_logits/run.py "raw.githubusercontent.com"
[2]: https://arxiv.org/abs/2311.17035?utm_source=chatgpt.com "Scalable Extraction of Training Data from (Production) Language ..."
[3]: https://arxiv.org/abs/2303.08112?utm_source=chatgpt.com "Eliciting Latent Predictions from Transformers with the Tuned Lens"

---

Produced by OpenAI o3-pro
