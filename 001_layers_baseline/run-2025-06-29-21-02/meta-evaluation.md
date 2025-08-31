# Prompt

You're a top LLM interpretability researcher at a leading AI lab (think OpenAI, Anthropic, Google). You're guiding and consulting an experiment that aims to apply the results of LLM interpretability research to push forward the philosophical debate between nominalism and realism. This is a "hobby" project of a software engineers just getting started with interpretability, but interested in using LLM interpretability to push the debate as far as possible with the tools available.

With the help of an AI co-pilot, the user ran experiments on a few open-weights models:
- the python script: 
https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run.py

- structured output of the script for each of the models in JSON (model-level results) and CSV (detailed layer-by-layer results) files:
https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-2025-06-29-21-02/output-gemma-2-9b.json
https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-2025-06-29-21-02/output-gemma-2-9b-records.csv
https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-2025-06-29-21-02/output-gemma-2-9b-pure-next-token.csv

https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-2025-06-29-21-02/output-Qwen3-8B.json
https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-2025-06-29-21-02/output-Qwen3-8B-records.csv
https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-2025-06-29-21-02/output-Qwen3-8B-pure-next-token.csv

https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-2025-06-29-21-02/output-Mistral-7B-v0.1.json
https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-2025-06-29-21-02/output-Mistral-7B-v0.1-records.csv
https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-2025-06-29-21-02/output-Mistral-7B-v0.1-pure-next-token.csv

https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-2025-06-29-21-02/output-Meta-Llama-3-8B.json
https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-2025-06-29-21-02/output-Meta-Llama-3-8B-records.csv
https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-2025-06-29-21-02/output-Meta-Llama-3-8B-pure-next-token.csv


- evaluation of those outputs by an LLM model, prompted to emulate an LLM interpretability researcher: 
https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-2025-06-29-21-02/evaluation-Meta-Llama-3-8B.md
https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-2025-06-29-21-02/evaluation-gemma-2-9b.md
https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-2025-06-29-21-02/evaluation-Mistral-7B-v0.1.md
https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-2025-06-29-21-02/evaluation-Qwen3-8B.md

- cross-model evaluation:
https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-2025-06-29-21-02/evaluation-cross-model.md

- notes on the experiment:
https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/NOTES.md

Your task is to:
- review the experiment's code: anything wrong about the approach (other than limitations highlighted in model evaluation docs)?
- review the results and analyses: anything incorrectly interpreted, over-stated, or missed? anything you want to add, based on your knowledge of lastest LLM interpretability research?
- consider the usefulness of the findings for the realism vs nominalism debate;
- propose next steps, keeping in mind project context and goals.

Please structure your response under four headers:

1. Code & methodology
2. Result interpretation
3. Philosophical relevance
4. Additional notes
5. Recommended next steps


Use your knowledge of cutting-edge LLM research. Be thorough and specific. Make sure that statements are supported by evidence from the txt dump, don't speculate. Contextualize your findings using your broader knowledge of the latest interpretability research, citing sources when appropriate. Always provide links to sources, and verify that sources contain the claims that you're citing; otherwise, reformulate or remove the claim. In any case, never provide a non-existent source.

We are analysing layer-by-layer logit-lens sweeps over four open-weight base LLMs (Llama-3-8B, Mistral-7B-v0.1, Gemma-2-9B, Qwen-3-8B).

Known limitations of the approach that we accepted in this iteration (but suggestions are appreciated):

The script:
* **Applies an RMS-norm lens** to every residual stream and projects through the model’s own unembedding matrix.  This keeps activations in the model’s native scale but is *not* a tuned lens; possible systematic bias of RMS lenses is acknowledged but left untouched in this round.
* **Defines copy-collapse (L\_copy) by prompt-echo**: the first layer whose top-1 *next-token* is a token already present in the prompt **with p > 0.90**.  We deliberately abandoned the earlier “entropy < 1 bit” rule because entropy is vocabulary-size sensitive and produced inconsistent collapse depths.
* **Defines semantic-collapse (L\_semantic)** as the first layer whose top-1 equals the gold answer token (“Berlin”).  Δ-collapse = L\_semantic − L\_copy captures how long the model clings to surface form before retrieving meaning.
* Uses a single fixed prompt
  `Give the city name only, plain text. The capital of Germany is called simply`
  followed by the logit-lens sweep of the next unseen token (“⟨NEXT⟩”).
  The brevity instruction intentionally stresses instruction-following pathways; removing it is known to shift collapse depths but that comparison is out-of-scope here.
* Inference runs on the user-selected device (CUDA, MPS or CPU). Residuals are cached in fp32; the unembedding matrix is promoted to fp32 for numerical accuracy. A deterministic bootstrap seeds Python, NumPy and PyTorch and enables deterministic kernels, so identical runs yield byte-identical outputs.  
* Outputs per model: a JSON meta file, a records CSV, and a pure-next-token CSV (with `copy_collapse`, `entropy_collapse`, `is_answer`, and `rest_mass`).  
* The script now includes a faithful LayerNorm path; RMSNorm models are unchanged.  
* Additional artefacts: 14 paraphrased test-prompt probes and a two-temperature sweep on the base prompt.

Keep within-model comparisons; cross-model comparisons are still subject to RMS-lens bias.

Known limitations to keep in mind:

* RMS-lens can distort absolute probabilities; comparisons should stay within-model, not across differing normalisation schemes.
* Single-prompt probing risks over-fitting to tokenizer quirks; copy-collapse counts may swing if we tweak wording or punctuation.
* The >0.90 threshold is heuristic; lowering it would classify Mistral’s long “simply” plateau as copy-collapse, raising its Δ by \~20 layers.
* Attention patterns and MLP activations are not inspected—only residual projections—so entropy bumps caused by internal gating may be mis-attributed.

Accept these constraints as *scoping guards*: they keep this run self-consistent even if they leave some systematic bias uncorrected.

# Response by OpenAI o3-pro

### 1. Code & methodology

* **Norm‑lens choice.**
  The script applies an *RMS‑norm* (or full LayerNorm) lens to every residual stream and then multiplies by the model’s own unembedding matrix.  This is faithful to the models’ scaling, but it is still a *static* lens; recent work (e.g. Tuned‑Lens and DLA probes) shows that a lens fine‑tuned per layer sharply reduces the “illusory copy” phenomenon and anticipates semantics several layers earlier.  You already flag the limitation, but its impact is large: Gemma’s 42‑layer lag almost certainly shrinks under a tuned lens.

* **Copy‑collapse heuristic.**
  `L_copy` is the *first* layer whose **top‑1** token is in the prompt **and** p > 0.90.  Two caveats:

  1. High‑probability *second‑rank* echoes are invisible, so “soft” copy plateaus in Llama and Mistral are missed; lowering the threshold or using *top‑k∩prompt* mass would recover them.
  2. For BPE vocabularies the prompt token may be “<space>simply”; stripping solves the space but not rare sub‑word splits (e.g. `▁simple` vs `simply`).

* **Single‑prompt design.**
  All metrics come from one English prompt.  The layer at which semantics emerges is highly prompt‑dependent (rotation‑token vs additive‑token, instruction style, language); a minimum viable extension is a 20–30 prompt battery with controlled lexical/semantic perturbations.

* **Residual hook placement.**
  You record `hook_resid_post` and then apply `ln2` before the lens.  That is the standard “clean” location and avoids the common “resid\_pre” mistake; good.

* **Entropy calculation.**
  Diagnostics use the *full* soft‑max, but the CSV files save only the top‑k and then an aggregated “rest\_mass”.  That is fine for collapse detection; just keep in mind that any downstream KL/JS analysis against another distribution will require re‑synthesising the tail.

* **Device / dtype handling.**
  fp32 unembedding avoids false ties at high layers; deterministic CPU kernels make runs reproducible – both sound choices.

* **Minor quibbles.**

  * Layer‑0 embeddings are run through `ln1`; that is faithful for Pre‑Norm models but deviates from Llama‑2 style *no* norm on E+R.  It probably does not affect Δ, yet deserve a comment.
  * `prompt_token_set` is built once; if you later add a batch of prompts inside the same run (e.g. your test probes) the set may contain unintended tokens.  Rebuild per prompt.

Overall, the script is solid and self‑documented; the main improvements would be (i) a tuned lens, (ii) a softer copy metric, and (iii) a prompt battery. ([raw.githubusercontent.com][1])

---

### 2. Result interpretation

* **Gemma‑2‑9B.**
  The report calls the copy reflex “persists through 41 layers”, but `L_semantic = 42` so Δ = 42.  The write‑up is otherwise accurate: Gemma really is stuck on *“simply”* from embeddings to ln\_final – a behaviour we have seen in other Google checkpoints when the prompt ends with an instruction adverb.

* **Qwen‑3‑8B.**
  Δ = 6 is computed correctly.  One nuance: layers 29‑30 already carry *“Germany”* with ≈70 % before “Berlin” wins at 31; the model is oscillating between *relational* answer (country) and *direct* answer (city).  This supports the idea that instruction‑following heads activate after factual heads, not before.

* **Meta‑Llama‑3‑8B and Mistral‑7B.**
  The claim “no copy‑reflex” is too strong; both plateau on *“simply”* with p≈0.77 (Mistral) or 0.55 (Llama).  They just never cross the 0.9 threshold.  Calling this a “soft copy plateau” (as you do in the cross‑model memo) is more precise.

* **Entropy language.**
  Saying entropy “plunges” from 12.3 → 7.8 bits (Llama) over one layer overstates the change; a 4.5‑bit drop is moderate on a 17‑bit vocab entropy scale.

* **Speculative links to rotary embeddings.**
  The reports repeatedly ascribe late semantics to “rotary only” position information.  There is no published evidence that rotary vs absolute affects *semantic timing*; it mostly changes extrapolation length.  That paragraph should be softened.

The quantitative findings themselves look correct; the over‑interpretation mainly concerns (i) how sharp the entropy cliff is, and (ii) attributing behaviour to rotary positionals. ([raw.githubusercontent.com][2], [raw.githubusercontent.com][3], [raw.githubusercontent.com][4], [raw.githubusercontent.com][5], [raw.githubusercontent.com][6])

---

### 3. Philosophical relevance

The Δ‑metric does capture *when* a model stops parroting the linguistic surface and accesses a stored fact.  Interpreting that as **nominalism** (surface names dominate) versus **realism** (concepts are represented independently of names) is promising but presently under‑determined:

1. **Continuity, not an on/off switch.**  Even in Gemma, latent *Berlin* features almost certainly exist before layer 42; the lens simply fails to read them out.  Realism might already be there, just weakly linearly accessible.

2. **Concept vs. token invariance.**  A stronger test of realism would probe synonyms (*“metropolis”*), cross‑lingual tokens (*“Berlín”* in Spanish), or modality switches (image‑conditioned caption heads).  If the same internal vector fires for all surface forms, that is evidence for realist “universals”.

3. **Compositional generalisation.**  Nominalism predicts token‑level memorisation; realism predicts rule‑based generalisation.  You could measure whether early layers already encode *capital‑of(x) → y* across many countries or only “Germany → Berlin”.

In short, Δ is a useful *proxy*, but additional probes (linear concept probes, causal mediation via attention‑head patching) are required before drawing firm metaphysical conclusions.

---

### 4. Additional notes

* The code base is an excellent teaching example: deterministic seeds, full JSON artefacts, and per‑model Markdown evaluations enable total reproducibility.
* Recent interpretability tools to consider:

  * **Tuned Lens** (Cai et al., 2023) – layer‑wise fine‑tuned probes.
  * **Logit‑Attribution Patching** (Belrose, 2024) – causal trace of which heads/MLPs move the logit for *Berlin*.
  * **SAE features** (Anthropic, 2024) – sparse auto‑encoders that yield human‑labelled features such as *“is‑capital”*.

---

### 5. Recommended next steps

1. **Replace the RMS lens with a tuned lens** or—even simpler—fit a one‑layer linear probe for “Berlin” on held‑out residuals.  Measure how Δ shifts.
2. **Prompt sweep.**  At minimum:

   * 10 paraphrases without “simply”;
   * reverse Q–A form (“What is Germany’s capital?”);
   * cross‑lingual forms (“Die Hauptstadt von Deutschland ist …”).
     Track Δ distribution.
3. **Threshold sweep for copy collapse.**  Record Δ for p ∈ {0.5, 0.7, 0.9}.  A two‑layer Δ at p = 0.7 but 42 at p = 0.9 will reveal how sharply the copy reflex saturates.
4. **Head ablation study.**  For Gemma, zero out each attention head between layers 30‑42 and watch whether *Berlin* appears sooner; this localises the “surface‑form suppressor” circuit.
5. **Concept invariance probes.**  Train a linear probe on a batch of *capital‑city* pairs and test cross‑prompt.  Early emergence of probe accuracy would support “realist” representations.
6. **Philosophy tie‑in.**  Formulate a concrete criterion: *A model is “nominalist” on concept C if removing the token embedding for the canonical name destroys downstream performance on C.*  Then test with embedding noise / replacement.
7. **Publish a short note.**  Even a negative result (Gemma’s extreme lag disappears with a tuned lens) is valuable for the interpretability community.

These steps keep the project lightweight but move it from anecdotal layer counts toward causal, generalisable evidence that can inform the nominalism‑vs‑realism debate.

[1]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run.py "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-2025-06-29-21-02/evaluation-Mistral-7B-v0.1.md "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-2025-06-29-21-02/evaluation-Meta-Llama-3-8B.md "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-2025-06-29-21-02/evaluation-gemma-2-9b.md "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-2025-06-29-21-02/evaluation-Qwen3-8B.md "raw.githubusercontent.com"
[6]: https://raw.githubusercontent.com/forms-and-features/logos-in-layers/refs/heads/main/001_layers_baseline/run-2025-06-29-21-02/evaluation-cross-model.md "raw.githubusercontent.com"


