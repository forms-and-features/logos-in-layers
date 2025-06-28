# CROSS-EVAL

## 1. Result synthesis

Across the four 7–9 B-parameter decoder stacks we observe a shared pattern: entropy remains high for roughly the first two-thirds of the network, then collapses sharply onto the factual answer "Berlin", followed by a modest rebound after the unembed projection.

Layer index of collapse differs systematically by architecture.  In the two smaller RMS-norm models the collapse occurs in block 25 (≈ 78 % depth) as shown in
```41:48:001_layers_and_logits/evaluation-Meta-Llama-3-8B.md
| **L 25** | **0.43** | **' Berlin'** |
```
and
```43:47:001_layers_and_logits/evaluation-Mistral-7B-v0.1.md
| **L 25** | **0.56** | **'Berlin'** |
```
Qwen3, which has the same nominal width but divergent tokenizer and attention recipe, collapses slightly later at block 28
```38:44:001_layers_and_logits/evaluation-Qwen3-8B.md
| **L 28** | **0.43** | 'Berlin' |
```
whereas Gemma-2-9B defers the transition to block 35
```45:49:001_layers_and_logits/evaluation-gemma-2-9b.md
| L 35 | 0.235 | 'Berlin' |
```
This ordering mirrors the "longer to decide" trend reported in tuned-lens studies (arXiv:2303.08112) whereby architectures with deeper blocks or wider feed-forward modules defer semantic commitment.

The final softmax on the answer position re-introduces ≈ 1.7–2.0 bits of entropy in every model despite high certainty inside the stack, e.g.
```13:17:001_layers_and_logits/output-Meta-Llama-3-8B.json
"entropy": 1.7041
```
```13:17:001_layers_and_logits/output-Qwen3-8B.json
"entropy": 2.0197
```
which supports the "unembed temperature" effect noted in prior work (arXiv:2310.00430).

Qualitatively, two distinct pre-collapse regimes appear.  Gemma and Qwen exhibit early punctuation bias ("colon-spam") with near-zero entropy ≤ layer 8
```14:22:001_layers_and_logits/evaluation-gemma-2-9b.md
| L 0 | 0.000 | ':' |
```
whereas Llama-3 and Mistral stay multilingual or gibberish until about layer 20, after which both models pass through an abstract "capital" phase (8–10 bits) before converging on _Berlin_.  The shared intermediate prior suggests that an abstract geopolitical concept head (similar to the "capital-city" circuitry in GPT-2, see arXiv:2211.00593) is reused across open-weights models.

Auxiliary prompts in JSON corroborate directionality asymmetry: every model is far more certain on the cloze "Berlin is the capital of" (≈ 0.9 bits) than on "Germany's capital is" (> 5 bits), matching causal-attention findings in Llama-Lens.

## 2. Misinterpretations in existing EVALS

* `evaluation-Qwen3-8B.md` L54–61 labels the early multilingual noise as "colon-spam".  The CSV shows the top-1 tokens are Japanese and API-style strings, not ":" (see first 10 rows of `output-Qwen3-8B-records.csv`); the term is therefore mis-applied.
* `evaluation-gemma-2-9b.md` L17–25 rounds early entropies to **0.000** bits.  The CSV gives small but non-zero values (e.g. 7 × 10⁻²⁹ bits), so claiming perfect certainty overstates the effect.
* The same Gemma EVAL (L40–48) lists layer 42 entropy as "0.0000" yet simultaneously cites a 2-bit rebound in JSON; the table omits the unembed step and therefore mixes two different layers.

## 3. Usefulness for the Realism ↔ Nominalism project

The staggered collapse points offer a natural manipulation variable: if realism predicts an internal "concept" crystallising independent of lexical identity, we can test whether activation-patching _before_ the nominal layer (25–35) transfers across paraphrases.  The consistent rebound after unembed further suggests a late-stage nominal re-expansion; intervening on `model.unembed.W_U` while keeping residual fixed could isolate whether diversity is injected lexically or conceptually.

The intermediate "capital" prior common to Llama-3/Mistral provides a candidate subspace for nominalism: a token-agnostic slot that later binds to a specific city.  Probing that subspace for other country names may reveal whether it represents abstract roles or is merely a lexical mixture.

## 4. Limitations

The probe exercises a single prompt hard-coded at
```158:162:001_layers_and_logits/run.py
prompt = "Question: What is the capital of Germany? Answer:"
```
and therefore cannot distinguish representation generality from prompt-specific memorisation.  All runs use one forward pass on Apple-M-series `mps`; cross-device numerical drift is unmeasured.  Entropy is read through an RMS-lens (`USE_NORM_LENS = True`, L143) whose calibration is unverified for each norm variant, and only the top-20 logits are stored, hiding tail mass.  Finally, FP32 unembed is enabled (L151) which alters the final distribution, so comparisons to raw model decoding must control for that intervention.

Produced by OpenAI o3
