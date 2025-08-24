# Interpretability Project – Development Notes for AI Assistant

## Philosophical Project Context

**Goal.** Bring concrete interpretability data to the centuries‑old dispute between **nominalism** and **realism** about universals. The first iterations target the low bar: **austere (extreme) nominalism**, which claims that only particular tokens exist and any talk of properties or relations can be paraphrased away.

By showing that LLMs contain robust, reusable internal structures, detected through logit‑lens baselines and causal patches, we aim to gather empirical pressure against that austere view. Once those methods are sound and the anti‑austere evidence is in hand, the project will move to the harder task of discriminating between **metalinguistic nominalism** (which treats those structures as facts about words and predicates) and **realism** (which treats them as evidence of mind‑independent universals).

---

## Provenance & Conventions (read me first)

**Provenance.** Section **1.1**’s normalization fix is merged and active. The current `run-latest` (e.g., `timestamp-20250824-1549`) is **V2 (post‑fix)**. Only archived runs before the fix should be treated as **V1 (pre‑fix)**; label them explicitly when comparing depths or probabilities.
Runs after 2025‑08‑24 also decode logits with an fp32 unembedding when the model computes in bf16/fp16, and compute LN/RMS statistics in fp32 before casting back. This improves numerical stability without changing defaults for ≤27B CPU runs.

**Layer indexing.** We decode **post‑block** unless otherwise stated. Rows `layer = 0 … (n_layers − 1)` are post‑block residuals; `layer = n_layers` is the **final unembed head** (the model’s actual output distribution).

**Positional encoding.** Most models here use rotary position embeddings (RoPE). At layer 0 our “token‑only” wording indicates no additive positional vector; position is injected inside attention via RoPE (cf. RoFormer, arXiv:2104.09864).

**Answer matching is ID‑level.** We determine the gold **first answer token id** from the model’s tokenizer applied to `prompt + " Berlin"` (or control answer). All `is_answer` logic compares **token IDs**, not strings. We log the entire answer tokenisation for transparency (see §1.11).

**Cross‑model caution.** RMS/LN lenses can distort **absolute** probabilities and entropies in model‑specific ways. Only compare **within** a model unless using a Tuned Lens or a shared decoder (Logit Prism). Cross‑model claims should be phrased in terms of **relative** or **rank‑based** metrics (e.g., KL‑to‑final thresholds).

---

# Next steps

Items are ordered by the approximate engineering lift required.

---

## 1. Get the measurement right

Before we can claim that LLMs house structures too systematic for austere nominalism, our probes themselves must be trustworthy. This stage therefore focuses on scrubbing away every obvious source of numerical noise or probe artefact.

### 1.1. Fix the RMS/LN scaling path (γ + ε placement)

**Why.** If you normalise a *different* residual stream (post‑block) with γ that was trained for the *pre‑block* stream, logits are systematically mis‑scaled; early‑layer activations can be inflated by >10×. An incorrect ε outside the square‑root likewise shifts all norms upward. These distortions then propagate through the logit lens, giving spurious “early meaning” or hiding true signal. RMSNorm’s official formula places ε **inside** the √ and multiplies by γ afterwards ([arxiv.org][1]).

**What.** Apply RMS/LN γ and ε to the right residual stream; fix the ε‑outside‑sqrt bug (cf. arXiv:1910.07467).

**How.**

1. Inside `apply_norm_or_skip`:

```python
# compute rms with eps *inside* the sqrt
rms = torch.sqrt(residual.pow(2).mean(-1, keepdim=True) + norm_module.eps)
return residual / rms * scale
```

2. Instead of always pulling `block[i].ln1`, fetch the *next* norm that actually generated the residual being probed:

```python
norm_module = model.blocks[i].ln2 if probe_after_block else model.blocks[i].ln1
```

(Wrap this in a helper; autodetect whether the architecture is Pre‑Norm or Post‑Norm.)
3\) Add a unit test that decodes layer 0 twice: once with γ=1, once with learned γ. The KL between them should match KL between *raw* hidden states with and without γ, proving that scaling now matches semantics.

**✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)**

* All epsilon placement, architecture detection, and validation requirements implemented and unit‑tested.
* Numerical precision policy: When compute dtype is bf16/fp16 (e.g., large CPU runs), unembedding/decoding uses fp32 and LN/RMS statistics are computed in fp32 and cast back. This stabilizes small logit gaps and entropy at negligible memory cost.
* **Provenance note:** the current `run-latest` directory (timestamped 2025‑08‑24) is **post‑fix (V2)**. Only archived outputs predating the fix (or the referenced commit) are **pre‑fix (V1)**; re‑run those models if you need comparable V2 numbers.

---

### 1.2. Sub‑word‑aware copy‑collapse detector

**Why.** String‑level membership can both **miss** prompt‑echoes (multi‑piece tokens; whitespace variants) and **spuriously fire** on substrings (“lin” in “Berlin”). Detecting copy at the **token‑ID level** eliminates these errors and makes `L_copy` robust.

**What.** *Detect prompt echo when the **top‑1 token ID** (or a window of the last *k* top‑1 IDs) appears as a **contiguous subsequence** of the prompt’s token‑ID list **and** `p_top1 > THRESH`.*

**How.**

1. Precompute the prompt’s token IDs once:

   ```python
   ctx_ids = tokenizer.encode(prompt, add_special_tokens=False)
   ```
2. Maintain a rolling window of the last *k* top‑1 IDs (default `k=1`, optional `k∈{1,2,3}`):

   ```python
   window_ids.append(top1_id)
   if len(window_ids) > k: window_ids.pop(0)
   ```
3. Replace string membership with an **ID‑level contiguous subsequence** check:

   ```python
   def is_id_subseq(needle, haystack):
       # return True iff `needle` appears as a contiguous slice of `haystack`
       k = len(needle)
       return any(haystack[i:i+k] == needle for i in range(len(haystack)-k+1))

   collapse = is_id_subseq(window_ids, ctx_ids) and (p_top1 > THRESH)
   ```
4. Expose CLI knobs:

   * `--copy-thresh` (default `0.90`)
   * `--copy-window-k` (default `1`)
5. **Provenance.** Emit to JSON meta:

   ```json
   "copy_thresh": 0.90,
   "copy_window_k": 1,
   "copy_match_level": "id_subsequence"
   ```
6. **Note.** Detokenise only for **reporting** (pretty prints), **not** for detection.

---

### 1.3. Record top‑1 p, top‑5 p\_cumulative, **p\_answer**, **answer\_rank**, and KL‑to‑final

**Why.** Entropy blurs ties; probabilities depend on lens calibration. Adding **rank** of the gold token provides a calibration‑robust signal and clean thresholds (“rank ≤ 5/10”). KL to the final head diagnoses **amplification vs rotation**.

**What.** *Add five columns to every `*-pure-next-token.csv`:*

* `p_top1`, `p_top5` (cumulative), `p_answer`, `kl_to_final_bits`, **`answer_rank`**.
  *Add run‑summary fields:* `first_kl_below_0.5`, `first_kl_below_1.0`, **`first_rank_le_1`**, **`first_rank_le_5`**, **`first_rank_le_10`** (layer indices or `null`).

**How.**

1. Cache final distribution once:

   ```python
   final_probs = final_logits.softmax(dim=-1, dtype=torch.float32)
   ```
2. Per layer:

   ```python
   probs = layer_logits.softmax(dim=-1, dtype=torch.float32)
   p_top1   = probs[top1_id].item()
   p_top5   = probs[torch.topk(probs, 5).indices].sum().item()
   p_answer = probs[first_ans_id].item()
   kl_bits  = torch.kl_div(probs.log(), final_probs, reduction="sum") / math.log(2)
   answer_rank = 1 + (probs > p_answer).sum().item()  # integer rank (1 = top-1)
   ```
3. After the sweep, derive thresholds:

   ```python
   first_rank_le_1  = first_layer_where(answer_rank <= 1)
   first_rank_le_5  = first_layer_where(answer_rank <= 5)
   first_rank_le_10 = first_layer_where(answer_rank <= 10)
   ```
4. Persist all five fields in CSV; write the four summary indices into JSON.

---

### 1.4. Raw‑activation lens toggle

**Why.** If “early meaning” disappears when you skip normalisation, that meaning was an artefact of the lens, not the model.

**What.** *A boolean CLI flag `--raw-lens` that bypasses `apply_norm_or_skip`.*

**How.**

1. Add `parser.add_argument("--raw-lens", action="store_true")`.
2. Wrap the call:

```python
resid_to_decode = residual if args.raw_lens else apply_norm_or_skip(residual, norm_mod)
```

3. Output the flag value into the JSON meta file so that result artefacts are traceable.
4. Optional `--dual-lens N` to emit both raw and norm lens rows for layers `0..N-1` with a `lens` column ∈ {`raw`,`norm`} (debug only).

---

### 1.5. Representation‑drift cosine curve

**Why.** A realist reading predicts an answer‑token direction that exists early and merely grows in magnitude; a nominalist picture predicts the direction rotates into place late. Cosine similarity across depth quantifies which is true.

**What.** *A per‑layer scalar `cos_to_final` written alongside entropy metrics.*

**How.**

1. Compute once:

```python
final_dir = final_logits / final_logits.norm()
```

2. At each layer:

```python
curr_dir = layer_logits / layer_logits.norm()
cos = torch.dot(curr_dir, final_dir).item()
```

3. Write `cos_to_final` column; include in diagnostic plots.

**Note.** Cosine is computed on **logit directions** (not residuals). With a Tuned Lens (§1.9), `cos_to_final` measures closeness to the **tuned** head—still interpretable as “distance to the model’s decision boundary”.

---

### 1.6. Negative‑control prompt

**Why.** If, in the **France** control, the Berlin token outranks Paris, the probe is leaking lexical co‑occurrence. A **margin** makes leakage quantitative and comparable.

**What.** *Run a control prompt (“… The capital of **France** …”) alongside the positive prompt; log a **control margin** and a summary index.*

* Add a per‑layer column (control rows only): **`control_margin = p(Paris) − p(Berlin)`** using ID‑level `first_id` from §1.11.
* Add run‑summary fields: **`first_control_margin_pos = first layer with control_margin > 0`** and **`max_control_margin`**.

**How.**

1. `PROMPTS = [positive_prompt, control_prompt]`; add `prompt_id ∈ {pos, ctl}` to CSV/JSON.
2. For `prompt_id == ctl`, compute `p(Paris)` and `p(Berlin)` per layer and write `control_margin`.
3. After the sweep, compute and store `first_control_margin_pos` and `max_control_margin` in JSON meta.
4. In analysis, flag runs where `first_control_margin_pos` is `null` or late (possible leakage).

---

### 1.7. Ablate stylistic filler (`simply`)

**Why.** Gemma’s early copy‑collapse may be triggered by instruction‑style cues, not semantics. Removing “simply” tests that hypothesis.

**What.** *Rerun the base prompt with and without filler and compare `L_copy`, `L_semantic`.*

**How.**

1. Duplicate prompt; drop the adverb.
2. Record both runs with `prompt_variant` metadata.
3. Plot `Δ-collapse` for the two variants; a large shift confirms the stylistic‑cue explanation.

---

### 1.8. Lightweight CI / regression harness

**Why.** Top‑1 flips are brittle; KL thresholds are **more stable**. CI should guard **metrics and provenance**.

**What.** *A GitHub Actions workflow that executes a dry‑run and asserts:*

* JSON meta contains **schema & provenance** keys:

  * `schema_version`, `code_commit_sha`
  * `gold_answer.first_id`, `gold_answer.pieces`
  * `copy_thresh`, `copy_window_k`
  * lens flags (`use_norm_lens`, `raw_lens`, `use_tuned_lens`, `use_logit_prism`)
  * **if tuned lens is used:** `tuned_lens.{version, sha, train_corpus_id, num_steps, seed}`
* **`first_kl_below_1.0` remains within ±1 layer** of an expected value (store in `expected_meta.json`).
  *(Optionally also check `L_semantic` ±1 for visibility.)*

**How.**

1. Add `--dry-run` to load the model and decode first 5 layers.
2. Commit `expected_meta.json` with `first_kl_below_1.0` and `schema_version`.
3. GH Actions matrix: `{python-version: [3.10], torch: [2.3]}`; fail if any required key is missing or thresholds drift.
4. Print the guarded fields on CI for audit.

---

### 1.9. Integrate a Tuned Lens

**Why.** Tuned Lens compensates for scaling and basis rotation, lowering KL to final and stabilising early‑layer read‑outs (cf. arXiv:2303.08112). **Reproducible provenance** is required.

**What.** *Train once per model; load with `--use-tuned-lens`; record full training provenance.*

**How.**

1. Install and train:

   ```python
   from tuned_lens import TunedLens, train_all_layers
   lens = TunedLens(model)
   train_all_layers(
       lens, tokenizer, text_corpus,
       max_steps=500, lr=1e-4, seed=SEED
   )
   lens.save_pretrained(save_dir)
   ```
2. Load in the sweep (as already planned):

   ```python
   if args.use_tuned_lens:
       lens = TunedLens.from_pretrained(save_dir).to(model.device)
       logits = lens(hidden_states, layer_idx=ℓ)
   else:
       logits = (unembed @ resid)
   ```
3. **Provenance.** Persist a `tuned_lens.json` alongside weights and mirror it into the run meta:

   ```json
   "tuned_lens": {
     "version": "0.6.x",
     "sha": "<model-or-weights-hash>",
     "train_corpus_id": "pile-50k-shards_v1",
     "num_steps": 500,
     "seed": 316
   }
   ```
4. CI (see §1.8) asserts presence of the full `tuned_lens.*` block when `--use_tuned_lens` is set.

---

### 1.10. *(Optional)* Logit Prism shared decoder

**Why.** A single whitening + rotation matrix (`W_prism`) that works for *all* layers makes cross‑layer geometry directly comparable and reduces probe freedom ([neuralblog.github.io][3]).

**What.** *Fit `W_prism` on a slice of hidden states (e.g., 100k tokens, every other layer), save to disk, and expose `--use-logit-prism` which bypasses per‑layer lenses.*

**How.**

1. Sample hidden states `H` size `[N, d_model]`, compute per‑component variance and mean; whiten.
2. Solve for `R` that minimises reconstruction `‖W_U · R · H_i − logits_final‖`. `torch.linalg.lstsq` is fine at 8–9B scale.
3. Save `{mean, var, R}` to a `*.npz`; load at runtime and apply:

```python
logits = W_U @ (R @ (resid - mean) / sqrt(var))
```

4. Add a CLI flag parallel to `--use-tuned-lens`; mutual‑exclusion logic ensures the user picks exactly one decoding scheme.

Note. Logit Prism is currently documented via a non‑archival implementation blog ([3]); treat it as an engineering reference rather than a formal citation.

---

### 1.11. Gold‑token alignment (leading‑space, multi‑piece)

**Why.** Many tokenizers produce a leading‑space piece; string equality silently mismatches. **ID‑level matching** is robust. In multilingual runs, the **gold token(s) vary by language** and must be recorded per‑language.

**What.** *Compute the gold **first answer token id** from `tokenizer(prompt + " Berlin")` (or the language‑specific gold token) and record both the ID and full tokenisation in JSON.* In multilingual runs, also write a `gold_answer_by_lang` block.

**How.**

1. Monolingual:

   ```python
   ans_pieces = tokenizer.encode(" Berlin", add_special_tokens=False)
   ctx_ids    = tokenizer.encode(prompt, add_special_tokens=False)
   ctx_ans    = tokenizer.encode(prompt + " Berlin", add_special_tokens=False)
   first_ans_id = ctx_ans[len(ctx_ids)]
   ```
2. Use `first_ans_id` for `is_answer`, `p_answer`, and `answer_rank` (§1.3).
3. JSON meta:

   ```json
   "gold_answer": { "string": "Berlin", "pieces": ans_pieces, "first_id": first_ans_id }
   ```
4. **Multilingual extension.** When `prompt_lang` is set, compute and store:

   ```json
   "gold_answer_by_lang": {
     "en": { "string": "Berlin",  "pieces": [...], "first_id": 1234 },
     "de": { "string": "Berlin",  "pieces": [...], "first_id": 5678 },
     "es": { "string": "Berlín",  "pieces": [...], "first_id": 9012 },
     "...": { "...": "..."}
   }
   ```

   The analysis code must use the language‑appropriate `first_id` for all `p_answer` and `answer_rank` computations.

---

### 1.12. Metadata completeness

**Why.** Reproducibility and auditability require stable **schema** and **provenance**.

**What.** Add to JSON meta:

* Core: `schema_version`, `code_commit_sha`, `seed`, `device`, `dtype`, `unembed_dtype`
* Lens flags: `use_norm_lens`, `raw_lens` (if dual), `use_tuned_lens`, `use_logit_prism`
* Gold answer: `gold_answer` (and `gold_answer_by_lang` when multilingual)
* Copy parameters: `copy_thresh`, `copy_window_k`, `copy_match_level`
* Layer indexing: `layer_indexing: "post_block"`, `final_row_is_unembed: true`
* **Tuned Lens provenance** (when used): `tuned_lens.{version, sha, train_corpus_id, num_steps, seed}`

**How.** Populate from CLI/derived values at run start; print to console and write into the meta JSON. Bump `schema_version` when keys change.

---

#### Wrap‑up

Executing the items in **Group 1** upgrades the measurement pipeline from an informative prototype to a rigour‑grade toolchain. Only after this foundation is secure should we move on to the broader prompt battery and causal‑intervention work.

[1]: https://arxiv.org/abs/1910.07467 "Root Mean Square Layer Normalization"
[2]: https://arxiv.org/abs/2303.08112 "Eliciting Latent Predictions from Transformers with the Tuned Lens"
[3]: https://neuralblog.github.io/logit-prisms/ "Logit Prisms: Decomposing Transformer Outputs for Mechanistic ..."

---

## 2. Straight‑forward experimental variations on the current design

We run a first wave of low‑overhead variations that reuse the logit‑lens baseline while **adding causal or representational sanity checks wherever those tools are already available**. The purpose is two‑fold:

1. **Finish dismantling austere nominalism.** If a small change in wording or language leaves the same causal layer and vector structure intact, the model’s behaviour cannot be captured by listing concrete token‑tokens alone.
2. **Collect scaffolding for the harder metalinguistic‑nominalism vs realism tests.** Stability (or fragility) across these variations will tell us which relations and properties deserve a deeper causal follow‑up in Group 4.

#### Philosophical background

* **Austere nominalism** says all facts reduce to concrete token occurrences. If our probes keep finding *shared* internal drivers across token changes, that claim weakens.
* **Metalinguistic nominalism** treats any such shared driver as a sophisticated fact *about linguistic predicates themselves*. The experiments below do **not** decide between MN and realism; they only build a reliability map. ([plato.stanford.edu][5])
* **Realism** expects some level of cross‑prompt, cross‑language, or cross‑style invariance once token noise is factored out; large deviations would instead push explanation toward MN. ([plato.stanford.edu][4])

---

### 2.1. Threshold sweep for copy‑collapse

**Why.** Copy‑collapse is meant to record when the network *re‑uses a token already present in the prompt*—i.e., when it relies on **particular** lexical material. If that layer changes drastically when the probability threshold moves from 0.90 to 0.70, the phenomenon is fragile and more consonant with nominalist “name matching” than with an entrenched universal.

**What.** Add boolean columns `copy@0.70`, `copy@0.80`, `copy@0.90` to every `*-pure-next-token.csv`. Emit a short Markdown summary that lists `L_copy(0.70)`, `L_copy(0.80)`, `L_copy(0.90)` for each model (explicitly show `null` when no layer qualifies).

**How.** Compute `p_top1` per layer, evaluate the three inequalities, write booleans, and post an auto‑generated Markdown diff.

---

### 2.2. Multilingual prompt – preliminary pass

**Why.** Language‑independent behaviour is compatible with realism but not mandated by it; language‑dependent depths are prima facie evidence for predicate‑tied behaviour. A **per‑language gold‑token alignment** prevents tokenizer artefacts from polluting comparisons.

**What.** Translate the prompt into five major languages (matched subject–predicate order). Record normalised `L_sem / n_layers`, **`first_rank_le_{1,5,10}`**, and tuned‑lens KL thresholds; visualise variance. Use **ID‑level** gold tokens from `gold_answer_by_lang` (§1.11).

**How.**

1. Maintain a YAML of prompts keyed by ISO codes (`prompt_lang`); include `translation_ok: true/false`.
2. For each language, compute `first_id` and `pieces` and store under `gold_answer_by_lang` (§1.11).
3. Run sweeps; bar‑plot layer‑fraction variance and **rank thresholds**; highlight deviations `> 0.05` (fraction) or delays `> 2` layers in `first_rank_le_5`.
4. Prefer rank/KL‑threshold metrics over raw probabilities for cross‑language comparisons.

---

### Closing note on epistemic modesty

These variations are diagnostic, not decisive. Their job is to show which internal patterns ride above surface token variation and which do not. If the patterns hold, austere nominalism loses more credibility and we have a cleaner target set for the higher‑lift causal, multimodal, and synthetic‑language probes that might separate metalinguistic nominalism from realism in later stages.

### Caution on metrics

Raw “semantic‑collapse depth” (the layer where the gold token first becomes top‑1) is a correlational signal. Before drawing philosophical conclusions, validate any depth‑based claim with at least one causal or representational check (activation patching, tuned‑lens KL, concept‑vector alignment). See Group 3 & 4 tasks.
**Cross‑model caveat.** Absolute probabilities/entropies under a norm‑based lens are **not** comparable across models using different normalisers; use Tuned Lens (§1.9) or Logit Prism (§1.10) for cross‑model comparisons, or prefer rank/KL‑threshold metrics.

[4]: https://plato.stanford.edu/entries/properties/ "Properties — Stanford Encyclopedia of Philosophy"
[5]: https://plato.stanford.edu/entries/nominalism-metaphysics/ "Nominalism in Metaphysics — Stanford Encyclopedia of Philosophy"
[6]: https://plato.stanford.edu/entries/tropes/ "Tropes — Stanford Encyclopedia of Philosophy"

---

## 3. Advanced interpretability interventions

These tools move us beyond descriptive logit‑lens curves. They intervene directly in the computation graph so we can ask which internal components are necessary or sufficient for a factual prediction. That causal angle already strains austere nominalism (which would have to re‑paraphrase the interventions themselves) and lays the groundwork for later stages that try to tease apart metalinguistic nominalism from realism.

### 3.1. Layer‑wise activation patching (“causal tracing”)

**Why.** Causal flips show when enough information to force the answer is present. Splitting by **sublayer** (Attention vs MLP) around `L_sem` distinguishes **retrieval** from **construction** (cf. Geva et al., arXiv:2012.14913).

**What.** *Given a prompt pair (clean, corrupted), produce a CSV of “causal Δ log‑prob” per layer for **three modes** — `full` (standard residual patch), `attn_only`, `mlp_only` — and record:*

* `causal_L_sem` (full), **`causal_L_sem_attn`**, **`causal_L_sem_mlp`**
* **`delta_causal = causal_L_sem − L_semantic`**, plus **`delta_causal_attn`**, **`delta_causal_mlp`**

**How.**

1. Implement:

   ```python
   def patch_layer_full(h_clean, h_corr, ℓ): ...
   def patch_layer_attn_only(h_clean, h_corr, ℓ): ...
   def patch_layer_mlp_only(h_clean, h_corr, ℓ): ...
   ```

   Each returns patched hidden states at layer ℓ.
2. For each ℓ and **mode ∈ {full, attn_only, mlp_only}**:

   * Run forward with the patched stream,
   * **Decode with the same lens** as the baseline (Tuned Lens or Prism),
   * Log Δ log‑prob of the gold token (ID from §1.11).
3. Define `causal_L_sem*` as the earliest ℓ where the top‑1 flips to the gold token under that mode.
4. Write `causal_L_sem*` and **delta fields** into JSON meta; include all three per‑layer Δ values in the CSV (columns `dlogp_full`, `dlogp_attn`, `dlogp_mlp`).
5. CLI:

   * `--patching`
   * `--patching-mode {full,attn,mlp,all}` (default `all`)
   * `--corrupted-answer "Paris"`

---

### 3.2. Attention‑head fingerprinting near L\_sem

**Why.** A head that systematically links “Germany” to “Berlin” across prompts and languages suggests a dedicated mechanism. That concreteness challenges the idea that all structure is just diffuse word‑statistics, yet MN can still say the head embodies a predicate rule. Isolating the head is therefore a prerequisite for the stronger MN‑vs‑realism tests in Group 4 ([arxiv.org][9], [neelnanda.io][10]).

**What.** *Catalogue all heads in layers `L_sem − 2 … L_sem` for which:*

* `attn_weight ≥ top‑k(0.8 quantile)` across heads for that layer, and
* Zero‑ablation of the head drops answer log‑prob by ≥ 0.5 bits.
  Store a JSON manifest `relation_heads.json` listing `(layer, head)` tuples for every model.

**How.**

1. Hook attention weights in the forward pass; identify subject and candidate answer positions.
2. Compute head‑specific importance by zeroing its output vector and re‑running the remainder of the model.
3. Save heads meeting both criteria; visualise with a simple heat map.
4. Optional: run CHG (Causal Head Gating) to refine head attribution ([arxiv.org][9]).
5. **Fix random seeds** for zero‑ablation order and batch selection; emit `relation_heads.json` with the seed and `model_sha`.

---

### 3.3. Concept‑vector extraction via Causal Basis (CBE)

**Why.** Belrose et al. show a low‑rank subspace can *causally* steer the model’s logits ([arxiv.org][11]). If a low‑rank vector learned in one context reliably boosts the correct capital in unseen prompts, that shows the model stores a portable shard of “capital‑of” information — already more structure than austere nominalism predicts. Whether this portability counts against metalinguistic nominalism, or is fully compatible with it, cannot be settled here; the result simply gives us a concrete target for the follow‑up tests in Group 4 that are designed to probe that distinction (see also Elhage et al., “Toy Models of Superposition,” arXiv:2209.10652).

**What.** *Deliver a PyTorch module `CapitalDirection` with weights `{U, Σ}` such that adding `α · U Σ v` (for a learned v) to the residual stream at layer `L_sem` reliably increases the log‑prob of the correct capital across ≥ 80% of country prompts, while minimally disrupting unrelated outputs.*

**How.**

1. Sample 1,000 (country, capital) prompts.
2. Use the tuned lens to get layer‑ℓ logits; fit CBE on those activations to identify vectors that maximise Δ p(answer).
3. Freeze the top‑k singular directions; test generalisation on held‑out prompts.
4. Implement `apply_patch(resid, strength)` to inject the vector in new contexts.

---

### 3.4. Attribution patching for scalable causal maps

**Why.** Full activation‑patch grids scale O(L²) runs; attribution patching (gradient‑based approximation) gets the entire layer×token causal heat‑map from *three* passes ([neelnanda.io][12]). This enables causal tracing over the entire WikiData battery without prohibitive compute. Scaling causal maps to thousands of prompts lets us check whether causal responsibility clusters in a few modules or is smeared everywhere. Tight clustering adds tension for nominalist readings that lean heavily on token‑level variance.

**What.** *A script `attribution_patch.py` that, for a batch of prompts, outputs an HDF5 tensor `attr[L, T]` of estimated causal contributions for every layer L and token position T, plus a notebook that plots token‑level heat‑maps.*

**How.**

1. Implement the three‑pass protocol: clean forward, corrupted forward, backward pass on KL divergence.
2. Cache residuals and gradients; compute attribution scores per layer/token.
3. Validate against explicit patching on a 10‑prompt subset (correlation > 0.9).
4. Integrate into CI to run nightly on a 100‑prompt sample.

---

### 3.5. Cross‑model concept alignment (CCA / Procrustes)

**Why.** Convergent geometry across checkpoints trained on different seeds suggests architecture‑level constraints. That is hard to square with austere nominalism’s token‑listing strategy, though MN can still treat it as convergence on shared predicate statistics. Either way, the alignment gives us a common space to compare later multimodal tests ([arxiv.org][13]).

**What.** *Produce an analysis notebook `concept_alignment.ipynb` that:*

1. Collects layer‑`L_sem` activations for the token “Berlin” in the same prompt across all ten models.
2. Performs CCA or orthogonal Procrustes alignment to a shared 128‑D space.
3. Reports average inter‑model cosine similarity before vs after alignment and visualises clusters.

**How.**

1. Dump 10k activation vectors per model to disk.
2. Use `sklearn.cross_decomposition.CCA` (or `emalign` for Procrustes) to learn mappings.
3. Evaluate: if mean pairwise cosine ≥ 0.6 pre‑alignment, geometry is already convergent; if it jumps only post‑alignment, differences are mostly rotational. Interpret results in the accompanying markdown narrative.

---

### 3.6. (Optional) Causal scrubbing of candidate circuits

**Why.** Causal scrubbing replaces multiple intermediate signals at once to test entire hypothesised *circuits* for necessity and sufficiency. If a minimal circuit passes, the burden shifts to MN to reinterpret that circuit linguistically; failure would instead counsel caution against premature realist readings.

**What.** Encode a circuit hypothesis (subject‑head → MLP → answer) in a Python spec and automatically test all 2ᴺ subsets of components, outputting a table of accuracy drops.

**How.**

1. Adopt the open‑source `causal-scrubbing` library.
2. Write a spec file mapping nodes to model components.
3. Run exhaustive subset ablations on a 100‑prompt subset; visualise results as a lattice diagram.

---

### 3.7. *(Optional)* Targeted sparse autoencoders on decisive layers

**Why.** Low‑rank concept bases (CBE) identify **subspaces**; sparse autoencoders (SAEs) can surface **discrete, sparse features** within those subspaces that admit **naming and causal tests** across prompts/languages. A small, targeted SAE on the decisive layer(s) provides feature‑level objects to test portability and causality — stronger evidence against austere nominalism (cf. arXiv:2409.14507; arXiv:2402.12201; arXiv:2209.10652).

**What.** *Train a small SAE on residuals at `L_sem ± 1` for one 7–9B model; identify features whose activation **predicts and causally increases** the correct capital across prompts and in ≥3 languages; ship a minimal feature manifest.*

**How.**

1. **Data.** Collect residuals at `L_sem − 1, L_sem, L_sem + 1` over \~50k tokens drawn from country–capital prompts (and their multilingual variants).
2. **Model.** Train a top‑k or L1‑regularised SAE with 8–16× overcompleteness:

   ```python
   sae = SparseAutoencoder(d_model, d_hidden=16*d_model, sparsity='topk', k=64)
   sae.fit(residual_batch, epochs=2, seed=SEED)
   ```
3. **Screening.** For each learned feature:

   * compute correlation with `p_answer` at `L_sem`,
   * patch **feature ablation** (zero its coeff) and **feature activation** (+α along its decoder) and log Δ log‑prob of the gold token across a held‑out prompt set and ≥3 languages.
4. **Success criterion.** ≥1 feature with median **Δ log‑prob ≥ 0.5 bits** on activation and ≤0 on unrelated tokens in ≥70% of held‑out prompts; portability holds across languages.
5. **Outputs.**

   * `sae_config.json` (arch, k/λ, seed),
   * `sae_features.npz` (enc/dec),
   * `feature_manifest.json` entries: `{feature_id, sparsity, name (optional), Δ_logp_median, languages_passed}`.
6. **Scope.** Gate behind stability of §1.8 and availability of tuned/prism lens; keep to a single model/layer tranche to avoid scope creep.

---

## Philosophical pay‑off

* **Against austere nominalism.** Portable vectors, decisive heads, and convergent circuits all show regularities that outstrip any list of concrete token occurrences.
* **Setting the stage for metalinguistic nominalism vs realism.** By localising the drivers (vectors, heads, circuits) we create objects that MN can still call “sophisticated predicate routines” and realism can call “instantiated universals.” The follow‑up experiments in Group 4 are designed to stress‑test which story explains them more economically.
* **Methodological upgrade.** Manipulative evidence—patching, ablation, scrubbing—moves us from observational claims (“the logit went up”) to counterfactual ones (“if this head were silent, the answer would change”). Those counterfactuals are what philosophical theories must now accommodate.

[8]: https://arxiv.org/abs/2202.05262 "Locating and Editing Factual Associations in GPT"
[9]: https://www.arxiv.org/pdf/2505.13737 "[PDF] A Framework for Interpreting Roles of Attention Heads in Transformers"
[10]: https://www.neelnanda.io/mechanistic-interpretability/glossary "A Comprehensive Mechanistic Interpretability Explainer & Glossary"
[11]: https://arxiv.org/abs/2303.08112 "Eliciting Latent Predictions from Transformers with the Tuned Lens"
[12]: https://www.neelnanda.io/mechanistic-interpretability/attribution-patching "Attribution Patching: Activation Patching At Industrial Scale"
[13]: https://arxiv.org/html/2310.12794v2 "Are Structural Concepts Universal in Transformer Language Models …"

---

## 4. Consolidating the Case Against Austere Nominalism

Austere (extreme) nominalism says *every apparent regularity reduces to a list of concrete token‑tokens* — no predicates, no properties, no relations 〖SEP‑Nominalism〗〖Loux‑2023〗. The Group 3 tools (tuned lens, activation patching, head fingerprinting, concept vectors) are designed to test whether LLMs in fact contain reusable, portable structures that would resist such a paraphrase. If the experiments below confirm that hunch, austere nominalism loses its footing; if they do not, the debate stays open.

### 4.1. Instruction Words vs Capital‑Relation (Causal Check)

**Why.** If deleting “please” or “simply” leaves the causal layer and KL inflection unchanged, the capital‑relation circuit is insensitive to those extra tokens, contradicting austere nominalism’s token‑paraphrase strategy.

**What.** Run the original prompt and a “plain” prompt; record

* (a) the tuned‑lens KL‑curve inflection,
* (b) **causal `L_sem`** obtained by single‑layer activation patching, and
* (c) Δ log‑prob when the top two “style heads” (found via head fingerprinting) are zeroed.

**How.**

1. Generate both prompts; tag `variant=instruct/plain`.
2. For each, sweep layers; patch the corrupted prompt at ℓ until the answer flips; store causal `L_sem`.
3. During the clean run, zero candidate style heads and measure answer log‑prob drop.
4. Summarise: `Gemma‑9B — causal L_sem unchanged (45→45); style‑head ablation −0.1 bits ⇒ semantics robust to pragmatics.`

---

### 4.2. Paraphrase Robustness

**Why.** Ten English rewrites that keep predicate content but change wording. Stable causal `L_sem` and aligned concept vectors across them show a structure deeper than any one token string.

**What.** Ten English paraphrases. For each:

* (a) causal `L_sem`,
* (b) cosine similarity of the answer‑logit direction to the canonical Berlin vector after whitening,
* (c) tuned‑lens KL divergence at `L_sem`.
  Visualise variance; compute coefficient of variation (CV) of causal `L_sem`.

**How.**

1. Store paraphrases in YAML.
2. Batch‑run; cache residuals for concept‑vector whitening.
3. Use the concept‑vector module to obtain Berlin direction per paraphrase; compute cosines.
4. Plot violin of causal `L_sem`; print `CV = 0.06` (low) or `CV = 0.32` (high).

---

### 4.3. Multilingual Consistency (Text‑only Pass)

**Why.** If the same causal layer appears in German, Spanish, Arabic, etc., the relation transcends a single token inventory. That strains austere nominalism yet remains interpretable by metalinguistic nominalism (MN).

**What.** Five language versions of the prompt. Measure:

* (a) tuned‑lens KL inflection layer,
* (b) cosine between each language’s Berlin vector and the English one *after the language‑specific whitening transforms*,
* (c) causal `L_sem`.

**How.**

1. Verify translations keep subject–predicate order.
2. Extract concept vectors; apply whitening per language.
3. Compute pairwise cosines; output a short Markdown table of `⟨cos⟩ = 0.71 ± 0.05` or similar.
4. Flag languages whose causal `L_sem` deviates > 10% of depth.

---

### 4.4. Large WikiData Battery with Causal L\_sem

**Why.** A relation that generalises across 1,000 country–capital pairs is hard to restate as token‑lists. If token length and frequency fail to predict causal depth, austere nominalism loses more ground.

**What.** 1,000–5,000 (country, capital) prompts. For each: causal `L_sem`, answer token length, frequency. Output:

* Histogram of causal `L_sem`,
* OLS regression `L_sem ∼ len + log_freq`.

**How.**

1. Use activation patching in batched mode (two passes per prompt: clean & patch grid).
2. Compute causal `L_sem` for each.
3. Fit regression; print `R²`.
4. Store results in `battery_capital.csv`.

---

### 4.5. Lexical Ambiguity Stress Test

**Why.** Ambiguous names multiply particulars sharing one string. If entropy stays high and relation heads fire later only for ambiguous cases, that shows the model is doing sense‑resolution, which a bare token list cannot capture.

**What.** 50 ambiguous vs 50 control prompts. Metrics:

* (a) entropy plateau height (mean entropy over layers before causal `L_sem`),
* (b) first‑firing layer of the dominant relation head (from fingerprinting).
  Statistical test: Wilcoxon on each metric.

**How.**

1. Curate ambiguous list (“Georgia”, “Jordan”).
2. Run sweeps with attention recording.
3. Detect dominant head per prompt (`attn_weight > 0.2`).
4. Compute layer index; perform paired non‑parametric test; print p‑values.

---

### 4.6. Instruction‑Style Grid with Causal Metrics

**Why.** Checks if speech‑act markers shift causal semantics. Minimal shifts push further against token‑dependence.

**What.** 12 prompt styles (4 modifiers × 3 moods) run over the WikiData battery. For each cell:

* mean causal `L_sem`,
* mean log‑prob drop when style heads are ablated,
* mean tuned‑lens KL at `L_sem`.
  Heat‑map the three statistics.

**How.**

1. Auto‑generate prompt grid.
2. Batch activation patching; reuse style‑head list.
3. Aggregate per cell; render three matplotlib heat‑maps.

---

### Tally of Austere‑Nominalism Pressure

After the above, we will have: *portable concept vectors*, *head‑level causal circuits*, and *cross‑prompt and cross‑language invariance*, all of which resist reduction to token enumeration. This effectively **clears the ground** so later work can focus on MN vs realism.

---

## 5. First Probes at Metalinguistic Nominalism vs Realism (and a Trope Check)

*Metalinguistic nominalism (MN)* treats any internal regularity as a fact about the model’s predicate vocabulary rather than a mind‑independent universal 〖SEP‑Nominalism〗〖Brandom‑2000〗. *Trope theory* replaces universals with many particularised property‑instances (tropes) that resemble one another 〖SEP‑Tropes 2023〗. The experiments below look for patterns that strain an MN paraphrase or favour a trope interpretation, and where a realist story might do better. They remain speculative; negative or ambiguous results will still be philosophically useful.

### 5.1. Vector Portability Across Modalities

**Why.** If a capital‑of vector learned from text alone also raises the right city name in a vision‑language model when shown a map, the underlying pattern is not tied to any specific word‑token. That stretches MN, whose story centres on language, more than a realist reading. (If the vector fails to port, the result remains compatible with both MN and trope theory.)

**What.** Fine‑tune Llava‑1.6 on the same prompt; patch the text‑only vector at `L_sem` during multimodal inference; measure Δ log‑prob of the correct answer.

**How.** Extract vector from text checkpoint, inject into Llava’s language head, record success rate.

---

### 5.2. Synthetic Mini‑Language Swap

**Why.** MN predicts that changing every occurrence of the predicate token (“capital”) to a nonsense token (“blork”) should license the model to build a new, potentially different circuit, because the linguistic anchor has changed. A realist would expect the model to reconstruct a similar geometry for the underlying concept, merely keyed to a new embedding. Trope theory is agnostic: it allows many similar—but non‑identical—instantiated circuits. Measuring geometric overlap therefore places the explanatory burden on whichever view ends up with the more complex paraphrase.

**What.** Create a synthetic corpus with systematic token swap; fine‑tune Qwen‑3‑8B; rerun head fingerprinting and concept extraction.

**How.** Corpus generation script, LoRA fine‑tune, repeat fingerprints, compare vectors via Procrustes.

---

### 5.3. Statistical Scrambling Test

**Why.** Counter‑factually shuffle surface co‑occurrence while keeping underlying relations intact (Levinstein 2024). If capital‑vectors survive, they are not mere word‑statistics.

**What.** Generate a scrambled dataset where country and capital tokens never co‑occur in the same sentence; probe whether the original vector still pushes “Berlin” when patched in.

**How.** Data augmentation, re‑train small model, perform activation patch with original vector, log Δ.

---

### 5.4. Zero‑Shot Novel Syntax

**Why.** Hold out a rare syntactic frame (“Of Germany the capital is \_\_\_”) during training. If relation heads fire correctly on first exposure, they encode more than learned predicate strings.

**What.** Create held‑out eval prompts; record causal `L_sem` and answer accuracy.

**How.** Fine‑tune model with frame removed, evaluate, compare depths.

---

### 5.5. Cross‑Model Convergence After Token Swap

**Why.** If two models trained on disjoint corpora converge to similar relation heads, that hints at architecture‑level universals beyond shared predicates.

**What.** Train Mistral‑7B on Wikipedia vs Common Crawl subsets; run head fingerprinting; measure overlap of head coordinates after alignment.

**How.** Training scripts, CCA alignment, cosine similarity histogram.

---

### 5.6. Trope‑Sensitivity Probe

**Why.** Trope theory expects each context to instantiate its *own* “blackness” or “capital‑of” trope. If concept vectors extracted in different sentences diverge significantly and fail to transfer causally, that supports a trope interpretation; tight clustering and high transferability favour realism or MN.

**What.** Fifty noun‑adjective pairs (“black pawn”, “black asphalt”, …). For each context:

* extract a blackness vector with CBE;
* measure cosine dispersion across contexts;
* patch each vector into every other context and log Δ log‑prob for the adjective “black”.

**How.**

1. Run CBE per sentence at `L_sem`.
2. Compute pairwise cosines; report mean and standard deviation.
3. Patch vectors cross‑context; if median Δ log‑prob > 0.5 bits in ≥ 70% of cases, portability is high (anti‑trope); otherwise low portability supports a trope reading.

---

*Fictionalism*, which treats all universal talk as useful but literally false, can in principle accommodate any of the outcomes; strong results will therefore be framed in terms of **explanatory indispensability** rather than outright refutation 〖SEP‑Fictionalism 2021〗.

---

### Implementation Dependencies for Sections 4 & 5

* Tuned or Prism lens for logits and KL curves
* Validated activation patching (unit: causal `L_sem` within ±1 layer of probe for 95% of prompts)
* Head fingerprinting and concept‑vector modules
* Multimodal patching wrappers (Section 5.1)
* Data‑generation utilities for synthetic corpora and scrambling

---

### References

* SEP‑Nominalism — **“Nominalism in Metaphysics,”** *Stanford Encyclopedia of Philosophy* (2023).
* SEP‑Tropes — **“Tropes,”** *Stanford Encyclopedia of Philosophy* (2023).
* SEP‑Fictionalism — **“Fictionalism,”** *Stanford Encyclopedia of Philosophy* (2021).
* Loux‑2023 — Michael J. Loux, *Metaphysics*, 4th ed., Routledge (2023).
* Brandom‑2000 — Robert B. Brandom, *Articulating Reasons: An Introduction to Inferentialism*, Harvard UP (2000).
* RMSNorm — Zhang & Sennrich, “Root Mean Square Layer Normalization,” arXiv:1910.07467 (2019).
* RoFormer — Su et al., “RoFormer: Enhanced Transformer with Rotary Position Embedding,” arXiv:2104.09864 (2021).
* Superposition — Elhage et al., “Toy Models of Superposition,” arXiv:2209.10652 (2022).
* FFN as KV — Geva et al., “Transformer Feed‑Forward Layers Are Key‑Value Memories,” arXiv:2012.14913 (2020).
* Tuned Lens — Belrose et al., “Eliciting Latent Predictions from Transformers with the Tuned Lens,” arXiv:2303.08112 (2023).
* Logit Prisms — “Logit Prisms: Decomposing Transformer Outputs for Mechanistic …” (implementation blog), https://neuralblog.github.io/logit-prisms/.
* SAE Absorption — Chanin et al., “A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders,” arXiv:2409.14507 (2024).
* Dictionary Learning (Othello‑GPT) — He et al., “Dictionary Learning Improves Patch‑Free Circuit Discovery in Mechanistic Interpretability: A Case Study on Othello‑GPT,” arXiv:2402.12201 (2024).
* Monet — Park et al., “Monet: Mixture of Monosemantic Experts for Transformers,” arXiv:2412.04139 (2024).

---

# Audience

* Software engineer, growing ML interpretability knowledge
* No formal ML background but learning through implementation
* Prefers systematic, reproducible experiments with clear documentation
