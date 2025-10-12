# Interpretability Project – Development Notes for AI Assistant

## Philosophical Project Context

**Goal.** Bring concrete interpretability data to the centuries‑old dispute between **nominalism** and **realism** about universals. The first iterations target the low bar: **austere (extreme) nominalism**, which claims that only particular tokens exist and any talk of properties or relations can be paraphrased away.

By showing that LLMs contain robust, reusable internal structures, detected through logit‑lens baselines and causal patches, we aim to gather empirical pressure against that austere view. Once those methods are sound and the anti‑austere evidence is in hand, the project will move to the harder task of discriminating between **metalinguistic nominalism** (which treats those structures as facts about words and predicates) and **realism** (which treats them as evidence of mind‑independent universals).

---

## Provenance & Conventions (read me first)

**Provenance — date‑based, no manual edits.**
- `run-latest/` is a moving pointer to the most recent sweep; do not cite it directly.
- On each new run, the previous `run-latest/` is rotated to a stable folder `run-YYYYMMDD-HHMM/` (see `layers_core/run_dir.py`). Cite this folder name as the `run_id`.
- Always cite `run_id` (folder name) and `code_commit_sha` from the JSON meta when reporting numbers. If `schema_version` exists, you may include it, but prefer date‑based labels.
- §1.1 normalization fix cut‑over: merged on 2025‑08‑24 (UTC). Any archived run whose `run_id` date is strictly before 2025‑08‑24 is **pre‑§1.1**; runs on or after 2025‑08‑24 are **with‑§1.1**. Use these labels rather than version numbers.
- Future fixes (e.g., §1.2) will add a new cut‑over date here; until then, the rule above suffices.

**Precision policy.** Runs after 2025‑08‑24 decode logits with an fp32 unembedding when compute is bf16/fp16, and compute LN/RMS statistics in fp32 before casting back. This improves numerical stability without changing defaults for ≤27B CPU runs.

**Layer indexing.** We decode **post‑block** unless otherwise stated. Rows `layer = 0 … (n_layers − 1)` are post‑block residuals; `layer = n_layers` is the **final unembed head** (the model’s actual output distribution).

**Positional encoding.** Most models here use rotary position embeddings (RoPE). At layer 0 our “token‑only” wording indicates no additive positional vector; position is injected inside attention via RoPE (cf. RoFormer, arXiv:2104.09864).

**Answer matching is ID‑level.** We determine the gold **first answer token id** from the model’s tokenizer applied to `prompt + " Berlin"` (or control answer). All `is_answer` logic compares **token IDs**, not strings. We log the entire answer tokenisation for transparency (see §1.7).

**Cross‑model caution.** RMS/LN lenses can distort **absolute** probabilities and entropies in model‑specific ways. Only compare **within** a model unless using a Tuned Lens or a shared decoder (Logit Prism). Cross‑model claims should be phrased in terms of **relative** or **rank‑based** metrics (e.g., KL‑to‑final thresholds).

---

# Next steps

Items are ordered by the approximate engineering lift required.

---

## 1. Get the measurement right

Before we can claim that LLMs house structures too systematic for austere nominalism, our probes themselves must be trustworthy. This stage therefore focuses on scrubbing away every obvious source of numerical noise or probe artefact.

### 1.1. [x] Fix the RMS/LN scaling path (γ + ε placement)

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
* **Provenance note:** the current `run-latest` (timestamped 2025‑08‑24) is **with‑§1.1**. Only archived outputs with `run_id` dates before 2025‑08‑24 are **pre‑§1.1**; re‑run those models if you need comparable with‑§1.1 numbers.

---

### 1.2. [x] Sub‑word‑aware copy‑collapse detector

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

**✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)**

---

### 1.3. [x] Record top‑1 p, top‑5 p\_cumulative, **p\_answer**, **answer\_rank**, and KL‑to‑final

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

**✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)**

* Pure next‑token CSV now includes: `p_top1`, `p_top5`, `p_answer`, `kl_to_final_bits`, `answer_rank`.
* Diagnostics JSON includes summary thresholds: `first_kl_below_0.5`, `first_kl_below_1.0`, `first_rank_le_1`, `first_rank_le_5`, `first_rank_le_10`.
* KL is computed as KL(P_layer || P_final) in bits via a centralized helper `layers_core.numerics.kl_bits`.
* Probability/rank metrics are factored in `layers_core.metrics.compute_next_token_metrics` to avoid duplication in `run.py`.

---

### 1.4. [x] Raw‑vs‑Norm dual‑lens sanity (baked‑in)

**Why.** If “early meaning” disappears when you skip normalisation, that meaning was an artefact of the lens, not the model. This check should run repeatedly, not as a one‑off toggle, to guard against regressions and model‑family differences (pre‑ vs post‑norm; future Tuned‑Lens, etc.).

**What.** Baked‑in, low‑cost sanity comparisons between the normalized lens and the raw activation lens at a few sampled depths. Record divergence metrics in JSON; avoid growing the public CLI surface.

**✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)**

**How.**

1. At each sampled layer L ∈ {0, ⌊n/4⌋, ⌊n/2⌋, ⌊3n/4⌋}:
   - Capture `resid_raw` before normalization; compute `logits_raw` and `P_raw` for the pure next‑token (last position only).
   - Apply the correct normalizer (per §1.1) to get `resid_norm`; compute `logits_norm` and `P_norm`.
2. Compute per‑layer divergence metrics:
   - `kl_norm_vs_raw_bits = KL(P_norm || P_raw)` (bits)
   - `top1_agree` (bool)
   - `p_top1_norm`, `p_top1_raw`
   - `p_answer_norm`, `p_answer_raw`, `answer_rank_norm`, `answer_rank_raw` (when gold first‑token id is known; see §1.3)
3. Persist into JSON under `raw_lens_check.samples` with entries:
   ```json
   {"layer": L, "kl_norm_vs_raw_bits": x, "top1_agree": true/false,
    "p_top1_norm": ..., "p_top1_raw": ..., "p_answer_norm": ..., "p_answer_raw": ...,
    "answer_rank_norm": ..., "answer_rank_raw": ...}
   ```
4. Add run‑summary fields:
   - `first_norm_only_semantic_layer` = first L where `is_answer_norm` is true and `is_answer_raw` is false
   - `max_kl_norm_vs_raw_bits` over sampled layers
   - `lens_artifact_risk` ∈ {`low`,`medium`,`high`} via simple heuristics (tune empirically; e.g., `low` < 0.5 bits, `high` ≥ 1.0 bits or any early norm‑only semantics)

**Defaults & cost.**

- Default mode is “sampled”: only a handful of layers and only the pure next‑token, so runtime overhead is negligible (a second unembed+softmax at ~3–4 layers).
- No change to CSV schemas by default; JSON carries the sanity results. A debug sidecar CSV can be added later if needed.

**Optional controls (no new public CLI flag).**

- Environment variable: `LOGOS_RAW_LENS ∈ {off|sample|full}` with default `sample`.
  - `off`: disable the check entirely.
  - `sample`: run the default sampled comparison and write JSON summaries.
  - `full`: compute dual‑lens for all layers (pure next‑token), optionally emitting a small sidecar CSV with a `lens ∈ {raw,norm}` column for inspection.
- In `--self-test`, auto‑escalate to `full` and warn/error if large `kl_norm_vs_raw_bits` or early norm‑only semantics are detected.

**Why not a CLI flag.** Keeping this as a baked‑in QA signal avoids a “one‑and‑done” toggle and ensures every run remains robust to lens‑induced artefacts without burdening the user with additional switches.

---

### 1.5. [x] Representation‑drift cosine curve

**Why.** A realist reading predicts an answer‑token direction that exists early and merely grows in magnitude; a nominalist picture predicts the direction rotates into place late. Cosine similarity across depth quantifies which is true.

**What.** *A per‑layer scalar `cos_to_final` written alongside entropy metrics.*

**✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)**
* Implemented in `run.py` for the pure next‑token at layer 0 and all post‑block layers.
* Persisted to `output-*-pure-next-token.csv` as `cos_to_final` (see 001 README Outputs).

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

3. Write `cos_to_final` column; included in pure next‑token CSV. (Plots to be added in analysis notebooks.)

**Note.** Cosine is computed on **logit directions** (not residuals). With a Tuned Lens (§1.12), `cos_to_final` measures closeness to the **tuned** head—still interpretable as “distance to the model’s decision boundary”.

---

### 1.6. [x] Last‑layer consistency check (lens vs final head)

Why. The last post‑block row should agree with the model’s true output distribution. If `KL(P_last || P_final)` stays large (e.g., Gemma‑2‑9B ≈ 1.0 bits), the discrepancy is likely a family‑specific final‑head transform (e.g., temperature or softcap), not a bug in the lens. Making this check permanent prevents subtle regressions across families and highlights when a model’s head applies extra calibration.

What. Always write a JSON diagnostic comparing the lens’ last‑layer distribution to the model’s final head, then probe simple transforms:

```json
"diagnostics": {
  "last_layer_consistency": {
    "kl_to_final_bits": <float>,
    "top1_agree": true,
    "p_top1_lens": <float>,
    "p_top1_model": <float>,
    "p_answer_lens": <float>,
    "answer_rank_lens": <int|null>,
    "temp_est": <float|null>,
    "kl_after_temp_bits": <float|null>,
    "warn_high_last_layer_kl": <bool>,
    "cfg_transform": { "scale": <float|null>, "softcap": <float|null> },
    "kl_after_transform_bits": { "scale": <float|null>, "softcap": <float|null>, "scale_then_softcap": <float|null> }
  }
}
```

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)
* The core comparison and a scalar‑temperature probe (`temp_est`, `kl_after_temp_bits`) are implemented.
* A follow‑up adds detection of simple family‑specific transforms (scale/softcap) via model/config and reports KL after applying them at the last layer only (no change to earlier layers).

How.

1. Cache `final_probs = softmax(final_logits)` once per run.
2. On the last post‑block layer, compute `P_last = softmax(last_logits_lens)`; record KL, top‑1 agreement, and answer metrics (ID‑level).
3. Fit a scalar temperature `s ∈ [0.1,10]` minimizing `KL(softmax(z/s) || final_probs)`; record `temp_est`, `kl_after_temp_bits`.
4. If the model/config exposes head transforms (e.g., `final_logit_scale`, `final_logit_softcap`), apply them to the last‑layer lens logits only and report `kl_after_transform_bits`.

Note on scope. Only the final row is “aligned” to the model’s head. All earlier layers remain decoded with the standard normalized lens to preserve comparability of depth‑wise metrics (entropy, ranks, KL‑to‑final, cosine).

### [x] 1.7. Gold‑token alignment (leading‑space, multi‑piece)

Why. Tokenization differences (leading spaces, multi‑piece tokens) can create apparent mismatches if we compare strings instead of IDs. Making the gold tokenization explicit prevents such drift.

What. Persist `gold_answer` in JSON with `{ string, pieces, first_id }`. Use `first_id` for `is_answer`, `p_answer`, and `answer_rank` throughout.

How.

1. Compute once via tokenizer (no special tokens):

   ```python
   ctx_ids  = tokenizer.encode(prompt, add_special_tokens=False)
   ctx_ans  = tokenizer.encode(prompt + " Berlin", add_special_tokens=False)
   first_id = ctx_ans[len(ctx_ids)]
   pieces   = tokenizer.convert_ids_to_tokens(ctx_ans[len(ctx_ids):len(ctx_ids)+3])
   ```
2. Use `first_id` for `is_answer`, `p_answer`, and `answer_rank` (§1.3), and store in JSON:

   ```json
   "gold_answer": { "string": "Berlin", "pieces": pieces, "first_id": first_id }
   ```

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)
- Central helper: `layers_core/gold.py` computes `first_id`, `pieces`, and selects with/without leading space.
- Runner integration: `run.py` persists `json.gold_answer = {string, pieces, first_id, answer_ids, variant}` and sets `diagnostics.gold_alignment`.
- Semantics: `is_answer` prefers ID/rank (`answer_rank == 1`), with string fallback only if rank is unavailable.
- Tests: `tests/test_gold_alignment.py` covers with-space preference, no-space fallback, unresolved path, and sequence-based helper.

### [x] 1.8. Negative‑control prompt

Why. If, in the France control, the Berlin token outranks Paris, the probe is leaking lexical co‑occurrence. A margin makes leakage quantitative and comparable.

What. Run a control prompt (“… The capital of France …”) alongside the positive prompt; log a control margin and a summary index.

• Per‑layer (control rows only): `control_margin = p(Paris) − p(Berlin)` using ID‑level `first_id` from §1.7.
• Summary: `first_control_margin_pos` (first layer with `control_margin > 0`) and `max_control_margin`.

How.

1. `PROMPTS = [positive_prompt, control_prompt]`; add `prompt_id ∈ {pos, ctl}` to CSV/JSON.
2. For `prompt_id == ctl`, compute the two probabilities and write `control_margin` per layer.
3. After the sweep, store the two summary indices in JSON meta and annotate late/null cases as possible leakage.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)
- Control pass added (France → Paris) with ID‑level gold alignment; records are tagged with `prompt_id`.
- Pure next‑token CSV: new `control_margin = p(Paris) − p(Berlin)` column for control rows; empty for positive rows.
- Both CSVs now include a leading `prompt_id` column.
- JSON persists `control_prompt` (context, `gold_answer`, `gold_alignment`) and `control_summary` with `first_control_margin_pos` and `max_control_margin`.

### [x] 1.9. Ablate stylistic filler ("simply")

Why. Gemma’s early copy‑collapse may be triggered by instruction‑style cues, not semantics. Removing “simply” tests that hypothesis.

What. Rerun the base prompt with and without the filler and compare `L_copy`, `L_semantic`.

How.

1. Duplicate prompt; drop the adverb.
2. Record both runs with `prompt_variant` metadata.
3. Plot `Δ-collapse` for the two variants; a large shift confirms the stylistic‑cue explanation.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)
- Added a no‑filler positive pass (`context_prompt_nf = "… is called"`) alongside the original prompt.
- Both CSVs now include a leading `prompt_variant` column (`orig` | `no_filler`); `prompt_id` remains `pos` for both variants.
- JSON includes `ablation_summary` with collapse indices and deltas:
  `{ L_copy_orig, L_sem_orig, L_copy_nf, L_sem_nf, delta_L_copy, delta_L_sem }`.
- Control rows are not ablated; they are tagged as `prompt_id=ctl, prompt_variant=orig`.

### [x] 1.10. Logit Prism shared decoder

Why. A single whitening + rotation matrix (`W_prism`) that works for all layers makes cross‑layer geometry directly comparable and reduces probe freedom. A shared decoder is a strong baseline/control against “lens‑induced early semantics”, and a cheap preconditioning step for Tuned Lens.

Benefits.
- Reduced probe freedom: one decoder for all layers (vs per‑layer heads) minimizes overfitting to layer idiosyncrasies.
- Cross‑layer comparability: statements like “rotation → alignment, then amplification” are cleaner when the basis is fixed.
- Better calibration than raw logit‑lens with low complexity; KL/rank milestones tend to be more stable early.
- Synergy with Tuned Lens: Prism can initialize or regularize TL translators (A_ℓ) and serve as a regression guardrail.

What. Fit a single linear map that (i) whitens pooled residuals, then (ii) applies an orthogonal rotation into the model’s unembedding basis (or a low‑rank subspace of it). At run time, we decode every layer with both lenses: keep the norm‑lens sweep as primary, and in the same run also decode a Prism sweep from the cached residuals (no second forward).

How (minimal, reproducible).
1) Data sampling (per model):
   - Collect residual snapshots at ~3 depths (≈25/50/75%) over ~100k tokens (batched). Store only summary stats if memory is tight.
   - Use fp32 for statistics; seed from the global SEED (316) for determinism.

2) Fit procedure:
   - Whitening: compute per‑feature mean/variance (or full covariance for true whitening); transform pooled residuals to zero‑mean, unit variance.
   - Rotation/alignment: solve an orthogonal Procrustes from the whitened residual subspace to the unembed subspace (optionally top‑k SVD of `W_U` for k≲d_model). Alternative: ridge to logits with an orthogonality constraint.
   - Output: `W_prism` (shape d_model×d_vocab or d_model×k then project with `W_U_k`), plus means/vars if whitening is applied on‑the‑fly.

3) Validation (held‑out prompt):
   - Report `prism_metrics`: KL(P_prism || P_final) in bits, top‑1 agreement, answer_rank milestones at 25/50/75% depths.
   - Compare to raw norm‑lens and note any early “norm‑only semantics”; Prism should be closer to final than raw in early/mid layers if calibration improves.

4) Integration in `run.py` (always‑on dual decode):
   - Primary: run norm‑lens sweep as today (write canonical CSVs).
   - Sidecar: after the sweep, reuse cached residuals to compute Prism logits for all layers and write sidecar CSVs: `*-records-prism.csv`, `*-pure-next-token-prism.csv` (schemas identical to primary CSVs). No forward duplication.
   - Numerics: compute whitening in fp32; cast back for matmul as needed; seeds/devices unchanged.
   - Provenance: in JSON, persist `prism_provenance` (artifact paths, version/sha, sample_size, depths, whitening kind, rank k, seed) and `prism_metrics` (KL bits, top‑1 agree by depth; optional Prism L_copy/L_sem summary). Primary CSVs remain norm‑lens; Prism lives in sidecars.

5) Storage & layout:
   - Save per‑model artifacts under `001_layers_baseline/prisms/<clean_model_name>/`:
     - `W_prism.pt` (tensor), `whiten.pt` (means/vars or Cholesky), `provenance.json`.
   - Mirror a short summary into each run’s JSON for auditability. If artifacts are missing, optionally perform a “quick fit” from current run cache (tag `fit_mode=quick`).

Lift (engineering).
- Moderate: an offline fit (few hundred MB of activations or aggregated stats), simple SVD/Procrustes, a guarded decode path, and a couple of unit tests. Runtime impact is minimal (one extra matmul per layer).

Trade‑offs / scope.
- Prism is not a tuned lens; it won’t perfectly match the final head. Its value is robustness and comparability, not exact replication. Keep CSV unchanged and treat Prism as a diagnostic/baseline mode.
- When TL is added (see §1.12), initialize TL translators from Prism and/or add a “shared‑decoder” ablation to emulate Prism for QA. Keep Prism available to catch lens‑induced regressions.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

### [x] 1.11. Soft copy indices (strict + soft; windowed)

**Why.** The strict copy rule (`p_top1 > 0.95`, `k=1`) rarely fires outside specific families (e.g., Gemma‑2), which leaves `Δ‑collapse = L_sem − L_copy` undefined and weakens across‑model comparisons. The baseline runs should therefore emit both strict **and** soft/windowed detectors by default so that the richer metrics land in every sweep, while still retaining the strict path for high‑precision claims.

**What.** Detect prompt‑echo at two sensitivities and multiple window sizes, and report both *per‑layer booleans* and *first‑hit layer indices*:

* **Strict copy (baseline default continues):** first layer `L_copy_strict` where the last **k=1** top‑1 token ID is a contiguous subsequence of the prompt IDs with `p_top1 > τ_strict` (default **0.95**).
* **Soft copy (baseline default emits alongside strict):** first layer `L_copy_soft[k]` where a window of the last **k ∈ {1,2,3}** top‑1 IDs forms a contiguous subsequence with `p_top1 > τ_soft` (default **0.33**).
* **Derived summaries:** `Δ_sem_minus_copy_strict = L_sem − L_copy_strict` (nullable) and `Δ_sem_minus_copy_soft[k] = L_sem − L_copy_soft[k]`.

**How.**

1. **Detection (pure next‑token path, per layer).**

```python
# rolling top-1 window (maintained elsewhere)
collapse_strict = is_id_subseq([top1_id], ctx_ids) and (p_top1 > τ_strict)

def collapse_soft_k(k):
    return is_id_subseq(window_ids[-k:], ctx_ids) and (p_top1 > τ_soft)
```

2. **CSV additions (pure next‑token).**

* Booleans: `copy_strict@0.95`, `copy_soft_k1@0.33`, `copy_soft_k2@0.33`, `copy_soft_k3@0.33`.
* (Optional) If `--copy-soft-thresh-list τ1,τ2` is set, emit additional columns `copy_soft_k{K}@{τi}` for each τ in the list.

3. **JSON meta (summary).**

```json
"copy_detector": {
  "strict": {"thresh": 0.95, "k": 1, "L_copy_strict": <int|null>},
  "soft": {
    "thresh": 0.33,
    "window_ks": [1,2,3],
    "L_copy_soft": {"k1": <int|null>, "k2": <int|null>, "k3": <int|null>}
  },
  "deltas": {
    "Δ_sem_minus_copy_strict": <int|null>,
    "Δ_sem_minus_copy_soft": {"k1": <int|null>, "k2": <int|null>, "k3": <int|null>}
  }
}
```

4. **CLI & defaults.**

* `--copy-thresh` (strict; default **0.95**), `--copy-window-k` (strict; default **1**)
* `--copy-soft-thresh` (default **0.33**), `--copy-soft-window-ks` (default **1,2,3**) — both applied by default so that strict + soft detections are always produced.
* Optional override: `--copy-soft-thresh-list` (comma‑sep) to emit additional soft detectors.

**Notes.**

* Keep **strict** `L_copy_strict` as the headline metric for conservatism; use **soft/windowed** indices to compute `Δ‑collapse` when strict is null and to stabilize cross‑model comparisons.
* Maintain existing whitespace/punctuation filtering and ID‑level contiguous matching for all variants.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

---

### [x] 1.12. Integrate a Tuned Lens (translator‑in‑d)

See: `001_layers_baseline/TUNED_LENS_PLAN.md`

**✅ IMPLEMENTATION STATUS: COMPLETED (2025‑09‑27)**

- Training fitter implemented with multi‑layer updates, per‑layer temperatures, width‑scaled rank (k up to 256), 32M‑token budget, and vocab‑aware auto‑scaling.
- Large‑vocab runtime optimizations: compute teacher logits only at sampled positions; allow fp16/bf16 unembed for `d_vocab ≥ 100k`.
- Runtime adapter loads translators on the target device, applies per‑layer temperature, and enforces last‑layer identity; last‑layer consistency check passes (near‑zero KL to model head).
- Outputs: tuned sidecar CSVs written alongside baseline; pure CSV now includes `teacher_entropy_bits` for drift checks.
- Results snapshot (single‑probe; see plan §15/§16 for details):
  - Mistral‑7B: large ΔKL improvements across mid‑depths; earlier `KL≤1.0` crossing; no copy‑collapse; last‑layer agreement passes.
  - Llama‑3‑8B (128k vocab): large ΔKL gains; rank‑earliness did not improve on the single probe; last‑layer agreement passes; entropy drift behaves as expected mid‑stack.

Notes
- “Gates” (to prefer tuned in summaries) remain an evaluator‑level policy and are intentionally kept out of the probe script.

---

### [x] 1.13. Surface‑mass diagnostic (EchoMass vs AnswerMass)

**Why.** Strict prompt‑echo (“copy‑collapse”) almost never fires on the capital question; yet **surface‑form pull** can still exist as diffuse probability mass on **prompt tokens**. A lens‑agnostic, bag‑of‑tokens diagnostic captures the **surface→meaning** transition without requiring contiguous echo or top‑1 dominance.

**What.** For each layer ℓ and chosen lens (norm & tuned), compute:

* **Prompt token set** ( \mathcal{V}_\text{prompt} ): the **unique token IDs** present anywhere in the prompt, **minus** the project’s ignore set ( \mathcal{S} ) (whitespace, punctuation, etc., reusing the copy detector mask).
* **Answer mass** ( \text{AnsMass}^{(\ell)} = P^{(\ell)}(\text{answer_first_id}) ) (the probability of the **gold answer’s first token ID** used by `L_semantic`).
* **Echo mass** ( \text{EchoMass}^{(\ell)} = \sum_{t \in \mathcal{V}_\text{prompt}} P^{(\ell)}(t) ).

Define the **surface→meaning crossover depth** ( L_{\text{surface}\to\text{meaning}} ) as the smallest ℓ such that
[
\text{AnsMass}^{(\ell)} \ge \text{EchoMass}^{(\ell)} + \delta \quad \text{with}\ \delta = 0.05.
]
Also record the **mass ratio** ( \text{AnsMass}^{(\ell)} / (\text{EchoMass}^{(\ell)} + 10^{-9}) ).

**How.**

* **Integration point.** In the per‑layer decode loop (already computing (P^{(\ell)}) for norm and tuned), construct ( \mathcal{V}_\text{prompt} ) once per run using the prompt’s token IDs and the existing ignore mask; cache it.
* **CSV (per layer).** Add: `echo_mass_prompt`, `answer_mass`, `mass_ratio_ans_over_prompt` **for each lens** (prefix columns with `norm_`/`tuned_` or write to the respective sidecar).
* **Run JSON (summary).** Add: `L_surface_to_meaning_norm`, `L_surface_to_meaning_tuned`, and the corresponding **confidence margins** at those depths: `answer_mass_at_L`, `echo_mass_at_L`.
* **Numerics.** Use fp32 softmaxes already in the pipeline; ignore tokens in ( \mathcal{S} ) exactly as in the copy detector.
* **Diagnostics.** The ignore mask (IDs + representative strings) is logged per model under `diagnostics.copy_mask`.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

---

### [x] 1.14. Geometric surface→meaning crossover (cosine to decoder vectors)

**Why.** Decoding thresholds can be noisy; the **residual‑space geometry** provides a complementary, threshold‑light view of when the state aligns more with the **answer direction** than with **prompt directions**.

**What.** Let ( W_U \in \mathbb{R}^{d \times |\mathcal{V}|} ) be the unembedding (already tied and fp32 in the analysis path). For each layer ℓ:

* Let ( h_\text{geom}^{(\ell)} ) be the **state whose logits are decoded** at layer ℓ:
  – **norm lens:** the **normalized** post‑block residual used by the norm lens;
  – **tuned lens:** the **translated** residual ( \tilde h^{(\ell)} ).
* Define **cosines** against **decoder columns**:
  ( \text{cos_to_answer}^{(\ell)} = \cos\big(h_\text{geom}^{(\ell)}, W_U[:, \text{answer_first_id}]\big) ).
  ( \text{cos_to_prompt_max}^{(\ell)} = \max_{t \in \mathcal{V}*\text{prompt}} \cos\big(h*\text{geom}^{(\ell)}, W_U[:, t]\big) ).
* Define the **geometric crossover depth** ( L_{\text{geom}} ) as the smallest ℓ where
  [
  \text{cos_to_answer}^{(\ell)} \ge \text{cos_to_prompt_max}^{(\ell)} + \gamma,\quad \gamma = 0.02.
  ]

**How.**

* **Integration point.** Reuse the same ( \mathcal{V}_\text{prompt} ) and ignore mask from §1.13. Compute the two cosines per layer and per lens before unembedding (use the fp32 `W_U`).
* **CSV (per layer).** Add: `cos_to_answer`, `cos_to_prompt_max`, and a boolean `geom_crossover` **for each lens**.
* **Run JSON (summary).** Add: `L_geom_norm`, `L_geom_tuned`, with `cos_to_answer_at_L`, `cos_to_prompt_max_at_L`.
* **Numerics.** Normalize vectors with an ε‑stabilized L2 norm; for multi‑query prompts, ( \mathcal{V}_\text{prompt} ) includes **all** prompt tokens (minus ( \mathcal{S} )).

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

---

### [x] 1.15. Prompt‑token Top‑K coverage decay

**Why.** Even when no single prompt token dominates, a **large share of high‑probability mass** can remain on prompt tokens early in depth. Tracking its **decay** gives a robust surface‑bias indicator.

**What.** For each layer ℓ and lens:

* Take the **Top‑K** tokens by ( P^{(\ell)} ) (default **K=50**).
* Define **prompt Top‑K mass share**:
  [
  \text{TopKPromptMass}^{(\ell)} = \sum_{t \in \text{TopK}^{(\ell)} \cap \mathcal{V}_\text{prompt}} P^{(\ell)}(t).
  ]
* Define the **half‑coverage depth** ( L_{\text{topK},\downarrow} ) as the smallest ℓ with ( \text{TopKPromptMass}^{(\ell)} \le \tau ) (default **τ=0.33**).

**How.**

* **Integration point.** In the per‑layer decode loop, after logits→probabilities, compute Top‑K once, intersect with ( \mathcal{V}_\text{prompt} ), and sum.
* **CSV (per layer).** Add: `topk_prompt_mass@50` **for each lens**.
* **Run JSON (summary).** Add: `L_topk_decay_norm`, `L_topk_decay_tuned`, with threshold τ and K.
* **Defaults.** K=50, τ=0.33. No new CLI; constants live next to existing copy‑detector config.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

---

### [x] 1.16. Norm‑lens per‑layer temperature control (diagnostic baseline)

**Why.** Tuned lenses include a learned per‑layer temperature; to attribute gains correctly to **rotation** (translator) rather than **calibration**, provide a **fair, temperature‑matched baseline** for the norm lens.

**What.** For each layer ℓ, learn a **scalar temperature** ( \tau_\ell^\text{norm} > 0 ) that minimizes
( \mathrm{CE}\big(\text{softmax}(\frac{z_\text{norm}^{(\ell)}}{\tau}),\ P_\text{final}\big) ) on a small **calibration stream** (e.g., the first **256k** tokens consumed by the TL fitter). Keep a **hold‑out stream** for reporting.

**How.**

* **Fitting.** One‑dimensional optimize ( \tau ) per layer by line search or Adam on the calibration stream; clamp (\tau\in[0.2,5.0]). Persist the vector `tau_norm_per_layer` in the run JSON.
* **Reporting.** Alongside existing `kl_to_final_bits`, add `kl_to_final_bits_norm_temp` per layer; in summaries, report **ΔKL** for:
  (i) **tuned vs norm**, and (ii) **norm_temp vs norm**.
* **No new CLI.** Calibration stream is drawn automatically during the run; artifacts live under the run’s `diagnostics` block.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

---

### [x] 1.17. Skip‑layers sanity (optional, fast)

**Why.** A classic tuned‑lens check: **replace the last m blocks** with the translator and measure next‑token loss. If perplexity barely degrades for small m, the translator captures the **final computation** faithfully.

**What.** For **m ∈ {2, 4, 8}**, at each evaluation position:

* Run the model **up to layer L−m** normally;
* Apply the translator at layer L−m to produce tuned logits;
* Compute CE to the gold next token; aggregate across an **evaluation shard** (~128k tokens).

**How.**

* **Implementation.** Add a light evaluation pass that short‑circuits the forward at L−m and calls the lens adapter for logits.
* **Reporting.** In run JSON: `skip_layers_sanity: {m: ppl_delta}` per m.
* **Guardrail.** If `ppl_delta` > 5% for m=2, flag `tuned_lens_regression=true` in diagnostics (the lens may be overfitting or under‑capacity).

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)


---

### [x] 1.18. Artifacts and schema changes (for §§1.13–1.17)

**What.** Extend outputs with new fields; **no new CLI**.

**How.**

* **Per‑layer CSVs** (norm & tuned sidecars):

  * `echo_mass_prompt`, `answer_mass`, `mass_ratio_ans_over_prompt`
  * `cos_to_answer`, `cos_to_prompt_max`, `geom_crossover`
  * `topk_prompt_mass@50`
  * `kl_to_final_bits_norm_temp` (in **norm** CSV only)
* **Run JSON additions** (top‑level `diagnostics` and `summary`):

  * `L_surface_to_meaning_{norm,tuned}`, `answer_mass_at_L`, `echo_mass_at_L`, `delta=answer_mass-echo_mass`
  * `L_geom_{norm,tuned}`, `cos_to_answer_at_L`, `cos_to_prompt_max_at_L`
  * `L_topk_decay_{norm,tuned}`, with `K=50`, `tau=0.33`
  * `tau_norm_per_layer` vector and `kl_to_final_bits_norm_temp@{25,50,75}%`
  * `skip_layers_sanity: { "m=2": ppl_delta, "m=4": ..., "m=8": ... }`
* **Defaults & provenance.** Record constants (`delta=0.05`, `gamma=0.02`, `K=50`, `tau=0.33`) in the run JSON under `surface_diagnostics_config`.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

---

### [x] 1.19. Windowed Raw‑vs‑Norm Escalation around Collapse Layers (automatic)

**Why.** The sampled dual‑lens sanity check (§1.4) can miss a narrow early window where “semantics” are induced by normalization. A lightweight, targeted escalation around candidate collapse layers prevents over‑reading early rank improvements.

**What.** After the baseline sweep, automatically perform a *windowed* dual‑lens pass (raw vs normalized) on a small set of *center layers*, covering a radius of ±4 layers (clipped to valid indices). Persist both a JSON summary and a compact sidecar CSV.

**How.**

1. **Center layers (union of available indices):**

   * `L_semantic` (first `answer_rank == 1` under the norm lens),
   * `first_rank_le_5` (norm lens),
   * `first_kl_below_1.0` (norm lens),
   * `L_copy_strict` (if not null),
   * earliest among `L_copy_soft[k]` for `k ∈ {1,2,3}` (if any).
2. **Window & decode.** For each center layer `c`, decode layers `ℓ ∈ [c−R, c+R]` (default **R = 4**) twice: **raw** (pre‑norm residual) and **norm** (post‑norm residual). Record for the pure next‑token:

   * `p_top1`, `top1_token_id`, `top1_token_str`,
   * `p_answer`, `answer_rank`,
   * `kl_norm_vs_raw_bits = KL(P_norm || P_raw)` (bits).
3. **Artifacts.**

   * **Sidecar CSV:** `output-<MODEL>-pure-next-token-rawlens-window.csv` with columns
     `layer, lens ∈ {raw,norm}, p_top1, top1_token_id, top1_token_str, p_answer, answer_rank, kl_norm_vs_raw_bits`.
   * **Run JSON (`diagnostics.raw_lens_window`):**

     ```json
     {
       "radius": 4,
       "center_layers": [ ... ],
       "layers_checked": [ ... ],
       "norm_only_semantics_layers": [ ... ],   // layers where is_answer_norm && !is_answer_raw
       "max_kl_norm_vs_raw_bits_window": <float>,
       "mode": "window"
     }
     ```
4. **Escalation gates.** If `raw_lens_check.summary.lens_artifact_risk == "high"` or any sampled layer had `kl_norm_vs_raw_bits ≥ 1.0`, set **R = 8**.
5. **CLI & cost.** No new CLI flags. Overhead is one extra unembed+softmax per layer in a narrow window.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

---

### [x] 1.20. Cosine Milestones & Normalized Depth Summaries

**Why.** Evaluations currently scan CSVs to find when `cos_to_final` crosses useful thresholds and to normalize depths by `n_layers`. Publishing these as summary fields reduces friction and prevents arithmetic drift across evaluators.

**What.** Add milestone and normalized‑depth summaries to run JSON.

**How.**

* **Cosine milestones (per lens present: `norm`, `tuned`):**

  ```json
  "summary": {
    "cos_milestones": {
      "norm": { "ge_0.2": L, "ge_0.4": L, "ge_0.6": L },
      "tuned": { "ge_0.2": L, "ge_0.4": L, "ge_0.6": L }
    }
  }
  ```

  (Use the smallest `L` where the inequality holds; `null` if never reached.)
* **Normalized depths (`/ n_layers`, rounded to 3 decimals):**

  ```json
  "summary": {
    "depth_fractions": {
      "L_semantic_frac": L_semantic / n_layers,
      "first_rank_le_5_frac": first_rank_le_5 / n_layers,
      "L_copy_strict_frac": L_copy_strict ? L_copy_strict / n_layers : null,
      "L_copy_soft_k1_frac": ...,
      "L_copy_soft_k2_frac": ...,
      "L_copy_soft_k3_frac": ...
    }
  }
  ```
* **No CLI changes.** Computed post‑sweep from existing per‑layer arrays.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

---

### [x] 1.21. Unified Sidecar Summaries for Prism & Tuned‑Lens

**Why.** Evaluations now recompute tuned/prism rank milestones and KL deltas from sidecar CSVs. Publishing a compact, consistent summary block avoids duplication and errors, and clarifies that Prism is a *shared‑decoder diagnostic*, not the model’s head.

**What.** Augment the run JSON with per‑lens summaries computed from the sidecars.

**How.**

* **Prism (`diagnostics.prism_summary.metrics`):**

  ```json
  "diagnostics": {
    "prism_summary": {
      "present": true,
      "compatible": true,
      "k": <int>,
      "metrics": {
        "rank_milestones": {
          "baseline": { "le_10": L, "le_5": L, "le_1": L },
          "prism":    { "le_10": Lp, "le_5": Lp, "le_1": Lp },
          "delta":    { "le_10": Lp-L, "le_5": Lp-L, "le_1": Lp-L }
        },
        "kl_bits_at_percentiles": {
          "baseline": { "p25": x, "p50": y, "p75": z },
          "prism":    { "p25": x', "p50": y', "p75": z' },
          "delta":    { "p25": x-x', "p50": y-y', "p75": z-z' }
        },
        "first_kl_le_1.0": { "baseline": Lb, "prism": Lp, "delta": Lp-Lb }
      }
    }
  }
  ```
* **Tuned (`tuned_lens.summary.metrics`):**

  ```json
  "tuned_lens": {
    "summary": {
      "rank_milestones": {
        "baseline": { "le_10": L, "le_5": L, "le_1": L },
        "tuned":    { "le_10": Lt, "le_5": Lt, "le_1": Lt },
        "delta":    { "le_10": Lt-L, "le_5": Lt-L, "le_1": Lt-L }
      },
      "kl_bits_at_percentiles": {
        "baseline": { "p25": x, "p50": y, "p75": z },
        "tuned":    { "p25": x', "p50": y', "p75": z' },
        "delta":    { "p25": x-x', "p50": y-y', "p75": z-z' }
      },
      "first_kl_le_1.0": { "baseline": Lb, "tuned": Lt, "delta": Lt-Lb }
    }
  }
  ```
* **Notes.** Keep these summaries *additive*; do not change CSV schemas.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

---

### [x] 1.22. Machine‑Readable Measurement Guidance (Evaluation Hints)

**Why.** Some models/families require rank‑first reporting (e.g., head‑calibrated final layers; norm‑only semantics). A small, machine‑readable block helps evaluators apply consistent guardrails.

**What.** Add a `measurement_guidance` block to run JSON with boolean gates and reasons.

**How.**

```json
"measurement_guidance": {
  "prefer_ranks": true,               // set if any gate below is true
  "suppress_abs_probs": true,         // same as prefer_ranks; kept explicit for clarity
  "reasons": [
    "warn_high_last_layer_kl",        // from last_layer_consistency
    "norm_only_semantics_window",     // from §1.19 window check
    "high_lens_artifact_risk"         // from raw_lens_check.summary
  ],
  "notes": "Family-level head calibration; treat probabilities comparatively only within model."
}
```

* **Setting logic.**
  `prefer_ranks = (warn_high_last_layer_kl == true) OR (raw_lens_check.summary.lens_artifact_risk == 'high') OR (diagnostics.raw_lens_window.norm_only_semantics_layers not empty)`.
  Mirror to `suppress_abs_probs`.
* **No CLI.** Purely an advisory for downstream evaluation prompts.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

---

### [x] 1.23. Threshold sweep for copy‑collapse

**Why.** Copy‑collapse should capture when the network re‑uses **particular prompt tokens** (ID‑level, contiguous subsequence), not when the lens/temperature makes generic echoes look confident. A small **threshold sweep** tests robustness: if the first copy layer moves a lot as τ changes (e.g., 0.95→0.70), copy is fragile and more consonant with nominalist “name matching.” The sweep now also cross‑checks **raw vs norm** in a narrow window so lens‑induced “copy” is downgraded.

**What.**

1. Add multiple **strict‑copy flags** per layer in `*-pure-next-token.csv`:
   `copy_strict@0.70`, `copy_strict@0.80`, `copy_strict@0.90`, `copy_strict@0.95` (k=1 window; same margin δ=0.10; ID‑contiguous subsequence; ignore whitespace/punctuation top‑1s).
2. Record the **earliest layer** for each threshold in run JSON, both as layers and normalized depth fractions.
3. Cross‑validate the earliest strict‑copy layer at each τ using the **Windowed Raw‑vs‑Norm** check (§1.19): flag `copy_norm_only@τ=true` if strict‑copy holds under the norm lens but not under the raw lens within ±4 (or ±8 if escalated) layers.

**How.**

* **CSV (per layer):** append boolean columns
  `copy_strict@0.70, copy_strict@0.80, copy_strict@0.90, copy_strict@0.95`.
* **Run JSON additions:**

  ```json
  {
    "summary": {
      "copy_thresholds": {
        "tau_list": [0.70, 0.80, 0.90, 0.95],
        "L_copy_strict": {
          "0.70": <int|null>,
          "0.80": <int|null>,
          "0.90": <int|null>,
          "0.95": <int|null>
        },
        "L_copy_strict_frac": {
          "0.70": <float|null>,
          "0.80": <float|null>,
          "0.90": <float|null>,
          "0.95": <float|null>
        },
        "norm_only_flags": {
          "0.70": <bool|null>,
          "0.80": <bool|null>,
          "0.90": <bool|null>,
          "0.95": <bool|null>
        },
        "stability": "<stable|mixed|fragile|none>"
      }
    }
  }
  ```

  * **Stability rule:**
    Let Δτ = |L_copy_strict(0.95) − L_copy_strict(0.70)| measured both in layers and as fraction of depth.
    • **stable** if Δτ ≤ 2 layers **or** ≤ 0.05·n_layers,
    • **fragile** if Δτ ≥ 6 layers **or** ≥ 0.15·n_layers,
    • **mixed** otherwise,
    • **none** if all `L_copy_strict(τ)` are null.
  * Set `norm_only_flags[τ]=true` if the earliest strict‑copy layer at τ **fails under the raw lens** anywhere in the §1.19 window around that layer.
* **Ablation to Δ‑collapse reporting (no breaking change):** keep Δ defined against **strict@0.95** when present; if null, fall back to the earliest `L_copy_soft[k]` (report k). Use the threshold sweep **only** for robustness commentary and the `stability` tag; do not change Δ’s primary definition.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

---

### [x] 1.24. Full Raw‑vs‑Norm dual‑lens sweep (all layers; sidecar by default)

**Why.** The sampled (§1.4) and windowed (§1.19) checks can miss narrow bands where normalization induces “early semantics”. A **full per‑layer** raw‑vs‑norm pass makes lens artefacts auditable across the entire depth for every model, not only “high‑risk” families.

**What.**
Emit a **default** sidecar with raw‑vs‑norm metrics for **every** post‑block layer at the pure next‑token position:
*File:* `output-<model>-pure-next-token-rawlens.csv`
*Columns (per layer):*
`layer, p_top1_raw, top1_token_id_raw, top1_token_str_raw, p_answer_raw, answer_rank_raw, p_top1_norm, top1_token_id_norm, top1_token_str_norm, p_answer_norm, answer_rank_norm, kl_norm_vs_raw_bits, norm_only_semantics` (boolean: `answer_rank_norm==1 && answer_rank_raw!=1`).
*Run JSON (`diagnostics.raw_lens_full`):*
`{ pct_layers_kl_ge_1.0, pct_layers_kl_ge_0.5, n_norm_only_semantics_layers, earliest_norm_only_semantic, max_kl_norm_vs_raw_bits, mode: "full" }`.

**How.**

1. During the standard sweep, capture the **pre‑norm** residual and compute `P_raw` alongside `P_norm` for the NEXT position at **every** layer; write a row per layer.
2. Populate `diagnostics.raw_lens_full` from the sidecar.
3. Promote the existing windowed CSV to `*-rawlens-window.csv` and leave it intact (§1.19).

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

---

### [x] 1.25. Confirmed‑semantics gate (norm corroborated by raw/Tuned)

**Why.** To avoid lens‑induced “early semantics”, declare **semantic onset** only when the norm‑lens rank‑1 is corroborated by a second view within a small window. This promotes **rank robustness** from single‑lens to multi‑lens evidence.

**What.**
Add a **confirmed** milestone:
`L_semantic_confirmed = min L s.t. (answer_rank_norm(L)==1) AND (∃ L' ∈ [L−Δ,L+Δ]: answer_rank_raw(L')==1 OR answer_rank_tuned(L')==1)` with default `Δ=2`.
*Run JSON (`summary.confirmed_semantics`):*
`{ L_semantic_norm, L_semantic_raw? (if exists), L_semantic_tuned? (if exists), Δ_window: 2, L_semantic_confirmed, confirmed_source: "raw"|"tuned"|"both"|"none" }`.

**How.**

1. Reuse per‑layer **raw** sidecar (§1.24) and **tuned** sidecar (§1.12) to scan ranks in `[L−Δ, L+Δ]`.
2. Write `L_semantic_confirmed` and `confirmed_source` into the run JSON.
3. If neither corroborates within the window, set `confirmed_source="none"` and emit `diagnostics.flags.norm_only_semantics_confirmed=false`.
4. Evaluators may prefer `L_semantic_confirmed` over `L_semantic_norm` when present (advisory only).

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

---

### [x] 1.26. Rotation‑vs‑Temperature attribution & “prefer_tuned” gate

**Why.** Tuned‑Lens gains mix **rotation** (translator) and **calibration** (temperature). The norm‑temp baseline (§1.16) provides a fair control; attributing ΔKL correctly prevents over‑crediting the translator.

**What.**
Compute, at depth percentiles {25, 50, 75}:
`ΔKL_tuned = KL_norm − KL_tuned` and `ΔKL_temp = KL_norm − KL_norm_temp`.
Define **rotation gain** `ΔKL_rot = ΔKL_tuned − ΔKL_temp`.
*Run JSON (`tuned_lens.attribution`):*

```
{ "percentiles": { "p25": { "ΔKL_tuned": x, "ΔKL_temp": y, "ΔKL_rot": x−y },
                   "p50": { ... }, "p75": { ... } },
  "prefer_tuned": boolean // see rule below
}
```

**Gate (prefer_tuned).** Set true if either (a) `ΔKL_rot(p50) ≥ 0.2` bits **or** (b) `first_rank_le_5` is earlier under tuned by ≥2 layers (or ≥0.05·n_layers).

**How.**

1. Persist `kl_to_final_bits_norm_temp` per layer (already in §1.16) and compute percentiles.
2. Read tuned sidecar KL per layer; compute `ΔKL_tuned`.
3. Derive `ΔKL_rot` and the `prefer_tuned` boolean; expose an advisory field in `measurement_guidance.preferred_lens_for_reporting ∈ {"norm","tuned"}`.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

---

### [x] 1.27. Lens‑artefact risk score (numeric) from full dual‑lens sweep

**Why.** The categorical `lens_artifact_risk` (§1.4) is useful, but a **numeric score** enables finer evaluation policies and trend tracking across runs/families once the **full** dual‑lens sweep (§1.24) exists.

**What.**
`lens_artifact_score ∈ [0,1] = 0.6·pct_layers(kl_norm_vs_raw_bits≥1.0) + 0.3·1[n_norm_only_semantics_layers>0] + 0.1·min(1, max_kl_norm_vs_raw_bits/5)`; thresholds: `<0.2=low`, `0.2–0.5=medium`, `>0.5=high`.
*Run JSON:*
`diagnostics.raw_lens_full.score = { lens_artifact_score, tier }`.
Mirror to `measurement_guidance.reasons += ["high_lens_artifact_score"]` when `tier="high"`.

**How.**
Aggregate directly from `*-rawlens.csv` (§1.24) and `diagnostics.raw_lens_full`. Keep weights/thresholds as constants in `diagnostics.config.lens_artifact_score`.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

---

### [x] 1.28. Evaluator‑readable guidance extensions

**Why.** Evaluations already honor `measurement_guidance` (§1.22); adding **preferred lens** and **confirmed‑semantics** hints reduces accidental misuse.

**What.**
Extend `measurement_guidance` with:

```
"preferred_lens_for_reporting": "norm"|"tuned",
"use_confirmed_semantics": true|false, // suggest L_semantic_confirmed when present
"notes_append": "Prism is diagnostic-only; treat probabilities comparatively within model."
```

**How.**
Set `preferred_lens_for_reporting` from §1.26. Set `use_confirmed_semantics=true` when `L_semantic_confirmed` exists (§1.25). Append notes verbatim.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

---

### [x] 1.29. Centralize KL helpers & de-duplicate docs/tests

**Why.** Production paths already compute KL as **KL(P_layer ∥ P_final)** via a helper, but scattered examples and the optional sanity test still show `torch.kl_div(...)`. Keeping all KL math and examples behind one helper eliminates confusion and prevents silent orientation regressions.

**What.**
Unify *all* KL computations and examples under `layers_core.numerics.kl_bits(p, q)`; remove remaining `torch.kl_div` from code snippets and tests.

**How.**

1. **Code:** replace `torch.kl_div(...)` in `kl_sanity_test.py` with `kl_bits(...)`.
2. **Tests:** add `assert_close(kl_bits(p, q), (p*(p.log()-q.log())).sum()/ln2)` and `kl_bits(p, p)≈0`; confirm asymmetry `kl_bits(p,q)!=kl_bits(q,p)`.
3. **Docs (§1.3):** swap the illustrative snippet to call `kl_bits(probs, final_probs)` and add: “All KL in this repo is **KL(P_layer ∥ P_final)** in **bits**, unless stated otherwise.”

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

---

### [x] 1.30. Normalizer provenance & per-layer effect logging

**Why.** Even with architecture‑aware selection, auditability requires explicit provenance. When results are surprising, evaluators should see **which normalization module** was applied and how much it changed the residual.

**What.**
Emit a `diagnostics.normalization_provenance` block and two per‑layer scalars quantifying normalization’s effect.

**How.**

1. **JSON (`diagnostics.normalization_provenance`):**

```json
{
  "arch": "pre_norm|post_norm",
  "strategy": "post_ln2|next_ln1|raw",
  "per_layer": [
    {"layer": 0, "ln_source": "blocks[0].ln1|ln2|final", "eps_inside_sqrt": true, "scale_gamma_used": true}
  ]
}
```

2. **CSV (pure next‑token, per layer):**

   * `resid_norm_ratio = ||h_norm||₂ / (||h_raw||₂ + 1e-12)`
   * `delta_resid_cos = cos(h_raw, h_norm)`
3. **Validator:** warn on spikes (norm change > 3× or cosine < 0.8) before L_semantic.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

---

### [x] 1.31. Unembedding-bias audit & bias-free cosine guarantee

**Why.** Some families keep a non‑zero `lm_head.bias`. Cosines and direction metrics should be **bias‑free**; otherwise bias can spuriously boost agreement.

**What.**
Audit the bias and guarantee that all geometric metrics use **bias‑free logits**.

**How.**

1. **JSON (`diagnostics.unembed_bias`):**

```json
{ "present": true|false, "l2_norm": float, "max_abs": float }
```

2. **Code:** ensure `cos_to_final`, `cos_to_answer`, `cos_to_prompt_max` use `(resid @ W_U)` (no `+ b`).
3. **Unit test:** cosines invariant to adding a constant bias vector to logits.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

---

### [x] 1.32. Determinism & environment capture

**Why.** Reproducibility across machines/runs is part of “measurement right.” Subtle changes in PyTorch/CUDA/mps or nondeterministic kernels can shift borderline thresholds.

**What.**
Capture deterministic flags and environment in JSON.

**How.**

1. **JSON (`provenance.env`):**

```json
{
  "torch_version": "...", "cuda_version": "...", "cudnn": "...",
  "device": "cuda|mps|cpu", "dtype_compute": "bf16|fp16|fp32",
  "deterministic_algorithms": true|false,
  "cudnn_benchmark": false|true,
  "seed": 316, "python": "...", "platform": "..."
}
```

2. **Runtime:** set `torch.use_deterministic_algorithms(True)` where available; set `torch.backends.cudnn.benchmark = False`.
3. **Flag:** add `diagnostics.flags.nondeterministic=true` when determinism cannot be guaranteed; propagate to `measurement_guidance.reasons`.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

---

### [x] 1.33. Numeric-health sentinels (NaN/Inf, overflow, underflow)

**Why.** Rare fp anomalies can silently corrupt a few layers and mislead collapse detection.

**What.**
Continuous runtime checks with a compact summary.

**How.**

1. **Checks (per layer, per lens):** `any_nan`, `any_inf` on residuals/logits; `max_abs_logit`; `min_prob` after softmax (fp32).
2. **JSON (`diagnostics.numeric_health`):**

```json
{ "any_nan": false, "any_inf": false, "max_abs_logit_p99": 18.4, "min_prob_p01": 1e-12, "layers_flagged": [ ... ] }
```

3. **Gate:** if flagged early layers overlap `L_copy*` or `L_semantic`, add a caution into `measurement_guidance.reasons`.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

---

### [x] 1.34. Copy-detector mask transparency & tests

**Why.** The “ignore set” (whitespace, punctuation, markup) materially affects copy detection and surface‑mass. It must be explicit and tested across tokenizers.

**What.**
Publish the mask per model and add tokenizer‑specific unit tests.

**How.**

1. **JSON (`diagnostics.copy_mask`):**

```json
{ "ignored_token_ids": [...], "ignored_token_str_sample": ["▁", "!", ".", "”"], "size": 123 }
```

2. **Tests:** for each tokenizer, assert expected coverage of whitespace/punctuation; regression‑test the mask size.
3. **Docs:** add a pointer in §1.11/§1.13 that mask provenance is logged.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

---

### [x] 1.35. Confidence margins for answer token

**Why.** Ranks are robust; margins quantify confidence and make threshold crossings interpretable (e.g., rank‑1 with tiny margin vs decisive win).

**What.**
Add two per‑layer scalars: *answer‑vs‑runner‑up* margin and *answer‑vs‑top1* margin when answer is not top‑1.

**How.**

1. **CSV (pure next‑token):**

   * `answer_logit_gap = logit(answer) − logit(second_best)` when `answer_rank==1`, else `null`.
   * `answer_vs_top1_gap = logit(answer) − logit(top1)` when `answer_rank>1`, else `null`.
2. **JSON (summary):** first layer where `answer_logit_gap ≥ {0.5, 1.0}` (nats or bits; specify unit).
3. **Diagnostics:** include an explicit `answer_margin_unit = "logit"` entry alongside the summary to remove ambiguity about the units.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

---

### [x] 1.36. Layer-index provenance map

**Why.** Off‑by‑one errors in indexing (post‑block vs pre‑block) are a classic source of confusion.

**What.**
Emit a compact map from `layer` indices to **model block names** and decoded stream descriptors.

**How.**

1. **JSON (`diagnostics.layer_map`):**

```json
[
  {"layer": 0, "block": "blocks[0]", "stream": "post_block", "norm": "ln2|next_ln1"},
  {"layer": N, "block": "final", "stream": "unembed_head"}
]
```

2. **Evaluator hint:** set `measurement_guidance.reasons += ["layer_map_missing"]` if absent.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

---

### [x] 1.37. Symmetric divergences & entropy gap in Raw‑vs‑Norm

**Why.** `KL(P_norm ∥ P_raw)` can explode when `P_raw` has near‑zeros, obscuring whether disagreement is local or global. Symmetric and entropy‑based views reduce this ambiguity and make the lens‑artefact score more diagnostic.
**What.**

* Per‑layer metrics alongside the existing raw‑vs‑norm fields:
  `js_divergence`, `kl_raw_to_norm_bits` (reverse KL), `entropy_bits_raw`, `entropy_bits_norm`, `entropy_gap_bits = entropy_bits_norm − teacher_entropy_bits` (already logged) and `l1_prob_diff = ∑|P_norm − P_raw|`.
* A compact summary in JSON: percentiles for `js_divergence` and `l1_prob_diff`, plus `first_js_le_0.1` and `first_l1_le_0.5`.
* Extend `diagnostics.raw_lens_full.score` with an optional, **non‑breaking** additive sub‑score (e.g., `+0.1·min(1, js_p50/0.1)`).
  **How.**
* Compute per‑layer with the existing prob vectors; reuse fp32 softmax.
* Write columns to the `*-rawlens.csv` sidecar; record summary percentiles in `diagnostics.raw_lens_full`.
* Keep the old score stable; surface a second `lens_artifact_score_v2` alongside (do not replace).

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

---

### [x] 1.38. Top‑K overlap to qualify raw‑vs‑norm drift

**Why.** A large KL can be driven by tail mass. A **set‑level** overlap makes visible whether disagreement affects the high‑probability region.
**What.**

* Per‑layer `topk_jaccard_raw_norm@K` (default `K=50`) and `topk_jaccard_consecutive@K` (overlap between successive layers under the **same** lens).
* JSON summary: `jaccard_raw_norm_p50`, `first_jaccard_raw_norm_ge_0.5`.
  **How.**
* Reuse the existing Top‑K machinery (already used for `topk_prompt_mass@50`).
* Emit two columns in both the raw‑lens and primary CSVs; add a small summary block under `diagnostics.topk_overlap`.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

---

### [x] 1.39. Repeatability micro‑check (same prompt, same run)

**Why.** Non‑deterministic kernels can shift borderline ranks. A cheap intra‑run repeatability gauge prevents over‑interpreting one‑off flips.
**What.**

* Run the **decode path only** twice for the same cached residuals at all layers/lenses; compute:
  `max_rank_dev`, `p95_rank_dev`, and `top1_flip_rate`.
* JSON: `diagnostics.repeatability = {max_rank_dev, p95_rank_dev, top1_flip_rate}` and gate `measurement_guidance.reasons += ["repeatability_variance_high"]` if `max_rank_dev > 5` or `top1_flip_rate > 0.02`.
  **How.**
* After the forward pass (residuals cached), re‑decode logits with a no‑grad path twice; compare per‑layer ranks.
* No CLI; auto‑skip if `provenance.env.deterministic_algorithms == true`.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

---

### [x] 1.40. Gold‑token alignment summary (battery‑ready)

**Why.** Alignment failures or variant fallbacks (leading‑space, multi‑piece) must be auditable before scaling to batteries/multilingual probes.
**What.**

* JSON: `diagnostics.gold_alignment = { ok: bool, variant: "with_space|no_space|fallback", first_id, answer_ids, pieces }` (already present) **plus** `gold_alignment_rate` (fraction over all prompts in the run; `1.0` for single‑prompt sweeps).
  **How.**
* Accumulate across prompts in the runner; for single‑prompt runs, write `1.0` and keep the richer fields verbatim.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)y

---

### [x] 1.41. Residual‑norm trajectory classifier

**Why.** Monotone vs non‑monotone norm growth hints at “iterative refinement” vs “modular subcomputations.” A coarse classifier helps interpret depth curves.
**What.**

* CSV already has `resid_norm_ratio`; add JSON:
  `diagnostics.norm_trajectory = { shape: "monotonic|plateau|non_monotonic|spike", slope, r2, n_spikes }`.
  **How.**
* Fit a per‑run linear model on log‑norms; set `shape` by slope sign and residual variance (spike = any early layer with `ratio > 3×` or `delta_resid_cos < 0.8`).
* No CLI; attach to existing normalization provenance.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)y

---

### [x] 1.42. Entropy columns for each lens (layer‑local)

**Why.** The “entropy − teacher” drift is informative only if both sides are logged per layer/lens, not inferred from a single side.
**What.**

* Add `entropy_bits` per layer in **primary** CSV and in tuned sidecars; JSON summaries: `entropy_gap_bits@{25,50,75}%`.
  **How.**
* Compute `−∑ p log₂ p` at decode time; reuse teacher entropy already persisted; add three percentile summaries into `summary.entropy`.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)y

---

### [x] 1.43. Tuned‑Lens Provenance & Generalization Audit (no refit)

**Why.** The tuned lens is already fitted and saved with sufficient provenance. Measurement integrity can be improved—without any re‑fitting—by (i) surfacing that provenance in run outputs, (ii) decomposing the loaded translator’s effect into **rotation‑only** vs **temperature‑only** at decode time, (iii) checking **positional generalization** (end‑of‑sequence OOD), and (iv) quantifying **final‑head mismatch calibration**. This makes tuned‑lens‑based milestones auditable and robust for later phases.

**What.**
Add a **read‑only audit** that runs on the loaded tuned lens and emits:

1. **Provenance snapshot (JSON, read‑only)**
   Mirror key metadata from the saved tuned‑lens bundle in the run JSON under `tuned_lens.provenance_snapshot`:

   * `dataset_id`, `dataset_revision`, `content_hash`
   * `train_pos_window = [pos_min, pos_max]` (e.g., `[0.60, 0.95]`)
   * `sampled_layers_count`, `sampled_positions_count`, `rank`, `preconditioner` (e.g., `{"whiten": bool, "orthogonal_rotation": bool}`)
   * `temperatures_stats = {p25, p50, p75, min, max}`, `final_layer_identity: true`
   * `fit_total_tokens_est`, `optimizer`, `schedule`

2. **Rotation‑only & Temperature‑only component ablations (no weight changes)**
   Decode each layer in four variants: **baseline norm lens**, **full tuned**, **rotation‑only** (set τ=1 at decode), **temperature‑only** (bypass affine map, keep τ). For each variant, compute:

   * `kl_to_final_bits`, `answer_rank`, `p_answer` (within‑model, for reference), `entropy_bits`
   * Deltas vs baseline: `delta_kl_bits_* = kl_baseline − kl_variant`, `rank_shift_* = answer_rank_baseline − answer_rank_variant`
     Emit a per‑layer **sidecar**:
   * **CSV:** `output-<MODEL>-pure-next-token-tuned-variants.csv`
     **Columns:**
     `layer, kl_bits_baseline, kl_bits_tuned, kl_bits_rot_only, kl_bits_temp_only, delta_kl_bits_tuned, delta_kl_bits_rot_only, delta_kl_bits_temp_only, delta_kl_bits_interaction, answer_rank_baseline, answer_rank_tuned, answer_rank_rot_only, answer_rank_temp_only, rank_shift_tuned, rank_shift_rot_only, rank_shift_temp_only, entropy_bits_baseline, entropy_bits_tuned`
     *Notes:* `delta_kl_bits_interaction = (kl_baseline − kl_tuned) − [(kl_baseline − kl_rot_only) + (kl_baseline − kl_temp_only)]`.

3. **Positional OOD check (end‑of‑sequence generalization)**
   Evaluate ΔKL improvements across a fixed grid of next‑token positions (fractions of sequence length):
   `pos_grid = {0.20, 0.40, 0.60, 0.80, 0.92, 0.96, 0.98, 1.00}` (clamp to valid decode sites).
   For each `pos` and `layer`, compute the same metrics as in (2) for **baseline** and **full tuned**; summarize:

   * `pos_ood_ge_0.96 = median(delta_kl_bits_tuned) over pos ∈ {0.96, 0.98, 1.00}`
   * `pos_in_dist_le_0.92 = median(delta_kl_bits_tuned) over pos ∈ {≤0.92}`
   * `pos_ood_gap = pos_ood_ge_0.96 − pos_in_dist_le_0.92`
     Emit:
   * **CSV:** `output-<MODEL>-positions-tuned-audit.csv`
     **Columns:**
     `pos_frac, pos_index, layer, kl_bits_baseline, kl_bits_tuned, delta_kl_bits_tuned, answer_rank_baseline, answer_rank_tuned, rank_shift_tuned, entropy_bits_baseline, entropy_bits_tuned`
   * **JSON (summary):** under `tuned_lens.audit_summary.positional`: `{grid, pos_ood_ge_0.96, pos_in_dist_le_0.92, pos_ood_gap}`

4. **Final‑head mismatch calibration sanity (model‑level)**
   At the **final layer**, compute a scalar `τ*_modelcal` that minimizes `KL(tuned_logits/τ ∥ P_final)`; report:

   * `kl_bits_tuned_final`, `kl_bits_tuned_final_after_tau_star`, `tau_star_modelcal`
   * If `kl_bits_tuned_final_after_tau_star` ≪ `kl_bits_tuned_final`, treat remaining mismatch as **global calibration** rather than representation.
     Emit:
   * **JSON:** `tuned_lens.audit_summary.head_mismatch = { kl_bits_tuned_final, kl_bits_tuned_final_after_tau_star, tau_star_modelcal }`

5. **Reporting gate (tuned‑is‑calibration‑only)**
   Derive a model‑level flag from component ablations:

   * If `delta_kl_bits_rot_only@p50 < 0.2` **and** `delta_kl_bits_temp_only@p50 ≥ 0.8 · delta_kl_bits_tuned@p50`, set
     `tuned_lens.audit_summary.tuned_is_calibration_only = true`.
   * Otherwise `false`.
     Add a hint to `measurement_guidance.reasons` when `true`.

6. **JSON roll‑up (single place to read)**
   Under `tuned_lens.audit_summary`, include:

   * `rotation_vs_temperature = { delta_kl_rot_p25, p50, p75, delta_kl_temp_p25, p50, p75, delta_kl_interaction_p50 }`
   * `positional` (as above)
   * `head_mismatch` (as above)
   * `preferred_semantics_lens_hint = "tuned" | "norm" | "tuned_for_calibration_only"` (non‑binding suggestion for Parts 2–5)

**How.**

* **Provenance snapshot:** when loading the tuned lens, copy metadata into the run JSON (`tuned_lens.provenance_snapshot`), including temperatures summary and translator rank.
* **Component ablations:** add decode‑time toggles (no weight edits): (a) set `τ=1` to obtain rotation‑only; (b) bypass the affine map to obtain temperature‑only. Reuse the same cached normalized residuals and unembed.
* **Positional OOD:** run decode at `pos_grid` by selecting earlier next‑token positions; reuse the same prompt and cached forward pass(es). Aggregate deltas and emit the positions sidecar plus JSON summary.
* **Head mismatch:** compute `τ*_modelcal` by a 1‑D line search on the final‑layer tuned logits; store values in JSON.
* **Plumbing:** write the new CSV sidecars alongside existing `*-tuned.csv` files; append the new JSON blocks under `tuned_lens.audit_summary` and `measurement_guidance.reasons` if the gate triggers.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)y

---

### [x] 1.44. LLM‑Eval Pack (derived metrics & citations)

**Why.** The evaluation prompts are executed by LLMs. Providing a compact, derived “eval view” of each run minimizes on‑the‑fly calculations, enforces consistency across models, and reduces errors. The pack aggregates key metrics introduced in §§1.37–1.43, adds normalized/derived summaries, and includes row‑level **citations** back to CSVs so LLMs can quote line numbers without scanning full files.

**What.**
Emit a per‑model **JSON** block `evaluation_pack` (embedded in the main run JSON) and two light **CSV** sidecars that standardize commonly referenced rows.

1. **JSON: `evaluation_pack` (embedded)**

   ```json
   {
     "model": "<name>",
     "n_layers": N,
     "preferred_lens_for_reporting": "norm|tuned",
     "use_confirmed_semantics": true|false,
     "milestones": {
       "L_copy_strict": L_or_null,
       "L_copy_soft": {"k": 1|2|3, "layer": L_or_null},
       "L_semantic_norm": L_or_null,
       "L_semantic_confirmed": {"layer": L_or_null, "source": "raw|tuned|both|none"},
       "depth_fractions": {
         "copy_strict_frac": f_or_null,
         "semantic_frac": f_or_null,
         "delta_hat": "(L_sem - L_copy)/n_layers"   // uses strict if present, else earliest soft {k}
       }
     },
     "artifact": {
       "lens_artifact_score": x,                      // legacy
       "lens_artifact_score_v2": x2,                  // §1.37
       "js_divergence_p50": x,
       "l1_prob_diff_p50": x,
       "first_js_le_0.1": L_or_null,
       "first_l1_le_0.5": L_or_null,
       "jaccard_raw_norm_p50": x,                     // §1.38
       "first_jaccard_raw_norm_ge_0.5": L_or_null,
       "pct_layers_kl_ge_1.0": p_or_null,
       "n_norm_only_semantics_layers": n_or_null,
       "earliest_norm_only_semantic": L_or_null,
       "risk_tier": "low|medium|high"
     },
     "repeatability": {
       "max_rank_dev": mrd, "p95_rank_dev": p95, "top1_flip_rate": r, "flag": "ok|high_variance"
     },
     "alignment": {
       "gold_alignment_rate": r,                      // §1.40
       "variant": "with_space|no_space|fallback"
     },
     "norm_trajectory": {                             // §1.41
       "shape": "monotonic|plateau|non_monotonic|spike",
       "slope": s, "r2": r2, "n_spikes": k
     },
     "entropy": {                                     // §1.42
       "entropy_gap_bits_p25": e25,
       "entropy_gap_bits_p50": e50,
       "entropy_gap_bits_p75": e75
     },
     "tuned_audit": {                                 // §1.43
       "rotation_vs_temperature": {
         "delta_kl_rot_p25": r25, "p50": r50, "p75": r75,
         "delta_kl_temp_p25": t25, "p50": t50, "p75": t75,
         "delta_kl_interaction_p50": i50
       },
       "positional": {
         "pos_grid": [ ... ],
         "pos_ood_ge_0.96": ood,
         "pos_in_dist_le_0.92": ind,
         "pos_ood_gap": gap
       },
       "head_mismatch": {
         "kl_bits_tuned_final": k0,
         "kl_bits_tuned_final_after_tau_star": k1,
         "tau_star_modelcal": tau_star
       },
       "tuned_is_calibration_only": true|false,
       "preferred_semantics_lens_hint": "tuned|norm|tuned_for_calibration_only"
     },
     "citations": {
       "layers": {
         "L_copy_strict_row": row_or_null,
         "L_copy_soft_row": row_or_null,
         "L_semantic_norm_row": row_or_null,
         "L_semantic_confirmed_row": row_or_null
       },
       "files": {
         "pure_csv": "output-<MODEL>-pure-next-token.csv",
         "tuned_pure_csv": "output-<MODEL>-pure-next-token-tuned.csv",
         "rawlens_full_csv": "output-<MODEL>-pure-next-token-rawlens.csv",
         "tuned_variants_csv": "output-<MODEL>-pure-next-token-tuned-variants.csv",
         "positions_audit_csv": "output-<MODEL>-positions-tuned-audit.csv"
       }
     }
   }
   ```

2. **CSV: `output-<MODEL>-milestones.csv`**
   **Columns:**
   `layer, is_copy_strict, is_copy_soft_k, is_semantic_norm, is_semantic_confirmed, answer_rank, p_answer, kl_to_final_bits, entropy_bits, lens`
   *Content:* exactly **four** rows for quick quoting: `L_copy_strict` (if present), earliest `L_copy_soft[k]` (if strict null), `L_semantic_norm`, `L_semantic_confirmed` (if present), each with the **row index** from the pure CSV.

3. **CSV: `output-<MODEL>-artifact-audit.csv`**
   **Columns:**
   `layer, js_divergence, kl_raw_to_norm_bits, l1_prob_diff, topk_jaccard_raw_norm@50, lens_artifact_score_v2, risk_tier`
   *Content:* per‑layer values for quick scan, plus a header row with p25/p50/p75 rolled‑up values (annotated in comments or as a `_summary` row).

**How.**

* Populate `evaluation_pack` after all §1.37–1.43 computations, pulling directly from existing diagnostics and sidecars (no recomputation).
* For `citations.layers`, record **0‑based row indices** from the pure CSV where each milestone occurs. If a milestone is null, set the row to null.
* Generate the two CSVs from existing arrays; write them alongside the other sidecars.
* Keep all additions **read‑only**; no model refits or translator edits.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

### [x] 1.45. Above‑uniform **semantics margin** gate

**Why.**
Rank‑1 alone can occur at near‑uniform probabilities (especially with large vocabularies), which risks over‑interpreting noise as “semantic onset.” A small, uniform‑referenced margin disambiguates genuine signal from flat distributions without changing the rest of the pipeline.

**What.**
Augment rank‑based milestones with a **uniform‑baseline margin**:

* Let (p_\mathrm{uni} = 1/|\mathcal{V}|) (the softmax vocabulary size at decode time).
* Define the **uniform margin** (m^{(\ell)} = p_\mathrm{answer}^{(\ell)} - p_\mathrm{uni}).
* A layer qualifies as **rank‑1 with margin** when `answer_rank == 1` **and** (m^{(\ell)} \ge \delta_\mathrm{abs}). Default (\delta_\mathrm{abs} = 0.002).

Report **first** layer satisfying this gate and surface a warning when `L_semantic_norm` is rank‑only with **insufficient** margin.

**How.**

1. **Decode‑loop additions (pure next‑token; norm & tuned where applicable).**

   * Compute once per model: `p_uniform = 1.0 / vocab_size`.
   * Per layer, compute:
     `answer_minus_uniform = p_answer - p_uniform` (float; write even if `answer_rank>1`).
   * Add per‑layer boolean (norm lens only):
     `semantic_margin_ok = (answer_rank == 1) and (answer_minus_uniform >= δ_abs)`.

2. **CSV changes (pure next‑token).**

   * New columns: `answer_minus_uniform`, `semantic_margin_ok` (norm lens).
     *(Keep tuned/prism in their sidecars; no schema change there.)*

3. **Run JSON (summary).**

   ```json
   "summary": {
     "semantic_margin": {
       "delta_abs": 0.002,
       "p_uniform": 0.0000,
       "L_semantic_margin_ok_norm": <int|null>,
       "margin_ok_at_L_semantic_norm": true|false,
       "p_answer_at_L_semantic_norm": <float|null>
     }
   }
   ```

   * If `L_semantic_confirmed` exists, also add:
     `"L_semantic_confirmed_margin_ok_norm": <int|null>` (the **confirmed** layer that also passes the margin gate at the **norm** lens, if any).

4. **Measurement guidance (evaluation hints).**

   * When `L_semantic_norm` exists but `margin_ok_at_L_semantic_norm == false`, set:

     ```json
     "measurement_guidance": {
       "reasons": ["rank_only_near_uniform", ...],
       "prefer_ranks": true,
       "suppress_abs_probs": true
     }
     ```
   * *Advisory:* Evaluators **prefer** `L_semantic_confirmed_margin_ok_norm` when present; otherwise annotate the semantic onset as **weak (near‑uniform)**.

5. **CLI / config.**

   * Optional flag: `--semantic-margin-abs 0.002` (float).
     Also read environment override: `LOGOS_SEMANTIC_MARGIN_ABS`.

**Defaults & cost.**
Negligible runtime (one subtraction per layer). No extra forwards.

**✅ Provenance fields.**
Embed constants under `diagnostics.surface_diagnostics_config.semantic_margin = { "delta_abs": 0.002 }` for reproducibility.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

---

### [x] 1.46. Minimal **micro‑suite of isomorphic facts** (N=5) for robustness

**Why.**
Single‑fact probing is brittle w.r.t. tokenization quirks and idiosyncratic memorization. A **tiny** suite of equally trivial facts yields variance/median estimates of collapse and semantics depths **within** a model, improving reliability without expanding conceptual scope.

**What.**
Run the same measurement pass over **five** positive prompts (same template), keeping the existing France control:

* **Positives (default list):**
  `Germany→Berlin` *(baseline)*, `France→Paris`, `Italy→Rome`, `Japan→Tokyo`, `Canada→Ottawa`.
* **Control:** keep the existing **France** control margin (`p(Paris) − p(Berlin)`), unchanged.

**How.**

1. **Prompts.**
   Extend the runner to iterate over the `facts_micro_suite` list and emit rows for each *positive* fact using the same `prompt_variant` handling (`orig` / `no_filler`). Controls remain as today (a single France control per run).

2. **CSV additions.**
   Add leading identifiers:

   * `fact_key` (string; e.g., `"Germany→Berlin"`, `"France→Paris"`, ...),
   * `fact_index` (0..N−1).
     `prompt_id` remains `pos` for positives and `ctl` for the France control.

3. **Run JSON (summary).**

   ```json
   "prompts": {
     "facts_micro_suite": ["Germany→Berlin","France→Paris","Italy→Rome","Japan→Tokyo","Canada→Ottawa"]
   },
   "summary": {
     "micro_suite": {
       "n": 5,
       "L_copy_strict_median": <int|null>,
       "L_copy_soft_k1_median": <int|null>,
       "L_semantic_norm_median": <int|null>,
       "L_semantic_confirmed_median": <int|null>,
       "L_semantic_margin_ok_norm_median": <int|null>,
       "delta_hat_median": <float|null>,   // (L_sem - L_copy)/n_layers using strict else earliest soft
       "L_semantic_norm_iqr": [p25, p75],  // optional dispersion
       "notes": "control is France→Paris; other positives do not add new controls"
     }
   }
   ```

4. **Evaluation‑pack (derived view).**

   ```json
   "evaluation_pack": {
     "micro_suite": {
       "facts": [
         {"fact_key":"Germany→Berlin","L_copy_strict":...,"L_semantic_norm":...,"L_semantic_confirmed":...,"L_semantic_margin_ok_norm":...},
         {"fact_key":"France→Paris", ...}
       ],
       "aggregates": {
         "L_semantic_confirmed_median": ...,
         "delta_hat_median": ...,
         "n_missing": <count_of_facts_missing_milestones>
       },
       "citations": {
         "fact_rows": { "Germany→Berlin": <row_index>, ... }
       }
     }
   }
   ```

**Scope discipline.**
No new concepts, prompts remain capital‑facts; the micro‑suite only multiplies the **same** measurement to obtain robust medians/variances.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

### [x] 1.47. **Top‑2 (runner‑up) margin gate for semantic onset**

**Why.** Rank‑1 at small logit gaps can be transient and calibration‑driven, even when above‑uniform (§1.45). A **runner‑up (Top‑2) margin** ensures the answer decisively beats the second‑best token at the layer where semantics are claimed.

**What.** Add a **Top‑2 margin gate** and corresponding milestones:

* **Per‑layer scalar (norm lens):** `answer_logit_gap = logit(answer) − logit(second_best)` (already computed in §1.35 for rank‑1 layers).
* **Threshold:** `δ_top2_logit = 0.5` (logit units).
* **Milestones (norm lens):**

  * `L_semantic_top2_ok_norm` = first layer with `answer_rank==1` **and** `answer_logit_gap ≥ δ_top2_logit`.
  * `L_semantic_top2_ok_norm_frac = L_semantic_top2_ok_norm / n_layers` (nullable).
* **Roll‑up (JSON)** under a new block:

```json
"summary": {
  "semantic_gate": {
    "delta_top2_logit": 0.5,
    "L_semantic_top2_ok_norm": null,
    "L_semantic_top2_ok_norm_frac": null,
    "gap_at_L_semantic_norm": null
  }
}
```

**How.**

1. During the existing pure next‑token decode, the project already writes `answer_logit_gap` when `answer_rank==1` (§1.35). Reuse it to derive `L_semantic_top2_ok_norm` post‑sweep.
2. Persist the fields above into `summary.semantic_gate`.
3. No CSV schema changes; JSON only.
4. Add `diagnostics.surface_diagnostics_config.semantic_gate = {"delta_top2_logit": 0.5}` for provenance.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

---

### [x] 1.48. **Stability gate (run‑of‑two) for semantic onset**

**Why.** One‑layer rank‑1 “blips” can occur near lens transitions or temperature inflections. Requiring **two consecutive rank‑1 layers** improves robustness without extra forwards.

**What.** Add **run‑length** gates and “strong” composites:

* **Run‑of‑two milestone (norm lens):**
  `L_semantic_run2 = min L s.t. answer_rank(L)==1 and answer_rank(L+1)==1`.
* **Strong + stable (composite):**
  `L_semantic_strong = first L where (answer_rank(L)==1) ∧ (answer_minus_uniform(L) ≥ δ_abs from §1.45) ∧ (answer_logit_gap(L) ≥ δ_top2_logit from §1.47)`.
  `L_semantic_strong_run2 = min L s.t. L and L+1 both satisfy the **strong** criteria above.
* **Optional corroboration window (advisory only; no new forwards):** if `L_semantic_confirmed` (§1.25) exists, report whether the 2‑layer band `[L, L+1]` overlaps the ±Δ window used there.

**How.**

1. Scan the existing per‑layer arrays of `answer_rank`, `answer_minus_uniform` (§1.45), and `answer_logit_gap` (§1.35).
2. Write to JSON (all nullable when not satisfied):

```json
"summary": {
  "semantic_gate": {
    "L_semantic_run2": null,
    "L_semantic_strong": null,
    "L_semantic_strong_run2": null
  }
}
```

3. Add normalized depths in `summary.depth_fractions` for any non‑null of the above.
4. No CSV schema changes.

✅ IMPLEMENTATION STATUS: COMPLETED (active in current runs)

---

### [ ] 1.49. **Control “strong” indicator (runner‑up margin on control prompt)**

**Why.** A positive **control margin** (`p(Paris) − p(Berlin) > 0`) can be minuscule. Flagging **strong control**—where the control answer also decisively beats its runner‑up—helps separate harmless calibration wiggles from genuine lexical leakage.

**What.**

* **Per‑layer (control rows):** compute `control_top2_logit_gap = logit(Paris) − logit(second_best)` when `Paris` is top‑1 under the control prompt.
* **Threshold:** `δ_top2_logit_ctl = 0.5` (logit units; reuse §1.47 if desired).
* **Milestone & summary:**

  * `first_control_strong_pos` = first control layer with `control_margin > 0` **and** `control_top2_logit_gap ≥ δ_top2_logit_ctl`.
  * `max_control_top2_logit_gap` over control layers.
  * JSON roll‑up:

```json
"control_summary": {
  "first_control_margin_pos": null,
  "max_control_margin": null,
  "first_control_strong_pos": null,
  "max_control_top2_logit_gap": null,
  "delta_top2_logit_ctl": 0.5
}
```

**How.**

1. In the control pass (already present in §1.8), compute `control_top2_logit_gap` at layers where `Paris` is top‑1; track the earliest layer crossing both gates.
2. Persist fields above; no CSV schema changes required.

---

#### Wrap‑up

Executing the items in **Group 1** upgrades the measurement pipeline from an informative prototype to a rigour‑grade toolchain. Only after this foundation is secure should we move on to the broader prompt battery and causal‑intervention work.

[1]: https://arxiv.org/abs/1910.07467 "Root Mean Square Layer Normalization"
[2]: https://arxiv.org/abs/2303.08112 "Eliciting Latent Predictions from Transformers with the Tuned Lens"
[3]: https://neuralblog.github.io/logit-prisms/ "Logit Prisms: Decomposing Transformer Outputs for Mechanistic ..."

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
