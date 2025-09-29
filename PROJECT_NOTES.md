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

### 1.20. Cosine Milestones & Normalized Depth Summaries

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

---

### 1.21. Unified Sidecar Summaries for Prism & Tuned‑Lens

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

---

### 1.22. Machine‑Readable Measurement Guidance (Evaluation Hints)

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

---

### 1.23. Threshold sweep for copy‑collapse

**Why.** Copy‑collapse should capture when the network re‑uses **particular prompt tokens** (ID‑level, contiguous subsequence), not when the lens/temperature makes generic echoes look confident. A small **threshold sweep** tests robustness: if the first copy layer moves a lot as τ changes (e.g., 0.95→0.70), copy is fragile and more consonant with nominalist “name matching.” The sweep now also cross‑checks **raw vs norm** in a narrow window so lens‑induced “copy” is downgraded.

**What.**

1. Add multiple **strict‑copy flags** per layer in `*-pure-next-token.csv`:
   `copy_strict@0.70`, `copy_strict@0.80`, `copy_strict@0.90`, `copy_strict@0.95` (k=1 window; same margin δ=0.10; ID‑contiguous subsequence; ignore whitespace/punctuation top‑1s).
2. Record the **earliest layer** for each threshold in run JSON, both as layers and normalized depth fractions.
3. Cross‑validate the earliest strict‑copy layer at each τ using the **Windowed Raw‑vs‑Norm** check (§1.19): flag `copy_norm_only@τ=true` if strict‑copy holds under the norm lens but not under the raw lens within ±4 (or ±8 if escalated) layers.
4. Emit an **auto Markdown summary** per model (`copy-thresholds.md`) with a one‑line stability classification.

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
* **CLI:** add `--copy-thresholds 0.70,0.80,0.90,0.95` (default as shown). No other flag changes.
* **Markdown (`run-latest/copy-thresholds.md`):** one line per model, e.g.
  `Meta‑Llama‑3‑8B: L_copy_strict@{0.95,0.90,0.80,0.70} = {null, 27, 24, 23} (frac {—, .53, .47, .45}); norm_only@{0.95,0.90,0.80,0.70}={—, false, false, false}; stability=mixed.`

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

### 2.1. Multilingual prompt – preliminary pass

**Why.** Language‑independent behaviour is compatible with realism but not mandated by it; language‑dependent depths are prima facie evidence for predicate‑tied behaviour. A **per‑language gold‑token alignment** prevents tokenizer artefacts from polluting comparisons.

**What.** Translate the prompt into five major languages (matched subject–predicate order). Record normalised `L_sem / n_layers`, **`first_rank_le_{1,5,10}`**, and tuned‑lens KL thresholds; visualise variance. Use **ID‑level** gold tokens from `gold_answer_by_lang` (§1.7).

**How.**

1. Maintain a YAML of prompts keyed by ISO codes (`prompt_lang`); include `translation_ok: true/false`.
2. For each language, compute `first_id` and `pieces` and store under `gold_answer_by_lang` (§1.7).
3. Run sweeps; bar‑plot layer‑fraction variance and **rank thresholds**; highlight deviations `> 0.05` (fraction) or delays `> 2` layers in `first_rank_le_5`.
4. Prefer rank/KL‑threshold metrics over raw probabilities for cross‑language comparisons.

---

### 2.2. Predicate‑Permutation Control (Quine guard)

**Why.** Quine‑style “inscrutability of reference” argues that empirical evidence can be preserved under a systematic **re‑labelling** of terms. A global **permutation** of country (and optionally capital) names is a lightweight guard: if our heads/vectors merely track arbitrary labels, many of our metrics should look similar under the permutation; if they track **the original relation**, they should **fail** under the permutation in diagnostic ways.

**What.** *Create a permuted control set by applying a fixed bijection `π` to the set of country tokens (optionally also to capital tokens) across the **entire** prompt battery. Evaluate whether the same heads/vectors that succeed on clean prompts also succeed under `π` when “truth” is still computed in the **original** (unpermuted) mapping.*

* Add a per‑run block to JSON meta:

  ```json
  "permutation_control": {
    "enabled": true,
    "permute_countries": true,
    "permute_capitals": false,
    "perm_seed": 316,
    "perm_coverage": 100
  }
  ```
* Add per‑row fields to the CSV when permutation is active:

  * `is_permuted ∈ {0,1}`
  * `subject_id_permuted` (the ID of `π(country)`)
  * `answer_id_permuted` (the ID of `π(capital)`, if capitals are permuted)

**How.**

1. **Construct `π`.** Build a random bijection over the set of countries used in the run; record `perm_seed` and the mapping in a sidecar `perm_map.json`. (Optional: a second run with capitals also permuted.)
2. **Run the three diagnostics below** on the **same** models/prompts as the baseline:

   * **(A) Control margin under permutation.** On permuted prompts, log per‑layer
     `margin_perm = p(original_capital) − p(capital_of(π(country)))`.
     *Summary:* `first_margin_perm_pos` (first layer with `margin_perm > 0`), `max_margin_perm`. Expect **late or absent** positive margins if structure truly keys to the original relation.
   * **(B) Vector‑portability gap.** Extract the **CapitalDirection** (§3.3) on clean prompts. On permuted prompts, inject the **same** vector and log
     `Δ_true = Δ log p(original_capital)`, `Δ_perm = Δ log p(π(capital))`, and
     `portability_ratio = Δ_true / (Δ_true on clean)`.
     *Summary:* median `portability_ratio` and median `Δ_perm`. Expect **low** `Δ_perm` and **low** `portability_ratio` if the vector does not just track labels.
   * **(C) Head‑consistency drop.** Take `relation_heads.json` (§3.2) from the clean run. On permuted prompts, measure the share of those heads that still meet both criteria (high attention + ≥0.5‑bit effect on the *original* capital).
     *Summary:* `head_consistency_ratio`. Expect a **drop** under permutation.
3. **Report.** Add a one‑page Markdown summary per model with the four scalars: `first_margin_perm_pos`, `max_margin_perm`, `portability_ratio (median)`, `head_consistency_ratio`. Flag models where any scalar indicates **permutation‑robust success**, which would warrant a deeper check.

---

### 2.3. Rank‑centric prompt battery (100–1,000 country→capital items)

**Why.** Single‑prompt results can overfit tokenizer/stylistic quirks. A larger, rank‑centric battery provides robust distributions of semantic‑collapse depths without relying on lens‑calibrated probabilities, and is cheap to run on the existing pipeline.

**What.** Expand to a 100–1,000 item country→capital set using the same brevity instruction. Reuse ID‑level gold alignment (§1.7). For each model, report distributions of:

* Normalised collapse depth `L_sem / n_layers`
* `first_rank_le_{10,5,1}` (layer indices)

Prefer rank/KL thresholds for summary; avoid absolute probability comparisons across models.

**How.**

1. Maintain a simple CSV/YAML prompt list (`prompts/country_capital.csv`) with columns: `country`, `capital`, `prompt_text` (optional override). Ensure single‑token answers where feasible; record exceptions.
2. For each prompt, compute `gold_answer.first_id` via the model’s tokenizer (§1.7) and run the standard sweep (positive + control; ablation optional).
3. Aggregate per‑model summaries: histograms of `L_sem / n_layers` and counts of `first_rank_le_{10,5,1}`. Persist a `battery_summary.json` per model with `run_id`, `code_commit_sha`, and sample sizes.
4. In cross‑model write‑ups, compare distributions qualitatively (within‑family when in doubt) and emphasise rank milestones.

---

### Closing note on epistemic modesty

These variations are diagnostic, not decisive. Their job is to show which internal patterns ride above surface token variation and which do not. If the patterns hold, austere nominalism loses more credibility and we have a cleaner target set for the higher‑lift causal, multimodal, and synthetic‑language probes that might separate metalinguistic nominalism from realism in later stages.

### Caution on metrics

Raw “semantic‑collapse depth” (the layer where the gold token first becomes top‑1) is a correlational signal. Before drawing philosophical conclusions, validate any depth‑based claim with at least one causal or representational check (activation patching, tuned‑lens KL, concept‑vector alignment). See Group 3 & 4 tasks.
**Cross‑model caveat.** Absolute probabilities/entropies under a norm‑based lens are **not** comparable across models using different normalisers; use Tuned Lens (§1.12) or Logit Prism (§1.10) for cross‑model comparisons, or prefer rank/KL‑threshold metrics.

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
   * Log Δ log‑prob of the gold token (ID from §1.7).
3. Define `causal_L_sem*` as the earliest ℓ where the top‑1 flips to the gold token under that mode.
4. Write `causal_L_sem*` and **delta fields** into JSON meta; include all three per‑layer Δ values in the CSV (columns `dlogp_full`, `dlogp_attn`, `dlogp_mlp`).
5. CLI:

   * `--patching`
   * `--patching-mode {full,attn,mlp,all}` (default `all`)
   * `--corrupted-answer "Paris"`

**Pilot.** Start with a cleanly calibrated base model (e.g., Mistral‑Small‑24B or Llama‑3‑70B, where final‑row KL≈0). Report `causal_L_sem`, `causal_L_sem_attn`, and `causal_L_sem_mlp` heat‑maps around `L_sem`, and include per‑layer Δ log‑prob traces (full/attn/MLP) to separate retrieval vs construction dynamics.

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

### 3.7. Targeted Sparse Autoencoders on decisive layers — **ambitious, feature‑level causal tests** *(replaces current §3.7; no scaffolding changes to `run.py`)*

**Objective.** Move beyond depth curves to **feature‑level** evidence. Identify sparse features at the decisive layers that **predict** and **causally control** capital answers across **prompts and languages**, with **reliability gates** to avoid seed/capacity artefacts.

**Scope & placement.** One base model to start (use a model already in `run-latest/`). Layers: `L_sem − 1`, `L_sem`, `L_sem + 1` from that model’s JSON summary.

**Data.** \~50k country→capital prompts including paraphrases and ≥3 languages (small fixed list). Use the same gold‑token IDs procedure as §1.7 for each language.

**Method (single, focused script or notebook: `sae_pass.py` — edit defaults inline; no new CLI knobs):**

1. **Collect residuals.** Forward the base model once per prompt, hook the **post‑block residual** at each target layer (the same tap point used in `run.py`), and store activations to a local `.npz`.
   *Normalization and unembed follow whatever `run.py` used; do not change lensing here.*

2. **Train SAEs (reliability via replication).** Train **three** SAEs with the **same architecture** (Top‑K or L1; 8–16× overcomplete) but **different seeds** on the concatenated residuals from the 3 layers. Save encoders/decoders.

3. **Screen features (per seed).** For each latent:

   * **Predictive link.** Compute Pearson correlation between latent activation and `p_answer` at `L_sem` across the dataset. Keep latents with corr ≥ **0.2** (weak but consistent).
   * **Causal link (held‑out).** On a held‑out set, run **latent ablation** (zero its coeff) and **latent activation** (+α·decoder with α∈{0.25,0.5,1.0}) at the layer where correlation peaked. Log **Δ log‑prob** for the correct capital and for the top distractor, per language.

4. **Cross‑seed consensus (stability gate).** Match latents across seeds by **decoder‑cosine ≥ 0.8** and **Jaccard ≥ 0.5** overlap of top‑activating examples. Define `consensus_score = matched_seeds / 3`. Retain only latents with `consensus_score ≥ 2/3`.

5. **Decomposition check (non‑canonicity gate).** For each retained latent, fit a **small meta‑SAE** on its **top‑activating residual snippets**. If ≥30% of its variance is explained by ≥2 sub‑latents with **distinct** activation profiles, mark it as a **bundle**; otherwise **unitary**.

6. **Negative & permutation controls.**

   * **Predicate‑permutation:** Repeat causal tests under the §2.2 permutation prompts; expect **loss** of effect on the *original* correct capital.
   * **France‑control:** Ensure steering does **not** increase `p(Berlin)` in “France → ?” prompts.

7. **Success criteria (advance only if all hold).**

   * Median **Δ log‑prob ≥ 0.5 bits** for the correct capital at some α, **and** ≤0 for unrelated token set (months/colors) on the same prompts.
   * **Portability:** effect remains positive across **≥3 languages** with median Δ ≥ 0.25 bits each.
   * **Stability:** `consensus_score ≥ 2/3`.
   * **Controls:** effect **drops** under permutation control.

8. **Outputs (written alongside `run-latest/…`):**

   * `sae_config.json` (arch, sparsity, seeds, layers used),
   * `sae_features.npz` (enc/dec),
   * `feature_manifest.json` entries:
     `{feature_id, layer, consensus_score, unitary_or_bundle, Δ_logp_median, Δ_logp_by_lang, α_star, control_drop}`,
   * `sae_reliability.json` (cross‑seed matches, decomposition flags, validation snippets).

> *Implementation note:* Keep this pass isolated (separate script/notebook). Defaults live in the file; edit them directly for each iteration. No CI, no profiles, no extra flags.

---

### 3.8. SAE reliability & consensus scoring *(implementation detail; pairs with §3.7)*

**Goal.** Standardize how stability and non‑canonicity are reported, without adding harness.

**Procedure.**

* **Consensus across seeds.** Build bipartite matches between latents from seed‑A and seed‑B by **decoder‑cosine**; extend to seed‑C transitively. Compute `consensus_score` per latent (0–1).
* **Decomposition flag.** From the meta‑SAE (§3.7‑5), record `unitary` vs `bundle`.
* **Run‑level summary.** Print to console and save `sae_reliability.json` with: median `consensus_score`, `% bundle`, and a short list of advanced features with their scores.

**Why it matters.** Prevents over‑claiming from one‑off latents; makes later philosophical claims depend on **replicable** causal features.

---

### 3.9. Cross‑model SAE universality probe *(exploratory; ambitious but compact)*

**Objective.** Test for **model‑independent** real patterns by checking whether **analogous** sparse features exist—and are **causally efficacious**—in **two different base models**.

**Setup.** Choose two models already run in `run-latest/`. Train/obtain SAEs at `L_sem` for each (repeat §3.7 on both, but you can skip the meta‑SAE on the second model if compute is tight).

**Method (single notebook/script `universality_probe.py`):**

1. **Activation signatures.** For each model, compute per‑feature **activation signatures** over a **shared 500‑prompt set** (mean activation per prompt).
2. **Alignment & matching.** Align the two feature collections using **SVCCA** (or RSA on signatures). Select top‑k **matched pairs** (highest similarity with one‑to‑one matching).
3. **Causal concordance.** For each pair, **steer** each model with its own feature (+α at the same relative layer) and compute standardized effect sizes (Δ log‑prob / σ across prompts).
4. **Universality metrics (report 3 scalars + plots):**

   * `feature_match_ratio = matched_pairs / min(nA, nB)`,
   * `effect_size_corr` (Pearson over prompts between the two models’ effect size vectors),
   * `universality_pass = share of pairs with positive effect in both models`.
5. **Interpretation guardrails.** Treat positive results as **real‑pattern** evidence; negative results are informative—record and move on. No weight‑sharing or latent transfer attempted.

**Outputs.** `universality_summary.json` (the three metrics), plus a small PDF/PNG plot showing matched pairs and effect‑size scatter.



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

---

**Operational notes (for all four sections):**

* Do **not** modify `run.py`. Each pass is a **single, focused script or notebook** with **defaults defined inside**; edit once per iteration.
* Reuse artifacts from `run-latest/` (gold IDs, `L_sem`, prompts) to stay consistent with the main sweep.
---



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

### 4.7. Zero‑Shot Flag‑Grounding Check (Minimal)

**Why.** To pressure **purely metalinguistic** explanations, add a small, low‑lift probe that routes **non‑linguistic** evidence (country flags) into the text‑only pipeline without fine‑tuning the LLM. A positive result (even if modest) strengthens the case that the “capital‑of” machinery is not *only* about word‑tokens.

**What.** *Two quick tests using frozen components:*

* **(A) VLM zero‑shot baseline (sanity).** With an off‑the‑shelf VLM (e.g., Qwen‑VL or LLaVA) **without fine‑tuning**, prompt each flag image with “Which country’s flag is this?”; take the top country string and feed it into the **standard text‑only** capital prompt. Log whether the overall pipeline returns the correct capital. *(This is a control that confirms the image→text hand‑off works; it is not itself an interpretability result.)*
* **(B) Minimal vision→LM bridge (no LM training).** Use a frozen **OpenCLIP** image encoder to get a flag embedding `z_img`. Learn a **linear projector** `P` from CLIP **text** embeddings of country names to the LLM’s **residual space at the subject position** using **≈100** (country name) pairs (least‑squares; no LM gradients). At inference, project `z_img` via `P` and **inject** at `L_sem−1` in the sentence “The capital of ⟨IMG⟩ is …”. Measure Δ log‑prob for the correct capital vs the top distractor.

**How.**

1. **Data.** 50 countries with high‑quality flag images (SVG or PNG). Hold out 10 for evaluation of the bridge.
2. **Bridge fit (text‑only supervision).** Compute CLIP **text** embeddings for the 40 training country names; compute the LLM **residual vectors** at the subject token for the same names; solve `min_P ‖P·clip_text − resid‖²`. Record `n_pairs`, `seed`, and norms in JSON meta under `vision_bridge.*`.
3. **Injection.** For each held‑out flag, compute `z_img`, project `r̂ = P·z_img`, and add `α·r̂` at `L_sem−1` (scan `α ∈ {0.25, 0.5, 1.0}`) before decoding with the **same** lens used in Group 1. Log Δ log‑prob for the correct capital and for a frequency‑matched distractor.
4. **Success criteria.** Report median Δ log‑prob ≥ **0.25 bits** at some `α` across the 10 held‑out flags, with ≤ **0** median gain for distractors. Include a per‑model summary block:

   ```json
   "vision_bridge": {
     "clip_model": "openclip_ViT-B/32",
     "n_pairs": 40,
     "alpha_grid": [0.25, 0.5, 1.0],
     "median_delta_logp_bits": 0.31,
     "median_delta_logp_distractor_bits": -0.02
   }
   ```
5. **Scope guard.** This is a **one‑day spike**: no VLM fine‑tuning, no LM weight updates, only a linear map and a single‑layer injection. If it fails noisily, defer multimodal work to §5.1.


---

### 4.8. Feature‑steering side‑effects & risk profile *(minimal instrumentation, strong safeguards)*

**Purpose.** Show that steering a “capital‑of” feature **does what is intended** with **limited collateral effects**—critical for credible philosophical claims.

**Method (small script/notebook `steering_profile.py`; uses features advanced in §3.7):**

* **Neutral set.** 200 simple sentences unrelated to geography (in‑repo text file).
* **For each advanced feature and α ∈ {0.25, 0.5, 1.0}:**

  1. **Target efficacy.** Re‑measure Δ log‑prob for the correct capital on the capital prompts; record the minimal α that meets the §3.7 threshold (call it `α*`).
  2. **Collateral metrics on neutral set:**

     * `KL_drift_bits`: mean KL(logits\_steered ∥ logits\_base),
     * `PPL_delta`: relative perplexity change,
     * `unrelated_token_shift`: mean Δ log‑prob on a frequency‑matched distractor list,
     * `target_vs_distractor_ratio`: (Δ on correct capital) / (max Δ over top‑5 distractors) on the capital prompts.
* **Report.** Emit `steering_profile.json` per feature with metrics at each α and highlight `α*` with the **lowest** KL\_drift that still satisfies target efficacy.

**Success criterion.** At `α*`, `KL_drift_bits` small (target: ≤0.05 bits), `PPL_delta` near 0, `unrelated_token_shift ≤ 0`, and `target_vs_distractor_ratio ≥ 2`.

**Why this matters.** Prevents over‑interpreting features that “work” only by globally perturbing the model; strengthens the case that a **specific mechanism** underlies the observed competence.

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
