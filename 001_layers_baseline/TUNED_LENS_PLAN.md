# Integrate a Tuned Lens (translator‑in‑d) 

**Goal.** Replace brittle raw/norm logit‑lens readouts with a **per‑layer affine translator** that maps each (properly normalized) residual to a representation whose decoded distribution—using the model’s own unembedding—**matches the model’s final next‑token distribution** at the same position. This **Tuned Lens (TL)** yields earlier and more faithful rank/KL milestones and stabilizes depth‑wise trajectories against basis drift. ([arXiv][1])

**Deliverables (per model).**

* TL weights `{U_ℓ, V_ℓ, c_ℓ}` for all **post‑block** layers ℓ (final post‑block translator fixed to identity).
* Sidecar CSVs (schemas identical to current lens CSVs) with tuned‑lens metrics.
* JSON provenance (training snapshot, config, acceptance gates, metrics).
* Optional HF dataset repo artifacts (`tuned_lenses/<model>/…`) with upload/download scripts.

---

## 1) Definitions and placement

* **Where translators apply.** Exactly where the current **NormLensAdapter** reads: **post‑block residual**, normalized with the **architecture‑correct** norm (pre‑norm: next block’s `ln1` or `ln_final`; post‑norm: current block’s `ln2`) in fp32. TL operates **after** normalization.
* **Decoder is fixed.** Decode with the model’s tied unembedding `(W_U, b_U)`—the same fp32 “analysis” copy used in `run.py`. Translators **do not** introduce a new decoder.
* **Translator parameterization (identity‑plus‑low‑rank).**

  $$
  \tilde h^{(\ell)} = h_{\text{norm}}^{(\ell)} + \big(h_{\text{norm}}^{(\ell)} U_\ell\big) V_\ell^\top + c_\ell,\ \ \ U_\ell, V_\ell \in \mathbb{R}^{d\times k}
  $$

  Use width‑scaled rank $k$ (see §4). The final post‑block translator is **frozen to identity** ($U=V=0, c=0$).
* **Objective.** Minimize **CE/KL** between the tuned distribution $P^{(\ell)}$ and the model’s **final** distribution $P_{\text{final}}$ at the **same position** (distillation). ([arXiv][1])

---

## 2) Training data

**Source (pinned).** Hugging Face **`HuggingFaceFW/fineweb-edu`**, **subset** `CC-MAIN-2025-26` (Jan–Jun 2025), **revision** `v1.4.0`, **split** `train`, **streaming** mode. This revision explicitly adds the 2025‑05 … 2025‑26 dumps and documents all available subsets. ([Hugging Face][2])

Use the model’s own tokenizer. Pack to sequence length L=512 (safe & fast across models) with left‑to‑right contiguous packing and no EOS insertion beyond what tokenization adds.

**Budget (per model, fixed).**

* **Tokens:** **16,000,000** tokens.
* **Pack length:** **512** tokens per packed sequence.
* **Positions per sequence:** **8** positions, sampled uniformly from the last **60–95%** of the context.

**Why this budget.** At seq‑len 512, 16M tokens ≈ 31,250 sequences. With an **effective batch** of 32 sequences (see §4), this yields \~**975 optimizer steps**, enough for stable translators while keeping single‑GPU wall‑time modest (model forward under `no_grad`, translators only get gradients).

**Loader call (for implementers).**

```python
from datasets import load_dataset
ds = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="CC-MAIN-2025-26",         # subset/config
    split="train",
    streaming=True,
    revision="v1.4.0"               # pinned dataset version
)  # iterate -> tokenize with the model's tokenizer; pack to length 512
```

(Keep a **content hash** of sampled shard indices and record `repo_id`, `name`, `revision` in provenance.) ([Hugging Face][2])

---

## 3) Losses, regularization, and (optional) preconditioning

**Primary loss.**

$$
\mathcal{L}_{\text{CE}} = \operatorname{CE}\big(P_{\text{final}}(\cdot \mid x, t),\ P^{(\ell)}(\cdot \mid x, t)\big).
$$

**Regularizers.**

* **Smallness:** L2 on $U_\ell, V_\ell, c_\ell$ with weight **1e‑4**.
* **Depth smoothness:** $\sum_\ell \|U_{\ell+1}-U_\ell\|_F^2 + \|V_{\ell+1}-V_\ell\|_F^2$ with weight **1e‑4**.
* **(Optional) Prism/ZCA anchor:** If a **Prism** rotation $R$ is present for the model, compose in the $R$-basis (no extra CLI): internally whiten `h_norm` by $R$ and parameterize $U,V$ in that basis. Prism remains a shared‑decoder baseline and stabilizer; no separate flags needed. ([LessWrong][3])

**Last‑layer policy.** The last post‑block translator is fixed to identity; diagnostics compare **norm‑lens after scalar temperature** to the model head to confirm calibrated agreement (report `kl_after_temp_bits`). ([arXiv][1])

---

## 4) Capacity, numerics, and optimization

* **Rank schedule (width‑scaled):** $k = \mathrm{clip}(\lfloor d/32 \rfloor,\ 64,\ 256)$.
* **Micro‑batch:** **8 sequences** (seq‑len 512).
* **Grad accumulation:** **4** (effective batch = 32 sequences).
* **LR schedule:** AdamW (lr **1e‑3**, betas **0.9,0.999**, weight decay **1e‑4**), **cosine decay to 0** with **warmup steps = ⌈0.1 × total\_steps⌉** (computed from §2 budget and this batch shape).
* **Numerics:** Run model forward under `torch.no_grad()`; extract normalized residuals; compute translator logits and CE in **fp32**. No AMP on translators.
* **Total steps (derived, fixed budget):** `total_steps = ceil((16_000_000 / 512) / (8 * 4)) = 975`. Thus **warmup = 98** steps (exact).
* **Early stopping:** Track mean KL(P_TL‖P_final) on a held‑out shard at depth percentiles {25,50,75} (layers mapped to percentiles). Stop on no improvement for N=3 validations.

---

## 5) Implementation plan (files, APIs)

### New module

`001_layers_baseline/layers_core/tuned_lens.py`

**Status:** ✅ Implemented (`layers_core/tuned_lens.py` with translator, preconditioner, save/load helpers).

* `class TunedTranslator(nn.Module)`: per‑layer $U_\ell, V_\ell, c_\ell$; `forward(h_norm, layer_idx) → tuned_logits`.
* `class TunedLensAdapter`: mirrors `NormLensAdapter`; takes `UnembedContext` + `TunedTranslator`; `decode_next_token(h_norm, layer_idx)`.
* `save_tuned_lens(path)` / `load_tuned_lens(path)` writing `weights.pt`, `precond.pt` (if Prism used), and `provenance.json`.

### Training script (single‑flag CLI)

`001_layers_baseline/tuned_lens_fit.py`

**Status:** ✅ Implemented (`tuned_lens_fit.py` trains translators with streaming loader and saves artifacts).

**CLI:**

```
--model-id <hf_repo/model>
```

Everything else uses the **fixed defaults** above:

* Dataset: `HuggingFaceFW/fineweb-edu`, `name="CC-MAIN-2025-26"`, `revision="v1.4.0"`, `split="train"`, streaming. ([Hugging Face][2])
* Tokens: 16,000,000; seq‑len: 512; positions per seq: 8; micro‑batch: 8; grad‑accum: 4.
* Rank: `clip(d//32, 64, 256)`; regularizers as in §3; LR/schedule as in §4.
* **Optional Prism preconditioning** auto‑enabled if a Prism artifact exists under `001_layers_baseline/prisms/<clean_model_name>/` (no flags). ([LessWrong][3])
* **Save path:** `001_layers_baseline/tuned_lenses/<clean_model_name>/`.

**Loop sketch (deterministic seed=316):**

1. Stream dataset; tokenize with the **model’s tokenizer**; pack to 512; sample 8 positions/seq (last 60–95%).
2. `no_grad()` model forward → final logits at sampled positions; extract **normalized post‑block residual** at a randomly sampled layer ℓ (uniform over post‑blocks).
3. Compute tuned logits via TL adapter; CE to final; **update only layer ℓ** translators.
4. Repeat until exactly **16M tokens** consumed; run validation every 100 steps on a small held‑out shard (same config), reporting percentile KLs; save the best checkpoint.

### Integration in `run.py`

**Status:** ✅ Implemented (auto-load tuned lens, dual-lens emission, tuned sidecar CSVs).

* At startup, **auto‑load** TL from `001_layers_baseline/tuned_lenses/<clean_model_name>/`.

  * **If present:** run the standard norm‑lens sweep **and** a TL sweep **from cached residuals**; summaries prefer TL where **acceptance gates** pass (§6).
  * **If missing:** **fail fast** with a clear message: “Tuned Lens not found for `<clean_model_name>`. Train it with `tuned_lens_fit.py --model-id <hf_repo/model>`.”
* Sidecars: write `*-records-tuned.csv`, `*-pure-next-token-tuned.csv` (same schema as norm; or a merged file with `lens ∈ {norm,tuned}` if preferred internally).

### Storage layout (per model)

```
001_layers_baseline/tuned_lenses/<clean_model_name>/
  weights.pt                  # state_dict with U_l, V_l, c_l
  precond.pt                  # optional Prism/ZCA info (if used)
  provenance.json             # see §7
```

### HF Hub mirrors

* `scripts/tuned_lens_upload.py` / `scripts/tuned_lens_download.py` modeled on Prism scripts; default repo `logos-in-layers-tuned`; path `tuned_lenses/<clean_model_name>/…`. (Use the same `huggingface_hub` utilities as the Prism scripts.)

  **Status:** ✅ Implemented (scripts generate `INDEX.json`, compute checksums, and upload/download tuned lens artifacts).

---

## 6) Validation, reporting, and acceptance gates

**Compute both baselines on a small held‑out prompt suite:**

* **Norm‑lens (last layer with scalar temperature)** vs model head → `kl_after_temp_bits`.
* **Tuned‑lens** at all layers.

**KPIs (persist to run JSON + TL provenance).**

* `median_delta_kl_bits@{25,50,75}%` = median over positions of $\mathrm{KL}_{\text{norm}} - \mathrm{KL}_{\text{TL}}$ at depth percentiles 25/50/75.
* `earliness_shift_first_rank_le_{10,5,1}` = positive = earlier layer index under TL.
* `last_layer_kl_after_temp_bits` ≤ 0.05 (agreement at final post‑block).
* `rank1_coverage_gain` = % layers where answer moves into top‑k under TL vs norm.

**Acceptance gates (to prefer TL in summaries).**

* **Gate‑A:** `median_delta_kl_bits` ≥ {0.2, 0.3, 0.3} bits at {25,50,75}%.
* **Gate‑B:** `earliness_shift_first_rank_le_10` ≥ **+4** layers, and `≤5` ≥ **+2** layers on ≥60% of prompts.
* **Gate‑C:** `last_layer_kl_after_temp_bits` ≤ **0.05**.
* **Gate‑D:** No percentile shows **negative ΔKL** worse than **−0.1** bits.

(Also ensure **Raw‑vs‑Norm sanity** does not regress when TL is used in reporting.)

**QA hooks already in project.**

* **Raw‑vs‑Norm sanity (§1.4)**: ensure no regression in lens_artifact_risk
* **Last‑layer consistency (§1.6)**: TL must not “improve” the last post‑block distribution (it’s forced identity); the model head is the source of truth.
* **Cosine curves (§1.5)**: include cos_to_final for TL logits; interpret as “distance to the decision boundary”.

---

## 7) Provenance schema (append to run JSON; write in TL folder)

```json
"tuned_lens": {
  "enabled": true,
  "model_id": "<hf_repo/model>",
  "clean_model_name": "<name>",
  "sha": "<git_commit>",
  "seed": 316,

  "dataset": {
    "repo_id": "HuggingFaceFW/fineweb-edu",
    "name": "CC-MAIN-2025-26",
    "split": "train",
    "revision": "v1.4.0",
    "sample_description": "streamed; tokenized with model tokenizer; packed to 512; positions=8 in [0.6,0.95] of context",
    "content_hash": "<sha256 of sampled shard indices>"
  },

  "budget": { "tokens": 16000000, "seq_len": 512, "positions_per_seq": 8 },
  "batching": { "micro_batch_seqs": 8, "grad_accum": 4, "effective_batch_seqs": 32 },
  "capacity": { "rank_k": "clip(d//32,64,256)" },

  "optim": {
    "lr": 0.001, "betas": [0.9,0.999], "weight_decay": 0.0001,
    "schedule": "cosine_to_zero",
    "total_steps": 975, "warmup_steps": 98
  },

  "regularizers": { "l2_smallness": 0.0001, "smooth_depth": 0.0001, "precond": "auto_if_prism" },
  "preconditioner": { "type": "prism|none", "path": "001_layers_baseline/prisms/<clean_model_name>/" },

  "val_kl_bits@{25,50,75}%": [0.42, 0.61, 0.58],
  "kpis": {
    "median_delta_kl_bits@{25,50,75}%": [0.18, 0.33, 0.31],
    "earliness_shift_first_rank_le_{10,5,1}": [5, 3, 1],
    "last_layer_kl_after_temp_bits": 0.02,
    "rank1_coverage_gain": 0.11
  },
  "acceptance": {"gate_A": true, "gate_B": true, "gate_C": true, "gate_D": true}
}
```

---

## 8) Analysis & plots

* Plot **KL vs depth** (norm vs TL) with **`first_rank_le_{10,5,1}`** crossings.
* Plot **cos\_to\_final** (norm vs TL) to visualize “rotation → amplification.”
* Report **ΔKL percentiles** and **earliness shifts**; show **norm‑lens after temperature** vs TL at the last post‑block to avoid overstating TL gains. ([arXiv][1])

---

## 9) Non‑goals / guardrails

* No non‑linear translators; linear, identity‑plus‑low‑rank only.
* Single translator per layer (no per‑position translators).
* Decoder remains tied and frozen.
* Prism stays an optional shared‑decoder baseline/preconditioner; auto‑used if present; no extra flags. ([LessWrong][3])

---

## 10) Example nuance (adapter forward)

```python
# h_norm: [B, d] normalized residual at a post-block layer
def decode_next_token(h_norm, layer_idx):
    U, V, c = self.U[layer_idx], self.V[layer_idx], self.c[layer_idx]  # [d,k], [d,k], [d]
    tuned = h_norm + (h_norm @ U) @ V.T + c
    logits = tuned @ unembed_ctx.W + unembed_ctx.b
    return logits.float()  # softmax in fp32 for CE/KL
```


### 11) Runtime and storage expectations

* **Compute shape.** Training processes **16,000,000 tokens** per model at **seq‑len 512**, with **micro‑batch = 8** sequences and **grad‑accum = 4** (effective batch = 32 sequences). This yields **total\_steps = 975** with **warmup\_steps = 98** (cosine to zero thereafter).
* **Per‑layer compute overhead (sweep / inference).** TL adds two low‑rank projections per layer:

  * `h_norm @ U_ℓ` (d×k) and `(…) @ V_ℓ^T` (k×d) plus bias/adds → about **4·d·k FLOPs** per token per layer.
  * With the fixed rank schedule $k=\mathrm{clip}(\lfloor d/32\rfloor, 64, 256)$, this is small compared to the unembedding matmul and negligible versus the forward pass (the sweep reuses cached residuals).
* **Parameter count and disk footprint (identity‑plus‑low‑rank).** Per layer: **2·d·k + d** parameters (fp32).

  * Typical 4k‑d, 32‑layer model (e.g., many 7–8B bases), $k=128$: \~**33.7M params** → \~**129 MB**.
  * Typical 8k‑d, 80‑layer model (e.g., many \~70B bases), $k=256$: \~**336.2M params** → \~**1.25 GiB**.
  * 4k‑d, 60‑layer (e.g., some \~30–35B bases), $k=128$: \~**63.2M params** → \~**241 MB**.
* **Optimizer state during training.** AdamW moments roughly **2× parameters**; gradients another **1×**. Expect peak TL‑specific optimizer memory ≈ **\~3× parameter bytes** (on the device doing TL updates). The base model runs under `no_grad` and is not optimized.
* **Sweep storage overhead.** Two additional CSV sidecars per run (`*-records-tuned.csv`, `*-pure-next-token-tuned.csv`) with the **same schema** as norm‑lens; size is negligible relative to model artifacts.
* **Device placement.**

  * **Training:** keep TL parameters + optimizer on the training device; base model forward in `no_grad`. Hook only the **selected layer ℓ** per batch to bound activation memory.
  * **Inference (sweep):** load TL weights to the inference device. For very large TLs, optionally keep `U,V,c` on CPU and move **one layer at a time** to the device during decoding.
* **I/O.** Artifacts live under `001_layers_baseline/tuned_lenses/<clean_model_name>/`:

  * `weights.pt` (TL parameters), `precond.pt` (if Prism/ZCA used), `provenance.json`.
  * Optional HF mirrors under `tuned_lenses/<clean_model_name>/…` (dataset repo).
* **Determinism.** Fixed **seed = 316** for TL parameter initialization and sampler order. Provenance records the dataset revision, subset name, and a **content hash** of sampled shard indices for reproducibility.

---

### 12) Step‑by‑step “getting started”

1. **(Optional) Prepare Prism for the model**
   If a Prism artifact exists at `001_layers_baseline/prisms/<clean_model_name>/`, TL will auto‑use it as a preconditioning basis. If not present, skip—TL proceeds without Prism.

2. **Train the Tuned Lens**

   ```
   python 001_layers_baseline/tuned_lens_fit.py \
     --model-id <hf_repo/model>
   ```

   Defaults:

   * Dataset: `HuggingFaceFW/fineweb-edu`, subset `CC-MAIN-2025-26`, revision `v1.4.0`, split `train`, streaming; tokenize with the model’s tokenizer; pack to 512; sample 8 positions per sequence (last 60–95% of context).
   * Budget: 16,000,000 tokens; micro‑batch 8; grad‑accum 4; total\_steps 975; warmup 98.
   * Rank schedule: $k=\mathrm{clip}(\lfloor d/32\rfloor, 64, 256)$; smallness and depth‑smoothness regularizers both **1e‑4**; AdamW with lr **1e‑3**, betas **(0.9, 0.999)**, weight decay **1e‑4**, cosine to zero.
   * Last post‑block translator is fixed to identity.
   * Artifacts saved to `001_layers_baseline/tuned_lenses/<clean_model_name>/` with `weights.pt`, optional `precond.pt`, and `provenance.json`.

3. **(Optional) Mirror artifacts to Hugging Face**

   ```
   python scripts/tuned_lens_upload.py \
     --repo-id <org>/logos-in-layers-tuned \
     --models <clean_model_name>
   ```

   Use the companion download script for CI or multi‑machine runs.

4. **Run the layer sweep with TL auto‑integration**
   Execute the standard probe runner as usual. The runner:

   * Auto‑loads TL from `001_layers_baseline/tuned_lenses/<clean_model_name>/`.
   * Runs the **norm‑lens sweep** and a **TL sweep** from cached residuals.
   * Writes sidecars `*-records-tuned.csv`, `*-pure-next-token-tuned.csv` and mirrors TL provenance into the run JSON.
     If no TL is found for the model, the runner **fails fast** with a clear instruction to train TL (step 2).

5. **Review acceptance gates (built into the run JSON)**
   TL is **preferred in summaries** only if all gates pass:

   * Gate‑A: `median_delta_kl_bits ≥ {0.2, 0.3, 0.3}` at depth percentiles `{25,50,75}`.
   * Gate‑B: `earliness_shift_first_rank_le_10 ≥ +4` layers and `≤5 ≥ +2` on ≥60% of prompts.
   * Gate‑C: `last_layer_kl_after_temp_bits ≤ 0.05`.
   * Gate‑D: no percentile with negative ΔKL worse than −0.1 bits.
     The JSON also reports `rank1_coverage_gain` and percentile KLs for quick validation.

6. **Troubleshooting & guardrails**

   * **Missing TL artifacts:** ensure `weights.pt` and `provenance.json` exist under the expected path; re‑run step 2.
   * **Dataset access issues:** verify HF auth and that `HuggingFaceFW/fineweb-edu` `CC-MAIN-2025-26` at `v1.4.0` is accessible.
   * **Memory pressure (very large models):** set the runner to **stream TL layer weights** from CPU to device per layer during decoding; training already hooks only one layer per batch.
   * **Acceptance gates failing:** TL will still be written but not preferred in summaries. Re‑train (same defaults) or check for Prism presence; ensure the dataset stream reached the full **16M tokens**.
   * **Reproducibility:** the `provenance.json` must include `repo_id`, `name` (subset), `revision`, and `content_hash` of sampled shards, plus `sha` (project Git commit).

---

### 13) Publishing Tuned Lenses to Hugging Face

**Objective.** Make the trained Tuned Lens artifacts easy to discover, reproduce, and consume—by this project and by others—while keeping licensing and provenance airtight.

#### 13.1 Repository type, naming, and structure

* **Repo type:** Hugging Face **dataset** repo (not a model repo), e.g. `forms-and-features/logos-in-layers-tuned-lenses`. Dataset repos are ideal for arbitrary binary artifacts and multi-model trees.
* **Top-level layout:**

  ```
  tuned_lenses/
    meta-llama/Meta-Llama-3-8B/
      weights.pt
      precond.pt               # optional (Prism/ZCA); omit if unused
      provenance.json
      README.md                # short per-model card (auto-generated)
    meta-llama/Meta-Llama-3-70B/
      ...
    mistralai/Mistral-7B-v0.1/
      ...
    google/gemma-2-9b/
      ...
    01-ai/Yi-34B/
      ...
    Qwen/Qwen3-14B/
      ...
    Qwen/Qwen2.5-72B/
      ...
  INDEX.json                   # machine-readable map for consumers
  LICENSES/
    LLAMA_3_LICENSE.txt
    GEMMA_2_TERMS.txt
    QWEN_LICENSE.txt
    MISTRAL_LICENSE.txt
    YI_LICENSE.txt
  README.md                    # dataset card (see 13.3)
  ```
* **`INDEX.json` schema (one entry per model):**

  ```json
  {
    "meta-llama/Meta-Llama-3-8B": {
      "path": "tuned_lenses/meta-llama/Meta-Llama-3-8B",
      "sha256": {
        "weights.pt": "<sha256>",
        "precond.pt": "<sha256-or-null>",
        "provenance.json": "<sha256>"
      },
      "tl_version": "v2025.09.24",
      "model_commit": "<hf_model_snapshot_sha>",
      "tokenizer_commit": "<hf_tokenizer_snapshot_sha>"
    },
    "...": { }
  }
  ```

#### 13.2 Versioning & releases

* **Release tag format:** `vYYYY.MM.DD` (e.g., `v2025.09.24`). Treat releases as **immutable**; never overwrite artifacts—add a new tag for updates (e.g., re-trains on new data).
* **Revision pinning:** Each `provenance.json` must pin:

  * base **model HF repo id** and **snapshot commit**,
  * **tokenizer** snapshot (if different),
  * **training dataset** (`HuggingFaceFW/fineweb-edu`, `name="CC-MAIN-2025-26"`, `revision="v1.4.0"`),
  * **budget** (exact 16,000,000 tokens), **seed** (316), **rank schedule**, and acceptance KPIs.
* **Checksums:** Include **SHA-256** for every file in `INDEX.json` and repeat within each `provenance.json` for redundancy.

#### 13.3 Dataset card (top-level `README.md`)

Include:

* **What this is:** Per-layer affine **Tuned Lens** translators for specific open-weight LLMs; decoder tied to original unembedding.
* **How they were trained:** 16M tokens on **FineWeb-Edu** (`CC-MAIN-2025-26`, `revision=v1.4.0`), seq-len 512, 8 positions/seq, uniform depth sampling, identity-plus-low-rank with rank $k=\mathrm{clip}(\lfloor d/32\rfloor, 64, 256)$, AdamW, cosine schedule with 10% warmup, last post-block fixed to identity.
* **Acceptance gates:** List the four gates (ΔKL percentiles, earliness shifts, last-layer KL after temperature, no negative ΔKL worse than −0.1). Mark pass/fail **per model**.
* **Licenses & terms:** Summarize upstream model licenses and include full texts in `LICENSES/`. State that the lenses are **model-specific** and must be used under the corresponding base model’s license/terms.
* **Provenance & reproducibility:** Point to `provenance.json` and `INDEX.json`. Clarify that **no project evaluation prompts** were included in training.
* **Citation & contact:** How to cite this artifact; issue tracker / contact for corrections.

#### 13.4 Per-model `README.md`

Each model folder should include a short card with:

* Model id + exact HF snapshot commit,
* TL version, parameter count (layers × (2·d·k + d)), on-disk size,
* Whether **Prism/ZCA** preconditioning was used,
* KPI table (ΔKL at {25,50,75}%, earliness shifts, last-layer KL after temp, rank1 coverage gain),
* Quick-start code (see 13.6).

#### 13.5 Upload & CI automation

* **Upload:** Reuse Prism’s pattern. Provide:

  * `scripts/tuned_lens_upload.py` (uploads `tuned_lenses/` subtree, regenerates `INDEX.json`, computes SHA-256, commits with tag `vYYYY.MM.DD`).
  * `scripts/tuned_lens_download.py` (downloads either one model folder or the entire `tuned_lenses/` subtree at a specified **revision**).
* **CI (optional but recommended):**

  * On new TL artifacts in `main`, a CI job:

    1. regenerates `INDEX.json` + checksums,
    2. cross-checks `provenance.json` fields,
    3. runs a tiny **verification suite** (loads weights, decodes 10 random layers on 100 random positions; asserts `Gate-C` and ΔKL>0 at median),
    4. publishes to HF with a new tag.

#### 13.6 Consumption patterns (examples)

**A) Using the project loader (recommended):**

```python
from huggingface_hub import snapshot_download
from layers_core.tuned_lens import load_tuned_lens

local = snapshot_download(
    repo_id="forms-and-features/logos-in-layers-tuned-lenses",
    repo_type="dataset",
    allow_patterns="tuned_lenses/meta-llama/Meta-Llama-3-8B/*",
    revision="v2025.09.24",   # pin a release
)
tl_path = f"{local}/tuned_lenses/meta-llama/Meta-Llama-3-8B"
tuned = load_tuned_lens(tl_path)  # returns TunedTranslator + metadata
```

**B) Auto-integration in `run.py`:** simply place the downloaded folder under
`001_layers_baseline/tuned_lenses/<clean_model_name>/` and run the probe script; it auto-loads TL and emits `*-tuned.csv` sidecars.

#### 13.7 Licensing & attribution

* **Bundle upstream licenses/terms** in `LICENSES/` and link to them from the dataset card.
* **Folder-level notices:** Add a short `NOTICE` file in each model folder referencing the upstream license, tokenizer license (if separate), and any additional attribution requirements.
* **Usage reminder:** State clearly that **decoder tying** makes the lens specific to the exact model (weights + tokenizer). Using different weights/tokenizers is unsupported.

#### 13.8 Quality & safety notes

* **No PII or sensitive content** is stored; artifacts are pure linear weights + calibration stats.
* **Determinism:** `provenance.json` records seed (316), dataset revision, subset name, and **content hash** of sampled shards.
* **Known limitations:** Linear translators; identity at last post-block; results depend on the base model snapshot and tokenizer revision.

#### 13.9 Roadmap for updates

* **When to re-train:** If base model or tokenizer snapshot changes, or if a new FineWeb-Edu revision/subset becomes the project default.
* **Extended coverage:** Add lenses for additional sizes/variants as needed (e.g., instruct-tuned bases, with a clear label that training used base data and distillation to each model’s final head).
* **Community alignment:** Open an issue/PR upstream (e.g., in the `tuned-lens` docs) to list this repo under “pre-trained lenses,” enabling broader discoverability.

### Smoke test

####  Steps

  1. HF auth (if needed)
     ```
     huggingface-cli login       # only once, for gated models
     ```

  2. Activate the project venv
     ```
     cd logos-in-layers
     source venv/bin/activate
     ```

  3. Run the trainer
     ```
     python 001_layers_baseline/tuned_lens_fit.py --model-id mistralai/Mistral-7B-v0.1
     ```
      - The script streams FineWeb‑Edu (CC-MAIN-2025-26) and writes artifacts to `001_layers_baseline/tuned_lenses/Mistral-7B-v0.1/`.
      - Expect a 16M-token run (about 975 steps with the default LR schedule). On CPU this can take a while; use `--device mps` if you’re on Apple Silicon and want the faster path.

  4. Inspect outputs
     The folder should now contain:
     ```
     weights.pt
     provenance.json
     (optional) precond.pt
     ```
     provenance.json will record the dataset revision, seed, rank, etc.

  5. Verify integration
     Re-run the analysis script to see tuned sidecar CSVs emitted automatically:
     ```
     cd 001_layers_baseline
     python run.py mistralai/Mistral-7B-v0.1
     ```
     In `run-latest/` you should now see the tuned CSVs (*-records-tuned.csv, *-pure-next-token-tuned.csv) and the run JSON will include a tuned_lens block referencing the artifacts.

  Once that works, you can exercise the upload script:
  ```
  python scripts/tuned_lens_upload.py --repo-id <user>/logos-in-layers-tuned
  ```
  (Optionally stage just the new model before running it.)

---

## References

* **Tuned Lens (paper & code).** Belrose et al., *Eliciting Latent Predictions from Transformers with the Tuned Lens*; AlignmentResearch/**tuned‑lens**. ([arXiv][1])
* **FineWeb‑Edu dataset (HF dataset card).** Subsets, changelog (`v1.4.0` adds `CC‑MAIN‑2025‑26`), and loading examples. ([Hugging Face][2])
* **Logit Prisms (background & positioning).** Nguyen, “Logit Prisms: Decomposing Transformer Outputs …” (LessWrong post + explainer). ([LessWrong][3])

[1]: https://arxiv.org/abs/2303.08112?utm_source=chatgpt.com "Eliciting Latent Predictions from Transformers with the Tuned Lens"
[2]: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu "HuggingFaceFW/fineweb-edu · Datasets at Hugging Face"
[3]: https://www.lesswrong.com/posts/TKRp7inbiLRmzNMFB/logit-prisms-decomposing-transformer-outputs-for-mechanistic?utm_source=chatgpt.com "Logit Prisms: Decomposing Transformer Outputs for ..."

---

## 14) Findings From First Run + Path Forward

This section records the first end‑to‑end tuned‑lens run on Mistral‑7B (local training; 16M tokens; rank=128; identity‑plus‑low‑rank; no per‑layer temperature), and lays out concrete next steps to reach the acceptance gates reliably.

### 14.1 What worked

- Integration behaves as designed: tuned artifacts are auto‑loaded when present (`--tuned auto`), skipped otherwise; sidecar CSVs are emitted without changing baseline outputs; last‑layer identity constraint holds (KL≈0 vs model head).
- Diagnostics and provenance are complete (rank, depth, dataset hash, version, device/dtype). No TODO placeholders remain.

### 14.2 What didn’t move enough (on a single probe)

- On Mistral‑7B (Germany→Berlin probe), the tuned lens reduces KL by ~0.085–0.096 bits at the 25/50/75% depth slices and shows 0‑layer earliness shift for rank≤{1,5,10}. Gate‑A targets (≥{0.2, 0.3, 0.3} bits) are not met on this single prompt.
- Root causes (expected):
  - Per‑layer updates are sparse (one layer sampled per optimizer step) → ~30 update steps per layer across the entire 16M‑token budget.
  - No per‑layer temperature → tuned logits may remain miscalibrated even when the direction improves.
  - Single‑prompt snapshot is noisy; gates are intended for medians across a prompt suite.

### 14.3 Plan to increase signal per layer and improve calibration

1) Train multiple layers per step (high‑leverage)

- Change the fitter to update several randomly sampled post‑block layers per forward pass (retain identity at the last layer). Default: sample `L_sample = 8` layers per step without replacement (uniform over layers each step). For large GPUs, allowing `L_sample = all` is fine; keep the default at 8 for portability (MPS/CPU).
- Loss: sum (or mean) cross‑entropy to the final head over the sampled layers and positions; same regularizers as §3. This multiplies the update count per layer by ~8× without extra forwards.

2) Add per‑layer scalar temperature (optional but recommended)

- Introduce a learned positive scalar `τ_ℓ` per post‑block layer; implement as `log_tau_ℓ` with `τ_ℓ = softplus(log_tau_ℓ) + ε`, initialize near 1.0. Apply as `logits/τ_ℓ` after the translator and before softmax.
- Keep the last layer fixed: `τ_{last} = 1.0` (no effect on Gate‑C).
- Fair baseline calibration: optionally learn a single scalar per layer for the norm lens when reporting gates (configurable; default off initially). This isolates the effect of the translator vs pure temperature.

3) Capacity tweak (rank)

- Bump rank to 192 for models with `d_model ≥ 4096` (e.g., Mistral‑7B, Llama‑3‑8B). New policy:
  - `rank_k = min(256, max(base_rank, floor(d_model/32)))`, with `base_rank = 192` if `d_model ≥ 4096` else `base_rank = 128`.
- Rationale: rank=128 is often enough, but 192 provides headroom for mid‑stack layers without exploding parameters; 256 remains a hard cap for very wide models.

4) Evaluate on a prompt suite (not a single probe)

- Add a small, fixed suite (10–20 prompts) and compute Gate‑A medians and earliness shifts across prompts; record per‑model summaries in the run JSON.
- Keep the single‐probe CSVs for visualization and debugging; use the suite medians for gating decisions.

5) Reporting/diagnostics additions

- Record `L_sample` and whether per‑layer temperature was enabled in `provenance.json`.
- Add tuned‑vs‑norm ΔKL medians at {25,50,75}% and earliness shifts to the tuned lens block in the run JSON.
- Log step throughput and ETA every N steps (already added) for long runs.

### 14.4 Acceptance gates (unchanged), and how we’ll judge them

- Gate‑A: `median_delta_kl_bits` ≥ {0.2, 0.3, 0.3} bits at {25,50,75}% — computed over the prompt suite, using tuned vs norm with the chosen calibration setting.
- Gate‑B: earliness shifts for `first_rank_le_{10,5}` ≥ {+4,+2} layers on ≥60% of prompts.
- Gate‑C: `last_layer_kl_after_temp_bits` ≤ 0.05 (tuned last layer is identity; norm lens may use a scalar temperature for the fair comparison if enabled).
- Gate‑D: no depth percentile with negative ΔKL worse than −0.1 bits.

### 14.5 Implementation outline (follow‑up work)

- Fitter
  - Sample `L_sample=8` layers per step; compute CE across all sampled layers from the cached residuals; keep optimizer/regularizers unchanged.
  - Add per‑layer `log_tau_ℓ` parameters (freeze last layer); apply temperature at decode.
  - Rank policy update as above; record in `provenance.json` (`translator.rank`).
  - No new CLI flags for defaults; keep behavior deterministic with seed=316; continue to skip models when artifacts already exist.

- Runtime (run.py)
  - Load temperatures from the tuned artifacts and apply at decode.
  - Optional fair calibration for norm lens (single scalar per layer) can be toggled later; keep off by default to avoid changing baseline semantics.
  - Emit tuned suite metrics in the diagnostics when a suite is present; otherwise leave per‑probe summaries as is.

- Evaluation
  - Add a small prompt suite and a helper to compute ΔKL medians and earliness shifts from CSVs per model; store suite summaries in the run JSON.

### 14.6 Roll‑forward strategy

1) Implement multi‑layer updates + temperature + rank policy; retrain Mistral‑7B at 16M tokens.
2) Evaluate on the prompt suite; if Gate‑A is still borderline at 75% depth, consider `L_sample = all` for GPUs or increasing tokens to 24M for that model only.
3) If gates pass, proceed to train tuned lenses for the remaining mid‑sized models; defer giants until hardware or sharding support is available.

---

## 15) Second Iteration (H200 CUDA) — Results and Runtime Adjustments

This section documents the second end‑to‑end pass after implementing §14.3 and adding additional runtime optimizations for large vocabularies. The run used an H200‑class GPU with the updated fitter defaults and 32M tokens per model.

### 15.1 Changes implemented in this iteration

- Multi‑layer updates: sample `L_sample=24` layers/step (identity at last layer) and average losses over sampled layers and positions.
- Per‑layer temperature: learn τℓ; last layer fixed to τ=1.0. Runtime divides tuned logits by τℓ.
- Capacity: default rank schedule now yields `k=256` for `d_model ≥ 4096` (was 128; interim 192).
- Data budget: increased to 32M tokens (total_steps doubled to 1954 with effective batch 32×512).
- Throughput UX: human‑readable ETA; periodic tokens/sec logging.
- CUDA path: larger micro‑batch (32) and grad‑accum=1; pinned H2D and non‑blocking copies; keep tokenizer parallelism on by prefetching before tokenizer/model construction.
- Vocab‑aware auto‑scaling: scale per‑step work by `√(32k / d_vocab)` with floors (`layers ≥ 8`, `positions ≥ 12`); record actual schedule in provenance.
- Large‑vocab acceleration:
  - Build teacher logits only at sampled positions via final post‑block residual + correct final norm + tied unembedding (no full B×T×V logits tensor).
  - For `d_vocab ≥ 100k`, allow fp16/bf16 unembed (do not force fp32) to speed the large GEMMs; otherwise keep fp32 unembed for stability.

### 15.2 Mistral‑7B results (Germany→Berlin probe; tuned lens trained on CUDA)

- KL improvements (medians at depth percentiles; bits):
  - 25% (L8): ΔKL = +4.03 (norm 10.26 → tuned 6.22)
  - 50% (L16): ΔKL = +3.74 (norm 10.33 → tuned 6.58)
  - 75% (L24): ΔKL = +7.09 (norm 9.05 → tuned 1.97)
- Earliness (single probe diagnostics):
  - First KL ≤ 1.0 layer: norm L32 → tuned L26 (earlier by 6 layers).
  - First rank ≤ {10,5,1}: norm L{20,22,24} vs tuned L{22,24,25} (rank‑based earliness did not improve on this prompt).
- Sanity checks:
  - Last layer identity holds: `kl_after_temp_bits ≈ 3.9e‑06`; tuned last post‑block decoded ≈ model head.
  - No copy‑collapse flags at NEXT token across layers.

Interpretation: Gate‑A and Gate‑C are clearly satisfied on this probe; Gate‑B requires suite evaluation across prompts per §14.4.

### 15.3 Large‑vocab models (Llama‑3‑8B, Qwen‑3, Gemma‑2)

- Observation: with `d_vocab ≈ 128k–256k`, naive per‑step cost was dominated by O(V) unembedding, yielding ~2.1–2.4k tok/s on Llama‑3‑8B even after multi‑layer sampling.
- Remedies applied:
  - Auto‑scaled schedule (e.g., Llama‑3‑8B: 24/16 → 12/12).
  - Teacher logits computed only at sampled positions (remove B×T×V cost).
  - fp16/bf16 unembed for `d_vocab ≥ 100k`.
- Result: throughput rose into the ~3.5k tok/s range on H200 (exact figure depends on clocks). Further gains are available via top‑K CE distillation if needed (not implemented yet).

### 15.4 Runtime and cost expectations (H200)

- 32k vocab, 7–8B class (Mistral‑7B): ~0.66 h
- 128k vocab, 8B class (Llama‑3‑8B): ~2.5–3.5 h after optimizations.
- 152k vocab, 8/14B class (Qwen‑3‑8B/14B): ~3.5 h / ~6 h after optimizations.
- 256k vocab, 9–27B class (Gemma‑2‑9B/27B): ~6–7 h / ~18–20 h after optimizations.
- Yi‑34B (32–64k): ~3–6 h depending on tokenizer.
- 70B on single H200 likely does not fit; use B200 or sharded multi‑GPU.

These figures assume 32M tokens/model, prefetch enabled, and the auto‑scaled schedule. See §11 for general runtime/storage notes.

### 15.5 Risks, regressions, and mitigations

- Rank‑based earliness may not improve on every prompt even when KL decreases substantially; evaluate Gate‑B on a prompt suite as intended.
- fp16 unembed for very large `d_vocab` slightly changes calibration tails; per‑layer temperatures largely compensate. Validate Gate‑C and entropy drift on the suite.
- Auto‑scaled layers/positions reduce per‑step coverage for large vocabs; we kept TOTAL_TOKENS at 32M to preserve data volume. If a model is borderline on Gate‑A/B, prefer increasing tokens over increasing layers/positions (to avoid re‑introducing the O(V) bottleneck).

### 15.6 Next actions

1) Add prompt‑suite evaluator to compute Gate‑A/B medians and shifts; store suite summaries in the run JSON and TL provenance.
2) Retrain/evaluate additional models with the auto‑scaled schedule; prioritize 8–14B sizes where hardware fit and cost are favorable.
3) If needed for 128k–256k vocabs, implement top‑K teacher CE (union of per‑position teacher top‑K, K≈4–8k) to further cut unembed cost while preserving rank/top‑k signals; validate entropy impact.
4) Optional: add fair scalar temperature calibration for the norm‑lens when reporting (off by default) to isolate translator vs calibration effects in gates.

---

## 16) Brief Findings Snapshot (2025‑09‑27)

- Mistral‑7B (local first run vs CUDA retrain):
  - ΔKL medians at depth percentiles improved from ~0.09 bits (early draft) to multi‑bit gains after multi‑layer updates and temps (e.g., L8/L16/L24: +4.03/+3.74/+7.09 bits). First `KL≤1.0` moved earlier by ~6 layers; no copy‑collapse; last‑layer agreement near‑zero KL.
- Llama‑3‑8B (H200, 32M tokens, auto‑scaled 12 layers × 12 positions):
  - Large ΔKL gains across the stack (e.g., L8/L16/L24: +4.20/+4.31/+3.97 bits). Rank‑earliness did not improve on the single Berlin probe; last‑layer agreement near‑zero KL. Mid‑stack tuned entropy remains higher than teacher (as expected when the teacher is already low‑entropy on this prompt).
- Engineering improvements landed since the first draft:
  - Vocab‑aware auto‑scaling; teacher logits only at sampled positions; fp16/bf16 unembed for very large vocabs; UTC finish‑time logging; teacher_entropy_bits emitted in pure CSVs.
- Policy: gating (Gate‑A/B/C/D) remains evaluator‑level; the probe emits both tuned and baseline sidecars without making the selection.
