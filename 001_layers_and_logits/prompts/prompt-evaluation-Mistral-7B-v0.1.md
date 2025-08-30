ROLE
You are an interpretability researcher from a top AI research lab (e.g. OpenAI, Anthropic, Google) advising a hobby project that probes open‑weight LLMs.
You are reviewing results of a probe of one LLM model.

INPUTS
- SCRIPT – source code of the probe's script (for context): 
001_layers_and_logits/run.py

- JSON  – one json file with structured results of the probe of the model (first part of results):
001_layers_and_logits/run-latest/output-Mistral-7B-v0.1.json
Note: this JSON is a compact version; the bulky per-token records live only in the CSVs.

Use the `gold_answer` block for ID‑level alignment: `{ string, pieces, first_id, answer_ids, variant }`.
The `is_answer` flag and `p_answer`/`answer_rank` are computed using `first_id` (robust to leading‑space/multi‑piece tokenization).
Also read `diagnostics.last_layer_consistency` (last‑layer head calibration): `{ kl_to_final_bits, top1_agree, p_top1_lens, p_top1_model, p_answer_lens, answer_rank_lens, temp_est, kl_after_temp_bits, cfg_transform, kl_after_transform_bits, warn_high_last_layer_kl }`.


- CSV  - two csv files with detailed layer-level results of the probe of the model (second part of results):
001_layers_and_logits/run-latest/output-Mistral-7B-v0.1-records.csv
001_layers_and_logits/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv
Each CSV now includes a `rest_mass` column (probability not covered by the listed top-k tokens); the pure-next-token CSV also adds boolean flags `copy_collapse`, `entropy_collapse`, and `is_answer` produced by the script.
The pure-next-token CSV further includes per-layer probability and calibration fields: `p_top1`, `p_top5` (cumulative), `p_answer`, `answer_rank`, and `kl_to_final_bits` (bits), and `cos_to_final` (cosine similarity to the final logits direction; PROJECT_NOTES §1.5).

- Parameters (copy-collapse): copy_threshold = 0.95, copy_margin = 0.10

- Your own research knowledge.

- EVAL output file: 001_layers_and_logits/run-latest/evaluation-Mistral-7B-v0.1.md

You're asked to evaluate only one specific model; cross-model analysis and comparison will be done separately, so you MUST ONLY read the files mentioned here.

TASK
Write EVAL in GitHub‑flavoured Markdown with the sections below **in order**.  
If a claim cannot be grounded in a quoted CSV/JSON line, omit it.
The result of your evaluation must be in that file, don't put it into your response to me.

The brevity instruction is intentionally preserved to
(a) ensure single-token answers across all models, and
(b) expose the depth gap Δ between copy/filler collapse and semantic collapse.

1. Overview  
2 – 3 sentences: model name, size, run date, summary of what the probe captures.

2. Method sanity‑check  
One paragraph: do JSON and CSV confirm that positional encodings and the intended norm lens are applied? Quote ≤ 2 console lines with line numbers.
Verify context_prompt ends with “called simply” (no trailing space).
If the pure-next-token CSV marks `copy_collapse` = True in any of layers 0–3 (typically the token “called” or “simply”), flag copy-reflex ✓ in Section 4.
Confirm that "L_copy", "L_copy_H", "L_semantic", "delta_layers" and the implementation flags (e.g. "use_norm_lens", "unembed_dtype") are present in diagnostics. The copy rule is ID-level contiguous subsequence (k=1) with threshold τ=0.95 and margin δ=0.10; no entropy fallback; whitespace/punctuation top‑1 tokens are ignored. Cite `copy_thresh`, `copy_window_k`, `copy_match_level` from diagnostics. Gold‑token alignment: see `gold_answer` in JSON; alignment is ID‑based. Confirm `diagnostics.gold_alignment` is `ok`. If `unresolved`, note fallback to string matching and prefer rank‑based statements.
Report summary indices from diagnostics: `first_kl_below_0.5`, `first_kl_below_1.0`, `first_rank_le_1`, `first_rank_le_5`, `first_rank_le_10`. Confirm units for KL/entropy are bits. Last‑layer head calibration: verify CSV final `kl_to_final_bits` ≈ 0 and that `diagnostics.last_layer_consistency` exists. If not ≈ 0, quote `top1_agree`, `p_top1_lens` vs `p_top1_model`, `temp_est` and `kl_after_temp_bits`. If `warn_high_last_layer_kl` is true, flag final‑head calibration and prefer rank‑based statements over absolute probabilities. Note: this behaviour is expected for the Gemma family; be vigilant if the same signature appears in other families.
Copy-collapse flag check: first row with `copy_collapse = True`  
  layer = … , token_id₁ = … , p₁ = … , token_id₂ = … , p₂ = …  
  ✓ rule satisfied / ✗ fired spuriously

Lens sanity (JSON `raw_lens_check`): note `mode` (sample/full) and summarize `summary`: `lens_artifact_risk`, `max_kl_norm_vs_raw_bits`, and `first_norm_only_semantic_layer` (if any). If `first_norm_only_semantic_layer` is not null, flag “norm‑only semantics” and caution that early semantics may be lens‑induced.


3. Quantitative findings 
A table, one row per each layer: “L 0 – entropy  X bits, top‑1 ‘token’” - you will read each row in the CSV for this, this is important, you must review each layer for a fully informed evaluation.
Bold semantic layer (L_semantic) – first layer where `is_answer = True` (ID‑level gold token). For reference, `gold_answer.string` is “Berlin”.
Use only the pure-next-token CSV (it already contains entropy in bits plus the collapse flags).  The `rest_mass` column is provided for KL/entropy sanity-checks.

Add beneath the table:
ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = …
Confidence milestones (from pure CSV):  
p_top1 > 0.30 at layer …, p_top1 > 0.60 at layer …, final-layer p_top1 = …
Rank milestones (from diagnostics):  
rank ≤ 10 at layer …, rank ≤ 5 at layer …, rank ≤ 1 at layer …
KL milestones (from diagnostics):  
first_kl_below_1.0 at layer …, first_kl_below_0.5 at layer …; comment on whether KL decreases with depth and is ≈ 0 at final. If not ≈ 0 at final, annotate it, reference `diagnostics.last_layer_consistency`, and do not treat final `p_top1` as directly comparable across families.
Cosine milestones (from pure CSV):  
first `cos_to_final ≥ 0.2` at layer …, `≥ 0.4` at layer …, `≥ 0.6` at layer …; final `cos_to_final = …`.

You may consult records CSV for additional context,
but do not use it for the table or for bolding the collapse layer.


4. Qualitative patterns & anomalies  
- Concise but thorough paragraph for each notable behaviour; use your knowledge of the latest LLM interpretability research; cite sources.
- In addition to layer-by-layer results in CSV and JSON, evaluate JSON for notable results for other test prompts and temperature exploration.
- Negative control: Pay special attention to the test prompt “Berlin is the capital of”. Quote top-5 for test prompt “Berlin is the capital of”. If Berlin still appears, write: “semantic leakage: Berlin rank … (p = …)”.
- Investigate "records" CSV and write a paragraph on the evolution "important words" (as defined in the SCRIPT) alongside the expected answer ("Berlin") throughout the layers as well as words semantically close to the expected answer.
- Comment on whether the collapse-layer index shifts when the “one-word” instruction is absent, citing the test-prompt JSON block.
- Rest-mass sanity: “Rest_mass falls steadily; max after L_semantic = …” (or) “Rest_mass spikes to 0.37 at layer …, suggesting precision loss.”
- Rotation vs amplification: Compare decreasing `kl_to_final_bits` with rising `p_answer`, improving `answer_rank`, and rising `cos_to_final`. If `cos_to_final` rises early while KL stays high, note “early direction, late calibration”. If final-layer KL is not ≈ 0, flag “final‑head calibration” and prefer rank-based statements.
- Head calibration (final layer): If `warn_high_last_layer_kl` is true, briefly report `temp_est` and `kl_after_temp_bits`. If `cfg_transform` or `kl_after_transform_bits` are present, summarize them as calibration diagnostics only (do not adjust metrics). Known family pattern: Gemma models often show this; watch for similar behaviour in other families.
- Lens sanity: Quote the JSON `raw_lens_check.summary` and, if helpful, one sampled `raw_lens_check.samples` row. If `lens_artifact_risk` is `high` or `first_norm_only_semantic_layer` is present, explicitly caution that early semantics may be lens‑induced; prefer rank‑based statements and within‑model comparisons.
- Temperature robustness: “At T = 0.1, Berlin rank 1 (p = …); at T = 2.0, Berlin rank … (p = …). Entropy rises from … bits to … bits.”
- Important-word trajectory — “Berlin first enters any top-5 at layer …, stabilises by layer …. Germany remains in top-5 through layer …. capital drops out after layer ….”
- To support the claims, add a short inline quote + line number, e.g. > “… (‘Berlin’, 0.92)” [L541].
- Checklist (✓/✗/n.a.) at end of section:
    - RMS lens?  
    - LayerNorm bias removed?  
    - Entropy rise at unembed?
    - FP32 un-embed promoted? (see "use_fp32_unembed" in diagnostics)
    - Punctuation / markup anchoring?
    - Copy-reflex? ✓ if any of layers 0-3 in the pure-next-token CSV have copy_collapse = True (i.e. the model’s top-1 token is copied from the prompt with p > τ and margin > δ).
    - Grammatical filler anchoring? (mark if the top-1 token in layers 0–5 is in the set {“is”, “the”, “a”, “of”})
– Feel free to quote interesting rows from either CSV.
– If you notice early layers dominated by punctuation, fillers, or copy-tokens in records.csv, flag that under “Punctuation / filler anchoring”.

5. Limitations & data quirks  
Anything that reduces confidence; keep to facts.
Rest_mass > 0.3 after L_semantic indicates potential norm-lens mis-scale.
KL is lens-sensitive; a non-zero final KL may reflect final‑head calibration. Prefer rank milestones for cross-model claims; treat KL trends qualitatively. If `warn_high_last_layer_kl` is true, treat final‑row probability calibration as family-specific; rely on rank milestones and KL thresholds; use within‑model trends for conclusions.
Raw‑vs‑norm lens differences: consult `raw_lens_check` and note the `mode`; if only `sample`, treat findings as sampled sanity rather than exhaustive.

6. Model fingerprint (one sentence)  
Example: “Llama‑3‑8B: collapse at L 32; final entropy 1.8 bits; ‘Paris’ appears rank 2 mid‑stack.”

STYLE GUIDELINES
- Be conscise but thorough.
- Prefer paragraphs over lists.
- Quotes ≤ 2 lines, include references that can be located unambiguous: `"(layer 31, token = ‘Berlin’, p = 0.62)”  [row 32 in CSV]`
- Cite outside research only if you have the DOI/arXiv number handy (e.g. “Tuned‑Lens 2303.08112”); otherwise omit.  
- Ground every claim in the probe data; outside papers may only be cited to interpret the pattern, not to assert extra facts. 

At the end of the markdown file, add the following:

---
Produced by OpenAI GPT-5
