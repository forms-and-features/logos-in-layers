ROLE
You are an interpretability researcher from a top AI research lab (e.g. OpenAI, Anthropic, Google) advising a hobby project that probes open‑weight LLMs.
You are reviewing results of a probe of one LLM model.

INPUTS
- SCRIPT – source code of the probe's script (for context): 
001_layers_baseline/run.py

- JSON  – one json file with structured results of the probe of the model (first part of results):
001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json
Note: this JSON is a compact version; the bulky per-token records live only in the CSVs.
Also read `diagnostics.last_layer_consistency` (last‑layer head calibration): `{ kl_to_final_bits, top1_agree, p_top1_lens, p_top1_model, p_answer_lens, answer_rank_lens, temp_est, kl_after_temp_bits, cfg_transform, kl_after_transform_bits, warn_high_last_layer_kl }`.

If present, also read `diagnostics.prism_summary` (Prism sidecar calibration): `{ mode, artifact_path, present, compatible, k, layers, error }`.
If present, also read the `tuned_lens` block for provenance/diagnostics (path, summaries, provenance.translator/use_temperature/temperatures).

Also read:
* diagnostics.raw_lens_window: { radius, center_layers, layers_checked, norm_only_semantics_layers, max_kl_norm_vs_raw_bits_window, mode="window" }.
* summary.cos_milestones and summary.depth_fractions.
* summary.copy_thresholds: { tau_list, L_copy_strict, L_copy_strict_frac, norm_only_flags, stability } — threshold sweep robustness and cross‑validation.
* diagnostics.prism_summary.metrics and tuned_lens.summary.metrics.
* measurement_guidance: { prefer_ranks, suppress_abs_probs, reasons[], notes }.

Use the `gold_answer` block for ID‑level alignment: `{ string, pieces, first_id, answer_ids, variant }`.
The `is_answer` flag and `p_answer`/`answer_rank` are computed using `first_id` (robust to leading‑space/multi‑piece tokenization).

Also read if present (advisory blocks):

• provenance.env — { torch_version, cuda_version, cudnn, device, dtype_compute, deterministic_algorithms, cudnn_benchmark, seed, python, platform }.
• diagnostics.normalization_provenance — { arch, strategy, per_layer[].ln_source, eps_inside_sqrt, scale_gamma_used }.
• diagnostics.unembed_bias — { present, l2_norm, max_abs }. Treat all cosines as bias‑free; if bias present, note it as a calibration detail.
• diagnostics.numeric_health — { any_nan, any_inf, max_abs_logit_p99, min_prob_p01, layers_flagged } — flag anomalies if they overlap candidate collapse layers.
• diagnostics.copy_mask — { ignored_token_ids, ignored_token_str_sample, size } — note mask scope when interpreting copy detectors.

Ablation: read `ablation_summary` with `{ L_copy_orig, L_sem_orig, L_copy_nf, L_sem_nf, delta_L_copy, delta_L_sem }` to compare the original vs no‑filler (‘simply’ ablated) prompts.


- CSV  - two csv files with detailed layer-level results of the probe of the model (second part of results):
001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-records.csv
001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token.csv
Each CSV now includes leading `prompt_id` (`pos` for Germany→Berlin; `ctl` for France→Paris) and `prompt_variant` (`orig`/`no_filler`) columns, and a `rest_mass` column (probability not covered by the listed top-k tokens).
 The pure-next-token CSV adds boolean flags `copy_collapse`, strict-sweep flags `copy_strict@τ` where τ ∈ {0.70, 0.80, 0.90, 0.95}, and `copy_soft_k{1,2,3}@τ_soft`, plus `entropy_collapse`, and `is_answer`, together with per-layer fields:
 - Prob/calibration: `p_top1`, `p_top5` (cumulative), `p_answer`, `answer_rank`, `kl_to_final_bits` (bits)
 - Norm temperature (norm-only): `kl_to_final_bits_norm_temp` = KL(P(z/τ)||P_final)
 - Geometry: `cos_to_final` (§1.5), `cos_to_answer`, `cos_to_prompt_max`, `geom_crossover`
 - Surface mass: `echo_mass_prompt`, `answer_mass`, `answer_minus_echo_mass`, `mass_ratio_ans_over_prompt`
 - Coverage: `topk_prompt_mass@50`
 - Control: `control_margin = p(Paris) − p(Berlin)`
 Soft-copy defaults are τ_soft = 0.33 with windows k ∈ {1,2,3}; extra thresholds may appear when configured. Treat soft-copy primarily as an early-layer (L0–L3) sensitivity check; de-emphasize late soft hits near semantic collapse.

If present, include the Prism sidecar CSVs for calibration comparison:
001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-records-prism.csv
001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token-prism.csv

If present, also include the Tuned‑Lens sidecar CSVs (same schema) for comparison:
001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-records-tuned.csv
001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token-tuned.csv
The pure CSVs include `teacher_entropy_bits` (entropy of the teacher’s final distribution at NEXT). Use it to quantify entropy drift: `(entropy − teacher_entropy_bits)`.

If present, also include Windowed Raw‑vs‑Norm sidecar:
001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token-rawlens-window.csv
Columns: layer, lens ∈ {raw,norm}, p_top1, top1_token_id, top1_token_str, p_answer, answer_rank, kl_norm_vs_raw_bits, and (if present) strict-sweep flags `copy_strict@{0.70,0.80,0.90,0.95}`.

If present, also include Full Raw‑vs‑Norm sidecar:
001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501-pure-next-token-rawlens.csv
(Per-layer raw vs norm for all layers at NEXT; columns: layer, p_top1_raw, top1_token_id_raw, top1_token_str_raw, p_answer_raw, answer_rank_raw, p_top1_norm, top1_token_id_norm, top1_token_str_norm, p_answer_norm, answer_rank_norm, kl_norm_vs_raw_bits, norm_only_semantics.)

Also read diagnostics.raw_lens_full when present:
{ pct_layers_kl_ge_1.0, pct_layers_kl_ge_0.5, n_norm_only_semantics_layers, earliest_norm_only_semantic, max_kl_norm_vs_raw_bits, score: { lens_artifact_score ∈ [0,1], tier ∈ {low,medium,high} }, mode: "full" }.

Also read summary.L_semantic_confirmed (if present):
{ L_semantic_confirmed, L_semantic_norm, L_semantic_raw?, L_semantic_tuned?, Δ_window, confirmed_source ∈ {raw,tuned,both,none} }.

Also read tuned_lens.attribution (if present):
Attribution at depth percentiles {25,50,75} with:
- ΔKL_tuned = KL_norm − KL_tuned
- ΔKL_temp  = KL_norm − KL_norm_temp
- ΔKL_rot   = ΔKL_tuned − ΔKL_temp
and the advisory gate prefer_tuned (boolean).

measurement_guidance may include:
{ prefer_ranks, suppress_abs_probs, reasons[], preferred_lens_for_reporting?, use_confirmed_semantics? }.
Honor these in reporting.

- Parameters (copy-collapse): copy_threshold = 0.95, copy_margin = 0.10

- Your own research knowledge.

- EVAL output file: 001_layers_baseline/run-latest/evaluation-Mistral-Small-24B-Base-2501.md

You're asked to evaluate only one specific model; cross-model analysis and comparison will be done separately, so you MUST ONLY read the files mentioned here.

TASK
Write EVAL in GitHub‑flavoured Markdown with the sections below **in order**.  
If a claim cannot be grounded in a quoted CSV/JSON line, omit it.
The result of your evaluation must be in that file, don't put it into your response to me.

The brevity instruction is intentionally preserved to
(a) ensure single-token answers across all models, and
(b) expose the depth gap Δ between copy/filler collapse and semantic collapse.

Cautions
- Do not treat `rest_mass` as a lens-fidelity metric; it is top‑k coverage only. Diagnose fidelity/calibration via `diagnostics.last_layer_consistency` (final `kl_to_final_bits` ≈ 0 for well‑aligned heads) and `raw_lens_check.summary` (`lens_artifact_risk`, `max_kl_norm_vs_raw_bits`).
- Honor measurement_guidance:
  - If prefer_ranks = true or suppress_abs_probs = true, lead with rank thresholds and avoid absolute probability comparisons.
  - If preferred_lens_for_reporting is provided, cite depth milestones (e.g., first_rank≤{10,5,1}, L_surface_to_meaning) under that lens by default (still report baseline for context).
  - If use_confirmed_semantics = true and summary.L_semantic_confirmed is present, report L_semantic_confirmed (note confirmed_source) alongside L_semantic_norm.
- If `raw_lens_check.summary.lens_artifact_risk` is `high` or `first_norm_only_semantic_layer` is present, treat any pre‑final “early semantics” cautiously and prefer rank milestones (`first_rank_le_{10,5,1}`) over absolute probabilities; report the risk tier and `max_kl_norm_vs_raw_bits`.
- Cosine is a within‑model trajectory only; if citing thresholds (e.g., cos_to_final ≥ 0.2/0.4/0.6), include the layer indices from the pure CSV, and avoid cross‑family comparisons of absolute cosine values.
- When “top‑1” does not refer to the answer (pre‑semantic layers), label it as generic top‑1 (not `p_answer`). Use `p_answer`/`answer_rank` for semantic claims and always include the layer index when citing milestones (KL, cosine, rank, probabilities).
 - Interpret norm temperature fields (`tau_norm_per_layer`, `kl_to_final_bits_norm_temp`, `kl_to_final_bits_norm_temp@{25,50,75}%`) as calibration diagnostics vs the teacher; compare KL vs KL_temp to separate calibration from rotation.

Tuned‑Lens usage (if present)
- Present baseline (norm) and tuned side‑by‑side where relevant; do not suppress baseline.
- Attribute gains: at depth percentiles ≈{25,50,75}, report ΔKL_tuned = KL_norm − KL_tuned and ΔKL_temp = KL_norm − KL_norm_temp; define ΔKL_rot = ΔKL_tuned − ΔKL_temp and interpret positive ΔKL_rot as rotation (translator) gains beyond temperature.
- Rank earliness: report Δ in first_rank_le_{10,5,1} (tuned − norm). If unchanged, say so.
- Respect gate: if tuned_lens.attribution.prefer_tuned = true or measurement_guidance.preferred_lens_for_reporting = "tuned", cite tuned milestones in summaries by default (still include baseline values for context).
- Last‑layer agreement: verify diagnostics.last_layer_consistency.kl_after_temp_bits ≈ 0; quote the line.
- Entropy drift: compare entropy vs teacher_entropy_bits at mid‑depths; note sign/magnitude.

1. Overview  
2 – 3 sentences: model name, size, run date, summary of what the probe captures.

2. Method sanity‑check  
One paragraph: do JSON and CSV confirm that positional encodings and the intended norm lens are applied? Quote ≤ 2 console lines with line numbers.
Verify context_prompt ends with “called simply” (no trailing space).

If the pure-next-token CSV marks `copy_collapse` = True in any of layers 0–3 (typically the token “called” or “simply”), or `copy_soft_k1@τ_soft` = True in layers 0–3 (τ_soft=0.33), flag copy-reflex ✓ in Section 4. Do not count soft-copy hits that first appear near L_semantic as copy-reflex evidence.

Confirm that "L_copy", "L_copy_soft[k], "L_semantic", "delta_layers", "L_copy_soft" (per k), and "delta_layers_soft" are present in diagnostics alongside the implementation flags (e.g. "use_norm_lens", "unembed_dtype").

Additionally verify (quote one line each if present):
* Normalizer provenance: diagnostics.normalization_provenance.strategy and per_layer[0].ln_source / per_layer[last].ln_source.
* Per-layer normalizer effect: resid_norm_ratio and delta_resid_cos are well-behaved (no extreme spikes before L_semantic).
* Unembedding bias: diagnostics.unembed_bias.present and l2_norm — note that all geometric metrics are bias‑free.
* Environment & determinism: provenance.env (torch_version, device, deterministic_algorithms, seed). If deterministic_algorithms=false, add a reproducibility caution.
* Numeric health: diagnostics.numeric_health.any_nan/any_inf and layers_flagged (treat overlaps with early copy/semantic layers as suspect).
* Copy mask: diagnostics.copy_mask.size and sample — ensure the ignore set is plausible for the tokenizer.

The strict copy rule remains ID‑level contiguous subsequence (k=1) with threshold τ=0.95 and margin δ=0.10. Soft detectors use τ_soft = 0.33 with window_ks from `copy_soft_config.window_ks`. When `L_copy_strict` is null, compute Δ using the earliest `L_copy_soft[k]` (report k).

Cite `copy_thresh`, `copy_window_k`, `copy_match_level`, and `copy_soft_config` (threshold, window_ks, extra_thresholds); confirm `copy_flag_columns` mirrors these labels in the JSON/CSV. Gold‑token alignment: see `gold_answer`; alignment is ID-based. Confirm `diagnostics.gold_alignment` is `ok`. If `unresolved`, note fallback to string matching and prefer rank-based statements. Negative control: confirm `control_prompt` and `control_summary`. Ablation: confirm `ablation_summary` exists and that positive rows appear under both `prompt_variant = orig` and `no_filler`. For the main table, filter to `prompt_id = pos`, `prompt_variant = orig`.

Report summary indices from diagnostics: `first_kl_below_0.5`, `first_kl_below_1.0`, `first_rank_le_1`, `first_rank_le_5`, `first_rank_le_10`. Confirm units for KL/entropy are bits. Last‑layer head calibration: verify CSV final `kl_to_final_bits` ≈ 0 and that `diagnostics.last_layer_consistency` exists. If not ≈ 0, quote `top1_agree`, `p_top1_lens` vs `p_top1_model`, `temp_est` and `kl_after_temp_bits`. If `warn_high_last_layer_kl` is true, flag final‑head calibration and prefer rank‑based statements over absolute probabilities. Note: this behaviour is expected for the Gemma family; be vigilant if the same signature appears in other families.

Measurement guidance: quote `measurement_guidance` (prefer_ranks/suppress_abs_probs and reasons).

Raw‑vs‑Norm window (if present): list `center_layers`, `radius`, and any `norm_only_semantics_layers`; cite the largest `kl_norm_vs_raw_bits_window`.
Lens sanity (JSON `raw_lens_check`): note `mode` (sample/full) and summarize `summary`: `lens_artifact_risk`, `max_kl_norm_vs_raw_bits`, and `first_norm_only_semantic_layer` (if any). If `first_norm_only_semantic_layer` is not null, flag “norm‑only semantics” and caution that early semantics may be lens‑induced.

If the Full Raw‑vs‑Norm sidecar and diagnostics.raw_lens_full are present, also report:
pct_layers_kl_ge_1.0, pct_layers_kl_ge_0.5, n_norm_only_semantics_layers, earliest_norm_only_semantic, max_kl_norm_vs_raw_bits, and lens_artifact_score (tier).
Treat larger values/tier as higher artefact risk; prefer rank milestones and L_semantic_confirmed accordingly.

Threshold sweep sanity: confirm `summary.copy_thresholds` is present; quote the `stability` tag and earliest `L_copy_strict` at τ=0.70 and τ=0.95. If `norm_only_flags[τ]` is true for the earliest layer at τ, caution that strict‑copy appears only under the norm lens within the window; use the raw‑vs‑norm sidecar rows to substantiate.

If any early rank‑1 under the norm lens is not corroborated by raw or tuned within ±2 layers, label it as “norm‑only semantics” and do not treat it as semantic onset unless L_semantic_confirmed matches it.

Copy-collapse flag check: first row with `copy_collapse = True`  
  layer = … , token_id₁ = … , p₁ = … , token_id₂ = … , p₂ = …  
  ✓ rule satisfied / ✗ fired spuriously

Soft copy flags: record earliest layer where `copy_soft_k1@τ_soft = True` (and optionally k2/k3); note if soft fires while strict stays null.

Lens selection:
- Surface→meaning: Report L_surface_to_meaning for both lenses when available. If L_semantic_confirmed exists, include it and the (answer_mass_at_L, echo_mass_at_L) at that confirmed layer. Use preferred_lens_for_reporting to decide which value to foreground in prose.
- Geometry: Report L_geom for both lenses when available; treat as within‑model only.
- Coverage: Report L_topk_decay (K=50, τ=0.33) for both lenses when available.
- Norm temp: Include tau_norm_per_layer presence and kl_to_final_bits_norm_temp@{25,50,75} snapshots; optionally note per‑layer rows illustrating KL vs KL_temp.

3. Quantitative findings 
A table, one row per each layer: “L 0 – entropy  X bits, top‑1 ‘token’” — build the table from positive rows only: `prompt_id = pos`, `prompt_variant = orig`.
Bold semantic layer (default = L_semantic_norm). If summary.L_semantic_confirmed exists and/or measurement_guidance.use_confirmed_semantics = true, bold L_semantic_confirmed and note the corroborating source (raw/tuned). When preferred_lens_for_reporting="tuned", also report the tuned collapse layer in-line.

Control margin (from JSON `control_summary`):
- `first_control_margin_pos` = first layer with control_margin > 0;
- `max_control_margin` = maximum margin across layers.
If gold alignment for either gold token is unresolved, note the limitation and avoid absolute probability claims; prefer rank thresholds.
Use the pure-next-token CSV for ranks, KL, cosine, masses, and coverage. If an entropy column (e.g., entropy_bits) is present, use it; otherwise either (a) omit entropy deltas, or (b) consult the records CSV specifically for entropy values (do not recompute from partial outputs). If present, also cite:
* resid_norm_ratio and delta_resid_cos (normalizer effect);
* answer_logit_gap and answer_vs_top1_gap (confidence margins).

Ablation (no‑filler). From JSON `ablation_summary`, report:
- `L_copy_orig = …`, `L_sem_orig = …`
- `L_copy_nf = …`, `L_sem_nf = …`
- `ΔL_copy = L_copy_nf − L_copy_orig`, `ΔL_sem = L_sem_nf − L_sem_orig`
Interpretation: a large positive `ΔL_sem` (e.g., ≥ ~10% of `n_layers`) suggests stylistic‑cue sensitivity. If any value is null, note the limitation and rely on rank milestones.

Add beneath the table:
- ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = …
- Soft ΔHₖ (bits) = entropy(L_copy_soft[k]) − entropy(L_semantic) for k ∈ window_ks.
- If any `L_copy_soft[k]` differs materially from `L_copy`, highlight it (e.g., "k=2 soft copy at L … while strict null").
- Confidence milestones (from pure CSV):
  p_top1 > 0.30 at layer …, p_top1 > 0.60 at layer …, final-layer p_top1 = …
- Rank milestones (from diagnostics):
  rank ≤ 10 at layer …, rank ≤ 5 at layer …, rank ≤ 1 at layer …
  When preferred_lens_for_reporting is set, cite the milestones under that lens first, then provide the baseline as parenthetical.
- KL milestones (from diagnostics):
  first_kl_below_1.0 at layer …, first_kl_below_0.5 at layer …; comment on whether KL decreases with depth and is ≈ 0 at final. If not ≈ 0 at final, annotate it, reference `diagnostics.last_layer_consistency`, and do not treat final `p_top1` as directly comparable across families.
- Cosine milestones (from pure CSV):
  first `cos_to_final ≥ 0.2` at layer …, `≥ 0.4` at layer …, `≥ 0.6` at layer …; final `cos_to_final = …`.
  If `cos_milestones` is present in JSON, prefer those layer indices for thresholds ge_{0.2,0.4,0.6}; otherwise compute from the CSV.

Use `summary.depth_fractions` (if present) to report normalized depths (e.g., `L_semantic_frac`), and `summary.cos_milestones` to reference cosine thresholds without scanning the CSV; still include the corresponding layer indices.

Copy robustness (threshold sweep)
Report the `summary.copy_thresholds.stability` tag and earliest layers for strict copy at τ ∈ {0.70, 0.95}. If `norm_only_flags[τ]` is true, flag potential lens‑induced copy (norm‑only) and reference the raw‑vs‑norm sidecar (if available). Treat this as robustness commentary, not as a redefinition of Δ (which stays tied to strict@0.95 when present).

Prism Sidecar Analysis (if present)
Prism is a shared‑decoder diagnostic for robustness/comparability, not the model’s head. Judge Helpful/Neutral/Regressive strictly relative to the norm lens baseline, not vs the final head.
- Presence: if `diagnostics.prism_summary.compatible != true`, skip Prism analysis and note unavailability.
- Early-depth stability: at L≈0, ⌊n/4⌋, ⌊n/2⌋, ⌊3n/4⌋, compare KL(P_layer||P_final) from baseline vs `*-pure-next-token-prism.csv`.
- Rank milestones: compute `first_rank_le_{10,5,1}` from Prism pure CSV and report deltas vs baseline.
- Top‑1 agreement: at sampled depths, note any Prism→final top‑1 agreements where baseline disagreed (and vice versa).
- Cosine drift: compare `cos_to_final` at early/mid layers (Prism vs baseline); call out earlier stabilization if observed.
- Copy flags: verify `copy_collapse` calls don’t spuriously flip under Prism; explain any flips plausibly.
- Verdict: Helpful (clear KL drop ≥0.5 bits at early layer and same/earlier `first_rank_le_1`), Neutral (±0.2 bits, no qualitative shifts), or Regressive (KL increases or later rank milestones).

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
  Additionally, consult `summary.copy_thresholds.norm_only_flags` for norm‑only strict‑copy at τ ∈ {0.70, 0.80, 0.90, 0.95}; if any true, note that copy signals may be lens‑induced.
- Temperature robustness: “At T = 0.1, Berlin rank 1 (p = …); at T = 2.0, Berlin rank … (p = …). Entropy rises from … bits to … bits.”
- Important-word trajectory — “Berlin first enters any top-5 at layer …, stabilises by layer …. Germany remains in top-5 through layer …. capital drops out after layer ….”
- Stylistic ablation: summarize whether removing “simply” delays or advances semantics (`ΔL_sem`) or copy (`ΔL_copy`); if large, attribute likely guidance‑style anchoring rather than semantics.
- To support the claims, add a short inline quote + line number, e.g. > “… (‘Berlin’, 0.92)” [L541].
- Checklist (✓/✗/n.a.) at end of section:
    - RMS lens?  
    - LayerNorm bias removed?  
    - Entropy rise at unembed?
    - FP32 un-embed promoted? (see "use_fp32_unembed" in diagnostics)
    - Punctuation / markup anchoring?
    - Copy-reflex? ✓ if any of layers 0-3 in the pure-next-token CSV have `copy_collapse = True` (strict τ, δ) or `copy_soft_k1@τ_soft = True`.
    - Grammatical filler anchoring? (mark if the top-1 token in layers 0–5 is in the set {“is”, “the”, “a”, “of”})
    - Preferred lens honored in milestones (from measurement_guidance.preferred_lens_for_reporting or tuned_lens.attribution.prefer_tuned)
    - Confirmed semantics reported when available (summary.L_semantic_confirmed + source)
    - Full dual‑lens metrics cited (pct_layers_kl_ge_1.0, n_norm_only_semantics_layers, earliest_norm_only_semantic, lens_artifact_score tier)
    - Tuned‑lens attribution done (ΔKL_tuned, ΔKL_temp, ΔKL_rot at ~25/50/75%)
    - normalization_provenance present (ln_source verified at layer 0 and final)
    - per-layer normalizer effect metrics present (resid_norm_ratio, delta_resid_cos)
   - unembed bias audited (bias-free cosine guaranteed)
   - deterministic_algorithms = true (or caution noted)
   - numeric_health clean (no NaN/Inf; no flagged early layers)
   - copy_mask present and plausible for tokenizer
   - layer_map present (if available) for indexing audit

– Feel free to quote interesting rows from either CSV.
– If you notice early layers dominated by punctuation, fillers, or copy-tokens in records.csv, flag that under “Punctuation / filler anchoring”.

5. Limitations & data quirks  
Anything that reduces confidence; keep to facts.
Rest_mass > 0.3 after L_semantic indicates potential norm-lens mis-scale.
KL is lens-sensitive; a non-zero final KL can reflect final‑head calibration. Prefer rank milestones for cross-model claims; treat KL trends qualitatively. If `warn_high_last_layer_kl` is true, treat final‑row probability calibration as family-specific; rely on rank milestones and KL thresholds; use within‑model trends for conclusions.
Raw‑vs‑norm lens differences: consult raw_lens_check.mode and diagnostics.raw_lens_full (if present). If only sample, treat findings as sampled sanity rather than exhaustive; if full is present and lens_artifact_score is medium/high or norm_only_semantics_layers>0, caution that early semantics may be lens‑induced and prefer rank milestones and confirmed semantics.
Surface‑mass relies on tokenizer‑level prompt vocab; cross‑model comparisons of absolute masses can be confounded by tokenization differences—prefer within‑model trends and rank milestones.

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
