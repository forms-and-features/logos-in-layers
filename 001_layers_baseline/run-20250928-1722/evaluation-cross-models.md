**Cross‑Model Evaluation**

This synthesis reads script logic (001_layers_baseline/run.py) and the per‑model artifacts under `001_layers_baseline/run-latest/` (JSON metadata, pure‑next‑token and records CSVs, plus Prism/Tuned sidecars). All quantitative claims are grounded in these files; ranks/KL are within‑model unless stated. When final‑row lens vs head calibration shows non‑zero last‑layer KL, rank thresholds are preferred over probabilities.

**Method Notes**
- Norm lens is architecture‑aware and uses fp32 un‑embed by default (run.py). Last‑layer agreement is checked (`diagnostics.last_layer_consistency`). Tuned‑Lens and Prism sidecars are present for several models; when available, they are summarized within model. For broader calibration context, see Tuned‑Lens (arXiv:2303.08112).

### 1) Result synthesis

Copy‑reflex (L0–L3). Only the Gemma family shows an early copy reflex. In gemma‑2‑9B the first four layers set `copy_collapse=True` and `copy_soft_k1@0.5=True` at NEXT, e.g. L0: `copy_collapse=True, copy_soft_k1@0.5=True` (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2). Gemma‑2‑27B also fires at L0 and retains soft‑copy at L3 (001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2,4). All other models have no early copy flags (checked in the pure CSVs). This isolates Gemma as the outlier in early copy behavior.

Rank milestones (within model). Using JSON diagnostics:
- gemma‑2‑9B: first rank≤{10,5,1} at L=42 (001_layers_baseline/run-latest/output-gemma-2-9b.json:902–904). gemma‑2‑27B: all at L=46 (001_layers_baseline/run-latest/output-gemma-2-27b.json:906–908). Both collapse only at the final head and have non‑zero last‑layer KL (see below).
- Meta‑Llama‑3‑8B: rank≤{10,5,1} at {24,25,25} (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:892–894).
- Meta‑Llama‑3‑70B: rank≤{10,5,1} at {38,38,40} (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:940–942).
- Mistral‑7B‑v0.1: rank≤{10,5,1} at {23,25,25} (001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:892–894).
- Mistral‑Small‑24B: rank≤{10,5,1} at {30,30,33} (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:900–902).
- Qwen3‑8B: rank≤{10,5,1} at {29,29,31} (001_layers_baseline/run-latest/output-Qwen3-8B.json:896–898).
- Qwen3‑14B: rank≤{10,5,1} at {32,33,36} (001_layers_baseline/run-latest/output-Qwen3-14B.json:900–902).
- Qwen2.5‑72B: rank≤{10,5,1} at {74,78,80} (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:940–942).
- Yi‑34B: rank≤{10,5,1} at {43,44,44} (001_layers_baseline/run-latest/output-Yi-34B.json:920–922).

KL milestones and head calibration. Treat KL milestones qualitatively and within‑model; exclude families with non‑zero last‑layer KL from KL‑based cross‑model conclusions. Gemma shows calibrated‑head mismatch: `kl_to_final_bits≈1.01–1.14` and `warn_high_last_layer_kl=true` (001_layers_baseline/run-latest/output-gemma-2-9b.json:962; 001_layers_baseline/run-latest/output-gemma-2-27b.json:966). Others align at final: e.g., Meta‑Llama‑3‑8B `kl_to_final_bits=0.0` (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:952), Meta‑Llama‑3‑70B `0.000729…` (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1000), Qwen3‑14B `0.0` (001_layers_baseline/run-latest/output-Qwen3-14B.json:960), Yi‑34B `0.000278…` (001_layers_baseline/run-latest/output-Yi-34B.json:980).

Cosine alignment (within model, qualitative). cos_to_final rises with depth and peaks at the final head; absolute levels are not comparable across families. For Qwen3‑14B, cos_to_final ≈0.24 by L5 (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:7), ≈0.45 by L29 (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:31), and ≈0.61 at L36 where rank flips to 1 (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:38). Similar within‑model rises appear broadly across families.

Lens sanity (raw vs norm). From `raw_lens_check.summary`:
- Norm‑only semantics flagged: Meta‑Llama‑3‑8B `first_norm_only_semantic_layer=25` (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1368); Yi‑34B `46` (001_layers_baseline/run-latest/output-Yi-34B.json:1396). Treat pre‑final “early semantics” cautiously and prefer rank milestones.
- Risk tiers and maxima: Meta‑Llama‑3‑70B `lens_artifact_risk=low; max_kl_norm_vs_raw_bits=0.0429` (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1191–1192); Mistral‑Small‑24B `low; 0.1793` (001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1376–1377). Others are `high`, e.g., Qwen3‑14B `high; 17.6735` (001_layers_baseline/run-latest/output-Qwen3-14B.json:1376–1377), Qwen3‑8B `high; 13.6049` (001_layers_baseline/run-latest/output-Qwen3-8B.json:1373–1374), gemma‑2‑27B `high; 80.1001` (001_layers_baseline/run-latest/output-gemma-2-27b.json:1383–1384).

Family similarities/differences.
- Qwen (8B vs 14B vs 72B). The 8B and 14B show late collapse (rank‑1 at L=31/36) and clean final calibration (001_layers_baseline/run-latest/output-Qwen3-8B.json:896; 001_layers_baseline/run-latest/output-Qwen3-14B.json:900,960). The 14B exhibits a genuine separation between `L_copy_H` and `L_semantic` (Δ=4 layers in JSON “tuned” summaries; baseline diagnostics place `L_copy_H≈32`, `L_semantic=36”), yielding a non‑trivial entropy drop (see below). The 72B collapses only at the very end (rank‑1 at L=80) with low final KL (001_layers_baseline/run-latest/output-Qwen2.5-72B.json:940,1000).
- Gemma (9B vs 27B). Both display a strong early copy‑reflex and final‑only semantic collapse with non‑zero last‑layer KL (001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2–4; 001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2; and JSON lines above). The pattern suggests shared family calibration choices at the head, not earlier rank differences.

Collapse earliness and Δ layers. Grouped by relative depth (L_sem/n_layers):
- Early (<70%): only Meta‑Llama‑3‑70B (40/80 = 50%; 001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:933).
- Late (≥70%): gemma‑2‑9B (42/42 = 100%; 001_layers_baseline/run-latest/output-gemma-2-9b.json:895), gemma‑2‑27B (46/46 = 100%; 001_layers_baseline/run-latest/output-gemma-2-27b.json:899), Meta‑Llama‑3‑8B (25/32 ≈ 78%; 001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:885), Mistral‑7B (25/32 ≈ 78%; 001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:885), Mistral‑Small‑24B (33/40 = 82.5%; 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:893), Qwen3‑8B (31/36 ≈ 86%; 001_layers_baseline/run-latest/output-Qwen3-8B.json:889), Qwen3‑14B (36/40 = 90%; 001_layers_baseline/run-latest/output-Qwen3-14B.json:893), Qwen2.5‑72B (80/80=100%; 001_layers_baseline/run-latest/output-Qwen2.5-72B.json:933), Yi‑34B (44/60 ≈ 73%; 001_layers_baseline/run-latest/output-Yi-34B.json:913).
Within‑family Δ trends: in Qwen, Δ = L_sem − L_copy is 0 for 8B (copy and semantic hit coincide at L=31) and 4 for 14B (L_copy_H≈32 → L_sem=36), yielding relative Δ̂≈0.10 for 14B. This suggests a clearer separation of surface→meaning at larger size within Qwen.

Entropy drop shape (ΔH). Compute ΔH = entropy(L_copy) − entropy(L_semantic) from pure CSVs (pos/orig):
- Qwen3‑14B: ΔH ≈ 0.50 bits (entropy@L_copy≈0.815 vs entropy@L_sem≈0.312) — a modest but real plunge (001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:34,38). Qwen3‑8B: ΔH ≈ 0.0 (copy and semantic coincide at L=31; same row). Gemma shows negative ΔH due to final‑head mismatch inflating final‑row entropy under the lens (e.g., gemma‑2‑9B ΔH≈−0.37; see the calibration warning above). Treat Gemma’s ΔH as calibration‑confounded and rely on ranks.

Width/heads vs collapse sharpness. A consistent cross‑family link between d_model/n_heads and sharper ΔH is not evident here. Within Qwen, the 14B (d_model=5120, n_heads=40; 001_layers_baseline/run-latest/output-Qwen3-14B.json:1311–1314) shows a cleaner separation than 8B (d_model=4096, n_heads=32; 001_layers_baseline/run-latest/output-Qwen3-8B.json:1307–1313). Across families, absolute ΔH and cosine levels are lens‑ and tokenizer‑sensitive; avoid over‑interpreting.

Surface/Geometry/Coverage snapshots (norm; within‑model). Using diagnostics:
- Qwen3‑14B: L_surface_to_meaning_norm=36 with answer_mass≈0.9530 vs echo_mass≈4.39e‑06 (001_layers_baseline/run-latest/output-Qwen3-14B.json:947–953). L_geom_norm=35 with cos_to_answer≈0.081 and cos_to_prompt_max≈0.059 (001_layers_baseline/run-latest/output-Qwen3-14B.json:950–952). L_topk_decay_norm=0; early prompt coverage near‑zero (001_layers_baseline/run-latest/output-Qwen3-14B.json:953–955).
- Qwen3‑8B: L_surface_to_meaning_norm=31 with answer_mass≈0.936 vs echo_mass≈0.011 (001_layers_baseline/run-latest/output-Qwen3-8B.json:947–953). L_geom_norm=34; L_topk_decay_norm=0.
- Meta‑Llama‑3‑8B: L_surface_to_meaning_norm=32 with answer_mass≈0.520 vs echo_mass≈0.024 (001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:947–953). L_geom_norm=26.
- Yi‑34B: L_surface_to_meaning_norm=51 with answer_mass≈0.0597 vs echo_mass≈0.0063 (001_layers_baseline/run-latest/output-Yi-34B.json:967–973). L_geom_norm=46.
These are within‑model indicators of when surface echo gives way to meaning and when geometric alignment crosses its chosen criterion; they should not be compared across families.

Entropy drift at mid‑depth. `(entropy − teacher_entropy_bits)` at ≈50% depth shows sizeable positive drift across most families (lens distribution more entropic than the final head): Meta‑Llama‑3‑8B ≈ +13.89 bits, Meta‑Llama‑3‑70B ≈ +14.34 bits, Qwen3‑14B ≈ +13.35 bits, Qwen3‑8B ≈ +13.79 bits, Mistral‑7B ≈ +10.99 bits, Mistral‑Small‑24B ≈ +13.56 bits; Yi‑34B ≈ +12.61 bits. Gemma is inconsistent due to calibration mismatch (e.g., gemma‑2‑9B ≈ −1.07 bits; gemma‑2‑27B ≈ +4.12 bits). Values are from the pure CSV at L≈n/2 (file lines omitted for brevity; see mid‑depth rows for each model).

Relation to public scores (MMLU/ARC). Within this single‑prompt probe, early semantic collapse does not straightforwardly predict MMLU/ARC. The only “early” model is Llama‑3‑70B (50% depth; strong public scores). High‑scoring Yi‑34B collapses late (≈73%). Within Qwen, the 14B (higher MMLU/ARC) shows a slightly clearer separation (Δ̂≈0.10) than 8B (Δ̂=0), but both collapse late. With the present single‑prompt, treat these as qualitative within‑family tendencies rather than across‑family predictors.

Prism summary across models. All models report `prism_summary.compatible=true`. However, Prism rarely improves rank milestones and often increases KL at early/mid depths. For Qwen3‑14B, Prism never reaches rank≤10 (baseline: rank≤10 at L32; 001_layers_baseline/run-latest/output-Qwen3-14B.json:900–902), and KL deltas at ≈{25,50,75}% are small positive (worse): ΔKL≈{+0.26,+0.25,+0.71} bits (prism − baseline at L={10,20,30}). Similar non‑gains occur for Meta‑Llama‑3‑8B (ΔKL≈{+5.37,+8.29,+9.78}), Mistral‑7B (≈{+12.81,+17.54,+17.50}), Qwen3‑8B (≈{+0.36,+0.59,+7.03}), and Meta‑Llama‑3‑70B (≈{+0.89,+1.00,+1.16}). Some models show modest early KL reductions: gemma‑2‑27B (ΔKL≈{−22.57,−23.73,−23.08}) and Yi‑34B (≈{−0.94,−1.36,+1.01}); Qwen2.5‑72B improves at earlier depths (≈{−3.16,−2.83,+0.54}). Overall, Prism gains do not track `lens_artifact_risk` monotonically; several high‑risk models see regressions. Claims remain within‑model; absolute levels are not compared across families.

Tuned‑Lens (multi‑model). Tuned reduces KL substantially at early/mid depths across most models but often delays rank milestones slightly. Examples (ΔKL = tuned − baseline at ≈{25,50,75}%): Qwen3‑14B ≈{−4.68,−4.49,−3.90}; Qwen3‑8B ≈{−4.14,−4.00,−1.40}; Mistral‑7B ≈{−4.03,−3.75,−7.08}; Mistral‑Small‑24B ≈{−4.19,−4.59,−5.35}; Meta‑Llama‑3‑8B ≈{−4.35,−4.31,−3.95}; Yi‑34B ≈{−5.88,−6.10,−8.39}. In all these, last‑layer agreement remains good in the baseline lens (`kl_after_temp_bits≈0`; e.g., Qwen3‑14B: 0.0 at final; 001_layers_baseline/run-latest/output-Qwen3-14B.json:967). Rank earliness shifts are small and sometimes later (e.g., Qwen3‑14B baseline r1=36 vs tuned r1=39; see tuned sidecar vs baseline). Treat tuned improvements as within‑model calibration/rotation gains rather than cross‑family earliness.

Skip‑layers sanity (advisory). Where present, `diagnostics.skip_layers_sanity` varies widely (e.g., Qwen3‑14B m=2 ≈ −0.56; Meta‑Llama‑3‑8B m=2 ≈ 715.81; gemma‑2‑27B very large), underscoring that translator/rotation composition and layer decimation can interact non‑trivially. This is advisory only.

—

### 2) Misinterpretations in existing EVALS

- Meta‑Llama‑3‑70B: “Rest‑mass falls late; max after L_semantic = 0.9999 … consistent with narrow top‑k coverage in mid‑stack rather than fidelity loss.” While the caution about coverage is helpful, the single sentence could be read as over‑confident on rest_mass’s diagnostic value. Prefer pairing with explicit top‑k coverage (`topk_prompt_mass@50`) and last‑layer calibration quotes to avoid implying any fidelity test (001_layers_baseline/run-latest/evaluation-Meta-Llama-3-70B.md:137).
- Yi‑34B: “Confidence milestones (pure CSV, pos/orig): p_top1 > 0.30 at layer 60; … final‑layer p_top1 = 0.5555.” This is accurate within model, but given `raw_lens_check.summary.lens_artifact_risk=high` and `first_norm_only_semantic_layer=46`, consider foregrounding ranks over probabilities in the summary paragraph (001_layers_baseline/run-latest/evaluation-Yi-34B.md:123–130; 001_layers_baseline/run-latest/output-Yi-34B.json:1396–1398).
- Several EVALs mention large mid‑depth KL drops or spikes qualitatively without always quoting the corresponding `raw_lens_check.mode` and `max_kl_norm_vs_raw_bits`. Adding those quotes would strengthen the caveat that KL is lens‑sensitive (e.g., 001_layers_baseline/run-latest/evaluation-Qwen3-8B.md:16 vs 001_layers_baseline/run-latest/output-Qwen3-8B.json:1316–1374).

—

### 3) Limitations

- RMS‑lens can distort absolute probabilities; comparisons should stay within‑model, not across differing normalisation schemes.
- Single‑prompt probing may over‑fit tokenizer quirks; copy‑collapse depth can change if wording or punctuation shifts.
- Attention patterns and MLP activations are not inspected—only residual projections—so entropy bumps caused by internal gating may be mis‑attributed.
- Un‑embed weights may be promoted to FP32 ("use_fp32_unembed": true) in some models, slightly shrinking entropy gaps; keep comparisons qualitative.
- Final‑lens vs final‑head mismatches can keep last‑layer KL > 0 for some families/precisions; prefer rank thresholds for cross‑model conclusions and treat KL trends qualitatively within model.
- Layer counts differ (8B ≈ 32 layers, 34B ≈ 48+); compare relative depths, not absolute indices.
- Current results are correlation‑only; causal evidence (patching) awaits a later run.

—
Produced by OpenAI GPT-5

