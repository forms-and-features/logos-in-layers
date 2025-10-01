**Cross‑Model Evaluation**

**Result Synthesis**
- Copy‑reflex (layers 0–3): Only the Gemma family shows an early copy reflex under the strict rule (τ=0.95) and soft k1@0.5. In gemma‑2‑9B the first strict and soft hits are at L0 (pure CSV shows copy_strict@0.95=True, copy_soft_k1@0.5=True) and the JSON summary sets L_copy=0 with L_semantic=42 (output-gemma-2-9b.json:934,936). In gemma‑2‑27B strict L_copy=0 and soft k1=0 (output-gemma-2-27b.json:938,956), with L_semantic=46 (output-gemma-2-27b.json:940). All other models have no strict or soft hits at L0–L3 (e.g., Qwen3‑8B pure CSV shows False across L0–3 for both; output-Qwen3-8B-pure-next-token.csv:2–5). Outliers: gemma‑2‑9B is the strongest copy‑reflex (flags True at L0–3), gemma‑2‑27B shows L0 and occasional L3 soft flag, while Llama‑3, Mistral, Qwen, Yi show none.

- Rank milestones (within model; use ≤{10,5,1}):
  - Llama‑3‑70B: first_rank_le_1=40 and first_rank_le_5=38 (output-Meta-Llama-3-70B.json:981,982) ⇒ meaning emerges early (≈50% depth). KL last‑layer ≈0 (output-Meta-Llama-3-70B.json:1115).
  - Llama‑3‑8B: first_rank_le_1=25 (output-Meta-Llama-3-8B.json:933) but raw‑vs‑norm window flags norm‑only semantics around L25,27–30 (output-Meta-Llama-3-8B.json:1053). Treat early semantics cautiously here.
  - Mistral‑7B‑v0.1: first_rank_le_1=25, le_5=25, le_10=23 (output-Mistral-7B-v0.1.json:933–935) ⇒ mid/late earliness; last‑layer KL ≈0 (output-Mistral-7B-v0.1.json:1061).
  - Mistral‑Small‑24B‑Base‑2501: first_rank_le_1=33, le_5=30 (output-Mistral-Small-24B-Base-2501.json:941–942); last‑layer KL ≈0 (output-Mistral-Small-24B-Base-2501.json:1071).
  - Qwen3‑8B: first_rank_le_1=31, le_5=29 (output-Qwen3-8B.json:937–938); last‑layer KL ≈0 (output-Qwen3-8B.json:1064–1069).
  - Qwen3‑14B: first_rank_le_1=36, le_5=33, le_10=32 (output-Qwen3-14B.json:941–943); last‑layer KL ≈0 (output-Qwen3-14B.json:1068–1073).
  - Qwen2.5‑72B: first_rank_le_1=80, le_5=78, le_10=74 (output-Qwen2.5-72B.json:981–983); last‑layer KL ≈0 (output-Qwen2.5-72B.json:1104–1111). This is a late collapse (≈100% depth).
  - Yi‑34B: first_rank_le_1=44, le_5=44 (output-Yi-34B.json:961–962) with some norm‑only layers in the window (output-Yi-34B.json:1083).
  - Gemma‑2‑9B / ‑27B: collapse only at the final layer (L_semantic=42 and 46 respectively; output-gemma-2-9b.json:936, output-gemma-2-27b.json:940). Both exhibit last‑layer head mismatch (warn_high_last_layer_kl=true; output-gemma-2-9b.json:1069–1081, output-gemma-2-27b.json:1073–1081) ⇒ exclude from KL‑based cross‑model conclusions; use ranks only.

- KL milestones and calibration: Within‑model qualitative trends only. Where last‑layer KL≈0, KL typically declines with depth alongside rank improvements (e.g., Llama‑3‑70B last‑layer KL≈7.29e‑4; output-Meta-Llama-3-70B.json:1115). For Gemma, last‑layer KL≈1 bit with warn_high_last_layer_kl=true (output-gemma-2-9b.json:1069–1081; output-gemma-2-27b.json:1073–1081), indicating a calibrated final head; do not compare final probabilities across families in these cases.

- Direction alignment (cos_to_final; within‑model):
  - Llama‑3‑70B: cos thresholds reached only late (ge_{0.2,0.4,0.6}=80; output-Meta-Llama-3-70B.json:1066) ⇒ alignment consolidates near the end.
  - Llama‑3‑8B: ge_{0.2,0.4,0.6}≈{20,30,32} (output-Meta-Llama-3-8B.json:1018) ⇒ rising alignment before final.
  - Qwen3‑8B: ge_{0.2,0.4,0.6}=36 across the board (output-Qwen3-8B.json:1022), with L_geom_norm=34 (output-Qwen3-8B.json:1014).
  - Mistral‑7B‑v0.1: ge_0.4≈25 (output-Mistral-7B-v0.1.json:1021); L_geom_norm not reported (null; output-Mistral-7B-v0.1.json:1010). Final cos approaches 1.0 (pure CSV last row shows ≈0.9999999).
  - Yi‑34B: L_geom_norm=46 (output-Yi-34B.json:1038); cos milestones present (output-Yi-34B.json:1046).
  Treat all cosine comments as within‑model; do not compare absolute levels across families.

- Lens sanity (raw‑vs‑norm):
  - Models with first_norm_only_semantic_layer present: Llama‑3‑8B (25; output-Meta-Llama-3-8B.json:1583) and Yi‑34B (46; output-Yi-34B.json:1615). For these, downgrade confidence in “early semantics” and prefer ranks.
  - Lens artifact risk: “high” for Qwen3‑8B/14B/2.5‑72B (output-Qwen3-8B.json:1584; output-Qwen3-14B.json:1587; output-Qwen2.5-72B.json:1296), Gemma‑2‑9B/27B (output-gemma-2-9b.json:1589; output-gemma-2-27b.json:1593), Mistral‑7B‑v0.1 (output-Mistral-7B-v0.1.json:1581), Yi‑34B (output-Yi-34B.json:1617). “Low” for Llama‑3‑70B and Mistral‑Small‑24B‑Base (output-Meta-Llama-3-70B.json:1307; output-Mistral-Small-24B-Base-2501.json:1590).
  - Raw‑vs‑Norm window norm‑only semantics: Gemma‑2‑9B [42] (output-gemma-2-9b.json:1061), Gemma‑2‑27B [46] (output-gemma-2-27b.json:1065), Llama‑3‑8B [25,27,28,29,30] (output-Meta-Llama-3-8B.json:1053), Llama‑3‑70B [79,80] (output-Meta-Llama-3-70B.json:1106), Qwen2.5‑72B [80] (output-Qwen2.5-72B.json:1096), Yi‑34B [44,45,46,47,48,56,60] (output-Yi-34B.json:1083). Early semantics that disappear under the raw lens in ±4–8 layers should be treated as lens‑induced.

- Family similarities/differences:
  - Qwen (8B/14B/72B): L_semantic moves later with scale (Frac≈0.861/0.9/1.0; output-Qwen3-8B.json:1030; output-Qwen3-14B.json:— depth fractions key not mirrored; ranks suffice; output-Qwen2.5-72B.json:—), and aligns with stronger MMLU in the table. However, the late collapse in 72B (L_sem=80; output-Qwen2.5-72B.json:974) indicates that higher scores do not imply earlier collapse under this probe; instead, confidence consolidates at the end.
  - Gemma (9B/27B): both show strong early copy (L_copy=0; output-gemma-2-9b.json:934; output-gemma-2-27b.json:938) and only final‑layer semantics (42/46). Both have warn_high_last_layer_kl with calibrated heads (output-gemma-2-9b.json:1069–1086; output-gemma-2-27b.json:1073–1081).

- Δ and Δ̂ timing: Using strict L_copy when present; otherwise earliest soft k∈{1,2,3}.
  - Gemma‑2‑9B: Δ = 42 − 0 = 42; Δ̂ = 42/42 = 1.0 (output-gemma-2-9b.json:934,936; model_stats.num_layers=42 at output-gemma-2-9b.json:1523). Entropy drop between L_copy and L_semantic is sharp: H(0)≈1.67e−5 vs H(42)≈0.370 bits (pure CSV rows for L0 and L42).
  - Gemma‑2‑27B: Δ = 46 − 0 = 46; Δ̂ = 46/46 = 1.0 (output-gemma-2-27b.json:938,940; model_stats.num_layers=46 at output-gemma-2-27b.json:1527). Entropy drops from ≈4.97e−4 at L0 to ≈0.118 bits at L46 (pure CSV rows L0,L46).
  - Others: no strict/soft L_copy detected ⇒ Δ undefined; summarize via rank milestones and KL trends. For most bases here, collapse is late (≥70% depth) except Llama‑3‑70B (~50%).

- Entropy‑drop shape H(L_copy)→H(L_sem):
  - Gemma‑2‑9B: “cliff” to the end (near‑zero at L0 due to copy; moderate 0.37 bits at L42; pure CSV L0/L42).
  - Gemma‑2‑27B: similar “cliff” (pure CSV L0/L46).
  - Others (no copy layer): use mid‑depth drift as qualitative proxy; e.g., Qwen3‑8B mid‑depth drift (entropy − teacher_entropy_bits) ≈ +13.79 bits at L18 (output-Qwen3-8B-pure-next-token.csv:—), consistent with sharpening late.

- Early (<70%) vs late (≥70%) grouping by L_semantic_frac:
  - Early: Llama‑3‑70B only (L_semantic_frac=0.5; output-Meta-Llama-3-70B.json:1074).
  - Late: all others (e.g., Qwen3‑8B 0.861 at output-Qwen3-8B.json:1030; Qwen3‑14B 0.9 at output-Qwen3-14B.json:1034; Llama‑3‑8B 0.781 at output-Meta-Llama-3-8B.json:1026; Mistral‑7B 0.781 at output-Mistral-7B-v0.1.json:1026; Qwen2.5‑72B 1.0 at output-Qwen2.5-72B.json:1074; Gemma variants 1.0).

- Relation to scores (MMLU/ARC; within‑family caution): In Qwen, higher‑capacity models have higher public scores yet later collapse under this probe (8B→14B→72B). In Gemma, larger 27B does not “solve” late collapse; both end at the final layer. No clear correlation between sharper ΔH and d_model or n_heads: gemma‑2‑9B (d_model=3584,n_heads=16; output-gemma-2-9b.json:1524–1525) and gemma‑2‑27B (4608,32; output-gemma-2-27b.json:1528–1529) both show sharp ΔH, while Llama‑3‑70B (8192,64; output-Meta-Llama-3-70B.json:1517–1519) collapses early but gradually.

Prism Summary Across Models
- Artifacts present and compatible in all listed runs (k=512; e.g., Qwen3‑8B shows k=512 at output-Qwen3-8B.json:825–838). Rank deltas are null across models (no earlier ≤{10,5,1} milestones under Prism). KL deltas (baseline − Prism, positive means Prism reduces KL) cluster as:
  - Gemma‑2‑27B: large KL improvements at 25/50/75% (Δ≈+22.6/+23.7/+23.1 bits; output-gemma-2-27b.json:868–870) — Helpful, concentrated mid‑stack. High lens‑artifact risk model benefits more.
  - Gemma‑2‑9B: negative deltas (−11.6/−10.3/−24.4; output-gemma-2-9b.json:868–870) — Regressive.
  - Llama‑3‑8B: negative deltas (≈−5.37/−8.29/−9.78; output-Meta-Llama-3-8B.json:868–870) — Regressive at early/mid depths.
  - Llama‑3‑70B: small negative deltas (≈−0.89/−1.00/−1.16; output-Meta-Llama-3-70B.json:868–870) — Slightly regressive.
  - Mistral‑7B‑v0.1: larger negative deltas (≈−12.8/−17.5/−17.5; output-Mistral-7B-v0.1.json:868–870) — Regressive.
  - Mistral‑Small‑24B‑Base: modest negatives (≈−1.93/−5.98/−4.97; output-Mistral-Small-24B-Base-2501.json:868–870) — Slightly regressive.
  - Qwen3‑8B: small negatives (≈−0.36/−0.59/−7.03; output-Qwen3-8B.json:868–870) — Regressive (stronger at 75%).
  - Qwen3‑14B: small negatives (≈−0.26/−0.25/−0.71; output-Qwen3-14B.json:868–870) — Regressive.
  - Qwen2.5‑72B: mixed (≈+3.16/+2.83/−0.54; output-Qwen2.5-72B.json:868–870) — Neutral overall (early/mid gains, late flat/slight loss).
  - Yi‑34B: small positives at 25/50% and slight negative at 75% (≈+0.94/+1.36/−1.01; output-Yi-34B.json:868–870) — Neutral.
Interpretation: Prism gains are larger in a high‑risk lens regime for Gemma‑2‑27B (big mid‑depth KL reduction), but generally regressive for Llama‑3/Mistral/Qwen, and mixed for Yi/Qwen‑72B. Keep claims strictly within‑model; Prism is a shared‑decoder diagnostic, not the model head.

Tuned‑Lens (if present)
- Reported where sidecars exist. For Llama‑3‑8B, tuned median ΔKL (norm−tuned) ≈ +4–5 bits at depth percentiles, with last‑layer agreement KL≈0 (output-Meta-Llama-3-8B.md: Quantitative). For Gemma models, tuned summaries mirror late collapse (e.g., L_surface_to_meaning_tuned=46 in 27B; output-gemma-2-27b.json:1420–1427). Entropy drift at a mid‑depth snapshot remains large (e.g., Qwen3‑8B L≈18 shows +13.79 bits drift; pure CSV). Rank earliness: tuned often does not move rank milestones earlier in these single‑prompt runs.

Surface/Geometry/Coverage (norm; tuned when present)
- L_surface_to_meaning_norm and masses: Qwen3‑8B shows L_surface_to_meaning_norm=31 with answer_mass 0.936 and echo_mass 0.011 at L (output-Qwen3-8B.json:1011–1013). Llama‑3‑8B: L_surface_to_meaning_norm=32, answer_mass 0.520, echo_mass 0.024 (output-Meta-Llama-3-8B.json:1007–1009). Yi‑34B: L_surface_to_meaning_norm=51, answer_mass 0.060, echo_mass 0.006 (output-Yi-34B.json:1035–1037). Treat mass comparisons qualitatively within model.
- L_geom_norm and cosines: Llama‑3‑8B L_geom_norm=26 with cos_to_answer≈0.127 and cos_to_prompt_max≈0.097 (output-Meta-Llama-3-8B.json:1010). Qwen3‑8B L_geom_norm=34 (output-Qwen3-8B.json:1014). Yi‑34B L_geom_norm=46 (output-Yi-34B.json:1038). Tuned L_geom_tuned mirrors late/near‑final alignment for Gemma.
- L_topk_decay_norm (K=50, τ=0.33): many models show early near‑zero prompt coverage (e.g., Llama‑3‑70B topk_prompt_mass_at_L_norm=0.0 at L_topk_decay_norm=0; output-Meta-Llama-3-70B.json:1055–1058). Interpret as within‑model trends only.

Copy robustness (strict/soft)
- summary.copy_thresholds.stability: “mixed” for Gemma (output-gemma-2-9b.json:1015; output-gemma-2-27b.json:1019), “none” for Llama‑3/Mistral/Qwen/Yi (e.g., output-Meta-Llama-3-8B.json:1005; output-Mistral-Small-24B-Base-2501.json:1005; output-Qwen3-8B.json:1009; output-Yi-34B.json:1033). No norm_only_flags at τ∈{0.70,0.95} observed in these runs.

Link to public scores (advisory): Higher MMLU/ARC does not guarantee earlier collapse across families. Within Qwen, the highest‑scoring 72B collapses latest (L_sem=80). Within Llama, 70B collapses earlier (L≈0.5) than 8B (≈0.78), aligning loosely with its stronger scores.

References: Trends match prior interpretability observations that linear probes/lenses often show “early direction, late calibration” (arXiv:2309.00941; arXiv:2303.02436), and that normalization choices influence lens fidelity (arXiv:2309.08600). Use ranks and KL trends within model; avoid cross‑family absolute probabilities.

**Misinterpretations in Existing EVALS**
- Qwen3‑8B: The text asserts “no rows with copy_collapse=True in pos/orig for layers 0–36” without citing the earliest strict/soft sweep indices; it would be safer to reference the sweep block (copy_thresholds.stability="none") with exact JSON lines (output-Qwen3-8B.json:984–1009) alongside the CSV scan.
- Meta‑Llama‑3‑8B: Early semantics at L25 are described, but raw‑vs‑norm window lists norm‑only semantics at L25 and 27–30; the EVAL should explicitly downgrade these claims per raw‑lens window (output-Meta-Llama-3-8B.json:1053). The file notes caution elsewhere, but the punchline could over‑emphasize “early” without this caveat in the summary lines.
- Gemma‑2‑9B/‑27B: Some phrasing leans on final probabilities despite warn_high_last_layer_kl=true; cross‑model probability comparisons should be avoided and ranks emphasized (output-gemma-2-9b.json:1069–1086; output-gemma-2-27b.json:1073–1081). When citing final p_top1, include the mismatch note.
- Mistral‑7B‑v0.1: The fingerprint line claims “rest_mass declines to 0.230 by the final layer”; ensure this is clearly marked as coverage (not fidelity) and avoid implying calibration differences across models (output-Mistral-7B-v0.1.md:183). The EVAL body notes this elsewhere, but the bullet could mislead out of context.

**Limitations**
- RMS‑lens can distort absolute probabilities; keep comparisons within model, not across differing normalisation schemes.
- Single‑prompt probing may over‑fit tokenizer quirks; copy‑collapse depth can shift with wording/punctuation.
- Attention patterns and MLP activations are not inspected—only residual projections—so entropy bumps from internal gating may be mis‑attributed.
- Un‑embed weights may be promoted to FP32 (use_fp32_unembed=true) in some models, slightly shrinking entropy gaps; keep comparisons qualitative.
- Final‑lens vs final‑head mismatches can keep last‑layer KL>0 for some families/precisions; prefer rank thresholds for cross‑model conclusions and treat KL trends qualitatively within model.
- Layer counts differ (8B≈32, 34B≈48, 70B≈80); compare relative depths, not raw indices.
- Results are correlational; causal evidence (ablation/patching) awaits later runs.

---
Produced by OpenAI GPT-5
