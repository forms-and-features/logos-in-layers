# Cross‑Model Evaluation (Logit‑Lens, single‑prompt Germany→Berlin)

## 1) Result synthesis

Across ten models, semantic rank milestones cluster mid–late in depth, with copy‑reflex largely absent except in Gemma‑2. We ground comparisons in rank thresholds and within‑model KL/cosine trends, following best practice for lens analyses (cf. tuned‑lens style calibration checks; arXiv:2303.08112).

Copy‑reflex. Only Gemma‑2 shows an early copy reflex by our rule (any of copy_collapse or copy_soft_k1@τ_soft firing in L0–L3 on the pure CSV). Gemma‑2‑9B: “copy_collapse=True … copy_soft_k1@0.5=True” at L0 with top‑1 “simply” p=0.9999993 [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2]. Gemma‑2‑27B: same pattern at L0 with “simply” p=0.999976 [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2]. All other models have no strict or soft copy flag in L0–L3 (e.g., Meta‑Llama‑3‑70B rows 0–3 all False [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:1]). Gemma is therefore the outlier family for copy‑reflex in this prompt.

Emergence timing (rank milestones; relative depth). We categorize “early” (<70% depth) vs “late” (≥70%).
- Meta‑Llama‑3‑8B (32L): first_rank_le_1 at L25 (78%) — late; “first_rank_le_1: 25” [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:851].
- Meta‑Llama‑3‑70B (80L): first_rank_le_1 at L40 (50%) — early; “first_rank_le_1: 40” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:851].
- Mistral‑7B‑v0.1 (32L): first_rank_le_1 at L25 (78%) — late; “first_rank_le_1: 25” [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:851].
- Mistral‑Small‑24B‑Base‑2501 (40L): first_rank_le_1 at L33 (83%) — late; “first_rank_le_1: 33” [001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:851].
- Qwen‑3‑8B (36L): first_rank_le_1 at L31 (86%) — late; “first_rank_le_1: 31” [001_layers_baseline/run-latest/output-Qwen3-8B.json:851].
- Qwen‑3‑14B (40L): first_rank_le_1 at L36 (90%) — late; “first_rank_le_1: 36” [001_layers_baseline/run-latest/output-Qwen3-14B.json:851].
- Qwen‑2.5‑72B (80L): first_rank_le_1 at L80 (100%) — late; “first_rank_le_1: 80” [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:851].
- Yi‑34B (60L): first_rank_le_1 at L44 (73%) — late (borderline); “first_rank_le_1: 44” [001_layers_baseline/run-latest/output-Yi-34B.json:851].
- Gemma‑2‑9B (42L): first_rank_le_1 at L42 (100%) — late; “first_rank_le_1: 42” [001_layers_baseline/run-latest/output-gemma-2-9b.json:851].
- Gemma‑2‑27B (46L): first_rank_le_1 at L46 (100%) — late; “first_rank_le_1: 46” [001_layers_baseline/run-latest/output-gemma-2-27b.json:851].

Δ and Δ̂ (semantic − copy). Using L_copy (or fallback L_copy_H) and L_semantic from JSON diagnostics:
- Gemma‑2‑9B: L_copy=0; L_sem=42; Δ=42; Δ̂=1.00. “L_copy: 0 … L_semantic: 42” [001_layers_baseline/run-latest/output-gemma-2-9b.json:842,844].
- Gemma‑2‑27B: L_copy=0; L_sem=46; Δ=46; Δ̂=1.00. “L_copy: 0 … L_semantic: 46” [001_layers_baseline/run-latest/output-gemma-2-27b.json:842,844].
- Qwen‑3‑8B: L_copy_H=31; L_sem=31; Δ=0; Δ̂=0.00. “L_copy_H: 31 … L_semantic: 31” [001_layers_baseline/run-latest/output-Qwen3-8B.json:843,844].
- Qwen‑3‑14B: L_copy_H=32; L_sem=36; Δ=4; Δ̂=0.10. “L_copy_H: 32 … L_semantic: 36” [001_layers_baseline/run-latest/output-Qwen3-14B.json:843,844].
Other models have null L_copy and L_copy_H; we omit Δ in those cases.

Entropy shape (L_copy→L_semantic). We quote the pure CSV entropies in bits.
- Gemma‑2‑9B: near‑zero entropy at copy (L0, H≈1.7e‑05) rising to 0.370 bits at L_sem=42 — an “entropic bump,” not a drop: “entropy,1.6721e‑05 … copy_collapse=True” [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2]; “…,42,…,entropy,0.370067… is_answer=True” [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:42].
- Gemma‑2‑27B: H(L0)=0.00050 → H(L46)=0.118 — small increase toward semantics: “entropy,0.0004968 … copy_collapse=True” [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:2]; “…,46,…,entropy,0.1180477… is_answer=True” [001_layers_baseline/run-latest/output-gemma-2-27b-pure-next-token.csv:48].
- Qwen‑3‑8B: H(L31)=0.454 at both L_copy_H and L_sem (ΔH≈0): “…,31,…,entropy,0.4538646…, is_answer=True” [001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:33].
- Qwen‑3‑14B: H drops 0.816→0.312 bits (ΔH≈0.504) as rank becomes 1: “…,32,…,entropy,0.815953…, answer_rank,10” and “…,36,…,entropy,0.312212…, answer_rank,1” [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:34,38].
- Meta‑Llama‑3‑8B: L_sem at still‑high entropy (16.81 bits): “…,25,…,entropy,16.814249…, answer_rank,1” [001_layers_baseline/run-latest/output-Meta-Llama-3-8B-pure-next-token.csv:29].
- Meta‑Llama‑3‑70B: L_sem at 16.94 bits: “…,40,…,entropy,16.937426…, answer_rank,1” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:42].
- Mistral‑7B‑v0.1: L_sem at 13.60 bits: “…,25,…,entropy,13.598551…, answer_rank,1” [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:27].
- Qwen‑2.5‑72B: final L_sem at 4.12 bits: “…,80,…,entropy,4.115832…, answer_rank,1” [001_layers_baseline/run-latest/output-Qwen2.5-72B-pure-next-token.csv:138].
- Yi‑34B: L_sem at 15.33 bits: “…,44,…,entropy,15.327294…, answer_rank,1” [001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:46].
Within‑model cosine to final generally rises with depth: e.g., Gemma‑2‑9B cos 0→L_sem 0.9993 [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2,49]; Qwen‑3‑14B −0.137→0.610 [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:2,38]; Mistral‑7B −0.330→0.425 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:1,27]. We avoid cross‑family cosine comparisons.

KL trends and head calibration. We treat KL milestones as within‑model diagnostics and ignore models with non‑zero last‑layer head KL in cross‑model KL claims (Gemma‑2). Meta‑Llama‑3‑8B: first_kl_below_1.0 at L32 (final) and last‑layer KL≈0 (“first_kl_below_1.0: 32”; “kl_to_final_bits: 0.0”) [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:849,900]. Meta‑Llama‑3‑70B: KL<1.0 only at L80 (final) with small final KL (“first_kl_below_1.0: 80”; “kl_to_final_bits: 0.000729…”) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:849,900]. Qwen‑2.5‑72B and Qwen‑3‑14B similarly hit KL<1.0 only at the final layer [001_layers_baseline/run-latest/output-Qwen2.5-72B.json:849; 001_layers_baseline/run-latest/output-Qwen3-14B.json:849]. Gemma‑2 (9B/27B) have warn_high_last_layer_kl=true with final KL ≈1 bit (“warn_high_last_layer_kl: true … kl_to_final_bits: 1.01–1.14”) — annotate as a final‑lens vs final‑head mismatch and avoid probability conclusions [001_layers_baseline/run-latest/output-gemma-2-9b.json:917; 001_layers_baseline/run-latest/output-gemma-2-27b.json:917].

Lens sanity. Raw‑vs‑norm sampling flags “norm‑only semantics” in two families: Meta‑Llama‑3‑8B “first_norm_only_semantic_layer: 25 … lens_artifact_risk: high” [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:1072–1074] and Yi‑34B “first_norm_only_semantic_layer: 46 … lens_artifact_risk: high” [001_layers_baseline/run-latest/output-Yi-34B.json:1072–1074]. Low risk examples include Meta‑Llama‑3‑70B and Mistral‑Small‑24B (“lens_artifact_risk: low”; max_kl_norm_vs_raw_bits≈0.04/0.18) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1073–1074; 001_layers_baseline/run-latest/output-Mistral-Small-24B-Base-2501.json:1073–1074]. For models with “high” risk (Qwen, Gemma, Yi, Mistral‑7B), we downgrade confidence in pre‑final “early semantics” and rely on rank thresholds.

Family contrasts.
- Qwen‑3 (8B vs 14B): Neither shows early copy; both are late semantics (86–90% depth). L_copy_H suggests a brief non‑answer confidence surge before semantics at 14B (Δ=4, Δ̂=0.10) but no lag at 8B (Δ̂=0). “L_copy_H: 32/31; L_semantic: 36/31” [001_layers_baseline/run-latest/output-Qwen3-14B.json:843–844; output-Qwen3-8B.json:843–844]. Entropy drops modestly into semantics (0.82→0.31 bits at 14B) [001_layers_baseline/run-latest/output-Qwen3-14B-pure-next-token.csv:34,38]. Last‑layer KL≈0 for both.
- Gemma‑2 (9B vs 27B): Strong copy reflex at L0 for both, with semantics only at the very end (Δ̂≈1.0). Final‑head KL≈1 bit with warn_high flags [001_layers_baseline/run-latest/output-gemma-2-9b.json:900,917; output-gemma-2-27b.json:900,917]. Removing the stylistic filler does not shift semantics (ΔL_sem=0) [001_layers_baseline/run-latest/output-gemma-2-9b.json:1083–1089; output-gemma-2-27b.json:1083–1090].

Relation to model size and topology. We do not observe a simple association between sharper collapse and wider d_model or more heads. Llama‑3‑70B (d_model=8192, 64 heads) is “early” by rank (50% depth) but maintains high entropy at L_sem (16.94 bits) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1120; output-Meta-Llama-3-70B-pure-next-token.csv:42]. Qwen‑2.5‑72B (8192, 64) is “late” (100%) yet achieves the best public MMLU≈86% in the table, contradicting a monotonic “earlier semantics ⇒ higher exam scores” hypothesis. Overall, Δ/Δ̂ and entropy sharpness do not map cleanly onto d_model/n_heads.

External eval linkage (MMLU/ARC; base vs instruct). Early‑vs‑late semantics does not straightforwardly predict public exam scores. High‑scoring Qwen‑2.5‑72B (MMLU≈86%) is late (100%); Meta‑Llama‑3‑70B (MMLU≈79.5%) is early (50%); Mistral‑Small‑24B (MMLU≈80.7%) is late (83%). Yi‑34B is late (73%) with very strong scores (MMLU≈76%). Instruction‑tuning vs base also interacts (some bases here are quite competitive), suggesting that lens emergence depth is more a description of internal calibration dynamics than a proxy for factual‑reasoning capability.

Prism summary across models. All artifacts are present/compatible (k=512; sampled layers differ per model, e.g., Llama‑3‑8B: layers [embed,7,15,23] [001_layers_baseline/run-latest/output-Meta-Llama-3-8B.json:825]). We compare Prism vs baseline at ~25/50/75% depth using KL_to_final and first rank milestones.
- Gemma‑2‑9B (high lens‑risk). ΔKL>0 (worse) at 25/50/75%: +11.63/+10.33/+24.42 bits; Prism never reaches earlier rank thresholds (p1/p5/p10: n/a vs 42/42/42).
- Gemma‑2‑27B (high lens‑risk). ΔKL<0 (better) at 25/50/75%: −22.57/−23.73/−23.08 bits; no earlier rank thresholds (Prism p1/p5/p10: n/a vs 46/46/46).
- Meta‑Llama‑3‑8B (high lens‑risk due to norm‑only semantics). ΔKL>0 (worse): +5.37/+8.29/+9.78 bits; no earlier rank thresholds (n/a vs 24–25).
- Meta‑Llama‑3‑70B (low lens‑risk). Small ΔKL>0 at all three points (~+0.9/+1.0/+1.16); no earlier ranks.
- Mistral‑7B‑v0.1 (high lens‑risk). ΔKL>0 large (+12.81/+17.54/+17.50); no earlier ranks.
- Mistral‑Small‑24B (low lens‑risk). ΔKL>0 moderate (+1.93/+5.98/+4.97); no earlier ranks.
- Qwen‑3‑8B and Qwen‑3‑14B (high lens‑risk). ΔKL>0 small→moderate (+0.36/+0.59/+7.03; +0.26/+0.25/+0.71); no earlier ranks.
- Qwen‑2.5‑72B (high lens‑risk). ΔKL<0 at 25/50% (−3.16/−2.83), +0.54 at 75%; no earlier ranks.
- Yi‑34B (high lens‑risk; norm‑only semantics). ΔKL<0 at 25/50% (−0.94/−1.36), +1.01 at 75%; no earlier ranks.
Overall, Prism gains (ΔKL<0) occur sporadically and mostly at early/mid layers when the baseline lens‑artifact risk is high (e.g., Gemma‑27B, Qwen‑2.5‑72B, Yi‑34B), but they rarely translate into earlier rank milestones in this setup; several models show clear regressions. Keep claims within‑model (no cross‑family probability comparisons).

## 2) Misinterpretations in existing EVALS
- Yi‑34B EVAL states “Prism pure CSV lacks answer_rank fields (null)” [001_layers_baseline/run-latest/evaluation-Yi-34B.md:148]. The Prism CSV includes an `answer_rank` column (see header) and populated values (e.g., line 2 shows `answer_rank,48899`) [001_layers_baseline/run-latest/output-Yi-34B-pure-next-token-prism.csv:1–2]. The lack of earlier rank milestones reflects ranks not crossing ≤{10,5,1}, not missing fields.
- Gemma‑2‑9B EVAL highlights “final‑layer p_top1 = 0.9298” as a confidence milestone without re‑emphasizing the warn_high_last_layer_kl=true flag [001_layers_baseline/run-latest/evaluation-gemma-2-9b.md:64 vs output-gemma-2-9b.json:917]. Given final‑head KL≈1.01 bits [output-gemma-2-9b.json:900], absolute probabilities at the final lens should not be used beyond within‑model qualitative trends; rank thresholds are the safe cross‑model comparator.

## 3) Limitations
- RMS‑lens can distort absolute probabilities; comparisons should stay within‑model, not across differing normalisation schemes.
- Single‑prompt probing may over‑fit tokenizer quirks; copy‑collapse depth can change if wording or punctuation shifts.
- Attention patterns and MLP activations are not inspected—only residual projections—so entropy bumps caused by internal gating may be mis‑attributed.
- Un‑embed weights may be promoted to FP32 ("use_fp32_unembed": true) in some models, slightly shrinking entropy gaps; keep comparisons qualitative.
- Final‑lens vs final‑head mismatches can keep last‑layer KL > 0 for some families/precisions; prefer rank thresholds for cross‑model conclusions and treat KL trends qualitatively within model.
- Layer counts differ (8B ≈ 32 layers, 34B ≈ 48–60); compare relative depths, not absolute indices.
- Current results are correlation‑only; causal evidence (patching) awaits a later run.

---
Produced by OpenAI GPT-5

