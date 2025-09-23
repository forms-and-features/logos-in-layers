**Overview**
- Model: mistralai/Mistral-7B-v0.1 (7B). Probe runs a norm-lens logit-lens pass over all layers and logs per-layer entropy, KL-to-final, cosine-to-final, copy flags, and answer calibration.
- The analysis targets the next token after “...Germany is called simply”, with ID-aligned gold token “Berlin”, plus a France→Paris control and a no‑filler ablation.

**Method Sanity‑Check**
- Norm lens and positional info: “use_norm_lens": true … “layer0_position_info": "token_only_rotary_model” [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:807,816].
- Context prompt ends with “called simply” (no trailing space): “context_prompt": "… Germany is called simply” [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4].
- Final‑head calibration present and ≈0 KL: “last_layer_consistency.kl_to_final_bits": 0.0; top1_agree: true; p_top1_lens=p_top1_model=0.3822 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:899–907].
- Copy metrics and config present: “L_copy": null, “L_copy_H": null, “L_semantic": 25, “delta_layers": null [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:842–845]; “copy_thresh": 0.95, “copy_window_k": 1, “copy_match_level": "id_subsequence” [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:846–848]. Soft config: “threshold": 0.5, “window_ks": [1,2,3], “extra_thresholds": [] [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:833–841].
- Flags mirrored in CSV/JSON: copy_flag_columns = [copy_strict@0.95, copy_soft_k{1,2,3}@0.5] [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1077–1082].
- Gold‑token alignment: diagnostics.gold_alignment = "ok” [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:898]. Alignment is ID‑based (gold_answer.first_id=8430) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1115].
- Negative control present: control_prompt/context (France→Paris) and summary: first_control_margin_pos=2; max_control_margin=0.6539 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1091–1109].
- Ablation present: ablation_summary with L_sem_nf=24 (ΔL_sem = −1) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1084–1089]. Both variants appear in CSV (e.g., “pos,no_filler,0,…” [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:35]).
- Summary indices (bits/ranks): first_kl_below_0.5=32; first_kl_below_1.0=32; first_rank_le_1=25; first_rank_le_5=25; first_rank_le_10=23 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:849–853]. KL/entropy units are bits (CSV “entropy”, “kl_to_final_bits”).
- Lens sanity (raw vs norm): mode=sample; lens_artifact_risk=high; max_kl_norm_vs_raw_bits=1.1739; first_norm_only_semantic_layer=null [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1016,1071–1075]. Caution: treat any “early semantics” conservatively; prefer rank milestones.
- Copy‑collapse flag check (strict τ=0.95, δ=0.10): no “copy_collapse=True” in layers 0–3 (pure CSV rows 2–5). Soft copy (τ_soft=0.5, k∈{1,2,3}) never fires: L_copy_soft.k{1,2,3}=null [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:860–868].
- First strict copy row: none. Earliest soft: none. ✓ rule not triggered.

**Quantitative Findings**
- Per‑layer (pos, orig):
  - L 0 – entropy 14.9614 bits, top‑1 ‘dabei’
  - L 1 – entropy 14.9291 bits, top‑1 ‘biologie’
  - L 2 – entropy 14.8254 bits, top‑1 ‘"’
  - L 3 – entropy 14.8771 bits, top‑1 ‘[…]’
  - L 4 – entropy 14.8538 bits, top‑1 ‘[…]’
  - L 5 – entropy 14.8265 bits, top‑1 ‘[…]’
  - L 6 – entropy 14.8378 bits, top‑1 ‘[…]’
  - L 7 – entropy 14.8049 bits, top‑1 ‘[…]’
  - L 8 – entropy 14.8210 bits, top‑1 ‘[…]’
  - L 9 – entropy 14.7755 bits, top‑1 ‘[…]’
  - L 10 – entropy 14.7816 bits, top‑1 ‘[…]’
  - L 11 – entropy 14.7363 bits, top‑1 ‘[…]’
  - L 12 – entropy 14.6418 bits, top‑1 ‘[…]’
  - L 13 – entropy 14.7261 bits, top‑1 ‘[…]’
  - L 14 – entropy 14.6531 bits, top‑1 ‘[…]’
  - L 15 – entropy 14.4497 bits, top‑1 ‘[…]’
  - L 16 – entropy 14.5998 bits, top‑1 ‘[…]’
  - L 17 – entropy 14.6278 bits, top‑1 ‘[…]’
  - L 18 – entropy 14.5197 bits, top‑1 ‘[…]’
  - L 19 – entropy 14.5104 bits, top‑1 ‘[…]’
  - L 20 – entropy 14.4242 bits, top‑1 ‘simply’
  - L 21 – entropy 14.3474 bits, top‑1 ‘simply’
  - L 22 – entropy 14.3874 bits, top‑1 ‘“’
  - L 23 – entropy 14.3953 bits, top‑1 ‘simply’
  - L 24 – entropy 14.2124 bits, top‑1 ‘simply’
  - **L 25 – entropy 13.5986 bits, top‑1 ‘Berlin’**
  - L 26 – entropy 13.5409 bits, top‑1 ‘Berlin’
  - L 27 – entropy 13.2964 bits, top‑1 ‘Berlin’
  - L 28 – entropy 13.2962 bits, top‑1 ‘Berlin’
  - L 29 – entropy 11.4269 bits, top‑1 ‘""""’
  - L 30 – entropy 10.7970 bits, top‑1 ‘“’
  - L 31 – entropy 10.9943 bits, top‑1 ‘""""’
  - L 32 – entropy 3.6110 bits, top‑1 ‘Berlin’
  (Source: 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv; rows for pos,orig.)

- Control margin (France→Paris): first_control_margin_pos=2; max_control_margin=0.6539 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1106–1109].

- Ablation (no‑filler): L_copy_orig=null; L_sem_orig=25; L_copy_nf=null; L_sem_nf=24; ΔL_copy=null; ΔL_sem=−1 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1084–1089]. Interpretation: removal of “simply” advances semantics slightly (−1 layer), suggesting mild stylistic anchoring.

- ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n/a (L_copy=null).
- Soft ΔHk (k∈{1,2,3}) = n/a (no soft‑copy layers; all null).
- Confidence milestones (pure CSV): p_top1 > 0.30 at layer 32; p_top1 > 0.60 = n/a; final‑layer p_top1 = 0.3822 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:34].
- Rank milestones (diagnostics): rank ≤ 10 at L23; rank ≤ 5 at L25; rank ≤ 1 at L25 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:851–853].
- KL milestones (diagnostics): first_kl_below_1.0 at L32; first_kl_below_0.5 at L32; KL decreases with depth and is ≈0 at final [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:849–850].
- Cosine milestones (pure CSV): cos_to_final ≥0.2 at L11; ≥0.4 at L25; ≥0.6 at L26; final cos_to_final = 0.99999988 (≈1.0) [derived from 001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv].

Prism Sidecar Analysis
- Presence: compatible=true, mode=auto, k=512, layers=[embed,7,15,23] [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:819–831].
- Early‑depth stability (KL): baseline vs Prism at sampled depths:
  - L0 KL: 10.17 (baseline) vs 10.40 (Prism)
  - L7 KL: 10.28 vs 24.94
  - L15 KL: 10.27 vs 30.65
  - L23 KL: 9.45 vs 26.85
  - L32 KL: 0.00 vs 44.38
  (Source: pure CSVs; see exemplar rows printed in analysis.) Clear KL increase under Prism at all sampled depths.
- Rank milestones (Prism pure): first_rank_le_10=31; first_rank_le_5=never; first_rank_le_1=never (answer_rank at L32=49; is_answer=false) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token-prism.csv:34].
- Top‑1 agreement: Prism top‑1 diverges strongly from final answer at all depths including final (top‑1 is not ‘Berlin’ at L32) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token-prism.csv:34].
- Cosine drift: Prism cos_to_final remains small/negative at mid‑stack (e.g., L15 cos≈−0.44) vs baseline trending positive and >0.4 by L25.
- Copy flags: no spurious flips observed (strict/soft false in both CSVs at early layers).
- Verdict: Regressive (KL increases markedly and first_rank_le_1 never occurs under Prism).

**Qualitative Patterns & Anomalies**
The negative control “Berlin is the capital of” produces the expected country with Berlin not in the top‑5: “Germany, 0.8966; the, 0.0539; both, 0.00436; a, 0.00380; Europe, 0.00311” and “Berlin, 0.00284” at rank 6 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:10–35]. No semantic leakage to the city in the top‑5; semantic leakage: Berlin rank 6 (p = 0.00284).

Across prompt tokens, “important words” rise with depth before semantic collapse. “Berlin” first appears in any top‑5 around L22 (e.g., token “is”: “… Berlin, 0.0020” [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-records.csv:390]) and strengthens by L24 (“is”: “Berlin, 0.0133”; “called”: “Berlin, 0.0108”) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-records.csv:424–425]. At L25 it dominates the next‑token position: “(layer 25, token = ‘Berlin’, p = 0.0335)” [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:27]. “Germany” remains prominent in top‑5 through L24–L27 at multiple positions (e.g., L24 pos 13: “Germany, 0.00477 … Berlin, 0.00139”) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-records.csv:423]. “capital” features in the mid‑stack before the city overtakes (e.g., L23 pos 14: “capital, 0.0070; Berlin, 0.0036”) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-records.csv:407].

The no‑filler ablation advances semantics slightly (L_sem 24 vs 25; ΔL_sem = −1) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1084–1089], suggesting the “simply” adverb acts as stylistic guidance rather than essential content. Rest_mass declines with depth after collapse (e.g., rest_mass 0.9106 at L25 → 0.2298 at L32) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:27,34], consistent with sharper distributions; note rest_mass is top‑k coverage only, not a fidelity metric.

Rotation vs amplification: cosine to the final direction rises early (≥0.2 by L11) while KL to final remains ≈10 bits until late, indicating “early direction, late calibration.” Semantic collapse (rank→1) occurs at L25 while KL continues decreasing toward ≈0 by L32. With last‑layer calibration clean (kl_to_final_bits=0.0; temp_est=1.0) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:899–907], probability statements at the final layer are well‑calibrated in‑model.

Lens sanity: raw‑vs‑norm check flags lens_artifact_risk = high and max_kl_norm_vs_raw_bits ≈1.17 bits [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1071–1075]. No “norm‑only semantics” layer is reported (first_norm_only_semantic_layer=null). Treat any apparent “early semantics” prior to L25 cautiously and prefer rank thresholds over raw probabilities.

Temperature robustness: at T=0.1, “Berlin” rank 1 with p≈0.9996 and entropy≈0.005 bits [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:669–676,671]. At T=2.0, “Berlin” remains top‑1 with p≈0.036 and entropy≈12.22 bits [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:736–744,738].

Checklist
- RMS lens? ✓ (RMSNorm; normalized lens) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:807,810–816]
- LayerNorm bias removed? ✓ (not needed for RMS) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:812]
- Entropy rise at unembed? ✗ (entropy decreases toward final; 10.99 → 3.61 bits) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:33–34]
- FP32 un‑embed promoted? ✓ (unembed_dtype="torch.float32") [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:809]
- Punctuation / markup anchoring? ✓ (mid/late layers show quotes/marks as top‑1) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:31–33]
- Copy‑reflex? ✗ (no strict/soft copy flags at L0–3) [pure CSV rows 2–5]
- Grammatical filler anchoring? ✗ (L0–5 top‑1 not in {is,the,a,of}) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:2–7]

**Limitations & Data Quirks**
- raw_lens_check marks lens_artifact_risk=high; treat pre‑final “semantics” cautiously and rely on rank milestones [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1071–1075].
- Rest_mass after L_semantic remains high early (e.g., 0.91 at L25) and declines to 0.23 at final; rest_mass is top‑k coverage only and not a fidelity metric [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:27,34].
- Prism sidecar is regressive for this model/prompt (KL grows; answer never reaches rank 1, even at L32), so Prism‑based probabilities are not directly comparable here [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token-prism.csv:34].
- Findings use sampled raw‑vs‑norm checks (mode=sample), so raw/norm delta is a sanity indicator rather than exhaustive measurement [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:1016].

**Model Fingerprint**
“Mistral‑7B‑v0.1: collapse at L 25; final entropy 3.61 bits; ‘Berlin’ stabilizes top‑1 from mid‑20s.”

---
Produced by OpenAI GPT-5 

