**Overview**
- Model: 01-ai/Yi-34B (pre_norm; 60 layers)
- The probe tracks layer-by-layer next-token behavior under a norm lens, with tuned-lens and Prism sidecars. Confirmed semantic collapse occurs in the mid–late stack; KL-to-final drops to ~0 by the last layer.

**Method Sanity-Check**
- Norm lens enabled and RMSNorm pipeline engaged; positional info is rotary-only at layer 0: “use_norm_lens: true” and “layer0_position_info: token_only_rotary_model” (001_layers_baseline/run-latest/output-Yi-34B.json:807,816). The context prompt ends with “called simply” (001_layers_baseline/run-latest/output-Yi-34B.json:817).
- Gold alignment is ok (ID-level) and last-layer head calibration agrees: “gold_alignment: ok” (001_layers_baseline/run-latest/output-Yi-34B.json:1107); “kl_to_final_bits: 0.000278…, top1_agree: true, p_top1_lens: 0.5555, p_top1_model: 0.5627, temp_est: 1.0, warn_high_last_layer_kl: false” (001_layers_baseline/run-latest/output-Yi-34B.json:1109–1116,1126).
- Copy detection config/flags present and consistent across JSON/CSV: copy_strict τ∈{0.70,0.80,0.90,0.95} and soft k∈{1,2,3} with threshold 0.5; flag columns: ["copy_strict@0.95","copy_strict@0.7","copy_strict@0.8","copy_strict@0.9","copy_soft_k1@0.5","copy_soft_k2@0.5","copy_soft_k3@0.5"] (001_layers_baseline/run-latest/output-Yi-34B.json:1654–1661). No strict or soft copy layer found (all null) across thresholds (001_layers_baseline/run-latest/output-Yi-34B.json:1758–1769,1770–1776,1806–1819,1861–1872,1873–1879).
- Raw-vs-Norm window: radius=4, centers=[44,60], with norm-only semantics in layers [44,45,46,47,48,56,60]; max KL(norm||raw) in window = 90.47 bits; mode=window (001_layers_baseline/run-latest/output-Yi-34B.json:1062–1066,1083–1092). Full check: pct_kl≥1.0=0.656, pct_kl≥0.5=0.820, n_norm_only_semantics_layers=14, earliest_norm_only_semantic=44, max_kl_norm_vs_raw_bits=90.47, tier=high (001_layers_baseline/run-latest/output-Yi-34B.json:1096–1105).
- Diagnostics indices present: first_kl_below_1.0=60, first_kl_below_0.5=60, first_rank_le_1=44, first_rank_le_5=44, first_rank_le_10=43 (001_layers_baseline/run-latest/output-Yi-34B.json:841–843,873–877,2160–2164).
- Measurement guidance: prefer ranks; suppress abs probs; preferred lens for reporting = “tuned”; use_confirmed_semantics = true (001_layers_baseline/run-latest/output-Yi-34B.json:2189–2200). Confirmed semantics: L_semantic_confirmed=44, source=tuned (001_layers_baseline/run-latest/output-Yi-34B.json:1475–1482).
- Prism sidecar: compatible=true; percentiles show baseline→Prism KL changes p25: 13.12→12.18 (Δ≈0.94), p50: 13.54→12.17 (Δ≈1.36), p75: 11.16→12.17 (Δ≈-1.01) (001_layers_baseline/run-latest/output-Yi-34B.json:856–871).

Copy-collapse flags: no strict or soft hits. First row with copy_collapse=True: n.a. ✓ rule did not fire spuriously.
Soft copy: no k∈{1,2,3} hits at τ_soft=0.5.

Lens selection: preferred lens for milestones is “tuned”, but baseline is always reported for context. Confirmed semantics reported (L_semantic_confirmed=44 via tuned corroboration). Geometry, coverage, and norm-temperature diagnostics are present: L_geom_norm=46; cosine milestones (norm) ge_0.2:1, ge_0.4:44, ge_0.6:51 (001_layers_baseline/run-latest/output-Yi-34B.json:1038–1046). Norm-temperature snapshots present at 25/50/75%: 12.21, 12.26, 6.77 bits (001_layers_baseline/run-latest/output-Yi-34B.json:1128–1136).

**Quantitative Findings**
Table (pos, orig, NEXT only; one row per layer)

- L 0 – entropy 15.962 bits, top‑1 ' Denote'
- L 1 – entropy 15.942 bits, top‑1 '.'
- L 2 – entropy 15.932 bits, top‑1 '.'
- L 3 – entropy 15.839 bits, top‑1 'MTY'
- L 4 – entropy 15.826 bits, top‑1 'MTY'
- L 5 – entropy 15.864 bits, top‑1 'MTY'
- L 6 – entropy 15.829 bits, top‑1 'MTQ'
- L 7 – entropy 15.862 bits, top‑1 'MTY'
- L 8 – entropy 15.873 bits, top‑1 '其特征是'
- L 9 – entropy 15.836 bits, top‑1 '审理终结'
- L 10 – entropy 15.797 bits, top‑1 '~\\\\'
- L 11 – entropy 15.702 bits, top‑1 '~\\\\'
- L 12 – entropy 15.774 bits, top‑1 '~\\\\'
- L 13 – entropy 15.784 bits, top‑1 '其特征是'
- L 14 – entropy 15.739 bits, top‑1 '其特征是'
- L 15 – entropy 15.753 bits, top‑1 '其特征是'
- L 16 – entropy 15.714 bits, top‑1 '其特征是'
- L 17 – entropy 15.714 bits, top‑1 '其特征是'
- L 18 – entropy 15.716 bits, top‑1 '其特征是'
- L 19 – entropy 15.696 bits, top‑1 'ncase'
- L 20 – entropy 15.604 bits, top‑1 'ncase'
- L 21 – entropy 15.609 bits, top‑1 'ODM'
- L 22 – entropy 15.620 bits, top‑1 'ODM'
- L 23 – entropy 15.602 bits, top‑1 'ODM'
- L 24 – entropy 15.548 bits, top‑1 'ODM'
- L 25 – entropy 15.567 bits, top‑1 'ODM'
- L 26 – entropy 15.585 bits, top‑1 'ODM'
- L 27 – entropy 15.227 bits, top‑1 'ODM'
- L 28 – entropy 15.432 bits, top‑1 'MTU'
- L 29 – entropy 15.467 bits, top‑1 'ODM'
- L 30 – entropy 15.551 bits, top‑1 'ODM'
- L 31 – entropy 15.531 bits, top‑1 ' 版的'
- L 32 – entropy 15.455 bits, top‑1 'MDM'
- L 33 – entropy 15.455 bits, top‑1 'XFF'
- L 34 – entropy 15.477 bits, top‑1 'XFF'
- L 35 – entropy 15.471 bits, top‑1 'Mpc'
- L 36 – entropy 15.433 bits, top‑1 'MDM'
- L 37 – entropy 15.454 bits, top‑1 'MDM'
- L 38 – entropy 15.486 bits, top‑1 'MDM'
- L 39 – entropy 15.504 bits, top‑1 'MDM'
- L 40 – entropy 15.528 bits, top‑1 'MDM'
- L 41 – entropy 15.519 bits, top‑1 'MDM'
- L 42 – entropy 15.535 bits, top‑1 'keV'
- L 43 – entropy 15.518 bits, top‑1 ' "'
- L 44 – entropy 15.327 bits, top‑1 'Berlin'  ← confirmed semantic collapse
- L 45 – entropy 15.293 bits, top‑1 'Berlin'
- L 46 – entropy 14.834 bits, top‑1 'Berlin'
- L 47 – entropy 14.731 bits, top‑1 'Berlin'
- L 48 – entropy 14.941 bits, top‑1 'Berlin'
- L 49 – entropy 14.696 bits, top‑1 'Berlin'
- L 50 – entropy 14.969 bits, top‑1 'Berlin'
- L 51 – entropy 14.539 bits, top‑1 'Berlin'
- L 52 – entropy 15.137 bits, top‑1 'Berlin'
- L 53 – entropy 14.870 bits, top‑1 'Berlin'
- L 54 – entropy 14.955 bits, top‑1 'Berlin'
- L 55 – entropy 14.932 bits, top‑1 'Berlin'
- L 56 – entropy 14.745 bits, top‑1 'Berlin'
- L 57 – entropy 14.748 bits, top‑1 ' '
- L 58 – entropy 13.457 bits, top‑1 ' '
- L 59 – entropy 7.191 bits, top‑1 ' '
- L 60 – entropy 2.981 bits, top‑1 'Berlin'

Control margin (JSON): first_control_margin_pos=1; max_control_margin=0.5836 (001_layers_baseline/run-latest/output-Yi-34B.json:1686–1688).

Ablation (no‑filler): L_copy_orig=null, L_sem_orig=44; L_copy_nf=null, L_sem_nf=44; ΔL_copy=null, ΔL_sem=0 (001_layers_baseline/run-latest/output-Yi-34B.json:1664–1669).

- ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (L_copy_strict=null; soft also null).
- Soft ΔHₖ (bits) = n.a. (no soft copy layer across k∈{1,2,3}).
- Confidence milestones (p_top1, baseline): >0.30 at L 60; >0.60: n.a.; final-layer p_top1=0.5555.
- Rank milestones (preferred tuned first, baseline in parentheses): rank≤10 at L 44 (43); rank≤5 at L 44 (44); rank=1 at L 46 (44) (001_layers_baseline/run-latest/output-Yi-34B.json:2127–2136).
- KL milestones (baseline): first_kl_below_1.0 at L 60; first_kl_below_0.5 at L 60; KL decreases with depth and is ≈0 at final (001_layers_baseline/run-latest/output-Yi-34B.json:2160–2164,1109).
- Cosine milestones (norm): cos_to_final ≥0.2 at L 1; ≥0.4 at L 44; ≥0.6 at L 51; final cos_to_final ≈1.0 (001_layers_baseline/run-latest/output-Yi-34B.json:1038–1046,1109–1114).
- Depth fractions: L_semantic_frac=0.733 (001_layers_baseline/run-latest/output-Yi-34B.json:1054–1060).

Copy robustness (threshold sweep): stability=“none”; earliest L_copy_strict at τ=0.70 and τ=0.95 are null; norm_only_flags at these τ are null as well (001_layers_baseline/run-latest/output-Yi-34B.json:1752–1777,1958–1983).

Prism Sidecar Analysis: compatible=true (001_layers_baseline/run-latest/output-Yi-34B.json:828–835). Early/mid KL vs final (baseline→Prism): p25 13.12→12.18 (Δ≈0.94 bits), p50 13.54→12.17 (Δ≈1.36 bits), p75 11.16→12.17 (Δ≈−1.01 bits) (001_layers_baseline/run-latest/output-Yi-34B.json:856–871). Rank milestones are null in the JSON summary. Sampling Prism pure CSV shows top‑1 tokens do not align with final head at key depths (e.g., “L 44, top‑1 ‘upl’, p≈9.69e-05” [001_layers_baseline/run-latest/output-Yi-34B-pure-next-token-prism.csv: row with layer=44]). Verdict: Neutral to regressive (mixed KL deltas; no earlier rank‑1 corroboration; frequent top‑1 disagreement).

**Qualitative Patterns & Anomalies**
The negative control “Berlin is the capital of” yields a strong country token: “Germany, 0.8398; the, 0.0537; which, 0.0288; what, 0.0120; Europe, 0.0060” (001_layers_baseline/run-latest/output-Yi-34B.json:1–40). No semantic leakage of “Berlin” is observed in this control (top‑5 shown).

In the baseline norm lens, rank‑1 for the gold answer appears at L 44 (confirmed), with KL to final decaying thereafter and cosine to final rising. Concrete slices illustrate rotation followed by calibration: at L 44 “p_answer=0.00846, KL=11.41 bits, cos_to_final=0.432” (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:46); at L 60 “p_answer=0.5555, KL≈0.00028 bits, cos_to_final≈1.0” (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:row with layer=60). This is consistent with “early direction, late calibration” behavior.

Records (prompt tokens) show early layers dominated by punctuation, non‑word and markup‑like tokens rather than semantic anchors (e.g., for the prompt token “ capital”, the top‑1 entries include punctuation/markers across shallow layers) (001_layers_baseline/run-latest/output-Yi-34B-records.csv:13,30,47). In NEXT distribution, semantically related tokens begin to appear near L≈44, with “Berlin”, “capital”, and “Germany” all in the top‑k: “(layer 46, token = ‘Berlin’, p = 0.0345)” (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:48).

Removing the stylistic filler (“simply”) does not shift collapse: L_sem_nf remains 44 (ΔL_sem=0) (001_layers_baseline/run-latest/output-Yi-34B.json:1664–1669). Rest_mass declines with depth but is high at the onset of semantics (coverage only): “rest_mass at L 44 = 0.9807; final rest_mass = 0.1753”.

Rotation vs amplification: KL-to-final decreases monotonically post‑collapse while answer rank is 1 and p_answer rises, with cos_to_final increasing from ≈0.43 at L 44 to ≈1.0 at L 60 (001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:46, row with layer=60). Last‑layer head calibration is clean (kl_after_temp_bits≈0; temp_est=1.0) (001_layers_baseline/run-latest/output-Yi-34B.json:1109–1116,1126).

Lens sanity: raw_lens_check indicates high artefact risk and norm‑only semantics (earliest at L 46; max KL 80.57 bits) (001_layers_baseline/run-latest/output-Yi-34B.json:1649–1652). The full dual‑lens summary corroborates a high artefact tier (score≈0.793, tier=high) with 14 norm‑only semantics layers (001_layers_baseline/run-latest/output-Yi-34B.json:1096–1105). Given measurement_guidance (prefer ranks/suppress abs probs; prefer tuned), statements are rank‑centric and rely on confirmed semantics.

Temperature robustness: at T=0.1, entropy collapses to ~7.1e‑06 bits; at T=2.0, “Berlin” remains top‑k with p≈0.0488 and entropy≈12.49 bits (001_layers_baseline/run-latest/output-Yi-34B.json:720–752). This reflects expected broadening under high temperature.

Important‑word trajectory: “Berlin” first appears rank‑1 at L 44 (confirmed), and remains rank‑1 across most mid‑late layers; semantically close words like “capital” and “Germany” co‑occur in top‑k around collapse (e.g., “(layer 45, token = ‘Berlin’, p = 0.0075; ‘capital’, 0.0054)” [001_layers_baseline/run-latest/output-Yi-34B-pure-next-token.csv:47]).

Checklist
- RMS lens? ✓ (RMSNorm; norm lens on) (001_layers_baseline/run-latest/output-Yi-34B.json:807,810–816)
- LayerNorm bias removed? ✓ (not needed for RMS) (001_layers_baseline/run-latest/output-Yi-34B.json:812)
- Entropy rise at unembed? n.a. (not directly reported)
- FP32 un‑embed promoted? ✓ (use_fp32_unembed: true; unembed_dtype: torch.float32) (001_layers_baseline/run-latest/output-Yi-34B.json:808–809)
- Punctuation / markup anchoring? ✓ (records show punctuation/markers early) (001_layers_baseline/run-latest/output-Yi-34B-records.csv:13,30,47)
- Copy‑reflex? ✗ (no strict/soft hits in layers 0–3) (001_layers_baseline/run-latest/output-Yi-34B.json:1758–1777)
- Grammatical filler anchoring? ✗ (no {is,the,a,of} as top‑1 in layers 0–5 of NEXT)
- Preferred lens honored? ✓ (tuned milestones foregrounded; baseline quoted) (001_layers_baseline/run-latest/output-Yi-34B.json:2127–2136,2189–2200)
- Confirmed semantics reported? ✓ (L_semantic_confirmed=44; source=tuned) (001_layers_baseline/run-latest/output-Yi-34B.json:1475–1482)
- Full dual‑lens metrics cited? ✓ (pct≥1.0, n_norm_only, earliest_norm_only, tier) (001_layers_baseline/run-latest/output-Yi-34B.json:1096–1105)
- Tuned‑lens attribution done? ✓ (ΔKL_tuned, ΔKL_temp, ΔKL_rot at 25/50/75%) (001_layers_baseline/run-latest/output-Yi-34B.json:2168–2185)

**Limitations & Data Quirks**
- High raw‑vs‑norm discrepancies (max KL≈90.47 bits; tier=high artefact risk) and multiple norm‑only semantics layers imply potential lens‑induced early semantics; ranks and confirmed semantics are preferred over absolute probabilities for interpretation (001_layers_baseline/run-latest/output-Yi-34B.json:1062–1092,1096–1105,1649–1652,2189–2200).
- Rest_mass is top‑k coverage, not fidelity. It remains high near collapse (e.g., 0.98 at L 44) and drops by the final layer; use KL/rank/confirmed semantics for robustness.
- Prism is diagnostic only; mixed KL deltas and no earlier rank corroboration suggest it is not helpful for stabilization on this prompt.

**Model Fingerprint**
Yi‑34B: confirmed collapse at L 44 (tuned corroboration); KL→~0 by L 60; cosine crosses 0.4 at L 44 and 0.6 at L 51.

---
Produced by OpenAI GPT-5 

