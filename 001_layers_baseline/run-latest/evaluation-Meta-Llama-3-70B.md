# Evaluation Report: meta-llama/Meta-Llama-3-70B

*Run executed on: 2025-10-04 18:04:23*
**1. Overview**

Meta‑Llama‑3‑70B (80 layers) evaluated on 2025‑10‑04. The probe measures layer‑by‑layer next‑token behavior under a norm‑lens, tracking copy vs. semantics onset, calibration to the final head (KL), cosine trajectory to final, and surface‑mass/coverage.

**2. Method Sanity‑Check**

The prompt is correctly formed and ends with “called simply” with no trailing space: “Give the city name only, plain text. The capital of Germany is called simply” [JSON L4]. The run uses the norm lens with FP32 unembedding: "use_norm_lens": true, "use_fp32_unembed": true, "unembed_dtype": "torch.float32" [JSON L807–L809]. Normalizer provenance matches a pre‑norm RMS pipeline with next‑block ln1 and epsilon inside sqrt at early layers (e.g., layer 0: "ln_source": "blocks[0].ln1", "eps_inside_sqrt": true) [JSON L7184–L7193], and the final head uses ln_final [JSON L8320–L8323]. Unembedding bias is absent (present=false; l2_norm=0.0) [JSON L826–L830].

Copy detection configuration and outputs are present: "copy_thresh": 0.95, "copy_window_k": 1, "copy_match_level": "id_subsequence" [JSON L7030–L7033]; strict sweep tau_list = [0.7, 0.8, 0.9, 0.95] with all L_copy_strict = null and stability="none" [JSON L7086–L7111]. The CSV includes the expected flags: copy_strict@{0.95,0.7,0.8,0.9}, copy_soft_k{1,2,3}@0.5 [JSON L8565–L8572]. Gold alignment is ok [JSON L8332]. Summary indices: first_kl_below_0.5=80, first_kl_below_1.0=80, first_rank_le_1=40, first_rank_le_5=38, first_rank_le_10=38 [JSON L7033–L7037].

Last‑layer head calibration is clean: kl_to_final_bits=0.000729, top1_agree=true, p_top1_lens=0.4783 vs p_top1_model=0.4690, temp_est=1.0, warn_high_last_layer_kl=false [JSON L8334–L8351]. The CSV’s final layer matches: layer 80 has kl_to_final_bits=0.0007293, answer_rank=1, p_top1=0.4783 (“Berlin”) [row 82 in CSV].

Raw‑vs‑Norm window: radius=4; center_layers=[38,40,80]; norm_only_semantics_layers=[79,80]; max_kl_norm_vs_raw_bits_window=1.2437 [JSON L7140–L7169]. Full raw‑vs‑norm: pct_layers_kl_ge_1.0=0.0123, pct_layers_kl_ge_0.5=0.0247, n_norm_only_semantics_layers=2, earliest_norm_only_semantic=79, max_kl_norm_vs_raw_bits=1.2437, tier="medium" [JSON L7171–L7181]. Measurement guidance requests rank‑first reporting and suppresses absolute probabilities: prefer_ranks=true, suppress_abs_probs=true, preferred_lens_for_reporting="norm", use_confirmed_semantics=true, reasons=["norm_only_semantics_window","normalization_spike"] [JSON L8606–L8616].

Per‑layer normalizer effect metrics are present and stable in early layers (e.g., layer 0: resid_norm_ratio=3.889, delta_resid_cos=0.170 [JSON L7192–L7193]; layer 3: resid_norm_ratio=7.862, delta_resid_cos=0.823 [JSON L7216–L7217]). Numeric health is clean: any_nan=false, any_inf=false, layers_flagged=[] [JSON L8325–L8331]. Environment/determinism: torch 2.8.0 (cpu), dtype_compute=torch.bfloat16, deterministic_algorithms=true, seed=316 [JSON L8553–L8562]. Copy mask is large and plausible (punctuation‑heavy sample), size=6022 [JSON L7006–L7024]. Control summary: first_control_margin_pos=0, max_control_margin=0.5168 [JSON L8597–L8600].

Copy‑reflex check (layers 0–3): no strict copy_collapse and no soft k1@0.5 hits in early layers; all False in CSV rows for L0–L3 [rows 2–5 in CSV].

Prism sidecar is present and compatible [JSON L834–L840]. Rank milestones are not reported by Prism (null), and KL is higher than baseline at sampled depths (e.g., p25 delta ≈ −0.89 bits) [JSON L865–L870, L353–L361]. Tuned‑lens is missing [JSON L8365–L8369, L8601–L8605].

**3. Quantitative Findings**

| Layer | Entropy (bits) | Top-1 token |
|---:|---:|:---|
| 0 | 16.968050 |  winding |
| 1 | 16.960091 | cepts |
| 2 | 16.963358 | улю |
| 3 | 16.962561 | zier |
| 4 | 16.958609 | alls |
| 5 | 16.957176 | alls |
| 6 | 16.956129 | alls |
| 7 | 16.953325 | NodeId |
| 8 | 16.959366 | inds |
| 9 | 16.959677 | NodeId |
| 10 | 16.952387 | inds |
| 11 | 16.956022 | inds |
| 12 | 16.956364 | lia |
| 13 | 16.955206 | eds |
| 14 | 16.950441 | idders |
| 15 | 16.953262 |  Kok |
| 16 | 16.952202 | /plain |
| 17 | 16.948030 |  nut |
| 18 | 16.944304 |  nut |
| 19 | 16.947536 |  nut |
| 20 | 16.946392 |  nut |
| 21 | 16.937977 |  burge |
| 22 | 16.937849 |  simply |
| 23 | 16.935751 |  bur |
| 24 | 16.949749 |  bur |
| 25 | 16.937452 | � |
| 26 | 16.938286 | � |
| 27 | 16.937248 | za |
| 28 | 16.932850 | /plain |
| 29 | 16.932850 |  plain |
| 30 | 16.938557 | zed |
| 31 | 16.925108 |  simply |
| 32 | 16.940559 |  simply |
| 33 | 16.927097 |  plain |
| 34 | 16.932306 |  simply |
| 35 | 16.929186 |  simply |
| 36 | 16.939707 |  simply |
| 37 | 16.934622 |  simply |
| 38 | 16.934185 |  simply |
| 39 | 16.934874 |  simply |
| **40** | 16.937426 |  Berlin |
| 41 | 16.936241 |  " |
| 42 | 16.944441 |  " |
| 43 | 16.941313 |  Berlin |
| 44 | 16.925947 |  Berlin |
| 45 | 16.940191 |  " |
| 46 | 16.955173 |  " |
| 47 | 16.939295 |  " |
| 48 | 16.938839 |  " |
| 49 | 16.936884 |  " |
| 50 | 16.943811 |  " |
| 51 | 16.940096 |  " |
| 52 | 16.921988 |  Berlin |
| 53 | 16.933023 |  Berlin |
| 54 | 16.942423 |  Berlin |
| 55 | 16.941885 |  Berlin |
| 56 | 16.920958 |  Berlin |
| 57 | 16.933500 |  Berlin |
| 58 | 16.941111 |  Berlin |
| 59 | 16.944092 |  Berlin |
| 60 | 16.922909 |  Berlin |
| 61 | 16.939604 |  Berlin |
| 62 | 16.950912 |  Berlin |
| 63 | 16.945812 |  Berlin |
| 64 | 16.926292 |  Berlin |
| 65 | 16.933395 |  " |
| 66 | 16.940704 |  Berlin |
| 67 | 16.930429 |  Berlin |
| 68 | 16.924042 |  Berlin |
| 69 | 16.931520 |  Berlin |
| 70 | 16.925699 |  Berlin |
| 71 | 16.922638 |  Berlin |
| 72 | 16.922110 |  Berlin |
| 73 | 16.918125 |  " |
| 74 | 16.914299 |  Berlin |
| 75 | 16.912664 |  Berlin |
| 76 | 16.919010 |  Berlin |
| 77 | 16.909897 |  Berlin |
| 78 | 16.918554 |  Berlin |
| 79 | 16.942253 |  Berlin |
| 80 | 2.589049 |  Berlin |

Control margin (JSON): first_control_margin_pos=0; max_control_margin=0.5168 [JSON L8597–L8600].

Ablation (no‑filler): L_copy_orig=null, L_sem_orig=40; L_copy_nf=null, L_sem_nf=42; ΔL_copy=null, ΔL_sem=2 [JSON L8574–L8580]. Interpretation: semantics are delayed by ~2 layers (≈2.5% of depth) when filler is removed.

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (L_copy_strict=null). Soft ΔHk (k∈{1,2,3}) = n.a. (all L_copy_soft[k]=null) [JSON L7044–L7053].

Confidence milestones (CSV): p_top1>0.30 at layer 80; p_top1>0.60 not reached; final‑layer p_top1=0.4783 (“Berlin”) [row 82 in CSV].

Rank milestones (JSON): rank≤10 at L38, rank≤5 at L38, rank=1 at L40 [JSON L7035–L7037].

KL milestones (JSON/CSV): first_kl_below_1.0 at L80; first_kl_below_0.5 at L80 [JSON L7033–L7034]. KL decreases sharply only at the end and is ≈0 at the final layer (0.000729) [JSON L8334; row 82 in CSV].

Cosine milestones (JSON/CSV): cos_to_final ≥0.2, ≥0.4, ≥0.6 all first at L80 [JSON L7124–L7129]; final cos_to_final=0.999989 [row 82 in CSV]. Depth fractions: L_semantic_frac=0.5 [JSON L7131–L7133].

Surface→meaning (norm lens): L_surface_to_meaning_norm=80 with answer_mass_at_L=0.4783, echo_mass_at_L=0.00477 [JSON L7113–L7116]. Geometry: L_geom_norm=54 [JSON L7116]. Coverage: L_topk_decay_norm=0; topk_prompt_mass_at_L_norm=0.0; τ=0.33 [JSON L7119–L7121]. Norm‑temp snapshots: kl_to_final_bits_norm_temp@{25,50,75}% = {10.686, 10.625, 4.350} [JSON L8353–L8363].

Copy robustness (threshold sweep): stability="none"; earliest strict L_copy at τ=0.70 and τ=0.95 both null; norm_only_flags all null [JSON L7086–L7111].

Prism sidecar: present (compatible=true). KL is higher vs baseline at early/mid depths (e.g., p25 baseline=10.45, prism=11.34, Δ=−0.89 bits) [JSON L865–L870]. Prism rank milestones are null [JSON L854–L858]. Sampled per‑layer KLs from the Prism CSV confirm higher KL at 0/20/40/60 and very high at 80 (≈26.88 bits). Verdict: Regressive.

**4. Qualitative Patterns & Anomalies**

Negative control shows clean behavior: for “Berlin is the capital of”, the model predicts country tokens, e.g., “ Germany” (p≈0.85) and “ the” (p≈0.079) [JSON L14–L19]. No “Berlin” appears in the top‑5; no leakage flagged.

Important‑word trajectory at NEXT: “Berlin” first enters the top‑5 by L38 and becomes top‑1 at L40. For example, L38 includes “ Berlin” as top‑3 (p≈2.50e‑05) [row 40 in CSV], L39 keeps “ Berlin” as top‑2 [row 41], and L40 promotes “ Berlin” to top‑1 (answer_rank=1) [row 42]. In records spanning later layers, “Berlin” and its morphological variants cluster in the top slots (e.g., L75 at pos=16 with " Berlin", " Ber", " BER" among top tokens) [records CSV L1609]. This pattern is consistent with token‑family consolidation before final calibration.

Instruction sensitivity: test prompts vary punctuation/wording; they report single‑step distributions only (no per‑layer traces), so collapse‑layer shifts cannot be directly measured. Several variants keep “ Berlin” as dominant completion (e.g., “Germany’s capital city is called”, p≈0.77 [JSON L49–L56]).

Rest‑mass sanity: after L_semantic, rest_mass remains high for a while (max≈0.9999 at L46) before falling to 0.107 at the final layer [row 82 in CSV]. This reflects top‑k coverage rather than lens fidelity.

Rotation vs amplification: direction forms early while calibration is late. Around L38–L40, answer ranks improve (3→1) and cos_to_final rises (≈0.084→0.097) while KL to final remains ≈10.4 bits [rows 40–42]. KL collapses only near the top (L80), consistent with “early direction, late calibration”.

Head calibration: last‑layer KL is ≈0 with top‑1 agreement and temp_est=1.0; no family‑level high‑KL warning [JSON L8334–L8351].

Lens sanity: sampled raw‑vs‑norm risk is low [JSON L8545–L8549], but the full scan marks tier="medium" with 2 norm‑only semantic layers (earliest=79) [JSON L7171–L7181]. Measurement guidance explicitly requests rank‑first, probability‑suppressed reporting [JSON L8606–L8616].

Temperature robustness: n.a. (no explicit temperature sweep in this run). Norm‑temperature diagnostics are reported instead (see §3 snapshots) [JSON L8353–L8363].

Checklist
✓ RMS lens (RMSNorm; next_ln1) [JSON L7184–L7193]
✓ LayerNorm bias removed (bias‑free unembed) [JSON L826–L830]
✓ Entropy rise at unembed (final entropy drops to 2.589 bits) [row 82]
✓ FP32 un‑embed promoted [JSON L808–L809]
✗ Punctuation / markup anchoring (early top‑1 not dominated by {is,the,a,of}) [rows 2–5]
✗ Copy‑reflex (no strict or soft‑k1 hits at L0–L3) [rows 2–5]
✓ Preferred lens honored (norm; rank‑first) [JSON L8606–L8616]
✓ Confirmed semantics reported (L_semantic_confirmed=40; source=raw) [JSON L8371–L8377]
✓ Full dual‑lens metrics cited (pct_layers_kl_ge_1.0, n_norm_only_semantics_layers, earliest_norm_only_semantic, tier) [JSON L7171–L7181]
n.a. Tuned‑lens attribution (status=missing) [JSON L8365–L8369]
✓ normalization_provenance present (ln_source verified at layer 0 and final) [JSON L7188–L7193, L8320–L8323]
✓ per‑layer normalizer metrics present (resid_norm_ratio, delta_resid_cos) [JSON L7192–L7193]
✓ unembed bias audited (bias‑free cosine guaranteed) [JSON L826–L830]
✓ deterministic_algorithms=true [JSON L8558]
✓ numeric_health clean (no NaN/Inf; no flagged layers) [JSON L8325–L8331]
✓ copy_mask present and plausible for tokenizer (size=6022) [JSON L7006–L7024]
n.a. layer_map (not required for indexing)

**5. Limitations & Data Quirks**

Measurement guidance is rank‑first and probability‑suppressed; absolute probabilities should be compared only within‑model [JSON L8606–L8616]. KL is lens‑sensitive; although final KL≈0 indicates good head alignment here, raw‑vs‑norm full scan reports medium artefact risk with norm‑only semantics appearing at 79–80 [JSON L7171–L7181], so early “semantics” should be treated cautiously and primarily via rank milestones. Rest_mass measures top‑k coverage only; its high values after L_semantic (e.g., ≈0.9999 at L46) do not imply fidelity issues, and the final rest_mass (≈0.107) remains <0.3 [row 82]. Prism is diagnostic only and regressive in this run (higher KL), so its ranks are not used to override baseline.

**6. Model Fingerprint**

Meta‑Llama‑3‑70B: collapse at L 40 (confirmed raw); final entropy 2.59 bits; “Berlin” stable top‑1 late‑stack and KL→0.

---
Produced by OpenAI GPT-5 
