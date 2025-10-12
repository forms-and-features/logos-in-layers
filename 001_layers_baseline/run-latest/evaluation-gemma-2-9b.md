# Evaluation Report: google/gemma-2-9b

**Overview**
This report evaluates gemma-2-9b on a single-fact probe run on 2025-10-11, measuring copy-reflex versus semantic emergence with layer-by-layer ranks, KL-to-final-head, cosine geometry, and entropy, plus lens diagnostics (raw vs norm), Tuned-Lens audit, and Prism.
The script `001_layers_baseline/run.py` runs a deterministic pass with a LayerNorm-corrected logit lens and emits unified metrics and sidecars across lenses and audits.

**Method Sanity‑Check**
- Prompt & indexing: context ends with “called simply” (no trailing space): "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply"  [001_layers_baseline/run-latest/output-gemma-2-9b.json:4]. Positive baseline rows exist: "Germany→Berlin,0,pos,orig,0 …"  [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2].
- Normalizer provenance: "strategy": "next_ln1"  [001_layers_baseline/run-latest/output-gemma-2-9b.json:5916]; first/last sources "ln_source": "blocks[0].ln1" and "ln_source": "ln_final"  [001_layers_baseline/run-latest/output-gemma-2-9b.json:5922, 6314].
- Per‑layer normalizer effect: early layers show no extreme pre‑semantic spikes but a final spike is flagged; examples: L0 "resid_norm_ratio": 0.757…, "delta_resid_cos": 0.929…  [001_layers_baseline/run-latest/output-gemma-2-9b.json:5924–5926]; final lens row "resid_norm_ratio": 0.2199…, "delta_resid_cos": 0.745…  [001_layers_baseline/run-latest/output-gemma-2-9b.json:6316–6318]. Flag recorded: "normalization_spike": true  [001_layers_baseline/run-latest/output-gemma-2-9b.json:836–842].
- Unembed bias: "unembed_bias": { "present": false, "l2_norm": 0.0 }  [001_layers_baseline/run-latest/output-gemma-2-9b.json:832–838]. Cosines are bias‑free by construction.
- Environment & determinism: "device": "cpu", "deterministic_algorithms": true, "seed": 316  [001_layers_baseline/run-latest/output-gemma-2-9b.json:8665–8670]. Reproducibility OK.
- Numeric health: "any_nan": false, "any_inf": false, "layers_flagged": []  [001_layers_baseline/run-latest/output-gemma-2-9b.json:6570–6576].
- Copy mask: ignored_token_ids size = 4668; e.g., sample [108, 109, 110, 111, 112] … [255994, 255995, 255996, 255997, 255998]  [001_layers_baseline/run-latest/output-gemma-2-9b.json:949–955, 1456–1460]. Plausible for tokenizer punctuation/whitespace.
- Gold alignment: { "ok": true, "variant": "with_space", "pieces": ["▁Berlin"] }  [001_layers_baseline/run-latest/output-gemma-2-9b.json:6579–6597].
- Repeatability: deterministic environment → "repeatability": { "status": "skipped", "reason": "deterministic_env" }  [001_layers_baseline/run-latest/output-gemma-2-9b.json:6576–6579]; evaluation_pack repeats with null metrics and flag "skipped"  [001_layers_baseline/run-latest/output-gemma-2-9b.json:10441–10448].
- Norm trajectory: "shape": "spike", "slope": 0.0517, "r2": 0.987, "n_spikes": 1  [001_layers_baseline/run-latest/output-gemma-2-9b.json:10451–10457].
- Measurement guidance: { "prefer_ranks": true, "suppress_abs_probs": true, "preferred_lens_for_reporting": "norm", "use_confirmed_semantics": true }  [001_layers_baseline/run-latest/output-gemma-2-9b.json:10389–10403].
- Semantic margin: { "delta_abs": 0.002, "p_uniform": 3.9e‑06, "L_semantic_margin_ok_norm": 42, "L_semantic_confirmed_margin_ok_norm": 42 }  [001_layers_baseline/run-latest/output-gemma-2-9b.json:10292–10299]. Margin gate passes at L_semantic_norm.
- Micro‑suite: evaluation_pack aggregates present with n=5, n_missing=0; medians L_semantic_confirmed_median=42, delta_hat_median=1.0  [001_layers_baseline/run-latest/output-gemma-2-9b.json:10572–10590].

**Quantitative Findings (Layer‑by‑Layer)**
- L 0 — entropy 1.67e‑05 bits, top‑1 "simply"; copy flags set (strict@{0.70,0.95}); answer_rank=1468  [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2].
- L 18 — entropy 3.34 bits, top‑1 "simply"; answer_rank=35160; KL_to_final≈18.15 bits (norm lens).
- L 28 — entropy 1.23 bits, top‑1 '"'; answer_rank=5954; KL_to_final≈1.58 bits (norm lens).
- L 34 — entropy 0.090 bits, top‑1 '"'; answer_rank=649; KL_to_final≈1.84 bits (norm lens)  [001_layers_baseline/run-latest/output-gemma-2-9b.json:5647–5651].
- L 42 — entropy 0.370 bits, top‑1 "Berlin"; answer_rank=1; KL_to_final=1.013 bits; cos_to_final≈0.9993  [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:49].
- Control margin: first_control_margin_pos=18; max_control_margin=0.8677  [001_layers_baseline/run-latest/output-gemma-2-9b.json:8717–8719].
- Micro‑suite (medians): L_semantic_confirmed=42, Δ̂=1.0; e.g., "Germany→Berlin" cites row 42 for confirmed semantics  [001_layers_baseline/run-latest/output-gemma-2-9b.json:10558–10590].
- Entropy drift (gaps vs teacher): p25=−2.89 bits; p50=−2.80; p75=−1.63  [001_layers_baseline/run-latest/output-gemma-2-9b.json:10460–10466].
- Confidence margins and normalizer snapshots at L 42: answer_logit_gap=2.588 (logit units); resid_norm_ratio=0.2199; delta_resid_cos=0.7451; cos_to_answer=0.1642; cos_to_prompt_max=0.1056  [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:49].

**Qualitative Findings**

**4.1. Copy vs semantics (Δ‑gap)**
Early layers exhibit a strong copy‑reflex: at L0 the model echoes the prompt continuation token (“simply”) with copy_collapse=True and strict copy at τ∈{0.70,0.95}  [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2]. Using evaluation-pack depth fractions, Δ̂ = 1.0 (semantic at last layer, strict copy at L0)  [001_layers_baseline/run-latest/output-gemma-2-9b.json:10420–10428]. Copy threshold stability tag is “mixed”; earliest strict copy is L0 for τ=0.70 and τ=0.95; norm_only_flags false  [001_layers_baseline/run-latest/output-gemma-2-9b.json:5706–5730].

**4.2. Lens sanity: Raw‑vs‑Norm**
Artifact risk is high: lens_artifact_score_v2=0.5906 (tier=high) with js_divergence_p50=0.0063 and l1_prob_diff_p50=0.0292; first_js_le_0.1=0 and first_l1_le_0.5=0  [001_layers_baseline/run-latest/output-gemma-2-9b.json:10424–10440]. Top‑K overlap is moderate (jaccard_raw_norm_p50=0.639; first_jaccard_raw_norm_ge_0.5=3). 30.2% of layers have KL(raw∥norm)≥1.0; there is 1 norm‑only semantics layer at 42 (earliest=42)  [001_layers_baseline/run-latest/output-gemma-2-9b.json:5768–5820]. Caution: favor rank milestones and confirmed semantics; early semantics could be lens‑induced under high‑risk tiers.

**4.3. Tuned‑Lens analysis**
Preference: tuned_is_calibration_only=false and preferred_semantics_lens_hint="norm" → report semantics under norm; use tuned for calibration context  [001_layers_baseline/run-latest/output-gemma-2-9b.json:10490–10494, 10389–10403]. Attribution: ΔKL rotation vs temperature at percentiles shows small net effects (delta_kl_rot_p50≈0, delta_kl_temp_p50=−0.013; interaction_p50≈0.0026)  [001_layers_baseline/run-latest/output-gemma-2-9b.json:10460–10466]. Rank earliness unchanged (first_rank_le_{10,5,1}=42 under both)  [001_layers_baseline/run-latest/output-gemma-2-9b.json:5649–5651, 8733–8737]. Positional generalization: pos_ood_ge_0.96=0.0; pos_in_dist_le_0.92≈0.00028; gap≈−0.00028  [001_layers_baseline/run-latest/output-gemma-2-9b.json:10466–10484]. Head mismatch improves with model-calibrated τ*: KL_tuned_final 1.0823 → 0.4059 after τ*; τ_star_modelcal≈2.85  [001_layers_baseline/run-latest/output-gemma-2-9b.json:10484–10490]. Last‑layer agreement holds: "top1_agree": true with warn_high_last_layer_kl=true  [001_layers_baseline/run-latest/output-gemma-2-9b.json:6591–6620].

**4.4. KL, ranks, cosine, entropy milestones**
KL thresholds: first_kl_below_{1.0,0.5} are null under the preferred lens (norm), and final KL to the model head remains >0, consistent with calibrated‑head mismatch  [001_layers_baseline/run-latest/output-gemma-2-9b.json:5647–5649, 6591–6620]. Ranks: first_rank_le_{10,5,1}=42 (norm; tuned identical)  [001_layers_baseline/run-latest/output-gemma-2-9b.json:5649–5651, 8733–8737]. Cosine milestones (norm lens): ge_0.2 at L1; ge_{0.4,0.6} at L42  [001_layers_baseline/run-latest/output-gemma-2-9b.json:5744–5750]. Entropy decreases substantially vs teacher across the depth (median gap −2.80 bits), consistent with increasing certainty as ranks improve late; pair this with high final‑head KL to avoid over‑interpreting absolute probabilities  [001_layers_baseline/run-latest/output-gemma-2-9b.json:10458–10466]. Margin gate passes at L_semantic_norm and at confirmed semantics  [001_layers_baseline/run-latest/output-gemma-2-9b.json:10292–10299].

**4.5. Prism (shared‑decoder diagnostic)**
Prism artifacts are present and compatible; sampled layers [embed, 9, 20, 30]  [001_layers_baseline/run-latest/output-gemma-2-9b.json:842–858]. KL deltas (baseline−prism) are negative at p50 (≈−10.33 bits), indicating higher KL under Prism; rank milestones show no improvement (null deltas)  [001_layers_baseline/run-latest/output-gemma-2-9b.json:858–898]. Verdict: Regressive.

**4.6. Ablation & stress tests**
No‑filler ablation leaves milestones unchanged: { L_copy_orig=0, L_sem_orig=42, L_copy_nf=0, L_sem_nf=42, delta_L_copy=0, delta_L_sem=0 }  [001_layers_baseline/run-latest/output-gemma-2-9b.json:8683–8690]. Control summary present (first_control_margin_pos=18); for the test prompt “Berlin is the capital of”, the target country appears in top‑k and “Berlin” also appears among candidates  [001_layers_baseline/run-latest/output-gemma-2-9b.json:1–24, 8717–8719].

**4.7. Checklist (✓/✗/n.a.)**
- RMS lens ✓
- LayerNorm bias removed ✓
- FP32 unembed promoted ✓ (per env/provenance and defaults)  [001_layers_baseline/run-latest/output-gemma-2-9b.json:808–812]
- Punctuation / markup anchoring noted ✓ (top‑1 quotes in mid‑layers)
- Copy‑reflex ✓ (L0 strict/soft)  [001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv:2]
- Preferred lens honored ✓ (norm; confirmed semantics used)  [001_layers_baseline/run-latest/output-gemma-2-9b.json:10389–10403]
- Confirmed semantics reported ✓ (L=42, source=tuned)  [001_layers_baseline/run-latest/output-gemma-2-9b.json:10416–10423]
- Dual‑lens artefact metrics (v2, JS/Jaccard/L1) cited ✓  [001_layers_baseline/run-latest/output-gemma-2-9b.json:10424–10440]
- Tuned‑lens audit (rotation/temp/positional/head) ✓  [001_layers_baseline/run-latest/output-gemma-2-9b.json:10460–10490]
- normalization_provenance present (ln_source @ L0/final) ✓  [001_layers_baseline/run-latest/output-gemma-2-9b.json:5916–5926, 6312–6318]
- per‑layer normalizer effect present (resid_norm_ratio, delta_resid_cos) ✓  [001_layers_baseline/run-latest/output-gemma-2-9b.json:5922–5926, 6316–6318]
- deterministic_algorithms true ✓  [001_layers_baseline/run-latest/output-gemma-2-9b.json:8665–8670]
- numeric_health clean ✓  [001_layers_baseline/run-latest/output-gemma-2-9b.json:6570–6576]
- copy_mask plausible ✓  [001_layers_baseline/run-latest/output-gemma-2-9b.json:949–955]
- milestones.csv or evaluation_pack.citations used ✓  [001_layers_baseline/run-latest/output-gemma-2-9b-milestones.csv:3–4, 001_layers_baseline/run-latest/output-gemma-2-9b.json:10494–10506]

---
Produced by OpenAI GPT-5

*Run executed on: 2025-10-11 21:50:12*
