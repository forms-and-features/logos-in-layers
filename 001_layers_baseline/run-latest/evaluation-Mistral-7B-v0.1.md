# Evaluation Report: mistralai/Mistral-7B-v0.1

**1. Overview**
- Model: `mistralai/Mistral-7B-v0.1` (7B, 32 layers). Run timestamp: Experiment started: 2025-10-04 18:04:23 (001_layers_baseline/run-latest/timestamp-20251004-1804:1).
- Probe captures layer-by-layer next-token behavior with a normalization-aware logit lens, copy detectors, KL-to-final, cosine trajectory, and sidecars (Prism, Tuned-Lens) for calibration/robustness.

**2. Method Sanity-Check**
- Prompting and lens: use_norm_lens = true and FP32 unembed are recorded (“use_norm_lens”: true; “unembed_dtype”: “torch.float32”) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:807] [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:809]. Positional handling reports token-only rotary at layer 0 (“layer0_position_info”: “token_only_rotary_model”) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:816].
- Context prompt ends with “called simply” (no trailing space) in code and JSON: context_prompt = “Give the city name only, plain text. The capital of Germany is called simply” [001_layers_baseline/run.py:254] and “context_prompt”: “… called simply” [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:4].
- Normalizer provenance: pre-norm, strategy “next_ln1” with epsilon inside sqrt and γ scaling; first layer uses blocks[0].ln1 (“strategy”: “next_ln1”; “ln_source”: “blocks[0].ln1”) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2288] [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2292]. Final mapping shows ln_final at unembed (“norm”: “ln_final”) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2896].
- Per-layer normalizer effect: resid_norm_ratio and delta_resid_cos present; early layers are large (e.g., layer 0 resid_norm_ratio ≈115.17, delta_resid_cos ≈0.308) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2293] [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2890]. A “normalization_spike” flag is set [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:823].
- Unembedding bias: present = false; l2_norm = 0.0; all cosines are bias-free [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:826].
- Environment & determinism: torch 2.8.0+cu128; device “cpu”; dtype_compute “torch.float32”; deterministic_algorithms = true; seed = 316 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3324].
- Numeric health: any_nan = false; any_inf = false; layers_flagged = [] [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2756].
- Copy mask: ignored_token_ids list present (diagnostic mask for punctuation/markup etc.) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2249].
- Copy-detector config and flags: copy_thresh = 0.95, copy_window_k = 1, match_level = “id_subsequence”; soft-copy threshold = 0.5; window_ks = [1,2,3]; copy_flag_columns mirror these labels [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2964] [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3336]. Strict L_copy_strict and soft L_copy_soft[k] are null across the sweep; stability = “none” [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2976].
- Gold alignment: diagnostics.gold_alignment = “ok” [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2763].
- Control prompt + summary present (Paris): first_control_margin_pos = 2; max_control_margin ≈ 0.6539 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3368].
- Last-layer head calibration: kl_to_final_bits = 0.0; top1_agree = true; p_top1_lens = p_top1_model = 0.38216; temp_est = 1.0; kl_after_temp_bits = 0.0; warn_high_last_layer_kl = false [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2764]. Final CSV row also shows kl_to_final_bits = 0.0 and answer_rank = 1 (“Berlin”, p=0.3822) (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:34).
- Measurement guidance: prefer_ranks = true; suppress_abs_probs = true; preferred_lens_for_reporting = “tuned”; use_confirmed_semantics = true; reasons include “norm_only_semantics_window”, “high_lens_artifact_risk”, “high_lens_artifact_score”, “normalization_spike” [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3854].
- Raw-vs-Norm window: radius = 4; center_layers = [25,32]; norm_only_semantics_layers = [32]; max_kl_norm_vs_raw_bits_window = 8.5569 bits; mode = “window” [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2248]. Lens sanity (sample): lens_artifact_risk = “high”; max_kl_norm_vs_raw_bits ≈ 1.174 at a sampled layer [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3316]. Full raw-vs-norm: pct_layers_kl_ge_1.0 = 0.242; n_norm_only_semantics_layers = 1 at earliest 32; max_kl_norm_vs_raw_bits = 8.557; tier = “high” [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2274].
- Threshold sweep: summary.copy_thresholds.stability = “none”; earliest L_copy_strict at τ ∈ {0.70,0.95} are null; norm_only_flags[τ] are null [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2969].
- Copy-reflex check in early layers (0–3): no strict copy_collapse rows, and no soft k1 hits at τ_soft=0.5 in 0–3 (scan of pure-next-token CSV shows no “…,True” for these flags in layers 0–3).

Prism/Tuned sidecars
- Prism sidecar present and compatible; however metrics indicate strong regression (KL↑): p50 KL baseline ≈10.33 vs prism ≈27.87 bits [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:874]. Rank milestones are null for Prism [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:858].
- Tuned-Lens present and loaded; guidance prefers tuned. Attribution at percentiles: ΔKL_tuned ≈ {p25: 4.03, p50: 3.75, p75: 7.08}; ΔKL_temp ≈ {−0.24, −0.28, 4.23}; ΔKL_rot ≈ {4.27, 4.03, 2.85}; prefer_tuned = true [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3828].

**3. Quantitative Findings**
Table built from pure-next-token CSV, filtered to prompt_id = pos, prompt_variant = orig. Each row: L i — entropy X bits, top‑1 ‘token’.
- L 0 — entropy 14.9614 bits, top‑1 ‘dabei’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:2)
- L 1 — entropy 14.9291 bits, top‑1 ‘biologie’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:3)
- L 2 — entropy 14.8254 bits, top‑1 ‘"’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:4)
- L 3 — entropy 14.8771 bits, top‑1 ‘[…]’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:5)
- L 4 — entropy 14.8538 bits, top‑1 ‘[…]’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:6)
- L 5 — entropy 14.8265 bits, top‑1 ‘[…]’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:7)
- L 6 — entropy 14.8378 bits, top‑1 ‘[…]’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:8)
- L 7 — entropy 14.8049 bits, top‑1 ‘[…]’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:9)
- L 8 — entropy 14.8210 bits, top‑1 ‘[…]’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:10)
- L 9 — entropy 14.7755 bits, top‑1 ‘[…]’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:11)
- L 10 — entropy 14.7816 bits, top‑1 ‘[…]’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:12)
- L 11 — entropy 14.7363 bits, top‑1 ‘[…]’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:13)
- L 12 — entropy 14.6418 bits, top‑1 ‘[…]’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:14)
- L 13 — entropy 14.7261 bits, top‑1 ‘[…]’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:15)
- L 14 — entropy 14.6531 bits, top‑1 ‘[…]’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:16)
- L 15 — entropy 14.4497 bits, top‑1 ‘[…]’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:17)
- L 16 — entropy 14.5998 bits, top‑1 ‘[…]’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:18)
- L 17 — entropy 14.6278 bits, top‑1 ‘[…]’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:19)
- L 18 — entropy 14.5197 bits, top‑1 ‘[…]’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:20)
- L 19 — entropy 14.5104 bits, top‑1 ‘[…]’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:21)
- L 20 — entropy 14.4242 bits, top‑1 ‘simply’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:22)
- L 21 — entropy 14.3474 bits, top‑1 ‘simply’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:23)
- L 22 — entropy 14.3874 bits, top‑1 ‘“’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:24)
- L 23 — entropy 14.3953 bits, top‑1 ‘simply’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:25)
- L 24 — entropy 14.2124 bits, top‑1 ‘simply’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:26)
- L 25 — entropy 13.5986 bits, top‑1 ‘Berlin’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:27)
- L 26 — entropy 13.5409 bits, top‑1 ‘Berlin’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:28)
- L 27 — entropy 13.2964 bits, top‑1 ‘Berlin’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:29)
- L 28 — entropy 13.2962 bits, top‑1 ‘Berlin’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:30)
- L 29 — entropy 11.4269 bits, top‑1 ‘"""’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:31)
- L 30 — entropy 10.7970 bits, top‑1 ‘“’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:32)
- L 31 — entropy 10.9943 bits, top‑1 ‘"""’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:33)
- L 32 — entropy 3.6110 bits, top‑1 ‘Berlin’ (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:34)

Collapse layer and milestones
- Preferred lens honored per guidance (tuned) and confirmed semantics used. confirmed_semantics: L_semantic_confirmed = 25 (source = “raw”); L_semantic_norm = 25; L_semantic_tuned = 32; Δ_window = 2 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3165].
- Surface→meaning: baseline L_semantic_norm = 25; tuned summary shows L_surface_to_meaning_tuned ≈ 27 at top-level and 26/25 in per-prompt summaries; we foreground tuned per guidance (27) while noting baseline 25 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3040].
- Control margin (JSON): first_control_margin_pos = 2; max_control_margin ≈ 0.6539 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3368].
- Ablation (no‑filler): L_copy_orig = null; L_sem_orig = 25; L_copy_nf = null; L_sem_nf = 24; ΔL_copy = null; ΔL_sem = −1 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3345].

Derived summaries
- ΔH (bits) = entropy(L_copy) − entropy(L_semantic). Strict L_copy_strict is null; using earliest soft copy is also null ⇒ ΔH not defined.
- Confidence milestones (p_top1 from pure CSV): p_top1 > 0.30 at layer 32; p_top1 > 0.60 not reached. Final-layer p_top1 = 0.3822 (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:34). As per guidance, treat probabilities within-model only.
- Rank milestones (diagnostics): rank ≤ 10 at layer 22; rank ≤ 5 at layer 24; rank ≤ 1 at layer 25 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2964].
- KL milestones (diagnostics): first_kl_below_1.0 at layer 27; first_kl_below_0.5 at layer 32; final KL ≈ 0 as expected [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2964] [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2764].
- Cosine milestones: cos_to_final ≥ {0.2, 0.4, 0.6} at layers {11, 25, 26} (norm); final cos_to_final ≈ 1.0 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2233] and (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:34).
- Depth fractions: L_semantic_frac ≈ 0.781; first_rank_le_5_frac ≈ 0.781 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2240].
- Copy robustness: summary.copy_thresholds.stability = “none”; earliest L_copy_strict at τ=0.70 and τ=0.95 are null; norm_only_flags are null [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2969].

Prism Sidecar Analysis
- Presence: compatible = true [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:834].
- Early-depth stability: KL(P_layer||P_final) worsens under Prism at percentiles (p50 baseline ≈10.33 vs Prism ≈27.87 bits), indicating regression [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:888].
- Rank milestones: no Prism rank milestones reported (nulls) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:858].
- Top‑1 agreement and cosine drift: no evidence of improvements at sampled depths; no earlier stabilization.
- Copy flags: no strict/soft flips observed in early layers.
- Verdict: Regressive (KL increases substantially; no earlier rank milestones).

Tuned‑Lens (preferred) brief
- Rank earliness: tuned first_rank_le_1 = 32 vs baseline 25 (later); however attribution shows positive rotation gains; use tuned for reporting per guidance [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2708] [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3828].
- Entropy drift at mid-depth (L16): baseline entropy ≈14.60 vs teacher 3.611 ⇒ drift ≈ +10.99 bits (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:18); tuned entropy ≈9.63 vs teacher 3.611 ⇒ drift ≈ +6.01 bits (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token-tuned.csv:22).

**4. Qualitative Patterns & Anomalies**
Negative control and leakage
- Test prompt “Berlin is the capital of” top‑5: Germany (0.8966), the (0.0539), both (0.00436), a (0.00380), Europe (0.00311), Berlin also appears at 0.00284 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:10]. Semantic leakage: Berlin rank present with p ≈ 0.00284 (ID-level) under this control phrasing.

Important-word trajectory (records CSV; prompt words and Berlin)
- “Berlin” becomes top‑1 by L25 and remains top‑k thereafter; at L25 the CSV shows top‑1 ‘Berlin’, p≈0.0335 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:27]. Early layers are dominated by punctuation/other tokens in records.csv (e.g., token noise at positions across the prompt) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-records.csv:1].

Instruction brevity sensitivity (one‑word): ablation shows L_sem_nf = 24 vs L_sem_orig = 25 (ΔL_sem = −1), indicating negligible shift [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3345].

Rest‑mass sanity and precision
- Rest_mass declines with depth; at final layer it is ≈0.2298 (top‑k coverage improves) (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:34). No spikes after L_semantic noted.

Rotation vs amplification
- KL decreases with depth and reaches ≈0 at final [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2764]; answer rank improves to 1 by L25 [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2964]; cosine to final rises early (≥0.2 by L11; ≥0.4 by L25; ≥0.6 by L26) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2233]. Pattern: early direction, later calibration.

Head calibration (final layer)
- Final-head calibration is clean (warn_high_last_layer_kl = false; kl_to_final_bits = 0; temp_est = 1.0) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2764].

Lens sanity and artefact risk
- Raw‑vs‑Norm: sample risk “high” (max KL ≈1.174 bits at sampled layer) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3316]; full check shows high risk with norm‑only semantics at L32 and max KL ≈8.557 bits [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2274]. Measurement guidance accordingly prefers ranks and tuned lens [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3854].

Temperature robustness (tuned)
- Mid‑depth entropy is substantially closer to teacher under tuned (drift reduces by ≈4–5 bits at p50) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3790].

Checklist
- RMS lens? ✓ (RMSNorm architecture; next_ln1 strategy) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2288]
- LayerNorm bias removed? n.a. (RMSNorm; bias-free unembed) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:826]
- Entropy rise at unembed? ✗ (final entropy is low: 3.611 bits) (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:34)
- FP32 un-embed promoted? ✓ (“unembed_dtype”: “torch.float32”) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:809]
- Punctuation / markup anchoring? ✓ (early layers dominated by quotes/punctuation) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:31]
- Copy-reflex? ✗ (no strict or soft k1 hits in L0–L3)
- Grammatical filler anchoring? ✓ (“simply” top‑1 around L20–24) (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:22)
- Preferred lens honored in milestones? ✓ (tuned per guidance) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3854]
- Confirmed semantics reported? ✓ (L_semantic_confirmed = 25, source = raw) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3165]
- Full dual‑lens metrics cited? ✓ (pct_layers_kl_ge_1.0, n_norm_only_semantics_layers, earliest_norm_only_semantic, tier) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2274]
- Tuned‑lens attribution done? ✓ (ΔKL_tuned, ΔKL_temp, ΔKL_rot at 25/50/75%) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3828]
- normalization_provenance present? ✓ (ln_source verified at layer 0 and final) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2292] [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2896]
- per-layer normalizer effect metrics present? ✓ (resid_norm_ratio, delta_resid_cos) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2368]
- unembed bias audited? ✓ (bias-free cosine guaranteed) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:826]
- deterministic_algorithms = true? ✓ [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3324]
- numeric_health clean? ✓ (no NaN/Inf; no flagged layers) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2756]
- copy_mask present and plausible? ✓ (ignored_token_ids present) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2249]
- layer_map present for indexing audit? ✓ (per-layer mapping block lists ln sources and final ln_final) [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2700]

**5. Limitations & Data Quirks**
- High raw‑vs‑norm artefact risk (tier “high”) with a norm‑only semantics layer at L32; prefer rank milestones and confirmed semantics for onset claims [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:2274].
- Measurement guidance explicitly requests rank‑first reporting and suppressing absolute probability comparisons across models [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:3854].
- Rest_mass is top‑k coverage only (not fidelity). It falls to ≈0.2298 at final (001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv:34), but should not be used as a lens‑fidelity metric.
- Prism sidecar is diagnostic-only here and regresses KL; do not substitute its metrics for the model head [001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json:888].

**6. Model Fingerprint**
“Mistral‑7B‑v0.1: collapse at L 25 (confirmed raw); tuned lens reports later rank‑1 at L 32; final entropy 3.61 bits; ‘Berlin’ stabilizes top‑1 in late layers.”

---
Produced by OpenAI GPT-5
