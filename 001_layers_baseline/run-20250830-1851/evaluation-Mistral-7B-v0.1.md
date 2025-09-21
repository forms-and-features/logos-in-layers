**Overview**
- This run analyzes mistralai/Mistral-7B-v0.1 (32 layers) on a one-word capital probe with a norm lens and per-layer logit-lens metrics. The run targets next-token semantics (“Berlin”) and collapse dynamics under stylistic filler.
- Outputs include JSON diagnostics, per-layer pure next-token CSV, and records CSV; semantic collapse occurs mid–late stack with clean final-head consistency.

**Method Sanity-Check**
The JSON confirms the intended norm lens, RMSNorm architecture, and rotary/token-only position info. The context prompt ends with “called simply” (no trailing space):
> "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply"  [L817–L818 in JSON]
> "layer0_position_info": "token_only_rotary_model"  [L816 in JSON]
Implementation flags and copy-rule settings are present and aligned with the spec:
> "use_norm_lens": true, "unembed_dtype": "torch.float32"  [L807–L809 in JSON]
> "copy_thresh": 0.95, "copy_window_k": 1, "copy_match_level": "id_subsequence"  [L823–L825 in JSON]
Diagnostics contain the expected indices and units (bits):
> "L_copy": null, "L_copy_H": null, "L_semantic": 25, "first_kl_below_0.5": 32, "first_kl_below_1.0": 32, "first_rank_le_1": 25, "first_rank_le_5": 25, "first_rank_le_10": 23, "gold_alignment": "ok"  [L819–L831 in JSON]
Last-layer head calibration is consistent (KL≈0, identical top-1 between lens and model, temp≈1):
> "kl_to_final_bits": 0.0, "top1_agree": true, "p_top1_lens": 0.38216, "p_top1_model": 0.38216, "temp_est": 1.0, "warn_high_last_layer_kl": false  [L833–L851 in JSON]
CSV final row also shows KL≈0: 
> "...,3.611043930053711,Berlin,0.38216,...,kl_to_final_bits,0.0,..."  [row 34 in pure CSV]

Gold-token alignment is ID-based and OK:
> "gold_answer": { "string": "Berlin", "pieces": ["▁Berlin"], "first_id": 8430, ... }, "variant": "with_space"  [L1037–L1047 in JSON]
Negative control scaffolding is present:
> "control_prompt": "... The capital of France is called simply", "gold_alignment": "ok"  [L1018–L1031 in JSON]
> "control_summary": { "first_control_margin_pos": 2, "max_control_margin": 0.6539 }  [L1033–L1035 in JSON]
Ablation summary exists and both prompt variants appear in the CSV:
> "ablation_summary": { "L_copy_orig": null, "L_sem_orig": 25, "L_copy_nf": null, "L_sem_nf": 24, "delta_L_sem": -1 }  [L1010–L1016 in JSON]
> First no‑filler rows present (e.g., "pos,no_filler,0,...")  [row 35 in pure CSV]

Copy-collapse early-layer check (τ=0.95, δ=0.10; k=1, ID-level contiguous subsequence): no rows with copy_collapse=True in layers 0–3 (or any layer), matching L_copy=null.
> No matches for "copy_collapse,True" in pure CSV  [search]

Lens sanity (dual lens):
> "raw_lens_check": { "mode": "sample", summary: { "max_kl_norm_vs_raw_bits": 1.1739, "lens_artifact_risk": "high", "first_norm_only_semantic_layer": null } }  [L948–L1008 in JSON]
Note: “high” artifact risk in sampled checks; no norm‑only semantic layer flagged. Prefer within‑model trends and rank milestones when interpreting early signals.

Summary indices (from diagnostics): first_kl_below_0.5 = 32; first_kl_below_1.0 = 32; first_rank_le_10 = 23; first_rank_le_5 = 25; first_rank_le_1 = 25. KL/entropy units are bits (column name "kl_to_final_bits" in CSV; entropy is reported as bits in CSV).

**Quantitative Findings**
Per-layer pure next-token (prompt_id=pos, prompt_variant=orig):
- L 0 – entropy 14.961 bits, top-1 'dabei'
- L 1 – entropy 14.929 bits, top-1 'biologie'
- L 2 – entropy 14.825 bits, top-1 ',\r'
- L 3 – entropy 14.877 bits, top-1 '[…]'
- L 4 – entropy 14.854 bits, top-1 '[…]'
- L 5 – entropy 14.827 bits, top-1 '[…]'
- L 6 – entropy 14.838 bits, top-1 '[…]'
- L 7 – entropy 14.805 bits, top-1 '[…]'
- L 8 – entropy 14.821 bits, top-1 '[…]'
- L 9 – entropy 14.776 bits, top-1 '[…]'
- L 10 – entropy 14.782 bits, top-1 '[…]'
- L 11 – entropy 14.736 bits, top-1 '[…]'
- L 12 – entropy 14.642 bits, top-1 '[…]'
- L 13 – entropy 14.726 bits, top-1 '[…]'
- L 14 – entropy 14.653 bits, top-1 '[…]'
- L 15 – entropy 14.450 bits, top-1 '[…]'
- L 16 – entropy 14.600 bits, top-1 '[…]'
- L 17 – entropy 14.628 bits, top-1 '[…]'
- L 18 – entropy 14.520 bits, top-1 '[…]'
- L 19 – entropy 14.510 bits, top-1 '[…]'
- L 20 – entropy 14.424 bits, top-1 'simply'
- L 21 – entropy 14.347 bits, top-1 'simply'
- L 22 – entropy 14.387 bits, top-1 '“'
- L 23 – entropy 14.395 bits, top-1 'simply'
- L 24 – entropy 14.212 bits, top-1 'simply'
- L 25 – entropy 13.599 bits, top-1 '**Berlin**'  [is_answer=True]
- L 26 – entropy 13.541 bits, top-1 'Berlin'  [is_answer=True]
- L 27 – entropy 13.296 bits, top-1 'Berlin'  [is_answer=True]
- L 28 – entropy 13.296 bits, top-1 'Berlin'  [is_answer=True]
- L 29 – entropy 11.427 bits, top-1 '"'
- L 30 – entropy 10.797 bits, top-1 '“'
- L 31 – entropy 10.994 bits, top-1 '"'
- L 32 – entropy 3.611 bits, top-1 'Berlin'  [is_answer=True]

Ablation (no‑filler): L_copy_orig = null, L_sem_orig = 25; L_copy_nf = null, L_sem_nf = 24; ΔL_copy = null; ΔL_sem = −1. A one‑layer advance with no copy signal suggests the stylistic filler "simply" slightly delays semantics rather than providing a semantic cue.

ΔH (bits) = n/a (no copy collapse flagged) − entropy(L_semantic=25) = n/a.

Confidence milestones (pure CSV):  
p_top1 > 0.30 at layer 32; p_top1 > 0.60: n/a; final-layer p_top1 = 0.3822 [row 34 in pure CSV].

Rank milestones (diagnostics):  
rank ≤ 10 at layer 23; rank ≤ 5 at layer 25; rank ≤ 1 at layer 25  [L826–L830 in JSON].

KL milestones (diagnostics):  
first_kl_below_1.0 at layer 32; first_kl_below_0.5 at layer 32  [L826–L827 in JSON]. KL decreases to ≈0 at the final layer; last-layer consistency confirms KL=0 and temp≈1 [L833–L840 in JSON].

Cosine milestones (pure CSV):  
first cos_to_final ≥ 0.2 at layer 11; ≥ 0.4 at layer 25; ≥ 0.6 at layer 26; final cos_to_final = ~1.00 [row 34 in pure CSV].

**Qualitative Patterns & Anomalies**
Negative control shows clean country recall and low Berlin leakage: for “Berlin is the capital of”, the top‑5 are dominated by “Germany” (p≈0.897) with minimal “Berlin” mass (p≈0.00284):
> "topk": [ ["Germany", 0.896605...], ["the", 0.0539369], ["both", 0.00436054], ["a", 0.00380174], ["Europe", 0.00311423], ["Berlin", 0.00284079] ... ]  [L13–L36 in JSON]
Across the prompt (records.csv), important-word positions ("is", "called", "Germany", "simply") start to admit "Berlin" into their top‑5 around L22–L25, then strengthen alongside related terms. Examples:
> "...,22,14,is,... Berlin,0.00202 ..."  [L390 in records.csv]
> "...,23,15,called,... Berlin,0.00357 ..."  [L408 in records.csv]
> "...,24,16,simply,... Berlin,0.00389, Germany,0.00362, Frankfurt,0.00248 ..."  [L426 in records.csv]
> "...,25,14,is,... Berlin,0.08978, ... Germany,0.03455 ..."  [L441 in records.csv]
At the next-token slot, “Berlin” becomes top‑1 at L25 and stays top‑1 through L28, then shares headroom with quote/markup tokens near the unembed before reasserting strongly at L32:
> "pos,orig,25,16,⟨NEXT⟩,... Berlin,0.0335, ... is_answer,True ..."  [row 27 in pure CSV]
> "pos,orig,32,16,⟨NEXT⟩,3.611..., Berlin,0.38216, ... kl_to_final_bits,0.0 ..."  [row 34 in pure CSV]
Rotation vs amplification: cos_to_final rises early (≥0.2 by L11) while KL remains high until very late (first_kl_below_1.0 at L32), indicating an early alignment of direction with late calibration of scale (“early direction, late calibration”). Final-head calibration is clean (KL=0, temp≈1), so final probabilities are directly interpretable within this family.
Rest-mass sanity: rest_mass drops from ~0.999 in early layers to 0.2298 by the final layer [row 34 in pure CSV], with max after L_semantic = 0.9106 at L25 [row 27], consistent with concentration into the recorded top‑k as semantics consolidate.
Stylistic ablation: removing “simply” shifts L_semantic earlier by one layer (25→24) with no copy signal (both L_copy=null) [L1010–L1016 in JSON], consistent with mild guidance-style anchoring rather than content semantics.
Temperature robustness and additional test prompts: the family of test prompts shows stable top‑1 “Berlin” under slight rephrasings (e.g., “Germany's capital city is called” gives p≈0.811 for Berlin) and more diffuse distributions for more complex variants:
> "..., prompt": "Germany's capital city is called", ..., ["Berlin", 0.81055], ["\"", 0.04029], ...  [L245–L258 in JSON]
> "The capital city of Germany is named simply" top‑1 Berlin with lower mass (p≈0.346) amid punctuation [L107–L118 in JSON].

Temperature exploration explicitly confirms robustness to T changes: at T=0.1, Berlin rank 1 (p≈0.99961; entropy≈0.005 bits); at T=2.0, Berlin rank 1 (p≈0.036; entropy≈12.22 bits).
> "temperature": 0.1, "entropy": 0.005019..., ["Berlin", 0.999608...]  [L670–L676 in JSON]
> "temperature": 2.0, "entropy": 12.21997..., ["Berlin", 0.0359827...]  [L737–L744 in JSON]

Collapse-layer index under missing one‑word instruction: the test-prompt block reports only final next-token distributions (no layer‑by‑layer traces), so L_sem cannot be read off for those variants; rely on the ablation summary for stylistic effects.

Checklist (✓/✗/n.a.)
- RMS lens?  ✓  (RMSNorm detected; pre_norm)  [L810–L812, L946]
- LayerNorm bias removed?  n.a. (not needed; RMS model)  [L812]
- Entropy rise at unembed?  ✗ (entropy decreases sharply by final; quotes rise near L29–L31)
- FP32 un-embed promoted?  ✓ (unembed_dtype torch.float32; mixed-precision fix in place)  [L809, L255]
- Punctuation / markup anchoring?  ✓ (quotes top‑1 at L29–L31 in pure CSV)
- Copy-reflex?  ✗ (no copy_collapse=True in layers 0–3; L_copy=null)  [L819–L821]
- Grammatical filler anchoring?  ✓ ("simply" top‑1 at L20–L21, L23–L24 in pure CSV)

**Limitations & Data Quirks**
- Rest_mass remains >0.3 after L_semantic (0.9106 at L25; 0.2298 at L32), reflecting that only a fraction of probability mass is captured by the listed top‑k early in semantic emergence; rely on rank milestones and trends rather than absolute probabilities for cross-model claims.
- KL is lens-sensitive; while final KL≈0 here, early “lens_artifact_risk: high” in the sampled raw‑vs‑norm check suggests caution in interpreting early-layer probabilities; within‑model trends and ranks are preferred.
- Raw‑lens checks are sampled (mode: sample), not exhaustive; treat lens sanity as indicative rather than definitive.

**Model Fingerprint**
Mistral‑7B‑v0.1: semantic collapse at L 25; final entropy 3.611 bits; “Berlin” reasserts after punctuation near the unembed with clean final-head calibration.

---
Produced by OpenAI GPT-5
