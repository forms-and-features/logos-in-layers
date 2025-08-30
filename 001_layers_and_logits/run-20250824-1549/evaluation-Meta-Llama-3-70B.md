# Evaluation Report: meta-llama/Meta-Llama-3-70B

## 1. Overview
Meta‑Llama‑3‑70B (80 layers) evaluated on 2025‑08‑24 with normalized residual lens and FP32 unembedding. The probe captures layer‑by‑layer entropy and next‑token distributions, collapse flags, and diagnostics for scaling/lens configuration.

## 2. Method sanity‑check
Diagnostics confirm the intended norm lens and positional handling: “use_norm_lens”: true and FP32 unembed are set, and layer‑0 interpretation accounts for rotary PE via token‑only embeddings with real ln1 applied. Examples: “use_norm_lens”: true; “layer0_position_info”: “token_only_rotary_model”; “layer0_norm_fix”: “using_real_ln1_on_embeddings”.

> "use_norm_lens": true  [output-Meta-Llama-3-70B.json L807]
>
> "layer0_position_info": "token_only_rotary_model"  [L816]

Prompt integrity: context_prompt ends with “called simply” (no trailing space).

> "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply"  [L817]

Diagnostics fields present: L_copy, L_copy_H, L_semantic, delta_layers, implementation flags (use_fp32_unembed, unembed_dtype, ln types).

> "L_copy": null, "L_copy_H": null, "L_semantic": 40, "delta_layers": null  [L819–L822]

Copy‑collapse flag check (copy_threshold=0.90, copy_margin=0.05): No rows with copy_collapse=True in pure-next-token CSV across layers 0–80; first four layers all False (e.g., layer 0 …, copy_collapse=False). Rule did not fire; no spurious triggers.

> "…,0.9999548513,…,False,False,False" (layer 0)  [output-Meta-Llama-3-70B-pure-next-token.csv row 2]

## 3. Quantitative findings
Per‑layer next‑token probe (entropy bits, top‑1 token). Bold indicates first semantic layer (top‑1 = “Berlin”).

| Layer | Entropy (bits) | Top‑1 |
|---:|---:|:---|
| 0 | 16.9681 | ‘winding’ |
| 1 | 16.9601 | ‘cepts’ |
| 2 | 16.9634 | ‘улю’ |
| 3 | 16.9626 | ‘zier’ |
| 4 | 16.9586 | ‘alls’ |
| 5 | 16.9572 | ‘alls’ |
| 6 | 16.9561 | ‘alls’ |
| 7 | 16.9533 | ‘NodeId’ |
| 8 | 16.9594 | ‘inds’ |
| 9 | 16.9597 | ‘NodeId’ |
| 10 | 16.9524 | ‘inds’ |
| 11 | 16.9560 | ‘inds’ |
| 12 | 16.9564 | ‘lia’ |
| 13 | 16.9552 | ‘eds’ |
| 14 | 16.9504 | ‘idders’ |
| 15 | 16.9533 | ‘Kok’ |
| 16 | 16.9522 | ‘/plain’ |
| 17 | 16.9480 | ‘nut’ |
| 18 | 16.9443 | ‘nut’ |
| 19 | 16.9475 | ‘nut’ |
| 20 | 16.9464 | ‘nut’ |
| 21 | 16.9380 | ‘burge’ |
| 22 | 16.9378 | ‘simply’ |
| 23 | 16.9357 | ‘bur’ |
| 24 | 16.9497 | ‘bur’ |
| 25 | 16.9375 | ‘�’ |
| 26 | 16.9383 | ‘�’ |
| 27 | 16.9373 | ‘za’ |
| 28 | 16.9328 | ‘/plain’ |
| 29 | 16.9328 | ‘plain’ |
| 30 | 16.9386 | ‘zed’ |
| 31 | 16.9251 | ‘simply’ |
| 32 | 16.9406 | ‘simply’ |
| 33 | 16.9271 | ‘plain’ |
| 34 | 16.9323 | ‘simply’ |
| 35 | 16.9292 | ‘simply’ |
| 36 | 16.9397 | ‘simply’ |
| 37 | 16.9346 | ‘simply’ |
| 38 | 16.9342 | ‘simply’ |
| 39 | 16.9349 | ‘simply’ |
| 40 | 16.9374 | **‘Berlin’** |
| 41 | 16.9362 | ‘" ""’ |
| 42 | 16.9444 | ‘" ""’ |
| 43 | 16.9413 | ‘Berlin’ |
| 44 | 16.9260 | ‘Berlin’ |
| 45 | 16.9402 | ‘" ""’ |
| 46 | 16.9552 | ‘" ""’ |
| 47 | 16.9393 | ‘" ""’ |
| 48 | 16.9388 | ‘" ""’ |
| 49 | 16.9369 | ‘" ""’ |
| 50 | 16.9438 | ‘" ""’ |
| 51 | 16.9401 | ‘" ""’ |
| 52 | 16.9220 | ‘Berlin’ |
| 53 | 16.9330 | ‘Berlin’ |
| 54 | 16.9424 | ‘Berlin’ |
| 55 | 16.9419 | ‘Berlin’ |
| 56 | 16.9210 | ‘Berlin’ |
| 57 | 16.9335 | ‘Berlin’ |
| 58 | 16.9411 | ‘Berlin’ |
| 59 | 16.9441 | ‘Berlin’ |
| 60 | 16.9229 | ‘Berlin’ |
| 61 | 16.9396 | ‘Berlin’ |
| 62 | 16.9509 | ‘Berlin’ |
| 63 | 16.9458 | ‘Berlin’ |
| 64 | 16.9263 | ‘Berlin’ |
| 65 | 16.9334 | ‘" ""’ |
| 66 | 16.9407 | ‘Berlin’ |
| 67 | 16.9304 | ‘Berlin’ |
| 68 | 16.9240 | ‘Berlin’ |
| 69 | 16.9315 | ‘Berlin’ |
| 70 | 16.9257 | ‘Berlin’ |
| 71 | 16.9226 | ‘Berlin’ |
| 72 | 16.9221 | ‘Berlin’ |
| 73 | 16.9181 | ‘" ""’ |
| 74 | 16.9143 | ‘Berlin’ |
| 75 | 16.9127 | ‘Berlin’ |
| 76 | 16.9190 | ‘Berlin’ |
| 77 | 16.9099 | ‘Berlin’ |
| 78 | 16.9185 | ‘Berlin’ |
| 79 | 16.9422 | ‘Berlin’ |
| 80 | 2.5890 | ‘Berlin’ |

Notes and references:
- L40 is first with top‑1 ‘Berlin’ (is_answer=True): “40,…, Berlin,…, …, …, …, …, …, …, …, …, …, …, …,False,False,True” [output‑Meta‑Llama‑3‑70B‑pure‑next‑token.csv row 42].

Confidence milestones:
- p > 0.30 at layer 80 (p = 0.4783) [pure‑next‑token.csv row 82].
- p > 0.60: n.a. (never exceeds 0.60 in layers 0–80).
- Final‑layer p = 0.4783; entropy = 2.5890 bits [row 82].

ΔH (bits) = n.a. (no copy‑collapse layer; “L_copy”: null) [output‑Meta‑Llama‑3‑70B.json L819].

## 4. Qualitative patterns & anomalies
Punctuation anchoring competes with semantics after emergence. Quote‑like tokens (rendered as '" ""') frequently top the list in mid‑stack layers (e.g., L41–42, L45–51, L65, L73), consistent with known markup anchoring in logit‑lens probes where stylistic or quoting tokens momentarily outrank content words (cf. Tuned‑Lens 2303.08112). For example: “41,…, top‑1 '" ""' …; 42,…, top‑1 '" ""' …” [pure‑next‑token.csv rows 43–44].

Negative control (prompt: “Berlin is the capital of”): top‑5 are Germany (0.8516), the (0.0791), and (0.0146), modern (0.0048), Europe (0.0031) — no leakage of “Berlin” into the continuation. “topk”: [“ Germany”, 0.85156], [“ the”, 0.07910], [“ and”, 0.01465], [“ modern”, 0.00476], [“ Europe”, 0.00307] [output‑Meta‑Llama‑3‑70B.json L14–L31].

Important‑word trajectory: On the pure next‑token probe, “Berlin” first appears in any top‑5 at L38 and becomes top‑1 at L40: “38,…, top‑3 ‘Berlin’, 2.50e‑05; …” [pure‑next‑token.csv row 40], then “40,… top‑1 ‘Berlin’ … is_answer=True” [row 42]. It stabilises as top‑1 by L52 onward, with occasional quote‑token outrankings (e.g., L65, L73). Semantically related words co‑rise: “Germany” enters the top‑5 alongside Berlin at L38–41 (e.g., L39 lists ‘Germany’ in top‑5) [rows 41–43]. The instruction word “simply” dominates top‑1 throughout L22–39 before yielding to “Berlin,” reflecting grammatical/format anchoring prior to semantic collapse.

Prompt variant sensitivity: Across test prompts that omit the “one‑word” instruction, “Berlin” remains rank‑1 with substantial probability for Germany‑focused variants: e.g., “Germany has its capital at” → “ Berlin” 0.8789 [output‑Meta‑Llama‑3‑70B.json L343–L345], while the negative control (“Berlin is the capital of”) correctly predicts the country, not the city. Collapse layer index (first ‘Berlin’ top‑1) is computed from the main probe and is reported as L_semantic=40 [JSON L821]; test prompts do not redefine this index.

Rest‑mass sanity: Rest_mass remains ~0.9999 through mid‑layers (expected under high entropy), then drops sharply only at the final layer: rest_mass = 0.1074 at L80 [pure‑next‑token.csv row 82]. The maximum after L_semantic is ≈0.999917 at L46 [row 48], indicating no precision loss spikes.

Temperature robustness: At T=0.1, Berlin rank 1 (p = 0.9933; entropy 0.058 bits) [JSON L670–L676]. At T=2.0, Berlin remains rank 1 (p = 0.0357; entropy 14.464 bits) [JSON L742–L750]. Entropy rises with temperature as expected.

Checklist:
- RMS lens? ✓ (RMSNorm; “first_block_ln1_type”: “RMSNorm”) [JSON L816–L817]
- LayerNorm bias removed? ✓/n.a. (“not_needed_rms_model”) [L818]
- Entropy rise at unembed? ✓ small (final_prediction 2.597 vs L80 2.589 bits) [JSON L826; pure row 82]
- FP32 un‑embed promoted? ✓ (“use_fp32_unembed”: true; “unembed_dtype”: “torch.float32”) [L813–L815]
- Punctuation / markup anchoring? ✓ (quote tokens outrank mid‑stack)
- Copy‑reflex? ✗ (no copy_collapse=True in layers 0–3; all False) [pure rows 2–5]
- Grammatical filler anchoring? ✗ (no {“is”, “the”, “a”, “of”} as top‑1 in layers 0–5)

## 5. Limitations & data quirks
- No copy‑collapse detected (L_copy = null), so ΔH relative to copy‑collapse is undefined. This is consistent with the strict threshold (τ=0.90, δ=0.05) and the prompt design.
- Rest_mass > 0.99 after L_semantic is expected given high entropy and top‑20 truncation; no evidence of norm‑lens mis‑scale. The large final drop (to 0.107) reflects concentration at the last layer, not precision loss.

## 6. Model fingerprint
Meta‑Llama‑3‑70B: semantic collapse at L 40; final entropy 2.59 bits; “Berlin” becomes reliably top‑1 by mid‑50s with intermittent quote‑token competition.

---
Produced by OpenAI GPT-5 
