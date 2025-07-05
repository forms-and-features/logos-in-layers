# Evaluation Report: Qwen/Qwen3-8B

*Run executed on: 2025-07-05 09:51 (UTC)*

## 1. Overview
Qwen-3-8B (≈8 billion parameters) was probed on 2025-07-05 using the layer-by-layer RMS-norm lens script.  The probe records residual-stream entropies and top-k predictions at every block, yielding per-layer collapse metrics and diagnostic metadata for the base prompt and several sanity prompts.

## 2. Method sanity-check
The script prints that the RMS-norm lens is applied and that rotary models bypass additive position hooks:
```464:469:001_layers_and_logits/run.py
print("Using NORMALIZED residual stream (RMS + learned scale)")
```
```495:495:001_layers_and_logits/run.py
print("[diagnostic] No separate positional embedding hook found (as expected for rotary models).")
```
The JSON shows the exact prompt string (no trailing space):
```3:3:001_layers_and_logits/run-latest/output-Qwen3-8B.json
"context_prompt": "Give the city name only, plain text. The capital of Germany is called simply",
```
Diagnostics block contains the expected keys (`use_norm_lens`, `unembed_dtype`, `L_copy`, `L_semantic`, …) – e.g.
```820:822:001_layers_and_logits/run-latest/output-Qwen3-8B.json
"L_semantic": 31,
```
In layers 0–3 the `copy_collapse` flag is never `True`; therefore copy-reflex is **not** triggered.

## 3. Quantitative findings
| Layer | Entropy (bits) | Top-1 token |
|-------|---------------|-------------|
| L 0 | 17.21 | `CLICK` |
| L 1 | 17.19 | `обязан` |
| L 2 | 15.86 | `hook` |
| L 3 | 13.65 | `对学生` |
| L 4 | 15.69 | `原因之一` |
| L 5 | 16.77 | `(PR` |
| L 6 | 17.01 | `呼和` |
| L 7 | 16.77 | `的方式来` |
| L 8 | 16.91 | `的方式来` |
| L 9 | 17.06 | `俗` |
| L 10 | 17.02 | `名字` |
| L 11 | 17.04 | `名字` |
| L 12 | 17.07 | `简单` |
| L 13 | 17.07 | `简单` |
| L 14 | 17.06 | `ifiable` |
| L 15 | 17.06 | `名字` |
| L 16 | 17.06 | `ifiable` |
| L 17 | 17.06 | `名称` |
| L 18 | 17.06 | `名称` |
| L 19 | 17.05 | `名称` |
| L 20 | 17.03 | `flat` |
| L 21 | 17.01 | `这个名字` |
| L 22 | 16.93 | `______` |
| L 23 | 16.91 | `______` |
| L 24 | 16.84 | `这个名字` |
| L 25 | 16.76 | `simply` |
| L 26 | 16.71 | `simply` |
| L 27 | 16.60 | `simply` |
| L 28 | 16.47 | `simply` |
| L 29 | 15.76 | `Germany` |
| L 30 | 14.46 | `Germany` |
| **L 31** | **9.69** | **`Berlin`** |
| L 32 | 2.73 | `Germany` |
| L 33 | 2.56 | `Berlin` |
| L 34 | 4.87 | `Berlin` |
| L 35 | 10.57 | `Berlin` |
| L 36 | 3.12 | `Berlin` |

Semantic collapse therefore occurs first at L 31 (matching `L_semantic` in diagnostics).

## 4. Qualitative patterns & anomalies
Entropy stays >16 bits for the first 25 layers and is filled with rare multilingual or struct-like tokens, indicating nearly uniform logits rather than prompt copying.  From L 25–28 the top-1 flips to the filler word "simply", reflecting the final word of the context and showing a shallow grammatical anchor but no high-confidence echo (`copy_collapse` never fires).  A two-step transition follows: "Germany" dominates layers 29–30, then true semantic convergence ("Berlin") snaps in at L 31 and remains stable thereafter.

The pure-next-token entropy drops sharply (17 → 9.7 bits) at the semantic layer, then further to ~2.5 bits when "Berlin" versus "Germany" compete (L 33).  Records CSV shows the answer token already climbing the ranks earlier; e.g.
```520:522:001_layers_and_logits/run-latest/output-Qwen3-8B-records.csv
31,13, is,0.6906…, Berlin,0.83  …
```
Test prompts corroborate: for "Berlin is the capital of" the model answers *Germany* with 73 % [L14]; for the one-word instruction prompt it gives *Berlin* at 77 % [L40], implying the collapse layer shifts when the instruction is removed.
Temperature sweep shows extreme certainty at T 0.1 (p≈0.999 for "Berlin", entropy ≈0.01 bits) yet still prefers "Berlin" at T 2.0 (p≈0.042), suggesting a relatively peaked head even at high temperature.

Checklist:  
✓ RMS lens applied  
✓ LayerNorm bias removed (n.a. for RMSNorm)  
✗ Punctuation anchoring  
✓ Entropy rise at un-embed (spike before collapse)  
✗ FP32 un-embed promoted  
✗ Punctuation / markup anchoring  
✗ Copy reflex  
✗ Grammatical filler anchoring (layers 0–5 not an English filler)

## 5. Limitations & data quirks
Early-layer top-1 tokens are random multilingual morphemes, hinting at tokenizer edge cases rather than meaningful features; this inflates early entropies.  The lack of a copy-collapse layer leaves `delta_layers` undefined, limiting cross-model comparisons.  Diagnostics use FP16 un-embedding, so sub-bit differences near collapse may be under-resolved.

## 6. Model fingerprint
Qwen-3-8B: semantic collapse at L 31; final next-token entropy ≈3 bits; "Berlin" stabilises after a brief Germany/Berlin rivalry.

---
Produced by OpenAI o3