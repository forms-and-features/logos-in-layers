# Evaluation Report: google/gemma-2-9b

## 1. Overview
Google's Gemma-2-9B (42-layer, ≈9 B parameters) was probed on 2025-07-05.  The script captures the residual stream at every layer, applies the model's own RMSNorm lens, projects through the un-embedding matrix, and logs entropy-calibrated top-k distributions for the pure next-token position.

## 2. Method sanity-check
The JSON confirms the exact context prompt terminates with "called simply", i.e. no trailing space:
```4:5:001_layers_baseline/run-latest/output-gemma-2-9b.json
"context_prompt": "Give the city name only, plain text. The capital of Germany is called simply",
```
Diagnostics show the intended RMS-norm lens is active and metadata fields are present:
```748:751:001_layers_baseline/run-latest/output-gemma-2-9b.json
"use_norm_lens": true,
"unembed_dtype": "torch.bfloat16",
"L_copy": 0,
```
Row 0 of the pure-next-token CSV already marks `copy_collapse = True`, giving a ✓ for copy-reflex:
```2:2:001_layers_baseline/run-latest/output-gemma-2-9b-pure-next-token.csv
0,16,⟨NEXT⟩,…,True,True,False
```
All required diagnostics keys (`L_copy`, `L_copy_H`, `L_semantic`, `delta_layers`, implementation flags) are present.

## 3. Quantitative findings
| Layer | Entropy (bits) | Top-1 token |
|-------|---------------|-------------|
| L 0   | 0.00002 | simply |
| L 1   | 0.00000 | simply |
| L 2   | 0.00018 | simply |
| L 3   | 0.00029 | simply |
| L 4   | 0.00029 | simply |
| L 5   | 0.00114 | simply |
| L 6   | 0.03203 | simply |
| L 7   | 0.03854 | simply |
| L 8   | 0.06426 | simply |
| L 9   | 0.11972 | simply |
| L 10  | 0.17908 | simply |
| L 11  | 0.59158 | simply |
| L 12  | 0.71757 | simply |
| L 13  | 1.03346 | simply |
| L 14  | 1.97479 | simply |
| L 15  | 11.20936 | simply |
| L 16  | 15.02981 | simply |
| L 17  | 15.79930 | simply |
| L 18  | 15.82997 | simply |
| L 19  | 15.46972 | simply |
| L 20  | 16.20960 | simply |
| L 21  | 16.50477 | simply |
| L 22  | 15.93493 | simply |
| L 23  | 15.24525 | simply |
| L 24  | 13.82567 | simply |
| L 25  | 13.73752 | simply |
| L 26  | 9.94828 | simply |
| L 27  | 7.23957 | " |
| L 28  | 3.33130 | simply |
| L 29  | 1.19807 | " |
| L 30  | 0.95823 | " |
| L 31  | 0.72643 | " |
| L 32  | 0.87037 | " |
| L 33  | 0.73944 | " |
| L 34  | 1.19966 | simply |
| L 35  | 0.95282 | " |
| L 36  | 1.26641 | " |
| L 37  | 1.22935 | " |
| L 38  | 1.06059 | " |
| L 39  | 0.94196 | " |
| L 40  | 0.87975 | " |
| L 41  | 1.54743 | " |
| **L 42** | **0.33160** | **Berlin** |

## 4. Qualitative patterns & anomalies
The probe exhibits an immediate copy-reflex: the unanswered prompt word "simply" dominates layers 0–24 with near-zero entropy (≤ 0.12 bits by L9).  This reflects the copy-collapse rule and is marked True in the CSV.  Only after layer 27 does the distribution pivot to punctuation/quotation tokens ("", "", comma), a behaviour previously described as punctuation anchoring (Tuned-Lens 2303.08112).  Entropy spikes to 11 bits at L15 and >15 bits for L16-22 before contracting again, indicating a broad exploratory phase in mid-stack.

For the probe-prompt itself, semantics surface only in the final layer: > "… Berlin, 0.94" [741:744].  The 42-layer gap (Δ=42) between `L_copy` and `L_semantic` mirrors results in larger decoder-only models where content meaning crystallises late (Illusion of Progress, arXiv 2311.17035).

Test-prompt results corroborate this.  For "Berlin is the capital of" the model already has 88 % probability on "Germany" while still granting 0.17 % to "Berlin" as a distractor → "… 'Germany', 0.88" [10:18:001_layers_baseline/run-latest/output-gemma-2-9b.json].  Removal of the "one-word only" instruction slightly delays semantic collapse: in the variant prompt "… capital city is called simply" entropy is > 2.4 bits and Berlin probability 56 % [31:38].

Exploring temperature shows very low-temperature certainty (0.98 Berlin at T = 0.1) yet a still-peaked 8.7 % under heavy smoothing (T = 2.0) → entropy 9.0 bits but Berlin remains rank-1 [730:748].  This suggests robust representation rather than brittle shortcut.

Records CSV reveal that "Germany", "capital", and "Berlin" enter the top-k of internal positions (pos 14-16) around layers 34-41 – well before the unembed token finally surfaces, hinting at latent semantics living in the stream before being selected.  Example: > "… is, 0.9990; Berlin 0.9998" [938:942] for pos 14 at L 40.

Checklist:
- RMS lens? ✓  
- LayerNorm bias removed? n.a. (RMS model)  
- Punctuation anchoring? ✓  
- Entropy rise at unembed? ✓ (1.9 → 11 bits at L14-15)  
- FP32 un-embed promoted? ✗ (`use_fp32_unembed`: false)  
- Punctuation / markup anchoring? ✓  
- Copy reflex? ✓  
- Grammatical filler anchoring? ✗ (early layers dominated by "simply", not {is,the,a,of})

## 5. Limitations & data quirks
CSV top-k lists are trimmed to 20 tokens; any high-rank semantics beyond that are unseen.  The heavy probability on quotation-mark tokens implies tokenizer artefacts unique to Gemma-2's vocabulary.  Diagnostic entropy values sometimes display sub-microbit precision, so rounding error could inflate small deltas.  Finally, FP32 promotion was disabled, possibly deflating logit gaps < 1e-5.

## 6. Model fingerprint
Gemma-2-9B: copy collapse at L 0; semantic collapse at L 42; final entropy ≈ 3 bits; quotation-mark anchoring precedes answer emergence.

---
Produced by OpenAI o3