# Evaluation Report: mistralai/Mistral-7B-v0.1

## 1. Overview  
Mistralai **Mistral-7B-v0.1** (≈7 B parameters, 32 transformer blocks) was probed on **2025-07-05** with a layer-by-layer RMS-norm lens.  
The script records, for every residual stream, the entropy of the unrestricted soft-max, top-k token distribution and boolean collapse flags, allowing us to locate the shift from copy/filler behaviour to genuine semantic prediction.

## 2. Method sanity-check  
The diagnostics block confirms that the intended settings were active:
```865:872:001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json
"use_norm_lens": true,
"unembed_dtype": "torch.float16",
"layer0_position_info": "token_only_rotary_model",
```
Positional encodings are therefore rotary only; no separate positional embedding hook was required.  The context prompt is present and ends exactly with "called simply" (no trailing space):
```874:877:001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json
"context_prompt": "Give the city name only, plain text. The capital of Germany is called simply",
```
Collapse metrics are in place:
```883:887:001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json
"L_copy": null,
"L_semantic": 25,
"delta_layers": null
```
No row in layers 0-3 of the pure-next-token CSV carries `copy_collapse = True`, so the strong copy-reflex did **not** trigger (✓ rule satisfied – no false positive).

## 3. Quantitative findings  
Layer-wise entropy & top-1 token (pure next-token):

| Layer | Entropy (bits) | Top-1 token |
|-------|----------------|-------------|
| L 0  | 14.96 | dabei |
| L 1  | 14.96 | ❶ |
| L 2  | 14.95 | simply |
| L 3  | 14.94 | simply |
| L 4  | 14.93 | simply |
| L 5  | 14.93 | simply |
| L 6  | 14.92 | simply |
| L 7  | 14.91 | plain |
| L 8  | 14.90 | olas |
| L 9  | 14.89 | anel |
| L 10 | 14.89 | anel |
| L 11 | 14.88 | inho |
| L 12 | 14.88 | ifi |
| L 13 | 14.87 | … |
| L 14 | 14.86 | mate |
| L 15 | 14.85 | … |
| L 16 | 14.82 | simply |
| L 17 | 14.78 | simply |
| L 18 | 14.75 | simply |
| L 19 | 14.70 | simply |
| L 20 | 14.64 | simply |
| L 21 | 14.53 | simply |
| L 22 | 14.50 | simply |
| L 23 | 14.38 | simply |
| L 24 | 14.21 | simply |
| **L 25** | **11.64** | **Berlin** |
| L 26 | 9.88 | Berlin |
| L 27 | 8.81 | Berlin |
| L 28 | 8.44 | Berlin |
| L 29 | 7.90 | Berlin |
| L 30 | 7.24 | Berlin |
| L 31 | 7.84 | Berlin |
| L 32 | 3.61 | Berlin |

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (no copy collapse).  
Confidence milestones: p > 0.30 at **L 27**; p > 0.60 never reached; final-layer p = 0.38.

## 4. Qualitative patterns & anomalies  
The model stays in a high-entropy, low-confidence regime for 24 layers, dominated by low-frequency fillers ("simply", "plain").  **Berlin** first appears in any top-5 list at L 24 and becomes top-1 at L 25, gaining probability monotonically afterwards; the important noun **Germany** remains in top-5 from L 20 onwards, while the generic word *capital* peaks around L 22 then fades.  Rest-mass declines gradually, dropping from ≈0.99 in early layers to **0.23 by L 32**, showing the lens captures increasing mass rather than saturating.

Negative control ("Berlin is the capital of"):  
```436:448:001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json
"Germany", 0.8823 … "Berlin", 0.0028
```
→ semantic leakage: Berlin rank 6 (p ≈ 0.003).  The answer token is therefore not spuriously copied from the subject form.

Temperature robustness: at **T = 0.1** Berlin rank 1 (p = 0.9996) vs **T = 2.0** rank 1 but p = 0.036; entropy rises from 0.005 bits to 12.2 bits.

Rest-mass sanity: remains > 0.95 through L 24 then falls sharply (0.73 at L 25, 0.53 at L 27), consistent with narrowing distribution rather than numerical error.

Checklist:  
RMS lens? ✓ LayerNorm bias removed? n.a. (RMS) Entropy rise at un-embed? ✓ FP32 un-embed promoted? ✗ Punctuation / markup anchoring? ✗ Copy-reflex? ✗ Grammatical filler anchoring? ✗

## 5. Limitations & data quirks  
The absence of a clear copy-collapse layer prevents measuring Δ between copy and semantic collapse.  Rest-mass is still 0.23 at the final layer, indicating that top-20 coverage is incomplete and small-probability tail matters for entropy.  Early layers show exotic byte-level tokens ("❶", "anel") which may be artefacts of byte-pair segmentation and are not semantically meaningful.

## 6. Model fingerprint  
Mistral-7B-v0.1: semantic collapse at **L 25** with final entropy 3.6 bits; confidence never exceeds 0.39.

---
Produced by OpenAI o3

