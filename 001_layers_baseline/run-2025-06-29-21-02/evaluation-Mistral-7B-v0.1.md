# Replacing entire placeholder content with full evaluation
# Evaluation Report: mistralai/Mistral-7B-v0.1

## 1. Overview
mistralai/Mistral-7B-v0.1 (≈7 B parameters) was probed on 2025-06-29 21:02 with a layer-wise norm-lens sweep that logs entropy and top-k tokens for every layer and emits diagnostics in JSON and CSV.

## 2. Method sanity-check
JSON diagnostics confirm that the probe used RMS-Norm lens on both embeddings and blocks and that positional information is rotary only:
```9:12:001_layers_baseline/run-2025-06-29-21-02/output-Mistral-7B-v0.1.json
    "first_block_ln1_type": "RMSNormPre",
```
```6:6:001_layers_baseline/run-2025-06-29-21-02/output-Mistral-7B-v0.1.json
    "use_norm_lens": true,
```
The `context_prompt` string ends exactly with "called simply" (no trailing whitespace) [16]. Layers 0–3 do **not** predict "called" or "simply" as top-1, so no early copy-reflex is flagged. Diagnostics block exposes all three metrics: `L_copy`, `L_semantic`, `delta_layers` [18-21].

## 3. Quantitative findings
| Layer | Entropy (bits) & top-1 token |
|-------|-----------------------------|
| L 0 | 13.46 bits, '******/' |
| L 1 | 13.77 bits, '❶' |
| L 2 | 13.71 bits, '❶' |
| L 3 | 13.74 bits, 'eston' |
| L 4 | 13.74 bits, 'simply' |
| L 5 | 13.60 bits, 'simply' |
| L 6 | 13.64 bits, 'simply' |
| L 7 | 13.72 bits, 'plain' |
| L 8 | 13.74 bits, 'olas' |
| L 9 | 13.75 bits, 'anel' |
| L 10 | 13.76 bits, 'anel' |
| L 11 | 13.77 bits, 'inho' |
| L 12 | 13.76 bits, 'ifi' |
| L 13 | 13.81 bits, 'ív' |
| L 14 | 13.74 bits, 'ív' |
| L 15 | 13.65 bits, 'simply' |
| L 16 | 13.41 bits, 'simply' |
| L 17 | 13.15 bits, 'simply' |
| L 18 | 12.16 bits, 'simply' |
| L 19 | **4.17 bits, 'simply'** |
| L 20 | 3.36 bits, 'simply' |
| L 21 | 5.39 bits, 'simply' |
| L 22 | 7.81 bits, 'simply' |
| L 23 | 4.65 bits, 'simply' |
| L 24 | 5.24 bits, 'simply' |
| **L 25** | **2.02 bits, 'Berlin'** |
| L 26 | 1.63 bits, 'Berlin' |
| L 27 | 1.24 bits, 'Berlin' |
| L 28 | 1.44 bits, 'Berlin' |
| L 29 | 2.54 bits, 'Berlin' |
| L 30 | 2.38 bits, 'Berlin' |
| L 31 | 3.05 bits, 'Berlin' |
| L 32 | 3.61 bits, 'Berlin' |

Copy-collapse layer (L_copy) is *n.a.* – no layer reached p > 0.9 for any prompt token. Semantic layer is L 25; Δ cannot be computed.

## 4. Qualitative patterns & anomalies
The residual stream holds noisy filler tokens up to the mid-stack. A long plateau (L 4–L 24) is dominated by the instruction word "simply", reaching 0.77 p at L 20 yet never crossing the 0.9 copy-collapse threshold – a "soft copy reflex". Once the model's MLP stack resolves semantic content, "Berlin" becomes top-1 at L 25 and stabilises (> 0.8 at L 27). Low entropy at L 19 (4.17 bits) suggests sharpening before meaning clicks.

Test prompt "Berlin is the capital of" shows 0.95 bits and p(Germany)=0.90 [Lines 31-40], meaning removal of the "one-word" instruction shifts collapse one token earlier but still yields high certainty.

Temperature sweep confirms logit dominance: at τ = 0.1 "Berlin" holds 99.96 % [lines 710-720]; at τ = 2.0 it still ranks 1st with 3.6 %. Punctuation anchoring persists (quotes/commas) across temperatures.

Checklist:  
✓ RMS lens  
✓ LayerNorm bias removed  
✓ Punctuation anchoring  
✗ Entropy rise at unembed (entropy mostly falls)  
✓ Punctuation / markup anchoring  
✗ Copy reflex  
✗ Grammatical filler anchoring ("is/the/of" top-1 never observed)

## 5. Tentative implications for Realism ↔ Nominalism
1. Does the "simply" plateau indicate an intermediate lexical compression stage where the network aligns to instruction tokens before resolving semantics?  
2. Could the delayed semantic collapse (Δ n.a.) imply that Mistral's copy mechanism is less aggressive than Llama-3, favouring gradual entropy decay?  
3. Does the sharp entropy drop at L 19-20 correspond to a specific attention head group specialised for instruction compliance?  
4. Would pruning/ablating the heads active at L 19 reduce the "simply" fixation without harming answer accuracy?

## 6. Limitations & data quirks
L_copy is null; the copy-collapse heuristic may miss sub-90 % but still decisive plateaus. Early layers emit non-alphabetic tokens ('******/', '❶') that are unlikely in natural text, hinting at tokenizer edge-cases. CSV entropies remain high (>13 bits) for long stretches, limiting precision when comparing with other models.

## 7. Model fingerprint
"Mistral-7B-v0.1: semantic collapse at L 25; final entropy 3.61 bits; 'Berlin' gains >80 % by L 27."

---
Produced by OpenAI o3

