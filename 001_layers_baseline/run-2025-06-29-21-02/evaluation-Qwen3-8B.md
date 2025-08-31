# Evaluation Report: Qwen/Qwen3-8B

*Run executed on: 2025-06-29 21:02:18*

## 1. Overview  
Qwen3-8B is an 8-billion-parameter base model released by Alibaba/Qwen.  The probe (see `run.py`) performs a layer-by-layer logit-lens sweep with RMS/LayerNorm corrections, recording the entropy and top-k tokens at every residual stream.  Results were collected on 2025-06-29 and saved to the JSON/CSV artefacts referenced below.

## 2. Method sanity-check  
The diagnostics block confirms that the intended normalised lens (RMSNorm Pre) was applied and that positional information is rotary (no additive embeddings).  Key console lines:
```6:10:001_layers_baseline/run-2025-06-29-21-02/output-Qwen3-8B.json
    "use_norm_lens": true,
    "first_block_ln1_type": "RMSNormPre",
    "final_ln_type": "RMSNormPre",
    "layernorm_bias_fix": "not_needed_rms_model",
    "norm_alignment_fix": "using_ln2_rmsnorm_for_post_block",
```
The prompt string ends exactly with "called simply" (no trailing space):
```15:16:001_layers_baseline/run-2025-06-29-21-02/output-Qwen3-8B.json
    "layer0_position_info": "token_only_rotary_model",
    "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply",
```
Layer-0–3 top-1 tokens are 'CLICK' and several non-prompt Chinese tokens (见 §3); neither "called" nor "simply", hence no early copy-reflex flag.  The JSON diagnostics include all required metrics:  
```18:20:001_layers_baseline/run-2025-06-29-21-02/output-Qwen3-8B.json
    "L_copy": 25,
    "L_copy_H": 25,
    "L_semantic": 31,
```

## 3. Quantitative findings  
| Layer | Entropy (bits) | Top-1 token |
|------|---------------|-------------|
| L 0 | 7.83 | 'CLICK' |
| L 1 | 8.53 | '湾' |
| L 2 | 8.59 | '湾' |
| L 3 | 8.82 | '湾' |
| L 4 | 7.65 | '湾' |
| L 5 | 11.19 | '院子' |
| L 6 | 9.57 | '-minded' |
| L 7 | 8.74 | 'mente' |
| L 8 | 7.91 | 'tion' |
| L 9 | 7.69 | 'ifiable' |
| L 10 | 4.50 | 'ifiable' |
| L 11 | 2.76 | 'ifiable' |
| L 12 | 3.31 | 'ifiable' |
| L 13 | 4.01 | 'ifiable' |
| L 14 | 6.21 | 'ifiable' |
| L 15 | 4.97 | 'ifiable' |
| L 16 | 4.07 | 'name' |
| L 17 | 2.25 | 'name' |
| L 18 | 2.93 | 'name' |
| L 19 | 3.94 | 'names' |
| L 20 | 4.06 | '这个名字' |
| L 21 | 1.46 | '这个名字' |
| L 22 | 3.20 | '______' |
| L 23 | 2.32 | '______' |
| L 24 | 1.15 | '这个名字' |
| **L 25** | **0.08** | **'simply'** |
| L 26 | 2.91 | '______' |
| L 27 | 2.40 | '"""' |
| L 28 | 1.65 | '"""' |
| L 29 | 1.07 | 'Germany' |
| L 30 | 0.88 | 'Germany' |
| **L 31** | **0.35** | **'Berlin'** |
| L 32 | 1.52 | 'Berlin' |
| L 33 | 0.65 | 'Berlin' |
| L 34 | 0.08 | 'Berlin' |
| L 35 | 1.64 | 'Berlin' |
| L 36 | 3.12 | 'Berlin' |

Δ = L_semantic − L_copy = **6 layers**.

## 4. Qualitative patterns & anomalies  
A striking "long copy-reflex" emerges: layer 25 confidently echoes the prompt word "simply" (p ≈ 0.99) before any sign of semantic grounding, and it takes another six transformer blocks for 'Berlin' to rise to dominance.  The copy collapse row is illustrated below:
```27:27:001_layers_baseline/run-2025-06-29-21-02/output-Qwen3-8B-pure-next-token.csv
25,15,⟨NEXT⟩,0.081…, simply,0.9916…, …,True,True,False
```
By L 31 the model's entropy rebounds slightly (0.35 bits) yet decisively selects 'Berlin' with 94 % probability, confirming the semantic layer:
```33:33:001_layers_baseline/run-2025-06-29-21-02/output-Qwen3-8B-pure-next-token.csv
31,15,⟨NEXT⟩,0.3457…, Berlin,0.9382…, …,False,True,True
```
Test-prompt blocks show that removing the "one-word" instruction ("Berlin is the capital of") still yields the correct answer 'Germany' at 73 % [29:43-54].  Collapse depth therefore appears robust to instruction framing; the model defaults to the factual pair (Berlin↔Germany) when the linguistic direction reverses.

Temperature exploration corroborates a sharp entropy drop at T = 0.1 (0.01 bits) and a broadening at T = 2.0 (13.4 bits), consistent with a well-calibrated unembedding distribution.

Checklist:  
✓ RMS lens | n.a. LayerNorm bias | ✓ Punctuation anchoring (" / '______') | ✓ Entropy rise at unembed (0.35 → 3.12 bits) | ✓ Mark-up anchoring | ✓ Copy reflex | ✗ Grammatical filler anchoring.

## 5. Tentative implications for Realism ↔ Nominalism  
1. Does the six-layer gap indicate a distinct "surface-form" sub-space that the network must first exit before accessing factual memory?  
2. Could the persistence of punctuation placeholders ('______') reflect a nominalist bias wherein tokens are treated as abstract labels until a later semantic binding stage?  
3. Does rotary-only positional encoding delay semantic resolution compared with additive embeddings seen in other architectures?  
4. How does heavy Chinese/English token competition in early layers affect the realism of language-agnostic representations?

## 6. Limitations & data quirks  
The probe ran on CPU with fp32 weights, increasing latency and possibly cache eviction.  CSV shows truncated top-k strings; some tokens ('"""'', '______') are artefacts of the tokenizer, which could inflate copy-collapse detection.  Absence of per-layer results for alternative prompts limits analysis of collapse-layer shift.

## 7. Model fingerprint  
"Qwen3-8B: copy collapse at L 25; semantic token 'Berlin' appears at L 31; final entropy 3.1 bits."

---
Produced by OpenAI o3

