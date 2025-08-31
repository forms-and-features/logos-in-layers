# Evaluation Report: 01-ai/Yi-34B

*Run executed on: 2025-07-05 16:29:56*

## 1. Overview  
01-ai/Yi-34B (34 B parameters) was probed on 5 July 2025.  The script performs a layer-by-layer logit-lens sweep over a fixed prompt, recording per-layer entropy and top-k tokens (raw and RMS-normalised streams) plus specialised collapse flags.  Here we review the single-model results.

## 2. Method sanity-check  
JSON diagnostics confirm that the probe used the RMS-norm lens (`"first_block_ln1_type": "RMSNorm"` [line 809]) with `"use_norm_lens": true` [line 806], and no FP32 promotion (`"unembed_dtype": "torch.float16"`).  The `context_prompt` appears exactly as intended and ends with "called simply" [line 3].  Positional embeddings were additive-free (`layer0_position_info": "token_only_rotary_model"`).  The pure-next-token CSV reports no `copy_collapse = True` in layers 0–3, so no copy-reflex was triggered; consequently `L_copy`, `L_copy_H` are `null`, while `L_semantic = 44` and `delta_layers = null` are present in diagnostics.  Copy-collapse rule check: *n.a.* – flag never fired.

## 3. Quantitative findings
| Layer | Entropy (bits) | Top-1 token |
|------:|---------------:|-------------|
| 0 | 15.96 | Denote |
| 1 | 15.97 | PREFIX |
| 2 | 15.96 | ifiable |
| 3 | 15.96 | . |
| 4 | 15.96 | Marketing |
| 5 | 15.96 | 拐 |
| 6 | 15.96 | ostream |
| 7 | 15.95 | tons |
| 8 | 15.95 | tons |
| 9 | 15.95 | 诉讼代理人 |
| 10 | 15.94 | 诉讼代理人 |
| 11 | 15.95 | 诉讼代理人 |
| 12 | 15.94 | Marketing |
| 13 | 15.94 | 避 |
| 14 | 15.93 | 避 |
| 15 | 15.93 | 的人民 |
| 16 | 15.93 | 诉讼代理人 |
| 17 | 15.93 | jo |
| 18 | 15.93 | ERE |
| 19 | 15.93 | ibo |
| 20 | 15.93 | MDM |
| 21 | 15.91 | MDM |
| 22 | 15.91 | MDM |
| 23 | 15.88 | . |
| 24 | 15.89 | fing |
| 25 | 15.90 | fing |
| 26 | 15.89 | fing |
| 27 | 15.89 | fing |
| 28 | 15.89 | MDM |
| 29 | 15.88 | MDM |
| 30 | 15.88 | ANE |
| 31 | 15.87 | MDM |
| 32 | 15.86 | MDM |
| 33 | 15.85 | MDM |
| 34 | 15.84 | aday |
| 35 | 15.83 | 日 |
| 36 | 15.82 | geries |
| 37 | 15.80 | 踊 |
| 38 | 15.78 | 什么呢 |
| 39 | 15.74 | simply |
| 40 | 15.74 | capital |
| 41 | 15.71 | """ |
| 42 | 15.66 | """ |
| 43 | 15.59 | """ |
| **44** | **15.34** | **Berlin** |
| 45 | 15.23 | Berlin |
| 46 | 14.68 | Berlin |
| 47 | 14.62 | Berlin |
| 48 | 14.96 | Berlin |
| 49 | 14.57 | Berlin |
| 50 | 14.54 | Berlin |
| 51 | 14.60 | Berlin |
| 52 | 14.86 | Berlin |
| 53 | 14.88 | Berlin |
| 54 | 14.91 | Berlin |
| 55 | 14.93 | Berlin |
| 56 | 15.05 | Berlin |
| 57 | 14.69 | Berlin |
| 58 | 14.85 | Berlin |
| 59 | 14.08 |  (space) |
| 60 | 2.96 | Berlin |

ΔH (bits) = n.a. (no copy-collapse layer)  
Confidence milestones: p > 0.30 at layer 60; p > 0.60 – not reached; final-layer p = 0.56.

## 4. Qualitative patterns & anomalies  
Early layers remain high-entropy (> 15.9 bits) and dominated by low-probability junk or Chinese legal phrases, consistent with random-looking activations before attention has focused (Tuned-Lens 2303.08112).  No layer satisfies the stricter copy-collapse rule; the model resists echoing the prompt despite many prompt tokens appearing in top-k.

Semantic meaning surfaces abruptly: "Berlin" first appears as top-1 at layer 44 and stays dominant, with probability climbing from 1 % to 56 % by the output layer.  Rest-mass remains > 0.9 through most of the stack and is still 0.97 at L 44, falling to 0.17 only in the final projection – evidence that the norm-lens under-resolves the tail distribution.

Negative control "Berlin is the capital of" shows no leakage of the answer; top-5 are "Germany", "the", "which", "what", "Europe" (p = 0.85, 0.05, 0.027, 0.011, 0.006) [lines 8-22], supporting that semantics are tied to the country cue rather than the word "Berlin".

Important-word trajectory from `records.csv` confirms that "Germany" and "capital" ascend in probability from mid-stack (≈ layer 30) while "Berlin" only overtakes them at layer 44, mirroring the classic filler → country → city pathway noted in logit-lens studies.  The absence of the one-word instruction in alternative prompts delays but does not remove the semantic collapse: in "Germany has its capital at the city called simply" the model already gives p = 0.55 for Berlin [line 64], suggesting the instruction mainly sharpens rather than creates the representation.

Temperature robustness: at T = 0.1 the model is almost deterministic (p(Berlin)=0.9999995, entropy ≈ 1.1 × 10⁻⁵ bits [line 671]), whereas at T = 2.0 p(Berlin)=0.048 and entropy jumps to 12.5 bits [line 686].

Checklist:  
✓ RMS lens applied  
✓ LayerNorm bias removal n.a. (RMS model)  
✗ Entropy rise at un-embed (entropy falls sharply)  
✗ FP32 un-embed promoted  
✓ Punctuation / markup anchoring (layers 39–43 dominated by quotes)  
✗ Copy-reflex  
✗ Grammatical filler anchoring (low-level fillers never top-1 before L 39)

## 5. Limitations & data quirks  
The very high rest-mass (> 0.9) until late layers implies that the 20-token cut-off underestimates mass concentrated in the tail, making per-layer entropies upper bounds.  The norm-lens may still be mis-scaled for Yi-34B, and lack of FP32 promotion can blur logit gaps.  Slight entropy oscillations after L 44 hint at precision loss in mixed-fp16 weights.

## 6. Model fingerprint  
Yi-34B: first semantic collapse at L 44, final entropy 3 bits; Berlin reaches 56 % in the last layer.

---
Produced by OpenAI o3

