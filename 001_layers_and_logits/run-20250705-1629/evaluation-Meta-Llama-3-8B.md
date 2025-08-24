# Evaluation Report: meta-llama/Meta-Llama-3-8B

## 1. Overview
meta-llama/Meta-Llama-3-8B (≈ 8 B params) was probed on 2025-07-05 16:29. The script records layer-by-layer next-token distributions using a norm-lens over RMS-normed residuals and emits entropy, top-k and collapse flags for every layer. Two CSVs capture detailed per-layer logits; a compact JSON contains diagnostics and test-prompt probes.

## 2. Method sanity-check
The diagnostics block confirms the intended configuration: 
> "use_norm_lens": true, "unembed_dtype": "torch.float16" [line 758]  
> "layer0_position_info": "token_only_rotary_model" [line 768]

The `context_prompt` ends exactly with "called simply" (no trailing space). Implementation flags `L_copy`, `L_copy_H`, `L_semantic` and `delta_layers` are present (lines 772-777). In the pure-next-token CSV layers 0-3 all have `copy_collapse = False`, so no early copy-reflex ✓ to flag; first (and only) `is_answer = True` is at layer 25. Copy-collapse rule therefore did **not** fire spuriously.

## 3. Quantitative findings
| Layer | Entropy (bits) | Top-1 token |
|------:|---------------:|-------------|
| 0 | 16.955 | `itzer` |
| 1 | 16.958 | `mente` |
| 2 | 16.955 | `tics` |
| 3 | 16.947 | `Simply` |
| 4 | 16.942 | `aires` |
| 5 | 16.936 | `` |
| 6 | 16.932 | `rops` |
| 7 | 16.926 | ` bul` |
| 8 | 16.926 | ` bul` |
| 9 | 16.924 | `单` |
| 10 | 16.923 | `ully` |
| 11 | 16.919 | ` h` |
| 12 | 16.920 | `283` |
| 13 | 16.924 | ` Freed` |
| 14 | 16.919 | ` simply` |
| 15 | 16.913 | ` simply` |
| 16 | 16.909 | ` simply` |
| 17 | 16.905 | ` simply` |
| 18 | 16.901 | ` simply` |
| 19 | 16.900 | ` simply` |
| 20 | 16.897 | `'` |
| 21 | 16.894 | `'` |
| 22 | 16.886 | ` simply` |
| 23 | 16.892 | ` simply` |
| 24 | 16.886 | ` capital` |
| **25** | **16.879** | **`Berlin`** |
| 26 | 16.869 | `Berlin` |
| 27 | 16.868 | `Berlin` |
| 28 | 16.859 | `Berlin` |
| 29 | 16.843 | `Berlin` |
| 30 | 16.830 | `Berlin` |
| 31 | 16.754 | `"""` |
| 32 | 2.966 | `Berlin` |

ΔH (bits) = n.a. (no copy-collapse layer)

Confidence milestones: p > 0.30 at layer 32; p > 0.60 not reached; final-layer p = 0.518.

## 4. Qualitative patterns & anomalies
Entropy stays near 17 bits until layer 24, then slips slightly before a dramatic drop (2.97 bits) at the output head, typical of RMS-norm lens behaviour. Early layers cycle through filler or punctuation (e.g. token `' simply'` from layers 14-19, `' '` at 20-21), indicating grammatical anchoring. The negative-control prompt shows minimal leakage: > "… "Berlin", 0.0029)" [lines 13-18] where Berlin ranks 6, p≈0.003. Test prompts that include the "city-name" instruction, however, place Berlin at rank 1 with p ≈ 0.37–0.43 (e.g. "Germany's capital city is called simply" [lines 40-48]). Temperature robustness is good: at T = 0.1 Berlin dominates (p = 0.99997, entropy 0.0005 bits); at T = 2.0 Berlin drops to rank 1 but with p = 0.037 and entropy 13.87 bits.

Important-word trajectory: "Germany" and "capital" appear in the prompt-echo layers but Berlin only enters top-1 at layer 25 and remains through the stack. "Capital" peaks as top-1 at layer 24 before giving way to Berlin. Rest-mass declines gradually, staying > 0.99 through layer 24, 0.99967 at L_semantic, then falling to 0.163 at the final layer, indicating that probability mass concentrates into the listed top-k rather than numerical loss.

Checklist:  
✓ RMS lens  
✓ LayerNorm bias n.a./removed  
✓ Entropy rise at unembed  
✗ FP32 un-embed promoted  
✓ Punctuation / markup anchoring  
✗ Copy-reflex  
✗ Grammatical filler anchoring ('is/the/a/of' never dominate top-1)

## 5. Limitations & data quirks
Rest-mass remains > 0.16 even at the final layer, suggesting long tails beyond top-20 which may slightly distort entropy estimates. Sparse probabilities in early layers (<10⁻⁴) can be unstable under float16 un-embed, though no overflow observed. Absence of copy-collapse could stem from aggressive margin (0.05) rather than true behaviour.

## 6. Model fingerprint
Meta-Llama-3-8B: no copy-collapse, semantic shift at L 25→32, final entropy ≈ 3 bits; Berlin probability 0.52 at head.

---
Produced by OpenAI o3

