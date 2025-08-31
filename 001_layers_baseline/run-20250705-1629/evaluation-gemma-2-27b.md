# Evaluation Report: google/gemma-2-27b

*Run executed on: 2025-07-05 16:29:56*

## 1. Overview  
Google's Gemma-2 · 27 B (46-layer, rotary-RMS stack) was probed on 2025-07-05 using a norm-lens script that records layer-by-layer logits and entropies while preserving the model's native RMS normalisation.  The probe focuses on the single-token continuation after the prompt "Give the city name only, plain text. The capital of Germany is called simply".  The attached JSON provides run metadata and diagnostics, while the two CSVs give the full per-layer logit lens trace.

## 2. Method sanity-check  
Diagnostics confirm that the run applied the intended RMS-norm lens and left the unembedding in bf16:  
> "use_norm_lens": true, … "unembed_dtype": "torch.bfloat16" [L806-808 in JSON]  
Positional handling is rotary-only, matching the model family:  
> "layer0_position_info": "token_only_rotary_model" [L815].  
The `context_prompt` ends exactly with "called simply" (no trailing space) [L3].  
Pure-next-token CSV marks `copy_collapse = True` in layers 0-8 (e.g. layer 0: p₁ ≈ 0.99998 for "simply", p₂ ≈ 7 × 10⁻⁶) showing an early copy reflex:  
`0, …, simply,0.999977…, merely,6.9e-06, …,True,…` [row 2 in CSV] – criterion ✓ satisfied.  
Diagnostics block contains `L_copy = 0`, `L_copy_H = 0`, `L_semantic = 36`, `delta_layers = 36`, alongside all implementation flags, so the script's bookkeeping is intact.

## 3. Quantitative findings  
| Layer | Entropy (bits) | Top-1 token |
|------:|--------------:|-------------|
| 0 | 0.0005 | simply |
| 1 | 0.9891 | the |
| 2 | 0.0003 | simply |
| 3 | 0.0081 | simply |
| 4 | 0.7790 | simply |
| 5 | 0.0016 | simply |
| 6 | 0.2358 | simply |
| 7 | 0.2145 | simply |
| 8 | 3.3899 | simply |
| 9 | 1.6736 | simply |
| 10 | 14.3030 | simply |
| 11 | 17.0622 | simply |
| 12 | 6.4475 | ſſel |
| 13 | 17.0337 | plain |
| 14 | 17.6165 | plain |
| 15 | 17.5781 | civilian |
| 16 | 17.7926 | civilian |
| 17 | 17.5941 | Wikimedia |
| 18 | 17.6323 | juges |
| 19 | 17.8084 | Bones |
| 20 | 17.8370 | contemporáneo |
| 21 | 17.8798 | مقاومت |
| 22 | 17.9083 | movi |
| 23 | 17.9092 | médic |
| 24 | 17.8292 | malades |
| 25 | 17.7843 | plain |
| 26 | 17.6055 | plain |
| 27 | 17.6866 | plain |
| 28 | 17.5683 | plain |
| 29 | 17.5632 | enough |
| 30 | 17.5995 | without |
| 31 | 17.5310 | enough |
| 32 | 17.6122 | just |
| 33 | 17.6159 | just |
| 34 | 17.6232 | just |
| 35 | 17.6641 | just |
| **36** | **17.5549** | **Berlin** |
| 37 | 17.5282 | Berlin |
| 38 | 17.4271 | Berlin |
| 39 | 17.5404 | Berlin |
| 40 | 17.5782 | Berlin |
| 41 | 16.8781 | Berlin |
| 42 | 17.5092 | Berlin |
| 43 | 17.8752 | Berlin |
| 44 | 17.2790 | Berlin |
| 45 | 17.0417 | """ |
| 46 | 0.1301 | Berlin |

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = 0.0005 − 17.55 ≈ −17.55 bits  
Confidence milestones: p("Berlin") > 0.30 at layer 46, > 0.60 also at 46; final-layer p = 0.982.

## 4. Qualitative patterns & anomalies  
The first eight layers echo the prompt wholesale: "simply" dominates with p > 0.97 (layer 0) and the copy-collapse flags remain True until layer 8, confirming a strong copy reflex ✓.  Entropy jumps from sub-bit values (0.00–0.24 bits) to >14 bits by layer 10, mirroring the "entropy spike" reported in Tuned-Lens 2303.08112, then plateaus around 17–18 bits for most of the stack.

Negative control "Berlin is the capital of" yields top-5  
> " Germany" 0.86, " the" 0.07, " and" 0.007, " a" 0.006, " Europe" 0.006 [lines 12-20 in JSON] – Berlin is absent, so no semantic leakage.

Important-word trajectory from records CSV shows "Germany" and "capital" tokens receive high probability already by layer 4 but decay after layer 15, while "Berlin" first cracks the per-token top-5 at layer 32 and becomes dominant by layer 36 (p ≈ 0.0014) before exploding to 0.98 in the final layer.  This late surge echoes the delta-layers = 36 score and suggests a deep semantic route rather than surface copying.

Removing the one-word instruction ("Give the city name...") hardly changes the collapse index: alternate test prompts ("Germany's capital city is called simply") still place Berlin top-1 with p ≈ 0.54 (entropy 2.77 bits) [JSON L60-70], indicating that the semantic circuit is robust to the framing but not earlier-activated.

Rest-mass sanity: rest_mass stays <10⁻³ through layer 7, spikes to 0.10 at layer 8 and 0.78 at layer 10, then hovers ≈0.99 for mid-stack layers, finally collapsing to 6.8 × 10⁻⁸ at layer 46.  The mid-stack spike suggests the 20-token cut-off under-captures the long-tail as logits spread, not a precision loss.

Temperature robustness: at T = 0.1 Berlin rank 1 (p = 0.88, entropy 0.53 bits); at T = 2.0 Berlin rank 1 but with p = 0.050 (entropy 12.57 bits).  Entropy rises by ~12 bits, showing the usual softening yet preservation of rank order.

Checklist  
• RMS lens? ✓  
• LayerNorm bias removed? n.a. (RMS model)  
• Entropy rise at unembed? ✓ (0.53 → 12.6 bits with temperature)  
• FP32 un-embed promoted? ✗ (`use_fp32_unembed = false`)  
• Punctuation / markup anchoring? ✗  
• Copy-reflex? ✓  
• Grammatical filler anchoring? ✓ (layers 1-4 top-1 = "the"/"simply")

## 5. Limitations & data quirks  
The large rest_mass (>0.7) across layers 10-35 means the top-20 slice covers <30 % of probability mass, so entropy numbers there are governed by the tail and should be interpreted cautiously.  Weird unicode tokens ("ſſel", Arabic "مقاومت") appear mid-stack, likely artefacts of high-entropy sampling rather than genuine hypotheses.  Final layer re-introduces triple-quote punctuation as top-1 before the fp16-unembed correction, hinting at mild quantisation noise.

## 6. Model fingerprint  
Gemma-2-27B: copy collapse at L 0, semantic collapse at **L 36**, final entropy 0.13 bits; answer probability skyrockets from 0.14 % to 98 % in the last ten layers.

---
Produced by OpenAI o3

