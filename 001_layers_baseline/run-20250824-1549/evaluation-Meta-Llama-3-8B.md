# Evaluation Report: meta-llama/Meta-Llama-3-8B

*Run executed on: 2025-08-24 15:49:44*
**1. Overview**

- Model: meta-llama/Meta-Llama-3-8B (8B); run date: 2025-08-24.
- Probe captures layerwise next-token distributions with a norm lens, entropy in bits, top-k, rest_mass, and collapse flags; semantic target is “Berlin”.

**2. Method Sanity-check**

JSON diagnostics confirm the norm lens and positional handling: “layer0_norm_fix”: “using_real_ln1_on_embeddings” [L814], “layer0_position_info”: “token_only_rotary_model” [L816]. The context_prompt ends with “called simply” and has no trailing space [L4]. Diagnostics include “use_norm_lens”, “use_fp32_unembed”, “unembed_dtype”, and the collapse indices: “L_copy”, “L_copy_H”, “L_semantic”, “delta_layers” [L807–L821].

Copy-collapse flag check: no rows with copy_collapse = True found in the pure-next-token CSV; thus no valid first row to report (✓ rule satisfied — no spurious fires). Early-layer reflex check: layers 0–3 have copy_collapse = False throughout (pure CSV).

**3. Quantitative Findings**

- L 0 – entropy 16.957 bits, top-1 'itzer'
- L 1 – entropy 16.942 bits, top-1 'mente'
- L 2 – entropy 16.876 bits, top-1 'mente'
- L 3 – entropy 16.894 bits, top-1 'tones'
- L 4 – entropy 16.899 bits, top-1 'interp'
- L 5 – entropy 16.873 bits, top-1 '�'
- L 6 – entropy 16.880 bits, top-1 'tons'
- L 7 – entropy 16.881 bits, top-1 'Exited'
- L 8 – entropy 16.862 bits, top-1 'надлеж'
- L 9 – entropy 16.867 bits, top-1 'biased'
- L 10 – entropy 16.851 bits, top-1 'tons'
- L 11 – entropy 16.854 bits, top-1 'tons'
- L 12 – entropy 16.877 bits, top-1 'LEGAL'
- L 13 – entropy 16.843 bits, top-1 'macros'
- L 14 – entropy 16.835 bits, top-1 'tons'
- L 15 – entropy 16.847 bits, top-1 ' simply'
- L 16 – entropy 16.847 bits, top-1 ' simply'
- L 17 – entropy 16.848 bits, top-1 ' simply'
- L 18 – entropy 16.839 bits, top-1 ' simply'
- L 19 – entropy 16.840 bits, top-1 ' ''
- L 20 – entropy 16.830 bits, top-1 ' ''
- L 21 – entropy 16.834 bits, top-1 ' ''
- L 22 – entropy 16.826 bits, top-1 'tons'
- L 23 – entropy 16.828 bits, top-1 'tons'
- L 24 – entropy 16.830 bits, top-1 ' capital'
- **L 25 – entropy 16.814 bits, top-1 ' Berlin'**
- L 26 – entropy 16.828 bits, top-1 ' Berlin'
- L 27 – entropy 16.819 bits, top-1 ' Berlin'
- L 28 – entropy 16.819 bits, top-1 ' Berlin'
- L 29 – entropy 16.799 bits, top-1 ' Berlin'
- L 30 – entropy 16.795 bits, top-1 ' Berlin'
- L 31 – entropy 16.838 bits, top-1 ':'
- L 32 – entropy 2.961 bits, top-1 ' Berlin'

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (no L_copy)

Confidence milestones:  
p > 0.30 at layer 32,  p > 0.60 at layer n.a.,  final-layer p = 0.5202

Quotes for grounding: “(layer 25, token = ‘Berlin’, p = 0.00013, rest_mass = 0.99955)” [row 29 in pure CSV]; “(layer 32, token = ‘Berlin’, p = 0.5202, entropy = 2.961 bits)” [row 36 in pure CSV].

**4. Qualitative Patterns & Anomalies**

Semantic emergence is late and sharp. “Berlin” first becomes top-1 at L25 with very low confidence (p ≈ 1.3e−4) and remains top-1 through L30, with a final jump in confidence at L32 (p = 0.520) where entropy collapses to 2.96 bits [row 36 in pure CSV]. This is consistent with the tuned-lens literature on late-stage semantic linearization (e.g., Tuned-Lens 2303.08112) but here the logit gap only becomes decisive at the final layer.

Negative control shows correct country continuation: for “Berlin is the capital of”, top-5 = [“ Germany”, 0.8955], [“ the”, 0.0525], [“ and”, 0.0075], [“ germany”, 0.0034], [“ modern”, 0.0030] [L14–L32]. Semantic leakage is present: Berlin rank 6 (p = 0.0029) [L34–L36].

Important-word trajectory (records.csv): at the “ simply” position, “Berlin” first enters any top-5 at L25 and stabilises from that layer onward; “Germany” appears in the same top-5 intermittently and last occurs by L27. No top-5 “Berlin” appears at the earlier “ capital” or “ Germany” positions. Example: “… (‘Berlin’, …) stabilises by layer 25” [simply-pos analysis].

Instruction sensitivity: prompts without the “one-word” instruction still yield strong final probabilities for “Berlin” (e.g., “Germany’s capital city is called” → p = 0.7524 [L249–L251]), but test prompts are final-layer probes only; no layerwise shift of collapse index can be inferred (n.a.).

Rest-mass sanity: rest_mass declines sharply only at the end; max after L_semantic = 0.99955 at layer 25, then drops to 0.1625 by the final layer [rows 29, 36 in pure CSV]. This pattern is consistent with a very diffuse distribution mid-stack that concentrates only at the head.

Temperature robustness: At T = 0.1, Berlin rank 1 (p = 0.999965; entropy 0.00057 bits). At T = 2.0, Berlin rank 1 (p = 0.0366; entropy 13.87 bits). The target stays ranked first across temperatures, while entropy rises with temperature (JSON temperature_exploration).

Checklist:
- RMS lens?: ✓ (RMSNorm; “use_norm_lens”: true) [L807]
- LayerNorm bias removed?: n.a. (“not_needed_rms_model” in diagnostics)
- Entropy rise at unembed?: n.a.
- FP32 un-embed promoted?: ✗ (use_fp32_unembed=false; unembed_dtype=torch.float32) [L808–L809]
- Punctuation / markup anchoring?: ✓ (layers 19–21 top-1 are quotes/punctuation in pure CSV)
- Copy-reflex?: ✗ (no layers 0–3 with copy_collapse=True in pure CSV)
- Grammatical filler anchoring?: ✗ (no {is,the,a,of} as top-1 in layers 0–5)

**5. Limitations & Data Quirks**

- Very high rest_mass persists even after L_semantic (0.9996 at L25), implying that “top-20” excludes most mass until the final layer; interpretations before L32 should be cautious.
- CSV rows contain embedded newlines in tokens; line-numbering uses file row numbers found via regex (reported above).

**6. Model Fingerprint**

“Llama‑3‑8B: collapse at L 25; final entropy 2.96 bits; ‘Berlin’ only crosses p>0.30 at the final layer.”

---
Produced by OpenAI GPT-5 
