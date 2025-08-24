# Evaluation Report: Qwen/Qwen3-14B

*Run executed on: 2025-07-05 16:29:56*

# 1. Overview
Qwen/Qwen3-14B (≈ 14 B params) was probed on 2025-07-05 using the layer-by-layer RMS-lens script.  The run captures per-layer entropy and top-1 predictions for the pure next-token as well as diagnostics, test-prompt probes and temperature sweeps.

# 2. Method sanity-check
Diagnostics confirm the intended setup: RMS lens is active (`"use_norm_lens": true`) with rotary token-only positional encoding (`"layer0_position_info": "token_only_rotary_model"`) and first pre-norm layer is RMSNorm (`"first_block_ln1_type": "RMSNorm"`).  The context prompt in JSON ends exactly with "called simply" (no trailing space).

Implementation flags are present: `L_copy`, `L_copy_H`, `L_semantic`, `delta_layers`, `unembed_dtype` and others are all populated in the `diagnostics` block.

Copy-collapse flag scan of the pure-next-token CSV shows *no* row with `copy_collapse = True` in layers 0-3 – e.g. layer 0: `copy_collapse = False`  [line 2] – therefore the copy-reflex detector did **not** fire (✓ rule satisfied).

# 3. Quantitative findings
| Layer | Entropy (bits) | Top-1 token |
|-------|---------------:|-------------|
|L 0 | 17.213 | 梳 |
|L 1 | 17.199 | 线条 |
|L 2 | 16.834 | 几十年 |
|L 3 | 16.329 | 几十年 |
|L 4 | 16.497 | 几十年 |
|L 5 | 16.362 | 几十年 |
|L 6 | 16.737 | 过去的 |
|L 7 | 16.548 | 几十年 |
|L 8 | 16.823 | 不再 |
|L 9 | 17.103 | 让人们 |
|L 10| 17.134 | 候 |
|L 11| 17.114 | 时代的 |
|L 12| 17.089 | esor |
|L 13| 17.080 | simply |
|L 14| 17.084 | 迳 |
|L 15| 17.087 | hausen |
|L 16| 17.096 |  |
|L 17| 17.106 | urname |
|L 18| 17.106 | SimpleName |
|L 19| 17.097 | tics |
|L 20| 17.088 | SimpleName |
|L 21| 17.098 | lobs |
|L 22| 17.091 | Ticker |
|L 23| 17.084 | -minded |
|L 24| 17.086 | -minded |
|L 25| 17.070 | urname |
|L 26| 17.038 | 这个名字 |
|L 27| 16.986 | 这个名字 |
|L 28| 16.921 | 这个名字 |
|L 29| 16.846 | tons |
|L 30| 16.772 | 这个名字 |
|L 31| 16.655 | 这个名字 |
|**L 32**| **16.486** | **Berlin** |
|L 33| 15.839 | Berlin |
|L 34| 14.982 | Berlin |
|L 35| 11.874 | Berlin |
|L 36| 12.064 | Berlin |
|L 37| 9.178  | Berlin |
|L 38| 12.273 | Berlin |
|L 39| 10.295 | Berlin |
|L 40| 3.581  | Berlin |

ΔH (bits) = n/a (no copy-collapse layer) – entropy drops 12.63 bits between L 32 and final layer.

Confidence milestones: p > 0.30 first achieved at layer 37; p > 0.60 never reached; final-layer p = 0.347.

# 4. Qualitative patterns & anomalies
The RMS lens reveals a long plateau of high entropy (≈ 17 bits) and low-confidence, non-semantic tokens through layer 31.  The answer "Berlin" first appears as top-1 at L 32 (p ≈ 0.0026) — "Berlin rank 1 (p = 0.0026)" [line 34] — and probability climbs gradually reaching 0.366 at L 37 then 0.347 at the unembed, mirroring the "semantic sharpening" reported in Tuned-Lens 2303.08112.

Negative control shows no leakage: for "Berlin is the capital of" the model answers "Germany" with 63 % followed by "which" (25 %) and no Berlin in top-5  [JSON lines 11-17].

Records CSV confirms important words surge: "Germany" and "capital" appear in top-k from early layers, but "Berlin" only enters any top-5 around layer 30 and dominates after L 33; "capital" drops out by layer 35.

Rest-mass drops from 0.992 at L 32 to 0.236 at L 40, indicating the top-20 list captures most probability only after semantic convergence; no precision-loss spikes observed.

Temperature robustness: at T = 0.1 Berlin rank 1 with p = 0.974 (entropy 0.17 bits); at T = 2.0 rank 1 persists but with p = 0.036 (entropy 13.16 bits), confirming answer stability under sampling though uncertainty grows.

Checklist:  
✓ RMS lens ✓ LayerNorm bias n/a (RMS) ✓ Entropy rise controlled ✗ FP32 un-embed (flag false) ✗ No punctuation anchoring ✗ Copy-reflex ✗ Filler anchoring.

# 5. Limitations & data quirks
The model never enters copy-collapse so ΔH cannot be computed; early layers produce Chinese substrings unrelated to the prompt, hinting at tokenizer mismatch noise.  Rest-mass remains > 0.23 even at final layer, so entropy is still dispersed across the large 151 k-token vocabulary.

# 6. Model fingerprint
Qwen-3-14B: semantic collapse at L 32; entropy falls from 16.5 → 3.6 bits; confidence tops at p ≈ 0.37.

---
Produced by OpenAI o3

