# 1. Overview

Qwen3-8B (≈8 billion parameters) was probed on 2025-07-05.  The script records layer-wise logits, entropy and top-k tokens under an RMS-norm "norm-lens"; results are split into a compact JSON summary plus two CSVs containing per-layer and pure-next-token statistics.

# 2. Method sanity-check

Diagnostics confirm the intended setup: positional information is rotary only (`"layer0_position_info": "token_only_rotary_model"`) and the norm-lens is active with RMSNorm weights
> "use_norm_lens": true, "unembed_dtype": "torch.float16"  [L808-L809 in JSON]

The context prompt in JSON ends exactly with "called simply" (no trailing space).  The pure-next-token CSV shows no `copy_collapse = True` in layers 0–3, so the copy-reflex heuristic did **not** fire.  Diagnostics block contains `L_copy`, `L_copy_H`, `L_semantic = 31` and the expected implementation flags.

Copy-collapse flag check: first (and only) row with `copy_collapse = True`   — *none present* → ✓ rule **did not** fire spuriously.

# 3. Quantitative findings

| Layer | Entropy (bits) | Top-1 token |
|-------|----------------|-------------|
| L 0 – 17.21 bits, 'CLICK' |
| L 1 – 17.19 bits, 'обязан' |
| L 2 – 15.87 bits, 'hook' |
| L 3 – 13.65 bits, '对学生' |
| L 4 – 15.69 bits, '原因之一' |
| L 5 – 16.78 bits, '(PR' |
| L 6 – 17.01 bits, '呼和' |
| L 7 – 16.77 bits, '的方式来' |
| L 8 – 16.91 bits, '的方式来' |
| L 9 – 17.06 bits, '俗' |
| L 10 – 17.02 bits, '名字' |
| L 11 – 17.04 bits, '名字' |
| L 12 – 17.07 bits, '简单' |
| L 13 – 17.07 bits, '简单' |
| L 14 – 17.06 bits, 'ifiable' |
| L 15 – 17.06 bits, '名字' |
| L 16 – 17.06 bits, 'ifiable' |
| L 17 – 17.06 bits, '名称' |
| L 18 – 17.06 bits, '名称' |
| L 19 – 17.05 bits, '名称' |
| L 20 – 17.03 bits, 'flat' |
| L 21 – 17.01 bits, '这个名字' |
| L 22 – 16.93 bits, '______' |
| L 23 – 16.91 bits, '______' |
| L 24 – 16.84 bits, '这个名字' |
| L 25 – 16.76 bits, 'simply' |
| L 26 – 16.71 bits, 'simply' |
| L 27 – 16.60 bits, 'simply' |
| L 28 – 16.47 bits, 'simply' |
| L 29 – 15.76 bits, 'Germany' |
| L 30 – 14.46 bits, 'Germany' |
| **L 31 – 9.70 bits, 'Berlin'** |
| L 32 – 2.73 bits, 'Germany' |
| L 33 – 2.56 bits, 'Berlin' |
| L 34 – 4.86 bits, 'Berlin' |
| L 35 – 10.57 bits, 'Berlin' |
| L 36 – 3.13 bits, 'Berlin' |

ΔH (bits) = n.a. (no copy-collapse)

Confidence milestones: p > 0.30 at layer 33 (0.50), p > 0.60 at layer 34 (0.60), final-layer p = 0.43.

# 4. Qualitative patterns & anomalies

Early layers are dominated by unrelated multilingual fragments; rest-mass stays > 0.99 until L 28, indicating a very flat distribution and no token certainty.  Important words from the prompt ("Germany", "capital", "Berlin") enter the top-k only after L 25; "Germany" becomes top-1 at L 29, and **Berlin first appears as top-1 at L 31** > "... ('Berlin', 0.28)" [line 33 in CSV].

Negative control shows weak semantic leakage: for the reversed prompt "Berlin is the capital of" the model still lists Berlin 9-th (p ≈ 4.6 e-4) > "..., " Berlin", 0.00046" [JSON L15-L27], suggesting token rehearsal rather than full answer recall.

Rest-mass peaks at 0.54 exactly at L 31, then falls to 0.11 by L 33 and stabilises ≤ 0.24 thereafter, implying that the norm-lens scaling is conservative but not lossy.  Temperature robustness is sound: at T = 0.1 Berlin rank 1 (p = 0.999) with entropy 0.01 bits; at T = 2.0 Berlin rank 1 (p = 0.042) with entropy 13.39 bits → entropy rises over three orders of magnitude, as expected for softmax temperature.

The absence of any copy-collapse flag and the late semantic convergence (Δlayers undefined) suggest that this model dedicates deep layers to disambiguation rather than early lexical echo, consistent with findings in Tuned-Lens 2303.08112 for larger rotary-RMS stacks.  Punctuation/filler anchoring is mild: fillers like "simply" dominate L 25-28 but core grammatical cues ("is", "the") never top-1.  Overall the probe behaves consistently with an RMS-norm lens: rotary position handled early, semantic features accumulate late.

Checklist:  
✓ RMS lens  |  n.a. LayerNorm bias  |  ✓ Entropy cast before unembed  |  ✗ FP32 unembed  |  ✓ Punctuation / filler anchoring  |  ✗ Copy-reflex  |  ✗ Grammatical filler anchoring

# 5. Limitations & data quirks

• The unusually high rest-mass (0.54) at L 31 indicates probability mass spread across tail tokens, so entropy values pre- and post-semantic collapse may be under-resolved.  
• No copy-collapse detected; cannot compute ΔH.  
• Early multilingual noise suggests tokenizer artefacts that the "important-word" filter misses, possibly skewing verbosity detection.

# 6. Model fingerprint

Qwen3-8B: semantic collapse at L 31; final-layer entropy 3.1 bits; Berlin probability rises from 0.28 → 0.60 within two layers.

---
Produced by OpenAI o3

