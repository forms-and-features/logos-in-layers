## 1. Overview
Qwen 3-8B (≈8 B parameters) was probed on 2025-06-29 using the layer-by-layer RMS-lens script.  The run collects residual-stream predictions at the embedding layer and after every transformer block, extracting entropy and top-k tokens plus flags for copy- and semantic-collapse.

## 2. Method sanity-check
The JSON diagnostics confirm that the probe applied the intended **RMS lens** and worked on RMS-normed residuals:
> "use_norm_lens": true [L7-8]  
> "first_block_ln1_type": "RMSNormPre" [L9-10]

The `context_prompt` ends exactly with "called simply" (no trailing space) as required:
> "context_prompt": "…called simply" [L14-16]

`L_copy`, `L_semantic`, and `delta_layers` fields are present in the diagnostics block:
> "L_copy": 25, "L_semantic": 31, "delta_layers": 6 [L17-19]

Layers 0–3 top-1 tokens are "CLICK", "湾", "湾", "湾"; none are "called"/"simply", so early copy-reflex is **not** triggered.

## 3. Quantitative findings
| Layer summary |
| --- |
| L 0 – entropy 7.84 bits, top-1 'CLICK' |
| L 1 – entropy 8.53 bits, top-1 '湾' |
| L 2 – entropy 8.59 bits, top-1 '湾' |
| L 3 – entropy 8.82 bits, top-1 '湾' |
| L 4 – entropy 7.65 bits, top-1 '湾' |
| L 5 – entropy 11.19 bits, top-1 '院子' |
| L 6 – entropy 9.57 bits, top-1 '-minded' |
| L 7 – entropy 8.74 bits, top-1 'mente' |
| L 8 – entropy 7.91 bits, top-1 'tion' |
| L 9 – entropy 7.69 bits, top-1 'ifiable' |
| L 10 – entropy 4.50 bits, top-1 'ifiable' |
| L 11 – entropy 2.76 bits, top-1 'ifiable' |
| L 12 – entropy 3.31 bits, top-1 'ifiable' |
| L 13 – entropy 4.01 bits, top-1 'ifiable' |
| L 14 – entropy 6.21 bits, top-1 'ifiable' |
| L 15 – entropy 4.97 bits, top-1 'ifiable' |
| L 16 – entropy 4.07 bits, top-1 'name' |
| L 17 – entropy 2.25 bits, top-1 'name' |
| L 18 – entropy 2.92 bits, top-1 'name' |
| L 19 – entropy 3.94 bits, top-1 'names' |
| L 20 – entropy 4.06 bits, top-1 '这个名字' |
| L 21 – entropy 1.46 bits, top-1 '这个名字' |
| L 22 – entropy 3.20 bits, top-1 ' ______' |
| L 23 – entropy 2.32 bits, top-1 ' ______' |
| L 24 – entropy 1.15 bits, top-1 '这个名字' |
| **L 25 – entropy 0.08 bits, top-1 'simply'** |
| L 26 – entropy 2.91 bits, top-1 ' ______' |
| L 27 – entropy 2.40 bits, top-1 ' "' |
| L 28 – entropy 1.65 bits, top-1 ' "' |
| L 29 – entropy 1.07 bits, top-1 'Germany' |
| L 30 – entropy 0.88 bits, top-1 'Germany' |
| **L 31 – entropy 0.35 bits, top-1 'Berlin'** |
| L 32 – entropy 1.52 bits, top-1 'Berlin' |
| L 33 – entropy 0.65 bits, top-1 'Berlin' |
| L 34 – entropy 0.08 bits, top-1 'Berlin' |
| L 35 – entropy 1.64 bits, top-1 'Berlin' |
| L 36 – entropy 3.12 bits, top-1 'Berlin' |

Δ = L_semantic − L_copy = **6** layers.

## 4. Qualitative patterns & anomalies
The model exhibits a **long copy-reflex** (Δ 6 layers): by layer 25 the network is already certain (p ≈ 0.99, entropy 0.08 bits) that the next token is the prompt word "simply", yet six additional layers are needed before "Berlin" overtakes with 94 % probability at layer 31 > "... 'Berlin', 0.938" [CSV L32].  Such a gap echoes "Tuned-Lens" findings that semantic signals surface later than surface-form echoes (2303.08112).

Entropy collapses sharply from ≈7 bits (layer 0) to <1 bit at L25, then rises again when switching from copy to semantics, peaking modestly (1.52 bits) at L32 before settling near 0.08 bits at L34, consistent with information-overwriting dynamics reported in "Vision Decoding" (2212.10554).

Test prompts show robust directional knowledge: for the reversed relation prompt "Berlin is the capital of" the top-1 is "Germany" with ≈0.73 probability > "Germany", 0.728 [L46-52], indicating that semantic recall is symmetric.  Removing the "one-word" instruction (prompt "Germany has its capital at") does **not** shift the semantic-collapse index: the model still outputs "Berlin" confidently (0.84) at final layer, suggesting instruction tokens mainly affect early copy stages, not deep semantics.

Temperature sweep reinforces this: at τ = 0.1 the distribution is almost deterministic (>99 % Berlin, entropy 0.01 bits) whereas at τ = 2.0 entropy balloons to 13.4 bits with Berlin only 4 %.  This wide dynamic range confirms calibrated logits downstream of the RMS lens.

Checklist:  
✓ RMS lens  
✓ LayerNorm bias removed  
✗ Punctuation anchoring  
✓ Entropy rise at unembed  
✓ Punctuation / markup anchoring  
✓ Copy reflex  
✗ Grammatical filler anchoring (layers 0–5 dominated by non-word tokens rather than {is,the,a,of}).

## 5. Tentative implications for Realism ↔ Nominalism
1. Does the six-layer gap imply a dedicated sub-stack that cleans surface echoes before abstractive recall, supporting a realist "latent concept" interpretation?
2. Could the persistence of German-centered tokens between L29–30 indicate nominalist recycling of earlier cue words rather than true concept formation?
3. Would pruning layers 25–30 degrade factual recall more than text coherence, hinting at separable pathways?
4. How does the late semantic consolidation interact with temperature-controlled entropy, and might this reflect hierarchical logit-lens calibration rather than genuine knowledge emergence?

## 6. Limitations & data quirks
The early layers emit rare Chinese and subword fragments ("湾", "院子"), likely tokenizer quirks that skew entropy downward.  Several mid-stack layers focus on placeholder strings ("______") or quotation marks, complicating copy-collapse detection.  The probe runs on CPU, slowing throughput and possibly affecting RNG-seed determinism.  File timestamps proxy run date; no explicit seed or commit hash recorded.

## 7. Model fingerprint
Qwen-3-8B: copy-collapse at L 25; semantic answer at L 31; final entropy 3.1 bits; late-stack Berlin logits dominate.

---
Produced by OpenAI o3
