# Evaluation – Qwen3-8B

## 1. Overview
Qwen 3-8B (≈ 8 B parameters, 36-layer Transformer) was probed on 2025-06-28 using the `001_layers_and_logits/run.py` script.  The probe records the entropy of the next-token distribution and the top-k tokens extracted from the residual stream at every layer, under a norm-lens that re-scales the residual by the layer's RMSNorm parameters.  Results are saved as structured JSON and per-layer CSV.

## 2. Method sanity-check
The JSON header confirms that the run used the intended norm-lens and captured positional embeddings:

> "use_norm_lens": true, "first_block_ln1_type": "RMSNormPre"  [L7-9:001_layers_and_logits/output-Qwen3-8B.json]
> Using NORMALIZED residual stream (RMS + learned scale)  [L246:001_layers_and_logits/run.py]

`hook_pos_embed` is present in the script, so the positional encoding stream is cached and included in layer 0.  Together, the console and JSON evidence indicate that both the positional embeddings and the RMS-based norm-lens were applied as designed.

## 3. Quantitative findings
| Layer | Entropy (bits) | Top-1 token |
|-------|----------------|-------------|
| L 0 | 5.67 | 'いらっ' |
| L 1 | 10.97 | 'ListViewItem' |
| L 2 | 10.00 | 'Buccane' |
| L 3 | 9.90 | 'Lauderdale' |
| L 4 | 11.34 | 'Buccane' |
| L 5 | 12.91 | '直接影响' |
| L 6 | 14.15 | '我省' |
| L 7 | 13.43 | 'portion' |
| L 8 | 12.07 | 'steller' |
| L 9 | 11.30 | 'Mus' |
| L 10 | 11.53 | '在游戏中' |
| L 11 | 12.15 | '在游戏中' |
| L 12 | 12.16 | 'Answer' |
| L 13 | 11.68 | 'Answer' |
| L 14 | 10.95 | 'Binary' |
| L 15 | 11.60 | 'Answer' |
| L 16 | 11.71 | 'Answer' |
| L 17 | 12.63 | 'Answer' |
| L 18 | 10.84 | 'Answer' |
| L 19 | 8.56 | 'Answer' |
| L 20 | 3.71 | 'Answer' |
| L 21 | 2.60 | 'Answer' |
| L 22 | 3.37 | 'Answer' |
| L 23 | 3.55 | 'Answer' |
| L 24 | 4.10 | '______' |
| L 25 | 1.41 | 'Germany' |
| L 26 | 2.86 | '____' |
| L 27 | 2.92 | 'Germany' |
| **L 28** | **0.43** | **'Berlin'** |
| L 29 | 0.19 | 'Berlin' |
| L 30 | 0.87 | 'Berlin' |
| L 31 | 0.011 | 'Berlin' |
| L 32 | 0.076 | 'Berlin' |
| L 33 | 0.011 | 'Berlin' |
| L 34 | 0.012 | 'Berlin' |
| L 35 | 0.13 | 'Berlin' |
| L 36 | 2.02 | 'Berlin' |

The entropy drops steadily after L 20 and **collapses below 1 bit at L 28**, where 'Berlin' becomes the confident prediction.

## 4. Qualitative patterns & anomalies
From L 0–L 19 the model's next-token guesses are dominated by rare English or multilingual junk tokens (e.g. 'Lauderdale', '直接影响'), suggesting that early residual features are noisy for this prompt.  Entropy then free-falls between L 19-L 21 (8.6 → 2.6 bits) before stabilising and ultimately collapsing at L 28 where 'Berlin' attains 0.93 probability and entropy 0.43 bits: "... 'Berlin', 0.93)" [L528].  The collapse persists through L 34 with sub-0.08 bit entropy, indicating a sharp phase change rather than gradual sharpening.

Test prompts reinforce the selectivity: for "Berlin is the capital of", 'Germany' is top-1 with 0.73 probability and entropy 1.20 bits [L45-60].  Temperature exploration shows extreme confidence at τ = 0.1 (entropy ≈ 0) and a still-dominant Berlin at τ = 2.0 (6.6 % mass), hinting at a high-margin logit gap.

An entropy rebound is visible after the final LN/unembed: entropy jumps from 0.13 bits (L 35) to 2.02 bits in the final logits (JSON `final_prediction`).  This echoes observations in the tuned-lens literature (2303.08112) that unembedding sometimes "de-sharpens" over-confident internal states.

Checklist
- RMS lens? ✓
- LayerNorm? ✗ (RMSNorm only)
- Colon-spam? ✗ (underscore tokens dominate instead)
- Entropy rise at unembed? ✓

## 5. Tentative implications for Realism ↔ Nominalism
1. Does the abrupt entropy collapse at L 28 imply that a discrete feature representing `<capital-city answer>` becomes linearly readable only after a long chain of composition?
2. Might the sustained sub-bit entropy in L 28-L 34 indicate that the model internally "decides" on 'Berlin' well before the final LN but then rescales logits for calibration, supporting a nominalist view of late-layer adjustment rather than new knowledge?
3. Could the entropy rebound at the unembed layer reflect an architectural regulariser that buffers over-confidence, suggesting a realism-compatible smoothing mechanism?
4. Does the symmetric performance on reverse prompts ("Berlin is the capital of ..." point to shared bidirectional representations, challenging a strictly causal realist stance?

## 6. Limitations & data quirks
All computations ran on CPU, extending runtime and potentially altering numerical noise profiles.  Early-layer top-k vocab is dominated by spurious multilingual fragments, hinting at tokenisation quirks or insufficient context length for the lens.  The CSV uses the ':' token as the answer slot; any mis-alignment would skew entropy values.  Finally, the probe records only the first 20 alternatives, so low-probability but semantically relevant tokens may be missing.

## 7. Model fingerprint
Qwen3-8B: collapse at L 28; final entropy ≈ 2.0 bits; 'Berlin' reaches > 99 % by L 31.

---

Produced by OpenAI o3