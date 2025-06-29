# 1. Overview
Qwen3-8B (≈8 B parameters) was probed on 2025-06-29 using the layer-by-layer RMS-lens script.  The sweep captures per-layer entropy and top-k lexical predictions for the first *unseen* token following the prompt "The capital of Germany is …", plus auxiliary probes for paraphrased prompts and temperature sweeps.

# 2. Method sanity-check
`output-Qwen3-8B.json` confirms that the run used the intended normalised lens and respected the model's rotary positional encoding.  The diagnostics block shows
> "use_norm_lens": true [L5]  
> "layer0_position_info": "token_only_rotary_model" [L17]
Both CSVs contain only a single position (`pos = 5`), indicating that predictions were indeed collected for the next token rather than for teacher-forced tokens.

# 3. Quantitative findings
| Layer | Entropy (bits) | Top-1 token |
|-------|----------------|-------------|
| L 0 | 2.06 | "墈" |
| L 1 | 13.10 | "单职业" |
| L 2 | 14.20 | " Liberties" |
| L 3 | 14.88 | "总队" |
| L 4 | 15.18 | "arring" |
| L 5 | 15.68 | "arring" |
| L 6 | 15.45 | "arring" |
| L 7 | 13.21 | "less" |
| L 8 | 12.16 | "less" |
| L 9 | 11.44 | "哈尔" |
| L 10 | 11.71 | "直辖" |
| L 11 | 10.27 | "呃" |
| L 12 | 10.77 | "呃" |
| L 13 | 11.47 | "呃" |
| L 14 | 10.69 | "_assets" |
| L 15 | 11.15 | "呃" |
| L 16 | 10.11 | "呃" |
| L 17 | 10.59 | "呃" |
| L 18 | 10.22 | "请选择" |
| L 19 | 9.21 | "呃" |
| L 20 | 8.53 | "一头" |
| L 21 | 3.90 | "eway" |
| L 22 | 8.91 | "ㅤ" |
| L 23 | 4.76 | " ____" |
| L 24 | 3.16 | " ____" |
| L 25 | 2.51 | " ____" |
| L 26 | 1.68 | " ____" |
| L 27 | 1.55 | " ____" |
| L 28 | 1.56 | " ____" |
| L 29 | 2.06 | " ____" |
| L 30 | 1.54 | " ____" |
| L 31 | 2.53 | " ____" |
| L 32 | 2.39 | " Berlin" |
| L 33 | 1.85 | " Berlin" |
| L 34 | 3.90 | " Berlin" |
| L 35 | 5.01 | " blank" |
| L 36 | 3.67 | "．" |

No layer attains < 1 bit; therefore **no collapse layer is bolded**.

# 4. Qualitative patterns & anomalies
The probe uncovers three broad regimes.  Early layers (0–10) oscillate around unrelated Chinese or English morphemes with entropy > 10 bits, suggesting token-level noise prior to consolidation.  Middle layers (11–20) gradually narrow the distribution yet remain trapped in filler tokens such as "呃" or "一头".  A sharp entropy drop at L 21 (3.9 bits) coincides with the first high-probability English sub-word "eway", but the effect rebounds, implying transient alignment rather than stable collapse.  Only from L 32 onward does "Berlin" dominate (> 0.3 prob) though entropy never falls below 1 bit, indicating partial but incomplete convergence.

The final prediction shows the model still prefers a full-width period over the correct answer:
> "．", 0.31; "�", 0.15 [L30-34]
This punctuation bias echoes the temperature exploration where at τ = 0.1 "Berlin" becomes deterministic (prob ≈ 1.0) while at τ = 2.0 it merely reaches 6 % [L320-338].  Paraphrased prompts yield mixed fidelity: "Germany has its capital at " returns 0.40 prob on "Berlin" with entropy 2.8 bits [L150-165], yet "The capital city of Germany is named " falls back to 0.18 prob with entropy 5.7 bits [L110-125].

Checklist:
✓ RMS lens  
✗ LayerNorm  
✗ Colon-spam  
✗ Entropy rise at unembed

# 5. Tentative implications for Realism ↔ Nominalism
1. Does the late but never-complete convergence on "Berlin" imply that the concept of *capital-city* is distributed across the upper third of the stack rather than localised to a single "naming" layer?

2. Could the persistent punctuation / underscore preference indicate nominal artefacts of the tokenizer that override semantic realism even in the unembedding space?

3. Might the rebound in entropy after L 21 suggest a realist encoding of multiple candidate answers that gets re-expanded by subsequent attention heads to maintain distributional diversity?

4. Does temperature-dependent determinism (τ = 0.1) versus ambiguity (τ = 2.0) support a nominalist view in which the semantic pointer is latent but not salient until logit sharpening?

# 6. Limitations & data quirks
The probe ran on CPU, increasing runtime and possibly affecting numerical precision.  No run timestamp is embedded, reducing reproducibility.  CSV rows show several non-ASCII fillers ("__", "____") whose semantics are unclear; they may be artefacts of the tokenizer rather than genuine model beliefs.  The entropy valley at L 21 is isolated and followed by rebound, hinting at hook mis-alignment rather than genuine information collapse.  Finally, the first five layers show extraordinarily high entropy (> 14 bits), beyond the theoretical maximum for a 151 k-vocab (≈ 17.2 bits), suggesting the FP32 unembedding trick may still under-sample tail probabilities.

# 7. Model fingerprint
"Qwen3-8B: no < 1 bit collapse; earliest Berlin appears at L 32; final entropy 3.7 bits, punctuation ('．') top-ranked."

---
Produced by OpenAI o3
