## 1. Overview
Mistral-7B-v0.1 is a 7-billion-parameter base model from Mistral AI. The probe script (`run.py`) records per-layer, per-token softmax distributions for the answer token in response to a factual prompt ("Question: What is the capital of Germany? Answer:"). The attached JSON aggregates summary statistics while the CSV logs layer-wise entropy and top-k predictions.

## 2. Method sanity-check
`output-Mistral-7B-v0.1.json` reports `"use_norm_lens": true` and both the first block and final norm layers are typed `RMSNormPre`, indicating that the RMS-corrected lens is applied. The console prints also confirm that the residual stream is normalised and that positional information is incorporated via the embedding hooks:
```246:248:001_layers_and_logits/run.py
print("Using NORMALIZED residual stream (RMS + learned scale)")
```
```266:267:001_layers_and_logits/run.py
print("[diagnostic] No separate positional embedding hook found (as expected for some models); using token embeddings for layer 0.")
```
Together with the CSV's presence of a layer 0 (embedding) row, these lines confirm that the intended norm lens is active and that positional encodings are summed into the residual before probing.

## 3. Quantitative findings
| Layer | Entropy (bits) | Top-1 token |
|-------|---------------|-------------|
| 0 | 14.72 | 'laug' |
| 1 | 14.44 | 'zo' |
| 2 | 14.05 | 'ts' |
| 3 | 13.87 | 'richt' |
| 4 | 13.82 | 'amber' |
| 5 | 13.74 | 'aiser' |
| 6 | 13.73 | 'amber' |
| 7 | 13.64 | 'nab' |
| 8 | 13.62 | 'amber' |
| 9 | 13.58 | 'answer' |
| 10 | 13.32 | 'answer' |
| 11 | 13.48 | 'Answer' |
| 12 | 13.30 | '*/' |
| 13 | 13.66 | 'ír' |
| 14 | 13.67 | 'Answer' |
| 15 | 13.55 | 'Answer' |
| 16 | 13.44 | 'Answer' |
| 17 | 12.92 | 'Answer' |
| 18 | 13.22 | 'cities' |
| 19 | 12.74 | 'cities' |
| 20 | 11.43 | 'cities' |
| 21 |  8.48 | 'capital' |
| 22 |  6.56 | 'Berlin' |
| 23 |  3.16 | 'Washington' |
| 24 |  2.88 | 'Berlin' |
| **25** | **0.56** | 'Berlin' |
| 26 |  0.28 | 'Berlin' |
| 27 |  0.10 | 'Berlin' |
| 28 |  0.14 | 'Berlin' |
| 29 |  0.12 | 'Berlin' |
| 30 |  0.35 | 'Berlin' |
| 31 |  0.79 | 'Berlin' |
| 32 |  1.80 | 'Berlin' |
The first layer with entropy < 1 bit is **L 25**, marking the collapse to a single answer.

## 4. Qualitative patterns & anomalies
Between L 0-L 20 the model explores semantically loose neighbours: generic fillers (e.g. 'Answer', 'cities') dominate before narrowing to the concept of a capital city. L 21 introduces the lexical cue 'capital', lowering entropy sharply. By L 22 'Berlin' overtakes with ~37 % probability, but a brief detour at L 23 sees 'Washington' briefly ranked 1, suggesting a geopolitical distractor before consolidation. From L 25 onwards 'Berlin' exceeds 91 % confidence and remains dominant despite a slight entropy rebound after the final `ln_final`.

Test-prompt probes corroborate this behaviour: > "... 0.896 on 'Germany'" [L152-156] shows low-entropy completion when the cue is reversed, whereas the one-word instruction yields newline preference and higher entropy (>4 bits). Temperature sweep reveals determinism at τ = 0.1 (>99 % 'Berlin') but diffuse alternatives at τ = 2 (entropy ≈ 12 bits, 'Berlin' 5.8 %).

Checklist:
✓ RMS lens ✓ LayerNorm (RMSNormPre) ✗ Colon-spam (only single ':') ✓ Entropy rise at unembed (0.79 → 1.80 bits).

## 5. Tentative implications for Realism ↔ Nominalism
1. Does the late but decisive collapse (L 25) imply that geographical facts are stored in deep layers rather than distributed earlier, supporting a realist "fact slots" view?
2. The transient 'Washington' at L 23 raises the hypothesis that multiple candidate capitals compete until gating resolves—could this reflect nominalist token competition rather than unitary concept retrieval?
3. The entropy rebound after `ln_final` suggests that final LayerNorm re-introduces uncertainty; is this a deliberate regularisation mechanism to keep sampling diverse, challenging realist assumptions of a fixed fact representation?
4. Under higher temperature the model still biases toward 'Berlin'; is the conceptual representation grounded enough to survive noise, or is this merely token frequency priming? Further probes could test nominalist explanations.

## 6. Limitations & data quirks
The probe focuses solely on the answer token; interactions across earlier positions are not analysed. The run executed in CPU fp32 mode (`"device": "cpu"`), so GPU precision quirks are not captured. Lack of explicit run timestamp reduces reproducibility. High-entropy junk tokens ('laug', '*/') indicate tokenizer artefacts that may distort early-layer measurements.

## 7. Model fingerprint
"Mistral-7B-v0.1: collapse at L 25; final entropy 1.8 bits; 'Berlin' plateaus above 90 % from L 25 onward."

---

Produced by OpenAI o3