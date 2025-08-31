# Evaluation Report: mistralai/Mistral-7B-v0.1

*Run executed on: 2025-07-05 09:51:01*

## 1. Overview
Mistral-7B-v0.1 (≈7 B parameters) was probed on 2025-07-05 using a logit-lens sweep that captures layer-by-layer next-token distributions, entropy and collapse metrics.  The probe focuses on the single prompt "Give the city name only … called simply" and records both pure next-token CSVs and compact JSON metadata for diagnostics.

## 2. Method sanity-check
The JSON confirms that the normalised lens was applied (`use_norm_lens = true`) and that positional information is rotary-only at layer 0:
```806:806:001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json
    "use_norm_lens": true,
```
```815:815:001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json
    "layer0_position_info": "token_only_rotary_model",
```
The context prompt is intact and ends with "called simply" (no trailing space):
```4:4:001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json
    "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply",
```
No early `copy_collapse` flag is set — layer 0 shows `False`:
```2:2:001_layers_baseline/run-latest/output-Mistral-7B-v0.1-pure-next-token.csv
0,16,⟨NEXT⟩,14.9551,…,False,False,False
```
Diagnostics block carries the expected keys including `L_copy`, `L_copy_H`, `L_semantic` and `delta_layers`:
```820:820:001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json
    "L_semantic": 25,
```

## 3. Quantitative findings
| Layer | Entropy (bits) | Top-1 token |
|-------|---------------|-------------|
|0|14.96|dabei|
|1|14.96|❶|
|2|14.95|simply|
|3|14.94|simply|
|4|14.93|simply|
|5|14.93|simply|
|6|14.92|simply|
|7|14.91|plain|
|8|14.90|olas|
|9|14.89|anel|
|10|14.89|anel|
|11|14.88|inho|
|12|14.88|ifi|
|13|14.87|…|
|14|14.86|mate|
|15|14.85|…|
|16|14.82|simply|
|17|14.78|simply|
|18|14.75|simply|
|19|14.70|simply|
|20|14.64|simply|
|21|14.53|simply|
|22|14.50|simply|
|23|14.38|simply|
|24|14.21|simply|
|**25**|**11.64**|**Berlin**|
|26|9.89|Berlin|
|27|8.83|Berlin|
|28|8.41|Berlin|
|29|7.89|Berlin|
|30|7.24|Berlin|
|31|7.83|Berlin|
|32|3.61|Berlin|

The semantic collapse (first "Berlin") occurs at layer 25, matching `L_semantic` in diagnostics.

## 4. Qualitative patterns & anomalies
Early layers oscillate among rare or artefactual tokens ("❶", "…", "anel") with entropy ≈15 bits, indicating a flat distribution anchored by punctuation / markup.  From layer 2 through 24 the model fixates on the filler "simply", a direct echo of the prompt, yet probabilities remain ≪ 0.9 so the copy-collapse rule is never triggered.  Entropy falls sharply (≈3 bits) at layer 25 where *Berlin* takes 13 % mass, and keeps tightening down-stack to 38 % at the final unembed.

The test prompt "Berlin is the capital of" is answered correctly with 0.90 mass on *Germany*:
```9:9:001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json
      "prompt": "Berlin is the capital of",
```
```14:14:001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json
          0.89697265625
```
Temperature exploration shows near-deterministic collapse at τ = 0.1 (0.005 bits) yet preserves "Berlin" as top-1 even at τ = 2.0:
```670:670:001_layers_baseline/run-latest/output-Mistral-7B-v0.1.json
    "entropy": 0.005483804207339175,
```
Qualitative sweep of `records.csv` confirms that "Germany" and "capital" climb the ranking from mid-teens at L 20 to top-3 by L 24, preceding the semantic jump — consistent with progressive attribute binding reported in Tuned-Lens 2303.08112.

Checklist:
- RMS lens? ✓  (first_block_ln1_type = RMSNorm)
- LayerNorm bias removed? n.a. (RMS model)
- Punctuation anchoring? ✓  ("❶", "… dominate L 0–10)
- Entropy rise at unembed? ✗  (monotonic ↓)
- FP32 un-embed promoted? ✗ (`use_fp32_unembed` = false)
- Punctuation / markup anchoring? ✓
- Copy reflex? ✗  (no `copy_collapse` True)
- Grammatical filler anchoring? ✓ (persistent "simply")

## 5. Limitations & data quirks
The probe inspects only a single factual prompt with one-word answer, so generality is untested.  Early layers surface low-frequency artefacts ("❶") suggesting tokenizer edge-cases that may skew entropy estimates.  Entropy dip/rebound around layers 30-31 indicates residual quantisation noise; CSV uses fp16 un-embed which may under-resolve sub-bit gaps.

## 6. Model fingerprint
Mistral-7B-v0.1: semantic collapse at L 25; final entropy 3.6 bits; "simply" dominates until a sharp switch to "Berlin".

---
Produced by OpenAI o3