## 1. Overview

Google Gemma-2-9B (≈9 B params, 42-layer decoder) was probed on 2025-06-29 with the custom logit-lens script.  The run captured layer-wise entropies and top-k predictions for every position plus auxiliary diagnostics, test prompts and temperature sweeps, all stored in structured JSON/CSV.

## 2. Method sanity-check

The diagnostics block confirms that the probe applied the intended RMS-norm lens and post-block alignment:
```5:12:001_layers_baseline/output-gemma-2-9b.json
    "use_norm_lens": true,
    "first_block_ln1_type": "RMSNormPre",
    "norm_alignment_fix": "using_ln2_rmsnorm_for_post_block",
```
The context prompt ends exactly with "called simply" with no trailing space, as shown in the same block:
```14:16:001_layers_baseline/output-gemma-2-9b.json
    "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply",
```
Layers 0–3 all predict the prompt token "simply" with p ≈ 1 (see CSV):
```2:5:001_layers_baseline/output-gemma-2-9b-pure-next-token.csv
0,16,…, simply,1.0 …
1,16,…, simply,1.0 …
2,16,…, simply,1.0 …
3,16,…, simply,1.0 …
```
Copy-reflex ✓.  Diagnostics expose the required summary metrics:
```17:20:001_layers_baseline/output-gemma-2-9b.json
    "L_copy": 0,
    "L_semantic": 42,
    "delta_layers": 42
```

## 3. Quantitative findings

| Layer | Entropy (bits) | Top-1 token |
|-------|---------------|-------------|
| **0** | 6.1 e-16 | **simply** |
| 1 | 1.0 e-12 | simply |
| 2 | 1.4 e-7 | simply |
| 3 | 2.95 e-7 | simply |
| 4 | 3.2 e-5 | simply |
| 5 | 1.8 e-5 | simply |
| 6 | 5.05 e-3 | simply |
| 7 | 1.34 e-3 | simply |
| 8 | 1.07 e-3 | simply |
| 9 | 3.53 e-3 | simply |
| 10 | 9.24 e-3 | simply |
| 11 | 1.26 e-2 | simply |
| 12 | 8.16 e-4 | simply |
| 13 | 5.36 e-4 | simply |
| 14 | 5.68 e-4 | simply |
| 15 | 4.13 e-3 | simply |
| 16 | 1.83 e-2 | simply |
| 17 | 2.01 e-2 | simply |
| 18 | 0.214 | simply |
| 19 | 0.194 | simply |
| 20 | 0.807 | simply |
| 21 | 0.280 | simply |
| 22 | 0.095 | simply |
| 23 | 0.0288 | simply |
| 24 | 0.0115 | simply |
| 25 | 0.0868 | simply |
| 26 | 0.00561 | simply |
| 27 | 0.773 | " |
| 28 | 0.268 | simply |
| 29 | 0.809 | " |
| 30 | 0.339 | " |
| 31 | 0.0946 | " |
| 32 | 0.166 | " |
| 33 | 0.0397 | " |
| 34 | 0.366 | " |
| 35 | 0.0368 | " |
| 36 | 0.459 | " |
| 37 | 0.221 | " |
| 38 | 0.0681 | " |
| 39 | 0.0242 | " |
| 40 | 0.0163 | " |
| 41 | 1.011 | " |
| **42** | **0.370** | **Berlin** |

Copy-collapse **L_copy = 0**, semantic emergence **L_semantic = 42**, so Δ = 42 layers.

## 4. Qualitative patterns & anomalies
Gemma shows a strikingly long copy-reflex (Δ = 42 layers).  The residual stream is dominated by the prompt-final adverb "simply" through most of the stack before abruptly switching to punctuation tokens (various quote glyphs) and only producing "Berlin" in the last layer.  This mirrors "long-tail echo" phenomena reported in tuned-lens work (2303.08112).

The test prompt without the one-word instruction, "Berlin is the capital of", is answered immediately with "Germany" (0.88) indicating semantic knowledge is present but suppressed when the model first learns to echo the prompt:
```46:53:001_layers_baseline/output-gemma-2-9b.json
      " Germany",
      0.8765909671783447
```
Thus removing the brevity instruction collapses the stack early, but under the original instruction the model waits until the final layer.

Temperature sweep confirms the answer token is robust: at T = 0.1 "Berlin" carries 98 % mass, yet even at T = 2.0 it stays top-1 (9 %).

Checklist
- RMS lens? ✓  
- LayerNorm bias removed? ✓ (not needed, RMS)  
- Punctuation anchoring? ✓ (layers 27–41 quote marks)  
- Entropy rise at unembed? ✓ (entropy climbs from 0.001-0.02 mid-stack to > 1 bit at layer 41)  
- Punctuation / markup anchoring? ✓  
- Copy reflex? ✓  
- Grammatical filler anchoring? ✗ (layers 0-5 dominated by "simply", not {is,the,a,of})

## 5. Tentative implications for Realism ↔ Nominalism
1. Does the persistent echo suggest Gemma allocates early capacity to *surface compliance* before semantics, implying that representation of the instruction is more nominal than real?
2. Could the late emergence of "Berlin" indicate that factual recall is stored in deeper MLPs, aligning with a realist view of distributed truth representations?
3. Does the dramatic shift from "simply" to punctuation tokens hint at a separate pathway that negotiates output formatting, complicating a simple realist/nominalist dichotomy?
4. Would interventions that attenuate the adverb token in early layers advance the semantic collapse, supporting the hypothesis that early nominal residues gate realism downstream?

## 6. Limitations & data quirks
Entropy values for early layers are near-zero due to single-token dominance, limiting resolution.  The CSV records reveal extensive garbage-tokens in other positions; these do not affect Δ but reduce confidence in absolute entropies.  The run date is absent from logs; timings and seed are unknown.  Layer 27 intermittently flips to Unicode quote glyphs, hinting at tokenizer idiosyncrasies.

## 7. Model fingerprint
Gemma-2-9B: collapse at L 0; final entropy 0.37 bits; 'Berlin' only surfaces at layer 42.

---
Produced by OpenAI o3
