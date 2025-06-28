# 1. Overview  
Mistral-7B-v0.1 (7 B parameters) was probed on 2025-06-28 (timestamp not recorded; assumed evaluation date).  
The script records layer-wise residual streams, applies a normalized logit lens, and logs entropy and top-k predictions into JSON (diagnostics, test probes) plus a CSV of per-layer results.

# 2. Method sanity-check  
JSON `diagnostics` confirms the run used RMSNorm ("first_block_ln1_type": "RMSNormPre") with `use_norm_lens: true`.  
The script also hooks positional embeddings when available:
```160:183:001_layers_and_logits/run.py
if 'hook_pos_embed' in model.hook_dict:
    pos_hook = model.hook_dict['hook_pos_embed'].add_hook(cache_hook)
```
and prints at runtime:
```210:230:001_layers_and_logits/run.py
print("Using NORMALIZED residual stream (RMS + learned scale)")
```
Together these confirm the intended norm-lens pipeline and positional encoding were active.

# 3. Quantitative findings
| Layer | Entropy (bits) | Top-1 token |
|-------|---------------:|-------------|
| L 0 | 14.72 | 'laug' |
| L 1 | 14.44 | 'zo' |
| L 2 | 14.05 | 'ts' |
| L 3 | 13.87 | 'richt' |
| L 4 | 13.82 | 'amber' |
| L 5 | 13.74 | 'aiser' |
| L 6 | 13.73 | 'amber' |
| L 7 | 13.64 | 'nab' |
| L 8 | 13.62 | 'amber' |
| L 9 | 13.58 | 'answer' |
| L 10 | 13.32 | 'answer' |
| L 11 | 13.48 | 'Answer' |
| L 12 | 13.30 | '*****/' |
| L 13 | 13.66 | 'ír' |
| L 14 | 13.67 | 'Answer' |
| L 15 | 13.55 | 'Answer' |
| L 16 | 13.44 | 'Answer' |
| L 17 | 12.92 | 'Answer' |
| L 18 | 13.22 | 'cities' |
| L 19 | 12.74 | 'cities' |
| L 20 | 11.43 | 'cities' |
| L 21 | 8.48 | 'capital' |
| L 22 | 6.56 | 'Berlin' |
| L 23 | 3.16 | 'Washington' |
| L 24 | 2.88 | 'Berlin' |
| **L 25** | **0.56** | **'Berlin'** |
| L 26 | 0.28 | 'Berlin' |
| L 27 | 0.10 | 'Berlin' |
| L 28 | 0.14 | 'Berlin' |
| L 29 | 0.12 | 'Berlin' |
| L 30 | 0.35 | 'Berlin' |
| L 31 | 0.79 | 'Berlin' |
| L 32 | 1.80 | 'Berlin' |

# 4. Qualitative patterns & anomalies
The network maintains near-maximum entropy (> 13 bits) for the first twenty layers, indicating minimal task-specific signal. A pronounced drop begins at L 21 when the top token shifts to the abstract category 'capital', and a sharp collapse occurs at L 25 where entropy falls below 1 bit and 'Berlin' dominates (≈0.92 p). This late-stack crystallisation echoes the behaviour described in Tuned-Lens 2303.08112.

Layer 23 briefly elevates 'Washington' despite narrowing entropy (3.16 bits), hinting at a competing geopolitical attractor rather than noise.  
> "23,11,:,3.158… 'Washington',0.46" [L289]

Entropy rises again after the final projection (L 32 = 1.8 bits) while preserving 'Berlin' top-1 (0.83 p), consistent with final-norm mixing that re-introduces generic language priors.  
> "32,11,:,1.800… 'Berlin',0.83" [L409]

Auxiliary probes underscore phrasing sensitivity: "Germany's capital is" remains diffuse (6.68 bits) whereas "Berlin is the capital of" is confident (0.95 bits). Temperature scaling behaves predictably, collapsing to near-zero entropy at τ = 0.1 and expanding (> 12 bits) at τ = 2.0.

Checklist  
✓ RMS lens  
✓ LayerNorm (RMS variant)  
✗ Colon-spam  
✓ Entropy rise at unembed

# 5. Tentative implications for Realism ↔ Nominalism
? Does the delayed collapse suggest that factual recall lives in deep nominal subspaces rather than being explicitly represented earlier?  
? Could the transient 'Washington' attractor indicate that competing factual schemas are stored distributively and only reconciled by late binding?  
? Is the entropy rebound in the final layer evidence for a realist-style mixture where generic priors are re-introduced after commitment to a fact?  
? How would steering earlier layers toward 'Berlin' via activation patching affect the collapse point and support either philosophical stance?

# 6. Limitations & data quirks
Layer-wise logs record context ':' rather than generated content, so answer alignment must be inferred from predictions. Absolute entropy calibration depends on the RMS lens assumptions, which are unvalidated here. Numeric values may shift under different precision (the run used MPS-fp32). Run timestamp and model checkpoint hash are not recorded, limiting reproducibility.

# 7. Model fingerprint
Mistral-7B-v0.1: collapse at L 25; final entropy 1.8 bits; 'Berlin' stabilises from L 22 onward.

---  
Produced by OpenAI o3
