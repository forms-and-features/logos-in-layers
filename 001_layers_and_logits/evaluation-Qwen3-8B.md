## 1. Overview
Qwen/Qwen3-8B (≈8 B parameters) was interrogated with the layer-wise logit-lens probe on an Apple M-series (mps) backend.  The probe captures, for every transform-er block, the entropy of the next-token distribution at the answer slot and the top-k lexical guesses, together with supplementary tests on prompt variants and temperature sweeps.

## 2. Method sanity-check
Both JSON diagnostics and probe code confirm that positional encodings are summed into the residual before the first lens application and that an RMS-norm lens is applied throughout.
> "first_block_ln1_type": "RMSNormPre", "final_ln_type": "RMSNormPre"  
> 8:18:001_layers_and_logits/output-Qwen3-8B.json
> resid = (residual_cache['hook_embed'] + residual_cache['hook_pos_embed']).to(model.cfg.device)  
> 277:278:001_layers_and_logits/run.py

## 3. Quantitative findings
| layer | entropy (bits) | top-1 token |
|-------|---------------|-------------|
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
| **L 28** | **0.43** | 'Berlin' |
| L 29 | 0.19 | 'Berlin' |
| L 30 | 0.87 | 'Berlin' |
| L 31 | 0.01 | 'Berlin' |
| L 32 | 0.08 | 'Berlin' |
| L 33 | 0.01 | 'Berlin' |
| L 34 | 0.01 | 'Berlin' |
| L 35 | 0.13 | 'Berlin' |
| L 36 | 2.02 | 'Berlin' |

The first entropy collapse (< 1 bit) occurs at **layer 28**.

## 4. Qualitative patterns & anomalies
The distribution is broad and multilingual up to layer 20, with random Japanese, Chinese and CJK control tokens dominating early guesses – a symptom of residual noise before accumulated attention ("colon-spam").  Entropy then declines steadily; a country-name prior ('Germany') appears around layer 25 (1.41 bits) and sharpens into the correct city by layer 28.  From layer 31 onward the distribution is nearly degenerate, e.g. > "... 'Berlin', 0.999" [443].  
The final unembedding layer slightly re-broadens the distribution (entropy rises from 0.13 → 2.02 bits), a known artefact when fp32 unembed weights are restored ("unembed temperature").  
Probe variants corroborate robustness: "Berlin is the capital of ⟶" has entropy 1.20 bits with 0.73 prob on 'Germany' [33:60], while a one-word prompt still yields 0.72 on 'Berlin' [66:80].  Temperature sweep shows deterministic certainty at τ = 0.1 (1.8 × 10⁻⁸ bits) and near-uniform 13.1 bits at τ = 2.0 [120:150].

Checklist:  
✓ RMS lens  
✓ LayerNorm (RMSPre)  
✓ Colon-spam early  
✗ Entropy rise at unembed? (yes, observed)

## 5. Tentative implications for Realism ↔ Nominalism
1. Does the late collapse (≈ ¾ through the stack) suggest that semantic realism (stable city concept) only crystallises after prolonged token-level nominal processing?
2. Could the transient 'Germany' prior indicate a mid-stack nominal abstraction layer that feeds into a realist city representation?  Might intervening on that representation alter downstream certainty?
3. Does the entropy rebound at unembed imply an architectural bias towards nominalism, where symbols are re-expanded into a lexical manifold even when an internal realist variable is fixed?
4. How would training with stronger supervised alignment affect the layer-index of collapse – bringing realist semantics earlier or merely sharpening nominal priors?

## 6. Limitations & data quirks
The probe inspects only one prompt and relies on single-batch caching; early layer entropies are noisy and include spurious non-Latin tokens, possibly due to tokenizer mismatch.  File output lacks a run timestamp, so reproducibility of hardware state is uncertain.  CSV top-k lists truncate at 20, hiding tail mass estimates.

## 7. Model fingerprint
Qwen3-8B: collapse at L 28; final entropy 2.0 bits; 'Berlin' surfaces with ≥ 0.99 prob from L 31 onward.

---
Produced by OpenAI o3
