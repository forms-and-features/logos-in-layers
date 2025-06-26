# 1. Executive summary
- All four open-weight models (Gemma-2-9B, Qwen-3-8B, Mistral-7B-v0.1, Llama-3-8B) apply RMS **pre-norm** and exhibit a sharp entropy "collapse" where the next-token distribution becomes near-deterministic.
- The collapse consistently locks onto **"Berlin"** except in Gemma, whose first collapse is a trivial ':' placeholder; Gemma only converges on Berlin much later, highlighting divergent abstraction timelines.
- Entropy **re-opens** after the final block in every model (≈ 1–2 bits), suggesting a common post-decision lexical diversification step.
- Mid-stack fixation on meta tokens (e.g. *Answer/answer*) appears in three models, hinting at a symbolic "slot-filling" phase before concrete entity resolution.
- Open question: Does the timing gap between Gemma's early syntactic collapse and later semantic convergence support a nominalist surface-token stage preceding realist concept formation?

# 2. Comparison table
| Model | Params | Norm type | Collapse layer (< 1 bit) | Final entropy (bits) | First token after collapse | Any anomaly |
|-------|--------|-----------|--------------------------|----------------------|----------------------------|--------------|
| Gemma-2-9B | 9 B | RMSNormPre | 0 | 2.003 | ':' | Early colon-spam & dual collapse |
| Qwen-3-8B  | 8 B | RMSNormPre | 28 | 2.020 | Berlin | Underscore "____" phase before collapse |
| Mistral-7B-v0.1 | 7 B | RMSNormPre | 25 | 1.800 | Berlin | Late mis-rank: Washington > Berlin at L23 |
| Llama-3-8B | 8 B | RMSNormPre | 25 | 1.704 | Berlin | Mild colon-spam around L19 |

# 3. Shared patterns
- RMS pre-norm confirmed in all logs:
  > "Block normalization type: RMSNormPre" — Llama-3 output [L8].
- Entropy cliff followed by 1–2 bit rebound at unembed:
  > "Model's final prediction (entropy: 1.704 bits)" — Llama-3 [L764].
- Collapse locks onto Berlin (or placeholder) and stays stable for several blocks:
  > "Layer 27 … 'Berlin' 0.99" — Mistral [L615].
- Mid-stack meta-token fixation (Answer/answer):
  > "Layer 10 … ' answer' (0.85)" — Gemma [L180].

# 4. Model-specific quirks & red flags
### Gemma-2-9B
- **Zero-entropy colon loop** for first 8 layers: `Layer 0 ':' (1.000000)` — output-Gemma [L31].
- **Secondary semantic collapse** at Layer 35 onto Berlin after long drift — eval-Gemma §3.
- Final entropy spike despite near-determinism inside model (`Model's final prediction 2.003 bits`) — output-Gemma [L1022].

### Qwen-3-8B
- **Delayed collapse** (Layer 28) compared with peers — output-Qwen [L662].
- **Underscore spam** in layers 24-26: `'____' 0.30` — output-Qwen [L613].
- Re-opening at final layer (2.020 bits) — output-Qwen [L866].

### Mistral-7B-v0.1
- **Washington distraction** outranks Berlin at Layer 23 (0.51 p) — output-Mistral [L536].
- Sharpest collapse (0.104 bits) by Layer 27 — output-Mistral [L657].
- Post-collapse entropy rebounds to 1.8 bits — output-Mistral [L759].

### Llama-3-8B
- **Collapse mirrors Mistral**: Layer 25 entropy 0.43 bits on 'Berlin' — output-Llama [L585].
- **Colon-spam blip** (`' (::'`) in Layer 19 — eval-Llama §4.
- Re-open to 1.7 bits at unembed — output-Llama [L764].

# 5. Preliminary implications for Realism ↔ Nominalism
- The universal entropy rebound hints that a realist internal commitment ("Berlin") fans out into multiple lexical tokens—supporting a nominalist surface realisation layer.
- Gemma's two-stage collapse (syntax then semantics) may indicate an intermediate nominal slot before realist grounding.
- Mid-stack 'Answer' fixation across models suggests a nominal label phase preceding concrete entity selection.
- The Washington mis-rank in Mistral shows competing realist candidates can intrude late, challenging a strictly serial realist narrative.

# 6. Methodological caveats
- RMS lens scaling identical across models but hidden fp16 ↔ fp32 differences could skew entropy magnitudes.
- Prompt tokenisation differs (special `<bos>` vs `<s>`), possibly affecting early layers.
- Console dumps truncate probabilities; low-precision zeros in Gemma may exaggerate "zero-entropy" claim.
- Single-prompt probe; no variance estimates. No control for temperature or sampling.

# 7. Priority next steps
1. **Cross-patch collapse residuals** — overwrite Gemma Layer 35 with Mistral Layer 25, re-run to test if semantic collapse transfers (§5 bullet 1).
2. **Paraphrase battery** — reuse `run.py` on 20 wording variants to quantify collapse-layer variance, addressing caveat on prompt idiosyncrasy (§6).
3. **Tuned-lens sweep** — apply RMS-lens vs tuned lens across checkpoints to see if rebound persists, probing nominal vs realist layering (§3 bullet 2).

# 8. Appendix: Evidence map
- Gemma params → evaluation-gemma-2-9b.md [L2]
- Gemma norm → output-gemma-2-9b.txt [L5]
- Gemma collapse layer → output-gemma-2-9b.txt [L29-31]
- Gemma final entropy → output-gemma-2-9b.txt [L1022]
- Qwen params → evaluation-Qwen3-8B.md [L3]
- Qwen norm → output-Qwen3-8B.txt [L8]
- Qwen collapse layer → output-Qwen3-8B.txt [L662]
- Qwen final entropy → output-Qwen3-8B.txt [L866]
- Mistral params → evaluation-Mistral-7B-v0.1.md [L2]
- Mistral norm → output-Mistral-7B-v0.1.txt [L6]
- Mistral collapse layer → output-Mistral-7B-v0.1.txt [L579-584]
- Mistral final entropy → output-Mistral-7B-v0.1.txt [L759]
- Llama params → evaluation-Meta-Llama-3-8B.md [L2]
- Llama norm → output-Meta-Llama-3-8B.txt [L8]
- Llama collapse layer → output-Meta-Llama-3-8B.txt [L585-590]
- Llama final entropy → output-Meta-Llama-3-8B.txt [L764]
