**Result Synthesis**
- **Setup sanity**: All runs use a normalized lens over the residual stream with RMSNorm LN and apply the first real `ln1` to embeddings (script: `apply_norm_or_skip(..., probe_after_block=False)`), then decode with an analysis unembed potentially promoted to FP32. Copy‑collapse uses a prompt‑membership rule with an entropy fallback at 1.0 bit (layers_core/collapse_rules.py: “Optional fallback… if entropy_bits < entropy_fallback_threshold” where `entropy_fallback_threshold=1.0`).
- **Copy‑reflex prevalence**: Only the Gemma‑2 family fires copy‑collapse in the first four layers. gemma‑2‑9b and gemma‑2‑27b have `copy_collapse=True` at layer 0 (e.g., gemma‑2‑27b row “0,… entropy 4.968e‑04 … ‘ simply’, 0.999976 …, True, True, False”); all other models have `False` in layers 0–3. Qwen3‑8B and Qwen3‑14B later show `copy_collapse=True` via the entropy fallback (L31 and L32 respectively), but not in 0–3; both files show the top‑1 is not a prompt token when the flag fires (e.g., Qwen3‑8B L31: “…, Berlin, 0.9359 …, True, True, True”).
- **Emergence depth (relative)**: Using `L_semantic` from diagnostics and normalizing by `num_layers`, meaning emerges “early” (< 70% depth) only for Meta‑Llama‑3‑70B (40/80 ≈ 50%). All others are “late” (≥ 70%): Meta‑Llama‑3‑8B (25/32 ≈ 78%), Mistral‑7B (25/32 ≈ 78%), Mistral‑Small‑24B (33/40 ≈ 82.5%), Yi‑34B (44/60 ≈ 73.3%), Qwen3‑8B (31/36 ≈ 86%), Qwen3‑14B (36/40 = 90%), Qwen2.5‑72B (80/80 = 100%), gemma‑2‑9b (42/42 = 100%), gemma‑2‑27b (46/46 = 100%).
- **Δ and Δ̂ across models**: Where `L_copy` exists (or `L_copy_H` fallback): gemma‑2‑9b Δ=42, Δ̂=1.00; gemma‑2‑27b Δ=46, Δ̂=1.00; Qwen3‑8B Δ=0, Δ̂=0.00; Qwen3‑14B Δ=4, Δ̂=0.10. Models with `L_copy=null` (Llama‑3‑8B/70B, Mistral‑7B, Mistral‑Small‑24B, Yi‑34B, Qwen2.5‑72B) do not admit a Δ.
- **Entropy drop shape between L_copy and L_semantic**: Using the pure‑next‑token CSV entropies at those layers:
  - gemma‑2‑9b: H(L_copy)=1.67e‑05 bits (L0) → H(L_sem)=0.370 bits (L42): non‑monotonic; early copy makes near‑zero entropy that rises and then falls late (ΔH ≈ −0.37).
  - gemma‑2‑27b: 4.97e‑04 bits (L0) → 0.118 bits (L46): similar U‑shape; ΔH ≈ −0.12.
  - Qwen3‑8B: 0.454 bits (L31) → 0.454 bits (L31): coincident copy/semantic; ΔH = 0.00 (degenerate case; copy flag fired via entropy fallback).
  - Qwen3‑14B: 0.816 bits (L32) → 0.312 bits (L36): gentle ~0.5‑bit taper across 4 layers.
  - Others (no `L_copy`): ΔH not defined; saliently, `H(L_sem)` varies widely: Llama‑3‑8B 16.81 bits, Llama‑3‑70B 16.94, Mistral‑7B 13.60, Mistral‑Small‑24B 16.77, Yi‑34B 15.33, Qwen2.5‑72B 4.12.
- **Width/heads vs sharpness**: No consistent association. The widest models (d_model=8192, n_heads=64 in Llama‑3‑70B and Qwen2.5‑72B) land at opposite ends: Llama‑3‑70B is early with high‑entropy semantics (H≈16.94) while Qwen2.5‑72B is maximally late with a comparatively low‑entropy semantic head (H≈4.12). Within Qwen3, 14B (d_model=5120, 40 heads) shows a slightly steeper local drop (ΔH≈0.5) than 8B (ΔH≈0), but Gemma‑2 (27B/9B) shows negative ΔH due to extreme early copy reflex.
- **Family patterns**:
  - Qwen family (Qwen3‑8B/14B, Qwen2.5‑72B): consistent absence of early copy‑reflex; semantics emerge very late (≥86% depth). Qwen3‑8B collapses copy/semantics at the same layer (L31), while Qwen3‑14B needs ~4 layers more to settle. Qwen2.5‑72B is the latest (final layer), with moderate concentration (H≈4.12, row “80,…, Berlin, 0.3395 …”).
  - Gemma‑2 family (9B/27B): both exhibit a strong layer‑0 copy‑reflex on the instruction word “ simply” with near‑certain mass (e.g., gemma‑2‑27b L0: “ simply, 0.999976 …, True, True”); semantics appear only at the last layer with very low entropy (0.37 and 0.118 bits respectively).
- **Link to capability scores (MMLU/ARC)**: Late emergence does not uniformly predict higher scores. High‑scoring Yi‑34B (MMLU 76.3, ARC 80) is late (≈73% depth) with high‑entropy semantics (H≈15.33), while Llama‑3‑70B (MMLU 79.5, ARC 93.0) is early (50%). Qwen2.5‑72B (MMLU 86.1) is extremely late (100%) with lower entropy at L_sem; Mistral‑Small‑24B‑Base (MMLU 80.7) is late (≈82.5%). Overall, emergence depth is not a reliable proxy for external scores; concentration at L_sem (ΔH where defined) also fails to track scores consistently.
- **Relation to prior work**: The late linearization of task‑relevant features with mid‑stack surface‑form anchoring (quotes/underscores), followed by a final‑layer confidence increase, mirrors tuned‑lens reports (arXiv:2303.08112). The early copy‑reflex in Gemma aligns with induction/echo behaviors commonly surfaced by logit‑lens‑style probes under instruction tokens, while Qwen avoids early echo and delays semantics until very late layers.

Quoted evidence (selected):
- Gemma‑2‑9B: “copy_collapse = True → layer = 0 … ‘ simply’, p1 ≈ 1.0” (evaluation‑gemma‑2‑9b.md L9) and final “L 42 – entropy 0.370 … ‘ Berlin’” (L57–L59).
- Gemma‑2‑27B: “L_copy: 0 … L_semantic: 46” (evaluation‑gemma‑2‑27b.md L8–L10), “L 46 — entropy 0.118048 … ‘Berlin’” (L58).
- Qwen3‑8B: “First flagged row … layer = 31 … top‑1 = ‘ Berlin’, p1 = 0.9359 → ✗ fired spuriously (entropy fallback)” (evaluation‑Qwen3‑8B.md L12), “L 31 – entropy 0.454 bits, top‑1 ‘Berlin’” (L46–L47).
- Qwen3‑14B: “L_copy: 32 … L_semantic: 36 … Δ=4” (evaluation‑Qwen3‑14B.md L10), “L 36 … 0.312 bits … Berlin” (L54). “L 37 … rest_mass 4.4e‑05” (CSV row quoted at evaluation L74).
- Llama‑3‑70B: “L_semantic: 40” (evaluation‑Meta‑Llama‑3‑70B.md L19) and “L 80 … 2.5890 bits … ‘Berlin’” (L110–L118).

Interpretation notes:
- Copy‑reflex outlier: Only Gemma‑2 (9B/27B) shows a layer‑0 copy‑reflex under the provided thresholds; all others lack early echo. Qwen’s fallback‑driven copy flags occur near semantic collapse and should not be read as prompt copying.
- Size effects: Δ̂ trends are inconsistent across sizes; both Gemma models span the full depth (Δ̂=1.0) but do so by starting at L_copy=0 and ending at L_sem=final, which is not a “long ramp” but rather “copy at 0 → semantic at end”. Qwen3‑14B’s shallow Δ̂ (0.10) reflects only a short post‑copy settling.
- Entropy shape: Many models maintain very high entropy until the head (e.g., Llama‑3‑8B H≈16.8 at L_sem), consistent with lens reports that unembed‑adjacent layers concentrate mass while mid‑layers host diffuse surface features (arXiv:2303.08112).

**Misinterpretations in Existing EVALS**
- evaluation‑Mistral‑Small‑24B‑Base‑2501.md L74: “Entropy rise at unembed? ✓ (entropy drops to 3.18 bits at L40; final decode)”. This contradicts itself (“rise” vs “drops”) and the JSON shows the final prediction entropy ≈ 3.18 bits, matching the lens row rather than rising.
- evaluation‑Qwen3‑8B.md L53: Treating ΔH = 0.000 as an interpretable “no‑drop” obscures that `copy_collapse=True` here is triggered by the entropy fallback, not prompt copying (L12). Δ/ΔH are degenerate and should not be over‑interpreted for this model.
- evaluation‑Meta‑Llama‑3‑8B.md L60: “Semantic emergence is late and sharp.” The lens entropy remains ≈16.8 bits at L_sem and only falls at the final layer (L32: 2.961 bits, L49). Describing the pre‑final change as “sharp” conflates final‑layer concentration with earlier layers’ high‑entropy semantics.
- evaluation‑Gemma‑2‑9B.md L73: “Rest‑mass peaks mid‑stack at 0.2563 … indicating no precision loss at collapse.” While accurate for that file, generalizing this as a “precision check” is overstated; rest_mass depends on top‑k truncation and varies widely across models. Cross‑model claims should be avoided (script computes rest_mass from top‑k only).
- evaluation‑Qwen2.5‑72B.md L131: “Rest_mass ≤ 0.30 post‑semantic” is vacuous here because `L_semantic` equals the final layer (80); there are no post‑semantic layers in this run to support that generalization.

**Limitations**
- **RMS‑lens comparability**: RMS‑lens can distort absolute probabilities; compare entropy/ΔH within a model, not across differing normalization schemes.
- **Prompt sensitivity**: Single‑prompt probing may over‑fit tokenizer quirks; early copy‑collapse depth can shift with wording/punctuation.
- **Hidden‑state coverage**: We do not inspect attention/MLP internals, only residual projections; entropy bumps from internal gating may be mis‑attributed.
- **Unembed precision**: Some runs promote unembed to FP32 (`"use_fp32_unembed": true`); this slightly shrinks entropy gaps. Keep comparisons qualitative.
- **Layer counts differ**: Depth varies (e.g., 32 vs 80); discuss relative depth (Δ̂), not absolute layer indices.
- **Correlation only**: No causal patching here; associations between depth/entropy and capability (MMLU/ARC) are descriptive only.
- **Cross‑model rest_mass**: `rest_mass` reflects top‑k truncation and is not directly comparable across models; high rest_mass around L_sem in some models (e.g., Llama‑3‑8B) is expected under diffuse distributions and does not imply poor calibration.

**Score Link (qualitative)**
- **Steeper ΔH vs MMLU**: No predictive pattern. The largest positive ΔH (Qwen3‑14B ≈ 0.5 bits) does not correspond to the top MMLU in this set (Qwen2.5‑72B 86.1%). Conversely, Gemma‑2 shows negative ΔH (artifact of early copy‑reflex) despite mid/high MMLU (63% at 27B). Treat ΔH as intra‑model shape, not a cross‑model capability proxy.

---
Produced by OpenAI GPT-5

