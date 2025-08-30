**Overview**
Model: 01-ai/Yi-34B (34B). Run date: 2025-08-24. The probe applies a norm-lens logit lens over all layers to track entropy and next-token top-1 evolution for the prompt; collapse flags identify copy-reflex and first semantic emergence of the correct answer.

**Method Sanity‑Check**
JSON diagnostics confirm rotary positional handling and norm-lens application: "use_norm_lens": true [JSON L807]; "layer0_position_info": "token_only_rotary_model" [JSON L816]. The context_prompt ends exactly with “called simply” (no trailing space): "Give the city name only, plain text. The capital of Germany is called simply" [JSON L4]. Diagnostics block includes required fields: L_copy, L_copy_H, L_semantic, delta_layers, and implementation flags (e.g., unembed_dtype): "L_semantic": 44 [JSON L821], "unembed_dtype": "torch.float32" [JSON L809]. Copy-collapse check in pure-next-token CSV: no rows with copy_collapse = True (none found) → n.a. (no trigger; no spurious fires).

**Quantitative Findings**
L 0 – entropy 15.962338 bits, top-1 ' Denote'
L 1 – entropy 15.941797 bits, top-1 '.'
L 2 – entropy 15.932033 bits, top-1 '.'
L 3 – entropy 15.839134 bits, top-1 'MTY'
L 4 – entropy 15.825938 bits, top-1 'MTY'
L 5 – entropy 15.863527 bits, top-1 'MTY'
L 6 – entropy 15.829475 bits, top-1 'MTQ'
L 7 – entropy 15.862338 bits, top-1 'MTY'
L 8 – entropy 15.872590 bits, top-1 '其特征是'
L 9 – entropy 15.836432 bits, top-1 '审理终结'
L 10 – entropy 15.797048 bits, top-1 '~\\\\'
L 11 – entropy 15.701524 bits, top-1 '~\\\\'
L 12 – entropy 15.773987 bits, top-1 '~\\\\'
L 13 – entropy 15.783698 bits, top-1 '其特征是'
L 14 – entropy 15.739416 bits, top-1 '其特征是'
L 15 – entropy 15.753135 bits, top-1 '其特征是'
L 16 – entropy 15.713601 bits, top-1 '其特征是'
L 17 – entropy 15.713733 bits, top-1 '其特征是'
L 18 – entropy 15.716445 bits, top-1 '其特征是'
L 19 – entropy 15.696079 bits, top-1 'ncase'
L 20 – entropy 15.604047 bits, top-1 'ncase'
L 21 – entropy 15.609381 bits, top-1 'ODM'
L 22 – entropy 15.620244 bits, top-1 'ODM'
L 23 – entropy 15.601881 bits, top-1 'ODM'
L 24 – entropy 15.547840 bits, top-1 'ODM'
L 25 – entropy 15.566961 bits, top-1 'ODM'
L 26 – entropy 15.585484 bits, top-1 'ODM'
L 27 – entropy 15.227417 bits, top-1 'ODM'
L 28 – entropy 15.431764 bits, top-1 'MTU'
L 29 – entropy 15.466778 bits, top-1 'ODM'
L 30 – entropy 15.550709 bits, top-1 'ODM'
L 31 – entropy 15.531230 bits, top-1 ' 版的'
L 32 – entropy 15.454513 bits, top-1 'MDM'
L 33 – entropy 15.455110 bits, top-1 'XFF'
L 34 – entropy 15.477488 bits, top-1 'XFF'
L 35 – entropy 15.471056 bits, top-1 'Mpc'
L 36 – entropy 15.432985 bits, top-1 'MDM'
L 37 – entropy 15.453636 bits, top-1 'MDM'
L 38 – entropy 15.485533 bits, top-1 'MDM'
L 39 – entropy 15.504303 bits, top-1 'MDM'
L 40 – entropy 15.527823 bits, top-1 'MDM'
L 41 – entropy 15.519179 bits, top-1 'MDM'
L 42 – entropy 15.534887 bits, top-1 'keV'
L 43 – entropy 15.517909 bits, top-1 ' "'
**L 44 – entropy 15.327293 bits, top-1 ' Berlin'**
L 45 – entropy 15.293162 bits, top-1 ' Berlin'
L 46 – entropy 14.833831 bits, top-1 ' Berlin'
L 47 – entropy 14.731156 bits, top-1 ' Berlin'
L 48 – entropy 14.941262 bits, top-1 ' Berlin'
L 49 – entropy 14.695848 bits, top-1 ' Berlin'
L 50 – entropy 14.969212 bits, top-1 ' Berlin'
L 51 – entropy 14.538900 bits, top-1 ' Berlin'
L 52 – entropy 15.137337 bits, top-1 ' Berlin'
L 53 – entropy 14.869741 bits, top-1 ' Berlin'
L 54 – entropy 14.955276 bits, top-1 ' Berlin'
L 55 – entropy 14.932296 bits, top-1 ' Berlin'
L 56 – entropy 14.745391 bits, top-1 ' Berlin'
L 57 – entropy 14.748362 bits, top-1 ' '
L 58 – entropy 13.457073 bits, top-1 ' '
L 59 – entropy 7.191097 bits, top-1 ' '
L 60 – entropy 2.981155 bits, top-1 ' Berlin'

ΔH (bits) = entropy(L_copy) − entropy(L_semantic) = n.a. (no copy layer)
Confidence milestones: p > 0.30 at layer 60; p > 0.60 not reached; final-layer p = 0.5555 [pure CSV row 63].

**Qualitative Patterns & Anomalies**
Semantic emergence is late and gradual. The first semantic layer is L44 with a low initial mass on the correct token: “… (Berlin, 0.00846)” [pure CSV row 46]. Confidence rises across late layers, dips into punctuation at L57–59 (top-1 becomes space/period), then recovers strongly at the final layer: “… (‘Berlin’, 0.5555)” [pure CSV row 63]. This late consolidation with an interim detour into punctuation is consistent with deep pre-norm transformers where high-level features sharpen after mid‑stack mixing (cf. Tuned‑Lens 2303.08112).

Negative control shows correct directional knowledge without leakage. For “Berlin is the capital of”, the model predicts the country, not the city: top‑5 → “ Germany” (0.8398), “ the” (0.0537), “ which” (0.0288), “ what” (0.0120), “ Europe” (0.0060) [JSON L14–L31]. Berlin does not appear here, so no semantic leakage.

Records CSV highlights the trajectory of important words around the answer position. At L43 on the final context token (“ simply”), ‘Berlin’ is already present in the wider candidate set but below the top‑5, alongside ‘capital’ and ‘German’: “… Berlin, 0.00053 … capital … German …” [records CSV row 770]. At L44, ‘Berlin’ becomes top‑1 across the final context positions: “ is … (‘Berlin’, 0.0105) …”, “ called … (‘Berlin’, 0.0083) …”, “ simply … (‘Berlin’, 0.0085)” [records CSV rows 786–788]. The top‑5 co-features include “capital”, “Germany/德国” and distractors like “Frankfurt/Munich” and “柏林” (Chinese for Berlin) [records CSV rows 804–842]. By L46–51, ‘Berlin’ strengthens across “ is / called / simply” with probabilities ~0.02–0.06, while ‘capital’ remains in top‑5 and ‘Germany’ appears intermittently [records CSV rows 822–914]. Important-word trajectory: Berlin first enters any top‑5 at layer 44 (pure CSV) and stabilises as the dominant candidate by L47–51; Germany remains intermittently in top‑5 at L44 and L46 [pure CSV rows 46, 48]; capital persists in top‑5 over many late layers.

Instructional phrasing effects at the final layer are visible in test prompts. Without the “simply”/one‑word nudge, completions sometimes favor discourse tokens (e.g., “ after”): “The capital city of Germany is named … ‘ after’ 0.334; ‘ Berlin’ 0.215” [JSON L296–L302]. Adding a minimal constraint helps: “Germany’s capital city is called simply … ‘ Berlin’ 0.469” [JSON L61–L67]; “Germany has its capital at the city called simply … ‘ Berlin’ 0.547” [JSON L155–L157]. We cannot assess collapse‑layer shifts from test prompts (no per‑layer logging), but surface probabilities suggest the instruction improves decisiveness.

Rest‑mass sanity: After L_semantic, rest_mass declines from 0.9807 at L44 to 0.1753 at L60 (max after L_semantic = 0.9777 at L45) [pure CSV rows 46, 47, 63], indicating the top‑k increasingly captures most probability mass; no precision‑loss spikes observed.

Temperature robustness: At T = 0.1, Berlin rank 1 (p = 0.9999996; entropy ≈ 7.1e‑06 bits) [JSON L674–L676, L671]. At T = 2.0, Berlin rank 1 (p = 0.0488) and entropy 12.488 bits [JSON L741–L743, L738]. Entropy rises sharply with temperature, as expected.

Checklist:
- RMS lens: ✓ (RMSNorm at ln1/ln_final; use_norm_lens true) [JSON L807, L810–L813]
- LayerNorm bias removed: ✓ (“not_needed_rms_model”) [JSON L812–L813]
- Entropy rise at unembed: n.a.
- FP32 un‑embed promoted: ✓ (“use_fp32_unembed”: true; “unembed_dtype”: torch.float32) [JSON L807–L811]
- Punctuation / markup anchoring: ✓ (top‑1 becomes space/period at L57–59) [pure CSV rows 59–61]
- Copy‑reflex: ✗ (no copy_collapse=True in layers 0–3; none at all)
- Grammatical filler anchoring: ✗ (layers 0–5 top‑1 not in {is, the, a, of})

**Limitations & Data Quirks**
- No copy‑collapse layer found (L_copy = null) [JSON L819–L821], so ΔH relative to L_copy is undefined.
- Early‑ to mid‑stack top‑1 tokens are heterogeneous artifacts (e.g., “MDM”, “MTU”, “keV”, non‑English tokens), which complicates linguistically grounded interpretation of pre‑semantic layers.
- Rest_mass remains high (>0.9) for many layers after L_semantic before collapsing late; while consistent with broad distributions, it reduces clarity on near‑term sharpening.

**Model Fingerprint**
Yi‑34B: collapse at L 44; final entropy 2.98 bits; ‘Berlin’ strengthens late with a brief punctuation detour at L57–59.

---
Produced by OpenAI GPT-5
