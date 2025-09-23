1. Overview

Meta‑Llama‑3‑70B (80 layers; pre‑norm). The probe analyzes layer‑by‑layer next‑token predictions under a norm lens, tracking copy/semantic collapse, entropy, KL to final, cosine drift, and calibration. Final head is well aligned (KL≈0), and the gold token is “Berlin”.

2. Method sanity‑check

Diagnostics confirm the norm lens and positional handling are active: “use_norm_lens": true” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:807] and “layer0_position_info": "token_only_rotary_model"” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:816]. The context prompt ends with “called simply” (no trailing space): “context_prompt": "Give the city name only, plain text. The capital of Germany is called simply"” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:4]. Copy detector settings are recorded: “copy_thresh": 0.95, "copy_window_k": 1, "copy_match_level": "id_subsequence"” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:846–848], with soft config “threshold": 0.5, "window_ks": [1,2,3]” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:833–858] and CSV/JSON flag labels “copy_strict@0.95 … copy_soft_k{1,2,3}@0.5” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1077–1082].

Gold alignment is ID‑based and ok: “gold_alignment": "ok"” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:898,1104]; “gold_answer": { … "string": "Berlin", "first_id": 20437 … }” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1110–1118]. Negative control is present with summary: “first_control_margin_pos": 0, "max_control_margin": 0.5168457566906” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1106–1108]. Ablation (no‑filler) exists: “L_sem_orig": 40 … "L_sem_nf": 42, "delta_L_sem": 2” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1084–1090]; positive rows are present for both prompt_variant=orig and no_filler (81 each) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:2–82, 83–163].

Summary indices (bits/ranks): “first_kl_below_0.5": 80, "first_kl_below_1.0": 80, "first_rank_le_1": 40, "first_rank_le_5": 38, "first_rank_le_10": 38” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:849–853]. Units are bits for KL and entropy (see “kl_to_final_bits” column and “entropy” field; final KL≈0). Last‑layer head calibration is recorded: “kl_to_final_bits": 0.000729…, "top1_agree": true, "p_top1_lens": 0.4783, "p_top1_model": 0.4689, "temp_est": 1.0, "warn_high_last_layer_kl": false” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:900–917], and matches the pure CSV final row “L 80 … p_top1 = 0.478338 … kl_to_final_bits = 0.000729 …” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:82].

Lens sanity: mode=sample; “lens_artifact_risk": "low", "max_kl_norm_vs_raw_bits": 0.0429” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1015–1075]. No “norm‑only semantics” layer is reported.

Copy‑collapse flags: no strict copy in layers 0–3 (no True in “copy_collapse” for L∈{0,1,2,3}) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:2–5]; ✓ rule not fired. Soft copy flags are absent as well: earliest of {copy_soft_k1@0.5, k2, k3} = none.

3. Quantitative findings

Per‑layer (pos, orig): “L i – entropy X bits, top‑1 ‘token’” — semantic layer in bold (is_answer=True). Built from 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv filtered to prompt_id=pos, prompt_variant=orig.

L 0 – entropy 16.968 bits, top-1 'winding'
L 1 – entropy 16.960 bits, top-1 'cepts'
L 2 – entropy 16.963 bits, top-1 'улю'
L 3 – entropy 16.963 bits, top-1 'zier'
L 4 – entropy 16.959 bits, top-1 'alls'
L 5 – entropy 16.957 bits, top-1 'alls'
L 6 – entropy 16.956 bits, top-1 'alls'
L 7 – entropy 16.953 bits, top-1 'NodeId'
L 8 – entropy 16.959 bits, top-1 'inds'
L 9 – entropy 16.960 bits, top-1 'NodeId'
L 10 – entropy 16.952 bits, top-1 'inds'
L 11 – entropy 16.956 bits, top-1 'inds'
L 12 – entropy 16.956 bits, top-1 'lia'
L 13 – entropy 16.955 bits, top-1 'eds'
L 14 – entropy 16.950 bits, top-1 'idders'
L 15 – entropy 16.953 bits, top-1 'Kok'
L 16 – entropy 16.952 bits, top-1 '/plain'
L 17 – entropy 16.948 bits, top-1 'nut'
L 18 – entropy 16.944 bits, top-1 'nut'
L 19 – entropy 16.948 bits, top-1 'nut'
L 20 – entropy 16.946 bits, top-1 'nut'
L 21 – entropy 16.938 bits, top-1 'burge'
L 22 – entropy 16.938 bits, top-1 'simply'
L 23 – entropy 16.936 bits, top-1 'bur'
L 24 – entropy 16.950 bits, top-1 'bur'
L 25 – entropy 16.937 bits, top-1 '�'
L 26 – entropy 16.938 bits, top-1 '�'
L 27 – entropy 16.937 bits, top-1 'za'
L 28 – entropy 16.933 bits, top-1 '/plain'
L 29 – entropy 16.933 bits, top-1 'plain'
L 30 – entropy 16.939 bits, top-1 'zed'
L 31 – entropy 16.925 bits, top-1 'simply'
L 32 – entropy 16.941 bits, top-1 'simply'
L 33 – entropy 16.927 bits, top-1 'plain'
L 34 – entropy 16.932 bits, top-1 'simply'
L 35 – entropy 16.929 bits, top-1 'simply'
L 36 – entropy 16.940 bits, top-1 'simply'
L 37 – entropy 16.935 bits, top-1 'simply'
L 38 – entropy 16.934 bits, top-1 'simply'
L 39 – entropy 16.935 bits, top-1 'simply'
**L 40 – entropy 16.937 bits, top-1 'Berlin'**
L 41 – entropy 16.936 bits, top-1 '"'
L 42 – entropy 16.944 bits, top-1 '"'
**L 43 – entropy 16.941 bits, top-1 'Berlin'**
**L 44 – entropy 16.926 bits, top-1 'Berlin'**
L 45 – entropy 16.940 bits, top-1 '"'
L 46 – entropy 16.955 bits, top-1 '"'
L 47 – entropy 16.939 bits, top-1 '"'
L 48 – entropy 16.939 bits, top-1 '"'
L 49 – entropy 16.937 bits, top-1 '"'
L 50 – entropy 16.944 bits, top-1 '"'
L 51 – entropy 16.940 bits, top-1 '"'
**L 52 – entropy 16.922 bits, top-1 'Berlin'**
**L 53 – entropy 16.933 bits, top-1 'Berlin'**
**L 54 – entropy 16.942 bits, top-1 'Berlin'**
**L 55 – entropy 16.942 bits, top-1 'Berlin'**
**L 56 – entropy 16.921 bits, top-1 'Berlin'**
**L 57 – entropy 16.934 bits, top-1 'Berlin'**
**L 58 – entropy 16.941 bits, top-1 'Berlin'**
**L 59 – entropy 16.944 bits, top-1 'Berlin'**
**L 60 – entropy 16.923 bits, top-1 'Berlin'**
**L 61 – entropy 16.940 bits, top-1 'Berlin'**
**L 62 – entropy 16.951 bits, top-1 'Berlin'**
**L 63 – entropy 16.946 bits, top-1 'Berlin'**
**L 64 – entropy 16.926 bits, top-1 'Berlin'**
L 65 – entropy 16.933 bits, top-1 '"'
**L 66 – entropy 16.941 bits, top-1 'Berlin'**
**L 67 – entropy 16.930 bits, top-1 'Berlin'**
**L 68 – entropy 16.924 bits, top-1 'Berlin'**
**L 69 – entropy 16.932 bits, top-1 'Berlin'**
**L 70 – entropy 16.926 bits, top-1 'Berlin'**
**L 71 – entropy 16.923 bits, top-1 'Berlin'**
**L 72 – entropy 16.922 bits, top-1 'Berlin'**
L 73 – entropy 16.918 bits, top-1 '"'
**L 74 – entropy 16.914 bits, top-1 'Berlin'**
**L 75 – entropy 16.913 bits, top-1 'Berlin'**
**L 76 – entropy 16.919 bits, top-1 'Berlin'**
**L 77 – entropy 16.910 bits, top-1 'Berlin'**
**L 78 – entropy 16.919 bits, top-1 'Berlin'**
**L 79 – entropy 16.942 bits, top-1 'Berlin'**
**L 80 – entropy 2.589 bits, top-1 'Berlin'**

Control margin (JSON control_summary): first_control_margin_pos = 0; max_control_margin = 0.5168457566906 [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1106–1108].

Ablation (no‑filler): L_copy_orig = null, L_sem_orig = 40; L_copy_nf = null, L_sem_nf = 42; ΔL_copy = null, ΔL_sem = 2 [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1084–1090].

ΔH (bits) = n/a (no strict L_copy). Soft ΔHk (bits) = n/a (no L_copy_soft for k∈{1,2,3}).

Confidence milestones (pure CSV, pos/orig): p_top1 > 0.30 at layer 80; p_top1 > 0.60 not reached; final-layer p_top1 = 0.4783 [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:82].

Rank milestones (JSON): rank ≤ 10 at layer 38; rank ≤ 5 at layer 38; rank ≤ 1 at layer 40 [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:851–853].

KL milestones (JSON/CSV): first_kl_below_1.0 at layer 80; first_kl_below_0.5 at layer 80 [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:849–850]. KL decreases by depth and is ≈ 0 at the final: 0.000729 bits [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:900] and [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:82].

Cosine milestones (pure CSV): first cos_to_final ≥ 0.2 at layer 80; ≥ 0.4 at 80; ≥ 0.6 at 80; final cos_to_final = 0.99999 [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:82].

Prism Sidecar Analysis

Presence: compatible=true (k=512; layers=[embed,19,39,59]) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:819–858].

Early‑depth stability (KL vs final): at L0/19/39/59, Prism KL is higher than baseline (e.g., L19 KL_base=10.456 vs KL_prism=11.347; L39 KL_base=10.419 vs KL_prism=11.491) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:21,41,61 and 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token-prism.csv:21,41,61].

Rank milestones (Prism): no early attainment — first_rank_le_{10,5,1} = none (not reached in sidecar depth range) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token-prism.csv].

Top‑1 agreement at sampled depths: baseline has Berlin as top‑1 by L59, Prism does not (L59 baseline: top‑1 ‘Berlin’; Prism: not ‘Berlin’) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:61; 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token-prism.csv:61].

Cosine drift: Prism does not stabilize earlier (e.g., at L39, cos_base≈0.094 vs cos_prism≈−0.009; at L59, cos_base≈−0.000 vs cos_prism≈−0.028) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:41,61; 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token-prism.csv:41,61].

Copy flags: no spurious flips (copy_collapse and copy_soft_k1@0.5 remain false at sampled depths) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token-prism.csv:2–62].

Verdict: Regressive (KL rises at early/mid layers and rank milestones are not earlier).

4. Qualitative patterns & anomalies

Negative control: “Berlin is the capital of” top‑5 are dominated by “ Germany” (0.8516) with function words following: “… [‘ Germany’, 0.8516], [‘ the’, 0.0791], [‘ and’, 0.0146], [‘ modern’, 0.0048], [‘ Europe’, 0.0031] …” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:10–31]. No leakage of ‘Berlin’ in this prompt (as expected for the inverse mapping).

Important‑word trajectory (records CSV; IMPORTANT_WORDS = ["Germany", "Berlin", "capital", "Answer", "word", "simply"]). Around the final context token (“ simply”, pos=16) ‘Berlin’ first appears in top‑5 at L38 (top‑3, p≈2.50e‑05) and becomes top‑1 at L40 (“(L40, pos=16, top‑1 ‘Berlin’, p≈2.39e‑05)”) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-records.csv:664,698]. ‘Germany’ is salient in nearby positions earlier (e.g., L38 pos=14 top‑1 ‘Germany’, p≈4.74e‑05) and remains in the top‑5 across adjacent positions/layers [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-records.csv:662–681]. ‘capital’ co‑occurs in top‑5 for pos 14–15 over L42–56 (e.g., L45 pos=15 top‑2 ‘ capital’, p≈2.44e‑05) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-records.csv:782]. This suggests late consolidation around the answer token with related context words present in late‑stack neighborhoods.

Collapse‑layer consistency across instruction variants: removing “simply” delays semantics slightly (L_sem 40 → 42; ΔL_sem = +2) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1084–1090]. This points to minor stylistic anchoring, not a large semantic shift.

Rest‑mass sanity: rest_mass remains very high pre‑final (reflecting extremely diffuse layer distributions); it spikes after L_semantic with a maximum ≈0.9999 at L46 [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:48], consistent with top‑20 covering negligible mass when the layer distribution is flat. Treat as top‑k coverage only, not lens fidelity.

Rotation vs amplification: KL to final falls only at the top (first_kl_below_1.0/0.5 at L80) while cos_to_final and p_top1 meaningfully rise only at L80 (cos≈1.0; p_top1≈0.48) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:849–850,900; 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:82]. The trajectory looks like late rotation and calibration rather than gradual amplification (no early direction with high cosine).

Head calibration (final): well‑aligned — “top1_agree": true; temp_est=1.0; kl_to_final_bits=0.000729 … warn_high_last_layer_kl=false” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:900–917]. Prefer within‑model ranks; absolute final p_top1 is trustworthy for this run/family.

Lens sanity: sampled raw‑vs‑norm checks show low artifact risk; “max_kl_norm_vs_raw_bits": 0.0429; first_norm_only_semantic_layer: null” [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1071–1075]. A sample at L41 shows norm/raw differences are small and ranks are consistent at late layers [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1045–1069].

Temperature robustness: at T=0.1 Berlin rank 1, p≈0.9933, entropy≈0.058 [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1018–1031]; at T=2.0 Berlin is still top‑1 but much softer, p≈0.0357, entropy≈14.46 [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1031–1039].

Final‑layer prediction profile includes heavy punctuation/quote mass alongside the answer (e.g., “ \"”, “ ‘”, “ ‘’, ‘ : ’) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:920–1004], consistent with post‑hoc framing tokens co‑activating near the unembed.

Checklist
– RMS lens? ✓ (first_block_ln1_type=RMSNorm; final_ln_type=RMSNorm) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:810–811]
– LayerNorm bias removed? ✓ (not needed on RMS models) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:812]
– Entropy rise at unembed? n.a. (tracked per‑layer; final entropy reported separately)
– FP32 un‑embed promoted? ✓ (“use_fp32_unembed": true; “mixed_precision_fix": "casting_to_fp32_before_unembed" ) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:808,815]
– Punctuation / markup anchoring? ✓ (quotes/colon appear in final top‑k) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:929–947,941–943]
– Copy‑reflex? ✗ (no copy_collapse True in layers 0–3) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:2–5]
– Grammatical filler anchoring? ✗ (layers 0–5 top‑1 not in {is,the,a,of}) [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:2–7]

5. Limitations & data quirks

Rest_mass remains >0.99 across many layers even after L_semantic (e.g., L46), indicating very flat layer distributions; treat as top‑k coverage only, not a fidelity metric [001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:48]. KL is lens‑sensitive but final‑layer KL≈0 and head calibration diagnostics are good; prefer rank milestones for qualitative cross‑model reasoning. Raw‑vs‑norm lens check used mode=sample, so findings are a sampled sanity rather than exhaustive [001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:1015–1075].

6. Model fingerprint (one sentence)

Meta‑Llama‑3‑70B: semantic collapse at L 40; KL≈0 only at L 80; final p_top1≈0.48 with quote co‑activations.

---
Produced by OpenAI GPT-5 

