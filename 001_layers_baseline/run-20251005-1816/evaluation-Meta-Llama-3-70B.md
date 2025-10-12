# Evaluation Report: meta-llama/Meta-Llama-3-70B

**Overview**
- Model: meta-llama/Meta-Llama-3-70B (n_layers=80); run timestamp 2025-10-05 18:16.
- Probe tracks copy vs. semantics using rank/KL/cosine/entropy trajectories with norm lens as default; includes raw-vs-norm lens diagnostics and artifact checks.

**Method Sanity‑Check**
- Prompt & indexing: context ends with “called simply” (no trailing space) — "context_prompt": "Give the city name only, plain text. The capital of Germany is called simply" (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:817). Positive rows exist (e.g., L0 and L40 under `pos,orig`) — see 001_layers_baseline/run-latest/output-Meta-Llama-3-70B-pure-next-token.csv:2 and :42.
- Normalizer provenance: arch="pre_norm", strategy="next_ln1" (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7378–7379). First layer ln_source="blocks[0].ln1" (…/output-Meta-Llama-3-70B.json:7383); final ln_source="ln_final" (…/output-Meta-Llama-3-70B.json:8103).
- Per‑layer normalizer effect: normalization spike flagged and trajectory shape="spike" — "normalization_spike": true (…/output-Meta-Llama-3-70B.json:833); "norm_trajectory": { "shape": "spike", "n_spikes": 15 } (…/output-Meta-Llama-3-70B.json:9024–9028).
- Unembed bias: "present": false, "l2_norm": 0.0 (…/output-Meta-Llama-3-70B.json:826–828). Cosines are bias‑free.
- Environment & determinism: device="cpu", dtype_compute="torch.bfloat16", deterministic_algorithms=true, seed=316 (…/output-Meta-Llama-3-70B.json:8905–8909). Reproducibility OK.
- Numeric health: any_nan=false, any_inf=false, layers_flagged=[] (…/output-Meta-Llama-3-70B.json:8601–8605).
- Copy mask: plausible punctuation/sample and size — "ignored_token_str_sample": ["!", '"', "#", …, ":"] (…/output-Meta-Llama-3-70B.json:7008–7024); "size": 6022 (…/output-Meta-Llama-3-70B.json:7025).
- Gold alignment: { ok: true, variant: "with_space", first_id: 20437, pieces: ["ĠBerlin"] } (…/output-Meta-Llama-3-70B.json:8611–8623).
- Repeatability (1.39): skipped under deterministic env — { "flag": "skipped" } (…/output-Meta-Llama-3-70B.json:9016–9020).
- Norm trajectory: shape="spike", slope=0.0498, r2=0.9367 (…/output-Meta-Llama-3-70B.json:9024–9028).
- Measurement guidance: prefer_ranks=true; suppress_abs_probs=true; preferred_lens_for_reporting="norm"; use_confirmed_semantics=true (…/output-Meta-Llama-3-70B.json:8966–8974).

**Quantitative Findings (Layer‑by‑Layer)**

Short table (positive prompt `pos,orig`; entropy in bits; top‑1 token):

| Layer | Entropy_bits | Top‑1 token | is_answer | answer_rank | CSV row |
|---|---|---|---|---|---|
| 0 | 16.968 | "winding" | False | 115765 | output-Meta-Llama-3-70B-pure-next-token.csv:2 |
| 38 | 16.934 | "simply" | False | 3 | output-Meta-Llama-3-70B-pure-next-token.csv:40 |
| 39 | 16.935 | "simply" | False | 2 | output-Meta-Llama-3-70B-pure-next-token.csv:41 |
| 40 | 16.937 | "Berlin" | True | 1 | output-Meta-Llama-3-70B-pure-next-token.csv:42 |
| 80 | 2.589 | "Berlin" | True | 1 | output-Meta-Llama-3-70B-pure-next-token.csv:82 |

- Semantic layer: L_semantic_norm = L40; confirmed semantics present at L40 with source="raw" (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8970–8979). Bolded row above corresponds to confirmed layer.
- Control margin: first_control_margin_pos=0; max_control_margin=0.5168 (…/output-Meta-Llama-3-70B.json:8958–8959).
- Entropy drift: entropy_gap_bits p25/p50/p75 ≈ 14.33/14.34/14.35 (…/output-Meta-Llama-3-70B.json:9031–9037). Teacher entropy bits=2.5969 (…/output-Meta-Llama-3-70B.json:7197).
- Confidence margins: answer_logit_gap rises from ≈0.031 at L40 (…/output-Meta-Llama-3-70B-pure-next-token.csv:42) to ≈0.564 at L80 (…/output-Meta-Llama-3-70B-pure-next-token.csv:82).
- Normalizer effect snapshots: at L40 resid_norm_ratio=1.2305, delta_resid_cos=0.9774 (…/output-Meta-Llama-3-70B.json:7746–7747); at L80 resid_norm_ratio=1.2479, delta_resid_cos=0.9806 (…/output-Meta-Llama-3-70B.json:8106–8107).

**Qualitative Findings**

4.1. Copy vs semantics (Δ‑gap)
The probe finds no copy reflex in early layers: strict and soft copy milestones are null across thresholds ("L_copy_strict": {0.7: null … 0.95: null}; stability="none") (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7098–7121). Thus Δ̂ is undefined (delta_hat=null) (…/output-Meta-Llama-3-70B.json:8992–9000). This aligns with the layerwise trajectory where "Berlin" only emerges near L38–L40: "… Berlin … Germany …" at L38 (…/output-Meta-Llama-3-70B-pure-next-token.csv:40), becomes top‑2 at L39 (…/output-Meta-Llama-3-70B-pure-next-token.csv:41), and top‑1 with is_answer=True at L40 (…/output-Meta-Llama-3-70B-pure-next-token.csv:42). Copy thresholds show no norm‑only flags at τ∈{0.70,0.95} (…/output-Meta-Llama-3-70B.json:7112–7120).

4.2. Lens sanity: Raw‑vs‑Norm
Dual‑lens checks indicate moderate artifact risk. lens_artifact_score_v2=0.3439; tier="medium" (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7372–7376). Robust metrics: js_divergence_p50≈0.00245; l1_prob_diff_p50≈0.0918; first_js_le_0.1=0; first_l1_le_0.5=0 (…/output-Meta-Llama-3-70B.json:7186–7195). Top‑K overlap: jaccard_raw_norm_p50≈0.515; first_jaccard_raw_norm_ge_0.5 at L11 (…/output-Meta-Llama-3-70B.json:7199–7201). Prevalence: pct_layers_kl_ge_1.0≈0.0123; n_norm_only_semantics_layers=2; earliest_norm_only_semantic=79 (…/output-Meta-Llama-3-70B.json:7179–7182). Note the window audit shows norm‑only semantics at L79–L80 (…/output-Meta-Llama-3-70B.json:7171–7174), but the confirmed semantic onset is at L40 via raw lens, reducing concern about lens‑induced early semantics.

4.3. Tuned‑Lens analysis
Tuned lens is not present for this run — "tuned_lens": { "status": "missing" } (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8961–8962). By guidance, prefer the norm lens for reporting milestones (…/output-Meta-Llama-3-70B.json:8966–8974). Last‑layer agreement is clean: kl_to_final_bits≈0.00073, top1_agree=true; warn_high_last_layer_kl=false (…/output-Meta-Llama-3-70B.json:8623–8641).

4.4. KL, ranks, cosine, entropy milestones
KL crosses below {1.0, 0.5} only at L80 (first_kl_below_1.0=80; first_kl_below_0.5=80) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:7034–7036). Final‑head calibration is good (last‑layer KL≈0) (…/output-Meta-Llama-3-70B.json:8623–8627). Ranks: first_rank_le_{10,5}=38 and first_rank_le_1=40 under the preferred norm lens (…/output-Meta-Llama-3-70B.json:7036–7039). Cosine milestones occur only at the surface (ge_{0.2,0.4,0.6}=80) (…/output-Meta-Llama-3-70B.json:7131–7135). Entropy decays monotonically: from ≈16.968 bits at L0 (…/output-Meta-Llama-3-70B-pure-next-token.csv:2) to ≈2.589 bits at L80 (…/output-Meta-Llama-3-70B-pure-next-token.csv:82), consistent with ranks/KL tightening late.

4.5. Prism (shared‑decoder diagnostic)
Prism is present/compatible. It increases KL at sampled depths (ΔKL p50≈−1.00 bits; higher is worse for prism here) and shows no earlier rank milestones (all null) (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:854–886). Verdict: Regressive.

4.6. Ablation & stress tests
Style ablation is stable: L_sem_orig=40 → L_sem_nf=42 (ΔL_sem=2 ≈2.5% of depth), L_copy remains null (001_layers_baseline/run-latest/output-Meta-Llama-3-70B.json:8922–8929). Control prompt summary present with healthy control margin (first_control_margin_pos=0; max=0.5168) (…/output-Meta-Llama-3-70B.json:8958–8959). Important‑word trajectory: "Berlin" appears in top‑k by L38 (…/output-Meta-Llama-3-70B-pure-next-token.csv:40), strengthens by L39 (…/output-Meta-Llama-3-70B-pure-next-token.csv:41), and is top‑1 by L40 (…/output-Meta-Llama-3-70B-pure-next-token.csv:42); "Germany" is also in the active set around L38–L45 (e.g., …/output-Meta-Llama-3-70B-pure-next-token.csv:40, :45, :46).

4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓
- LayerNorm bias removed ✓
- FP32 unembed promoted ✓
- Punctuation / markup anchoring noted ✓ (e.g., quotes/colon tokens near L38–L45; …/output-Meta-Llama-3-70B-pure-next-token.csv:40–46)
- Copy‑reflex ✗ (no strict/soft hits; …/output-Meta-Llama-3-70B.json:7098–7121)
- Preferred lens honored ✓ (norm)
- Confirmed semantics reported ✓ (L40 raw; …/output-Meta-Llama-3-70B.json:8970–8979)
- Dual‑lens artifact metrics (v2/JS/Jaccard/L1) cited ✓ (…/output-Meta-Llama-3-70B.json:7186–7201, 7368–7376)
- Tuned‑lens audit done n.a. (missing; …/output-Meta-Llama-3-70B.json:8961–8962)
- normalization_provenance present (ln_source @ L0/final) ✓ (…/output-Meta-Llama-3-70B.json:7383, 8103)
- per‑layer normalizer effect present (resid_norm_ratio, delta_resid_cos) ✓ (e.g., L40/L80; …/output-Meta-Llama-3-70B.json:7746–7747, 8106–8107)
- deterministic_algorithms true ✓ (…/output-Meta-Llama-3-70B.json:8907)
- numeric_health clean ✓ (…/output-Meta-Llama-3-70B.json:8601–8605)
- copy_mask plausible ✓ (…/output-Meta-Llama-3-70B.json:7008–7025)
- milestones.csv or evaluation_pack.citations used for quotes ✓ (…/output-Meta-Llama-3-70B.json:9040–9048)

---
Produced by OpenAI GPT-5 

*Run executed on: 2025-10-05 18:16:50*
