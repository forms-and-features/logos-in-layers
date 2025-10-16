# Evaluation Report: Qwen/Qwen3-8B

*Run executed on: 2025-10-16 07:26:19*

**Overview**
- Model: Qwen/Qwen3-8B (8B). Run start: 2025-10-16 (timestamp-20251016-0726).
- Probe measures copy vs. semantics with layer-wise KL-to-final, rank milestones, cosine/entropy trajectories, and dual-lens diagnostics (raw vs. norm, plus tuned/prism), including normalization provenance and gates.

**Method Sanity‑Check**
- Prompt & indexing: context ends with “called simply”; positive/original rows present. "Give the city name only, plain text. The capital of Germany is called simply" (001_layers_baseline/run-latest/output-Qwen3-8B.json:4). Example positive row: layer 31, top‑1 = Berlin (001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:33).
- Normalizer provenance: arch pre_norm, strategy primary=next_ln1 (001_layers_baseline/run-latest/output-Qwen3-8B.json:7380–7383). Per‑layer ln_source L0=blocks[0].ln1 (001_layers_baseline/run-latest/output-Qwen3-8B.json:7387–7392); final stream uses ln_final (unembed_head, 001_layers_baseline/run-latest/output-Qwen3-8B.json:7938–7946).
- Per‑layer normalizer effect: early spike flag present; high delta_resid_cos at L0 (0.9923) before semantics (001_layers_baseline/run-latest/output-Qwen3-8B.json:839–845, 7391–7393). At L31: resid_norm_ratio≈0.252, delta_resid_cos≈0.756 (001_layers_baseline/run-latest/output-Qwen3-8B.json:7663–7670).
- Unembed bias: present=false; l2_norm=0.0 (001_layers_baseline/run-latest/output-Qwen3-8B.json:834–838). Cosine metrics are bias‑free by construction.
- Environment & determinism: device=cpu, torch 2.8, deterministic_algorithms=true, seed=316 (001_layers_baseline/run-latest/output-Qwen3-8B.json:10326–10334).
- Repeatability (forward‑of‑two): mode=skipped_deterministic; pass1.layer=31; pass2.layer=null; gate.repeatability_forward_pass=null (001_layers_baseline/run-latest/output-Qwen3-8B.json:8107–8125).
- Decoding‑point ablation (pre‑norm): gate.decoding_point_consistent=false (001_layers_baseline/run-latest/output-Qwen3-8B.json:8017–8019). At L_semantic_norm=31: rank1_agree=true; jaccard@10=0.176 (001_layers_baseline/run-latest/output-Qwen3-8B.json:7979–7995).
- Numeric health: any_nan=false; any_inf=false; layers_flagged=[] (001_layers_baseline/run-latest/output-Qwen3-8B.json:7948–7956).
- Copy mask: ignored_token_ids listed (e.g., 0,1,2,3,…); plausible tokenizer control IDs (001_layers_baseline/run-latest/output-Qwen3-8B.json:960–976).
- Gold alignment: ok=true; pieces=[ĠBerlin] (001_layers_baseline/run-latest/output-Qwen3-8B.json:8095–8105). gold_alignment_rate=1.0 (001_layers_baseline/run-latest/output-Qwen3-8B.json:8139–8146).
- Repeatability (decode micro‑check §1.39): max_rank_dev=0.0, p95_rank_dev=0.0, top1_flip_rate=0.0 (001_layers_baseline/run-latest/output-Qwen3-8B.json:7952–7958).
- Norm trajectory: shape="spike"; slope≈0.129; r2≈0.922 (001_layers_baseline/run-latest/output-Qwen3-8B.json:9943–9949).
- Measurement guidance: prefer_ranks=true; suppress_abs_probs=true; preferred_lens_for_reporting=tuned; use_confirmed_semantics=true (001_layers_baseline/run-latest/output-Qwen3-8B.json:12186–12201).
- Semantic margin: δ_abs=0.002; margin_ok_at_L_semantic_norm=true (001_layers_baseline/run-latest/output-Qwen3-8B.json:12063–12069).
- Gate‑stability: both_gates_pass_frac at L_sem=1.0; min_both_gates_pass_frac=1.0 (001_layers_baseline/run-latest/output-Qwen3-8B.json:7235–7241).
- Position‑window: grid=[0.2…0.98]; rank1_frac=0.0 at L_sem → position‑fragile (001_layers_baseline/run-latest/output-Qwen3-8B.json:9929–9941).
- Micro‑suite: evaluation_pack.micro_suite.aggregates present; n=5, n_missing=0; L_semantic_confirmed_median=31 (001_layers_baseline/run-latest/output-Qwen3-8B.json:12224–12228, 12304–12314).

**Quantitative Findings (Layer‑by‑Layer)**
- Positive/original trajectory (selected layers):

  | Layer | Entropy_bits | Top‑1 token | Answer_rank | Citation |
  |---|---:|---|---:|---|
  | 0 | 17.21 | CLICK | — | 001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:2 |
  | 29 | 1.78 | ‑minded | 4 | 001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:31 |
  | 30 | 2.20 | Germany | 2 | 001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:32 |
  | **31** | 0.45 | Berlin | 1 | 001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:33 |

  - Bold semantic onset: L 31 — L_semantic_confirmed (raw), decoding‑point sensitive (001_layers_baseline/run-latest/output-Qwen3-8B.json:8017–8019).
- Controls: first_control_margin_pos=1; first_control_strong_pos=30 (001_layers_baseline/run-latest/output-Qwen3-8B.json:10382–10386).
- Micro‑suite: median L_semantic_confirmed=31; median Δ̂ across facts=0.0556 (001_layers_baseline/run-latest/output-Qwen3-8B.json:12085–12090). Example fact citation: Germany→Berlin row_index=31 → pure CSV line 33 (001_layers_baseline/run-latest/output-Qwen3-8B.json:12304–12310; 001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:33).
- Entropy drift: entropy_gap_bits p25=1.66, p50=13.79, p75=14.01 (001_layers_baseline/run-latest/output-Qwen3-8B.json:12232–12238).
- Normalizer snapshot at semantics: L31 resid_norm_ratio≈0.252; delta_resid_cos≈0.756 (001_layers_baseline/run-latest/output-Qwen3-8B.json:7663–7670).

**Qualitative Findings**

4.1. Copy vs semantics (Δ‑gap)
- No early copy‑reflex: layers 0–3 have copy_collapse=False and no early soft‑copy hits in the positive/original trajectory (e.g., L0 row shows copy flags False; 001_layers_baseline/run-latest/output-Qwen3-8B-pure-next-token.csv:2).
- Strict copy thresholds: L_copy_strict@{0.70,0.95}=null; norm_only_flags also null; stability="none" (001_layers_baseline/run-latest/output-Qwen3-8B.json:7150–7175). With no detected copy onset, Δ̂ is not defined here; semantics occurs late (semantic_frac≈0.861; 001_layers_baseline/run-latest/output-Qwen3-8B.json:12219–12222).

4.2. Lens sanity: Raw‑vs‑Norm
- Artifact tier: lens_artifact_score_v2=0.704 → tier=high (001_layers_baseline/run-latest/output-Qwen3-8B.json:7373–7377). Symmetric metrics: js_divergence_p50=0.358; l1_prob_diff_p50=1.134; first_js_le_0.1=0; first_l1_le_0.5=0 (001_layers_baseline/run-latest/output-Qwen3-8B.json:12228–12233).
- Top‑K overlap: jaccard_raw_norm_p50=0.282; first_jaccard_raw_norm_ge_0.5=0 (001_layers_baseline/run-latest/output-Qwen3-8B.json:12232–12237).
- Prevalence: pct_layers_kl_ge_1.0≈0.757; n_norm_only_semantics_layers=0; earliest_norm_only_semantic=null (001_layers_baseline/run-latest/output-Qwen3-8B.json:12234–12238).
- Lens‑consistency at semantic target is low (norm vs raw): jaccard@10=0.176; jaccard@50=0.370; spearman_top50≈0.341 at L=31 (001_layers_baseline/run-latest/output-Qwen3-8B.json:8043–8047). Caution: early “semantics” is view‑dependent; prefer ranks/KL and confirmed semantics.

4.3. Tuned‑Lens analysis
- Preference: tuned_is_calibration_only=false; preferred_semantics_lens_hint=tuned (001_layers_baseline/run-latest/output-Qwen3-8B.json:12196–12207; 12060–12069). Use confirmed semantics for reporting.
- Attribution: ΔKL_rot_p25/p50/p75≈0.81/0.92/1.01; ΔKL_temp_p25/p50/p75≈0.00/0.03/0.05; interaction_p50≈2.82 (001_layers_baseline/run-latest/output-Qwen3-8B.json:12260–12274).
- Rank earliness: first_rank_le_10 baseline=29 vs tuned=30; le_5 baseline=29 vs tuned=31; le_1 baseline=31 vs tuned=34 (001_layers_baseline/run-latest/output-Qwen3-8B.json:11916–11936). No earlier rank‑1 under tuned.
- Positional generalization: pos_in_dist_le_0.92≈5.00; pos_ood_ge_0.96≈3.77; pos_ood_gap≈−1.23 (001_layers_baseline/run-latest/output-Qwen3-8B.json:12020–12030).
- Head mismatch: kl_bits_tuned_final=0.0; tau_star_modelcal=1.0 (001_layers_baseline/run-latest/output-Qwen3-8B.json:12031–12038). Last‑layer consistency OK: kl_to_final_bits=0.0; top1_agree=true (001_layers_baseline/run-latest/output-Qwen3-8B.json:8127–8135).

4.4. KL, ranks, cosine, entropy milestones
- KL: first_kl_le_1.0 baseline=36; final KL≈0 (well‑calibrated final head) (001_layers_baseline/run-latest/output-Qwen3-8B.json:11936–11943, 8127–8135).
- Ranks (preferred lens=tuned; baseline in parentheses): le_10=30 (29); le_5=31 (29); le_1=34 (31) (001_layers_baseline/run-latest/output-Qwen3-8B.json:11916–11936). Uniform‑margin gate passes at L_semantic_norm=31 (001_layers_baseline/run-latest/output-Qwen3-8B.json:12063–12069). No strong/run‑of‑two milestone; treat onset as potentially unstable across seeds (forward‑of‑two skipped).
- Cosine: norm cos milestones ge_{0.2,0.4,0.6} all at L=36 (001_layers_baseline/run-latest/output-Qwen3-8B.json:7188–7193), indicating late alignment in cosine space.
- Entropy: large positive entropy_gap relative to teacher (p50≈13.79 bits; 001_layers_baseline/run-latest/output-Qwen3-8B.json:12232–12238); entropy collapses sharply by L30–31 where ranks improve.
- Advisory gates: decoding‑point gate fails (pre_norm), so the L=31 onset is decoding‑point sensitive (001_layers_baseline/run-latest/output-Qwen3-8B.json:8017–8019). Position‑window rank1_frac=0.0 → position‑fragile (001_layers_baseline/run-latest/output-Qwen3-8B.json:9938–9941).

4.5. Prism (shared‑decoder diagnostic)
- Presence/compatibility: prism present, k=512, sampled layers [embed,8,17,26] (001_layers_baseline/run-latest/output-Qwen3-8B.json:846–859).
- KL deltas (delta field): p25≈−0.36, p50≈−0.59, p75≈−7.03 bits (higher KL under prism; worse) (001_layers_baseline/run-latest/output-Qwen3-8B.json:902–931). No earlier rank milestones (le_1 unchanged/null) (001_layers_baseline/run-latest/output-Qwen3-8B.json:886–900).
- Verdict: Regressive.

4.6. Ablation & stress tests
- No‑filler ablation: L_sem_orig=31; L_sem_nf=31; ΔL_sem=0 (001_layers_baseline/run-latest/output-Qwen3-8B.json:10388–10390).
- Control prompt “Berlin is the capital of”: top‑1 token = “ Germany”; “Berlin” appears in top‑10 but not top‑1 (001_layers_baseline/run-latest/output-Qwen3-8B.json:12–20, 31–34).
- Important‑word trajectory (records): at L=31,pos=15, top‑1 “Berlin” (001_layers_baseline/run-latest/output-Qwen3-8B-records.csv:610). Earlier positions emphasize “Germany/located/called” (e.g., L=31,pos=12–14; 001_layers_baseline/run-latest/output-Qwen3-8B-records.csv:607–609).

4.7. Checklist (✓/✗/n.a.)
- RMS lens ✓
- LayerNorm bias removed ✓
- FP32 unembed promoted ✓
- Punctuation / markup anchoring noted ✓
- Copy‑reflex ✗
- Preferred lens honored ✓
- Confirmed semantics reported ✓
- Dual‑lens artefact metrics (incl. lens_artifact_score_v2, JS/Jaccard/L1) cited ✓
- Tuned‑lens audit done (rotation/temp/positional/head) ✓
- normalization_provenance present (ln_source @ L0/final) ✓
- per‑layer normalizer effect present (resid_norm_ratio, delta_resid_cos) ✓
- deterministic_algorithms true ✓
- numeric_health clean ✓
- copy_mask plausible ✓
- milestones.csv or evaluation_pack.citations used for quotes ✓
- gate_stability_small_scale reported ✓
- position_window stability reported ✓

---
Produced by OpenAI GPT-5
