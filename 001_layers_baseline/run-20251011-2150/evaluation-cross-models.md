# CROSS‑EVAL: Layer‑by‑layer probes across models

This cross‑model synthesis reads the latest JSON/CSV packs under `001_layers_baseline/run-latest`, using the script `001_layers_baseline/run.py` as implementation reference. Claims are grounded in per‑model JSON/CSVs and the evaluation packs. Following measurement_guidance, ranks/KL thresholds are preferred and absolute probabilities are avoided where suppressed.

## 1) Result synthesis

Across models, confirmed semantic layers cluster at the final third of depth, with family‑specific shifts and lens risk caveats:
- Qwen3‑8B: first_rank milestones `≤10/≤5/≤1 = 29/29/31`; confirmed semantics at L=31 (source=raw) (output‑Qwen3‑8B.json:7069,7088,7087,11748). Micro‑suite median Δ̂≈0.0556 (evaluation_pack) (output‑Qwen3‑8B.json:11813).
- Qwen3‑14B: `≤10/≤5/≤1 = 32/33/36`; confirmed semantics L=36 (raw) (output‑Qwen3‑14B.json:7093,7092,7091,9580). Tuned audit shows rotation dominates temperature (delta_kl_rot_p50≈1.97; temp_p50≈−0.0003) with pos_ood_gap≈−0.49 (output‑Qwen3‑14B.json:11640,11654,11667,11672).
- Qwen2.5‑72B: `≤10/≤5/≤1 = 74/78/80`; semantics at L=80 with no confirmed source (use_confirmed_semantics=false) (output‑Qwen2.5‑72B.json:7265,7264,7263,9076,9728).
- Meta‑Llama‑3‑8B: `≤10/≤5/≤1 = 24/25/25`; confirmed semantics L=25 (raw), but margin gate at L_semantic_norm fails (weak norm milestone) (output‑Meta‑Llama‑3‑8B.json:6996,6995,6994,11586,7110).
- Meta‑Llama‑3‑70B: `≤10/≤5/≤1 = 38/38/40`; confirmed semantics L=40 (raw); margin gate at norm is weak (output‑Meta‑Llama‑3‑70B.json:7044,7043,7042,9546,7157).
- Mistral‑7B‑v0.1: `≤10/≤5/≤1 = 23/25/25`; confirmed semantics L=25 (raw) (output‑Mistral‑7B‑v0.1.json:2152,2151,2150,6751).
- Mistral‑Small‑24B‑Base‑2501: `≤10/≤5/≤1 = 30/30/33`; confirmed semantics L=33 (raw) (output‑Mistral‑Small‑24B‑Base‑2501.json:3913,3912,3911,6405).
- Gemma‑2‑9B: `≤10/≤5/≤1 = 42/42/42`; confirmed semantics L=42; Δ̂ median=1.0 (copy at L0) (output‑gemma‑2‑9b.json:5651,5650,5649,8174,10382,10396).
- Gemma‑2‑27B: `≤10/≤5/≤1 = 46/46/46`; confirmed semantics L=46; Δ̂ median=1.0 (copy at L0); tuned_is_calibration_only=true (output‑gemma‑2‑27b.json:5655,5654,5653,8246,8332,10342).
- Yi‑34B: `≤10/≤5/≤1 = 43/44/44`; confirmed semantics L=44 (tuned preferred); margin gate OK (output‑Yi‑34B.json:2514,2513,2512,5353,2627).

Trends: Qwen and Llama families reach rank‑1 around 0.75–0.9 depth fraction, Gemma saturates at the final layer with robust margin gates, and Mistral sits mid‑pack by normalized depth. Tuned‑lens audits where present attribute most KL change to rotation rather than temperature in Qwen (output‑Qwen3‑8B.json:11560; output‑Qwen3‑14B.json:11643), while Gemma‑27B marks tuned as calibration‑only (output‑gemma‑2‑27b.json:10342).

## 2) Copy reflex (layers 0–3)

Copy‑reflex is observed in Gemma models at L0; others show no early copy flags:
- Gemma‑2‑9B: `L_copy_strict = 0` and `L_copy_soft(k=1) = 0`; micro‑suite medians confirm copy at L0 (output‑gemma‑2‑9b.json:5672,10396,10375).
- Gemma‑2‑27B: `L_copy_strict = 0` and `L_copy_soft(k=1) = 0`; micro‑suite medians confirm (output‑gemma‑2‑27b.json:5676,10396,8329).
- Qwen3‑8B (example CSV rows 0–3): `copy_collapse=False`, `copy_soft_k1@0.5=False` at early layers (output‑Qwen3‑8B‑pure‑next‑token.csv:1).

Within‑family, Gemma exhibits a strong copy‑reflex normalized to 0/n_layers, yielding Δ̂≈1.0 medians (output‑gemma‑2‑9b.json:10382; output‑gemma‑2‑27b.json:8332). Other families show no early copy collapse under these prompts.

## 3) Lens artefact risk

Raw‑vs‑Norm diagnostic metrics and artifact tiers:
- Qwen3‑8B: `lens_artifact_score_v2=0.704` (tier=high), `js_p50=0.358`, `l1_p50=1.134`, `jaccard_p50=0.282`, `pct_kl≥1.0=0.757`, `n_norm_only=0` (output‑Qwen3‑8B.json:7220,7240,7260,7260,7220).
- Qwen3‑14B: `v2=0.704` (high), `js_p50=0.513`, `l1_p50=1.432`, `jaccard_p50=0.250`, `pct_kl≥1.0=0.756`, `n_norm_only=0` (output‑Qwen3‑14B.json:11810,11819,11819).
- Qwen2.5‑72B: `v2=0.743` (high), `js_p50=0.105`, `l1_p50=0.615`, `jaccard_p50=0.316`, `pct_kl≥1.0=0.321`, `n_norm_only=1@80` (output‑Qwen2.5‑72B.json:9728,9766,9767,9773,9754).
- Meta‑Llama‑3‑8B: `v2=0.459` (medium), `js_p50=0.0168`, `l1_p50=0.240`, `jaccard_p50=0.408`, `pct_kl≥1.0=0.0303`, `n_norm_only=5` (output‑Meta‑Llama‑3‑8B.json:11598,11603,11603,11606).
- Meta‑Llama‑3‑70B: `v2=0.344` (medium), `js_p50=0.00245`, `l1_p50=0.0918`, `jaccard_p50=0.515`, `pct_kl≥1.0=0.0123`, `n_norm_only=2` (output‑Meta‑Llama‑3‑70B.json:9559,9569,9569).
- Mistral‑7B‑v0.1: `v2=0.670` (high), `js_p50=0.0741`, `l1_p50=0.505`, `jaccard_p50=0.408`, `pct_kl≥1.0=0.242`, `n_norm_only=1@32` (output‑Mistral‑7B‑v0.1.json:6754,6755,6765).
- Mistral‑Small‑24B: `v2=0.185` (low), `js_p50 not shown in pack excerpt`, `pct_kl≥1.0` low; risk tier low (output‑Mistral‑Small‑24B‑Base‑2501.json:4173,6880).
- Gemma‑2‑9B: `v2=0.591` (high), `js_p50=0.00625`, `l1_p50=0.0292`, `jaccard_p50=0.639`, `pct_kl≥1.0=0.302`, `n_norm_only=1@42` (output‑gemma‑2‑9b.json:10424,10440).
- Gemma‑2‑27B: `v2=1.00` (high), `js_p50=0.865`, `l1_p50=1.893`, `jaccard_p50=0.563`, `pct_kl≥1.0=0.979`, `n_norm_only=1@46` (output‑gemma‑2‑27b.json:10504,10516).
- Yi‑34B: `v2=0.943` (high), `js_p50≈0.28`, `l1_p50≈0.47`, `jaccard_p50≈0.33`, `pct_kl≥1.0` high; risk tier high (output‑Yi‑34B.json:2819,5827).

Interpretation: under high‑risk tiers, we rely on rank/KL milestones and confirmed semantics; where norm‑only layers exist near semantic onset, we avoid strong pre‑final conclusions.

## 4) Confirmed semantics

- Qwen3‑8B: L_semantic_confirmed=31 (source=raw); micro‑suite median confirmed L=31 with citations (Germany→Berlin row 31) (output‑Qwen3‑8B.json:11748,11813,11873).
- Qwen3‑14B: L_semantic_confirmed=36 (raw); micro‑suite median confirmed L=36 (Germany→Berlin row 36) (output‑Qwen3‑14B.json:9580,9666,9680).
- Qwen2.5‑72B: L_semantic_norm=80; confirmed absent (source=null); use ranks only (output‑Qwen2.5‑72B.json:9754,9080).
- Meta‑Llama‑3‑8B: L_semantic_confirmed=25 (raw); margin gate fails at L_semantic_norm (weak norm milestone); fact citation row 25 (output‑Meta‑Llama‑3‑8B.json:11586,7110,11694).
- Meta‑Llama‑3‑70B: L_semantic_confirmed=40 (raw); margin gate fails at norm; fact citation row 40 (output‑Meta‑Llama‑3‑70B.json:9546,7157,9597).
- Mistral‑7B‑v0.1: L_semantic_confirmed=25 (raw); median confirmed L=25 (Germany→Berlin row 25) (output‑Mistral‑7B‑v0.1.json:6751,6892).
- Mistral‑Small‑24B: L_semantic_confirmed=33 (raw); median confirmed L=33 (row 33) (output‑Mistral‑Small‑24B‑Base‑2501.json:6405,6489).
- Gemma‑2‑9B: L_semantic_confirmed=42; margin gate OK; citations row 42 (output‑gemma‑2‑9b.json:8174,10292,10396).
- Gemma‑2‑27B: L_semantic_confirmed=46; tuned is calibration‑only; citations row 46 (output‑gemma‑2‑27b.json:8246,10342,10449).
- Yi‑34B: L_semantic_confirmed=44; margin gate OK; citations row 44 (output‑Yi‑34B.json:5353,2627,7695).

Uniform‑margin gate: we treat norm milestones as weak when `margin_ok_at_L_semantic_norm=false` (e.g., Llama‑3 family) and prefer confirmed semantics.

## 5) Entropy and confidence

Entropy drift relative to teacher typically shrinks as rank improves and KL falls:
- Median entropy‑gap bits (evaluation_pack): Qwen3‑8B≈13.79 (output‑Qwen3‑8B.json:11788), Qwen3‑14B≈13.40 (output‑Qwen3‑14B.json:11863), Qwen2.5‑72B≈12.50 (output‑Qwen2.5‑72B.json:9797), Llama‑3‑8B≈13.88 (output‑Meta‑Llama‑3‑8B.json:11629), Llama‑3‑70B≈14.34 (output‑Meta‑Llama‑3‑70B.json:9589), Mistral‑7B≈10.99 (output‑Mistral‑7B‑v0.1.json:6785), Mistral‑24B≈13.59 (output‑Mistral‑Small‑24B‑Base‑2501.json:8685), Yi‑34B≈12.59 (output‑Yi‑34B.json:7655). Gemma shows atypical gaps due to head calibration (9B p50 negative; 27B lower), consistent with family‑level last‑layer calibration issues (output‑gemma‑2‑9b.json:10459; output‑gemma‑2‑27b.json:10536).

## 6) Normalization and numeric health

All runs detect pre‑norm architectures with `next_ln1` strategy (e.g., Qwen3‑8B, Llama‑3‑8B, Yi‑34B) and record early normalization spikes:
- Provenance examples: `arch: "pre_norm"`, `strategy: "next_ln1"` (output‑Qwen3‑8B.json:7246–7248; output‑Meta‑Llama‑3‑8B.json:7246–7248; output‑Yi‑34B.json:2824). Run‑level `normalization_spike=true` is set for several models (e.g., Qwen/Llama) (output‑Qwen3‑8B.json:834; output‑Meta‑Llama‑3‑8B.json:839).
- Numeric health: all models report `any_nan=false`, `any_inf=false`, with no flagged layers (e.g., Qwen3‑8B, Llama‑3‑8B) (output‑Qwen3‑8B.json:7905; output‑Meta‑Llama‑3‑8B.json:7756).

## 7) Repeatability

Repeatability benches are skipped across runs due to deterministic setup; treat near‑threshold rank differences cautiously: `status="skipped"`, `reason="deterministic_env"` (e.g., Qwen3‑8B, Mistral‑7B, Gemma‑2‑27B) (output‑Qwen3‑8B.json:7912; output‑Mistral‑7B‑v0.1.json:2908; output‑gemma‑2‑27b.json:6647).

## 8) Family patterns

- Qwen (3‑8B/14B, 2.5‑72B): Consistent late semantics (≥0.8 depth fraction) with high lens‑artifact tiers; tuned audits attribute most gains to rotation; head mismatch calibrated (τ*≈1.0) and final KL≈0 (output‑Qwen3‑8B.json:11560,11597; output‑Qwen3‑14B.json:11640,11672; output‑Qwen2.5‑72B.json:8865).
- Llama‑3 (8B/70B): Medium lens‑risk; norm milestones weak under uniform‑margin gate but confirmed semantics stable at 25/40; final‑head consistency clean (KL≈0) (output‑Meta‑Llama‑3‑8B.json:11598,7110,7772; output‑Meta‑Llama‑3‑70B.json:9559,7157,8656).
- Mistral (7B/24B): 7B shows high lens‑risk; 24B is low‑risk with later semantics (33) and tighter raw‑norm agreement (output‑Mistral‑7B‑v0.1.json:6754; output‑Mistral‑Small‑24B‑Base‑2501.json:4173,6405).
- Gemma‑2 (9B/27B): Strong copy‑reflex at L0 and semantics at final layer with margin OK; last‑layer KL warns of head calibration (warn_high_last_layer_kl) especially for 27B where tuned is calibration‑only (output‑gemma‑2‑9b.json:6591,6609,8170; output‑gemma‑2‑27b.json:6663,6681,8242,10342).
- Yi‑34B: High lens‑risk but stable confirmed semantics at L=44; prism helps (see §10) (output‑Yi‑34B.json:5827,5353,840).

## 10) Prism summary across models

Prism behaves as a shared‑decoder diagnostic with mixed outcomes:
- Helpful (KL deltas positive = lower prism KL): Gemma‑2‑27B (p50≈+23.73), Qwen2.5‑72B (p50≈+2.83), Yi‑34B (p50≈+1.36) (output‑gemma‑2‑27b.json:888–906; output‑Qwen2.5‑72B.json:871–889; output‑Yi‑34B.json:871–889).
- Regressive: Qwen3‑8B (p50≈−0.59), Qwen3‑14B (≈−0.25), Llama‑3‑8B (≈−0.59), Llama‑3‑70B (≈−1.00), Mistral‑7B (≈−17.54), Mistral‑24B (≈−5.98), Gemma‑2‑9B (≈−10.33) (output‑Qwen3‑8B.json:872–892; output‑Qwen3‑14B.json:871–889; output‑Meta‑Llama‑3‑8B.json:872–889; output‑Meta‑Llama‑3‑70B.json:872–889; output‑Mistral‑7B‑v0.1.json:871–889; output‑Mistral‑Small‑24B‑Base‑2501.json:872–889; output‑gemma‑2‑9b.json:871–889).

Interpretation: larger models with strong head calibration (Yi‑34B, Qwen2.5‑72B, Gemma‑27B) tend to see KL reductions under Prism; smaller/earlier‑stage models regress.

## 11) Within‑family similarities and differences

- Qwen: 8B and 14B confirm semantics earlier (31/36) than 72B (80), consistent with depth scaling. Both 8B/14B show high lens‑risk and prefer tuned lens for reporting; audits indicate rotation‑led improvements and τ*≈1.0 head calibration (output‑Qwen3‑8B.json:11720,11569; output‑Qwen3‑14B.json:11795,11667). 72B lacks confirmed semantics in this pack; conclusions rely on ranks/KL only (output‑Qwen2.5‑72B.json:9076).
- Gemma: Both 9B and 27B show copy at L0 and confirm semantics at the final layer with strong margins; last‑layer KL warnings persist (warn_high_last_layer_kl=true) and, for 27B, tuned is calibration‑only with τ*>1 (output‑gemma‑2‑9b.json:6560,6609,8174,10490; output‑gemma‑2‑27b.json:6636,6681,8246,10342).

## 13) Misinterpretations in existing EVALS

- Meta‑Llama‑3‑8B EVAL overstates Prism degradation: “KL deltas are negative (higher KL) at p50 (≈+8.29 bits vs baseline)” (001_layers_baseline/run-latest/evaluation‑Meta‑Llama‑3‑8B.md:63). The JSON shows p50 change ≈−0.59 (baseline 12.41 → prism 13.00; delta −0.588) (output‑Meta‑Llama‑3‑8B.json:872–889). Treat as minor; overall verdict “Regressive” still holds.
- Minor guidance violation: Qwen3‑8B EVAL quotes a near‑1 control margin (“max_control_margin: 0.9999977…”) despite `suppress_abs_probs=true` (001_layers_baseline/run-latest/evaluation‑Qwen3‑8B.md:27; output‑Qwen3‑8B.json:11720). Prefer rank/KL statements under high risk tiers.

## Notes and citations

- Diagnostics flags: final‑head calibration warnings in Gemma (warn_high_last_layer_kl=true) imply rank/KL focus at final layer (output‑gemma‑2‑9b.json:6591; output‑gemma‑2‑27b.json:6663).
- Gold alignment: All reported models show `gold_alignment_rate = 1.0` where present; Qwen2.5‑72B lacks confirmed semantics but alignment is ok (output‑Qwen3‑8B.json:7950; output‑Qwen3‑14B.json:—; output‑Qwen2.5‑72B.json:9736).
- Example fact citations: `Germany→Berlin` rows used above: Qwen3‑8B row 31 (output‑Qwen3‑8B.json:11873), Qwen3‑14B row 36 (output‑Qwen3‑14B.json:9672), Llama‑3‑8B row 25 (output‑Meta‑Llama‑3‑8B.json:11694), Llama‑3‑70B row 40 (output‑Meta‑Llama‑3‑70B.json:9597), Mistral‑7B row 25 (output‑Mistral‑7B‑v0.1.json:6892), Mistral‑24B row 33 (output‑Mistral‑Small‑24B‑Base‑2501.json:6426), Gemma‑2‑9B row 42 (output‑gemma‑2‑9b.json:10396), Gemma‑2‑27B row 46 (output‑gemma‑2‑27b.json:10449), Yi‑34B row 44 (output‑Yi‑34B.json:7695).

---
**Produced by OpenAI GPT-5**
---

