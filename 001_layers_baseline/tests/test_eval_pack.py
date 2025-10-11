#!/usr/bin/env python3

import _pathfix  # noqa: F401

from layers_core.eval_pack import build_evaluation_pack


def _pure_record(layer: int, answer_rank: int, p_answer: float, kl_bits: float, entropy_bits: float) -> dict:
    return {
        "prompt_id": "pos",
        "prompt_variant": "orig",
        "layer": layer,
        "pos": 5,
        "token": "⟨NEXT⟩",
        "entropy": entropy_bits,
        "entropy_bits": entropy_bits,
        "answer_rank": answer_rank,
        "p_answer": p_answer,
        "kl_to_final_bits": kl_bits,
    }


def test_build_evaluation_pack_minimal():
    json_data = {
        "pure_next_token_records": [
            _pure_record(0, 20, 0.01, 4.0, 3.0),
            _pure_record(1, 5, 0.10, 2.0, 2.5),
            _pure_record(2, 1, 0.50, 0.8, 2.0),
        ],
        "raw_lens_full_records": [
            {
                "prompt_id": "pos",
                "prompt_variant": "orig",
                "layer": 1,
                "js_divergence": 0.05,
                "kl_raw_to_norm_bits": 1.2,
                "l1_prob_diff": 0.4,
                "topk_jaccard_raw_norm@50": 0.6,
            }
        ],
    }
    diag = {
        "L_copy": 0,
        "copy_detector": {"soft": {"L_copy_soft": {"k1": 1}}},
        "L_semantic": 2,
        "confirmed_semantics": {"L_semantic_confirmed": 2, "confirmed_source": "both"},
        "raw_lens_full": {
            "pct_layers_kl_ge_1.0": 0.5,
            "n_norm_only_semantics_layers": 1,
            "earliest_norm_only_semantic": 2,
            "max_kl_norm_vs_raw_bits": 1.3,
            "js_divergence_percentiles": {"p50": 0.05},
            "l1_prob_diff_percentiles": {"p50": 0.4},
            "first_js_le_0.1": 1,
            "first_l1_le_0.5": 1,
            "topk_overlap": {"jaccard_raw_norm_p50": 0.6, "first_jaccard_raw_norm_ge_0.5": 1},
            "score": {"lens_artifact_score": 0.2, "lens_artifact_score_v2": 0.25, "tier": "medium"},
        },
        "repeatability": {"status": "ok", "max_rank_dev": 1.0, "p95_rank_dev": 0.5, "top1_flip_rate": 0.01},
        "gold_alignment_rate": 0.75,
        "gold_alignment": {"variant": "with_space"},
        "norm_trajectory": {"shape": "monotonic", "slope": 0.2, "r2": 0.9, "n_spikes": 0},
        "entropy_gap_bits_percentiles": {"p25": 0.1, "p50": 0.2, "p75": 0.3},
    }
    mg = {"preferred_lens_for_reporting": "norm", "use_confirmed_semantics": True}
    tuned_audit_summary = {"rotation_vs_temperature": {}, "positional": {}, "head_mismatch": {}, "tuned_is_calibration_only": False}
    pack, milestones_rows, artifact_rows = build_evaluation_pack(
        model_name="TestModel",
        n_layers=4,
        json_data=json_data,
        json_data_tuned=None,
        diag=diag,
        measurement_guidance=mg,
        tuned_audit_summary=tuned_audit_summary,
        tuned_audit_data=None,
        clean_name="TestModel",
    )

    assert pack["model"] == "TestModel"
    assert pack["milestones"]["L_copy_strict"] == 0
    assert pack["milestones"]["L_copy_soft"]["layer"] == 1
    assert pack["milestones"]["depth_fractions"]["delta_hat"] == 0.5
    assert pack["artifact"]["risk_tier"] == "medium"
    assert pack["repeatability"]["flag"] == "ok"
    assert pack["citations"]["layers"]["L_semantic_norm_row"] == 2
    assert pack["citations"]["files"]["tuned_pure_csv"] is None
    assert len(milestones_rows) == 4
    assert milestones_rows[0]["layer"] == 0
    assert milestones_rows[1]["is_copy_soft_k"] in ("", 1)
    assert artifact_rows[0]["layer"] == "_summary"


def test_build_evaluation_pack_micro_suite():
    json_data = {
        "pure_next_token_records": [
            _pure_record(0, 20, 0.01, 4.0, 3.0),
            _pure_record(1, 5, 0.10, 2.0, 2.5),
            _pure_record(2, 1, 0.50, 0.8, 2.0),
        ],
        "raw_lens_full_records": [],
    }
    diag = {
        "L_copy": 0,
        "copy_detector": {"soft": {"L_copy_soft": {"k1": 1}}},
        "L_semantic": 2,
        "micro_suite": {
            "facts": [
                {
                    "fact_key": "Germany→Berlin",
                    "fact_index": 0,
                    "L_copy_strict": 10,
                    "L_copy_soft_k1": 10,
                    "L_semantic_norm": 25,
                    "L_semantic_confirmed": 25,
                    "L_semantic_margin_ok_norm": 25,
                    "delta_hat": 0.2,
                    "row_index": 1,
                }
            ],
            "aggregates": {
                "L_semantic_confirmed_median": 25,
                "delta_hat_median": 0.2,
                "n_missing": 0,
                "n": 1,
            },
            "citations": {"fact_rows": {"Germany→Berlin": 1}},
        },
        "raw_lens_full": {},
        "repeatability": {},
    }
    pack, _, _ = build_evaluation_pack(
        model_name="MicroTest",
        n_layers=4,
        json_data=json_data,
        json_data_tuned=None,
        diag=diag,
        measurement_guidance={},
        tuned_audit_summary={},
        tuned_audit_data=None,
        clean_name="MicroTest",
    )
    assert "micro_suite" in pack
    ms = pack["micro_suite"]
    assert ms["facts"][0]["fact_key"] == "Germany→Berlin"
    assert ms["aggregates"]["L_semantic_confirmed_median"] == 25
    assert ms["aggregates"]["delta_hat_median"] == 0.2
    assert ms["citations"]["fact_rows"]["Germany→Berlin"] == 1
