#!/usr/bin/env python3
"""Unit tests for summaries helpers (CPU-only)."""

import _pathfix  # noqa: F401

from layers_core.summaries import summarize_pure_records


def test_summarize_pure_records():
    recs = [
        {
            "layer": 0,
            "copy_collapse": False,
            "entropy_collapse": False,
            "is_answer": False,
            "kl_to_final_bits": 2.0,
            "answer_rank": 100,
            "copy_soft_hits": {1: False, 2: False},
            "cos_to_final": 0.1,
            "entropy_bits": 3.2,
            "teacher_entropy_bits": 3.0,
            "answer_minus_uniform": -0.001,
            "semantic_margin_ok": False,
        },
        {
            "layer": 1,
            "copy_collapse": True,
            "entropy_collapse": False,
            "is_answer": False,
            "kl_to_final_bits": 1.2,
            "answer_rank": 50,
            "copy_soft_hits": {1: True, 2: False},
            "cos_to_final": 0.25,
            "entropy_bits": 2.9,
            "teacher_entropy_bits": 2.8,
            "answer_minus_uniform": -0.0005,
            "semantic_margin_ok": False,
        },
        {
            "layer": 2,
            "copy_collapse": False,
            "entropy_collapse": True,
            "is_answer": False,
            "kl_to_final_bits": 0.8,
            "answer_rank": 10,
            "copy_soft_hits": {1: False, 2: True},
            "cos_to_final": 0.45,
            "entropy_bits": 2.4,
            "teacher_entropy_bits": 2.7,
            "answer_minus_uniform": 0.0001,
            "semantic_margin_ok": False,
        },
        {
            "layer": 3,
            "copy_collapse": False,
            "entropy_collapse": False,
            "is_answer": True,
            "kl_to_final_bits": 0.3,
            "answer_rank": 1,
            "copy_soft_hits": {1: False, 2: False},
            "cos_to_final": 0.65,
            "entropy_bits": 2.1,
            "teacher_entropy_bits": 2.5,
            "answer_minus_uniform": 0.01,
            "semantic_margin_ok": True,
            "p_answer": 0.012,
        },
    ]

    diag = summarize_pure_records(
        recs,
        copy_threshold=0.95,
        copy_window_k=1,
        copy_soft_threshold=0.33,
        copy_soft_window_ks=(1, 2),
        copy_match_level="id_subsequence",
        n_layers=4,
        p_uniform=0.001,
    )

    assert diag["L_copy"] == 1
    assert diag["L_copy_H"] == 2
    assert diag["L_semantic"] == 3
    assert diag["delta_layers"] == 2
    assert diag["L_copy_soft"][1] == 1
    assert diag["L_copy_soft"][2] == 2
    assert diag["delta_layers_soft"][1] == 2
    assert diag["delta_layers_soft"][2] == 1
    assert diag["copy_detector"]["soft"]["L_copy_soft"]["k1"] == 1
    assert diag["copy_detector"]["deltas"]["Δ_sem_minus_copy_soft"]["k2"] == 1
    assert diag["first_kl_below_0.5"] == 3
    assert diag["first_kl_below_1.0"] == 2
    assert diag["first_rank_le_1"] == 3
    assert diag["first_rank_le_5"] is None  # no rank<=5 until layer 3 (which is <=1 already accounted)
    assert diag["first_rank_le_10"] == 2

    cos_milestones = diag["cos_milestones"]["norm"]
    assert cos_milestones == {"ge_0.2": 1, "ge_0.4": 2, "ge_0.6": 3}

    depth = diag["depth_fractions"]
    assert depth["L_semantic_frac"] == 0.75
    assert depth["first_rank_le_5_frac"] is None
    assert depth["L_copy_strict_frac"] == 0.25
    assert depth["L_copy_soft_k1_frac"] == 0.25
    assert depth["L_copy_soft_k2_frac"] == 0.5

    # Threshold sweep block exists with expected structure
    ct = diag.get("copy_thresholds")
    assert isinstance(ct, dict)
    assert ct.get("tau_list") == [0.7, 0.8, 0.9, 0.95]
    assert set(ct.get("L_copy_strict", {}).keys()) == {"0.7", "0.8", "0.9", "0.95"}
    assert set(ct.get("L_copy_strict_frac", {}).keys()) == {"0.7", "0.8", "0.9", "0.95"}
    assert set(ct.get("norm_only_flags", {}).keys()) == {"0.7", "0.8", "0.9", "0.95"}
    # stability is one of allowed values
    assert ct.get("stability") in {"stable", "mixed", "fragile", "none"}

    entropy_summary = diag.get("entropy_gap_bits_percentiles")
    assert isinstance(entropy_summary, dict)
    assert set(entropy_summary.keys()) == {"p25", "p50", "p75"}

    assert diag["L_semantic_margin_ok"] == 3
    sem_margin = diag["semantic_margin"]
    assert abs(sem_margin["delta_abs"] - 0.002) < 1e-9
    assert sem_margin["p_uniform"] == 0.001
    assert sem_margin["L_semantic_margin_ok_norm"] == 3
    assert sem_margin["margin_ok_at_L_semantic_norm"] is True
    assert abs(sem_margin["p_answer_at_L_semantic_norm"] - 0.012) < 1e-9
