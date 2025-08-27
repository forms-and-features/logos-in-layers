#!/usr/bin/env python3
"""Unit tests for summaries helpers (CPU-only)."""

import _pathfix  # noqa: F401

from layers_core.summaries import summarize_pure_records


def test_summarize_pure_records():
    recs = [
        {"layer": 0, "copy_collapse": False, "entropy_collapse": False, "is_answer": False, "kl_to_final_bits": 2.0, "answer_rank": 100},
        {"layer": 1, "copy_collapse": True,  "entropy_collapse": False, "is_answer": False, "kl_to_final_bits": 1.2, "answer_rank": 50},
        {"layer": 2, "copy_collapse": False, "entropy_collapse": True,  "is_answer": False, "kl_to_final_bits": 0.8, "answer_rank": 10},
        {"layer": 3, "copy_collapse": False, "entropy_collapse": False, "is_answer": True,  "kl_to_final_bits": 0.3, "answer_rank": 1},
    ]

    diag = summarize_pure_records(
        recs,
        copy_threshold=0.95,
        copy_window_k=1,
        copy_match_level="id_subsequence",
    )

    assert diag["L_copy"] == 1
    assert diag["L_copy_H"] == 2
    assert diag["L_semantic"] == 3
    assert diag["delta_layers"] == 2
    assert diag["first_kl_below_0.5"] == 3
    assert diag["first_kl_below_1.0"] == 2
    assert diag["first_rank_le_1"] == 3
    assert diag["first_rank_le_5"] is None  # no rank<=5 until layer 3 (which is <=1 already accounted)
    assert diag["first_rank_le_10"] == 2

