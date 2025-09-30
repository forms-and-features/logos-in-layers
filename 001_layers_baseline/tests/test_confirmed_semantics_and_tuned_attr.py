#!/usr/bin/env python3
"""CPU-only tests for confirmed semantics and tuned attribution/gate.

We fabricate minimal per-layer records to exercise the computations without
requiring model runs.
"""

import _pathfix  # noqa: F401

from layers_core.summaries import compute_confirmed_semantics, tuned_rotation_vs_temp_attribution


def test_confirmed_semantics_raw_only():
    # Baseline: L_semantic_norm = 2
    baseline = [
        {"layer": 0, "answer_rank": 99},
        {"layer": 1, "answer_rank": 5},
        {"layer": 2, "answer_rank": 1},
    ]
    raw_full = [
        {"layer": 0, "answer_rank_raw": 99, "prompt_id": "pos", "prompt_variant": "orig"},
        {"layer": 1, "answer_rank_raw": 99, "prompt_id": "pos", "prompt_variant": "orig"},
        {"layer": 2, "answer_rank_raw": 1, "prompt_id": "pos", "prompt_variant": "orig"},
    ]
    tuned = []
    conf = compute_confirmed_semantics(
        baseline_records=baseline,
        raw_full_rows=raw_full,
        tuned_records=tuned,
        L_semantic_norm=2,
        delta_window=2,
    )
    assert conf["L_semantic_norm"] == 2
    assert conf["L_semantic_confirmed"] == 2
    assert conf["confirmed_source"] in ("raw", "both")


def test_tuned_attribution_prefer_gate():
    # Baseline and tuned KLs at percentiles; baseline norm_temp also given
    baseline = [
        {"layer": 1, "kl_to_final_bits": 1.0, "kl_to_final_bits_norm_temp": 0.8, "answer_rank": 10},
        {"layer": 2, "kl_to_final_bits": 1.1, "kl_to_final_bits_norm_temp": 0.9, "answer_rank": 10},
        {"layer": 3, "kl_to_final_bits": 1.2, "kl_to_final_bits_norm_temp": 1.0, "answer_rank": 10},
    ]
    tuned = [
        {"layer": 1, "kl_to_final_bits": 0.6, "answer_rank": 10},
        {"layer": 2, "kl_to_final_bits": 0.7, "answer_rank": 4},  # first <=5 earlier than baseline
        {"layer": 3, "kl_to_final_bits": 0.8, "answer_rank": 4},
    ]
    attr = tuned_rotation_vs_temp_attribution(
        baseline_records=baseline,
        tuned_records=tuned,
        n_layers=3,
    )
    assert "percentiles" in attr and isinstance(attr["percentiles"], dict)
    assert isinstance(attr.get("prefer_tuned", False), bool)


def test_tuned_attribution_missing_fields():
    # Missing kl_to_final_bits_norm_temp and some layers absent; should not crash
    baseline = [
        {"layer": 0, "kl_to_final_bits": None, "answer_rank": None},
        {"layer": 1, "kl_to_final_bits": 0.9},
    ]
    tuned = [
        {"layer": 1, "kl_to_final_bits": None},
    ]
    out = tuned_rotation_vs_temp_attribution(
        baseline_records=baseline,
        tuned_records=tuned,
        n_layers=2,
    )
    assert "percentiles" in out
    assert set(out["percentiles"].keys()) == {"p25", "p50", "p75"}
