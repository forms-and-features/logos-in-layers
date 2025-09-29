#!/usr/bin/env python3
"""Unit tests for unified sidecar summaries (PROJECT_NOTES §1.21).

Covers:
- Rank milestone extraction and delta signs
- KL percentile sampling at 25/50/75 and deltas
- Null propagation when alt sidecar is empty/missing
- Filtering by prompt_id and prompt_variant
"""

import _pathfix  # noqa: F401

from typing import List, Dict, Any

from layers_core.summaries import build_unified_lens_metrics


def _rec(layer: int, *, prompt_id="pos", prompt_variant="orig", answer_rank=None, kl=None) -> Dict[str, Any]:
    return {
        "prompt_id": prompt_id,
        "prompt_variant": prompt_variant,
        "layer": layer,
        "answer_rank": answer_rank,
        "kl_to_final_bits": kl,
    }


def test_unified_metrics_rank_and_kl_deltas():
    # Baseline: first rank<=10 at L6, <=5 at L7, <=1 at L8
    base: List[Dict[str, Any]] = []
    for L in range(1, 9):
        ar = 100
        if L >= 6:
            ar = 10
        if L >= 7:
            ar = 5
        if L >= 8:
            ar = 1
        base.append(_rec(L, answer_rank=ar))

    # KL for percentiles with n_layers=8 → layers {2,4,6}
    for i, L in enumerate((2, 4, 6)):
        base[L - 1]["kl_to_final_bits"] = [1.2, 0.8, 0.4][i]

    # Alt (e.g., prism/tuned): reaches ranks one layer earlier across the board
    alt: List[Dict[str, Any]] = []
    for L in range(1, 9):
        ar = 100
        if L >= 5:
            ar = 10
        if L >= 6:
            ar = 5
        if L >= 7:
            ar = 1
        alt.append(_rec(L, answer_rank=ar))

    # Alt KL: slightly lower at p25 and p75, slightly higher at p50
    for i, L in enumerate((2, 4, 6)):
        alt[L - 1]["kl_to_final_bits"] = [1.0, 0.9, 0.3][i]

    out = build_unified_lens_metrics(
        baseline_records=base,
        alt_records=alt,
        n_layers=8,
        alt_label="prism",
        prompt_id="pos",
        prompt_variant="orig",
    )

    # Rank milestones deltas (alt - baseline): negative = earlier (improvement)
    rm = out["rank_milestones"]
    assert rm["baseline"] == {"le_10": 6, "le_5": 7, "le_1": 8}
    assert rm["prism"] == {"le_10": 5, "le_5": 6, "le_1": 7}
    assert rm["delta"] == {"le_10": -1, "le_5": -1, "le_1": -1}

    # KL deltas (baseline - alt): positive = lower KL in alt (improvement)
    kl = out["kl_bits_at_percentiles"]
    assert kl["baseline"] == {"p25": 1.2, "p50": 0.8, "p75": 0.4}
    assert kl["prism"] == {"p25": 1.0, "p50": 0.9, "p75": 0.3}
    assert kl["delta"] == {"p25": 0.2, "p50": -0.09999999999999998, "p75": 0.10000000000000003}

    # First KL <= 1.0: baseline at L4 (0.8), alt at L2 (1.0) → delta = 2 - 4 = -2
    fk = out["first_kl_le_1.0"]
    assert fk == {"baseline": 4, "prism": 2, "delta": -2}


def test_filtering_and_null_propagation():
    # Baseline has valid rows; alt sidecar has wrong prompt_id and variant → should be filtered out
    base = [_rec(2, answer_rank=50, kl=2.0), _rec(4, answer_rank=1, kl=0.5)]
    alt = [
        _rec(2, prompt_id="ctl", answer_rank=10, kl=1.0),
        _rec(4, prompt_variant="no_filler", answer_rank=1, kl=0.4),
    ]
    out = build_unified_lens_metrics(
        baseline_records=base,
        alt_records=alt,
        n_layers=4,
        alt_label="tuned",
    )

    # Alt filtered → milestones/percentiles become None; deltas propagate None
    rm = out["rank_milestones"]
    assert rm["baseline"]["le_1"] == 4
    assert rm["tuned"] == {"le_10": None, "le_5": None, "le_1": None}
    assert rm["delta"]["le_1"] is None

    kl = out["kl_bits_at_percentiles"]
    # n_layers=4 → percentiles target layers ~{1,2,3}; baseline missing p25/p75 → None
    assert kl["baseline"]["p50"] == 0.5  # layer 2 → p50
    assert kl["baseline"]["p25"] is None and kl["baseline"]["p75"] is None
    assert kl["tuned"] == {"p25": None, "p50": None, "p75": None}
    assert kl["delta"]["p50"] is None

    fk = out["first_kl_le_1.0"]
    assert fk["baseline"] == 4
    assert fk["tuned"] is None and fk["delta"] is None


def test_empty_and_none_alt_records_equivalence():
    base = [_rec(2, answer_rank=10, kl=1.1), _rec(3, answer_rank=1, kl=0.9)]
    out_none = build_unified_lens_metrics(
        baseline_records=base,
        alt_records=None,
        n_layers=3,
        alt_label="prism",
    )
    out_empty = build_unified_lens_metrics(
        baseline_records=base,
        alt_records=[],
        n_layers=3,
        alt_label="prism",
    )
    assert out_none == out_empty

