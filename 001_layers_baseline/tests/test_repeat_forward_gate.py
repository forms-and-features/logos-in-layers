#!/usr/bin/env python3
"""Unit tests for forward-of-two repeatability summary helpers."""

import _pathfix  # noqa: F401

from layers_core.repeat_forward import (
    build_repeatability_forward_summary,
    resolve_preferred_semantic_milestone,
)


def _record(layer: int, tokens, *, prompt_id="pos", prompt_variant="orig", fact_index=0):
    return {
        "prompt_id": prompt_id,
        "prompt_variant": prompt_variant,
        "layer": layer,
        "fact_index": fact_index,
        "topk": [[tok, 0.1] for tok in tokens],
    }


def test_resolve_preferred_semantic_milestone_priority():
    diag = {
        "semantic_gate": {
            "L_semantic_strong": 8,
            "L_semantic_strong_run2": 9,
        },
        "L_semantic_confirmed": 7,
        "L_semantic": 6,
    }
    name, layer = resolve_preferred_semantic_milestone(diag)
    assert name == "L_semantic_strong_run2"
    assert layer == 9


def test_repeatability_forward_pass_within_tolerance_and_jaccard():
    pass1_diag = {"semantic_gate": {"L_semantic_strong_run2": 12}}
    pass2_diag = {"semantic_gate": {"L_semantic_strong_run2": 13}}
    recs1 = [_record(12, ["Berlin", "Paris", "Rome"])]
    recs2 = [_record(12, ["Berlin", "Paris", "Rome"])]
    summary = build_repeatability_forward_summary(
        pass1_diag=pass1_diag,
        pass2_diag=pass2_diag,
        pass1_records=recs1,
        pass2_records=recs2,
        prompt_id="pos",
        prompt_variant="orig",
        fact_index=0,
        tolerance_layers=1,
        topk_k=10,
    )
    assert summary["milestones"]["primary"] == "L_semantic_strong_run2"
    assert summary["milestones"]["delta_layers"] == 1
    assert summary["topk_jaccard_at_primary_layer"] == 1.0
    assert summary["gate"]["repeatability_forward_pass"] is True


def test_repeatability_forward_fails_on_low_jaccard():
    pass1_diag = {"semantic_gate": {"L_semantic_strong": 10}}
    pass2_diag = {"semantic_gate": {"L_semantic_strong": 10}}
    recs1 = [_record(10, ["Berlin", "Paris", "Rome"])]
    recs2 = [_record(10, ["Paris", "London", "Ottawa"])]
    summary = build_repeatability_forward_summary(
        pass1_diag=pass1_diag,
        pass2_diag=pass2_diag,
        pass1_records=recs1,
        pass2_records=recs2,
        prompt_id="pos",
        prompt_variant="orig",
        fact_index=0,
        tolerance_layers=1,
        topk_k=10,
    )
    assert summary["topk_jaccard_at_primary_layer"] is not None
    assert summary["topk_jaccard_at_primary_layer"] < 0.5
    assert summary["gate"]["repeatability_forward_pass"] is False


def test_repeatability_forward_none_when_second_pass_missing():
    pass1_diag = {"L_semantic": 5}
    summary = build_repeatability_forward_summary(
        pass1_diag=pass1_diag,
        pass2_diag=None,
        pass1_records=[_record(5, ["Berlin"])],
        pass2_records=None,
        prompt_id="pos",
        prompt_variant="orig",
        fact_index=0,
        tolerance_layers=1,
        topk_k=10,
    )
    assert summary["gate"]["repeatability_forward_pass"] is None


def test_repeatability_forward_fails_on_mismatched_milestone():
    pass1_diag = {"semantic_gate": {"L_semantic_strong_run2": 14}}
    pass2_diag = {"semantic_gate": {"L_semantic_strong": 14}}
    recs = [_record(14, ["Berlin", "Paris"])]
    summary = build_repeatability_forward_summary(
        pass1_diag=pass1_diag,
        pass2_diag=pass2_diag,
        pass1_records=recs,
        pass2_records=recs,
        prompt_id="pos",
        prompt_variant="orig",
        fact_index=0,
        tolerance_layers=1,
        topk_k=10,
    )
    assert summary["gate"]["repeatability_forward_pass"] is False
