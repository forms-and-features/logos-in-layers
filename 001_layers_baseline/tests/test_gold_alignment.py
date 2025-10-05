#!/usr/bin/env python3
"""Unit tests for gold-alignment helper utilities."""

import _pathfix  # noqa: F401

from layers_core.gold import build_gold_alignment_entry, compute_gold_alignment_rate


def test_build_gold_alignment_entry_copies_lists():
    gold_info = {
        "status": "ok",
        "variant": "with_space",
        "first_id": 123,
        "answer_ids": [123, 456],
        "pieces": [" Ber", "lin"],
        "string": "Berlin",
    }
    entry = build_gold_alignment_entry("pos", "orig", gold_info)
    assert entry["prompt_id"] == "pos"
    assert entry["prompt_variant"] == "orig"
    assert entry["ok"] is True
    assert entry["status"] == "ok"
    assert entry["variant"] == "with_space"
    assert entry["first_id"] == 123
    assert entry["answer_ids"] == [123, 456]
    assert entry["pieces"] == [" Ber", "lin"]

    # mutating source lists should not affect the entry copies
    gold_info["answer_ids"].append(789)
    gold_info["pieces"].append("!")
    assert entry["answer_ids"] == [123, 456]
    assert entry["pieces"] == [" Ber", "lin"]


def test_compute_gold_alignment_rate():
    entries = [
        {"ok": True},
        {"ok": False},
        {"ok": True},
    ]
    rate = compute_gold_alignment_rate(entries)
    assert rate == 2 / 3

    empty_rate = compute_gold_alignment_rate([])
    assert empty_rate is None
