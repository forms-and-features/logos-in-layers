#!/usr/bin/env python3
"""CPU-only tests for collapse rules."""

import _pathfix  # noqa: F401

import torch

from layers_core.collapse_rules import detect_copy_collapse, is_semantic_top1


def test_detect_copy_collapse_threshold_and_margin():
    logits = torch.tensor([0.1, 0.2, 5.0, -1.0, 0.0], dtype=torch.float32)
    prompt_ids = {2, 4}
    collapsed = detect_copy_collapse(
        logits,
        prompt_ids,
        copy_threshold=0.6,
        copy_margin=0.2,
        entropy_bits=None,
    )
    assert collapsed is True

    collapsed = detect_copy_collapse(
        logits,
        {4},
        copy_threshold=0.6,
        copy_margin=0.2,
        entropy_bits=None,
    )
    assert collapsed is False


def test_entropy_fallback():
    logits = torch.tensor([3.0, 2.9, -5.0, -5.0], dtype=torch.float32)
    prompt_ids = {0, 1}
    collapsed = detect_copy_collapse(
        logits,
        prompt_ids,
        copy_threshold=0.99,
        copy_margin=0.5,
        entropy_bits=0.5,
        entropy_fallback_threshold=1.0,
    )
    assert collapsed is True


def test_is_semantic_top1():
    assert is_semantic_top1("Berlin", "Berlin")
    assert is_semantic_top1(" Berlin\n", "Berlin")
    assert not is_semantic_top1("Ber", "Berlin")

