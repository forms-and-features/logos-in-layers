#!/usr/bin/env python3
"""CPU-only tests for collapse rules."""

import _pathfix  # noqa: F401

import torch

from layers_core.collapse_rules import (
    detect_copy_collapse,
    is_semantic_top1,
    is_id_subseq,
    detect_copy_collapse_id_subseq,
    is_pure_whitespace_or_punct,
)


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


def test_is_id_subseq_basic():
    ctx = [10, 20, 30, 40]
    assert is_id_subseq([10], ctx)
    assert is_id_subseq([20, 30], ctx)
    assert is_id_subseq([30, 40], ctx)
    assert not is_id_subseq([10, 30], ctx)  # non-contiguous
    assert not is_id_subseq([], ctx)


def test_detect_copy_collapse_id_subseq_threshold_and_margin():
    # Logits where index 2 is very confident top-1
    logits = torch.tensor([0.1, 0.2, 5.0, -1.0, 0.0], dtype=torch.float32)
    ctx_ids = [9, 2, 4, 7]
    # window contains [2] (k=1)
    window = [2]
    # Tight thresholds should pass here
    collapsed = detect_copy_collapse_id_subseq(
        logits,
        ctx_ids,
        window,
        copy_threshold=0.95,
        copy_margin=0.10,
    )
    assert collapsed is True
    # Too strict margin should fail
    collapsed = detect_copy_collapse_id_subseq(
        logits,
        ctx_ids,
        window,
        copy_threshold=0.95,
        copy_margin=0.90,
    )
    assert collapsed is False
    # Window not in ctx should fail
    collapsed = detect_copy_collapse_id_subseq(
        logits,
        [1, 3, 4],
        window,
        copy_threshold=0.6,
        copy_margin=0.1,
    )
    assert collapsed is False


def test_is_pure_whitespace_or_punct():
    assert is_pure_whitespace_or_punct("   ")
    assert is_pure_whitespace_or_punct("\n\t")
    assert is_pure_whitespace_or_punct("!!!")
    assert not is_pure_whitespace_or_punct(" Berlin")
    assert not is_pure_whitespace_or_punct("simply")
