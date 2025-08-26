#!/usr/bin/env python3
"""Unit tests for answer_rank and related simple metrics (CPU-only).

These tests verify the rank calculation logic used in run.py (§1.3):
    answer_rank = 1 + (probs > p_answer).sum()

We also sanity-check p_top1 <= p_top5.
"""

import _pathfix  # noqa: F401

import torch


def answer_rank_from_probs(probs: torch.Tensor, answer_id: int) -> int:
    """Compute rank as implemented in run.py: 1 + number strictly greater.

    probs: 1D tensor that already sums to 1.
    """
    p_answer = probs[answer_id]
    return int(1 + (probs > p_answer).sum().item())


def test_answer_rank_basic():
    probs = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float32)
    assert answer_rank_from_probs(probs, 0) == 1
    assert answer_rank_from_probs(probs, 1) == 2
    assert answer_rank_from_probs(probs, 2) == 3


def test_answer_rank_with_ties_min_rank():
    # Ties should give minimum rank (strictly-greater counting)
    probs = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float32)
    assert answer_rank_from_probs(probs, 0) == 1
    assert answer_rank_from_probs(probs, 1) == 1


def test_p_top5_ge_p_top1():
    # Random-ish distribution; ensure p_top5 >= p_top1 by construction
    probs = torch.tensor([0.25, 0.15, 0.10, 0.05, 0.03, 0.02, 0.40], dtype=torch.float32)
    # Sort descending
    vals, idx = torch.sort(probs, descending=True)
    p_top1 = vals[0].item()
    k5 = min(5, probs.shape[0])
    p_top5 = vals[:k5].sum().item()
    assert 0.0 <= p_top1 <= 1.0
    assert 0.0 <= p_top5 <= 1.0
    assert p_top5 + 1e-9 >= p_top1

