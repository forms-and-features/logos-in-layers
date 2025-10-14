#!/usr/bin/env python3
"""Unit tests for decoding-point comparison helpers."""

import _pathfix  # noqa: F401

import torch

from layers_core.decoding_point import compare_decoding_strategies


def test_compare_decoding_strategies_identical_logits():
    logits = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32)
    final_probs = torch.softmax(logits, dim=0)
    result = compare_decoding_strategies(
        logits_same_ln2=logits,
        logits_next_ln1=logits,
        final_probs=final_probs,
        answer_token_id=4,
    )
    assert result["rank1_agree"] is True
    assert result["top1_token_agree"] is True
    assert result["delta_answer_rank"] == 0
    assert abs(result.get("jaccard@10", 0.0) - 1.0) < 1e-9
    assert abs(result.get("jaccard@50", 0.0) - 1.0) < 1e-9
    assert abs(result.get("spearman_top50", 1.0) - 1.0) < 1e-6


def test_compare_decoding_strategies_rank_difference():
    logits_next_ln1 = torch.tensor([0.0, 0.2, 0.4, 0.6], dtype=torch.float32)
    logits_same_ln2 = torch.tensor([0.6, 0.2, 0.1, 0.0], dtype=torch.float32)
    final_probs = torch.softmax(logits_next_ln1, dim=0)
    result = compare_decoding_strategies(
        logits_same_ln2=logits_same_ln2,
        logits_next_ln1=logits_next_ln1,
        final_probs=final_probs,
        answer_token_id=3,
    )
    assert result["rank1_agree"] is False
    assert result["top1_token_agree"] is False
    assert result["answer_rank_post_ln2"] is not None
    assert result["answer_rank_next_ln1"] == 1
    assert result["delta_answer_rank"] == result["answer_rank_post_ln2"] - 1
    j10 = result.get("jaccard@10")
    assert j10 is not None
    assert 0.0 <= j10 <= 1.0


def test_compare_decoding_strategies_missing_answer_id():
    logits = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    final_probs = torch.softmax(logits, dim=0)
    result = compare_decoding_strategies(
        logits_same_ln2=logits,
        logits_next_ln1=logits,
        final_probs=final_probs,
        answer_token_id=None,
    )
    assert result["rank1_agree"] is None
