#!/usr/bin/env python3
"""CPU-only tests for numerics helpers."""

import _pathfix  # noqa: F401

import math
import torch

from layers_core.numerics import bits_entropy_from_logits, safe_cast_for_unembed, kl_bits


def test_entropy_uniform_and_peaked():
    vocab = 10
    logits = torch.zeros(vocab)
    H_uniform = bits_entropy_from_logits(logits)
    assert abs(H_uniform - math.log2(vocab)) < 1e-5

    logits = torch.full((vocab,), -10.0)
    logits[3] = 10.0
    H_peaked = bits_entropy_from_logits(logits)
    assert H_peaked < 1e-3


def test_safe_cast_for_unembed():
    resid = torch.randn(3, 4, dtype=torch.float16)

    W_U_fp32 = torch.zeros(4, 4, dtype=torch.float32)
    out = safe_cast_for_unembed(resid, W_U_fp32, force_fp32_unembed=True)
    assert out.dtype == torch.float32

    resid32 = resid.float()
    W_U_fp16 = torch.zeros(4, 4, dtype=torch.float16)
    out = safe_cast_for_unembed(resid32, W_U_fp16)
    assert out.dtype == torch.float16

    W_U_int8 = torch.zeros(4, 4, dtype=torch.int8)
    out = safe_cast_for_unembed(resid32, W_U_int8)
    assert out.dtype == resid32.dtype


def test_entropy_bits_matches_manual_reference():
    torch.manual_seed(123)
    # A few vocab sizes to cover small/medium dimensions
    for vocab in (7, 257, 4096):
        logits = torch.randn(vocab, dtype=torch.float32) * 3.0
        H_bits = bits_entropy_from_logits(logits)
        # Manual reference in torch float32 (nats → bits)
        logp = torch.log_softmax(logits.to(torch.float32), dim=-1)
        ent_nats = -torch.sum(torch.exp(logp) * logp).item()
        H_manual = ent_nats / math.log(2)
        assert abs(H_bits - H_manual) < 1e-6


def test_kl_bits_basic_properties():
    torch.manual_seed(0)
    p = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float32)
    q = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32)
    kl_pq = kl_bits(p, q)
    manual = torch.sum(p * (torch.log(p + 1e-30) - torch.log(q + 1e-30))).item() / math.log(2)
    assert abs(kl_pq - manual) < 1e-6
    assert abs(kl_bits(p, p)) < 1e-9
    assert kl_bits(q, p) != kl_pq
