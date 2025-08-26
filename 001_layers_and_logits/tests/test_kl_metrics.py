#!/usr/bin/env python3
"""CPU-only tests for KL divergence helpers (bits).

Checks:
- KL(P||P) ≈ 0
- Asymmetry (KL(P||Q) != KL(Q||P))
- Monotonicity with perturbation size
- Agreement with nats/log2 conversion
"""

import _pathfix  # noqa: F401

import math
import torch

from layers_core.numerics import kl_bits


def _normalize(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp_min(0)
    s = x.sum()
    return (x / s) if s > 0 else torch.ones_like(x) / x.numel()


def test_kl_identity_zero_bits():
    p = _normalize(torch.rand(100, dtype=torch.float32))
    assert kl_bits(p, p) < 1e-8


def test_kl_asymmetry_and_monotonicity():
    p = _normalize(torch.rand(64, dtype=torch.float32))
    # Small perturbation
    q1 = _normalize(p + 0.01 * torch.randn_like(p))
    # Larger perturbation
    q2 = _normalize(p + 0.05 * torch.randn_like(p))

    kl_pq1 = kl_bits(p, q1)
    kl_qp1 = kl_bits(q1, p)
    kl_pq2 = kl_bits(p, q2)

    assert kl_pq1 > 0.0
    assert kl_qp1 > 0.0
    # Asymmetry: not equal within tight tolerance
    assert abs(kl_pq1 - kl_qp1) > 1e-6
    # Monotonicity wrt larger perturbation (probabilistic, but overwhelmingly likely)
    assert kl_pq2 > kl_pq1


def test_kl_bits_matches_nats_div_log2():
    p = _normalize(torch.rand(32, dtype=torch.float32))
    q = _normalize(torch.rand(32, dtype=torch.float32))
    # Compute nats explicitly
    eps = 1e-30
    p32 = p.to(torch.float32)
    q32 = q.to(torch.float32)
    nats = torch.sum(p32 * ((p32 + eps).log() - (q32 + eps).log())).item()
    bits_ref = nats / math.log(2)
    bits = kl_bits(p, q)
    assert abs(bits - bits_ref) < 1e-8

