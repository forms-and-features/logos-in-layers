#!/usr/bin/env python3
"""CPU-only tests for numerics helpers."""

import math
import torch

from layers_core.numerics import bits_entropy_from_logits, safe_cast_for_unembed


def test_entropy_uniform_and_peaked():
    vocab = 10
    logits = torch.zeros(vocab)
    H_uniform = bits_entropy_from_logits(logits)
    assert abs(H_uniform - math.log2(vocab)) < 1e-5

    logits = torch.full((vocab,), -10.0)
    logits[3] = 10.0
    H_peaked = bits_entropy_from_logits(logits)
    assert H_peaked < 1e-3  # ~0 bits for near-delta


def test_safe_cast_for_unembed():
    resid = torch.randn(3, 4, dtype=torch.float16)

    # Force FP32 when W_U is FP32 and flag set
    W_U_fp32 = torch.zeros(4, 4, dtype=torch.float32)
    out = safe_cast_for_unembed(resid, W_U_fp32, force_fp32_unembed=True)
    assert out.dtype == torch.float32

    # Cast to fp16 when W_U is fp16 and no force flag
    resid32 = resid.float()
    W_U_fp16 = torch.zeros(4, 4, dtype=torch.float16)
    out = safe_cast_for_unembed(resid32, W_U_fp16)
    assert out.dtype == torch.float16

    # Leave dtype unchanged for int8 W_U
    W_U_int8 = torch.zeros(4, 4, dtype=torch.int8)
    out = safe_cast_for_unembed(resid32, W_U_int8)
    assert out.dtype == resid32.dtype

