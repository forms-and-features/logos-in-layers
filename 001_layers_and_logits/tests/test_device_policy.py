#!/usr/bin/env python3
"""CPU-only tests for device/dtype policy."""

import _pathfix  # noqa: F401

import torch
from layers_core.device_policy import choose_dtype, should_auto_promote_unembed


def test_choose_dtype_table_and_gemma_override():
    assert choose_dtype("cpu", "any/model") == torch.float32
    assert choose_dtype("mps", "any/model") == torch.float16
    assert choose_dtype("cuda", "mistralai/Mistral-7B-v0.1") == torch.float16
    assert choose_dtype("cuda", "google/gemma-2-9b") == torch.bfloat16
    assert choose_dtype("cuda", "GEMMA-2-27B") == torch.bfloat16


def test_should_auto_promote_unembed():
    assert should_auto_promote_unembed(torch.float32) is True
    assert should_auto_promote_unembed(torch.float16) is False
    assert should_auto_promote_unembed(torch.bfloat16) is False

