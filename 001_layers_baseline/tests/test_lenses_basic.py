#!/usr/bin/env python3
"""CPU-only tests for lens adapters (baseline NormLensAdapter).

Verifies that the adapter reproduces the inline normalization+unembed path
exactly on deterministic tensors for both pre-norm and post-norm architectures.
"""

import _pathfix  # noqa: F401

import torch
import torch.nn as nn

from layers_core.lenses import NormLensAdapter
from layers_core.norm_utils import get_correct_norm_module, apply_norm_or_skip
from layers_core.numerics import safe_cast_for_unembed
from layers_core.unembed import unembed_mm


class PreNormBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model))


class PostNormBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model))
        self.ln2 = nn.LayerNorm(d_model)


class ModelStub(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        # ln_final only used when probing after last block; our tests avoid that
        self.ln_final = nn.LayerNorm(blocks[0].ln1.normalized_shape[0])


def _run_pair(model: nn.Module, layer_idx: int, probe_after_block: bool) -> None:
    torch.manual_seed(0)
    d_model = model.blocks[0].ln1.normalized_shape[0]
    seq_len, vocab = 4, 7
    resid = torch.randn(1, seq_len, d_model, dtype=torch.float32)
    W = torch.randn(d_model, vocab, dtype=torch.float32)
    b = torch.randn(vocab, dtype=torch.float32)

    # Inline baseline
    norm = get_correct_norm_module(model, layer_idx, probe_after_block=probe_after_block)
    resid_n = apply_norm_or_skip(resid, norm)
    casted = safe_cast_for_unembed(resid_n[0, :, :], W, force_fp32_unembed=False)
    logits_inline = unembed_mm(casted, W, b).float()

    # Adapter
    adapter = NormLensAdapter()
    logits_adapter = adapter.forward(
        model,
        layer_idx,
        resid,
        probe_after_block=probe_after_block,
        W_U=W,
        b_U=b,
        force_fp32_unembed=False,
        cache={},
    )

    assert logits_inline.shape == logits_adapter.shape
    assert logits_adapter.dtype == torch.float32
    assert torch.allclose(logits_inline, logits_adapter, atol=1e-6)


def test_norm_lens_adapter_prenorm_preblock():
    model = ModelStub([PreNormBlock(8), PreNormBlock(8)])
    _run_pair(model, layer_idx=0, probe_after_block=False)


def test_norm_lens_adapter_prenorm_postblock():
    model = ModelStub([PreNormBlock(8), PreNormBlock(8)])
    _run_pair(model, layer_idx=0, probe_after_block=True)


def test_norm_lens_adapter_postnorm_preblock():
    model = ModelStub([PostNormBlock(8), PostNormBlock(8)])
    _run_pair(model, layer_idx=0, probe_after_block=False)


def test_norm_lens_adapter_postnorm_postblock():
    model = ModelStub([PostNormBlock(8), PostNormBlock(8)])
    _run_pair(model, layer_idx=0, probe_after_block=True)

