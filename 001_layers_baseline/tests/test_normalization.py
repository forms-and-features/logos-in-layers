#!/usr/bin/env python3
"""Unit test for normalization scaling fixes (CPU-only)"""

import _pathfix  # noqa: F401

import torch
from run import apply_norm_or_skip, get_correct_norm_module, detect_model_architecture


def test_epsilon_placement():
    batch_size, seq_len, d_model = 1, 10, 64
    residual = torch.randn(batch_size, seq_len, d_model)

    class MockRMSNorm(torch.nn.Module):
        def __init__(self, d_model, eps=1e-5):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(d_model))
            self.eps = eps

    norm_module = MockRMSNorm(d_model)
    normalized = apply_norm_or_skip(residual, norm_module)
    expected_rms = torch.sqrt(residual.pow(2).mean(-1, keepdim=True) + norm_module.eps)
    expected_normalized = residual / expected_rms * norm_module.weight
    diff = torch.abs(normalized - expected_normalized).max().item()
    assert diff < 1e-6


def test_architecture_aware_norm_selection():
    class MockRMSNorm(torch.nn.Module):
        def __init__(self, d_model, eps=1e-5):
            super().__init__(); self.weight = torch.nn.Parameter(torch.ones(d_model)); self.eps = eps

    class MockBlock:
        def __init__(self):
            self.ln1 = MockRMSNorm(64)
            self.attn = "attn"; self.ln2 = MockRMSNorm(64); self.mlp = "mlp"
            self.hook_resid_pre = "mock_hook"
        def children(self):
            return [self.ln1, self.attn, self.ln2, self.mlp]

    class PostNormBlock:
        def __init__(self):
            self.attn = "attn"; self.ln1 = torch.nn.LayerNorm(64); self.mlp = "mlp"; self.ln2 = torch.nn.LayerNorm(64)
            self.hook_resid_post = "mock_hook"
        def children(self):
            return [self.attn, self.ln1, self.mlp, self.ln2, self.hook_resid_post]

    class PreNormModel:
        class Cfg: n_layers=3; model_name="llama-test"
        def __init__(self):
            self.cfg=self.Cfg(); self.blocks=[MockBlock() for _ in range(3)]; self.ln_final=MockRMSNorm(64)

    class PostNormModel:
        class Cfg: n_layers=3; model_name="gpt-j-test"
        def __init__(self):
            self.cfg=self.Cfg(); self.blocks=[PostNormBlock() for _ in range(3)]; self.ln_final=torch.nn.LayerNorm(64)

    pre_model = PreNormModel()
    assert detect_model_architecture(pre_model) == "pre_norm"
    assert get_correct_norm_module(pre_model, 0, True) is pre_model.blocks[1].ln1
    assert get_correct_norm_module(pre_model, 2, True) is pre_model.ln_final
    assert get_correct_norm_module(pre_model, 0, False) is pre_model.blocks[0].ln1

    post_model = PostNormModel()
    assert detect_model_architecture(post_model) == "post_norm"
    assert get_correct_norm_module(post_model, 0, True) is post_model.blocks[0].ln2
    assert get_correct_norm_module(post_model, 1, False) is post_model.blocks[1].ln1

