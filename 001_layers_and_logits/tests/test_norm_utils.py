#!/usr/bin/env python3
"""Unit tests for norm_utils slice extraction."""

import _pathfix  # noqa: F401

import torch

from layers_core.norm_utils import (
    apply_norm_or_skip,
    detect_model_architecture,
    get_correct_norm_module,
)


def test_epsilon_placement_norm_utils():
    batch, seq, d = 1, 7, 32
    residual = torch.randn(batch, seq, d)

    class MockRMS(torch.nn.Module):
        def __init__(self, d_model, eps=1e-5):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(d_model))
            self.eps = eps

    norm = MockRMS(d)
    out = apply_norm_or_skip(residual, norm)
    expected_rms = torch.sqrt(residual.pow(2).mean(-1, keepdim=True) + norm.eps)
    expected = residual / expected_rms * norm.weight
    assert torch.allclose(out, expected, atol=1e-6)


def test_arch_detection_and_norm_selection():
    class PreBlock:
        def __init__(self):
            self.ln1 = torch.nn.LayerNorm(16)
            self.ln2 = torch.nn.LayerNorm(16)
            self.mlp = "mlp"
        def children(self):
            return [self.ln1, self.ln2, self.mlp]

    class PreModel:
        def __init__(self):
            self.blocks = [PreBlock(), PreBlock(), PreBlock()]
            self.ln_final = torch.nn.LayerNorm(16)

    pre = PreModel()
    assert detect_model_architecture(pre) == "pre_norm"
    assert get_correct_norm_module(pre, 0, True) is pre.blocks[1].ln1
    assert get_correct_norm_module(pre, 2, True) is pre.ln_final

    class PostBlock:
        def __init__(self):
            self.attn = "attn"; self.ln1 = torch.nn.LayerNorm(16); self.mlp = "mlp"; self.ln2 = torch.nn.LayerNorm(16)
        def children(self):
            return [self.attn, self.ln1, self.mlp, self.ln2]

    class PostModel:
        def __init__(self):
            self.blocks = [PostBlock(), PostBlock(), PostBlock()]
            self.ln_final = torch.nn.LayerNorm(16)

    post = PostModel()
    assert detect_model_architecture(post) == "post_norm"
    assert get_correct_norm_module(post, 0, True) is post.blocks[0].ln2
    assert get_correct_norm_module(post, 1, False) is post.blocks[1].ln1

