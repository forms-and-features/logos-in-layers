#!/usr/bin/env python3
"""CPU-only tests for lens adapters (baseline NormLensAdapter).

Verifies that the adapter reproduces the inline normalization+unembed path
exactly on deterministic tensors for both pre-norm and post-norm architectures.
"""

import _pathfix  # noqa: F401

import torch
import torch.nn as nn

from layers_core.lenses import NormLensAdapter, PrismLensAdapter, TunedLensAdapter
from layers_core.norm_utils import get_correct_norm_module, apply_norm_or_skip
from layers_core.numerics import safe_cast_for_unembed
from layers_core.unembed import unembed_mm
from layers_core.prism import WhitenStats, whiten_apply


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

    assert logits_inline.shape == logits_adapter.shape, f"shape mismatch {logits_inline.shape} vs {logits_adapter.shape}"
    assert logits_adapter.dtype == torch.float32, f"dtype mismatch {logits_adapter.dtype}"
    close = torch.allclose(logits_inline, logits_adapter, atol=1e-6)
    if not close:
        diff = (logits_inline - logits_adapter).abs()
        print(f"[DEBUG lenses] max_abs_diff={float(diff.max())}")
    assert close


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


def _run_prism_pair(model: nn.Module, layer_idx: int, probe_after_block: bool) -> None:
    torch.manual_seed(0)
    d_model = model.blocks[0].ln1.normalized_shape[0]
    seq_len, vocab = 4, 7
    resid = torch.randn(1, seq_len, d_model, dtype=torch.float32)
    W = torch.randn(d_model, vocab, dtype=torch.float32)
    b = torch.randn(vocab, dtype=torch.float32)

    # Simple whitening stats (mean zero, var ones) and identity Q
    stats = WhitenStats(mean=torch.zeros(d_model), var=torch.ones(d_model), eps=1e-8)
    Q = torch.eye(d_model, dtype=torch.float32)

    # Inline baseline (normalize → whiten → Q → unembed)
    norm = get_correct_norm_module(model, layer_idx, probe_after_block=probe_after_block)
    resid_n = apply_norm_or_skip(resid, norm)
    Xw = whiten_apply(resid_n[0], stats)
    Xp = Xw @ Q
    logits_inline = unembed_mm(Xp, W, b).float()

    # Adapter
    adapter = PrismLensAdapter(stats, Q, active=True)
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

    assert logits_adapter is not None, "adapter returned None"
    assert logits_inline.shape == logits_adapter.shape, f"shape mismatch {logits_inline.shape} vs {logits_adapter.shape}"
    assert logits_adapter.dtype == torch.float32, f"dtype mismatch {logits_adapter.dtype}"
    close = torch.allclose(logits_inline, logits_adapter, atol=1e-6)
    if not close:
        diff = (logits_inline - logits_adapter).abs()
        print(f"[DEBUG lenses-prism] max_abs_diff={float(diff.max())}")
    assert close


def test_prism_lens_adapter_prenorm_preblock():
    model = ModelStub([PreNormBlock(8), PreNormBlock(8)])
    _run_prism_pair(model, layer_idx=0, probe_after_block=False)


def test_prism_lens_adapter_prenorm_postblock():
    model = ModelStub([PreNormBlock(8), PreNormBlock(8)])
    _run_prism_pair(model, layer_idx=0, probe_after_block=True)


def test_prism_lens_adapter_postnorm_preblock():
    model = ModelStub([PostNormBlock(8), PostNormBlock(8)])
    _run_prism_pair(model, layer_idx=0, probe_after_block=False)


def test_prism_lens_adapter_postnorm_postblock():
    model = ModelStub([PostNormBlock(8), PostNormBlock(8)])
    _run_prism_pair(model, layer_idx=0, probe_after_block=True)


def _run_tuned_pair(model: nn.Module, layer_idx: int, probe_after_block: bool) -> None:
    torch.manual_seed(0)
    d_model = model.blocks[0].ln1.normalized_shape[0]
    seq_len, vocab = 4, 7
    resid = torch.randn(1, seq_len, d_model, dtype=torch.float32)

    # Synthetic tuned head per layer
    Wt = torch.randn(d_model, vocab, dtype=torch.float32)
    bt = torch.randn(vocab, dtype=torch.float32)

    # Inline baseline: normalize → tuned head
    norm = get_correct_norm_module(model, layer_idx, probe_after_block=probe_after_block)
    resid_n = apply_norm_or_skip(resid, norm)
    X = resid_n[0, :, :]
    Xc = safe_cast_for_unembed(X, Wt, force_fp32_unembed=False)
    logits_inline = unembed_mm(Xc, Wt, bt).float()

    # Adapter with a head table for the correct probe timing
    if probe_after_block:
        adapter = TunedLensAdapter(weights_post={layer_idx: (Wt, bt)}, weights_pre={})
    else:
        adapter = TunedLensAdapter(weights_pre={layer_idx: (Wt, bt)}, weights_post={})

    logits_adapter = adapter.forward(
        model,
        layer_idx,
        resid,
        probe_after_block=probe_after_block,
        W_U=Wt,   # unused
        b_U=bt,   # unused
        force_fp32_unembed=False,
        cache={},
    )

    assert logits_inline.shape == logits_adapter.shape
    assert logits_adapter.dtype == torch.float32
    close = torch.allclose(logits_inline, logits_adapter, atol=1e-6)
    if not close:
        diff = (logits_inline - logits_adapter).abs().max().item()
        print(f"[DEBUG tuned] max_abs_diff={diff}")
    assert close


def test_tuned_lens_adapter_prenorm_preblock():
    model = ModelStub([PreNormBlock(8), PreNormBlock(8)])
    _run_tuned_pair(model, layer_idx=0, probe_after_block=False)


def test_tuned_lens_adapter_prenorm_postblock():
    model = ModelStub([PreNormBlock(8), PreNormBlock(8)])
    _run_tuned_pair(model, layer_idx=0, probe_after_block=True)


def test_tuned_lens_adapter_postnorm_preblock():
    model = ModelStub([PostNormBlock(8), PostNormBlock(8)])
    _run_tuned_pair(model, layer_idx=0, probe_after_block=False)


def test_tuned_lens_adapter_postnorm_postblock():
    model = ModelStub([PostNormBlock(8), PostNormBlock(8)])
    _run_tuned_pair(model, layer_idx=0, probe_after_block=True)


def test_tuned_lens_missing_head_non_strict_returns_none():
    # No heads provided; strict=False should not raise and should return None
    model = ModelStub([PreNormBlock(8), PreNormBlock(8)])
    adapter = TunedLensAdapter(weights_pre={}, weights_post={}, strict=False)
    resid = torch.randn(1, 4, 8, dtype=torch.float32)
    out = adapter.forward(
        model,
        0,
        resid,
        probe_after_block=False,
        W_U=torch.empty(8, 7),
        b_U=None,
        force_fp32_unembed=False,
        cache={},
    )
    assert out is None
    assert 0 in adapter.diag.get("missing_layers", [])


def test_tuned_lens_integer_weight_raises():
    model = ModelStub([PreNormBlock(8), PreNormBlock(8)])
    # int8 weight should be rejected
    W = torch.randint(-5, 5, (8, 7), dtype=torch.int8)
    b = torch.randint(-5, 5, (7,), dtype=torch.int8)
    adapter = TunedLensAdapter(weights_pre={0: (W, b)}, weights_post={})
    resid = torch.randn(1, 4, 8, dtype=torch.float32)
    try:
        _ = adapter.forward(
            model,
            0,
            resid,
            probe_after_block=False,
            W_U=torch.empty(8, 7),
            b_U=None,
            force_fp32_unembed=False,
            cache={},
        )
    except ValueError as e:
        assert "floating-point" in str(e)
    else:
        raise AssertionError("Expected ValueError for non-floating tuned weights")


def test_tuned_lens_shape_mismatch_raises():
    model = ModelStub([PreNormBlock(8), PreNormBlock(8)])
    # Wrong d_model in W (rows=9)
    W = torch.randn(9, 7, dtype=torch.float32)
    b = torch.randn(7, dtype=torch.float32)
    adapter = TunedLensAdapter(weights_pre={0: (W, b)}, weights_post={})
    resid = torch.randn(1, 4, 8, dtype=torch.float32)
    try:
        _ = adapter.forward(
            model,
            0,
            resid,
            probe_after_block=False,
            W_U=torch.empty(8, 7),
            b_U=None,
            force_fp32_unembed=False,
            cache={},
        )
    except ValueError as e:
        assert "W rows" in str(e)
    else:
        raise AssertionError("Expected ValueError for mismatched W rows vs d_model")


if __name__ == "__main__":
    import traceback
    print("Running lenses adapter tests…")
    ok = True
    try:
        test_norm_lens_adapter_prenorm_preblock(); print("✅ norm pre-norm pre-block")
        test_norm_lens_adapter_prenorm_postblock(); print("✅ norm pre-norm post-block")
        test_norm_lens_adapter_postnorm_preblock(); print("✅ norm post-norm pre-block")
        test_norm_lens_adapter_postnorm_postblock(); print("✅ norm post-norm post-block")
        test_prism_lens_adapter_prenorm_preblock(); print("✅ prism pre-norm pre-block")
        test_prism_lens_adapter_prenorm_postblock(); print("✅ prism pre-norm post-block")
        test_prism_lens_adapter_postnorm_preblock(); print("✅ prism post-norm pre-block")
        test_prism_lens_adapter_postnorm_postblock(); print("✅ prism post-norm post-block")
    except AssertionError as e:
        print("❌ assertion failed:", e); traceback.print_exc(); ok = False
    except Exception as e:
        print("❌ test crashed:", e); traceback.print_exc(); ok = False
    raise SystemExit(0 if ok else 1)
