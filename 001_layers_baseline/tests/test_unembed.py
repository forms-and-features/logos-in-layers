import _pathfix  # noqa: F401
import torch

from layers_core.unembed import prepare_unembed_weights, unembed_mm


def test_prepare_unembed_weights_fp32_promotion():
    W = torch.randn(4, 6, dtype=torch.float16)
    b = torch.randn(6, dtype=torch.float16)

    # No promotion
    W0, b0 = prepare_unembed_weights(W, b, force_fp32=False)
    assert W0.dtype == torch.float16
    assert b0 is not None and b0.dtype == torch.float16
    # Inputs are not mutated
    assert W.dtype == torch.float16 and b.dtype == torch.float16

    # Promotion to float32
    W1, b1 = prepare_unembed_weights(W, b, force_fp32=True)
    assert W1.dtype == torch.float32
    assert b1 is not None and b1.dtype == torch.float32
    # Originals unchanged
    assert W.dtype == torch.float16 and b.dtype == torch.float16


def test_unembed_mm_matches_naive():
    X = torch.randn(3, 4, dtype=torch.float32)
    W = torch.randn(4, 6, dtype=torch.float32)
    b = torch.randn(6, dtype=torch.float32)

    out1 = unembed_mm(X, W, b)
    out2 = X @ W + b
    assert torch.allclose(out1, out2, atol=1e-6)

    # Works with no bias
    out3 = unembed_mm(X, W, None)
    out4 = X @ W
    assert torch.allclose(out3, out4, atol=1e-6)


def test_unembed_mm_cache_is_optional_and_stable():
    X = torch.randn(2, 3, dtype=torch.float32)
    W = torch.randn(3, 5, dtype=torch.float32)
    b = torch.randn(5, dtype=torch.float32)
    cache = {}
    out_a = unembed_mm(X, W, b, cache=cache)
    out_b = unembed_mm(X, W, b, cache=cache)
    # At least numerically identical across repeated calls
    assert torch.allclose(out_a, out_b, atol=1e-6)
