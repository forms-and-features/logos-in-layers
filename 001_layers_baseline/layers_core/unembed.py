from __future__ import annotations

from typing import Optional, Dict, Any, Tuple

import torch


def prepare_unembed_weights(
    W_U: torch.Tensor,
    b_U: Optional[torch.Tensor],
    *,
    force_fp32: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return analysis-only unembedding weights (and bias) with optional fp32 promotion.

    - Never mutates the input tensors; makes float32 copies when promotion is requested.
    - If `force_fp32` is False or weights are already float32, returns inputs as-is.
    """
    if force_fp32 and W_U.dtype != torch.float32:
        W_prepared = W_U.float()
        b_prepared = b_U.float() if (b_U is not None and b_U.dtype != torch.float32) else b_U
        return W_prepared, b_prepared
    return W_U, b_U


def unembed_mm(
    X: torch.Tensor,
    W: torch.Tensor,
    b: Optional[torch.Tensor],
    *,
    cache: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Matrix multiply helper for unembedding with lightweight per-device cache.

    - If `W`/`b` are on a different device than `X`, moves once and caches copies in `cache`.
    - `cache` is a dict with keys: 'device', 'W', 'b'. If None, no caching is performed.
    - Returns logits of shape compatible with `X @ W (+ b)`.
    """
    dev = X.device

    W_use = W
    b_use = b

    if hasattr(W, "device") and W.device != dev:
        if cache is not None:
            if cache.get("device") != dev:
                cache["W"] = W.to(dev)
                cache["b"] = (b.to(dev) if (b is not None and hasattr(b, "device")) else b)
                cache["device"] = dev
            W_use = cache.get("W", W.to(dev))
            b_cached = cache.get("b", None)
            b_use = b_cached if b_cached is not None else (b.to(dev) if (b is not None and hasattr(b, "device")) else b)
        else:
            W_use = W.to(dev)
            b_use = (b.to(dev) if (b is not None and hasattr(b, "device")) else b)

    out = X @ W_use
    if b_use is not None:
        out = out + b_use
    return out

