from __future__ import annotations

from typing import Dict, Optional, Tuple, Any

import torch

from ..norm_utils import get_correct_norm_module, apply_norm_or_skip
from ..numerics import safe_cast_for_unembed
from ..unembed import unembed_mm
from .base import LensAdapter


class TunedLensAdapter(LensAdapter):
    """Tuned lens adapter: normalization → per-layer tuned head → logits.

    Mirrors the baseline normalization path but decodes with a provided per-layer
    tuned projection `(W, b)` instead of the model's `(W_U, b_U)`.

    - `weights_pre`: map layer→(W,b) for pre-block probes; `weights_post` for post-block.
    - Returns logits as float32. The matmul runs in the dtype of `W` unless callers
      provide fp32 weights (recommended). Residuals are cast via `safe_cast_for_unembed`.

    Operational notes
    - Use a distinct `cache` per lens to avoid cross-lens weight caching collisions.
    - This adapter expects floating-point tuned heads; quantized integer weights are
      rejected here. Upcasting/placement policies will live in the artifact loader.
    """

    def __init__(
        self,
        *,
        weights_pre: Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor]]] | None = None,
        weights_post: Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor]]] | None = None,
        strict: bool = True,
    ) -> None:
        self.weights_pre = dict(weights_pre or {})
        self.weights_post = dict(weights_post or {})
        self.strict = bool(strict)
        self.diag: Dict[str, Any] = {"missing_layers": []}

    def _get_head(self, layer_idx: int, probe_after_block: bool) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        table = self.weights_post if probe_after_block else self.weights_pre
        pair = table.get(layer_idx)
        if pair is None:
            if self.strict:
                raise KeyError(
                    f"No tuned head for layer {layer_idx} (probe_after_block={probe_after_block})"
                )
            # non-strict: record and skip this layer
            try:
                if layer_idx not in self.diag["missing_layers"]:
                    self.diag["missing_layers"].append(layer_idx)
            except Exception:
                self.diag["missing_layers"] = [layer_idx]
            return None
        return pair

    @staticmethod
    def _validate_head_shapes(W: torch.Tensor, b: Optional[torch.Tensor], X: torch.Tensor) -> None:
        if not torch.is_floating_point(W):
            raise ValueError("TunedLensAdapter expects floating-point W; got non-floating dtype")
        if b is not None and not torch.is_floating_point(b):
            raise ValueError("TunedLensAdapter expects floating-point b; got non-floating dtype")
        if W.dim() != 2:
            raise ValueError(f"W must be 2D (d_model×vocab); got shape {tuple(W.shape)}")
        if X.dim() != 2:
            raise ValueError(f"Input residual slice X must be 2D (seq×d_model); got shape {tuple(X.shape)}")
        d_model = X.shape[1]
        if W.shape[0] != d_model:
            raise ValueError(f"W rows {W.shape[0]} must equal d_model {d_model}")
        if b is not None:
            if b.dim() != 1:
                raise ValueError(f"b must be 1D (vocab,); got shape {tuple(b.shape)}")
            if b.shape[0] != W.shape[1]:
                raise ValueError(f"b length {b.shape[0]} must match W cols {W.shape[1]}")

    def forward(
        self,
        model: Any,
        layer_idx: int,
        residual: torch.Tensor,
        *,
        probe_after_block: bool,
        W_U: torch.Tensor,  # Unused; kept for interface compatibility
        b_U: Optional[torch.Tensor],  # Unused; kept for interface compatibility
        force_fp32_unembed: bool,
        cache: Optional[Dict[str, Any]] = None,
    ) -> Optional[torch.Tensor]:
        # Select tuned head for this probe timing
        pair = self._get_head(layer_idx, probe_after_block)
        if pair is None:
            return None
        W_t, b_t = pair

        # Normalize residual to match probe location
        norm_module = get_correct_norm_module(model, layer_idx, probe_after_block=probe_after_block)
        resid_norm = apply_norm_or_skip(residual, norm_module)

        # Vectorized decode for all positions with dtype-safe casting
        X = resid_norm[0, :, :]
        # Shape/dtype guards for clearer errors
        self._validate_head_shapes(W_t, b_t, X)
        X_cast = safe_cast_for_unembed(X, W_t, force_fp32_unembed=force_fp32_unembed)
        logits = unembed_mm(X_cast, W_t, b_t, cache=cache)
        return logits.float()

__all__ = ["TunedLensAdapter"]
