from __future__ import annotations

from typing import Optional, Dict, Any

import torch

from ..norm_utils import get_correct_norm_module, apply_norm_or_skip
from ..numerics import safe_cast_for_unembed
from ..unembed import unembed_mm
from .base import LensAdapter


class NormLensAdapter(LensAdapter):
    """Baseline normalization + unembedding path as a pluggable lens.

    Behavior-preserving: selects the same normalization module as run.py using
    get_correct_norm_module, applies normalization with fp32 stats, casts the
    residual to match W_U as needed, and performs the unembed matmul.
    """

    def forward(
        self,
        model: Any,
        layer_idx: int,
        residual: torch.Tensor,
        *,
        probe_after_block: bool,
        W_U: torch.Tensor,
        b_U: Optional[torch.Tensor],
        force_fp32_unembed: bool,
        cache: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        # Select and apply the correct normalization for this probe timing
        norm_module = get_correct_norm_module(model, layer_idx, probe_after_block=probe_after_block)
        resid_norm = apply_norm_or_skip(residual, norm_module)

        # Vectorized unembedding for all positions
        # Input residual shape is [1, seq, d_model]; take the batch=1 slice
        casted = safe_cast_for_unembed(
            resid_norm[0, :, :],
            W_U,
            force_fp32_unembed=force_fp32_unembed,
        )
        logits = unembed_mm(casted, W_U, b_U, cache=cache)
        return logits.float()

