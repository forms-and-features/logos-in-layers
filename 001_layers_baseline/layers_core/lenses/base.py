from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

import torch


class LensAdapter(ABC):
    """Minimal interface for a lens adapter that maps residuals to logits.

    Implementations must reproduce the existing path numerics exactly
    (normalization choice, dtype handling, and unembedding) so downstream
    metrics and outputs remain byte‑for‑byte identical.
    """

    @abstractmethod
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
        """Return logits for all positions (shape: [seq_len, vocab]) as float32."""
        raise NotImplementedError

