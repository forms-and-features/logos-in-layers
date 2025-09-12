from __future__ import annotations

from typing import Optional, Dict, Any

import torch

from ..norm_utils import get_correct_norm_module, apply_norm_or_skip
from ..unembed import unembed_mm
from ..prism import whiten_apply, WhitenStats
from .base import LensAdapter


class PrismLensAdapter(LensAdapter):
    """Prism lens: normalization → whitening → orthogonal map Q → unembed.

    Behavior mirrors the existing inline Prism path used by passes.py:
    - Uses architecture-aware normalization (fp32 stats) before whitening.
    - Applies diagonal whitening, then an orthogonal map Q (both fp32).
    - Uses float32 copies of W_U/b_U for the matmul to match prior numerics.

    Placement policy:
    - Q is moved to the device of the whitened residual on first use.
    - If placement fails, the adapter disables itself and surfaces a
      'placement_error' in `diag` and subsequent forward() calls no-op by
      returning None.
    """

    def __init__(self, stats: Optional[WhitenStats], Q: Optional[torch.Tensor], active: bool) -> None:
        self.enabled: bool = bool(active and (stats is not None) and (Q is not None))
        self.stats: Optional[WhitenStats] = stats
        self._Q: Optional[torch.Tensor] = Q.float() if Q is not None else None
        self._placed: bool = False
        self.diag: Dict[str, Any] = {}

    def _ensure_Q_on(self, device: torch.device) -> bool:
        if not self.enabled:
            return False
        if self._Q is None:
            self.enabled = False
            return False
        if self._placed and self._Q.device == device:
            return True
        try:
            self._Q = self._Q.to(device)
            self._placed = True
            return True
        except Exception as e:  # placement failed
            self.diag["placement_error"] = f"prism Q placement failed: {type(e).__name__}"
            self.enabled = False
            return False

    def forward(
        self,
        model: Any,
        layer_idx: int,
        residual: torch.Tensor,
        *,
        probe_after_block: bool,
        W_U: torch.Tensor,
        b_U: Optional[torch.Tensor],
        force_fp32_unembed: bool,  # Intentionally unused: Prism always unembeds in fp32 for parity
        cache: Optional[Dict[str, Any]] = None,
    ) -> Optional[torch.Tensor]:
        if not self.enabled:
            return None

        if self.stats is None or self._Q is None:
            self.enabled = False
            return None

        # Normalize matching the probe location
        norm_module = get_correct_norm_module(model, layer_idx, probe_after_block=probe_after_block)
        resid_norm = apply_norm_or_skip(residual, norm_module)

        # Whitening expects 2D (seq×d); residual is [1, seq, d]
        X = resid_norm[0]
        Xw = whiten_apply(X, self.stats)

        # Ensure Q is on the same device
        if not self._ensure_Q_on(Xw.device):
            return None

        # Apply orthogonal map Q: (seq×d) @ (d×d) → (seq×d)
        Xp = Xw @ self._Q  # type: ignore[arg-type]

        # Use float32 copies of W_U/b_U to match prior Prism numerics
        Wp = W_U.float() if W_U.dtype != torch.float32 else W_U
        bp = (b_U.float() if (b_U is not None and b_U.dtype != torch.float32) else b_U)
        logits = unembed_mm(Xp, Wp, bp, cache=cache)
        return logits.float()
