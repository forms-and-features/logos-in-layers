from __future__ import annotations

from typing import Dict, Optional, Any

import torch

from ..norm_utils import get_correct_norm_module, apply_norm_or_skip
from ..numerics import safe_cast_for_unembed
from ..unembed import unembed_mm
from ..tuned_lens import TunedTranslator
from .base import LensAdapter


class TunedLensAdapter(LensAdapter):
    """Adapter that runs the identity-plus-low-rank translator then decodes.

    The adapter mirrors the baseline normalization path but, instead of decoding
    the normalized residual directly, it passes the residual through a
    ``TunedTranslator`` before re-decoding with the model's tied unembedding.

    Translators are defined for *post-block* residuals only. When the caller
    probes a pre-block location (e.g. embeddings, layer 0) the adapter returns
    ``None`` unless ``strict`` is enabled, in which case a ``KeyError`` is
    raised. This keeps behaviour explicit while matching the plan's placement
    rules.
    """

    def __init__(self, *, translator: TunedTranslator, strict: bool = False) -> None:
        if not isinstance(translator, TunedTranslator):
            raise TypeError("translator must be a TunedTranslator instance")
        self.translator = translator
        self.strict = bool(strict)
        self.diag: Dict[str, Any] = {"missing_layers": []}

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
        if not probe_after_block:
            if self.strict:
                raise KeyError("Tuned lens translators are defined for post-block probes only")
            self.diag.setdefault("missing_layers", []).append(layer_idx)
            return None

        # Normalize residual to match probe location
        norm_module = get_correct_norm_module(model, layer_idx, probe_after_block=probe_after_block)
        resid_norm = apply_norm_or_skip(residual, norm_module)

        X = resid_norm[0, :, :]
        translated = self.translator(X, layer_idx)
        X_cast = safe_cast_for_unembed(translated, W_U, force_fp32_unembed=force_fp32_unembed)
        logits = unembed_mm(X_cast, W_U, b_U, cache=cache)
        tau = self.translator.temperature(layer_idx)
        logits = logits / tau.to(device=logits.device, dtype=logits.dtype)
        return logits.float()

__all__ = ["TunedLensAdapter"]
