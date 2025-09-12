from __future__ import annotations

from typing import Optional, Tuple
import torch


def ensure_prism_Q_on(Q: torch.Tensor, device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    """Best-effort placement of the Prism map `Q` onto `device`.

    Returns (Q_on_device, error). On success, error is None. On failure,
    returns (None, short_error_string) and the caller is expected to disable
    Prism use for the current pass.
    """
    try:
        if Q.device == device:
            return Q, None
        return Q.to(device), None
    except Exception as e:
        return None, f"prism Q placement failed: {type(e).__name__}"


__all__ = ["ensure_prism_Q_on"]

