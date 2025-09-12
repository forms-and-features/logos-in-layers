from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch


@dataclass
class UnembedContext:
    """Holds unembedding dependencies passed around during analysis-only matmuls."""
    W: torch.Tensor
    b: Optional[torch.Tensor]
    force_fp32: bool
    cache: Dict[str, Any]


@dataclass
class PrismContext:
    """Holds Prism whitening/projection state for a pass."""
    stats: Any
    Q: Any
    active: bool
    placement_error: Optional[str] = None


__all__ = ["UnembedContext", "PrismContext"]

