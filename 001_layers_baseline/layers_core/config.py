from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ExperimentConfig:
    device: str = "cuda"
    fp32_unembed: bool = False
    keep_residuals: bool = False
    copy_threshold: float = 0.95
    copy_margin: float = 0.10
    copy_window_k: int = 1
    entropy_collapse_threshold: float = 1.0
    out_dir: Optional[str] = None
    self_test: bool = False
