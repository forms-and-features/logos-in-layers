from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ExperimentConfig:
    device: str = "cuda"
    fp32_unembed: bool = False
    keep_residuals: bool = False
    copy_threshold: float = 0.90
    copy_margin: float = 0.05
    out_dir: Optional[str] = None
    self_test: bool = False

