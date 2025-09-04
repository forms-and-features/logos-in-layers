"""Core reusable helpers for layer-by-layer analysis.

Public API re-exports for convenience.
"""

from .norm_utils import (
    _get_rms_scale,
    apply_norm_or_skip,
    detect_model_architecture,
    get_correct_norm_module,
)
from .numerics import (
    bits_entropy_from_logits,
    safe_cast_for_unembed,
    kl_bits,
)
from .csv_io import (
    write_csv_files,
)
from .collapse_rules import (
    detect_copy_collapse,
    is_semantic_top1,
)
from .device_policy import (
    choose_dtype,
    should_auto_promote_unembed,
)
from .hooks import (
    build_cache_hook,
    attach_residual_hooks,
    detach_hooks,
)
from .run_dir import (
    setup_run_latest_directory,
)
from .config import (
    ExperimentConfig,
)
from .metrics import (
    compute_next_token_metrics,
)
from .windows import (
    WindowManager,
)
from .head_transforms import (
    detect_head_transforms,
)
from .unembed import (
    prepare_unembed_weights,
    unembed_mm,
)
from .records import (
    make_record,
    make_pure_record,
)

__all__ = [
    "_get_rms_scale",
    "apply_norm_or_skip",
    "detect_model_architecture",
    "get_correct_norm_module",
    # numerics
    "bits_entropy_from_logits",
    "safe_cast_for_unembed",
    "kl_bits",
    # csv io
    "write_csv_files",
    # collapse rules
    "detect_copy_collapse",
    "is_semantic_top1",
    # device policy
    "choose_dtype",
    "should_auto_promote_unembed",
    # hooks
    "build_cache_hook",
    "attach_residual_hooks",
    "detach_hooks",
    # run dir
    "setup_run_latest_directory",
    # config
    "ExperimentConfig",
    # metrics
    "compute_next_token_metrics",
    # windows
    "WindowManager",
    # head transforms
    "detect_head_transforms",
    # unembed helpers
    "prepare_unembed_weights",
    "unembed_mm",
    # records
    "make_record",
    "make_pure_record",
]
