import torch
import re
from typing import Optional, Tuple

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil may be unavailable in some envs
    psutil = None
import os


def choose_dtype(device: str, model_id: str) -> torch.dtype:
    """Return the compute dtype for a given device and model id.

    - cuda: float16 by default; Gemma models use bfloat16
    - mps: float16
    - cpu: float32 (but prefer bfloat16 for very large models to reduce RAM)
    """
    device = device.lower()
    base = {
        "cuda": torch.float16,
        "mps": torch.float16,
        "cpu": torch.float32,
    }[device]

    # CUDA Gemma override
    if device == "cuda" and "gemma" in model_id.lower():
        return torch.bfloat16

    # CPU: prefer bfloat16 for very large checkpoints to avoid FP32 blow-up
    if device == "cpu":
        params = resolve_param_count(model_id) or 0.0
        # Threshold chosen to catch 30B+ class (e.g., Yi-34B) on 256GB hosts
        if params >= 3.0e10:
            return torch.bfloat16

    return base


def should_auto_promote_unembed(compute_dtype: torch.dtype) -> bool:
    """Auto-promote unembed to fp32 when main compute runs in bf16/fp16.

    Rationale: Keeping weights in bf16/fp16 saves memory, but decoding logits via
    a float32 unembedding stabilizes small logit gaps and entropy with negligible
    memory cost. For pure fp32 compute, promotion is unnecessary.
    """
    return compute_dtype in (torch.float16, torch.bfloat16)


# ---- Dynamic fit policy -----------------------------------------------------

# Conservative parameter counts (approximate, used for fit estimation)
_PARAMS_MAP = {
    # Smaller
    "mistralai/Mistral-7B-v0.1": 7.3e9,
    "meta-llama/Meta-Llama-3-8B": 8.0e9,
    "Qwen/Qwen3-8B": 8.0e9,
    "google/gemma-2-9b": 9.2e9,
    # Larger
    "Qwen/Qwen3-14B": 14.0e9,
    "google/gemma-2-27b": 27.0e9,
    "01-ai/Yi-34B": 34.0e9,
}

_OVERHEAD_FACTOR = {
    # Activation + temp buffers for short prompts; conservative
    "cuda": 0.30,
    "mps": 0.35,
    "cpu": 0.20,
}

_RESERVE_BYTES = {
    # Headroom reserved for runtime/OS fragmentation
    "cuda": int(1.5 * (1024 ** 3)),  # ~1.5 GB
    "mps": int(3.5 * (1024 ** 3)),  # ~3.5 GB (UMA tends to be tighter)
    "cpu": int(4.0 * (1024 ** 3)),  # ~4 GB
}

# Extra reserve for very large CPU loads where transient duplication is likely
_LARGE_CPU_EXTRA_RESERVE_BYTES = int(12.0 * (1024 ** 3))  # ~12 GB


def _bytes_per_param(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype == torch.float32:
        return 4
    # Default conservative assumption
    return 4


def _assumed_checkpoint_bpp(model_id: str) -> int:
    """Best-effort guess of on-disk checkpoint bytes/param.

    Most modern large checkpoints ship in bf16/fp16 (2 bytes/param). Fall back
    to 4 if unsure. We bias toward 2 for common families to be realistic about
    load-time conversion peaks when CPU target dtype is fp32.
    """
    tail = model_id.lower()
    if any(k in tail for k in ["llama", "mistral", "gemma", "qwen", "yi", "falcon", "gpt-neox", "gpt-j"]):
        return 2
    return 2  # default optimistic but common in practice


def resolve_param_count(model_id: str) -> Optional[float]:
    """Return approximate parameter count for a model (in units of params).

    Uses a curated map first; falls back to parsing the trailing repo name for
    patterns like "7B", "8B", "14B", "27b", "34B". Returns None if unknown.
    """
    if model_id in _PARAMS_MAP:
        return float(_PARAMS_MAP[model_id])

    tail = model_id.split("/")[-1]
    m = re.search(r"(\d+)(?:\.(\d+))?\s*[bB]", tail)
    if m:
        whole = int(m.group(1))
        frac = m.group(2)
        val = whole + (int(frac) / (10 ** len(frac)) if frac else 0.0)
        return float(val) * 1e9
    return None


def estimate_model_peak_bytes(model_id: str, device: str, dtype: torch.dtype) -> Optional[int]:
    """Estimate peak memory requirement for loading + running short prompts.

    peak ≈ params * bytes_per_param * (1 + overhead_factor) + reserve_bytes

    Returns None if param count is unknown (caller can choose to attempt load or skip).
    """
    params = resolve_param_count(model_id)
    if params is None:
        return None
    bpp = _bytes_per_param(dtype)
    overhead = _OVERHEAD_FACTOR.get(device, 0.30)
    reserve = _RESERVE_BYTES.get(device, int(2.0 * (1024 ** 3)))
    weights = int(params * bpp)

    # Base estimate (steady-state after load)
    peak = int(weights * (1.0 + overhead) + reserve)

    # For very large CPU models targeting FP32, account for load-time duplication:
    # source (often bf16/fp16) + destination (fp32) may coexist transiently.
    if device == "cpu" and dtype == torch.float32 and params >= 2.5e10:
        src_bpp = _assumed_checkpoint_bpp(model_id)
        dup_bytes = int(params * src_bpp)
        # Apply overhead only once (assume duplication dominates activations), add extra reserve
        peak = int((weights + dup_bytes) * (1.0 + max(overhead, 0.10)) + reserve + _LARGE_CPU_EXTRA_RESERVE_BYTES)

    return peak


def available_memory_bytes(device: str) -> Optional[int]:
    """Best-effort available memory for the device.

    - cuda: torch.cuda.mem_get_info()[0]
    - mps/cpu: psutil.virtual_memory().available (if psutil present), else None
    """
    device = device.lower()
    if device == "cuda" and torch.cuda.is_available():
        try:
            free, total = torch.cuda.mem_get_info()  # type: ignore[attr-defined]
            return int(free)
        except Exception:
            return None
    # CPU/MPS: try psutil first, then os.sysconf fallback
    if psutil is not None:
        try:
            return int(psutil.virtual_memory().available)
        except Exception:
            pass
    # POSIX fallback without psutil
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")  # bytes
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        return int(page_size * avail_pages)
    except Exception:
        return None


def select_best_device(model_id: str) -> Optional[Tuple[str, torch.dtype, dict]]:
    """Pick the best single device that can fit the model.

    Preference order: cuda → mps → cpu. For each device, compute dtype via
    choose_dtype, estimate the peak bytes, compare with available memory. If
    fits, return (device, dtype, debug_info). If no device fits, return None.

    debug_info includes: params, dtype, est_peak, avail, overhead, reserve.
    """
    decisions = []
    for dev in ("cuda", "mps", "cpu"):
        # Skip devices that are not available at all
        if dev == "cuda" and not torch.cuda.is_available():
            continue
        if dev == "mps" and not torch.backends.mps.is_available():
            continue

        dtype = choose_dtype(dev, model_id)
        est = estimate_model_peak_bytes(model_id, dev, dtype)
        avail = available_memory_bytes(dev)
        overhead = _OVERHEAD_FACTOR.get(dev)
        reserve = _RESERVE_BYTES.get(dev)
        decisions.append((dev, dtype, est, avail, overhead, reserve))

        if est is None or avail is None:
            # If unknown, be conservative: do not select
            continue
        if est < avail:
            debug = {
                "params": resolve_param_count(model_id),
                "dtype": str(dtype),
                "est_peak": est,
                "available": avail,
                "overhead": overhead,
                "reserve": reserve,
            }
            return dev, dtype, debug

    return None
