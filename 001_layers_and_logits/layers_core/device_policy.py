import torch


def choose_dtype(device: str, model_id: str) -> torch.dtype:
    """Return the compute dtype for a given device and model id.

    - cuda: float16 by default; Gemma models use bfloat16
    - mps: float16
    - cpu: float32
    """
    device = device.lower()
    base = {
        "cuda": torch.float16,
        "mps": torch.float16,
        "cpu": torch.float32,
    }[device]

    if device == "cuda" and "gemma" in model_id.lower():
        return torch.bfloat16
    return base


def should_auto_promote_unembed(compute_dtype: torch.dtype) -> bool:
    """Auto-promote unembed to fp32 when the rest of the model runs in fp32."""
    return compute_dtype == torch.float32

