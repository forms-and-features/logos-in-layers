import math
import torch


def bits_entropy_from_logits(logits: torch.Tensor) -> float:
    """Shannon entropy in bits computed from logits, numerically safe."""
    eps = 1e-40
    probs = logits.softmax(dim=-1).float()
    log_probs = (probs + eps).log()
    ent_nats = -(probs * log_probs).sum()
    return (ent_nats / math.log(2)).item()


def safe_cast_for_unembed(resid: torch.Tensor, W_U: torch.Tensor, *, force_fp32_unembed: bool = False) -> torch.Tensor:
    """Cast residual to match unembedding expectations without breaking quantized kernels.

    - If `force_fp32_unembed` and `W_U` is float32, cast residual to float32.
    - If `W_U` is a floating-point tensor (fp16/bf16/fp32), cast residual to `W_U.dtype`.
    - Otherwise (e.g., int8/4-bit quantized), leave residual unchanged.
    """
    if force_fp32_unembed and W_U.dtype == torch.float32:
        return resid.float()
    elif torch.is_floating_point(W_U):
        return resid.to(dtype=W_U.dtype)
    else:
        return resid

