import math
import torch


def bits_entropy_from_logits(logits: torch.Tensor) -> float:
    """Shannon entropy in bits computed from logits, zero-safe and fp32-stable.

    Uses log_softmax to avoid catastrophic cancellation and masks the 0·log(0)
    case to prevent NaNs under saturated distributions. All math in float32.
    """
    logp = torch.log_softmax(logits.to(dtype=torch.float32), dim=-1)
    p = torch.exp(logp)
    term = torch.where(p > 0, p * logp, torch.zeros_like(p))
    ent_nats = -term.sum()
    return float(ent_nats / math.log(2))


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


def kl_bits(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-30) -> float:
    """Kullback–Leibler divergence KL(p || q) in bits.

    Expects 1D probability tensors over the same support. Performs fp32 math and
    guards logs with a small epsilon to avoid log(0). Returns a Python float.
    """
    p32 = p.to(dtype=torch.float32)
    q32 = q.to(dtype=torch.float32)
    logp = (p32 + eps).log()
    logq = (q32 + eps).log()
    kl_nats = torch.sum(p32 * (logp - logq)).detach().cpu().item()
    return kl_nats / math.log(2)
