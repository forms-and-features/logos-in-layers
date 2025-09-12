import torch
from typing import Callable, Iterable, List, Dict, Any

from .numerics import bits_entropy_from_logits


def emit_test_prompts(
    model,
    prompts: Iterable[str],
    decode_id: Callable[[Any], str],
) -> List[Dict[str, Any]]:
    """Emit per-prompt top-k summaries for a list of test prompts.

    Behavior is byte/shape-identical to the previous inline implementation in
    run.py: computes next-token logits for each prompt, clamps top-k to vocab
    size, returns a list of dict records with keys: type, prompt, entropy, topk.
    """
    records: List[Dict[str, Any]] = []
    with torch.no_grad():
        for test_prompt in prompts:
            # Let TL/Accelerate handle device placement
            test_tokens = model.to_tokens(test_prompt)
            test_logits = model(test_tokens)
            last_slice = test_logits[0, -1, :]
            k = min(10, int(last_slice.shape[-1]))
            _, top_idx = torch.topk(last_slice, k, largest=True, sorted=True)
            full_probs = torch.softmax(last_slice, dim=0)
            top_probs = full_probs[top_idx]
            entropy_bits = bits_entropy_from_logits(last_slice)
            rec = {
                "type": "test_prompt",
                "prompt": test_prompt,
                "entropy": float(entropy_bits),
                "topk": [[decode_id(idx), prob.item()] for prob, idx in zip(top_probs, top_idx)],
            }
            records.append(rec)
            # Cleanup per-iteration tensors promptly
            del test_tokens, test_logits, top_idx, full_probs, top_probs
    return records


def emit_temperature_exploration(
    model,
    prompt: str,
    decode_id: Callable[[Any], str],
) -> List[Dict[str, Any]]:
    """Emit two temperature-scaled top-k summaries for a fixed prompt.

    Temperatures are fixed at [0.1, 2.0]. Top-k is clamped to min(15, vocab).
    Matches previous inline logic in run.py including CUDA cleanup guards.
    """
    out: List[Dict[str, Any]] = []
    with torch.no_grad():
        temp_tokens = model.to_tokens(prompt)
        base_logits = model(temp_tokens)[0, -1, :]
        for temp in (0.1, 2.0):
            scaled_logits = (base_logits / temp).float()
            k = min(15, int(scaled_logits.shape[-1]))
            _, top_idx = torch.topk(scaled_logits, k, largest=True, sorted=True)
            full_probs = torch.softmax(scaled_logits, dim=0)
            top_probs = full_probs[top_idx]
            entropy_bits = bits_entropy_from_logits(scaled_logits)
            rec = {
                "type": "temperature_exploration",
                "temperature": float(temp),
                "entropy": float(entropy_bits),
                "topk": [[decode_id(idx), prob.item()] for prob, idx in zip(top_probs, top_idx)],
            }
            out.append(rec)
            del scaled_logits, top_idx, full_probs, top_probs
        del temp_tokens, base_logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    return out
