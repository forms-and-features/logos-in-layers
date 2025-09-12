import _pathfix  # noqa: F401
import torch

from layers_core.probes import emit_test_prompts, emit_temperature_exploration


class DummyModel:
    def __init__(self, vocab_size: int, seq_len: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def to_tokens(self, prompt: str):
        # encode length as a deterministic sequence of token ids (unused numerically)
        L = max(1, min(self.seq_len, len(prompt.split()) or 1))
        return torch.zeros(1, L, dtype=torch.long)

    def __call__(self, tokens):
        # Return deterministic logits per position
        B, L = tokens.shape
        torch.manual_seed(L + self.vocab_size)
        return torch.randn(B, L, self.vocab_size)


def _decode_id(i):
    return f"t{int(i)}"


def test_emit_test_prompts_shapes_and_keys():
    torch.manual_seed(0)
    model = DummyModel(vocab_size=11, seq_len=5)
    prompts = ["a b c", "x y", "single"]
    out = emit_test_prompts(model, prompts, _decode_id)
    assert isinstance(out, list) and len(out) == len(prompts)
    for rec, src in zip(out, prompts):
        assert rec["type"] == "test_prompt"
        assert rec["prompt"] == src
        assert isinstance(rec["entropy"], float)
        assert isinstance(rec["topk"], list)
        assert len(rec["topk"]) == min(10, model.vocab_size)
        tok, prob = rec["topk"][0]
        assert isinstance(tok, str) and tok.startswith("t")
        assert isinstance(prob, float)


def test_emit_temperature_exploration_records_and_lengths():
    torch.manual_seed(0)
    model = DummyModel(vocab_size=13, seq_len=4)
    recs = emit_temperature_exploration(model, "probe", _decode_id)
    assert isinstance(recs, list) and len(recs) == 2
    temps = {r["temperature"] for r in recs}
    assert temps == {0.1, 2.0}
    for r in recs:
        assert r["type"] == "temperature_exploration"
        assert isinstance(r["entropy"], float)
        assert len(r["topk"]) == min(15, model.vocab_size)


if __name__ == "__main__":
    import traceback
    print("Running probes tests…")
    ok = True
    try:
        test_emit_test_prompts_shapes_and_keys(); print("✅ emit_test_prompts")
        test_emit_temperature_exploration_records_and_lengths(); print("✅ emit_temperature_exploration")
    except AssertionError as e:
        print("❌ assertion failed:", e); traceback.print_exc(); ok = False
    except Exception as e:
        print("❌ test crashed:", e); traceback.print_exc(); ok = False
    raise SystemExit(0 if ok else 1)

