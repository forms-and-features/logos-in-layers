import _pathfix  # noqa: F401
from layers_core.records import make_record, make_pure_record


def test_make_record_schema_and_topk():
    toks = ["A", "B", "C"]
    probs = [0.5, 0.3, 0.2]
    rec = make_record(
        prompt_id="pos",
        prompt_variant="orig",
        layer=1,
        pos=0,
        token="Hello",
        entropy=2.0,
        top_tokens=toks,
        top_probs=probs,
    )
    assert rec["type"] == "record"
    assert rec["prompt_id"] == "pos"
    assert rec["prompt_variant"] == "orig"
    assert rec["layer"] == 1 and rec["pos"] == 0
    assert rec["token"] == "Hello" and rec["entropy"] == 2.0
    assert rec["topk"] == [["A", 0.5], ["B", 0.3], ["C", 0.2]]


def test_make_pure_record_includes_extra():
    toks = ["X", "Y"]
    class P:
        def __init__(self, v): self.v=v
        def item(self): return self.v
    probs = [P(0.8), P(0.1)]
    extra = {"copy_collapse": True, "answer_rank": 3}
    rec = make_pure_record(
        prompt_id="pos",
        prompt_variant="orig",
        layer=5,
        pos=7,
        token="⟨NEXT⟩",
        entropy=0.42,
        top_tokens=toks,
        top_probs=probs,
        extra=extra,
    )
    assert rec["type"] == "pure_next_token_record"
    assert rec["topk"] == [["X", 0.8], ["Y", 0.1]]
    assert rec["copy_collapse"] is True and rec["answer_rank"] == 3

