import _pathfix  # noqa: F401
import torch

from layers_core.token_utils import make_decode_id


class _TokDecode:
    def decode(self, ids):
        # expects a list of one id
        return f"T{int(ids[0])}"


class _TokConvertOnly:
    def convert_ids_to_tokens(self, ids):
        # returns list of tokens
        return [f"tok{int(ids[0])}"]


class _TokBroken:
    # no decode/convert methods
    pass


def test_make_decode_id_uses_decode_int_and_tensor():
    d = make_decode_id(_TokDecode())
    assert d(7) == "T7"
    assert d(torch.tensor(5)) == "T5"


def test_make_decode_id_fallback_convert_ids_to_tokens():
    d = make_decode_id(_TokConvertOnly())
    assert d(3) == "tok3"
    assert d(torch.tensor(9)) == "tok9"


def test_make_decode_id_string_fallback():
    d = make_decode_id(_TokBroken())
    assert d(42) == "42"
    assert d(torch.tensor(11)) == "11"


if __name__ == "__main__":
    import traceback
    print("Running token_utils tests…")
    ok = True
    try:
        test_make_decode_id_uses_decode_int_and_tensor(); print("✅ decode path")
        test_make_decode_id_fallback_convert_ids_to_tokens(); print("✅ convert fallback")
        test_make_decode_id_string_fallback(); print("✅ string fallback")
    except AssertionError as e:
        print("❌ assertion failed:", e); traceback.print_exc(); ok = False
    except Exception as e:
        print("❌ test crashed:", e); traceback.print_exc(); ok = False
    raise SystemExit(0 if ok else 1)

