import _pathfix  # noqa: F401
import torch

from layers_core.prism_utils import ensure_prism_Q_on


def test_ensure_prism_Q_on_success_cpu():
    Q = torch.eye(4)
    q_on, err = ensure_prism_Q_on(Q, torch.device("cpu"))
    assert err is None
    assert q_on is not None and q_on.device.type == "cpu"


def test_ensure_prism_Q_on_failure_with_non_tensor():
    class NotATensor:
        pass

    q_on, err = ensure_prism_Q_on(NotATensor(), torch.device("cpu"))  # type: ignore[arg-type]
    assert q_on is None and isinstance(err, str) and "prism Q placement failed" in err


if __name__ == "__main__":
    import traceback
    print("Running prism_utils tests…")
    ok = True
    try:
        test_ensure_prism_Q_on_success_cpu(); print("✅ success path")
        test_ensure_prism_Q_on_failure_with_non_tensor(); print("✅ failure path")
    except AssertionError as e:
        print("❌ assertion failed:", e); traceback.print_exc(); ok = False
    except Exception as e:
        print("❌ test crashed:", e); traceback.print_exc(); ok = False
    raise SystemExit(0 if ok else 1)

