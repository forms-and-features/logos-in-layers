#!/usr/bin/env python3
"""Unit tests for Prism whitening running moments and apply.

CPU-only; no external deps beyond torch.
"""

import _pathfix  # noqa: F401

import torch
import math

from layers_core.prism import RunningMoments, whiten_apply


def test_running_moments_and_whiten():
    torch.manual_seed(123)
    N, d = 1000, 7
    X = torch.randn(N, d) * 3.0 + 5.0  # nonzero mean, nonunit var
    rm = RunningMoments(dim=d, eps=1e-8)
    # Update in a few batches
    for i in range(0, N, 128):
        rm.update(X[i : i + 128])
    stats = rm.finalize()
    Xw = whiten_apply(X, stats)
    # Means close to 0; vars close to 1 (population moments)
    m = Xw.mean(dim=0)
    v = Xw.var(dim=0, unbiased=False)
    if not torch.allclose(m, torch.zeros_like(m), atol=5e-2) or not torch.allclose(v, torch.ones_like(v), atol=1e-1):
        print("[DEBUG prism-whiten] mean:", m.tolist())
        print("[DEBUG prism-whiten] var:", v.tolist())
    assert torch.allclose(m, torch.zeros_like(m), atol=5e-2)
    assert torch.allclose(v, torch.ones_like(v), atol=1e-1)


if __name__ == "__main__":
    import traceback
    print("Running prism whitening tests…")
    try:
        test_running_moments_and_whiten(); print("✅ running moments + whiten")
        raise SystemExit(0)
    except AssertionError as e:
        print("❌ assertion failed:", e); traceback.print_exc(); raise SystemExit(1)
    except Exception as e:
        print("❌ test crashed:", e); traceback.print_exc(); raise SystemExit(1)
