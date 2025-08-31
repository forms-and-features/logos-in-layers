#!/usr/bin/env python3
"""CPU-only tests for device/dtype policy."""

import _pathfix  # noqa: F401

import torch
from layers_core.device_policy import (
    choose_dtype,
    should_auto_promote_unembed,
    resolve_param_count,
    estimate_model_peak_bytes,
    available_memory_bytes,
    select_best_device,
)


def test_choose_dtype_table_and_gemma_override():
    assert choose_dtype("cpu", "any/model") == torch.float32
    assert choose_dtype("mps", "any/model") == torch.float16
    assert choose_dtype("cuda", "mistralai/Mistral-7B-v0.1") == torch.float16
    assert choose_dtype("cuda", "google/gemma-2-9b") == torch.bfloat16
    assert choose_dtype("cuda", "GEMMA-2-27B") == torch.bfloat16


def test_choose_dtype_cpu_large_prefers_bf16():
    # Very large CPU models (e.g., Yi-34B) should prefer bfloat16 to reduce RAM
    assert choose_dtype("cpu", "01-ai/Yi-34B") == torch.bfloat16
    # Below threshold (e.g., 27B) remains fp32 on CPU
    assert choose_dtype("cpu", "google/gemma-2-27b") == torch.float32


def test_should_auto_promote_unembed():
    # Auto-promote when main compute runs in low precision
    assert should_auto_promote_unembed(torch.float32) is False
    assert should_auto_promote_unembed(torch.float16) is True
    assert should_auto_promote_unembed(torch.bfloat16) is True


def test_resolve_param_count_known_and_parsed():
    assert resolve_param_count("mistralai/Mistral-7B-v0.1") is not None
    assert resolve_param_count("meta-llama/Meta-Llama-3-8B") == 8.0e9
    assert resolve_param_count("google/gemma-2-27b") == 27.0e9
    # Fallback parse from tail
    assert resolve_param_count("org/SomeModel-14B") == 14.0e9
    assert resolve_param_count("org/Model-9.2b") == 9.2e9


def test_estimate_model_peak_bytes_monotonic():
    model = "meta-llama/Meta-Llama-3-8B"
    fp32 = estimate_model_peak_bytes(model, "cpu", torch.float32)
    fp16 = estimate_model_peak_bytes(model, "cuda", torch.float16)
    assert fp32 is not None and fp16 is not None
    # fp32 should be roughly double fp16 weight size
    assert fp32 > fp16


def test_cpu_fp32_duplication_estimate_for_large_models():
    # For large params and CPU FP32, estimator should account for load-time duplication
    import layers_core.device_policy as dp
    model = "01-ai/Yi-34B"

    # Compute steady-state estimate (simulate by temporarily forcing small params duplication path off)
    params = dp.resolve_param_count(model)
    assert params is not None and params > 3.0e10

    # Manually compute steady-state baseline: weights*(1+overhead)+reserve
    bpp = 4  # fp32
    overhead = dp._OVERHEAD_FACTOR["cpu"]
    reserve = dp._RESERVE_BYTES["cpu"]
    steady = int(params * bpp * (1.0 + overhead) + reserve)

    est = estimate_model_peak_bytes(model, "cpu", torch.float32)
    # Expect the estimator to be strictly larger than the steady-state baseline
    assert est > steady

    # And when using bf16 on CPU, estimate should drop substantially
    est_bf16 = estimate_model_peak_bytes(model, "cpu", torch.bfloat16)
    assert est_bf16 < est


def test_select_best_device_prefers_gpu_when_memory_sufficient(monkeypatch):
    model = "meta-llama/Meta-Llama-3-8B"

    # Monkeypatch availability: pretend CUDA available
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    # Monkeypatch available memory to simulate capacities
    import layers_core.device_policy as dp

    def fake_avail(dev: str):
        # Plenty for cuda, little for cpu
        if dev == "cuda":
            return 40 * 1024**3  # 40 GB
        return 8 * 1024**3       # 8 GB

    monkeypatch.setattr(dp, "available_memory_bytes", fake_avail)

    sel = select_best_device(model)
    assert sel is not None
    dev, dtype, debug = sel
    assert dev == "cuda"
