"""Shared model registry for 001_layers_baseline.

Exposes the curated list of candidate models evaluated by the baseline suite
and (later) by the Prism fitter. Keep this as the single source of truth.
"""

# Candidate models (small → large). Device fit is decided per model at runtime.
CANDIDATE_MODELS = [
    "mistralai/Mistral-7B-v0.1",
    "meta-llama/Meta-Llama-3-8B",
    "Qwen/Qwen3-8B",
    "google/gemma-2-9b",
    "Qwen/Qwen3-14B",
    "google/gemma-2-27b",
    "01-ai/Yi-34B",
    "meta-llama/Meta-Llama-3-70B",
    "mistralai/Mistral-Small-24B-Base-2501",
    "Qwen/Qwen2.5-72B",
]

__all__ = ["CANDIDATE_MODELS"]

