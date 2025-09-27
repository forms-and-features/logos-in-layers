#!/usr/bin/env python3
"""Prefetch model safetensors + tokenizer/config into the HF cache.

Usage examples:
  - Use the project model list (excluding 70B by default):
      python scripts/prefetch_models.py --from-project
  - Prefetch specific repos:
      python scripts/prefetch_models.py --models mistralai/Mistral-7B-v0.1 google/gemma-2-9b
  - Include 70B as well:
      python scripts/prefetch_models.py --from-project --include-70b
  - Use a custom cache dir (fast NVMe), optionally enable hf_transfer first:
      HF_HOME=/mnt/nvme/hf_cache HF_HUB_ENABLE_HF_TRANSFER=1 \
      python scripts/prefetch_models.py --from-project

Notes:
  - For gated models (e.g., Llama family), run `huggingface-cli login` first.
  - This downloads only *.safetensors and minimal tokenizer/config/remote code files.
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable, List

try:
    from huggingface_hub import snapshot_download
except Exception as e:  # pragma: no cover
    print("❌ huggingface_hub is required. Install it in your venv:")
    print("   pip install huggingface_hub")
    raise


def project_models(include_70b: bool = False) -> List[str]:
    try:
        # Import within repo context
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "001_layers_baseline"))
        from models import CANDIDATE_MODELS  # type: ignore
        models = list(CANDIDATE_MODELS)
    except Exception:
        # Fallback to a hardcoded list matching 001_layers_baseline/models.py
        models = [
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

    if not include_70b:
        models = [m for m in models if "70B" not in m and "72B" not in m]
    return models


def prefetch(models: Iterable[str]) -> None:
    allow_patterns = [
        "*.safetensors",  # weights shards
        "*.json",         # config/tokenizer jsons
        "*.model",        # sentencepiece model
        "tokenizer.*",
        "vocab.*",
        "merges.*",
        "*.py",          # trust_remote_code modules if any
    ]
    for repo in models:
        print(f"[prefetch] {repo}")
        try:
            snapshot_download(
                repo_id=repo,
                allow_patterns=allow_patterns,
                local_files_only=False,
            )
        except Exception as e:
            print(f"[warn] failed to prefetch {repo}: {e}")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prefetch HF model safetensors + tokenizer/config")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--from-project", action="store_true", help="Use models from 001_layers_baseline/models.py")
    g.add_argument("--models", nargs="+", help="Explicit HF repo ids", default=None)
    p.add_argument("--include-70b", action="store_true", help="Include 70B/72B models when using --from-project")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    if args.from_project:
        repos = project_models(include_70b=args.include_70b)
    else:
        repos = args.models
    prefetch(repos)


if __name__ == "__main__":
    main()

