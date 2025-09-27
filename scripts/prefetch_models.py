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
from huggingface_hub import snapshot_download, HfApi
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


def _would_download(files: Iterable[str], allow_patterns: List[str], ignore_patterns: List[str]) -> List[str]:
    import fnmatch
    allow_any = lambda f: any(fnmatch.fnmatch(f, pat) for pat in allow_patterns)
    ignore_any = lambda f: any(fnmatch.fnmatch(f, pat) for pat in ignore_patterns)
    return [f for f in files if allow_any(f) and not ignore_any(f)]


def prefetch(models: Iterable[str], *, include_consolidated: bool = False, dry_run: bool = False) -> None:
    allow_patterns = [
        "*.safetensors",  # weights shards
        "*.json",         # config/tokenizer jsons
        "*.model",        # sentencepiece model
        "tokenizer.*",
        "vocab.*",
        "merges.*",
        "*.py",          # trust_remote_code modules if any
    ]
    ignore_patterns: List[str] = []
    if not include_consolidated:
        # Exclude legacy consolidated files and vendor-specific dumps we don't need
        ignore_patterns += [
            "consolidated*",   # e.g., consolidated.00.pth / consolidated.safetensors
            "original/*",      # some repos keep original dumps under this path
            "*/original/*",
        ]

    api = HfApi()
    for repo in models:
        try:
            files = api.list_repo_files(repo_id=repo)
        except Exception:
            files = []

        if dry_run:
            kept = _would_download(files, allow_patterns, ignore_patterns)
            print(f"[dry-run] {repo}: {len(kept)} files would be cached")
            for f in kept:
                print(f"  - {f}")
            continue

        print(f"[prefetch] {repo}")
        try:
            snapshot_download(
                repo_id=repo,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns or None,
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
    p.add_argument("--include-consolidated", action="store_true", help="Also fetch legacy consolidated* files (default: skipped)")
    p.add_argument("--dry-run", action="store_true", help="List files that would be fetched and exit")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    if args.from_project:
        repos = project_models(include_70b=args.include_70b)
    else:
        repos = args.models
    prefetch(repos, include_consolidated=args.include_consolidated, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
