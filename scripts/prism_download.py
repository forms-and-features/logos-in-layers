#!/usr/bin/env python3
"""
Download Prism artifacts from a Hugging Face dataset repo into the local folder.

Prereqs:
- If the repo is private, run `huggingface-cli login`.
- For public repos, auth is not required.

Usage examples:
  # Download a single model's artifacts
  python scripts/prism_download.py --repo-id <user>/logos-in-layers-prism --models Meta-Llama-3-8B

  # Download multiple models
  python scripts/prism_download.py --repo-id <user>/logos-in-layers-prism --models Meta-Llama-3-8B Qwen3-8B

  # Download all artifacts
  python scripts/prism_download.py --repo-id <user>/logos-in-layers-prism --all

By default, files are placed under `001_layers_baseline/prisms/` so run.py
in prism=auto mode will find them automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except Exception as e:
    print("Error: huggingface_hub is required. Install with `pip install huggingface_hub`.")
    raise


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Prism artifacts from a Hugging Face dataset repo")
    p.add_argument("--repo-id", required=True, help="Source HF dataset repo id, e.g., <user>/logos-in-layers-prism")
    p.add_argument("--repo-type", default="dataset", choices=["dataset"], help="HF repo type (fixed to dataset)")
    p.add_argument("--models", nargs="*", help="Optional list of clean model names to fetch (subfolders under prisms)")
    p.add_argument("--all", action="store_true", help="Fetch all models (overrides --models)")
    p.add_argument("--local-dir", default="001_layers_baseline", help="Local base directory to place artifacts")
    p.add_argument("--revision", default=None, help="Optional HF revision/commit to pin")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    local_dir = Path(args.local_dir)
    allow_patterns = None
    if args.all:
        allow_patterns = ["prisms/*/*"]
    elif args.models:
        allow_patterns = [f"prisms/{m}/*" for m in args.models]
    else:
        print("Nothing to download: pass --models <names> or --all")
        return

    print(f"Downloading from {args.repo_id} (type={args.repo_type}) with patterns: {allow_patterns}")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        allow_patterns=allow_patterns,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        revision=args.revision,
    )
    print(f"Done. Artifacts placed under: {local_dir / 'prisms'}")


if __name__ == "__main__":
    main()

