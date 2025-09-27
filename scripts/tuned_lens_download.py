#!/usr/bin/env python3
"""Download tuned lens artifacts from a Hugging Face dataset repository."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except Exception:
    print("Error: huggingface_hub is required. Install with `pip install huggingface_hub`.")
    raise


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download tuned lens artifacts from Hugging Face")
    p.add_argument("--repo-id", required=True, help="Source HF dataset repo id, e.g., <user>/logos-in-layers-tuned")
    p.add_argument("--repo-type", default="dataset", choices=["dataset"], help="HF repo type")
    p.add_argument("--models", nargs="*", help="Optional clean model names (subdirectories under tuned_lenses)")
    p.add_argument("--all", action="store_true", help="Fetch the entire tuned_lenses tree")
    p.add_argument("--local-dir", default="001_layers_baseline", help="Local base directory for artifacts")
    p.add_argument("--revision", default=None, help="Optional specific HF revision/commit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    local_dir = Path(args.local_dir)

    patterns = ["LICENSES/*", "README.md", "INDEX.json"]

    if args.all:
        patterns.append("tuned_lenses/*")
    elif args.models:
        for model in args.models:
            patterns.append(f"tuned_lenses/{model}/*")
    else:
        print("Nothing to download: pass --models <names> or --all")
        return

    print(f"Downloading from {args.repo_id} with patterns: {patterns}")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        allow_patterns=patterns,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        revision=args.revision,
    )
    print(f"Done. Artifacts available under {local_dir / 'tuned_lenses'}")


if __name__ == "__main__":
    main()

