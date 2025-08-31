#!/usr/bin/env python3
"""
Upload Prism artifacts to a Hugging Face dataset repo.

Prereqs:
- Run `huggingface-cli login` once (token with write permission).
- Generate artifacts locally via `cd 001_layers_baseline && python prism_fit.py`.

Usage examples:
  python scripts/prism_upload.py --repo-id <user_or_org>/logos-in-layers-prism
  python scripts/prism_upload.py --repo-id <user_or_org>/logos-in-layers-prism --models Meta-Llama-3-8B Qwen3-8B

Notes:
- By default uploads the entire `001_layers_baseline/prisms/` directory to `prisms/` in the repo.
- Use `--models` to upload only specific subfolders.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import upload_folder
except Exception as e:
    print("Error: huggingface_hub is required. Install with `pip install huggingface_hub`.")
    raise


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload Prism artifacts to a Hugging Face dataset repo")
    p.add_argument("--repo-id", required=True, help="Target HF dataset repo id, e.g., <user>/logos-in-layers-prism")
    p.add_argument("--repo-type", default="dataset", choices=["dataset"], help="HF repo type (fixed to dataset)")
    p.add_argument("--root", default="001_layers_baseline/prisms", help="Local artifacts root directory")
    p.add_argument("--path-in-repo", default="prisms", help="Destination path inside the repo")
    p.add_argument("--models", nargs="*", help="Optional list of clean model names to upload (subfolders under root)")
    p.add_argument("--commit-message", default="Add/Update Prism artifacts", help="Commit message")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        print(f"Error: local artifacts root not found: {root}")
        sys.exit(1)

    if args.models:
        for m in args.models:
            sub = root / m
            if not sub.exists():
                print(f"Warning: skipping missing model subfolder: {sub}")
                continue
            print(f"Uploading {sub} → {args.repo_id}:{args.path_in_repo}/{m}")
            upload_folder(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                folder_path=str(sub),
                path_in_repo=f"{args.path_in_repo}/{m}",
                commit_message=args.commit_message,
            )
    else:
        print(f"Uploading {root} → {args.repo_id}:{args.path_in_repo}")
        upload_folder(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            folder_path=str(root),
            path_in_repo=args.path_in_repo,
            commit_message=args.commit_message,
        )

    print("Done.")


if __name__ == "__main__":
    main()
