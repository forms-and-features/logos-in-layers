#!/usr/bin/env python3
"""Upload tuned lens artifacts to a Hugging Face dataset repository.

The script regenerates `INDEX.json` (with SHA-256 checksums per file) before
uploading the entire `001_layers_baseline/tuned_lenses/` subtree. This keeps the
remote dataset consistent with local artifacts.

Prerequisites
-------------
- Install `huggingface_hub` (`pip install huggingface_hub`).
- Run `huggingface-cli login` once if the target repo is private / requires
  authentication.

Usage examples
--------------
```
python scripts/tuned_lens_upload.py --repo-id <user>/logos-in-layers-tuned
python scripts/tuned_lens_upload.py --repo-id <user>/logos-in-layers-tuned --commit-message "Update tuned lens sweep"
```

Notes
-----
- By default the script uploads the full tuned lens tree so that `INDEX.json`
  and per-model metadata remain in sync. Use git to stage only the models you
  want to publish before running the upload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

try:
    from huggingface_hub import upload_folder
except Exception:
    print("Error: huggingface_hub is required. Install with `pip install huggingface_hub`.")
    raise


DEFAULT_ROOT = Path("001_layers_baseline/tuned_lenses")
INDEX_FILENAME = "INDEX.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_index(root: Path) -> Dict[str, Dict[str, object]]:
    index: Dict[str, Dict[str, object]] = {}
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        rel = model_dir.relative_to(root)
        model_key = rel.as_posix()

        sha_map: Dict[str, str] = {}
        for file in sorted(model_dir.glob("*")):
            if file.is_file():
                sha_map[file.name] = sha256_file(file)

        entry: Dict[str, object] = {
            "path": f"tuned_lenses/{model_key}",
            "sha256": sha_map,
        }

        provenance_path = model_dir / "provenance.json"
        if provenance_path.exists():
            try:
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                provenance = {}
            translator_meta = provenance.get("translator", {})
            entry["tl_version"] = translator_meta.get("version") or translator_meta.get("tl_version") or "unversioned"
            entry["model_commit"] = provenance.get("model_commit")
            entry["tokenizer_commit"] = provenance.get("tokenizer_commit")
        else:
            entry["tl_version"] = "missing"
            entry["model_commit"] = None
            entry["tokenizer_commit"] = None

        index[model_key] = entry

    return index


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload tuned lens artifacts to Hugging Face")
    p.add_argument("--repo-id", required=True, help="Target HF dataset repo id, e.g., <user>/logos-in-layers-tuned")
    p.add_argument("--repo-type", default="dataset", choices=["dataset"], help="HF repo type")
    p.add_argument("--root", default=str(DEFAULT_ROOT), help="Local tuned lens root directory")
    p.add_argument("--path-in-repo", default="tuned_lenses", help="Destination path inside the repo")
    p.add_argument("--commit-message", default=None, help="Commit message for the upload")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Error: local artifacts root not found: {root}")
        sys.exit(1)

    index = build_index(root)
    index_path = root / INDEX_FILENAME
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Regenerated {index_path}")

    commit_message = args.commit_message
    if not commit_message:
        commit_message = f"Update tuned lens artifacts ({datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')})"

    print(f"Uploading {root} → {args.repo_id}:{args.path_in_repo}")
    upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=str(root),
        path_in_repo=args.path_in_repo,
        commit_message=commit_message,
    )

    print("Done.")


if __name__ == "__main__":
    main()
