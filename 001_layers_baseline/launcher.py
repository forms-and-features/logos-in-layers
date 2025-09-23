#!/usr/bin/env python3
"""Launcher CLI for 001_layers_baseline.

Moves CLI parsing and multi-model orchestration out of run.py to keep the
worker slim. Backwards compatible: run.py sets default CLI_ARGS; this launcher
overrides it before invoking worker entrypoints.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime

# Support both package and script execution contexts
try:
    from . import run as worker
    from .layers_core.run_dir import setup_run_latest_directory
    from .layers_core.device_policy import select_best_device
except Exception:
    import run as worker  # type: ignore
    from layers_core.run_dir import setup_run_latest_directory  # type: ignore
    from layers_core.device_policy import select_best_device  # type: ignore


def parse_cli():
    p = argparse.ArgumentParser(description="Layer-by-layer logit-lens sweep")
    p.add_argument("--device",
                   default="auto",
                   choices=["auto", "cuda", "mps", "cpu"],
                   help="compute device to run on (default: auto picks best fit)")
    p.add_argument("--fp32-unembed",
                   action="store_true",
                   help="Use FP32 shadow unembedding for analysis-only decoding (do not mutate model params)")
    p.add_argument("--keep-residuals",
                   action="store_true",
                   help="Dump full residual tensors; if absent, keep only per-layer logits")
    p.add_argument("--copy-threshold", type=float, default=0.95,
                   help="Minimum P(top-1) for copy collapse")
    p.add_argument("--copy-margin", type=float, default=0.10,
                   help="Require P(top-1) − P(top-2) > margin for copy collapse")
    p.add_argument("--copy-soft-thresh", type=float, default=0.50,
                   help="Soft copy detector probability threshold (default: 0.50)")
    p.add_argument("--copy-soft-window-ks", default="1,2,3",
                   help="Comma-separated window sizes for soft copy detector (default: 1,2,3)")
    p.add_argument("--copy-soft-thresh-list", default=None,
                   help="Optional comma-separated list of additional soft thresholds to log")
    p.add_argument("model_id", nargs="?", default=None,
                   help="Model ID for single-run (when invoking as subprocess)")
    p.add_argument("--out_dir",
                   default=None,
                   help="Output directory to save CSV & JSON results (default: current script directory or value forwarded by parent launcher)")
    p.add_argument("--self-test",
                   action="store_true",
                   help="Run KL sanity test to validate normalization scaling (PROJECT_NOTES.md section 1.1). Can also run standalone: python kl_sanity_test.py MODEL_ID")
    # Prism sidecar controls
    p.add_argument("--prism",
                   default=os.environ.get("LOGOS_PRISM", "auto"),
                   choices=["auto", "on", "off"],
                   help="Prism sidecar mode: auto (default), on (require artifacts), off (disable)")
    p.add_argument("--prism-dir",
                   default="prisms",
                   help="Prism artifacts root directory (default: prisms under this script directory)")
    return p.parse_args()


def main():
    args = parse_cli()
    # Normalize CSV-like CLI inputs to typed lists before touching worker defaults
    if isinstance(args.copy_soft_window_ks, str):
        raw_parts = [part.strip() for part in args.copy_soft_window_ks.split(',') if part.strip()]
        args.copy_soft_window_ks = [int(part) for part in raw_parts] if raw_parts else [1, 2, 3]
    if args.copy_soft_thresh_list:
        raw_thresh = [part.strip() for part in args.copy_soft_thresh_list.split(',') if part.strip()]
        args.copy_soft_thresh_list = [float(part) for part in raw_thresh]
    else:
        args.copy_soft_thresh_list = []
    # Override worker defaults with CLI selections
    for k, v in vars(args).items():
        setattr(worker.CLI_ARGS, k.replace('-', '_'), v)

    # Single-model mode
    if args.model_id:
        ok = worker.run_single_model(args.model_id)
        sys.exit(0 if ok else 1)

    # Multi-model launcher
    print(f"🎯 Starting experiment launcher for {len(worker.CONFIRMED_MODELS)} models...")
    print("Each model will run in a separate process for clean memory isolation.")

    script_path = os.path.abspath(worker.__file__)
    run_dir = setup_run_latest_directory(os.path.dirname(script_path))

    print(f"📝 Creating empty evaluation markdown files...")
    for model_id in worker.CONFIRMED_MODELS:
        clean_name = worker.clean_model_name(model_id)
        eval_md_path = os.path.join(run_dir, f"evaluation-{clean_name}.md")
        with open(eval_md_path, 'w', encoding='utf-8') as f:
            f.write(f"# Evaluation Report: {model_id}\n\n")
            f.write(f"*Run executed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        print(f"   📄 Created: evaluation-{clean_name}.md")

    results = []
    launched = []
    for i, model_id in enumerate(worker.CONFIRMED_MODELS, 1):
        print(f"\n{'='*80}\n📋 Launching process {i}/{len(worker.CONFIRMED_MODELS)}: {model_id}\n{'='*80}")
        try:
            if args.device == "auto":
                sel = select_best_device(model_id)
                if sel is None:
                    print(f"⛔ Skipping {model_id}: no device fits (estimates)")
                    results.append((model_id, "SKIPPED_NO_FIT"))
                    continue
                dev, dtype, debug = sel
                print(f"📐 Decision: device={dev} dtype={dtype} est_peak={debug.get('est_peak')} avail={debug.get('available')}")
                chosen = dev
            else:
                chosen = args.device

            cmd = [
                sys.executable,
                script_path,
                "--device", chosen,
                "--out_dir", run_dir,
            ]
            if args.fp32_unembed:
                cmd.append("--fp32-unembed")
            if args.keep_residuals:
                cmd.append("--keep-residuals")
            cmd.extend(["--copy-threshold", str(args.copy_threshold)])
            cmd.extend(["--copy-margin", str(args.copy_margin)])
            cmd.append(model_id)

            r = subprocess.run(cmd, capture_output=False, text=True, check=False)
            if r.returncode == 0:
                print(f"✅ Process {i} completed successfully")
                results.append((model_id, "SUCCESS"))
                launched.append(model_id)
            else:
                print(f"❌ Process {i} failed with return code {r.returncode}")
                results.append((model_id, "FAILED"))
        except Exception as e:
            print(f"❌ Failed to launch subprocess for {model_id}: {e}")
            results.append((model_id, f"LAUNCH_FAILED: {e}"))

    print(f"\n{'='*80}\n🎉 All model processes completed!\n📁 Output files saved in: {run_dir}")
    print("\n📊 Results Summary:")
    for model_id, status in results:
        clean_name = worker.clean_model_name(model_id)
        status_emoji = "✅" if status == "SUCCESS" else "❌"
        print(f"   {status_emoji} {clean_name}: {status}")
    print("\n📄 Expected output files:")
    for model_id in (launched if launched else worker.CONFIRMED_MODELS):
        clean_name = worker.clean_model_name(model_id)
        print(f"   {os.path.join(run_dir, f'output-{clean_name}.json')}")
        print(f"   {os.path.join(run_dir, f'output-{clean_name}-records.csv')}")
        print(f"   {os.path.join(run_dir, f'output-{clean_name}-pure-next-token.csv')} ")
        print(f"   {os.path.join(run_dir, f'evaluation-{clean_name}.md')}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
