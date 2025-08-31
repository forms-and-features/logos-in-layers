import os
from datetime import datetime


def setup_run_latest_directory(script_dir: str, now_fn=datetime.now) -> str:
    """Set up the run-latest directory with rotation of previous runs.

    - If run-latest doesn't exist, create it.
    - If it exists, rename it to run-YYYYMMDD-HHMM based on its timestamp file if present,
      otherwise use the current timestamp with `-rotated` suffix.
    - Create a new run-latest directory and a timestamp file inside it.

    Returns: path to the (new) run-latest directory.
    """
    run_latest_dir = os.path.join(script_dir, "run-latest")
    current_timestamp = now_fn().strftime("%Y%m%d-%H%M")

    if os.path.exists(run_latest_dir):
        timestamp_files = [f for f in os.listdir(run_latest_dir) if f.startswith("timestamp-")]
        if timestamp_files:
            timestamp_file = timestamp_files[0]
            old_timestamp = timestamp_file.replace("timestamp-", "")
            rotated_name = f"run-{old_timestamp}"
        else:
            rotated_name = f"run-{current_timestamp}-rotated"
        rotated_dir = os.path.join(script_dir, rotated_name)
        os.rename(run_latest_dir, rotated_dir)

    os.makedirs(run_latest_dir, exist_ok=True)
    timestamp_file = os.path.join(run_latest_dir, f"timestamp-{current_timestamp}")
    with open(timestamp_file, 'w', encoding='utf-8') as f:
        f.write(f"Experiment started: {now_fn().strftime('%Y-%m-%d %H:%M:%S')}\n")

    return run_latest_dir

