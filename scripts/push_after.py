"""Wait until one Kaggle kernel run has finished, then push (start) another notebook.

Only reads the status of the running kernel; it never modifies or cancels it.

Usage: python scripts/push_after.py hamzadotexe/football-train-detector notebooks/04_train_pitch
"""
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = str(ROOT / ".venv" / "Scripts" / "kaggle")
FINAL = ("COMPLETE", "ERROR", "CANCEL", "FAIL")
env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

wait_for, folder = sys.argv[1], str((ROOT / sys.argv[2]).resolve())
# Hard cap on the pushed run (Kaggle kills it after this many seconds), so a bug can never burn the whole session.
MAX_RUN_SECONDS = int(sys.argv[3]) if len(sys.argv) > 3 else 5 * 3600


def log(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}", flush=True)


log(f"waiting for {wait_for} to finish before pushing {folder}")
while True:
    out = subprocess.run([KAGGLE, "kernels", "status", wait_for], capture_output=True, text=True, env=env)
    status = (out.stdout + out.stderr).strip().splitlines()[-1] if (out.stdout + out.stderr).strip() else "?"
    if any(s in status for s in FINAL):
        log(f"{wait_for} finished: {status}")
        break
    if "RUNNING" not in status and "QUEUED" not in status:
        log(f"unexpected status (will keep waiting): {status}")
    time.sleep(300)

push = subprocess.run([KAGGLE, "kernels", "push", "-p", folder, "-t", str(MAX_RUN_SECONDS)],
                      capture_output=True, text=True, env=env)
log("push result: " + (push.stdout + push.stderr).strip())
