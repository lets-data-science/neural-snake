"""Train the world model (Dev Suthar's Run 2) for at most --minutes, then evaluate the best checkpoint.

  python scripts/train.py --minutes 30     # Run 2 as shipped: up to 30 minutes, early stop after 10 without improvement
  python scripts/train.py --minutes 10     # a shorter run

Writes models/run2_best.pt (the best checkpoint, chosen every 2.5 minutes on 5,000 validation rows) and
runs/results_run2.json / .md (one-step and rollout results on 10,374 validation rows). Held-out rows are never loaded.
The device is chosen automatically: an NVIDIA GPU, an Apple GPU, or the CPU (set SNAKE_DEVICE to override).
"""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "training"))
import run1  # noqa: E402
import run2  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30.0)
    minutes = ap.parse_args().minutes
    print(f"Training on {run1.DEV} for at most {minutes:g} minutes.", flush=True)
    run2.main_train(minutes * 60)
