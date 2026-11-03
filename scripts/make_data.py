"""Generate both training datasets with the engine: data/spike_a (50,850 steps) and data/run1 (669,784 steps).

  python scripts/make_data.py

The episode seeds decide the split before any data exists: seed % 5 == 0 is held out (never used for training or
choosing a model), seed % 5 == 1 is validation, the rest is training. The same seeds give the same data every time.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "training"))
import gen  # noqa: E402
import gen_run1  # noqa: E402

if __name__ == "__main__":
    gen.main()       # data/spike_a: the first, smaller dataset (Phase 0)
    gen_run1.main()  # data/run1: the larger dataset with new seeds (Run 1 and Run 2)
