"""Restore one completed exercise from checkpoints/ into learner/, after backing up your version to .backups/.

  python scripts/restore.py encode      # learner/encode.py
  python scripts/restore.py food        # learner/food.py
  python scripts/restore.py survival    # learner/survival.py
"""
import os
import shutil
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXERCISES = ("encode", "food", "survival")


def main(argv):
    if len(argv) != 1 or argv[0] not in EXERCISES:
        print("Usage: python scripts/restore.py " + "|".join(EXERCISES))
        return 2
    name = argv[0]
    target = os.path.join(ROOT, "learner", name + ".py")
    os.makedirs(os.path.join(ROOT, ".backups"), exist_ok=True)
    if os.path.exists(target):
        backup = os.path.join(ROOT, ".backups", f"{name}-{time.strftime('%Y%m%d-%H%M%S')}.py")
        shutil.copyfile(target, backup)
        print(f"Your version is saved in {os.path.relpath(backup, ROOT)}")
    shutil.copyfile(os.path.join(ROOT, "checkpoints", name + ".py"), target)
    print(f"Restored learner/{name}.py from checkpoints/{name}.py. Read it, then run: python -m pytest tests/test_learner_{name}.py -q")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
