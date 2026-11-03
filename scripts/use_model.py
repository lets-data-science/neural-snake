"""Play a model you exported in the browser pages, or put Dev Suthar's shipped model back.

  python scripts/use_model.py runs/my-snake.onnx    # your model
  python scripts/use_model.py --shipped              # the model the project ships with (evidence/shipped-model/)

The pages always load web/model/snake.onnx and check it against web/model/selftest.json before playing, so both are
replaced together: the self-test cases are regenerated for the new model (scripts/make_selftest.py). The files you
replace are saved in .backups/ first.
"""
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL = os.path.join(ROOT, "web", "model")
SHIPPED = os.path.join(ROOT, "evidence", "shipped-model")
FILES = ("snake.onnx", "selftest.json", "export_info.json")


def backup():
    dest = os.path.join(ROOT, ".backups", "model-" + time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(dest, exist_ok=True)
    for name in FILES:
        if os.path.exists(os.path.join(MODEL, name)):
            shutil.copyfile(os.path.join(MODEL, name), os.path.join(dest, name))
    return os.path.relpath(dest, ROOT)


def main(argv):
    if argv == ["--shipped"]:
        saved = backup()
        for name in FILES:
            shutil.copyfile(os.path.join(SHIPPED, name), os.path.join(MODEL, name))
        print(f"The pages now use the shipped model again (previous files saved in {saved}).")
        return 0
    if len(argv) != 1 or not argv[0].endswith(".onnx") or not os.path.exists(argv[0]):
        print(__doc__)
        return 2
    source = os.path.abspath(argv[0])
    saved = backup()
    shutil.copyfile(source, os.path.join(MODEL, "snake.onnx"))
    info = source[:-5] + ".json"
    if os.path.exists(info):
        shutil.copyfile(info, os.path.join(MODEL, "export_info.json"))
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_selftest.py")], check=True)
    print(f"The pages now use {os.path.relpath(source, ROOT)} (previous files saved in {saved}). Reload the page to play it.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
