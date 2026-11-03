"""PyTorch vs onnxruntime parity: models/run2_best.pt against a fresh ONNX export of that same checkpoint.
LDS edition: the export is made here, in a temporary folder, so the test checks your model whichever model web/model/ holds.
Skips if the git-ignored checkpoint or data is missing. Validation rows only (seed % 5 == 1); held-out rows are never loaded."""
import os
import subprocess
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
needed = [os.path.join(ROOT, p) for p in ("models/run2_best.pt", "web/model/start_state.json", "data/spike_a/act.npy")]
pytestmark = pytest.mark.skipif(not all(os.path.exists(p) for p in needed), reason="checkpoint or data missing (train a model first)")


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    import onnxruntime as ort
    import export_onnx
    import parity_onnx
    onnx_path = str(tmp_path_factory.mktemp("export") / "parity.onnx")
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "export_onnx.py"), "--out", onnx_path], check=True, capture_output=True)
    return parity_onnx, export_onnx.load_net(), ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


def test_validation_rows_identical_grid_and_decisions(setup):
    p, net, sess = setup
    r = p.validation_parity(net, sess, 1000)
    assert r["rows"] == 1000
    assert r["grid_mismatches"] == 0 and r["ate_decision_mismatches"] == 0 and r["over_decision_mismatches"] == 0
    assert max(r["max_abs_prob_diff"].values()) < 1e-3


def test_closed_loop_rollouts_identical_states(setup):
    p, net, sess = setup
    r = p.closed_loop(net, sess, 20, 200)
    assert r["rollouts"] == 20 and r["steps_each"] == 200
    assert r["grid_mismatches"] == 0 and r["state_divergences"] == 0
    assert r["ate_decision_mismatches"] == 0 and r["over_decision_mismatches"] == 0
