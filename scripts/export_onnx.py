"""Export models/run2_best.pt to an ONNX file (single file, weights embedded). Default runs/my-snake.onnx; Dev Suthar's shipped
model is web/model/snake.onnx (python scripts/export_onnx.py --out web/model/snake.onnx reproduces that step).

Graph inputs : grid   int32 [1,12,12]  14-class grid (0 empty, 1 food, 2-5 head, 6-9 body, 10-13 tail; learner/encode.py)
               action int32 [1]        0 UP, 1 DOWN, 2 LEFT, 3 RIGHT, 4 NOOP
Graph outputs: next_grid int64 [1,12,12] argmax class per cell (BigInt64Array in JS; the pages convert with Int32Array.from(data, Number))
               food      float32 [1,144] softmax over cells (used only when ate > 0.5)
               ate       float32 [1]     sigmoid(ate logit - ate_shift)   (prior-shift buffer included)
               over      float32 [1]     sigmoid(over logit - over_shift) (prior-shift buffer included)
The one-hot of grid and action is built inside the graph. Run: python scripts/export_onnx.py [--out PATH] -> PATH and PATH with .json (export info)
"""
import argparse
import collections
import json
import os
import sys

import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "training"))
import v3  # noqa: E402

CKPT = os.path.join(ROOT, "models", "run2_best.pt")
OUT = os.path.join(ROOT, "runs", "my-snake.onnx")
N = 12


def load_net():
    net = v3.Net3("dil", ate=True, food_detach=True)
    net.load_state_dict(torch.load(CKPT, map_location="cpu")["state_dict"])
    return net.eval()


class Wrapped(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("cls_ids", torch.arange(14, dtype=torch.int32).view(1, 14, 1, 1))
        self.register_buffer("act_ids", torch.arange(5, dtype=torch.int32).view(1, 5))

    def forward(self, grid, action):
        x_grid = (grid.unsqueeze(1) == self.cls_ids).float()  # [1,14,12,12]
        x_act = (action.unsqueeze(1) == self.act_ids).float().view(-1, 5, 1, 1).expand(-1, 5, N, N)
        out = self.net(torch.cat([x_grid, x_act], 1))
        # next_grid stays ArgMax's int64 output: a following Cast(int64->int32) ran on the CPU EP under WebGPU (Chrome placement log), so JS converts instead
        return (out["cls"].argmax(1), torch.softmax(out["food"], 1), torch.sigmoid(out["ate"]), torch.sigmoid(out["over"]))


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT, help="ONNX file to write (default runs/my-snake.onnx)")
    OUT = os.path.abspath(ap.parse_args().out)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    net = load_net()
    w = Wrapped(net).eval()
    grid = torch.zeros(1, N, N, dtype=torch.int32)
    action = torch.zeros(1, dtype=torch.int32)
    opset = int(os.environ.get("OPSET", "18"))
    torch.onnx.export(w, (grid, action), OUT, input_names=["grid", "action"], output_names=["next_grid", "food", "ate", "over"],
                      dynamo=True, opset_version=opset)
    m = onnx.load(OUT)
    for node in m.graph.node:  # LDS edition: drop the per-node Python stack traces, which record this computer's folder paths
        keep = [p for p in node.metadata_props if p.key != "pkg.torch.onnx.stack_trace"]
        del node.metadata_props[:]
        node.metadata_props.extend(keep)
    onnx.save(m, OUT, save_as_external_data=False)  # embed weights: single-file model for the browser
    if os.path.exists(OUT + ".data"):
        os.remove(OUT + ".data")
    m = onnx.load(OUT)
    onnx.checker.check_model(m)
    ops = collections.Counter(n.op_type for n in m.graph.node)
    info = {"command": "scripts/export_onnx.py", "checkpoint": "models/run2_best.pt (git-ignored)", "torch": torch.__version__,
            "onnx": onnx.__version__, "onnxruntime_py": ort.__version__, "opset": {o.domain or "ai.onnx": o.version for o in m.opset_import},
            "file_bytes": os.path.getsize(OUT), "params": sum(p.numel() for p in net.parameters()),
            "over_shift": float(net.over_shift), "ate_shift": float(net.ate_shift), "op_counts": dict(sorted(ops.items())),
            "inputs": [(i.name, i.type.tensor_type.elem_type, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input],
            "outputs": [(o.name, o.type.tensor_type.elem_type, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]}
    info_path = os.path.join(os.path.dirname(OUT), "export_info.json") if OUT.endswith(os.path.join("web", "model", "snake.onnx")) else OUT[:-5] + ".json"
    json.dump(info, open(info_path, "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
