"""Export the winning checkpoint to ONNX (the deployable, PyTorch-free format Laya supports),
check the ONNX graph agrees with PyTorch on every hard-test case, and time both.

Usage: python export_final.py [--final runs/<run_id>/final]
Writes <final>/laya_fraud.onnx (+ .onnx.data) and <final>/export_report.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fraud_common import QUESTION, QUESTION_ID  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    final = a.final or json.load(open(os.path.join(HERE, "runs", "winner.json")))["final_dir"]
    onnx_path = os.path.join(final, "laya_fraud.onnx")
    subprocess.run([sys.executable, os.path.join(HERE, "tools", "export_onnx.py"),
                    "--model", final, "--output", onnx_path], check=True)

    import laya
    from laya.onnx_agent import ONNXAgent
    q = {QUESTION_ID: dict(QUESTION)}
    states = [json.loads(l)["state"] for l in open(os.path.join(HERE, "data", "hard_test.jsonl"), encoding="utf-8")]
    pt = laya.load(final, device=a.device)
    ox = ONNXAgent(final, onnx_path=onnx_path)

    def run(agent):
        t = time.perf_counter()
        p = [agent.predict(s, q)["answers"][QUESTION_ID]["noul"] for s in states]
        return np.array(p), (time.perf_counter() - t) * 1000 / len(states)

    p_pt, ms_pt = run(pt)
    p_ox, ms_ox = run(ox)
    rep = {"onnx": onnx_path, "cases": len(states),
           "decision_agreement": float(((p_pt > .5) == (p_ox > .5)).mean()),
           "max_abs_prob_diff": float(np.abs(p_pt - p_ox).max()),
           "ms_per_case_pytorch_" + a.device: ms_pt, "ms_per_case_onnx_cpu": ms_ox}
    json.dump(rep, open(os.path.join(final, "export_report.json"), "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
