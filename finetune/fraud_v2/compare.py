"""Print baseline vs fine-tuned for the winning run (or a run dir you pass).

Usage: python compare.py [runs/<run_id>]
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) > 1:
    run = sys.argv[1]
else:
    run = os.path.dirname(json.load(open(os.path.join(HERE, "runs", "winner.json")))["final_dir"])
rep = json.load(open(os.path.join(run, "final_report.json")))

rows = []
for s in ("val", "test", "hard_dev", "hard_test"):
    for k in ("accuracy", "balanced_accuracy", "log_loss", "recall_fraud", "recall_legit"):
        rows.append({"split": s, "metric": k, "base": rep["baseline"][s][k], "fine_tuned": rep["after"][s][k],
                     "delta": rep["after"][s][k] - rep["baseline"][s][k]})
print(pd.DataFrame(rows).round(4).to_string(index=False))
print("\nhard_test by category (base -> fine-tuned):")
for c, v in rep["after"]["hard_test"]["by_category"].items():
    print("  %-20s %.3f -> %.3f  (n=%d)" % (c, rep["baseline"]["hard_test"]["by_category"][c]["accuracy"], v["accuracy"], v["n"]))
print("\nDecision gate (BLOCK / ALLOW / NOUL -> human):  ALLOW if p<=%.5f, BLOCK if p>=%.5f" % (rep["gate"]["lo"], rep["gate"]["hi"]))
print("  bias:", rep["gate"]["bias"])
g = []
for s in ("val", "test", "hard_dev", "hard_test"):
    for who, src in (("base", rep["baseline"][s]["gate"]), ("fine_tuned", rep["after"][s]["gate"])):
        g.append({"split": s, "model": who, "block": src["block"], "allow": src["allow"], "noul->human": src["noul"],
                  "fraud_leaked": src["fraud_leak"], "false_block": src["false_block"],
                  "noul_rate": round(src["noul_rate"], 4)})
print(pd.DataFrame(g).to_string(index=False))
print("\nselected epoch %s | stop: %s | bias: %s" % (rep["selected_epoch"], rep["stop_reason"], rep["bias"]))
print("speed:", json.dumps(rep["speed"], indent=1))
p = pd.read_csv(os.path.join(run, "hard_test_predictions.csv"))
bad = p[p.correct == 0]
print("\nhard_test failures: %d / %d" % (len(bad), len(p)))
if len(bad):
    print(bad.sort_values("margin", key=abs).to_string(index=False, max_rows=60))
