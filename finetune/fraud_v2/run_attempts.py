"""Run the attempt ladder, one train_fraud.py run per rung, until the targets are met.

Every rung is a separate run directory and a separate line in runs/attempts_ledger.jsonl, so
every attempt (including ones that fail or are worse) stays on record. The decision to move to
the next rung and the choice of the winner use ONLY val and hard_dev - never test or hard_test.

Ladder (no training-data bias at any rung; the only bias is the gate's fraud-leak budget):
  a1_no_bias          defaults: soft-CE, encoder lr 1e-5, bottom 14 layers frozen, 12-16 epochs
  a2_gentler_longer   encoder lr 7e-6, up to 20 epochs, patience 3
  a3_more_data        same as a2 on a 3x larger train split (data_large/)
On CUDA out-of-memory a rung is retried once with --freeze-bottom 20 --micro-batch 2 --grad-accum 16
(logged). The decision gate (BLOCK / ALLOW / NOUL->human) is fitted inside every run.

Usage:  python run_attempts.py [--base convaiinnovations/laya] [--device cuda]
"""
import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def last_report(runs, tag):
    found = sorted(glob.glob(os.path.join(runs, "*_" + tag, "final_report.json")))
    return json.load(open(found[-1])) if found else None


def oom_in_last_log(runs, tag):
    found = sorted(glob.glob(os.path.join(runs, "*_" + tag, "train.log")))
    return bool(found) and "OutOfMemoryError" in open(found[-1], encoding="utf-8", errors="ignore").read()


def meets(rep, a):
    v, h = rep["after"]["val"], rep["after"]["hard_dev"]
    return (v["accuracy"] >= a.target_val_acc and v["log_loss"] <= a.target_val_loss
            and h["accuracy"] >= a.target_hard_dev
            and v["gate"]["fraud_leak_rate"] <= a.target_fraud_leak
            and h["gate"]["confident_wrong_rate"] <= a.target_hard_dev_confident_wrong)


def score(rep):   # winner key: val log-loss, then hard_dev accuracy (no test data involved)
    return (rep["after"]["val"]["log_loss"], -rep["after"]["hard_dev"]["accuracy"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="convaiinnovations/laya")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--target-val-acc", type=float, default=0.995)
    ap.add_argument("--target-val-loss", type=float, default=0.02)
    ap.add_argument("--target-hard-dev", type=float, default=0.97)
    ap.add_argument("--target-fraud-leak", type=float, default=0.005, help="val: share of fraud auto-ALLOWED")
    ap.add_argument("--target-hard-dev-confident-wrong", type=float, default=0.02)
    ap.add_argument("--extra", default="", help="extra args passed to every train_fraud.py run (e.g. '--micro-batch 4 --grad-accum 8')")
    a = ap.parse_args()

    data, data_large = os.path.join(HERE, "data"), os.path.join(HERE, "data_large")
    common = [PY, os.path.join(HERE, "train_fraud.py"), "--base", a.base, "--device", a.device, "--runs", a.runs] + a.extra.split()
    ladder = [
        ("a1_no_bias", ["--data", data]),
        ("a2_gentler_longer", ["--data", data, "--encoder-lr", "7e-6", "--max-epochs", "20", "--patience", "3"]),
        ("a3_more_data", ["--data", data_large, "--encoder-lr", "7e-6", "--max-epochs", "20", "--patience", "3"]),
    ]
    reports = {}
    for tag, extra in ladder:
        if tag == "a3_more_data" and not os.path.exists(os.path.join(data_large, "train.jsonl")):
            src = os.path.join(HERE, "source")
            subprocess.run([PY, os.path.join(HERE, "prepare_data.py"),
                            "--a", os.path.join(src, "sim_swap_fraud_dataset_100000.csv"),
                            "--b", os.path.join(src, "sim_swap_fraud_dataset_50000.csv"),
                            "--out", data_large, "--train", "72000", "--val", "3000", "--test", "6000"], check=True)
            for f in ("hard_dev", "hard_test"):   # identical hard sets so attempts are comparable
                for ext in (".csv", ".jsonl"):
                    with open(os.path.join(data, f + ext), "rb") as s, open(os.path.join(data_large, f + ext), "wb") as d:
                        d.write(s.read())
        print("\n=== attempt %s ===" % tag, flush=True)
        r = subprocess.run(common + ["--tag", tag] + extra)
        if r.returncode != 0 and oom_in_last_log(a.runs, tag):
            print("CUDA out of memory in %s: retrying once with a smaller footprint" % tag, flush=True)
            r = subprocess.run(common + ["--tag", tag] + extra +
                               ["--freeze-bottom", "20", "--micro-batch", "2", "--grad-accum", "16"])
        rep = last_report(a.runs, tag) if r.returncode == 0 else None
        if rep is None:
            print("attempt %s failed (see its train.log); moving on" % tag, flush=True)
            continue
        reports[tag] = rep
        if meets(rep, a):
            print("targets met by %s" % tag, flush=True)
            break
    if not reports:
        raise SystemExit("no attempt finished")
    win_tag = min(reports, key=lambda t: score(reports[t]))
    win = reports[win_tag]
    summary = {"winner": win_tag, "final_dir": win["final_dir"], "selected_epoch": win["selected_epoch"],
               "selection_rule": "lowest val log-loss, tie -> higher hard_dev accuracy (no test data used)",
               "targets": vars(a), "targets_met": meets(win, a),
               "attempts": {t: {"val_acc": r["after"]["val"]["accuracy"], "val_loss": r["after"]["val"]["log_loss"],
                                "hard_dev_acc": r["after"]["hard_dev"]["accuracy"], "bias": r["bias"]}
                            for t, r in reports.items()}}
    with open(os.path.join(a.runs, "winner.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
