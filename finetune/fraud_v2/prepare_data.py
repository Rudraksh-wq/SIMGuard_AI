"""Build Laya train / val / test JSONL from the public SIM-swap dataset.

  train + val : sim_swap_fraud_dataset_100000.csv   (disjoint random split)
  test        : sim_swap_fraud_dataset_50000.csv    (a different file; any row whose features
                also appear in the train/val file is dropped, so the test set is unseen)

Class balance is left exactly as the data has it (about 74% fraud). No re-weighting, no
re-sampling: that is the "no bias unless needed" default, and it is logged in data_report.json.

Usage:
  python prepare_data.py --a PATH_100000.csv --b PATH_50000.csv --out data \
      [--train 24000 --val 3000 --test 6000 --seed 13]
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fraud_common import FEATURES, LABEL, generator_label, to_row  # noqa: E402


def write_jsonl(df, path):
    with open(path, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            f.write(json.dumps(to_row(r, r[LABEL])) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--train", type=int, default=24000)
    ap.add_argument("--val", type=int, default=3000)
    ap.add_argument("--test", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    a = pd.read_csv(args.a).drop_duplicates()
    b = pd.read_csv(args.b).drop_duplicates()
    for name, df in (("A", a), ("B", b)):
        missing = set(FEATURES + [LABEL]) - set(df.columns)
        if missing:
            raise SystemExit("file %s is missing columns %s" % (name, sorted(missing)))

    # Audit only: does the recovered rule reproduce the labels? (not used for training)
    audit = {n: float((df.apply(generator_label, axis=1) == df[LABEL]).mean()) for n, df in (("A", a), ("B", b))}

    a = a.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    train = a.iloc[:args.train]
    val = a.iloc[args.train:args.train + args.val]
    seen = set(map(tuple, a[FEATURES].values.tolist()))
    b_unseen = b[[tuple(x) not in seen for x in b[FEATURES].values.tolist()]]
    test = b_unseen.sample(n=min(args.test, len(b_unseen)), random_state=args.seed)

    for name, df in (("train", train), ("val", val), ("test", test)):
        write_jsonl(df, os.path.join(args.out, name + ".jsonl"))
        df.to_csv(os.path.join(args.out, name + ".csv"), index=False)

    report = {
        "source_A": os.path.abspath(args.a), "source_B": os.path.abspath(args.b),
        "rows_A_dedup": len(a), "rows_B_dedup": len(b), "rows_B_unseen_in_A": len(b_unseen),
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "fraud_rate": {"train": float(train[LABEL].mean()), "val": float(val[LABEL].mean()),
                       "test": float(test[LABEL].mean())},
        "class_balancing": "none (natural prior kept)",
        "recovered_rule_label_agreement_audit_only": audit,
        "seed": args.seed,
    }
    with open(os.path.join(args.out, "data_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
