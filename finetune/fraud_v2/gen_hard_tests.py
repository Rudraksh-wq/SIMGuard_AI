"""Generate the hard / adversarial SIM-swap test sets as CSV (+ Laya JSONL).

Labels come from the dataset's own labelling rule (fraud_common.generator_score, which
reproduces all 200,000 public labels), so "correct" here means "what the dataset's generator
would say", not my opinion. Every case is inside the public data's value ranges.

Two files with different seeds and no shared rows:
  hard_dev.csv   (300)  - allowed for monitoring / early stopping during training
  hard_test.csv  (600)  - FINAL ONLY. Never used to pick a checkpoint or tune anything.

Categories (each roughly balanced between fraud / not-fraud where the rule allows):
  knife_edge_single   one identity change, |margin| <= 0.12 - decided by small geo/gap/extras differences
  knife_edge_double   two identity changes, |margin| <= 0.12
  two_changes_benign  two identity changes but same place + long gap -> NOT fraud (counter-intuitive)
  quiet_takeover      one identity change, no device/sim-type alarm, small extras tip it to fraud
  behaviour_storm     every behavioural red flag (new SIM, many swaps, IP change, gap 0) but no
                      identity change -> NOT fraud under the rule (trap)
  distractor_age      account age / tenure set opposite to intuition (these carry zero weight)
  flip_pair           counterfactual pairs: identical except one field (activation 6<->7 days,
                      swaps 1<->2, IP, geo +-0.03, gap +-~200 min) and the label flips
  far_but_flagged     obvious-looking cases a shortcut model gets wrong (e.g. 3+ identity
                      changes but every behavioural signal calm -> still fraud)

Usage: python gen_hard_tests.py --out data [--exclude data/train.csv data/val.csv data/test.csv]
"""
import argparse
import json
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fraud_common import FEATURES, LABEL, generator_score, to_row  # noqa: E402

IDENT = ["device_change_flag", "sim_type_change_flag", "imsi_change_flag", "iccid_change_flag"]


def base(rng):
    return {
        "sim_swap_time_gap_minutes": rng.randint(0, 10080),
        "device_change_flag": 0, "sim_type_change_flag": 0, "imsi_change_flag": 0, "iccid_change_flag": 0,
        "otp_and_sim_change_geo_hash_length": round(rng.randint(0, 100) / 100, 2),
        "recent_sim_activation_days": rng.randint(0, 365),
        "num_sim_changes_last_30d": rng.randint(0, 5),
        "previous_sim_holder_tenure_days": rng.randint(0, 1095),
        "account_age_days": rng.randint(1, 1825),
        "ip_change_flag": rng.randint(0, 1),
    }


def set_ident(r, rng, n):
    for k in IDENT:
        r[k] = 0
    for k in rng.sample(IDENT, n):
        r[k] = 1
    return r


def margin(r):
    return float(generator_score(r) - 3)


def label(r):
    return int(generator_score(r) > 3)


def gen_category(cat, rng, want_label):
    """Rejection-sample one case of `cat` with label `want_label` (None = whatever the rule allows)."""
    for _ in range(200000):
        r = base(rng)
        if cat == "knife_edge_single":
            set_ident(r, rng, 1)
            ok = 0.005 < abs(margin(r)) <= 0.12
        elif cat == "knife_edge_double":
            set_ident(r, rng, 2)
            ok = 0.005 < abs(margin(r)) <= 0.12
        elif cat == "two_changes_benign":
            set_ident(r, rng, 2)
            r["otp_and_sim_change_geo_hash_length"] = round(rng.randint(70, 100) / 100, 2)
            r["sim_swap_time_gap_minutes"] = rng.randint(5000, 10080)
            ok = margin(r) < -0.02
        elif cat == "quiet_takeover":
            set_ident(r, rng, 1)
            r["imsi_change_flag" if rng.random() < .5 else "iccid_change_flag"] = 1
            for k in ("device_change_flag", "sim_type_change_flag"):
                r[k] = 0
            if sum(r[k] for k in IDENT) != 1:
                continue
            ok = margin(r) > 0.02 and sum([r["recent_sim_activation_days"] < 7,
                                           r["num_sim_changes_last_30d"] >= 2, r["ip_change_flag"]]) == 1
        elif cat == "behaviour_storm":
            set_ident(r, rng, 0)
            r["recent_sim_activation_days"] = rng.randint(0, 6)
            r["num_sim_changes_last_30d"] = rng.randint(2, 5)
            r["ip_change_flag"] = 1
            r["sim_swap_time_gap_minutes"] = rng.randint(0, 60)
            # geo >= 0.02 keeps the case clearly below the threshold (no float-tie ambiguity)
            r["otp_and_sim_change_geo_hash_length"] = round(rng.randint(2, 10) / 100, 2)
            r["account_age_days"] = rng.randint(1, 30)
            r["previous_sim_holder_tenure_days"] = rng.randint(0, 20)
            ok = True
        elif cat == "distractor_age":
            set_ident(r, rng, rng.choice([1, 1, 2]))
            ok = abs(margin(r)) > 0.05
            if ok and label(r) == 1:   # fraud on a long-lived, loyal account
                r["account_age_days"] = rng.randint(1500, 1825)
                r["previous_sim_holder_tenure_days"] = rng.randint(900, 1095)
            elif ok:                   # legit on a brand-new account / new holder
                r["account_age_days"] = rng.randint(1, 14)
                r["previous_sim_holder_tenure_days"] = rng.randint(0, 14)
        elif cat == "far_but_flagged":
            if want_label == 1:        # many identity changes, everything else calm -> still fraud
                set_ident(r, rng, rng.choice([3, 4]))
                r["otp_and_sim_change_geo_hash_length"] = round(rng.randint(85, 100) / 100, 2)
                r["sim_swap_time_gap_minutes"] = rng.randint(8000, 10080)
                r["recent_sim_activation_days"] = rng.randint(100, 365)
                r["num_sim_changes_last_30d"] = 0
                r["ip_change_flag"] = 0
            else:                      # one change + two behavioural flags, still not enough
                set_ident(r, rng, 1)
                r["otp_and_sim_change_geo_hash_length"] = round(rng.randint(80, 100) / 100, 2)
                r["sim_swap_time_gap_minutes"] = rng.randint(4000, 10080)
            ok = abs(margin(r)) > 0.02
        else:
            raise ValueError(cat)
        if ok and (want_label is None or label(r) == want_label):
            return r
    raise RuntimeError("could not sample %s label=%s" % (cat, want_label))


def flip_pair(rng):
    """Two rows identical except one field, with opposite labels."""
    for _ in range(200000):
        r = set_ident(base(rng), rng, rng.choice([1, 2]))
        s = dict(r)
        kind = rng.choice(["activation", "swaps", "ip", "geo", "gap"])
        if kind == "activation":
            r["recent_sim_activation_days"], s["recent_sim_activation_days"] = 6, 7
        elif kind == "swaps":
            r["num_sim_changes_last_30d"], s["num_sim_changes_last_30d"] = 2, 1
        elif kind == "ip":
            r["ip_change_flag"], s["ip_change_flag"] = 1, 0
        elif kind == "geo":
            g = r["otp_and_sim_change_geo_hash_length"]
            s["otp_and_sim_change_geo_hash_length"] = round(min(1.0, g + 0.03), 2)
        else:
            d = rng.randint(100, 300)
            s["sim_swap_time_gap_minutes"] = min(10080, r["sim_swap_time_gap_minutes"] + d)
        if label(r) != label(s) and abs(margin(r)) > 0.005 and abs(margin(s)) > 0.005:
            return [(r, kind), (s, kind)]
    raise RuntimeError("flip pair")


PLAN = {  # counts per category for 600; dev uses half
    "knife_edge_single": 90, "knife_edge_double": 80, "two_changes_benign": 60, "quiet_takeover": 60,
    "behaviour_storm": 60, "distractor_age": 70, "far_but_flagged": 60, "flip_pair": 120,
}


def build(seed, scale, exclude):
    rng = random.Random(seed)
    rows, seen = [], set(exclude)

    def add(r, cat, note=""):
        key = tuple(r[f] for f in FEATURES)
        if key in seen:
            return False
        seen.add(key)
        rows.append(dict(r, category=cat, note=note, margin=round(margin(r), 4), **{LABEL: label(r)}))
        return True

    for cat, n in PLAN.items():
        n = int(round(n * scale))
        if cat == "flip_pair":
            pid = 0
            while sum(1 for x in rows if x["category"] == cat) < n:
                pair = flip_pair(rng)
                keys = [tuple(r[f] for f in FEATURES) for r, _ in pair]
                if any(k in seen for k in keys):
                    continue
                for r, kind in pair:
                    add(r, cat, "pair%03d:%s" % (pid, kind))
                pid += 1
            continue
        allowed = [0] if cat in ("two_changes_benign", "behaviour_storm") else \
                  [1] if cat == "quiet_takeover" else [0, 1]
        i = 0
        while sum(1 for x in rows if x["category"] == cat) < n:
            add(gen_category(cat, rng, allowed[i % len(allowed)]), cat)
            i += 1
    df = pd.DataFrame(rows)
    df.insert(0, "case_id", ["H%05d_%d" % (i, seed) for i in range(len(df))])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--exclude", nargs="*", default=[])
    args = ap.parse_args()
    exclude = set()
    for p in args.exclude:
        d = pd.read_csv(p)
        exclude |= set(map(tuple, d[FEATURES].values.tolist()))
    test = build(seed=20261113, scale=1.0, exclude=exclude)
    dev = build(seed=777, scale=0.5, exclude=exclude | set(map(tuple, test[FEATURES].values.tolist())))
    for name, df in (("hard_test", test), ("hard_dev", dev)):
        df.to_csv(os.path.join(args.out, name + ".csv"), index=False)
        with open(os.path.join(args.out, name + ".jsonl"), "w", encoding="utf-8") as f:
            for _, r in df.iterrows():
                row = to_row(r, r[LABEL])
                row["meta"] = {"case_id": r["case_id"], "category": r["category"], "margin": r["margin"]}
                f.write(json.dumps(row) + "\n")
        summary = df.groupby("category")[LABEL].agg(["count", "mean"]).round(3)
        print("%s: %d rows, fraud rate %.3f\n%s\n" % (name, len(df), df[LABEL].mean(), summary))


if __name__ == "__main__":
    main()
