"""The SIM-swap gate: turns Laya's calibrated P(fraud) into BLOCK / ALLOW / NOUL.

    p >= hi          -> BLOCK   (1)  stopped before it reaches the system or any employee
    p <= lo          -> ALLOW   (0)  proceeds automatically
    lo < p < hi      -> NOUL         "not sure": goes to the human review queue

lo / hi are NOT hand-picked and NOT trained as labels. They are fitted on a calibration slice
held out before training, under two error budgets:

    max_fraud_leak   share of real fraud allowed to be auto-ALLOWED   (strict)
    max_false_block  share of real legit allowed to be auto-BLOCKED   (looser)

Making the first budget stricter than the second is a deliberate bias toward catching fraud,
chosen by the company ("missing fraud is worse"). It is recorded in config.json, in
final_report.json and in the saved gate.json.

Use in SIMGuard:
    import json, laya
    from gate import route
    agent = laya.load("runs/<winner>/final")
    gate  = json.load(open("runs/<winner>/final/gate.json"))
    q     = json.load(open("runs/<winner>/final/questions.json"))
    p     = agent.predict(event_text, q)["answers"]["is_fraud"]["noul"]
    decision = route(p, gate)        # "BLOCK" | "ALLOW" | "NOUL"
"""
import numpy as np


def fit_gate(p, y, max_fraud_leak=0.002, max_false_block=0.01):
    """Fit (lo, hi) on held-out probabilities `p` (P fraud) and labels `y` (1 = fraud)."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    fraud_p = np.sort(p[y == 1])
    legit_p = np.sort(p[y == 0])
    n_f, n_l = len(fraud_p), len(legit_p)
    if n_f == 0 or n_l == 0:
        raise ValueError("need both classes in the calibration slice (fraud=%d legit=%d)" % (n_f, n_l))
    # ALLOW region is p <= lo: at most k_f fraud rows may sit there.
    k_f = int(np.floor(max_fraud_leak * n_f))
    lo = float(fraud_p[k_f]) - 1e-9 if k_f < n_f else 0.0          # just below the (k_f+1)-th lowest fraud
    # BLOCK region is p >= hi: at most k_l legit rows may sit there.
    k_l = int(np.floor(max_false_block * n_l))
    hi = float(legit_p[n_l - 1 - k_l]) + 1e-9 if k_l < n_l else 1.0  # just above the (k_l+1)-th highest legit
    lo = max(0.0, min(lo, 0.5))
    hi = min(1.0, max(hi, 0.5))
    if lo >= hi:   # budgets so loose the regions would overlap: collapse to 0.5, no NOUL band
        lo = hi = 0.5
    return {"lo": lo, "hi": hi, "max_fraud_leak": max_fraud_leak, "max_false_block": max_false_block,
            "fit_on": {"n_fraud": int(n_f), "n_legit": int(n_l)},
            "bias": "asymmetric budgets: fraud leak %.4f vs false block %.4f" % (max_fraud_leak, max_false_block)}


def route(p, gate):
    p = float(p)
    if p >= gate["hi"]:
        return "BLOCK"
    if p <= gate["lo"]:
        return "ALLOW"
    return "NOUL"


def gate_metrics(p, y, gate):
    """3-way outcome on labelled data. `confident_wrong` = real fraud ALLOWED or real legit BLOCKED."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    d = np.array([route(x, gate) for x in p])
    n = len(y)
    nf, nl = int((y == 1).sum()), int((y == 0).sum())
    fraud_leak = int(((d == "ALLOW") & (y == 1)).sum())
    false_block = int(((d == "BLOCK") & (y == 0)).sum())
    decided = d != "NOUL"
    return {
        "n": n, "block": int((d == "BLOCK").sum()), "allow": int((d == "ALLOW").sum()), "noul": int((d == "NOUL").sum()),
        "noul_rate": float((d == "NOUL").mean()),
        "fraud_leak": fraud_leak, "fraud_leak_rate": fraud_leak / max(1, nf),
        "false_block": false_block, "false_block_rate": false_block / max(1, nl),
        "confident_wrong": fraud_leak + false_block, "confident_wrong_rate": (fraud_leak + false_block) / max(1, n),
        "accuracy_on_decided": float((((d == "BLOCK").astype(int) == y)[decided]).mean()) if decided.any() else None,
        "fraud_sent_to_human": int(((d == "NOUL") & (y == 1)).sum()),
        "legit_sent_to_human": int(((d == "NOUL") & (y == 0)).sum()),
    }
