"""Shared helpers for the SIM-swap fine-tune: how a row becomes Laya text, and the question.

Nothing in here is used to *train* the model on the labelling rule. `generator_score` exists
only to (a) audit that the recovered rule reproduces the public dataset's labels and (b) label
the synthetic hard-test cases the same way the dataset itself was labelled. Training targets
always come from the dataset's own `sim_swap_flag` column.
"""
from fractions import Fraction

FEATURES = [
    "sim_swap_time_gap_minutes", "device_change_flag", "sim_type_change_flag",
    "imsi_change_flag", "iccid_change_flag", "otp_and_sim_change_geo_hash_length",
    "recent_sim_activation_days", "num_sim_changes_last_30d",
    "previous_sim_holder_tenure_days", "account_age_days", "ip_change_flag",
]
LABEL = "sim_swap_flag"

QUESTION_ID = "is_fraud"
# One fixed wording for every row (train, val, test, hard tests). No hints about the rule.
QUESTION = {
    "type": "noul",
    "instructions": "Does this SIM swap event look like SIM-swap fraud?",
}


def _yn(v):
    return "yes" if int(v) == 1 else "no"


def render_state(r) -> str:
    """Plain, neutral rendering of the raw features. No risk words, no derived scores, no
    thresholds: the model has to learn what matters from the labels alone."""
    gap = int(r["sim_swap_time_gap_minutes"])
    return (
        "SIM swap event. "
        f"Time between OTP request and SIM change: {gap} minutes ({gap / 1440:.2f} days). "
        f"Device changed: {_yn(r['device_change_flag'])}. "
        f"SIM type changed: {_yn(r['sim_type_change_flag'])}. "
        f"IMSI changed: {_yn(r['imsi_change_flag'])}. "
        f"ICCID changed: {_yn(r['iccid_change_flag'])}. "
        f"Geohash similarity between OTP location and SIM change location: "
        f"{float(r['otp_and_sim_change_geo_hash_length']):.2f} (0 = different place, 1 = same place). "
        f"Days since SIM activation: {int(r['recent_sim_activation_days'])}. "
        f"SIM changes in last 30 days: {int(r['num_sim_changes_last_30d'])}. "
        f"Previous SIM holder tenure: {int(r['previous_sim_holder_tenure_days'])} days. "
        f"Account age: {int(r['account_age_days'])} days. "
        f"IP address changed: {_yn(r['ip_change_flag'])}."
    )


def to_row(r, label=None) -> dict:
    """A Laya `{state, questions, gold}` row."""
    row = {"state": render_state(r), "questions": {QUESTION_ID: dict(QUESTION)}}
    if label is not None:
        y = int(label)
        row["gold"] = {QUESTION_ID: {"probabilities": {"false": float(1 - y), "true": float(y)}}}
    return row


def generator_score(r) -> Fraction:
    """Recovered labelling rule of the public dataset (reproduces 200,000/200,000 labels):

        score = 3*(device + sim_type + imsi + iccid) - 2*geo - 2*gap/10080
                + [activation_days < 7] + [sim_changes_30d >= 2] + ip_change
        fraud  <=> score > 3

    Exact arithmetic so margins near the threshold are not float noise.
    AUDIT / HARD-TEST LABELLING ONLY - never fed to training.
    """
    ident = (int(r["device_change_flag"]) + int(r["sim_type_change_flag"])
             + int(r["imsi_change_flag"]) + int(r["iccid_change_flag"]))
    geo = Fraction(str(round(float(r["otp_and_sim_change_geo_hash_length"]), 2)))
    gap = Fraction(int(r["sim_swap_time_gap_minutes"]), 10080)
    extra = (int(int(r["recent_sim_activation_days"]) < 7)
             + int(int(r["num_sim_changes_last_30d"]) >= 2)
             + int(r["ip_change_flag"]))
    return 3 * ident - 2 * geo - 2 * gap + extra


def generator_label(r) -> int:
    return int(generator_score(r) > 3)
