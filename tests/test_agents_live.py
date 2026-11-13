"""Laya text contract and the Qwen client/validator (no GPU or network needed)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

from agents import qwen_agent
from agents.features_text import FEATURE_COLUMNS, LAYA_QUESTION, laya_text

ROOT = Path(__file__).resolve().parents[1]


def test_laya_text_is_identical_to_the_training_renderer():
    sys.path.insert(0, str(ROOT / "finetune" / "fraud_v2"))
    try:
        from fraud_common import QUESTION, render_state
    finally:
        sys.path.pop(0)
    rows = pd.read_csv(ROOT / "sim_swap_fraud_dataset_100000.csv").head(300)
    for _, row in rows.iterrows():
        features = {c: row[c] for c in FEATURE_COLUMNS}
        assert laya_text(features) == render_state(row)
    assert LAYA_QUESTION == QUESTION


def test_laya_gate_uses_calibrated_cutoffs():
    from agents import laya_agent

    assert laya_agent.gate_for(0.99) == "block"
    assert laya_agent.gate_for(0.01) == "allow"
    assert laya_agent.gate_for(0.5) == "human_review"


PAYLOAD = {
    "case_id": "c1",
    "verdict": "flagged",
    "parameters": {
        "final_risk_score": 91.2,
        "laya": {"p_fraud": 0.9731},
        "event_features": {"sim_swap_time_gap_minutes": 45, "account_age_days": 120},
        "evidence_indicators": ["recent SIM activation (3 days ago)"],
    },
}


def test_validator_accepts_grounded_report():
    report = (
        "Summary: The case is flagged with a risk score of 91.2 and Laya p_fraud 97.3%. "
        "The SIM changed 45 minutes after the OTP and the account is 120 days old.\n"
        "Suggested review steps: verify the subscriber through a registered channel."
    )
    assert qwen_agent.validate_report(report, PAYLOAD) == []


def test_validator_rejects_invented_numbers_and_overclaims():
    invented = "Summary: risk score 91.2 and 17 failed logins from 3 countries were seen in 6000 ms."
    assert any("numbers not present" in p for p in qwen_agent.validate_report(invented, PAYLOAD))
    overclaim = "Summary: This is confirmed fraud. We have blocked the SIM after the 45 minute gap."
    assert any("overclaims" in p for p in qwen_agent.validate_report(overclaim, PAYLOAD))


def test_sse_parsing():
    stream = 'event: generating\ndata: null\n\nevent: complete\ndata: [{"report": "ok"}]\n\n'
    assert qwen_agent._parse_sse(stream) == [{"report": "ok"}]
    with pytest.raises(qwen_agent.QwenError):
        qwen_agent._parse_sse('event: error\ndata: "verdict must be flagged"\n')


def test_unconfigured_service_raises(monkeypatch):
    monkeypatch.setenv("QWEN_ENDPOINT", "")
    with pytest.raises(qwen_agent.QwenError):
        qwen_agent.call_service({"verdict": "none"})
    assert qwen_agent.ping()["configured"] is False
