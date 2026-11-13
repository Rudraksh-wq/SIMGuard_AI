import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agents.schemas import Event, LayaResult, MLResult
from api import db, main, pipeline


def make_event() -> Event:
    return Event(
        event_id="event-1",
        msisdn_hash="hashed-subscriber-id",
        timestamp="2026-11-13T12:00:00Z",
        sim_swap_time_gap_minutes=45,
        device_change_flag=1,
        sim_type_change_flag=0,
        imsi_change_flag=1,
        iccid_change_flag=1,
        otp_and_sim_change_geo_hash_length=0.12,
        recent_sim_activation_days=3,
        num_sim_changes_last_30d=2,
        previous_sim_holder_tenure_days=40,
        account_age_days=120,
        ip_change_flag=1,
    )


def install_pipeline_stubs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    ml_score: float = 30.0,
    p_laya: float | None = 0.95,
) -> list[dict[str, Any]]:
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("MOCK_MODELS", "0")
    monkeypatch.setenv("LAYA_FRAUD_THRESHOLD", "0.5")
    monkeypatch.setenv("LAYA_WEIGHT", "0.7")
    db.init_db()

    monkeypatch.setattr(
        pipeline,
        "score_events",
        lambda events: [
            MLResult(
                risk_score=ml_score,
                tier="low" if ml_score < 40 else "high",
                lgbm_prob=ml_score / 100,
                anomaly=0.1,
                contributions={},
                latency_ms=1.0,
            )
            for _ in events
        ],
    )
    monkeypatch.setattr(
        pipeline,
        "decide_events",
        lambda events: [
            LayaResult(p_fraud=p_laya, model="test-model", latency_ms=1.0) for _ in events
        ],
    )
    monkeypatch.setattr(pipeline, "extract_evidence", lambda event: ["test indicator"])
    captured_payloads = []

    def report(payload: dict[str, Any]) -> tuple[str, str]:
        captured_payloads.append(payload)
        return "A test report for the human analyst.", "qwen"

    monkeypatch.setattr(pipeline, "generate_report", report)
    return captured_payloads


def test_fraud_and_normal_report_behavior(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_pipeline_stubs(monkeypatch, tmp_path)
    fraud = pipeline.analyze(make_event())
    assert fraud.report is not None
    assert fraud.report_source in {"qwen", "template"}

    normal_dir = tmp_path / "normal"
    normal_dir.mkdir()
    install_pipeline_stubs(monkeypatch, normal_dir, ml_score=20, p_laya=0.05)
    normal = pipeline.analyze(make_event())
    assert normal.report is None
    assert normal.report_source == "none"
    assert normal.report_latency_ms is None


def test_qwen_failure_uses_template(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_pipeline_stubs(monkeypatch, tmp_path)

    def fail(payload: dict[str, Any]) -> None:
        raise TimeoutError("report timed out")

    monkeypatch.setattr(pipeline, "generate_report", fail)
    result = pipeline.analyze(make_event())

    assert result.report is not None
    assert result.report_source == "template"
    assert result.report_latency_ms is not None


def test_laya_unavailable_uses_ml_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_pipeline_stubs(monkeypatch, tmp_path, ml_score=42, p_laya=None)
    result = pipeline.analyze(make_event())

    assert result.agreement == "laya_unavailable"
    assert result.final.score == result.ml.risk_score
    assert result.final.laya_weight_used == 0.0


def test_qwen_payload_uses_final_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payloads = install_pipeline_stubs(monkeypatch, tmp_path, ml_score=30, p_laya=0.9)
    result = pipeline.analyze(make_event())

    assert payloads[0]["parameters"]["final_risk_score"] == result.final.score
    assert payloads[0]["verdict"] == "flagged"
    assert payloads[0]["parameters"]["event_features"]["imsi_change_flag"] == 1


def test_env_threshold_and_weight_affect_final_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_pipeline_stubs(monkeypatch, tmp_path, ml_score=30, p_laya=0.4)
    monkeypatch.setenv("LAYA_FRAUD_THRESHOLD", "0.3")
    high_threshold = pipeline.analyze(make_event())
    assert high_threshold.fraud_detected

    monkeypatch.setenv("LAYA_FRAUD_THRESHOLD", "0.7")
    low_threshold = pipeline.analyze(make_event())
    assert not low_threshold.fraud_detected
    assert high_threshold.final.score != low_threshold.final.score

    monkeypatch.setenv("LAYA_WEIGHT", "0.2")
    lower_weight = pipeline.analyze(make_event())
    monkeypatch.setenv("LAYA_WEIGHT", "0.9")
    higher_weight = pipeline.analyze(make_event())
    assert lower_weight.final.score != higher_weight.final.score


def test_every_final_score_is_bounded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_pipeline_stubs(monkeypatch, tmp_path, ml_score=100, p_laya=1)
    response = pipeline.analyze(make_event())
    assert 0 <= response.final.score <= 100


def test_mock_mode_runs_without_live_model_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DB_PATH", str(tmp_path / "mock.db"))
    monkeypatch.setenv("MOCK_MODELS", "1")
    monkeypatch.delenv("LAYA_FRAUD_THRESHOLD", raising=False)
    monkeypatch.delenv("LAYA_WEIGHT", raising=False)

    with TestClient(main.app) as client:
        fraud = client.post("/analyze", json=make_event().model_dump())
        assert fraud.status_code == 200
        assert fraud.json()["ml"]["risk_score"] == 90
        assert fraud.json()["evidence"] == []
        assert fraud.json()["report_source"] == "template"

        normal_event = make_event().model_copy(
            update={
                "device_change_flag": 0,
                "imsi_change_flag": 0,
                "otp_and_sim_change_geo_hash_length": 0.9,
            }
        )
        normal = client.post("/analyze", json=normal_event.model_dump())
        assert normal.status_code == 200
        assert normal.json()["ml"]["risk_score"] == 20
        assert normal.json()["report"] is None


def test_api_analyze_list_review_and_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_pipeline_stubs(monkeypatch, tmp_path)
    with TestClient(main.app) as client:
        analyzed = client.post("/analyze", json=make_event().model_dump())
        assert analyzed.status_code == 200
        case = analyzed.json()
        case_id = case["case_id"]

        monkeypatch.setattr(
            pipeline,
            "score_events",
            lambda events: [
                MLResult(
                    risk_score=90,
                    tier="high",
                    lgbm_prob=0.9,
                    anomaly=0.9,
                    contributions={},
                    latency_ms=1.0,
                )
                for _ in events
            ],
        )
        higher = client.post("/analyze", json=make_event().model_dump())
        assert higher.status_code == 200
        higher_case_id = higher.json()["case_id"]

        cases = client.get("/cases")
        assert cases.status_code == 200
        assert [item["case_id"] for item in cases.json()] == [
            higher_case_id,
            case_id,
        ]

        reviewed = client.post(
            f"/cases/{case_id}/review", json={"status": "confirmed"}
        )
        assert reviewed.status_code == 200
        assert reviewed.json()["status"] == "confirmed"

        bad_review = client.post(
            f"/cases/{case_id}/review", json={"status": "invalid"}
        )
        assert bad_review.status_code == 422
        assert client.get("/cases/unknown-case").status_code == 404

    with db._connect() as connection:
        actions = [
            row["action"]
            for row in connection.execute(
                "SELECT action FROM audit_log WHERE case_id = ?", (case_id,)
            )
        ]
    assert actions == ["analyze", "review"]


def test_input_endpoints_parse_text_csv_json_and_cap_qwen_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payloads = install_pipeline_stubs(monkeypatch, tmp_path, ml_score=90, p_laya=0.95)
    monkeypatch.setenv("QWEN_MAX_AUTO_REPORTS", "2")
    monkeypatch.setenv("QWEN_BACKGROUND", "0")
    features = {k: v for k, v in make_event().model_dump().items() if k not in {"event_id", "msisdn_hash", "timestamp"}}
    header = ",".join(features) + ",sim_swap_flag"
    row = ",".join(str(v) for v in features.values()) + ",1"
    csv_text = "\n".join([header, row, row, row, "1,2,3"])

    with TestClient(main.app) as client:
        out = client.post("/analyze/upload", files={"file": ("batch.csv", csv_text, "text/csv")})
        assert out.status_code == 200, out.text
        body = out.json()
        assert body["source"] == "csv"
        assert body["rows_analysed"] == 3 and len(body["errors"]) == 1
        sources = [r["report_source"] for r in body["results"]]
        assert sources.count("qwen") == 2 and sources.count("template") == 1
        assert len(payloads) == 2
        # the dataset label is kept as context in the UI but never sent to Qwen
        assert body["results"][0]["input"]["extra"] == {"sim_swap_flag": "1"}
        assert "input_context" not in payloads[0]["parameters"]

        over_budget = next(r for r in body["results"] if r["report_source"] == "template")
        regenerated = client.post(f"/cases/{over_budget['case_id']}/report")
        assert regenerated.status_code == 200
        assert regenerated.json()["report_source"] == "qwen"

        text = client.post("/analyze/input", json={"content": "device changed: yes", "kind": "text"})
        assert text.status_code == 422
        assert "missing fields" in text.text

        js = client.post("/analyze/input", json={"content": json.dumps({"events": [features]})})
        assert js.status_code == 200 and js.json()["source"] == "json"


def test_background_reports_fill_in_without_overwriting_review(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import threading

    payloads = install_pipeline_stubs(monkeypatch, tmp_path, ml_score=90, p_laya=0.95)
    gate = threading.Event()

    def slow_report(payload):
        gate.wait(5)
        payloads.append(payload)
        return "A test report for the human analyst.", "qwen"

    monkeypatch.setattr(pipeline, "generate_report", slow_report)
    result = pipeline.analyze_batch([(make_event(), None)], background=True)[0]
    assert result.report_source == "pending" and result.report  # template shown meanwhile
    db.update_status(result.case_id, "confirmed")  # analyst reviews before Qwen finishes
    gate.set()
    pipeline.wait_for_reports()
    stored = db.get_case(result.case_id)
    assert stored["report_source"] == "qwen"
    assert stored["status"] == "confirmed"
