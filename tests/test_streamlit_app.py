"""Headless tests of the Streamlit analyst UI (app/app.py) with a fake API client."""

import json
import sys
import types
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = st_testing.AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "app.py"
FEATURES = {
    "sim_swap_time_gap_minutes": 45.0, "device_change_flag": 1, "sim_type_change_flag": 0,
    "imsi_change_flag": 1, "iccid_change_flag": 1, "otp_and_sim_change_geo_hash_length": 0.12,
    "recent_sim_activation_days": 3.0, "num_sim_changes_last_30d": 2.0,
    "previous_sim_holder_tenure_days": 40.0, "account_age_days": 120.0, "ip_change_flag": 1,
}


def make_case(case_id="case-1", source="qwen", status="pending"):
    return {
        "case_id": case_id,
        "event": {"event_id": "evt-1", "msisdn_hash": "SYNTH-ABC", "timestamp": "", **FEATURES},
        "ml": {"risk_score": 99.0, "tier": "high", "lgbm_prob": 0.999, "anomaly": 0.97,
               "contributions": {"device_change_flag": 8.1, "sim_type_change_flag": -8.9}, "latency_ms": 2.5},
        "laya": {"p_fraud": 0.998, "model": "20261113_171802_a1_6ep/final", "latency_ms": 40.0,
                 "gate": "block", "error": None},
        "final": {"score": 99.4, "tier": "high", "laya_z": 5.0, "ml_z": 3.0, "laya_weight_used": 0.7},
        "evidence": ["Device changed", "IMSI changed"],
        "fraud_detected": True, "verdict": "flagged", "agreement": "agree",
        "report": "**Summary** Suspected SIM-swap fraud for human review.",
        "report_source": source, "report_latency_ms": 15000.0, "report_model": "Qwen/Qwen3.5-2B",
        "report_payload": {"case_id": case_id, "verdict": "flagged", "parameters": {"final_risk_score": 99.4}},
        "report_error": None,
        "input": {"source": "text", "raw": "SIM changed 45 minutes after an OTP request.", "extra": {}, "notes": ["read sim_swap_time_gap_minutes = '45 minutes'"]},
        "status": status, "created_at": "2026-11-14T10:00:00Z",
    }


HEALTH = {
    "mock_models": False, "db_ok": True,
    "ml": {"ok": True, "lgbm_trees": 640, "iforest_trees": 200,
           "metrics": {"precision": 0.9991, "recall": 0.9958, "pr_auc": 0.99993, "lgbm_rounds": 640,
                       "split_sizes": {"test": 15000}, "confusion_matrix": {"matrix": [[3904, 10], [47, 11039]]}}},
    "laya": {"loaded": True, "checkpoint": "a1_6ep/final", "gate": {"lo": 0.1197, "hi": 0.8828}, "error": None},
    "laya_checkpoint": "20261113_171802_a1_6ep/final", "laya_weight": 0.7, "laya_fraud_threshold": 0.5,
    "qwen": {"model": "Qwen/Qwen3.5-2B", "configured": True, "reachable": True,
             "endpoint": "https://x.gradio.live", "detail": "HTTP 200"},
}


@pytest.fixture
def fake_api(monkeypatch):
    calls = []
    api = types.ModuleType("api_client")
    api.API_URL = "http://127.0.0.1:8000"
    stored = {"case-1": make_case(), "case-new": make_case("case-new", source="pending")}

    def record(name, result):
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result(*args, **kwargs) if callable(result) else result
        return call

    batch = lambda *a, **k: {"source": "text", "rows_received": 1, "rows_analysed": 1,
                             "errors": [], "notes": [], "results": [make_case("case-new", source="pending")]}
    api.get_cases = record("get_cases", lambda: list(stored.values()))
    api.get_health = record("get_health", HEALTH)
    api.get_case = record("get_case", lambda case_id: stored.get(case_id, make_case(case_id)))
    api.analyze = record("analyze", lambda payload: make_case("case-api"))
    api.analyze_input = record("analyze_input", batch)
    all_fields = [{"field": f, "label": f.replace("_", " "), "kind": "yes/no" if f.endswith("_flag") else "number",
                   "example": "e.g. 3"} for f in FEATURES]
    api.parse_text = record("parse_text", lambda content, kind=None: {
        "rows": [{"ok": True, "found": FEATURES, "missing": [], "understood": ["gap: 45 (from \"45 minutes\")"], "error": None}],
        "fields": all_fields})
    api.ALL_FIELDS = all_fields
    api.analyze_upload = record("analyze_upload", batch)
    api.review_case = record("review_case", lambda case_id, status: {**make_case(case_id), "status": status})
    api.regenerate_report = record("regenerate_report", lambda case_id: make_case(case_id, source="qwen"))
    api.get_qwen_settings = record("get_qwen_settings", {})
    api.set_qwen_endpoint = record("set_qwen_endpoint", lambda url: {"override": url, "check": {"reachable": True}})
    api.set_qwen_settings = record("set_qwen_settings", lambda endpoint=None, user=None, password=None, save_even_if_unreachable=False:
                                   {"override": endpoint, "check": {"reachable": bool(endpoint), "endpoint": endpoint}})
    api.qwen_status = record("qwen_status", {"reachable": False, "endpoint": None, "detail": "HTTP 404",
                                              "user": "simguard", "password_set": True, "credentials_from": "env"})
    api.clear_qwen_endpoint = record("clear_qwen_endpoint", {"override": None})
    monkeypatch.setitem(sys.modules, "api_client", api)
    monkeypatch.syspath_prepend(str(APP.parent))
    return calls


def start(page="Overview"):
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["page"] = page
    at.session_state["top_workspace"] = page
    at.run()
    assert not at.exception, at.exception
    return at


def names(calls):
    return [c[0] for c in calls]


@pytest.mark.parametrize("page", ["Overview", "Cases", "Analyze", "Data & metrics", "System status"])
def test_every_page_renders(fake_api, page):
    start(page)


def test_text_prompt_goes_to_the_input_parser_and_shows_the_case(fake_api):
    at = start("Analyze")
    at.text_area(key="prompt_text").set_value("SIM changed 45 minutes after an OTP request.").run()
    at.button(key="analyze_text").click().run()
    assert not at.exception, at.exception
    assert ("parse_text", ("SIM changed 45 minutes after an OTP request.", "text"), {}) in fake_api  # read first
    call = next(c for c in fake_api if c[0] == "analyze_input")
    assert call[1] == ("SIM changed 45 minutes after an OTP request.", "text")
    shown = " ".join(m.value for m in at.markdown) + " ".join(i.value for i in at.info)
    assert "LAYA GATE: BLOCK" in shown  # gate shown in the dossier
    assert "Qwen 3.5 2B is writing the report" in shown  # background report state


def test_pending_report_is_swapped_in_when_qwen_finishes(fake_api):
    at = start("Analyze")
    at.text_area(key="prompt_text").set_value("SIM changed 45 minutes after an OTP request.").run()
    at.button(key="analyze_text").click().run()
    assert any("writing the report" in i.value for i in at.info)
    # Qwen finishes in the background: the next poll gets the finished case and redraws
    sys.modules["api_client"].get_case = lambda case_id: make_case(case_id, source="qwen")
    at.run()
    assert not at.exception, at.exception
    assert not any("writing the report" in i.value for i in at.info)
    assert any("Written by Qwen 3.5 2B" in c.value for c in at.caption)


def test_guided_form_uses_background_reports(fake_api):
    at = start("Analyze")
    at.button(key="FormSubmitter:event_form-Analyze event").click().run()
    assert not at.exception, at.exception
    call = next(c for c in fake_api if c[0] == "analyze_input")
    payload = json.loads(call[1][0])
    assert call[1][1] == "json" and payload["imsi_change_flag"] == 0 and payload["msisdn_hash"].startswith("SYNTH-")
    assert "analyze" not in names(fake_api)


def test_pasted_json_with_a_real_subscriber_id_is_refused(fake_api):
    at = start("Analyze")
    at.text_area(key="pasted_text").set_value(json.dumps({**FEATURES, "msisdn_hash": "9876543210"})).run()
    at.button(key="analyze_paste").click().run()
    assert any("SYNTH-" in e.value for e in at.error)
    assert "analyze_input" not in names(fake_api)


def test_pasted_synthetic_json_is_analyzed(fake_api):
    at = start("Analyze")
    at.text_area(key="pasted_text").set_value(json.dumps({**FEATURES, "msisdn_hash": "SYNTH-1"})).run()
    at.button(key="analyze_paste").click().run()
    assert ("analyze_input", (json.dumps({**FEATURES, "msisdn_hash": "SYNTH-1"}),), {}) in fake_api


def test_case_detail_offers_qwen_regeneration(fake_api):
    at = start("Cases")
    at.session_state["selected_case"] = "case-1"
    at.run()
    at.button(key="cases_report_case-1").click().run()
    assert not at.exception, at.exception
    assert ("regenerate_report", ("case-1",), {}) in fake_api


def test_status_page_shows_live_services_and_saves_the_qwen_login(fake_api):
    at = start("System status")
    page = " ".join(m.value for m in at.markdown)
    assert "LightGBM + Isolation Forest Pipeline" in page and "UNKNOWN" not in page
    assert "Save & Run All" in page  # Qwen offline: how to start the notebook by hand
    assert not any("Start Qwen" in b.label for b in at.button)  # no start webhook
    link, user, password = at.text_input
    link.set_value("https://new-link.gradio.live")
    user.set_value("simguard")
    password.set_value("s3cret")  # inside a form: submit in the same run
    at.button(key="FormSubmitter:qwen_login_form-Save login & link").click().run()
    assert not at.exception, at.exception
    call = next(c for c in fake_api if c[0] == "set_qwen_settings")
    assert call[1] == ("https://new-link.gradio.live", "simguard", "s3cret")
    # the login is also queued for this browser's local storage
    assert at.session_state["qwen_store_req"]["action"] == "set"
    assert at.session_state["qwen_store_req"]["value"]["password"] == "s3cret"


def test_forget_clears_browser_and_api(fake_api):
    at = start("System status")
    at.button(key="FormSubmitter:qwen_login_form-Forget saved login & link").click().run()
    assert not at.exception, at.exception
    assert at.session_state["qwen_store_req"]["action"] == "clear"
    assert "clear_qwen_endpoint" in names(fake_api)
    assert ("set_qwen_settings", (None, "", ""), {}) in fake_api


def partial_parse(missing_fields):
    all_fields = sys.modules["api_client"].ALL_FIELDS
    found = {k: v for k, v in FEATURES.items() if k not in missing_fields}
    return lambda content, kind=None: {
        "rows": [{"ok": False, "found": found, "error": "missing fields: " + ", ".join(missing_fields),
                  "missing": [f for f in all_fields if f["field"] in missing_fields],
                  "understood": ["Device changed: yes (from \"device changed\")"]}],
        "fields": all_fields}


def test_missing_values_are_named_and_can_be_filled_in_without_retyping(fake_api):
    api = sys.modules["api_client"]
    api.parse_text = partial_parse(["imsi_change_flag", "account_age_days"])
    at = start("Analyze")
    at.text_area(key="prompt_text").set_value("Device changed, 45 minutes after the OTP.").run()
    at.button(key="analyze_text").click().run()
    assert not at.exception, at.exception
    assert not any(c[0] == "analyze_input" for c in fake_api)  # nothing is analysed with gaps
    assert any("understood 9 of the 11 values" in w.value and "2 are still missing" in w.value for w in at.warning)
    shown = " ".join(m.value for m in at.markdown)
    assert "What I understood" in shown and "What I could not find" in shown

    # submitting with a blank answer names what is still needed
    at.button(key="FormSubmitter:text_gap_form-Analyze with these values").click().run()
    assert any("Still needed" in e.value and "imsi change flag" in e.value for e in at.error)
    assert not any(c[0] == "analyze_input" for c in fake_api)

    at.selectbox(key="gap_imsi_change_flag").set_value("No")
    at.text_input(key="gap_account_age_days").set_value("4 months")
    at.button(key="FormSubmitter:text_gap_form-Analyze with these values").click().run()
    assert not at.exception, at.exception
    call = next(c for c in fake_api if c[0] == "analyze_input")
    sent = call[1][0]
    assert sent.startswith("Device changed, 45 minutes after the OTP.")  # the user's own words are kept
    assert "imsi_change_flag: no" in sent and "account_age_days: 4 months" in sent and call[1][1] == "text"


def test_an_unusable_value_is_explained_not_just_rejected(fake_api):
    api = sys.modules["api_client"]
    original = partial_parse([])
    api.parse_text = lambda content, kind=None: {**original(content), "rows": [
        {**original(content)["rows"][0], "error": "otp_and_sim_change_geo_hash_length: Input should be less than or equal to 1"}]}
    at = start("Analyze")
    at.text_area(key="prompt_text").set_value("whatever").run()
    at.button(key="analyze_text").click().run()
    assert not at.exception, at.exception
    assert any("not usable" in e.value and "less than or equal to 1" in e.value for e in at.error)
