"""Free-text input: users describe the event in their own words, we extract the 11 values and say
exactly what is missing (never a template, never a guess)."""

import os

import pytest
from fastapi.testclient import TestClient

from api.input_parser import FIELD_HELP, field_help, parse_input
from agents.features_text import FEATURE_COLUMNS

GAP, DEV, SIMT, IMSI, ICCID = ("sim_swap_time_gap_minutes", "device_change_flag", "sim_type_change_flag",
                               "imsi_change_flag", "iccid_change_flag")
GEO, ACT, CHG, TEN, ACC, IP = ("otp_and_sim_change_geo_hash_length", "recent_sim_activation_days",
                               "num_sim_changes_last_30d", "previous_sim_holder_tenure_days",
                               "account_age_days", "ip_change_flag")

COMPLETE = [
    ("template",
     "SIM changed 45 minutes after an OTP request. Device changed, IMSI changed, ICCID changed, SIM type not "
     "changed. Geohash similarity 0.12. SIM activated 3 days ago, 2 SIM changes in last 30 days. Previous holder "
     "tenure 40 days, account age 120 days. IP changed.",
     {GAP: 45, DEV: 1, IMSI: 1, ICCID: 1, SIMT: 0, GEO: .12, ACT: 3, CHG: 2, TEN: 40, ACC: 120, IP: 1}),
    ("casual narrative",
     "Customer got a new SIM. The OTP went out and 20 minutes later the SIM was swapped. New handset, IMSI "
     "different, ICCID unchanged, same SIM type. Location similarity is 40%. The SIM is 5 days old with 1 SIM "
     "change this month. Previous owner had the number for 3 years. Customer for two years. Same IP.",
     {GAP: 20, DEV: 1, IMSI: 1, ICCID: 0, SIMT: 0, GEO: .4, ACT: 5, CHG: 1, TEN: 1095, ACC: 730, IP: 0}),
    ("number words",
     "The SIM was swapped forty five minutes after the OTP. Device and IMSI changed but not the ICCID or SIM type. "
     "Geohash similarity 0.3. The SIM was activated three days ago. Two SIM changes in the last thirty days. "
     "Previous holder tenure: one hundred and twenty days. Account age eighteen months. The IP address is new.",
     {GAP: 45, DEV: 1, IMSI: 1, ICCID: 0, SIMT: 0, GEO: .3, ACT: 3, CHG: 2, TEN: 120, ACC: 540, IP: 1}),
    ("key value mix",
     "device: yes, imsi: no, iccid: yes, sim type: no, geohash 0.9, ip: no, time gap 3 hours, sim activated 400 "
     "days ago, 0 sim changes, tenure 900 days, account age 4 years",
     {GAP: 180, DEV: 1, IMSI: 0, ICCID: 1, SIMT: 0, GEO: .9, ACT: 400, CHG: 0, TEN: 900, ACC: 1460, IP: 0}),
    ("everything matches",
     "Swap happened 2 days after the verification code was sent. Everything matches: same device, IMSI unchanged, "
     "ICCID unchanged, same SIM type, same IP. Same location. SIM activated 2 years ago. No other SIM changes. "
     "Holder tenure 5 years. Account is 8 years old.",
     {GAP: 2880, DEV: 0, IMSI: 0, ICCID: 0, SIMT: 0, IP: 0, GEO: 1, ACT: 730, CHG: 0, TEN: 1825, ACC: 2920}),
    ("hours, yesterday, one-time password",
     "SIM swapped 2 hours after the one-time password. New device, new IMSI, new ICCID, eSIM instead of physical so "
     "SIM type changed. Location similarity 5 percent. SIM activated yesterday. SIM changed twice in the last 30 "
     "days. Previous holder owned it for 18 months. Account created 6 months ago. IP changed.",
     {GAP: 120, DEV: 1, IMSI: 1, ICCID: 1, SIMT: 1, GEO: .05, ACT: 1, CHG: 2, TEN: 540, ACC: 180, IP: 1}),
    ("negations",
     "The SIM changed within 10 minutes of the OTP. The device did not change, no IMSI change, the ICCID has not "
     "changed, SIM type remained the same, and the IP wasn't changed. Geohash 0.77. Activated 12 days ago. Three SIM "
     "swaps in the past 30 days. Tenure 700 days. Account age 1500 days.",
     {GAP: 10, DEV: 0, IMSI: 0, ICCID: 0, SIMT: 0, IP: 0, GEO: .77, ACT: 12, CHG: 3, TEN: 700, ACC: 1500}),
    ("half hour and a shared verb",
     "Half an hour after the OTP was requested the SIM was replaced. Device, IMSI and ICCID were all changed. SIM "
     "type not changed, IP not changed. Geohash similarity 0.5. SIM activated 1 week ago. 1 SIM change in the last "
     "30 days. Previous holder tenure 2 years. The account is 3 months old.",
     {GAP: 30, DEV: 1, IMSI: 1, ICCID: 1, SIMT: 0, IP: 0, GEO: .5, ACT: 7, CHG: 1, TEN: 730, ACC: 90}),
    ("bullet list",
     "- Device changed: yes\n- SIM type changed: no\n- IMSI changed: yes\n- ICCID changed: yes\n- IP changed: no\n"
     "- Time between OTP and SIM change: 12 minutes\n- Location similarity: 0.05\n- SIM activated 1 day ago\n"
     "- SIM changes (last 30 days): 4\n- Previous holder tenure: 6 months\n- Account age: 1 year",
     {GAP: 12, DEV: 1, SIMT: 0, IMSI: 1, ICCID: 1, IP: 0, GEO: .05, ACT: 1, CHG: 4, TEN: 180, ACC: 365}),
    ("scattered wording",
     "The IMSI and the ICCID both changed, the device didn't, and neither did the SIM type. It took 3 hours "
     "between the OTP and the SIM swap. The OTP location and the SIM change location were 80% similar. SIM was "
     "activated a week ago. Just 1 SIM change recently... Tenure of the previous holder was 400 days. Account has "
     "been open for 2 years. IP changed.",
     {GAP: 180, IMSI: 1, ICCID: 1, DEV: 0, SIMT: 0, GEO: .8, ACT: 7, CHG: 1, TEN: 400, ACC: 730, IP: 1}),
    ("upper case and abbreviations",
     "SIM SWAPPED 5 MINS AFTER OTP!!! NEW IMEI, SAME IMSI, NEW ICCID, SIM TYPE CHANGED (ESIM), IP ADDRESS CHANGED. "
     "GEOHASH 0.02. ACTIVATED 2 DAYS AGO. 3 SIM CHANGES IN LAST 30 DAYS. PREVIOUS HOLDER TENURE 10 DAYS. ACCOUNT "
     "AGE 30 DAYS.",
     {GAP: 5, DEV: 1, IMSI: 0, ICCID: 1, SIMT: 1, IP: 1, GEO: .02, ACT: 2, CHG: 3, TEN: 10, ACC: 30}),
    ("decimals and other verbs",
     "Swap came 2.5 hours after OTP. SIM was changed 3 times in the last 30 days. Phone changed. No IMSI change. "
     "ICCID changed. Sim type is the same. IP address stayed the same. Geo-hash similarity of 0.6. SIM activation "
     "45 days ago. Previous owner kept it for 4 years. Customer since 5 years.",
     {GAP: 150, CHG: 3, DEV: 1, IMSI: 0, ICCID: 1, SIMT: 0, IP: 0, GEO: .6, ACT: 45, TEN: 1460, ACC: 1825}),
]


@pytest.mark.parametrize("name, text, expected", COMPLETE, ids=[c[0] for c in COMPLETE])
def test_any_wording_yields_all_eleven_values(name, text, expected):
    row = parse_input(text, kind="text").rows[0]
    assert row.missing == [] and row.event is not None, row.error
    for field, want in expected.items():
        assert float(row.found[field]) == pytest.approx(want, rel=1e-3), field
    assert set(row.found) == set(FEATURE_COLUMNS)


def test_partial_text_keeps_what_it_found_and_names_what_is_missing():
    row = parse_input("Someone swapped my SIM 5 minutes after an OTP and the device changed.", kind="text").rows[0]
    assert row.event is None
    assert row.found == {GAP: 5.0, DEV: 1}
    assert set(row.missing) == set(FEATURE_COLUMNS) - {GAP, DEV}
    assert "missing fields" in row.error


@pytest.mark.parametrize("text", [
    "hello there, can you check this for me please",
    "I have 3 cats and 2 dogs and 40 plants",
    "The new location is 1200 km away from the old one.",  # km is not a similarity
])
def test_unrelated_text_is_not_over_read(text):
    row = parse_input(text, kind="text").rows[0]
    assert row.found == {}


def test_a_sentence_with_commas_is_text_not_csv_when_auto_detected():
    result = parse_input("Device changed, IMSI changed, ICCID changed, SIM type not changed.")
    assert result.source == "text"


def test_added_key_value_lines_fill_gaps_in_the_original_text():
    text = "Someone swapped my SIM 5 minutes after an OTP and the device changed."
    filled = text + "\nimsi_change_flag: no\naccount_age_days: 4 months\nsim_type_change_flag: yes"
    found = parse_input(filled, kind="text").rows[0].found
    assert found[IMSI] == 0 and found[SIMT] == 1
    assert found[ACC] == pytest.approx(120)  # "4 months" -> days


def test_units_are_converted_in_every_input_format():
    row = parse_input('{"sim_swap_time_gap_minutes": "2 hours", "account_age": "1 year", "tenure": "6 months", '
                      '"geohash": "50%", "device": "yes", "imsi": 0, "iccid": 1, "sim_type": 0, "ip": "no", '
                      '"activation_days": 3, "sim_changes": 2}').rows[0]
    assert row.event is not None, row.error
    assert row.found[GAP] == 120 and row.found[ACC] == 365 and row.found[TEN] == 180 and row.found[GEO] == 0.5


def test_field_help_covers_every_feature():
    assert [item["field"] for item in field_help()] == list(FEATURE_COLUMNS)
    assert set(FIELD_HELP) == set(FEATURE_COLUMNS)


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MOCK_MODELS", "1")
    from api import main

    with TestClient(main.app) as c:
        yield c


def test_parse_endpoint_reports_understood_and_missing(client):
    text = "The OTP was sent and about twenty minutes later the SIM got swapped. New handset, IMSI different."
    body = client.post("/parse", json={"content": text, "kind": "text"}).json()
    row = body["rows"][0]
    assert row["ok"] is False and row["found"][GAP] == 20 and row["found"][DEV] == 1
    missing = {item["field"] for item in row["missing"]}
    assert ICCID in missing and GAP not in missing
    assert all(item["label"] and item["example"] for item in row["missing"])
    assert any("20" in note for note in row["understood"])


def test_analyze_error_explains_what_is_missing(client):
    response = client.post("/analyze/input", json={"content": "device changed: yes", "kind": "text"})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "could not find" in detail["message"] and "Device" not in detail["message"].split("find:")[1]
    assert detail["found"] == {DEV: 1} and len(detail["missing"]) == 10


def test_free_text_is_analysed_end_to_end(client):
    text = ("SIM was swapped 45 minutes after the OTP. New device, IMSI and ICCID both changed, but the SIM type "
            "stayed the same. Geohash similarity 12%. Activated three days ago, two SIM changes in the last 30 "
            "days. Previous holder tenure 40 days. Account is 4 months old. IP changed.")
    response = client.post("/analyze/input", json={"content": text, "kind": "text"})
    assert response.status_code == 200, response.text
    event = response.json()["results"][0]["event"]
    assert event[GAP] == 45 and event[SIMT] == 0 and event[GEO] == pytest.approx(0.12) and event[ACC] == 120
