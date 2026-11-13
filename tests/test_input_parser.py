import json

import pytest

from agents.features_text import FEATURE_COLUMNS
from api.input_parser import InputError, parse_input, strip_label

FRAUD_PROMPT = (
    "SIM changed 45 minutes after an OTP request. Device changed, IMSI changed, ICCID changed, "
    "SIM type not changed. Geohash similarity 0.12. SIM activated 3 days ago, 2 SIM changes in "
    "last 30 days. Previous holder tenure 40 days, account age 120 days. IP changed."
)
EXPECTED_FRAUD = {
    "sim_swap_time_gap_minutes": 45, "device_change_flag": 1, "sim_type_change_flag": 0,
    "imsi_change_flag": 1, "iccid_change_flag": 1, "otp_and_sim_change_geo_hash_length": 0.12,
    "recent_sim_activation_days": 3, "num_sim_changes_last_30d": 2,
    "previous_sim_holder_tenure_days": 40, "account_age_days": 120, "ip_change_flag": 1,
}


def test_free_text_prompt_is_parsed_into_all_features():
    result = parse_input(FRAUD_PROMPT, kind="text")
    assert result.source == "text" and not result.errors
    event, info = result.events[0]
    assert event.features() == EXPECTED_FRAUD
    assert info.raw == FRAUD_PROMPT


def test_key_value_prompt_with_units_and_percent():
    prompt = "\n".join([
        "time gap: 2 hours", "device changed: no", "sim type changed: no", "imsi changed: no",
        "iccid changed: yes", "geohash similarity: 94%", "days since sim activation: 400",
        "sim changes last 30d: 0", "previous holder tenure: 900 days", "account age: 1500 days",
        "ip changed: no",
    ])
    event, _ = parse_input(prompt).events[0]
    assert event.sim_swap_time_gap_minutes == 120
    assert event.otp_and_sim_change_geo_hash_length == pytest.approx(0.94)
    assert event.iccid_change_flag == 1 and event.device_change_flag == 0


def test_missing_fields_are_reported_not_invented():
    result = parse_input("device changed: yes. IMSI changed.", kind="text")
    assert not result.events
    assert "missing fields" in result.errors[0]["error"]
    assert "account_age_days" in result.errors[0]["error"]


def test_csv_with_aliases_label_and_bad_rows():
    header = "event_id,time_gap_minutes,device_changed,sim_type_change,imsi_change,iccid_change," \
             "geohash_similarity,sim_activation_days,sim_changes_30d,tenure_days,account_age,ip_changed,sim_swap_flag"
    csv_text = "\n".join([header, "e1,30,yes,1,1,1,0.05,2,3,20,60,true,1", "e2,abc,0,0,0,0,0.9,1,0,1,1,0,0"])
    result = parse_input(csv_text, filename="cases.csv")
    assert result.source == "csv"
    assert len(result.events) == 1 and len(result.errors) == 1
    event, info = result.events[0]
    assert event.event_id == "e1" and event.device_change_flag == 1 and event.ip_change_flag == 1
    context, label = strip_label(info.extra)
    assert label == "1" and context == {}


def test_json_shapes_are_all_accepted():
    row = dict(EXPECTED_FRAUD, event_id="x")
    for content in (json.dumps(row), json.dumps([row, row]), json.dumps({"events": [row]})):
        result = parse_input(content)
        assert result.source == "json" and result.events and not result.errors
    jsonl = json.dumps(row) + "\n" + json.dumps(row)
    assert len(parse_input(jsonl, filename="rows.jsonl").events) == 2


def test_csv_is_sniffed_without_a_filename():
    csv_text = ",".join(FEATURE_COLUMNS) + "\n" + "9932,1,1,0,1,0.41,237,0,917,1623,1"
    assert parse_input(csv_text).source == "csv"


def test_invalid_json_and_empty_prompt_raise():
    with pytest.raises(InputError):
        parse_input("{not json", kind="json")
    with pytest.raises(InputError):
        parse_input("   ", kind="text")


def test_out_of_range_values_are_rejected():
    bad = dict(EXPECTED_FRAUD, otp_and_sim_change_geo_hash_length=-3)
    result = parse_input(json.dumps(bad))
    assert not result.events and result.errors


@pytest.mark.parametrize(
    "prompt, expected",
    [
        (
            "gap 3 hours. new device, imsi changed, iccid unchanged, sim type unchanged, geohash 0.3, "
            "sim activation 10 days, 0 sim changes in last 30 days, tenure 300 days, account age 800 days, "
            "ip address not changed",
            {"sim_swap_time_gap_minutes": 180, "device_change_flag": 1, "iccid_change_flag": 0,
             "num_sim_changes_last_30d": 0, "previous_sim_holder_tenure_days": 300, "ip_change_flag": 0},
        ),
        (
            "gap 2 days; device changed: no; sim type: no; imsi: yes; iccid: yes; geohash similarity 88%; "
            "days since activation 90; sim changes last 30 days 1; previous holder tenure 500; "
            "account age is 2000 days; ip changed yes",
            {"sim_swap_time_gap_minutes": 2880, "otp_and_sim_change_geo_hash_length": 0.88,
             "num_sim_changes_last_30d": 1, "account_age_days": 2000, "ip_change_flag": 1},
        ),
    ],
)
def test_prompt_regressions(prompt, expected):
    event, _ = parse_input(prompt, kind="text").events[0]
    features = event.features()
    for key, value in expected.items():
        assert features[key] == pytest.approx(value), key
