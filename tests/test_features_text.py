import pytest

from agents.features_text import event_to_text


@pytest.fixture
def event():
    return {
        "sim_swap_time_gap_minutes": "9932",
        "device_change_flag": "1",
        "sim_type_change_flag": "1",
        "imsi_change_flag": "0",
        "iccid_change_flag": "1",
        "otp_and_sim_change_geo_hash_length": "0.41",
        "recent_sim_activation_days": "237",
        "num_sim_changes_last_30d": "0",
        "previous_sim_holder_tenure_days": "917",
        "account_age_days": "1623",
        "ip_change_flag": "1",
        "sim_swap_flag": "1",
    }


def test_event_text_uses_dataset_features_without_label_leakage(event):
    text = event_to_text(event)

    assert "Device changed: yes." in text
    assert "IMSI changed: no." in text
    assert "SIM swap time gap: 9932 minutes." in text
    assert "OTP and SIM change geo-hash length: 0.41." in text
    assert "sim_swap_flag" not in text
    assert " 1." not in text
    assert len(text.split()) < 200


def test_event_text_is_independent_of_target_label(event):
    positive = event_to_text({"sim_swap_flag": "1", **{k: v for k, v in event.items() if k != "sim_swap_flag"}})
    negative = event_to_text({"sim_swap_flag": "0", **{k: v for k, v in event.items() if k != "sim_swap_flag"}})

    assert positive == negative


def test_event_text_rejects_missing_or_unknown_features(event):
    with pytest.raises(ValueError, match="missing columns"):
        event_to_text({key: value for key, value in event.items() if key != "device_change_flag"})
    with pytest.raises(ValueError, match="unexpected columns"):
        event_to_text({**event, "request_hour": "3"})


def test_event_text_rejects_invalid_flags(event):
    event["device_change_flag"] = "2"

    with pytest.raises(ValueError, match="device_change_flag must be 0 or 1"):
        event_to_text(event)
