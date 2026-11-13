import pytest

from agents.evidence import extract_evidence


@pytest.fixture
def event():
    return {
        "sim_swap_time_gap_minutes": 100,
        "device_change_flag": 0,
        "sim_type_change_flag": 0,
        "imsi_change_flag": 0,
        "iccid_change_flag": 0,
        "otp_and_sim_change_geo_hash_length": 0.5,
        "recent_sim_activation_days": 30,
        "num_sim_changes_last_30d": 0,
        "previous_sim_holder_tenure_days": 100,
        "account_age_days": 100,
        "ip_change_flag": 0,
        "sim_swap_flag": 0,
    }


def test_evidence_thresholds_and_dataset_indicators(event):
    assert extract_evidence(event) == []

    event.update(
        {
            "device_change_flag": 1,
            "sim_type_change_flag": 1,
            "imsi_change_flag": 1,
            "iccid_change_flag": 1,
            "recent_sim_activation_days": 29,
            "num_sim_changes_last_30d": 2,
            "ip_change_flag": 1,
        }
    )

    assert extract_evidence(event) == [
        "Device changed",
        "SIM type changed",
        "IMSI changed",
        "ICCID changed",
        "recent SIM activation (29 days ago)",
        "2 SIM changes in the last 30 days",
        "IP address changed",
    ]


def test_evidence_does_not_depend_on_target_label(event):
    positive = extract_evidence({**event, "sim_swap_flag": 1})
    negative = extract_evidence({**event, "sim_swap_flag": 0})

    assert positive == negative


def test_evidence_rejects_non_finite_feature_values(event):
    event["recent_sim_activation_days"] = float("nan")

    with pytest.raises(ValueError, match="must be a finite number"):
        extract_evidence(event)
