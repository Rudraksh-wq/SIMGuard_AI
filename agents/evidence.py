from collections.abc import Mapping

from agents.features_text import _read_feature_values, _number_text


def extract_evidence(row: Mapping[str, object]) -> list[str]:
    values = _read_feature_values(row)
    evidence = []

    if values["device_change_flag"] == 1:
        evidence.append("Device changed")
    if values["sim_type_change_flag"] == 1:
        evidence.append("SIM type changed")
    if values["imsi_change_flag"] == 1:
        evidence.append("IMSI changed")
    if values["iccid_change_flag"] == 1:
        evidence.append("ICCID changed")
    if values["recent_sim_activation_days"] < 30:
        evidence.append(
            "recent SIM activation "
            f"({_number_text(values['recent_sim_activation_days'])} days ago)"
        )
    if values["num_sim_changes_last_30d"] > 0:
        evidence.append(
            f"{_number_text(values['num_sim_changes_last_30d'])} "
            "SIM changes in the last 30 days"
        )
    if values["ip_change_flag"] == 1:
        evidence.append("IP address changed")

    return evidence
