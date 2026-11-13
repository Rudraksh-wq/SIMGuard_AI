import math
from collections.abc import Mapping


FEATURE_COLUMNS = (
    "sim_swap_time_gap_minutes",
    "device_change_flag",
    "sim_type_change_flag",
    "imsi_change_flag",
    "iccid_change_flag",
    "otp_and_sim_change_geo_hash_length",
    "recent_sim_activation_days",
    "num_sim_changes_last_30d",
    "previous_sim_holder_tenure_days",
    "account_age_days",
    "ip_change_flag",
)
LABEL_COLUMN = "sim_swap_flag"
_BINARY_COLUMNS = {
    "device_change_flag",
    "sim_type_change_flag",
    "imsi_change_flag",
    "iccid_change_flag",
    "ip_change_flag",
}


def _read_feature_values(row: Mapping[str, object]) -> dict[str, int | float]:
    expected_columns = set(FEATURE_COLUMNS)
    missing_columns = expected_columns.difference(row)
    extra_columns = set(row).difference(expected_columns | {LABEL_COLUMN})
    if missing_columns or extra_columns:
        details = []
        if missing_columns:
            details.append(f"missing columns: {', '.join(sorted(missing_columns))}")
        if extra_columns:
            details.append(f"unexpected columns: {', '.join(sorted(extra_columns))}")
        raise ValueError("; ".join(details))

    values: dict[str, int | float] = {}
    for column in FEATURE_COLUMNS:
        try:
            value = float(row[column])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{column} must be a finite number") from error
        if not math.isfinite(value):
            raise ValueError(f"{column} must be a finite number")
        if column in _BINARY_COLUMNS:
            if value not in (0.0, 1.0):
                raise ValueError(f"{column} must be 0 or 1")
            values[column] = int(value)
        else:
            values[column] = value
    return values


def _number_text(value: int | float) -> str:
    if isinstance(value, int) or value.is_integer():
        return str(int(value))
    return format(value, ".15g")


def event_to_text(row: Mapping[str, object]) -> str:
    values = _read_feature_values(row)
    return (
        "SIM swap event features. "
        f"SIM swap time gap: {_number_text(values['sim_swap_time_gap_minutes'])} minutes. "
        f"Device changed: {'yes' if values['device_change_flag'] else 'no'}. "
        f"SIM type changed: {'yes' if values['sim_type_change_flag'] else 'no'}. "
        f"IMSI changed: {'yes' if values['imsi_change_flag'] else 'no'}. "
        f"ICCID changed: {'yes' if values['iccid_change_flag'] else 'no'}. "
        "OTP and SIM change geo-hash length: "
        f"{_number_text(values['otp_and_sim_change_geo_hash_length'])}. "
        "Recent SIM activation: "
        f"{_number_text(values['recent_sim_activation_days'])} days ago. "
        "SIM changes in the last 30 days: "
        f"{_number_text(values['num_sim_changes_last_30d'])}. "
        "Previous SIM holder tenure: "
        f"{_number_text(values['previous_sim_holder_tenure_days'])} days. "
        f"Account age: {_number_text(values['account_age_days'])} days. "
        f"IP changed: {'yes' if values['ip_change_flag'] else 'no'}."
    )


# --- Laya input text -------------------------------------------------------------------------
# Must stay byte-identical to finetune/fraud_v2/fraud_common.render_state: the fine-tuned Laya
# checkpoint only saw this wording during training (tests/test_laya_agent.py enforces it).
LAYA_QUESTION_ID = "is_fraud"
LAYA_QUESTION = {
    "type": "noul",
    "instructions": "Does this SIM swap event look like SIM-swap fraud?",
}


def _yes_no(value: int | float) -> str:
    return "yes" if int(value) == 1 else "no"


def laya_text(row: Mapping[str, object]) -> str:
    v = _read_feature_values(row)
    gap = int(v["sim_swap_time_gap_minutes"])
    return (
        "SIM swap event. "
        f"Time between OTP request and SIM change: {gap} minutes ({gap / 1440:.2f} days). "
        f"Device changed: {_yes_no(v['device_change_flag'])}. "
        f"SIM type changed: {_yes_no(v['sim_type_change_flag'])}. "
        f"IMSI changed: {_yes_no(v['imsi_change_flag'])}. "
        f"ICCID changed: {_yes_no(v['iccid_change_flag'])}. "
        f"Geohash similarity between OTP location and SIM change location: "
        f"{float(v['otp_and_sim_change_geo_hash_length']):.2f} (0 = different place, 1 = same place). "
        f"Days since SIM activation: {int(v['recent_sim_activation_days'])}. "
        f"SIM changes in last 30 days: {int(v['num_sim_changes_last_30d'])}. "
        f"Previous SIM holder tenure: {int(v['previous_sim_holder_tenure_days'])} days. "
        f"Account age: {int(v['account_age_days'])} days. "
        f"IP address changed: {_yes_no(v['ip_change_flag'])}."
    )
