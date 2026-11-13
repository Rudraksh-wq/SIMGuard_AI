"""Turn whatever the analyst gives us (CSV, JSON, or a free-text prompt) into Events.

Rules:
- Column / key names are normalised (case, spaces, dashes) and common aliases are accepted.
- Yes/no/true/false become 1/0. A geohash similarity given as a percentage (e.g. 40%) is scaled.
- Missing features are NEVER invented: the row is rejected with the list of missing fields.
- Non-feature columns (ids, labels, notes) are kept as context, not fed to the models.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from agents.features_text import FEATURE_COLUMNS, LABEL_COLUMN
from agents.schemas import Event, InputInfo

MAX_ROWS = 2000
RAW_PREVIEW_CHARS = 2000
ID_FIELDS = ("event_id", "msisdn_hash", "timestamp")

# alias (normalised) -> canonical feature name
_ALIASES: dict[str, str] = {}
_ALIAS_TABLE = {
    "sim_swap_time_gap_minutes": [
        "sim_swap_time_gap", "time_gap_minutes", "time_gap", "gap_minutes", "swap_gap_minutes",
        "otp_to_sim_change_minutes", "time_between_otp_and_sim_change", "minutes_since_otp",
        "time_between_otp_request_and_sim_change",
    ],
    "device_change_flag": ["device_change", "device_changed", "imei_changed", "new_device"],
    "sim_type_change_flag": ["sim_type_change", "sim_type_changed", "esim_change"],
    "imsi_change_flag": ["imsi_change", "imsi_changed"],
    "iccid_change_flag": ["iccid_change", "iccid_changed"],
    "otp_and_sim_change_geo_hash_length": [
        "geo_hash_length", "geohash_length", "geohash_similarity", "geo_similarity",
        "location_similarity", "otp_sim_geohash", "geohash",
        "geohash_similarity_between_otp_location_and_sim_change_location",
    ],
    "recent_sim_activation_days": [
        "sim_activation_days", "days_since_sim_activation", "days_since_activation",
        "sim_age_days", "recent_activation_days",
    ],
    "num_sim_changes_last_30d": [
        "sim_changes_last_30d", "sim_changes_30d", "num_sim_changes", "sim_changes",
        "sim_changes_in_last_30_days", "num_sim_changes_30d",
    ],
    "previous_sim_holder_tenure_days": [
        "previous_holder_tenure_days", "previous_sim_holder_tenure", "holder_tenure_days",
        "tenure_days", "tenure",
    ],
    "account_age_days": ["account_age", "acct_age_days", "customer_age_days"],
    "ip_change_flag": ["ip_change", "ip_changed", "ip_address_changed", "new_ip"],
}


def _norm(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")


for _canonical, _names in _ALIAS_TABLE.items():
    _ALIASES[_canonical] = _canonical
    for _name in _names:
        _ALIASES[_norm(_name)] = _canonical
for _id in ID_FIELDS:
    _ALIASES[_id] = _id
_ALIASES.update({"id": "event_id", "case_id": "event_id", "msisdn": "msisdn_hash", "time": "timestamp"})

_BINARY = {c for c in FEATURE_COLUMNS if c.endswith("_flag")}
_TRUE = {"1", "yes", "y", "true", "t", "changed", "on"}
_FALSE = {"0", "no", "n", "false", "f", "unchanged", "same", "off", "none"}


class InputError(ValueError):
    pass


@dataclass
class ParsedRow:
    event: Event | None
    info: InputInfo
    error: str | None = None


@dataclass
class ParseResult:
    source: str
    rows: list[ParsedRow] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def events(self) -> list[tuple[Event, InputInfo]]:
        return [(r.event, r.info) for r in self.rows if r.event is not None]

    @property
    def errors(self) -> list[dict[str, Any]]:
        return [{"row": i + 1, "error": r.error} for i, r in enumerate(self.rows) if r.error]


def _to_number(feature: str, value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if feature in _BINARY:
        if text in _TRUE:
            return 1.0
        if text in _FALSE:
            return 0.0
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not match:
        raise ValueError(f"{feature}: could not read a number from {value!r}")
    number = float(match.group())
    if feature == "otp_and_sim_change_geo_hash_length" and ("%" in text or number > 1):
        number /= 100.0
    if feature == "sim_swap_time_gap_minutes":
        if re.search(r"\bdays?\b", text):
            number *= 1440
        elif re.search(r"\b(hours?|hrs?|h)\b", text):
            number *= 60
    return number


def _row_from_mapping(record: dict[str, Any], source: str, raw: str | None) -> ParsedRow:
    features: dict[str, Any] = {}
    ids: dict[str, str] = {}
    extra: dict[str, Any] = {}
    notes: list[str] = []
    for key, value in record.items():
        canonical = _ALIASES.get(_norm(key))
        if canonical in ID_FIELDS:
            ids[canonical] = "" if value is None else str(value)
        elif canonical:
            if canonical != key:
                notes.append(f"'{key}' read as {canonical}")
            features[canonical] = value
        elif value not in (None, ""):
            extra[str(key)] = value
    info = InputInfo(source=source, raw=(raw or None) and raw[:RAW_PREVIEW_CHARS], extra=extra, notes=notes)

    missing = [c for c in FEATURE_COLUMNS if c not in features or features[c] in (None, "")]
    if missing:
        return ParsedRow(None, info, f"missing fields: {', '.join(missing)}")
    try:
        values = {c: _to_number(c, features[c]) for c in FEATURE_COLUMNS}
        values = {c: int(v) if c in _BINARY else v for c, v in values.items()}
        event = Event(**ids, **values)
    except (ValueError, ValidationError) as error:
        message = str(error).splitlines()
        return ParsedRow(None, info, "; ".join(m for m in message if m.strip())[:400])
    return ParsedRow(event, info)


def parse_csv(text: str) -> ParseResult:
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    if not reader.fieldnames:
        raise InputError("CSV has no header row")
    result = ParseResult("csv")
    for index, record in enumerate(reader):
        if index >= MAX_ROWS:
            result.notes.append(f"only the first {MAX_ROWS} rows were analysed")
            break
        raw = ",".join("" if v is None else str(v) for v in record.values())
        result.rows.append(_row_from_mapping(dict(record), "csv", raw))
    if not result.rows:
        raise InputError("CSV has a header but no data rows")
    return result


def parse_json(text: str) -> ParseResult:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise InputError(f"invalid JSON: {error.msg} (line {error.lineno})") from error
    if isinstance(data, dict):
        for key in ("events", "rows", "data", "records", "cases"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            if isinstance(data.get("event"), dict):
                data = [data["event"]]
            else:
                data = [data]
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise InputError("JSON must be an object, a list of objects, or {\"events\": [...]}")
    result = ParseResult("json")
    for index, record in enumerate(data):
        if index >= MAX_ROWS:
            result.notes.append(f"only the first {MAX_ROWS} records were analysed")
            break
        result.rows.append(_row_from_mapping(record, "json", json.dumps(record)))
    if not result.rows:
        raise InputError("JSON contains no records")
    return result


# --- free text -------------------------------------------------------------------------------
_NUM = r"(-?\d[\d,]*(?:\.\d+)?)\s*(%|minutes?|mins?|hours?|hrs?|days?)?"
_SEP = r"[\s:=(]*(?:(?:is|of|was|=|about|around|approx(?:imately)?)\s+)?[\s:=]*"
_TEXT_PATTERNS: list[tuple[str, str]] = [
    ("sim_swap_time_gap_minutes",
     r"(?:(?:sim (?:swap |change )?)?time\s*gap|gap|time between (?:the )?otp(?: request)? and (?:the )?sim change|"
     r"otp (?:request )?to sim change|sim swap time gap)" + _SEP + _NUM),
    ("otp_and_sim_change_geo_hash_length",
     r"(?:geo\s*-?hash(?:\s*(?:length|similarity))?|location similarity|geo similarity)"
     r"(?: between [a-z ]+? location)?" + _SEP + _NUM),
    ("recent_sim_activation_days",
     r"(?:(?:days since|recent) (?:sim )?activation|sim activat(?:ed|ion))" + _SEP + _NUM),
    ("num_sim_changes_last_30d",
     r"(?:sim changes?(?: in)?(?: the)? (?:last|past) 30 ?d(?:ays)?|sim changes? last 30d|"
     r"num(?:ber of)? sim changes?)" + _SEP + _NUM),
    ("previous_sim_holder_tenure_days",
     r"(?:previous (?:sim )?holder tenure|holder tenure|tenure)" + _SEP + _NUM),
    ("account_age_days", r"account(?: is)? age" + _SEP + _NUM),
]
_TEXT_FLAGS = {
    "device_change_flag": r"device(?:\s+change)?",
    "sim_type_change_flag": r"sim\s*type(?:\s+change)?",
    "imsi_change_flag": r"imsi(?:\s+change)?",
    "iccid_change_flag": r"iccid(?:\s+change)?",
    "ip_change_flag": r"ip(?:\s+address)?(?:\s+change)?",
}
_REVERSED_NUMBER = {  # "30 days old account", "2 sim changes in the last 30 days"
    "account_age_days": r"" + _NUM + r"[ -]?(?:old )?account",
    "num_sim_changes_last_30d": r"(\d+)()\s*sim changes?",
    "sim_swap_time_gap_minutes": _NUM + r"\s+(?:after|before|since|from|following)\s+(?:an |the )?otp",
    "recent_sim_activation_days": r"activated\s+" + _NUM + r"\s+ago",
}


def _flag_from_text(text: str, pattern: str) -> int | None:
    explicit = re.search(pattern + r"\s*(?:changed|change)?\s*[:=]\s*(yes|no|true|false|1|0|y|n)\b", text)
    if explicit:
        return 1 if explicit.group(1) in _TRUE else 0
    if re.search(r"(?:no|not|without|unchanged|same)\s+(?:\w+\s+){0,2}" + pattern + r"(?!\w)", text) or re.search(
        pattern + r"\s+(?:was |is |did )?(?:not changed|unchanged|did not change|same|no change)", text
    ):
        return 0
    if re.search(pattern + r"\s+(?:was |is |has been |got )?(?:changed|swapped|replaced|new|different)", text) or re.search(
        r"(?:changed|new|different|swapped)\s+(?:the\s+)?" + pattern + r"(?!\w)", text
    ):
        return 1
    return None


def parse_text(text: str) -> ParseResult:
    """Free-text prompt -> one event. `key: value` lines are tried first, then phrases."""
    stripped = text.strip()
    if not stripped:
        raise InputError("the prompt is empty")
    lowered = " ".join(stripped.lower().split())

    record: dict[str, Any] = {}
    for line in re.split(r"[\n;]|,(?=\s*[a-z_ ]+[:=])", stripped):
        if re.search(r"[:=]", line):
            key, _, value = re.split(r"([:=])", line, maxsplit=1)
            canonical = _ALIASES.get(_norm(key))
            if canonical and value.strip():
                record[canonical] = value.strip().rstrip(".")

    for feature, pattern in _REVERSED_NUMBER.items():
        if feature not in record:
            match = re.search(pattern, lowered)
            if match:
                record[feature] = (match.group(1).replace(",", "") + " " + (match.group(2) or "")).strip()
    for feature, pattern in _TEXT_PATTERNS:
        if feature in record:
            continue
        match = re.search(pattern, lowered)
        if match:
            record[feature] = (match.group(1).replace(",", "") + " " + (match.group(2) or "")).strip()
    for feature, pattern in _TEXT_FLAGS.items():
        if feature not in record:
            flag = _flag_from_text(lowered, pattern)
            if flag is not None:
                record[feature] = flag

    result = ParseResult("text")
    row = _row_from_mapping(record, "text", stripped)
    row.info.notes = [f"read {k} = {record[k]!r} from the prompt" for k in FEATURE_COLUMNS if k in record]
    result.rows.append(row)
    return result


def parse_input(content: str, filename: str | None = None, kind: str | None = None) -> ParseResult:
    """Detect the format (explicit kind > file extension > content sniffing) and parse."""
    kind = (kind or "").lower() or None
    if kind is None and filename:
        lower = filename.lower()
        kind = "csv" if lower.endswith((".csv", ".tsv")) else "json" if lower.endswith((".json", ".jsonl")) else None
    text = content.strip()
    if kind is None:
        if text[:1] in "[{":
            kind = "json"
        else:
            first = text.splitlines()[0] if text else ""
            known = sum(1 for part in first.split(",") if _ALIASES.get(_norm(part)) in FEATURE_COLUMNS)
            kind = "csv" if known >= 3 else "text"
    if kind == "json" and filename and filename.lower().endswith(".jsonl"):
        lines = [line for line in text.splitlines() if line.strip()]
        text = "[" + ",".join(lines) + "]"
    if kind == "csv":
        if filename and filename.lower().endswith(".tsv"):
            text = text.replace("\t", ",")
        return parse_csv(text)
    if kind == "json":
        return parse_json(text)
    if kind == "text":
        return parse_text(text)
    raise InputError(f"unknown input kind {kind!r}; use csv, json or text")


def strip_label(extra: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """Split the dataset label off the context so it is never sent to the models."""
    extra = dict(extra)
    label = None
    for key in list(extra):
        if _norm(key) in {LABEL_COLUMN, "label", "is_fraud", "fraud"}:
            label = extra.pop(key)
    return extra, label
