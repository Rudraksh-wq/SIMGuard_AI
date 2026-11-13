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
    "device_change_flag": ["device_change", "device_changed", "imei_changed", "new_device", "device", "handset_changed", "phone_changed"],
    "sim_type_change_flag": ["sim_type_change", "sim_type_changed", "esim_change", "sim_type"],
    "imsi_change_flag": ["imsi_change", "imsi_changed", "imsi"],
    "iccid_change_flag": ["iccid_change", "iccid_changed", "iccid"],
    "otp_and_sim_change_geo_hash_length": [
        "geo_hash_length", "geohash_length", "geohash_similarity", "geo_similarity",
        "location_similarity", "otp_sim_geohash", "geohash",
        "geohash_similarity_between_otp_location_and_sim_change_location",
    ],
    "recent_sim_activation_days": [
        "sim_activation_days", "days_since_sim_activation", "days_since_activation",
        "sim_age_days", "recent_activation_days", "sim_activation", "sim_activated", "activation_days", "days_since_activated",
    ],
    "num_sim_changes_last_30d": [
        "sim_changes_last_30d", "sim_changes_30d", "num_sim_changes", "sim_changes",
        "sim_changes_in_last_30_days", "num_sim_changes_30d",
    ],
    "previous_sim_holder_tenure_days": [
        "previous_holder_tenure_days", "previous_sim_holder_tenure", "holder_tenure_days",
        "tenure_days", "tenure", "previous_holder_tenure", "holder_tenure", "previous_owner_tenure",
    ],
    "account_age_days": ["account_age", "acct_age_days", "customer_age_days", "age_of_account", "account_age_in_days"],
    "ip_change_flag": ["ip_change", "ip_changed", "ip_address_changed", "new_ip", "ip", "ip_address"],
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
    found: dict[str, Any] = field(default_factory=dict)  # feature -> value understood so far
    missing: list[str] = field(default_factory=list)  # features that could not be found


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


_DAY_FEATURES = {"recent_sim_activation_days", "previous_sim_holder_tenure_days", "account_age_days"}


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
        lead = re.match(r"(yes|no|true|false|y|n|changed|unchanged|same)\b", text)
        if lead:
            return 1.0 if lead.group(1) in _TRUE else 0.0
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not match:
        raise ValueError(f"{feature}: could not read a number from {value!r}")
    number = float(match.group())
    if feature == "otp_and_sim_change_geo_hash_length" and ("%" in text or number > 1):
        number /= 100.0
    if feature == "sim_swap_time_gap_minutes" or feature in _DAY_FEATURES:
        unit = re.search(r"\d\s*(" + _UNIT_ALT + r")\b", text.replace(",", ""))
        per_unit = _UNIT_WORDS[unit.group(1)] if unit else (1 if feature == "sim_swap_time_gap_minutes" else 1440)
        number = number * per_unit if feature == "sim_swap_time_gap_minutes" else number * per_unit / 1440
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
    found: dict[str, Any] = {}
    for c in FEATURE_COLUMNS:
        if c in features and features[c] not in (None, ""):
            try:
                number = _to_number(c, features[c])
                found[c] = int(number) if c in _BINARY else round(number, 6)
            except ValueError:
                found[c] = features[c]
    if missing:
        return ParsedRow(None, info, f"missing fields: {', '.join(missing)}", found, missing)
    try:
        values = {c: _to_number(c, features[c]) for c in FEATURE_COLUMNS}
        values = {c: int(v) if c in _BINARY else v for c, v in values.items()}
        event = Event(**ids, **values)
    except (ValueError, ValidationError) as error:
        message = str(error).splitlines()
        return ParsedRow(None, info, "; ".join(m for m in message if m.strip())[:400], found, [])
    return ParsedRow(event, info, None, found, [])


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
# The analyst can write the event in any words. We never ask for a template: numbers, units,
# number words ("forty five minutes"), yes/no statements ("device and IMSI changed but not the
# ICCID") and `key: value` lines are all understood. Anything we cannot find is reported by name,
# never guessed.

_UNIT_WORDS = {  # unit -> minutes
    "min": 1, "mins": 1, "minute": 1, "minutes": 1,
    "h": 60, "hr": 60, "hrs": 60, "hour": 60, "hours": 60,
    "d": 1440, "day": 1440, "days": 1440,
    "wk": 10080, "wks": 10080, "week": 10080, "weeks": 10080,
    "month": 43200, "months": 43200,
    "yr": 525600, "yrs": 525600, "year": 525600, "years": 525600,
}
_UNIT_ALT = "|".join(sorted(_UNIT_WORDS, key=len, reverse=True))
_LONG_UNIT_ALT = "|".join(sorted((u for u in _UNIT_WORDS if len(u) > 1), key=len, reverse=True))
_NUM = r"(?P<num>\d+(?:\.\d+)?)"
_UNIT = rf"(?P<unit>{_LONG_UNIT_ALT})\b"
_OTP = r"(?:otp|one[- ]?time[- ]?(?:password|passcode|code|pin)|verification code|login code|auth(?:entication)? code)"
_FILL = r"[^0-9.;\n]"  # what may sit between a keyword and its number
_CHANGE_WORDS = r"(?:changes?|swaps?|replacements?)"
_CHANGES = rf"(?:sim\s+)?{_CHANGE_WORDS}"
_PRIOR = r"(?:(?:other|previous|prior|earlier|recent|additional|separate)\s+)*"

# feature -> regexes in priority order. Each has a `num` group and optionally a `unit` group.
_NUMERIC_PATTERNS: list[tuple[str, list[str]]] = [
    ("sim_swap_time_gap_minutes", [
        rf"{_NUM}\s*{_UNIT}\s+(?:after|later|before|following|since|from|past)\b[^.;\n]{{0,40}}?{_OTP}",
        rf"{_OTP}[^.;\n]{{0,70}}?{_NUM}\s*{_UNIT}\s+(?:later|after|before|earlier|apart)\b",
        rf"within\s+{_NUM}\s*{_UNIT}\s+(?:of|from)\b[^.;\n]{{0,25}}?{_OTP}",
        rf"{_NUM}\s*{_UNIT}\s+(?:between|separating|separates|apart)\b[^.;\n]{{0,40}}?{_OTP}",
        rf"(?:time\s*gap|gap|delay|interval|lag|elapsed|time\s+between|time\s+difference|difference)"
        rf"{_FILL}{{0,70}}?{_NUM}(?:\s*{_UNIT})?",
    ]),
    ("recent_sim_activation_days", [
        rf"(?:activated|activation|activate)\b{_FILL}{{0,40}}?{_NUM}\s*{_UNIT}",
        rf"(?:days\s+since|since)\s+(?:the\s+)?(?:sim\s+)?activation{_FILL}{{0,15}}?{_NUM}(?:\s*{_UNIT})?",
        rf"{_NUM}\s*{_UNIT}\s+(?:ago\s+)?(?:since\s+|after\s+)?(?:the\s+)?(?:sim\s+)?(?:was\s+)?activat",
        rf"sim\s+(?:card\s+)?(?:is|was|age|aged|has\s+been)?{_FILL}{{0,10}}?{_NUM}\s*{_UNIT}\s+old",
    ]),
    ("num_sim_changes_last_30d", [
        rf"{_NUM}\s+{_PRIOR}{_CHANGES}\b",
        rf"sim\s+{_CHANGE_WORDS}[\s(]*(?:in|during|over|within|for)?\s*(?:the\s+)?(?:last|past)\s+"
        rf"(?:30\s*days?|30d|month)\)?{_FILL}{{0,12}}?{_NUM}(?!\s*(?:days?|d\b))",
        rf"sim\s+(?:was\s+)?(?:changed|swapped|replaced)\s+{_NUM}\s+times?\b",
        rf"(?:number\s+of|count\s+of|total)\s+{_CHANGES}{_FILL}{{0,40}}?{_NUM}",
    ]),
    ("previous_sim_holder_tenure_days", [
        rf"(?:holder|owner)\b{_FILL}{{0,15}}?tenure{_FILL}{{0,15}}?{_NUM}(?:\s*{_UNIT})?",
        rf"tenure{_FILL}{{0,40}}?{_NUM}(?:\s*{_UNIT})?",
        rf"{_NUM}\s*{_UNIT}\s+(?:of\s+)?(?:previous\s+)?(?:holder\s+)?tenure",
        rf"(?:previous|former|prior|old|original|earlier)\s+(?:sim\s+)?(?:holder|owner|user|customer|subscriber)"
        rf"\b{_FILL}{{0,60}}?(?:held|had|owned|kept|used|for)\b{_FILL}{{0,25}}?{_NUM}\s*{_UNIT}",
    ]),
    ("account_age_days", [
        rf"account(?:\s+is|\s+was)?\s*(?:age|aged)?{_FILL}{{0,15}}?{_NUM}\s*{_UNIT}(?:\s+old)?",
        rf"{_NUM}[\s-]*{_UNIT}[\s-]*old\s+(?:customer\s+)?account",
        rf"(?:customer|subscriber|client|user)\s+(?:for|since)\s+{_NUM}\s*{_UNIT}",
        rf"with\s+us\s+(?:for|since)\s+{_NUM}\s*{_UNIT}",
        rf"(?:opened|created|registered|joined|started)\b{_FILL}{{0,15}}?{_NUM}\s*{_UNIT}\s+ago",
        rf"account\b{_FILL}{{0,30}}?\b(?:open|opened|existed|exists|active|held|for)\b{_FILL}{{0,12}}?{_NUM}\s*{_UNIT}",
    ]),
    ("otp_and_sim_change_geo_hash_length", [
        rf"geo\s*-?hash{_FILL}{{0,50}}?{_NUM}\s*(?P<pct>%)?(?!\s*(?:km|kilomet|miles?|m\b|meters?|metres?|{_LONG_UNIT_ALT}))",
        rf"(?:location|place|area|geo)\s+(?:similarity|match|overlap){_FILL}{{0,20}}?{_NUM}\s*(?P<pct>%)?",
        rf"{_NUM}\s*(?P<pct>%)?\s+(?:geo\s*-?hash\s+|location\s+)?(?:similarity|match|overlap)",
        rf"{_NUM}\s*(?P<pct>%)?\s+similar\b",
        rf"similarity{_FILL}{{0,15}}?{_NUM}\s*(?P<pct>%)?",
    ]),
]

_QUALITATIVE_ACTIVATION = re.compile(r"\bactivated\s+(?:just\s+)?(today|yesterday|just\s+now|this\s+morning|tonight|last\s+night)\b")
_SAME_PLACE = re.compile(r"\b(?:same|identical|matching)\s+(?:location|place|city|area|geo\s*-?hash)\b")
_FAR_PLACE = re.compile(r"\b(?:completely|entirely|totally|very)\s+different\s+(?:location|place|city|area|geo\s*-?hash)\b")
_ZERO_CHANGES = re.compile(rf"\b(?:no|zero|none|not\s+any|never)\s+{_PRIOR}{_CHANGES}\b")

_ENTITIES: list[tuple[str, re.Pattern]] = [
    ("sim_type_change_flag", re.compile(r"\bsim[\s-]*type\b|\be-?sim\b|\bsim\s+card\s+type\b|\btype\s+of\s+sim\b")),
    ("imsi_change_flag", re.compile(r"\bimsi\b")),
    ("iccid_change_flag", re.compile(r"\biccid\b|\bsim\s+(?:card\s+)?serial(?:\s+number)?\b")),
    ("device_change_flag", re.compile(r"\bdevice\b|\bhandset\b|\bimei\b|\bphone\b(?!\s+number)|\bmobile\b(?!\s+number)")),
    ("ip_change_flag", re.compile(r"\bip\b(?:[\s-]*address)?")),
]
_POSITIVE = re.compile(r"\b(?:chang\w*|swap\w*|replac\w*|new|different|differ\w*|updat\w*|switch\w*|mismatch\w*|alter\w*|modif\w*|spoof\w*)\b")
_NEGATIVE = re.compile(r"\b(?:no|not|never|without|nor|neither|except|excluding)\b|n't")
_SAME = re.compile(r"\b(?:unchanged|same|identical|unaltered|consistent|matching|familiar|known|existing|original)\b")
_HARD_SPLIT = re.compile(r",|\bbut\b|\bwhile\b|\bwith\b|\bplus\b|\bas well as\b|\bthen\b|/|:|\balthough\b|\bhowever\b")
_SOFT_SPLIT = re.compile(r"\band\b|\bor\b")
_SENTENCE_SPLIT = re.compile(r"(?<!\d)\.(?!\d)|[;!?\n]")

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_NUMBER_WORD_ALT = "|".join(list(_NUMBER_WORDS) + ["hundred", "thousand"])
_NUMBER_RUN = re.compile(rf"\b(?:{_NUMBER_WORD_ALT})\b(?:[\s-]+(?:and[\s-]+)?(?:{_NUMBER_WORD_ALT})\b)*")

FIELD_HELP: dict[str, tuple[str, str, str]] = {
    "sim_swap_time_gap_minutes": ("Time between the OTP request and the SIM change", "number",
                                  "e.g. '45 minutes after the OTP' or '2 hours'"),
    "device_change_flag": ("Device changed", "yes/no", "e.g. 'device changed' or 'same device'"),
    "sim_type_change_flag": ("SIM type changed (e.g. physical SIM to eSIM)", "yes/no",
                             "e.g. 'SIM type changed' or 'same SIM type'"),
    "imsi_change_flag": ("IMSI changed", "yes/no", "e.g. 'IMSI changed' or 'IMSI unchanged'"),
    "iccid_change_flag": ("ICCID changed", "yes/no", "e.g. 'ICCID changed' or 'ICCID unchanged'"),
    "otp_and_sim_change_geo_hash_length": (
        "Location similarity between the OTP request and the SIM change (0 = different place, 1 = same place)",
        "number 0-1", "e.g. 'location similarity 0.4', '40%' or 'same location'"),
    "recent_sim_activation_days": ("How long ago the new SIM was activated", "number",
                                   "e.g. 'SIM activated 3 days ago'"),
    "num_sim_changes_last_30d": ("SIM changes in the last 30 days", "number",
                                 "e.g. '2 SIM changes in the last 30 days' or 'no other SIM changes'"),
    "previous_sim_holder_tenure_days": ("How long the previous SIM holder had the number", "number",
                                        "e.g. 'previous holder tenure 40 days' or 'owned it for 3 years'"),
    "account_age_days": ("Account age", "number", "e.g. 'account is 4 months old'"),
    "ip_change_flag": ("IP address changed", "yes/no", "e.g. 'IP changed' or 'same IP'"),
}


def field_help() -> list[dict[str, str]]:
    return [{"field": f, "label": FIELD_HELP[f][0], "kind": FIELD_HELP[f][1], "example": FIELD_HELP[f][2]}
            for f in FEATURE_COLUMNS]


def _run_value(run: str) -> int:
    total = current = 0
    for word in re.split(r"[\s-]+", run):
        if word == "and":
            continue
        if word in _NUMBER_WORDS:
            current += _NUMBER_WORDS[word]
        elif word == "hundred":
            current = max(current, 1) * 100
        elif word == "thousand":
            total += max(current, 1) * 1000
            current = 0
    return total + current


def _normalize(text: str) -> str:
    """Lower-case and rewrite number words, 'an hour', 'twice', 'percent' into plain digits/units."""
    t = text.lower().replace("’", "'").replace("‘", "'")
    t = re.sub(r"\b(\d{1,3}(?:,\d{3})+)\b", lambda m: m.group(1).replace(",", ""), t)
    t = re.sub(r"\bhalf[\s-]+(?:an?\s+)?hour\b", "30 minutes", t)
    t = re.sub(r"\b(\d+(?:\.\d+)?)\s*(?:percent|per\s*cent|pct)\b", r"\1%", t)
    t = re.sub(r"\be-sim\b", "esim", t)
    t = re.sub(r"\bonce\b", "1 time", t)
    t = re.sub(r"\btwice\b", "2 times", t)
    t = re.sub(r"\bthrice\b", "3 times", t)
    t = re.sub(r"\bone[\s-]+time\b", "onetime", t)  # not the number 1
    t = _NUMBER_RUN.sub(lambda m: str(_run_value(m.group(0))), t)
    t = re.sub(r"\b(?:an?)\s+(minute|hour|day|week|month|year)\b", r"1 \1", t)
    return " ".join(t.split())


def _polarity(clause: str, has_entity: bool) -> int | None:
    """1 = changed, 0 = not changed, None = this clause does not say."""
    if _SAME.search(clause):
        return 0
    negated = bool(_NEGATIVE.search(clause))
    if _POSITIVE.search(clause):
        return 0 if negated else 1
    if negated and has_entity:  # "... but not the ICCID", "no IP"
        return 0
    return None


def _entities_in(clause: str) -> list[str]:
    return [name for name, pattern in _ENTITIES if pattern.search(clause)]


def _extract_flags(normalized: str, notes: list[str]) -> dict[str, int]:
    flags: dict[str, int] = {}
    for sentence in _SENTENCE_SPLIT.split(normalized):
        clauses: list[str] = []
        for chunk in _HARD_SPLIT.split(sentence):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = [p.strip() for p in _SOFT_SPLIT.split(chunk) if p.strip()]
            if len(parts) > 1 and sum(_polarity(p, bool(_entities_in(p))) is not None for p in parts) >= 2:
                clauses.extend(parts)  # "imsi changed and iccid unchanged": each side has its own verdict
            else:
                clauses.append(chunk)  # "device and imsi changed": the verdict is shared
        polarities = [_polarity(c, bool(_entities_in(c))) for c in clauses]
        for index, clause in enumerate(clauses):
            entities = _entities_in(clause)
            if not entities:
                continue
            verdict = polarities[index]
            source = clause
            if verdict is None:  # a bare list item ("device, imsi changed"): borrow the next verdict
                for later in range(index + 1, len(clauses)):
                    if polarities[later] is not None:
                        verdict, source = polarities[later], clauses[later]
                        break
            if verdict is None:
                continue
            for entity in entities:
                if entity not in flags:
                    flags[entity] = verdict
                    label = FIELD_HELP[entity][0]
                    notes.append(f'{label}: {"yes" if verdict else "no"} (from "{clause[:60]}")')
    return flags


def _extract_numbers(normalized: str, notes: list[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    used: list[tuple[int, int]] = []

    def free(span: tuple[int, int]) -> bool:
        return all(span[1] <= a or span[0] >= b for a, b in used)

    for feature, patterns in _NUMERIC_PATTERNS:
        for pattern in patterns:
            hit = next((m for m in re.finditer(pattern, normalized) if free(m.span("num"))), None)
            if not hit:
                continue
            used.append(hit.span("num"))
            number = hit.group("num")
            unit = hit.groupdict().get("unit")
            if feature == "otp_and_sim_change_geo_hash_length":
                value = f"{number}%" if hit.groupdict().get("pct") else number
            else:
                value = f"{number} {unit}" if unit else number
            found[feature] = value
            snippet = " ".join(hit.group(0).split())[:70]
            notes.append(f'{FIELD_HELP[feature][0]}: {value} (from "{snippet}")')
            break

    def put(feature: str, value: str, why: str) -> None:
        if feature not in found:
            found[feature] = value
            notes.append(f"{FIELD_HELP[feature][0]}: {value} ({why})")

    if "recent_sim_activation_days" not in found:
        q = _QUALITATIVE_ACTIVATION.search(normalized)
        if q:
            put("recent_sim_activation_days", "1 day" if "yesterday" in q.group(1) or "last night" in q.group(1) else "0 days",
                f'read "{q.group(0)}"')
    if "num_sim_changes_last_30d" not in found and _ZERO_CHANGES.search(normalized):
        put("num_sim_changes_last_30d", "0", "read as none")
    if "otp_and_sim_change_geo_hash_length" not in found:
        if _FAR_PLACE.search(normalized):
            put("otp_and_sim_change_geo_hash_length", "0", 'read "completely different location" as 0')
        elif _SAME_PLACE.search(normalized):
            put("otp_and_sim_change_geo_hash_length", "1", 'read "same location" as 1')
    return found


def parse_text(text: str) -> ParseResult:
    """Free text in any wording -> one event (or a precise list of what is missing).

    Order: explicit `key: value` lines, then numbers with units, then yes/no statements.
    """
    stripped = text.strip()
    if not stripped:
        raise InputError("the prompt is empty")

    record: dict[str, Any] = {}
    notes: list[str] = []
    for line in re.split(r"[\n;]|,(?!\d)", stripped):
        if re.search(r"[:=]", line):
            key, _, value = re.split(r"([:=])", line, maxsplit=1)
            canonical = _ALIASES.get(_norm(key))
            if canonical and value.strip():
                record[canonical] = value.strip().rstrip(".")
                if canonical in FIELD_HELP:
                    notes.append(f'{FIELD_HELP[canonical][0]}: {record[canonical]} (from "{line.strip()[:60]}")')

    normalized = _normalize(stripped)
    for feature, value in _extract_numbers(normalized, notes).items():
        record.setdefault(feature, value)
    for feature, value in _extract_flags(normalized, notes).items():
        record.setdefault(feature, value)

    result = ParseResult("text")
    row = _row_from_mapping(record, "text", stripped)
    row.info.notes = notes
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
            lines = text.splitlines()
            first = lines[0] if lines else ""
            known = sum(1 for part in first.split(",") if _ALIASES.get(_norm(part)) in FEATURE_COLUMNS)
            kind = "csv" if known >= 3 and len(lines) >= 2 else "text"
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
