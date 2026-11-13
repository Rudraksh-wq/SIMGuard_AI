"""Agent 2: Qwen 3.5 2B report writer, hosted by the Kaggle notebook `rudraksh12/qwen3-5`.

The notebook exposes a Gradio app (basic auth) with one API, `/generate_report`, that takes a
JSON *string* with a required `verdict` (flagged / not_flagged / none) and an optional
`parameters` object, and returns {verdict, report, model, generation_seconds, truncated,
requires_human_review}. It explains the supplied verdict; it never makes one.

Finding the service (a Kaggle share link is temporary). Known links are tried in this order;
the first that answers is remembered and a dead link (expired share link, restarted session)
is skipped automatically:
    1. the last link that worked in this process
    2. a link pasted in the frontend (saved in .qwen_endpoint.json, survives restarts)
    3. the link the notebook published to its discovery topic (QWEN_DISCOVERY)
    4. QWEN_ENDPOINT from .env (a fixed ngrok domain belongs here)

Config (.env):
    QWEN_ENDPOINT    fixed URL (ngrok static domain) or the current gradio.live link
    QWEN_DISCOVERY   ntfy:<topic>  (same value as the notebook's SIMGUARD_TOPIC secret)
    QWEN_USER        default "simguard"
    QWEN_PASSWORD    the notebook's API_PASSWORD
    QWEN_TIMEOUT_S   default 90

Every report goes through validate_report(). If the service is down or the report fails
validation twice, the pipeline uses the template report instead.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

API_NAME = "generate_report"
MAX_PAYLOAD_CHARS = 16000  # notebook limit
MODEL_ID = "Qwen/Qwen3.5-2B"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DISCOVERY_CACHE_S = 60.0
NTFY_BASE = "https://ntfy.sh"


class QwenError(RuntimeError):
    pass


class DeadEndpoint(QwenError):
    """The link does not answer (expired share link, stopped session, wrong URL)."""


@dataclass(frozen=True)
class QwenConfig:
    endpoint: str
    user: str
    password: str
    timeout_s: float
    discovery: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.endpoint or self.discovery or get_override().get("endpoint"))


def _clean_url(url: str | None) -> str:
    return (url or "").strip().rstrip("/")


def valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


# --- endpoint sources ------------------------------------------------------------------------
_state_lock = threading.Lock()
_active_endpoint: str | None = None
_discovered: tuple[float, str | None] = (0.0, None)


def _state_file() -> Path:
    return Path(os.getenv("QWEN_STATE_FILE") or PROJECT_ROOT / ".qwen_endpoint.json")


def get_override() -> dict[str, Any]:
    try:
        return json.loads(_state_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def set_override(url: str | None) -> None:
    """Save (or clear, with None/"") the link pasted in the frontend."""
    global _active_endpoint
    url = _clean_url(url)
    path = _state_file()
    with _state_lock:
        if not url:
            path.unlink(missing_ok=True)
        else:
            if not valid_url(url):
                raise ValueError("endpoint must be an http(s) URL")
            saved = {"endpoint": url, "saved_at": datetime.now(timezone.utc).isoformat()}
            path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        _active_endpoint = url or None


def reset_state() -> None:
    """Forget the remembered and discovered links (tests, config changes)."""
    global _active_endpoint, _discovered
    _active_endpoint = None
    _discovered = (0.0, None)


def discover(config: QwenConfig, force: bool = False) -> str | None:
    """Latest link the notebook published: `ntfy:<topic>`, or a URL whose body is the link."""
    global _discovered
    if not config.discovery:
        return None
    cached_at, cached = _discovered
    if not force and cached_at and time.monotonic() - cached_at < DISCOVERY_CACHE_S:
        return cached
    found = None
    try:
        if config.discovery.startswith("ntfy:"):
            topic = config.discovery[5:].strip()
            response = requests.get(
                f"{NTFY_BASE}/{topic}/json", params={"poll": "1", "since": "24h"}, timeout=8
            )
            newest = (-1, -1)
            for index, line in enumerate(response.text.splitlines()):
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                candidate = _clean_url(item.get("message", ""))
                stamp = (int(item.get("time") or 0), index)  # 1 s resolution: ties -> later line
                if valid_url(candidate) and stamp > newest:
                    newest, found = stamp, candidate
        else:
            lines = requests.get(config.discovery, timeout=8).text.strip().splitlines()
            candidate = _clean_url(lines[0] if lines else "")
            found = candidate if valid_url(candidate) else None
    except Exception:
        found = cached  # discovery hiccup: keep what we had
    _discovered = (time.monotonic(), found)
    return found


def candidate_endpoints(config: QwenConfig, refresh: bool = False) -> list[str]:
    ordered = [
        _active_endpoint,
        get_override().get("endpoint"),
        discover(config, force=refresh),
        config.endpoint,
    ]
    unique: list[str] = []
    for url in ordered:
        url = _clean_url(url)
        if url and url not in unique:
            unique.append(url)
    return unique


def get_config() -> QwenConfig:
    load_dotenv()
    return QwenConfig(
        endpoint=_clean_url(os.getenv("QWEN_ENDPOINT", "")),
        user=os.getenv("QWEN_USER", "simguard"),
        password=os.getenv("QWEN_PASSWORD", ""),
        timeout_s=float(os.getenv("QWEN_TIMEOUT_S", "90")),
        discovery=os.getenv("QWEN_DISCOVERY", "").strip(),
    )


# --- Gradio HTTP client ----------------------------------------------------------------------
_session_lock = threading.Lock()
_sessions: dict[tuple[str, str, str], requests.Session] = {}


def _login(endpoint: str, config: QwenConfig) -> requests.Session:
    session = requests.Session()
    if not config.password:
        return session
    try:
        response = session.post(
            f"{endpoint}/login",
            data={"username": config.user, "password": config.password},
            timeout=15,
        )
    except requests.RequestException as error:
        raise DeadEndpoint(f"{endpoint} unreachable ({type(error).__name__})") from error
    if response.status_code in (400, 401, 403):
        raise QwenError(f"Qwen login failed (HTTP {response.status_code}); check QWEN_USER / QWEN_PASSWORD")
    if response.status_code != 200:
        raise DeadEndpoint(f"{endpoint} answered HTTP {response.status_code} (link expired or stopped?)")
    return session


def _get_session(endpoint: str, config: QwenConfig, fresh: bool = False) -> requests.Session:
    key = (endpoint, config.user, config.password)
    with _session_lock:
        if fresh or key not in _sessions:
            _sessions[key] = _login(endpoint, config)
        return _sessions[key]


def _parse_sse(text: str) -> Any:
    """Return the data of the final `complete` event of a Gradio /call stream."""
    event = None
    for line in text.splitlines():
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data = line[5:].strip()
            if event == "complete":
                return json.loads(data)
            if event == "error":
                detail = data if data and data != "null" else "service returned an error"
                raise QwenError(f"Qwen service error: {detail}")
    raise QwenError("Qwen service closed the stream without a result")


def _call_once(
    session: requests.Session, endpoint: str, config: QwenConfig, payload_json: str
) -> dict[str, Any]:
    base = f"{endpoint}/gradio_api/call/{API_NAME}"
    try:
        start = session.post(base, json={"data": [payload_json]}, timeout=15)
    except requests.RequestException as error:
        raise DeadEndpoint(f"{endpoint} unreachable ({type(error).__name__})") from error
    if start.status_code in (401, 403):
        raise PermissionError("unauthorised")
    if start.status_code != 200:
        raise DeadEndpoint(f"{endpoint} answered HTTP {start.status_code} (link expired or stopped?)")
    event_id = start.json().get("event_id")
    if not event_id:
        raise QwenError("Qwen service did not return an event id")
    try:
        result = session.get(f"{base}/{event_id}", timeout=config.timeout_s)
    except requests.RequestException as error:
        raise QwenError(f"Qwen result fetch failed ({type(error).__name__})") from error
    if result.status_code != 200:
        raise QwenError(f"Qwen result fetch failed (HTTP {result.status_code})")
    data = _parse_sse(result.text)
    output = data[0] if isinstance(data, list) and data else data
    if not isinstance(output, dict) or not isinstance(output.get("report"), str):
        raise QwenError("Qwen service returned an unexpected response shape")
    return output


def _call_at(endpoint: str, config: QwenConfig, payload_json: str) -> dict[str, Any]:
    try:
        return _call_once(_get_session(endpoint, config), endpoint, config, payload_json)
    except PermissionError:
        return _call_once(_get_session(endpoint, config, fresh=True), endpoint, config, payload_json)


def call_service(payload: dict[str, Any], config: QwenConfig | None = None) -> dict[str, Any]:
    global _active_endpoint
    config = config or get_config()
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(payload_json) > MAX_PAYLOAD_CHARS:
        raise QwenError(f"payload is {len(payload_json)} chars; the service accepts {MAX_PAYLOAD_CHARS}")

    tried: list[str] = []
    dead: list[str] = []
    for refresh in (False, True):  # second pass asks the discovery topic for a fresh link
        for endpoint in candidate_endpoints(config, refresh=refresh):
            if endpoint in tried:
                continue
            tried.append(endpoint)
            try:
                output = _call_at(endpoint, config, payload_json)
            except DeadEndpoint as error:
                if _active_endpoint == endpoint:
                    _active_endpoint = None
                dead.append(str(error))
                continue
            _active_endpoint = endpoint
            return output
    if not tried:
        raise QwenError("no Qwen link known: set QWEN_ENDPOINT / QWEN_DISCOVERY or paste one in the UI")
    raise QwenError("Qwen service unreachable: " + "; ".join(dead))


def _ping_one(endpoint: str, config: QwenConfig) -> tuple[bool, str]:
    try:
        session = _get_session(endpoint, config, fresh=True)
        response = session.get(f"{endpoint}/gradio_api/info", timeout=8)
        ok = response.status_code == 200 and API_NAME in response.text
        return ok, f"HTTP {response.status_code}"
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def ping(config: QwenConfig | None = None, endpoint: str | None = None) -> dict[str, Any]:
    """Check one given link, or find (and remember) the first working known link."""
    global _active_endpoint
    config = config or get_config()
    if endpoint:
        url = _clean_url(endpoint)
        ok, detail = _ping_one(url, config)
        return {"configured": True, "reachable": ok, "endpoint": url, "detail": detail}
    tried: list[str] = []
    details: list[str] = []
    for refresh in (False, True):
        for url in candidate_endpoints(config, refresh=refresh):
            if url in tried:
                continue
            tried.append(url)
            ok, detail = _ping_one(url, config)
            if ok:
                _active_endpoint = url
                return {"configured": True, "reachable": True, "endpoint": url, "detail": detail}
            details.append(f"{url}: {detail}")
    if not tried:
        return {"configured": False, "reachable": False, "endpoint": None,
                "detail": "no Qwen link: set QWEN_ENDPOINT / QWEN_DISCOVERY or paste one in the UI"}
    if _active_endpoint in tried:
        _active_endpoint = None
    return {"configured": True, "reachable": False, "endpoint": tried[0], "detail": "; ".join(details)}


def endpoint_status(config: QwenConfig | None = None) -> dict[str, Any]:
    config = config or get_config()
    return {
        "active": _active_endpoint,
        "override": get_override().get("endpoint"),
        "override_saved_at": get_override().get("saved_at"),
        "discovery": config.discovery or None,
        "discovered": _discovered[1],
        "env_endpoint": config.endpoint or None,
        "user": config.user,
        "password_set": bool(config.password),
    }


# --- payload + validation --------------------------------------------------------------------
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
_FORBIDDEN = (
    "confirmed fraud",
    "is fraud",
    "definitely fraud",
    "account has been blocked",
    "we have blocked",
    "has been suspended",
    "we have suspended",
    "sim has been blocked",
)


def _numbers_in(value: Any, found: set[float]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        found.add(float(value))
    elif isinstance(value, str):
        for match in _NUMBER.findall(value):
            found.add(float(match))
    elif isinstance(value, dict):
        for key, item in value.items():
            _numbers_in(key, found)
            _numbers_in(item, found)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _numbers_in(item, found)


def validate_report(report: str, payload: dict[str, Any]) -> list[str]:
    """Return a list of problems; empty means the report is acceptable."""
    problems = []
    if not report or len(report.strip()) < 40:
        problems.append("report is empty or too short")
    lowered = report.lower()
    for phrase in _FORBIDDEN:
        if phrase in lowered and f"not {phrase}" not in lowered:
            problems.append(f"overclaims: '{phrase}'")
    if "<think>" in lowered:
        problems.append("contains chain-of-thought")

    allowed: set[float] = set()
    _numbers_in(payload, allowed)
    allowed |= {0.0, 1.0, 24.0, 30.0, 100.0}  # "0-1" scales, "24 h", "last 30 days", "100%"
    # list markers ("1.", "2)") are numbering, not facts
    body = re.sub(r"(?m)^\s*\d+[.)]\s", "", report)
    invented = []
    for match in _NUMBER.findall(body):
        number = float(match)
        if any(abs(number - known) <= max(0.051, abs(known) * 0.01) for known in allowed):
            continue
        if any(abs(number - round(known * 100, 1)) <= 0.051 for known in allowed if abs(known) <= 1):
            continue  # a probability written as a percentage
        invented.append(match)
    if invented:
        problems.append(f"numbers not present in the supplied data: {', '.join(sorted(set(invented))[:8])}")
    return problems


RETRY_NOTE = (
    "A previous draft of this report misquoted the data. Copy every number exactly as it "
    "appears in the JSON, do not round or shorten numbers, and do not add new numbers."
)


def generate_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Return {report, report_source, report_model, generation_seconds, ...} or raise.

    The Kaggle service decodes greedily (do_sample=False), so resending the same payload would
    return the same report. The retry therefore adds a writing note, which changes the input.
    Validation always checks against the original payload.
    """
    problems: list[str] = []
    for attempt in range(2):
        request = payload
        if attempt:
            request = {**payload, "parameters": {**payload.get("parameters", {}), "writing_note": RETRY_NOTE}}
        output = call_service(request)
        report = output["report"].strip()
        problems = validate_report(report, payload)
        if not problems:
            return {
                "report": report,
                "report_source": "qwen",
                "report_model": output.get("model", MODEL_ID),
                "generation_seconds": output.get("generation_seconds"),
                "truncated": bool(output.get("truncated", False)),
                "attempts": attempt + 1,
            }
    raise QwenError("Qwen report failed validation twice: " + "; ".join(problems))
