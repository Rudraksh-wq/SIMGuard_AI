"""Typed-at-the-boundary HTTP adapter for the current FastAPI contract."""

import os

import requests

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT_SECONDS = 30


def _json(response):
    response.raise_for_status()
    return response.json()


def analyze(payload):
    """POST one API Event and return the server's AnalyzeResponse unchanged."""
    if not isinstance(payload, dict):
        raise TypeError("Event payload must be an object")
    return _json(requests.post(f"{API_URL}/analyze", json=payload, timeout=TIMEOUT_SECONDS))


UPLOAD_TYPES = {".csv": "text/csv", ".tsv": "text/tab-separated-values", ".json": "application/json",
                ".jsonl": "application/json", ".txt": "text/plain"}


def _detail(response):
    """The API's own error message (FastAPI `detail`), so the UI can show what went wrong."""
    try:
        detail = response.json().get("detail")
    except ValueError:
        return response.text[:400]
    if isinstance(detail, dict):
        message = detail.get("message", "")
        errors = detail.get("errors") or []
        rows = "; ".join(f"row {e.get('row')}: {e.get('error')}" for e in errors[:5])
        return f"{message}. {rows}".strip(". ") if rows else message
    if isinstance(detail, list):
        return "; ".join(f"{'.'.join(str(p) for p in d.get('loc', []))}: {d.get('msg')}" for d in detail[:5])
    return str(detail)


class ApiError(RuntimeError):
    """A request the API rejected, carrying its readable reason."""


def _checked(response):
    if response.status_code >= 400:
        raise ApiError(f"HTTP {response.status_code}: {_detail(response)}")
    return response.json()


def analyze_upload(filename, content):
    """Submit a CSV / TSV / JSON / JSONL / TXT file to the API's parser and batch inference."""
    suffix = "." + filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if suffix not in UPLOAD_TYPES:
        raise ValueError("Upload a .csv, .tsv, .json, .jsonl or .txt file")
    return _checked(
        requests.post(
            f"{API_URL}/analyze/upload",
            files={"file": (filename, content, UPLOAD_TYPES[suffix])},
            timeout=180,
        )
    )


def analyze_input(content, kind=None):
    """Free-text prompt, pasted JSON or pasted CSV -> parsed events -> batch analysis.

    Qwen reports for flagged cases are written in the background (report_source "pending").
    """
    body = {"content": content}
    if kind:
        body["kind"] = kind
    return _checked(requests.post(f"{API_URL}/analyze/input", json=body, timeout=180))


def regenerate_report(case_id):
    """Ask Qwen 3.5 2B to (re)write the report for one case. Takes ~15-40 s."""
    return _checked(requests.post(f"{API_URL}/cases/{case_id}/report", timeout=240))


def get_qwen_settings():
    return _checked(requests.get(f"{API_URL}/settings/qwen", timeout=60))


def set_qwen_settings(endpoint=None, user=None, password=None, save_even_if_unreachable=False):
    """Give the API the Qwen link and/or login. The API tests a link before saving it unless
    save_even_if_unreachable; the login is kept in the API's memory only."""
    body = {"endpoint": endpoint or None, "user": user, "password": password,
            "save_even_if_unreachable": save_even_if_unreachable}
    return _checked(requests.post(f"{API_URL}/settings/qwen", json=body, timeout=60))


def set_qwen_endpoint(endpoint):
    """Save the Kaggle notebook's current link; the API tests it before saving."""
    return set_qwen_settings(endpoint=endpoint)


def qwen_status():
    """Whether Qwen answers, at which link, and whose login the API is using."""
    return _checked(requests.get(f"{API_URL}/qwen/status", timeout=90))


def clear_qwen_endpoint():
    return _checked(requests.delete(f"{API_URL}/settings/qwen", timeout=60))


def get_cases():
    result = _json(requests.get(f"{API_URL}/cases", timeout=TIMEOUT_SECONDS))
    if not isinstance(result, list):
        raise ValueError("The cases endpoint returned an unexpected response")
    return result


def get_case(case_id):
    return _json(requests.get(f"{API_URL}/cases/{case_id}", timeout=TIMEOUT_SECONDS))


def review_case(case_id, status):
    if status not in {"confirmed", "false_alarm"}:
        raise ValueError("Review status must be confirmed or false_alarm")
    return _json(requests.post(f"{API_URL}/cases/{case_id}/review", json={"status": status}, timeout=TIMEOUT_SECONDS))


def get_health():
    result = _json(requests.get(f"{API_URL}/health", timeout=TIMEOUT_SECONDS))
    if not isinstance(result, dict):
        raise ValueError("The health endpoint returned an unexpected response")
    return result
