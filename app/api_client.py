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


def analyze_upload(filename, content):
    """Submit a CSV file to the API's parser and batch inference endpoint."""
    if not filename.lower().endswith(".csv"):
        raise ValueError("Only CSV uploads are supported")
    return _json(
        requests.post(
            f"{API_URL}/analyze/upload",
            files={"file": (filename, content, "text/csv")},
            timeout=180,
        )
    )


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
