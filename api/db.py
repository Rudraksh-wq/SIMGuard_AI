import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agents.schemas import AnalyzeResponse
from api.mocks import utc_now_iso


def _db_path() -> str:
    load_dotenv()
    return os.getenv("DB_PATH", "simguard.db")


def _connect() -> sqlite3.Connection:
    path = _db_path()
    if path != ":memory:":
        path = str(Path(path).expanduser())
        Path(path).resolve().parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                final_score REAL NOT NULL,
                status TEXT NOT NULL,
                response_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY,
                case_id TEXT NOT NULL,
                action TEXT NOT NULL,
                actor TEXT NOT NULL,
                ts TEXT NOT NULL,
                detail TEXT NOT NULL
            )
            """
        )


def _response_json(resp: AnalyzeResponse) -> str:
    if hasattr(resp, "model_dump_json"):
        return resp.model_dump_json()
    return resp.json()


def save_case(resp: AnalyzeResponse) -> None:
    response_json = _response_json(resp)
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO cases (case_id, created_at, final_score, status, response_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                resp.case_id,
                resp.created_at,
                resp.final.score,
                resp.status,
                response_json,
            ),
        )
        connection.execute(
            """
            INSERT INTO audit_log (case_id, action, actor, ts, detail)
            VALUES (?, ?, ?, ?, ?)
            """,
            (resp.case_id, "analyze", "system", utc_now_iso(), "case created"),
        )


def _decode_case(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return json.loads(row["response_json"]) if row is not None else None


def get_case(case_id: str) -> dict[str, Any] | None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT response_json FROM cases WHERE case_id = ?", (case_id,)
        ).fetchone()
    return _decode_case(row)


def list_cases() -> list[dict[str, Any]]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT response_json FROM cases ORDER BY final_score DESC, created_at DESC"
        ).fetchall()
    return [json.loads(row["response_json"]) for row in rows]


def update_status(case_id: str, status: str) -> dict[str, Any]:
    if status not in {"confirmed", "false_alarm"}:
        raise ValueError("status must be 'confirmed' or 'false_alarm'")

    with _connect() as connection:
        cursor = connection.execute(
            "UPDATE cases SET status = ? WHERE case_id = ?", (status, case_id)
        )
        if cursor.rowcount == 0:
            raise KeyError(case_id)
        row = connection.execute(
            "SELECT response_json FROM cases WHERE case_id = ?", (case_id,)
        ).fetchone()
        response = json.loads(row["response_json"])
        response["status"] = status
        response_json = json.dumps(response, separators=(",", ":"))
        connection.execute(
            "UPDATE cases SET response_json = ? WHERE case_id = ?",
            (response_json, case_id),
        )
        connection.execute(
            """
            INSERT INTO audit_log (case_id, action, actor, ts, detail)
            VALUES (?, ?, ?, ?, ?)
            """,
            (case_id, "review", "analyst", utc_now_iso(), status),
        )
    return response
