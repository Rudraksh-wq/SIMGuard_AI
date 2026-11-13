import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from agents import qwen_agent
from agents.features_text import FEATURE_COLUMNS
from agents.schemas import AnalyzeResponse, Event, InputInfo
from api import db, mocks
from api.input_parser import InputError, parse_input
from api.pipeline import analyze, analyze_batch, laya_checkpoint, regenerate_report

MAX_UPLOAD_BYTES = 5 * 1024 * 1024


class ReviewRequest(BaseModel):
    status: Literal["confirmed", "false_alarm"]


class InputRequest(BaseModel):
    content: str
    kind: Literal["csv", "json", "text"] | None = None
    filename: str | None = None


class QwenEndpointRequest(BaseModel):
    endpoint: str | None = None
    user: str | None = None
    password: str | None = None
    save_even_if_unreachable: bool = False


class InputAnalyzeResponse(BaseModel):
    source: str
    rows_received: int
    rows_analysed: int
    errors: list[dict[str, Any]]
    notes: list[str]
    results: list[AnalyzeResponse]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    db.init_db()
    config = mocks.get_runtime_config()
    if not config.mock_models and os.getenv("LAYA_PRELOAD", "1") != "0":
        # load the Laya checkpoint + ML artifacts in the background so the first request is fast
        threading.Thread(target=_preload, daemon=True).start()
    yield


def _preload() -> None:
    try:
        warmup()
    except Exception:
        pass


app = FastAPI(title="SIMGuard AI", lifespan=lifespan)


@app.post("/analyze", response_model=AnalyzeResponse)
def post_analyze(event: Event) -> AnalyzeResponse:
    return analyze(event, InputInfo(source="api"))


def _run_input(content: str, filename: str | None, kind: str | None) -> InputAnalyzeResponse:
    try:
        parsed = parse_input(content, filename=filename, kind=kind)
    except InputError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    background = os.getenv("QWEN_BACKGROUND", "1") != "0"
    results = analyze_batch(parsed.events, background=background) if parsed.events else []
    if not results and parsed.errors:
        raise HTTPException(
            status_code=422,
            detail={"message": "no row could be analysed", "errors": parsed.errors[:20],
                    "expected_fields": list(FEATURE_COLUMNS)},
        )
    return InputAnalyzeResponse(
        source=parsed.source,
        rows_received=len(parsed.rows),
        rows_analysed=len(results),
        errors=parsed.errors[:200],
        notes=parsed.notes,
        results=results,
    )


@app.post("/analyze/input", response_model=InputAnalyzeResponse)
def post_analyze_input(request: InputRequest) -> InputAnalyzeResponse:
    """Text prompt, pasted JSON or pasted CSV."""
    return _run_input(request.content, request.filename, request.kind)


@app.post("/analyze/upload", response_model=InputAnalyzeResponse)
async def post_analyze_upload(
    file: UploadFile = File(...), kind: str | None = Form(default=None)
) -> InputAnalyzeResponse:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file is larger than 5 MB")
    try:
        content = data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise HTTPException(status_code=422, detail="file must be UTF-8 text (CSV/JSON/TXT)") from error
    return _run_input(content, file.filename, kind or None)


@app.get("/cases", response_model=list[AnalyzeResponse])
def get_cases() -> list[dict[str, Any]]:
    return db.list_cases()


@app.get("/cases/{case_id}", response_model=AnalyzeResponse)
def get_case(case_id: str) -> dict[str, Any]:
    response = db.get_case(case_id)
    if response is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return response


@app.post("/cases/{case_id}/report", response_model=AnalyzeResponse)
def post_case_report(case_id: str) -> AnalyzeResponse:
    """(Re)generate the Qwen report for one case, e.g. one over the auto-report budget."""
    case = db.get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return regenerate_report(case)


@app.post("/cases/{case_id}/review", response_model=AnalyzeResponse)
def review_case(case_id: str, review: ReviewRequest) -> dict[str, Any]:
    try:
        return db.update_status(case_id, review.status)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Case not found") from error


@app.get("/health")
def health() -> dict[str, object]:
    config = mocks.get_runtime_config()
    try:
        db.init_db()
        db_ok = True
    except Exception:
        db_ok = False

    ml: dict[str, Any]
    try:
        from ml.score import model_info

        ml = {"ok": True, **model_info()}
    except Exception as error:
        ml = {"ok": False, "error": f"{type(error).__name__}: {error}"}

    try:
        from agents import laya_agent

        laya = laya_agent.status()
    except Exception as error:
        laya = {"loaded": False, "error": str(error)}

    qwen = {"model": config.qwen_model, **qwen_agent.ping(cached=True)}
    return {
        "mock_models": config.mock_models,
        "db_ok": db_ok,
        "ml": ml,
        "laya": laya,
        "laya_checkpoint": laya_checkpoint(),
        "laya_weight": config.laya_weight,
        "laya_fraud_threshold": config.laya_fraud_threshold,
        "qwen": qwen,
        "qwen_model": config.qwen_model,
        "ollama_ok": qwen["reachable"],
        "feature_columns": list(FEATURE_COLUMNS),
    }


@app.get("/settings/qwen")
def get_qwen_settings() -> dict[str, object]:
    return {**qwen_agent.endpoint_status(), "check": qwen_agent.ping()}


@app.post("/settings/qwen")
def set_qwen_settings(request: QwenEndpointRequest) -> dict[str, object]:
    """Set the Qwen link and/or login from the frontend.

    Username / password are applied first (kept in memory only), so the link is tested with them.
    A link is saved only if Qwen answers there, unless save_even_if_unreachable is set.
    """
    if request.user is not None or request.password is not None:
        qwen_agent.set_credentials(request.user, request.password)
    if not request.endpoint or not request.endpoint.strip():
        return {**qwen_agent.endpoint_status(), "check": qwen_agent.ping()}
    url = request.endpoint.strip().rstrip("/")
    if not qwen_agent.valid_url(url):
        raise HTTPException(status_code=422, detail="endpoint must be an http(s) URL")
    check = qwen_agent.ping(endpoint=url)
    if not check["reachable"] and not request.save_even_if_unreachable:
        raise HTTPException(status_code=422, detail=f"Qwen did not answer at {url}: {check['detail']}")
    qwen_agent.set_override(url)
    return {**qwen_agent.endpoint_status(), "check": check}


@app.get("/qwen/status")
def qwen_status() -> dict[str, object]:
    """Is Qwen answering, at which link, and whose login is used (for the frontend panel).

    The Kaggle notebook is started by hand in the Kaggle editor; it shuts itself down when idle.
    """
    check = qwen_agent.ping(cached=True)
    return {
        "reachable": check["reachable"],
        "endpoint": check.get("endpoint"),
        "detail": check.get("detail"),
        **{k: v for k, v in qwen_agent.endpoint_status().items() if k in ("user", "password_set", "credentials_from", "discovered")},
    }


@app.delete("/settings/qwen")
def clear_qwen_settings() -> dict[str, object]:
    qwen_agent.set_override(None)
    return {**qwen_agent.endpoint_status(), "check": qwen_agent.ping()}


@app.post("/warmup")
def warmup() -> dict[str, object]:
    """Load ML artifacts and the Laya checkpoint now instead of on the first request."""
    from agents import laya_agent
    from ml.score import model_info

    model_info()
    try:
        laya_agent._ensure_loaded()
    except Exception:
        pass
    return laya_agent.status()


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "service": "SIMGuard AI API",
        "docs": "/docs",
        "frontend": os.getenv("FRONTEND_URL", "http://127.0.0.1:8501"),
    }
