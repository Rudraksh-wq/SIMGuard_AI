from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Any, Literal

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agents.schemas import AnalyzeResponse, Event
from api import db
from api.mocks import get_runtime_config
from api.pipeline import analyze, laya_checkpoint


class ReviewRequest(BaseModel):
    status: Literal["confirmed", "false_alarm"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    db.init_db()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/analyze", response_model=AnalyzeResponse)
def post_analyze(event: Event) -> AnalyzeResponse:
    return analyze(event)


@app.get("/cases", response_model=list[AnalyzeResponse])
def get_cases() -> list[dict[str, Any]]:
    return db.list_cases()


@app.get("/cases/{case_id}", response_model=AnalyzeResponse)
def get_case(case_id: str) -> dict[str, Any]:
    response = db.get_case(case_id)
    if response is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return response


@app.post("/cases/{case_id}/review", response_model=AnalyzeResponse)
def review_case(case_id: str, review: ReviewRequest) -> dict[str, Any]:
    try:
        return db.update_status(case_id, review.status)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Case not found") from error


@app.get("/health")
def health() -> dict[str, object]:
    config = get_runtime_config()
    try:
        ollama_ok = (
            requests.get(f"{config.ollama_url}/api/tags", timeout=2).status_code == 200
        )
    except requests.RequestException:
        ollama_ok = False

    try:
        db.init_db()
        db_ok = True
    except Exception:
        db_ok = False

    return {
        "laya_checkpoint": laya_checkpoint(),
        "qwen_model": config.qwen_model,
        "ollama_ok": ollama_ok,
        "mock_models": config.mock_models,
        "laya_weight": config.laya_weight,
        "laya_fraud_threshold": config.laya_fraud_threshold,
        "db_ok": db_ok,
    }
