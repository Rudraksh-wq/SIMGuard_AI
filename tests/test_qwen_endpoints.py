"""Finding the Qwen service when the Kaggle link changes."""

import json

import pytest
from fastapi.testclient import TestClient

from agents import qwen_agent


class FakeResponse:
    def __init__(self, text: str):
        self.text = text


def fake_service(monkeypatch, alive: set[str], calls: list[str]):
    def call_at(endpoint, config, payload_json):
        calls.append(endpoint)
        if endpoint not in alive:
            raise qwen_agent.DeadEndpoint(f"{endpoint} answered HTTP 502")
        return {"report": f"report from {endpoint}", "model": "Qwen/Qwen3.5-2B"}

    monkeypatch.setattr(qwen_agent, "_call_at", call_at)
    monkeypatch.setattr(qwen_agent, "_ping_one", lambda url, config: (url in alive, "HTTP 200" if url in alive else "HTTP 502"))


def test_dead_saved_link_falls_through_to_env_and_is_remembered(monkeypatch):
    calls: list[str] = []
    fake_service(monkeypatch, alive={"https://new.gradio.live"}, calls=calls)
    monkeypatch.setenv("QWEN_ENDPOINT", "https://new.gradio.live")
    qwen_agent.set_override("https://old.gradio.live")

    out = qwen_agent.call_service({"verdict": "flagged"})
    assert out["report"] == "report from https://new.gradio.live"
    assert calls == ["https://old.gradio.live", "https://new.gradio.live"]

    calls.clear()
    qwen_agent.call_service({"verdict": "flagged"})
    assert calls == ["https://new.gradio.live"]  # remembered: no time wasted on the dead link


def test_discovery_topic_supplies_the_newest_link(monkeypatch):
    calls: list[str] = []
    fake_service(monkeypatch, alive={"https://b.gradio.live"}, calls=calls)
    monkeypatch.setenv("QWEN_DISCOVERY", "ntfy:simguard-test-topic")
    stream = "\n".join(json.dumps({"event": "message", "message": url})
                       for url in ("https://a.gradio.live", "https://b.gradio.live/"))
    seen = {}

    def fake_get(url, params=None, timeout=None):
        seen["url"] = url
        return FakeResponse(stream)

    monkeypatch.setattr(qwen_agent.requests, "get", fake_get)
    out = qwen_agent.call_service({"verdict": "none"})
    assert out["report"] == "report from https://b.gradio.live"
    assert seen["url"] == "https://ntfy.sh/simguard-test-topic/json"


def test_discovery_is_refreshed_after_every_known_link_dies(monkeypatch):
    calls: list[str] = []
    fake_service(monkeypatch, alive={"https://fresh.gradio.live"}, calls=calls)
    monkeypatch.setenv("QWEN_DISCOVERY", "ntfy:t")
    published = ["https://stale.gradio.live"]
    monkeypatch.setattr(
        qwen_agent.requests, "get",
        lambda url, params=None, timeout=None: FakeResponse(json.dumps({"message": published[-1]})),
    )
    qwen_agent.discover(qwen_agent.get_config())  # cached: stale link
    published.append("https://fresh.gradio.live")  # notebook restarted and re-published
    out = qwen_agent.call_service({"verdict": "flagged"})
    assert out["report"] == "report from https://fresh.gradio.live"


def test_nothing_known_gives_a_clear_error():
    with pytest.raises(qwen_agent.QwenError, match="no Qwen link known"):
        qwen_agent.call_service({"verdict": "none"})


def test_settings_api_tests_before_saving(monkeypatch, tmp_path):
    from api import main

    monkeypatch.setenv("DB_PATH", str(tmp_path / "s.db"))
    monkeypatch.setenv("MOCK_MODELS", "1")
    fake_service(monkeypatch, alive={"https://good.gradio.live"}, calls=[])
    with TestClient(main.app) as client:
        bad = client.post("/settings/qwen", json={"endpoint": "https://dead.gradio.live"})
        assert bad.status_code == 422
        assert qwen_agent.get_override() == {}

        assert client.post("/settings/qwen", json={"endpoint": "not a url"}).status_code == 422

        good = client.post("/settings/qwen", json={"endpoint": "https://good.gradio.live/"})
        assert good.status_code == 200
        assert good.json()["override"] == "https://good.gradio.live"
        assert good.json()["check"]["reachable"] is True
        assert qwen_agent.get_override()["endpoint"] == "https://good.gradio.live"

        cleared = client.delete("/settings/qwen")
        assert cleared.status_code == 200 and cleared.json()["override"] is None
