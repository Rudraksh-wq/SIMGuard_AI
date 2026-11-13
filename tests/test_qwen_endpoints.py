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


def test_offline_message_from_the_notebook_hides_its_dead_link(monkeypatch):
    monkeypatch.setenv("QWEN_DISCOVERY", "ntfy:t")
    stream = "\n".join([
        json.dumps({"time": 100, "message": "https://old.gradio.live"}),
        json.dumps({"time": 200, "message": "offline"}),
    ])
    monkeypatch.setattr(qwen_agent.requests, "get", lambda url, params=None, timeout=None: FakeResponse(stream))
    assert qwen_agent.discover(qwen_agent.get_config(), force=True) is None
    # a newer link after the "offline" wins again
    stream += "\n" + json.dumps({"time": 300, "message": "https://new.gradio.live"})
    assert qwen_agent.discover(qwen_agent.get_config(), force=True) == "https://new.gradio.live"


def test_frontend_login_overrides_env_and_stays_in_memory(monkeypatch, tmp_path):
    monkeypatch.setenv("QWEN_USER", "env-user")
    monkeypatch.setenv("QWEN_PASSWORD", "env-pass")
    assert qwen_agent.get_config().user == "env-user"
    qwen_agent.set_credentials("browser-user", "browser-pass")
    config = qwen_agent.get_config()
    assert (config.user, config.password) == ("browser-user", "browser-pass")
    assert qwen_agent.endpoint_status()["credentials_from"] == "frontend"
    assert not any(tmp_path.iterdir())  # nothing written to disk
    qwen_agent.set_credentials("", "")
    assert qwen_agent.get_config().password == "env-pass"


def test_settings_api_applies_login_before_testing_the_link(monkeypatch, tmp_path):
    from api import main

    monkeypatch.setenv("DB_PATH", str(tmp_path / "s.db"))
    monkeypatch.setenv("MOCK_MODELS", "1")
    seen = []

    def ping_one(url, config):
        seen.append((url, config.user, config.password))
        return url == "https://good.gradio.live" and config.password == "secret", "HTTP 200"

    monkeypatch.setattr(qwen_agent, "_ping_one", ping_one)
    with TestClient(main.app) as client:
        login_only = client.post("/settings/qwen", json={"user": "simguard", "password": "secret"})
        assert login_only.status_code == 200 and login_only.json()["credentials_from"] == "frontend"
        saved = client.post("/settings/qwen", json={"endpoint": "https://good.gradio.live", "user": "simguard", "password": "secret"})
        assert saved.status_code == 200 and saved.json()["check"]["reachable"] is True
        assert ("https://good.gradio.live", "simguard", "secret") in seen
        status = client.get("/qwen/status")
        assert status.status_code == 200
        assert status.json()["reachable"] is True and status.json()["credentials_from"] == "frontend"
        assert client.post("/qwen/start").status_code in (404, 405)  # no start webhook


def test_status_ping_is_cached_and_settings_changes_clear_it(monkeypatch):
    calls = []
    monkeypatch.setenv("QWEN_ENDPOINT", "https://slow.gradio.live")

    def ping_one(url, config):
        calls.append(url)
        return False, "HTTP 404"

    monkeypatch.setattr(qwen_agent, "_ping_one", ping_one)
    qwen_agent.ping(cached=True)
    n = len(calls)
    assert n >= 1
    qwen_agent.ping(cached=True)
    qwen_agent.ping(cached=True)
    assert len(calls) == n  # offline Qwen: polling does not re-probe every time
    qwen_agent.set_credentials("u", "p")  # a settings change must be visible immediately
    qwen_agent.ping(cached=True)
    assert len(calls) > n


def test_one_operation_asks_the_discovery_service_once(monkeypatch):
    monkeypatch.setenv("QWEN_DISCOVERY", "ntfy:t")
    monkeypatch.setattr(qwen_agent, "_ping_one", lambda url, config: (False, "HTTP 404"))
    fetches = []

    def fake_get(url, params=None, timeout=None):
        fetches.append(url)
        return FakeResponse(json.dumps({"time": 1, "message": "https://a.gradio.live"}))

    monkeypatch.setattr(qwen_agent.requests, "get", fake_get)
    qwen_agent.ping()  # pass 1 fetches; the forced second pass must not fetch again
    assert len(fetches) == 1
    qwen_agent.ping()  # a later, separate operation may refresh again (cache is 60 s, so this reuses it)
    assert len(fetches) <= 2
