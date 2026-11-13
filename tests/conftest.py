import pytest


@pytest.fixture(autouse=True)
def _isolate_qwen_state(tmp_path, monkeypatch):
    """Never touch the real saved Qwen link or call the live service from tests."""
    from agents import qwen_agent

    monkeypatch.setenv("QWEN_STATE_FILE", str(tmp_path / "qwen_state.json"))
    monkeypatch.setenv("QWEN_ENDPOINT", "")
    monkeypatch.setenv("QWEN_DISCOVERY", "")
    qwen_agent.reset_state()
    yield
    qwen_agent.reset_state()
