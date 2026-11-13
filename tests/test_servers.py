"""The decision logic behind setup.bat's server handling (tools/servers.py)."""

import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import servers  # noqa: E402


def test_fingerprint_changes_only_when_server_code_changes(tmp_path):
    (tmp_path / "api").mkdir()
    (tmp_path / "app").mkdir()
    main = tmp_path / "api" / "main.py"
    main.write_text("print(1)")
    (tmp_path / "app" / "app.py").write_text("print(2)")
    first = servers.code_fingerprint(tmp_path)
    assert servers.code_fingerprint(tmp_path) == first  # stable

    (tmp_path / "README.md").write_text("docs are not server code")
    (tmp_path / "api" / "__pycache__").mkdir()
    (tmp_path / "api" / "__pycache__" / "main.pyc").write_bytes(b"x")
    assert servers.code_fingerprint(tmp_path) == first  # caches and docs do not count

    main.write_text("print(1)  # edited, and longer")
    assert servers.code_fingerprint(tmp_path) != first


@pytest.mark.parametrize("command, kind, expected", [
    (rf'"{servers.ROOT}\safe\Scripts\python.exe" -m uvicorn api.main:app --host 127.0.0.1 --port 8000', "api", True),
    (rf'"{servers.ROOT}\safe\Scripts\python.exe" -m streamlit run app.py --server.port 8501', "ui", True),
    (r'"C:\Other\python.exe" -m uvicorn other.main:app --port 8000', "api", False),  # another project
    (r'"C:\Other\python.exe" -m http.server 8000', "api", False),  # unrelated program
    (rf'"{servers.ROOT}\safe\Scripts\python.exe" -m uvicorn api.main:app', "ui", False),  # wrong kind
    ("", "api", False),
])
def test_only_our_own_servers_are_recognised(command, kind, expected):
    assert servers.is_ours(command, kind) is expected


def test_free_port_skips_a_port_another_program_holds():
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        busy = holder.getsockname()[1]
        assert servers.port_is_free(busy) is False
        chosen = servers.free_port(busy)
        assert chosen != busy and servers.port_is_free(chosen)


def test_listener_pid_is_read_from_netstat(monkeypatch):
    netstat = (
        "  Proto  Local Address          Foreign Address        State           PID\n"
        "  TCP    127.0.0.1:8000         0.0.0.0:0              LISTENING       4321\n"
        "  TCP    127.0.0.1:80001        0.0.0.0:0              LISTENING       1111\n"
        "  TCP    127.0.0.1:8001         127.0.0.1:50000        ESTABLISHED     2222\n"
        "  TCP    0.0.0.0:8501           0.0.0.0:0              LISTENING       9876\n"
    )
    monkeypatch.setattr(servers.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=netstat))
    assert servers.pid_on_port(8000) == 4321  # exact port only: 80001 is a different port
    assert servers.pid_on_port(8501) == 9876
    assert servers.pid_on_port(8001) is None  # established connections are not listeners


def test_options_are_read_from_the_command_line():
    args = ["--api-port", "8100", "--restart", "--ui-port", "8600"]
    assert servers.option(args, "--api-port", 8000) == 8100
    assert servers.option(args, "--ui-port", 8501) == 8600
    assert servers.option([], "--api-port", 8000) == 8000


def test_unknown_command_prints_usage():
    assert servers.main([]) == 2
    assert servers.main(["bogus"]) == 2
