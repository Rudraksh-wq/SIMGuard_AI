"""Start, verify, stop and report on the SIMGuard API and the Streamlit analyst frontend.

Used by setup.bat (`python tools/setup_helpers.py servers <start|verify|stop|status>`).

Why this exists instead of two `start` lines in the batch file:
- the servers run detached, with their output in logs/, so an error is never lost with a closed window
- a server that is already ours is reused; stale ones (code changed since they started) are replaced
- another program on port 8000 / 8501 is never mistaken for us: a free port is picked instead
- `verify` really exercises everything (engine, Laya, text reading, the frontend script) before
  the browser opens

Human-readable messages go to stderr; `KEY=value` lines for the batch file go to stdout.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = ROOT / "data" / ".servers.json"
LOGS = ROOT / "logs"
PY = sys.executable
SAMPLE_TEXT = ("SIM was swapped 45 minutes after an OTP. Device and IMSI changed, ICCID unchanged, same SIM type, "
               "same IP. Location similarity 0.12. SIM activated 3 days ago, 2 SIM changes in the last 30 days. "
               "Previous holder tenure 40 days, account age 120 days.")
STARTUP_TIMEOUT_S = 150
LOG_LIMIT_BYTES = 2_000_000


def say(message: str = "") -> None:
    print(message, file=sys.stderr, flush=True)


# --- state -----------------------------------------------------------------------------------
def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def code_fingerprint(root: Path = ROOT) -> str:
    """Changes whenever the server code or the trained scoring models change (git pull, local edits)."""
    digest = hashlib.sha256()
    for folder in ("api", "agents", "ml", "app"):
        base = root / folder
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".joblib", ".json", ".toml", ".html"}:
                info = path.stat()
                digest.update(f"{path.relative_to(root)}|{info.st_size}|{info.st_mtime_ns}\n".encode())
    return digest.hexdigest()[:16]


# --- processes and ports ---------------------------------------------------------------------
def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    return bool(re.search(rf"\b{pid}\b", out))


def pid_on_port(port: int) -> int | None:
    out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[3].upper() == "LISTENING" and parts[1].rsplit(":", 1)[-1] == str(port):
            return int(parts[4])
    return None


def command_line(pid: int) -> str:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"],
        capture_output=True, text=True, timeout=30,
    ).stdout
    return out.strip()


def is_ours(command: str, kind: str) -> bool:
    """Is this process command line one of our servers (not some other program)?"""
    text = command.lower().replace("/", "\\")
    markers = ("uvicorn", "api.main") if kind == "api" else ("streamlit", "app.py")
    where = (str(ROOT).lower(), str(Path(PY).parent.parent).lower())
    return all(m in text for m in markers) and any(w in text for w in where)


def ours_on_port(port: int, kind: str) -> int | None:
    pid = pid_on_port(port)
    if pid and is_ours(command_line(pid), kind):
        return pid
    return None


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def free_port(preferred: int, tries: int = 30) -> int:
    for port in range(preferred, preferred + tries):
        if port_is_free(port):
            return port
    raise RuntimeError(f"no free port found from {preferred} to {preferred + tries - 1}")


def kill_tree(pid: int) -> None:
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True)


# --- health checks ---------------------------------------------------------------------------
def api_ok(port: int, timeout: float = 4) -> bool:
    try:
        reply = requests.get(f"http://127.0.0.1:{port}/", timeout=timeout)
        return reply.status_code == 200 and reply.json().get("service") == "SIMGuard AI API"
    except (requests.RequestException, ValueError):
        return False


def ui_ok(port: int, timeout: float = 4) -> bool:
    try:
        return requests.get(f"http://127.0.0.1:{port}/_stcore/health", timeout=timeout).text.strip() == "ok"
    except requests.RequestException:
        return False


def log_tail(name: str, lines: int = 25) -> str:
    try:
        text = (LOGS / name).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(no log)"
    return "\n".join("    " + line for line in text[-lines:])


# --- starting --------------------------------------------------------------------------------
def spawn(args: list[str], log_name: str, env: dict, cwd: Path) -> subprocess.Popen:
    LOGS.mkdir(exist_ok=True)
    path = LOGS / log_name
    if path.exists() and path.stat().st_size > LOG_LIMIT_BYTES:
        path.write_bytes(b"")
    log = open(path, "ab")
    log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} start: {' '.join(args)}\n".encode())
    log.flush()
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(args, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, creationflags=flags, close_fds=True)


def wait_until(check, proc: subprocess.Popen, seconds: float) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if proc.poll() is not None:
            return False  # it exited: no point waiting
        if check():
            return True
        time.sleep(1)
    return False


def stop_all(state: dict, extra_ports: tuple[int, ...] = ()) -> None:
    for key in ("api", "ui"):
        pid = (state.get(key) or {}).get("pid")
        if pid and pid_alive(pid):
            kill_tree(pid)
    for key, kind in (("api", "api"), ("ui", "ui")):
        ports = {(state.get(key) or {}).get("port"), *extra_ports}
        for port in filter(None, ports):
            pid = ours_on_port(int(port), kind)
            if pid:
                kill_tree(pid)
    deadline = time.time() + 12
    ports = [p for p in ((state.get("api") or {}).get("port"), (state.get("ui") or {}).get("port"), *extra_ports) if p]
    while time.time() < deadline and any(not port_is_free(int(p)) for p in ports):
        time.sleep(0.5)


def cmd_stop(args: list[str]) -> int:
    state = load_state()
    extra = tuple(int(p) for p in args if p.isdigit())
    if not state and not extra:
        say("  nothing to stop (no servers were started by setup.bat)")
        return 0
    stop_all(state, extra)
    STATE_FILE.unlink(missing_ok=True)
    say("  SIMGuard AI stopped")
    return 0


def option(args: list[str], name: str, default: int) -> int:
    return int(args[args.index(name) + 1]) if name in args else default


def cmd_start(args: list[str]) -> int:
    api_port, ui_port = option(args, "--api-port", 8000), option(args, "--ui-port", 8501)
    state = load_state()
    fingerprint = code_fingerprint()
    if "--restart" in args:
        say("  restarting: stopping the running servers")
        stop_all(state, (api_port, ui_port))
        state = {}
    elif state and state.get("fingerprint") != fingerprint:
        say("  the code changed since the servers started: restarting them with the new code")
        stop_all(state, (api_port, ui_port))
        state = {}

    # ---- API
    api_state = state.get("api") or {}
    if api_state.get("pid") and pid_alive(api_state["pid"]) and api_ok(api_state["port"]):
        api_port = api_state["port"]
        say(f"  API already running on port {api_port}")
    elif api_ok(api_port) and ours_on_port(api_port, "api"):
        say(f"  API already running on port {api_port} (started outside setup.bat; use --restart to replace it)")
    else:
        chosen = free_port(api_port)
        if chosen != api_port:
            say(f"  port {api_port} is used by another program: starting the API on port {chosen} instead")
        api_port = chosen
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "FRONTEND_URL": f"http://127.0.0.1:{ui_port}"}
        proc = spawn([PY, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(api_port)], "api.log", env, ROOT)
        say(f"  starting the API on port {api_port} ...")
        if not wait_until(lambda: api_ok(api_port), proc, STARTUP_TIMEOUT_S):
            say("  THE API DID NOT START. Last lines of logs\\api.log:")
            say(log_tail("api.log"))
            if proc.poll() is None:
                kill_tree(proc.pid)
            return 1
        api_state = {"pid": proc.pid, "port": api_port}
        state["api"] = api_state
        state["fingerprint"] = fingerprint
        save_state(state)
        say("  API is up")

    # ---- frontend
    ui_state = state.get("ui") or {}
    if ui_state.get("pid") and pid_alive(ui_state["pid"]) and ui_ok(ui_state["port"]) \
            and ui_state.get("api_port") == api_port:
        ui_port = ui_state["port"]
        say(f"  frontend already running on port {ui_port}")
    else:
        if ui_state.get("pid") and pid_alive(ui_state["pid"]):
            kill_tree(ui_state["pid"])  # it points at a different API port: replace it
        if ui_ok(ui_port) and ours_on_port(ui_port, "ui") and not ui_state:
            say(f"  a frontend of this project is already on port {ui_port} (started outside setup.bat)")
        else:
            chosen = free_port(ui_port)
            if chosen != ui_port:
                say(f"  port {ui_port} is used by another program: starting the frontend on port {chosen} instead")
            ui_port = chosen
            env = {**os.environ, "API_URL": f"http://127.0.0.1:{api_port}", "PYTHONUNBUFFERED": "1"}
            proc = spawn([PY, "-m", "streamlit", "run", "app.py", "--server.address", "127.0.0.1",
                          "--server.port", str(ui_port), "--server.headless", "true"], "frontend.log", env, ROOT / "app")
            say(f"  starting the frontend on port {ui_port} ...")
            if not wait_until(lambda: ui_ok(ui_port), proc, STARTUP_TIMEOUT_S):
                say("  THE FRONTEND DID NOT START. Last lines of logs\\frontend.log:")
                say(log_tail("frontend.log"))
                if proc.poll() is None:
                    kill_tree(proc.pid)
                return 1
            state["ui"] = {"pid": proc.pid, "port": ui_port, "api_port": api_port}
            save_state(state)
            say("  frontend is up")
    print(f"API_URL=http://127.0.0.1:{api_port}")
    print(f"UI_URL=http://127.0.0.1:{ui_port}")
    return 0


# --- verifying -------------------------------------------------------------------------------
def check(label: str, ok: bool, detail: str = "", core: bool = True) -> bool:
    mark = "ok " if ok else ("FAIL" if core else "warn")
    say(f"  [{mark}] {label}" + (f": {detail}" if detail else ""))
    return ok or not core


def cmd_verify(args: list[str]) -> int:
    state = load_state()
    api_port = option(args, "--api-port", (state.get("api") or {}).get("port", 8000))
    ui_port = option(args, "--ui-port", (state.get("ui") or {}).get("port", 8501))
    base = f"http://127.0.0.1:{api_port}"
    good = True

    good &= check("API answers", api_ok(api_port), base)
    if not api_ok(api_port):
        return 1
    try:
        health = requests.get(f"{base}/health", timeout=90).json()
    except (requests.RequestException, ValueError) as error:
        good &= check("health report", False, str(error))
        return 1
    ml = health.get("ml", {})
    good &= check("LightGBM + Isolation Forest", bool(ml.get("ok")),
                  f"{ml.get('lgbm_trees')} + {ml.get('iforest_trees')} trees" if ml.get("ok") else str(ml.get("error")))
    good &= check("case database", bool(health.get("db_ok")))

    # Laya loads in the background after the API starts: wait for the final answer
    say("  waiting for Laya to load (first start takes about 10-30 seconds) ...")
    laya = health.get("laya", {})
    end = time.time() + 150
    while not laya.get("loaded") and not laya.get("error") and time.time() < end:
        time.sleep(3)
        try:
            laya = requests.get(f"{base}/health", timeout=90).json().get("laya", {})
        except requests.RequestException:
            pass
    check("Laya fine-tuned model", bool(laya.get("loaded")),
          str(laya.get("checkpoint")) if laya.get("loaded") else (laya.get("error") or "still loading")[:150],
          core=False)

    qwen = health.get("qwen", {})
    check("Qwen 3.5 2B reports", bool(qwen.get("reachable")),
          str(qwen.get("endpoint")) if qwen.get("reachable") else "offline: reports use the template until the Kaggle notebook runs",
          core=False)

    try:
        parsed = requests.post(f"{base}/parse", json={"content": SAMPLE_TEXT, "kind": "text"}, timeout=30).json()
        row = parsed["rows"][0]
        good &= check("reading free text", bool(row["ok"]), f"{len(row['found'])} of 11 values understood")
    except (requests.RequestException, KeyError, ValueError) as error:
        good &= check("reading free text", False, str(error))

    good &= check("frontend answers", ui_ok(ui_port), f"http://127.0.0.1:{ui_port}")
    if ui_ok(ui_port):
        good &= smoke_run_frontend(api_port)
    say("")
    say("  everything is working" if good else "  SOMETHING IS NOT WORKING: see the [FAIL] lines above and the logs folder")
    return 0 if good else 1


def smoke_run_frontend(api_port: int) -> bool:
    """Run every page of app/app.py headless against the live API: catches script errors the
    health endpoint cannot see. Read-only: it saves no cases."""
    os.environ["API_URL"] = f"http://127.0.0.1:{api_port}"
    sys.path.insert(0, str(ROOT / "app"))
    try:
        import logging

        from streamlit.testing.v1 import AppTest

        for name in list(logging.root.manager.loggerDict):  # "missing ScriptRunContext ... bare mode" noise
            if name.startswith("streamlit"):
                logging.getLogger(name).setLevel(logging.ERROR)
        problems = []
        for page in ("Overview", "Cases", "Analyze", "Data & metrics", "System status"):
            at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=90)
            at.session_state["page"] = page
            at.session_state["top_workspace"] = page
            at.run()
            if at.exception:
                problems.append(f"{page}: {at.exception[0].value[:100]}")
        return check("every frontend page renders", not problems, "; ".join(problems))
    except Exception as error:  # a broken import must be reported, not crash the checker
        return check("every frontend page renders", False, f"{type(error).__name__}: {error}"[:200])


def cmd_status(args: list[str]) -> int:
    state = load_state()
    if not state:
        say("  no servers were started by setup.bat")
        return 0
    for key, label, alive in (("api", "API", api_ok), ("ui", "frontend", ui_ok)):
        info = state.get(key) or {}
        port = info.get("port")
        running = bool(port) and pid_alive(info.get("pid")) and alive(port)
        say(f"  {label}: {'running' if running else 'NOT running'}" + (f" on http://127.0.0.1:{port}" if port else ""))
    return 0


def main(argv: list[str]) -> int:
    commands = {"start": cmd_start, "stop": cmd_stop, "verify": cmd_verify, "status": cmd_status}
    if not argv or argv[0] not in commands:
        say("usage: servers <start|verify|stop|status> [--api-port N] [--ui-port N] [--restart]")
        return 2
    return commands[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
