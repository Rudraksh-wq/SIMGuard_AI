"""Small helpers for setup.bat (kept in Python so the batch file stays simple and robust).

    python tools/setup_helpers.py deps-hash          print a hash of requirements + constraints
    python tools/setup_helpers.py laya-ensure DIR      download the fine-tuned Laya model into DIR if missing
    python tools/setup_helpers.py torch-info        print the installed torch build and device
    python tools/setup_helpers.py ensure-env         create .env from .env.example if missing
    python tools/setup_helpers.py wait-health PORT   wait until the API answers (max 120 s)
    python tools/setup_helpers.py wait-url URL       wait until URL answers 200 (max 120 s)
    python tools/setup_helpers.py status PORT       print a one-screen status of the running app
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DISCOVERY_TOPIC = "ntfy:simguard-2511bfdb6ca0500b"


def deps_hash() -> None:
    digest = hashlib.sha256()
    for name in ("requirements.txt", "constraints.txt"):
        digest.update((ROOT / name).read_bytes())
    digest.update(sys.version.encode())
    print(digest.hexdigest())


def _set(lines: list[str], key: str, value: str) -> list[str]:
    for index, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[index] = f"{key}={value}"
            return lines
    return lines + [f"{key}={value}"]


def ensure_env() -> None:
    env_file = ROOT / ".env"
    if env_file.exists():
        print("  .env already exists - kept as is")
        return
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    lines = _set(lines, "QWEN_ENDPOINT", os.getenv("QWEN_ENDPOINT", ""))
    lines = _set(lines, "QWEN_DISCOVERY", os.getenv("QWEN_DISCOVERY", DISCOVERY_TOPIC))
    lines = _set(lines, "QWEN_PASSWORD", os.getenv("QWEN_PASSWORD", ""))
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    has_password = bool(os.getenv("QWEN_PASSWORD"))
    print("  .env created" + ("" if has_password else " (no Qwen password: reports use the template until you add QWEN_PASSWORD)"))


def wait_health(port: str) -> None:
    import requests

    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=3).status_code == 200:
                print("  server is up")
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    print("  server did not answer within 120 s - see the 'SIMGuard AI API' window")
    sys.exit(1)


def wait_url(url: str) -> None:
    import requests

    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=3).status_code == 200:
                print("  frontend is up")
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    print(f"  {url} did not answer within 120 s - see the 'SIMGuard AI frontend' window")
    sys.exit(1)


def status(port: str) -> None:
    import requests

    health = requests.get(f"http://127.0.0.1:{port}/health", timeout=60).json()
    ml = health.get("ml", {})
    laya = health.get("laya", {})
    qwen = health.get("qwen", {})
    print(f"  Scoring engine : {'ok' if ml.get('ok') else 'ERROR ' + str(ml.get('error'))}")
    print(f"  Laya           : {'loaded' if laya.get('loaded') else 'loading in background' if not laya.get('error') else 'UNAVAILABLE - ' + str(laya.get('error'))[:120]}")
    print(f"  Qwen 3.5 2B    : {'online at ' + str(qwen.get('endpoint')) if qwen.get('reachable') else 'offline (template reports) - ' + str(qwen.get('detail'))[:120]}")


LAYA_REPO = "Kyoiske/simguard-laya-fraud"
LAYA_FILES = ("model.safetensors", "gate.json", "questions.json", "rl_agent_config.json",
              "encoder/config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json")


def laya_ensure(folder: str) -> None:
    """Make sure the fine-tuned Laya model is in `folder`; download it from the Hub if not."""
    path = Path(folder)
    missing = [name for name in LAYA_FILES if not (path / name).is_file()]
    if not missing:
        print(f"  found at {path}")
        return
    print(f"  downloading {LAYA_REPO} from Hugging Face - about 850 MB, please wait")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=LAYA_REPO, local_dir=str(path), allow_patterns=list(LAYA_FILES) + ["export_report.json"])
    except Exception as error:
        print(f"  download failed: {type(error).__name__}: {error}")
        print("  The app still runs with LightGBM + Isolation Forest; run setup.bat again to retry.")
        return
    still = [name for name in LAYA_FILES if not (path / name).is_file()]
    print("  downloaded" if not still else f"  still missing: {', '.join(still)}")


def py_version() -> None:
    print(f"Python {sys.version.split()[0]} at {sys.prefix}")


def torch_info() -> None:
    import torch

    print(torch.__version__, "GPU " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "deps-hash":
        deps_hash()
    elif command == "laya-ensure":
        laya_ensure(sys.argv[2])
    elif command == "py-version":
        py_version()
    elif command == "torch-info":
        torch_info()
    elif command == "ensure-env":
        ensure_env()
    elif command == "wait-health":
        wait_health(sys.argv[2])
    elif command == "wait-url":
        wait_url(sys.argv[2])
    elif command == "status":
        status(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
