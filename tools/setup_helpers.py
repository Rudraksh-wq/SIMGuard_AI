"""Small helpers for setup.bat (kept in Python so the batch file stays simple and robust).

    python tools/setup_helpers.py deps-hash          print a hash of requirements + constraints
    python tools/setup_helpers.py laya-ensure DIR      download the Laya model into DIR if missing or changed (--no-update: skip check)
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


MODEL_STATE = ".simguard_model_state.json"  # cached local hashes, keyed by size + mtime


def _local_digest(file: Path, lfs: bool, cache: dict) -> str:
    """sha256 for LFS files, git blob sha1 for small files (what the Hub reports)."""
    stat = file.stat()
    key = f"{file.name}|{stat.st_size}|{stat.st_mtime_ns}|{lfs}"
    if key in cache:
        return cache[key]
    if lfs:
        digest = hashlib.sha256()
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 24), b""):
                digest.update(chunk)
    else:
        data = file.read_bytes()
        digest = hashlib.sha1(b"blob %d\0" % len(data) + data)
    cache[key] = digest.hexdigest()
    return cache[key]


def laya_ensure(folder: str, update: bool = True) -> None:
    """Make sure the fine-tuned Laya model in `folder` is present and matches the Hub version.

    Downloads only missing or changed files. Offline, an existing local copy is kept as is.
    """
    import json as _json

    path = Path(folder)
    missing = [name for name in LAYA_FILES if not (path / name).is_file()]
    if not missing and not update:
        print(f"  found at {path} (update check skipped)")
        return
    try:
        from huggingface_hub import HfApi, hf_hub_download

        info = HfApi().model_info(LAYA_REPO, files_metadata=True)
    except Exception as error:
        if not missing:
            print(f"  found at {path} (could not reach Hugging Face to check for updates)")
        else:
            print(f"  cannot download the model: {type(error).__name__}: {error}")
            print("  The app still runs with LightGBM + Isolation Forest; run setup.bat again to retry.")
        return

    state_file = path / MODEL_STATE
    try:
        cache = _json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    remote = {s.rfilename: s for s in info.siblings}
    stale = []
    for name in list(LAYA_FILES) + ["export_report.json"]:
        sibling = remote.get(name)
        if sibling is None:
            continue
        local = path / name
        if not local.is_file():
            stale.append(name)
            continue
        lfs = getattr(sibling, "lfs", None)
        want = (lfs.sha256 if lfs else sibling.blob_id) if (lfs or sibling.blob_id) else None
        if want and _local_digest(local, bool(lfs), cache) != want:
            stale.append(name)

    if not stale:
        print(f"  found at {path} - same as the Hugging Face version")
    else:
        size = sum((remote[n].size or 0) for n in stale) / 1e6
        verb = "downloading" if missing else "updating"
        print(f"  {verb} {len(stale)} file(s) from {LAYA_REPO} - {size:.0f} MB, please wait")
        try:
            for name in stale:
                hf_hub_download(LAYA_REPO, name, local_dir=str(path), revision=info.sha)
        except Exception as error:
            print(f"  download failed: {type(error).__name__}: {error}")
            print("  The app still runs with LightGBM + Isolation Forest; run setup.bat again to retry.")
            return
        print(f"  model is now at Hugging Face revision {info.sha[:10]}")
    try:
        state_file.write_text(_json.dumps(cache), encoding="utf-8")
    except OSError:
        pass


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
        laya_ensure(sys.argv[2], update="--no-update" not in sys.argv[3:])
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
