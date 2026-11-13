# Worklog (every step, small to big; each step is its own git commit)

1. Env check: RTX 3050 6GB, venv torch was 2.15.0+cpu (no GPU). Installed torch 2.15.0+cu132 into D:\safe\safe; cuda.is_available() = True.
2. Decision (user-approved): Laya cannot be faithfully converted to GGUF (custom head); deliverable is ONNX. Ollama/Qwen stays for reports.
3. Gitignore: large checkpoints/ONNX excluded; logs, configs, reports, CSVs are tracked.
4. First ladder launch failed on all 3 attempts: ImportError `from laya import train` (venv had laya 0.3.6, no training module). Upgraded laya to 0.4.31 (--no-deps); torch cu132 kept; pip check clean.
