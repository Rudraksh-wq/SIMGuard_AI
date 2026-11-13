# Worklog (every step, small to big; each step is its own git commit)

1. Env check: RTX 3050 6GB, venv torch was 2.15.0+cpu (no GPU). Installed torch 2.15.0+cu132 into D:\safe\safe; cuda.is_available() = True.
2. Decision (user-approved): Laya cannot be faithfully converted to GGUF (custom head); deliverable is ONNX. Ollama/Qwen stays for reports.
3. Gitignore: large checkpoints/ONNX excluded; logs, configs, reports, CSVs are tracked.
