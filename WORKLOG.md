# Worklog (every step, small to big; each step is its own git commit)

1. Env check: RTX 3050 6GB, venv torch was 2.15.0+cpu (no GPU). Installed torch 2.15.0+cu132 into D:\safe\safe; cuda.is_available() = True.
2. Decision (user-approved): Laya cannot be faithfully converted to GGUF (custom head); deliverable is ONNX. Ollama/Qwen stays for reports.
3. Gitignore: large checkpoints/ONNX excluded; logs, configs, reports, CSVs are tracked.
4. First ladder launch failed on all 3 attempts: ImportError `from laya import train` (venv had laya 0.3.6, no training module). Upgraded laya to 0.4.31 (--no-deps); torch cu132 kept; pip check clean.
5. Relaunched ladder (run 20261113_164704_a1_no_bias, CUDA, 3050). onnxruntime has no wheel for Python 3.15 (onnx verification may need another route).
6. BASELINE (untouched Laya, before any training): val acc 0.2517 loss 1.735 | test acc 0.2565 | hard_dev acc 0.553 | hard_test acc 0.550 (bal_acc 0.50 everywhere = no signal). Gate sends ~all to NOUL. Speed p50 40.9 ms, p95 46.1 ms, batched 14.9 ms/item. Logged in runs/*/baseline.json.
