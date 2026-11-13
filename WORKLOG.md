# Worklog (every step, small to big; each step is its own git commit)

1. Env check: RTX 3050 6GB, venv torch was 2.15.0+cpu (no GPU). Installed torch 2.15.0+cu132 into D:\safe\safe; cuda.is_available() = True.
2. Decision (user-approved): Laya cannot be faithfully converted to GGUF (custom head); deliverable is ONNX. Ollama/Qwen stays for reports.
3. Gitignore: large checkpoints/ONNX excluded; logs, configs, reports, CSVs are tracked.
4. First ladder launch failed on all 3 attempts: ImportError `from laya import train` (venv had laya 0.3.6, no training module). Upgraded laya to 0.4.31 (--no-deps); torch cu132 kept; pip check clean.
5. Relaunched ladder (run 20261113_164704_a1_no_bias, CUDA, 3050). onnxruntime has no wheel for Python 3.15 (onnx verification may need another route).
6. BASELINE (untouched Laya, before any training): val acc 0.2517 loss 1.735 | test acc 0.2565 | hard_dev acc 0.553 | hard_test acc 0.550 (bal_acc 0.50 everywhere = no signal). Gate sends ~all to NOUL. Speed p50 40.9 ms, p95 46.1 ms, batched 14.9 ms/item. Logged in runs/*/baseline.json.
7. Epoch 1 of the 16-epoch ladder run was ~50 min -> est. ~13 h total. Aborted it (kept as runs/ABORTED_16ep_*), per user request to finish in 6-7 h.
8. Relaunched single attempt `a1_6ep` (same settings as a1_no_bias: no bias, enc lr 1e-5, freeze-bottom 14) with --min-epochs 4 --max-epochs 6 --patience 2 (started 17:18; est. finish ~23:00-23:30). Ladder a2/a3 not run; will only consider them if a1_6ep misses targets and time allows.
9. User asked for 10 epochs. Running job's LR schedule is fixed at 6, so: chained a continuation `a1_cont4` (4 more epochs, base = a1_6ep/final) that auto-starts when a1_6ep finishes and writes final/ (chain log: runs/chain.log). Est. a1_6ep done ~21:00, a1_cont4 done ~23:30. Total 6+4 = 10 epochs, LR schedule restarts at epoch 7.
10. User changed total from 10 to 8 epochs: continuation a1_cont4 now runs 2 epochs (--max-epochs 2) after a1_6ep (6+2=8). Tag name kept as a1_cont4 for continuity with step 9. Est. continuation ~1 h, done ~22:30.
