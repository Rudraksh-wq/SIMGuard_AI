# ⚠️ OBSOLETE: old HTML frontend

**Do not use or extend this folder.** It is the first, plain HTML/JS analyst page. It is kept
only for reference and will be removed.

The SIMGuard AI frontend is now the Streamlit analyst console in [`app/`](../app/README.md):

- `setup.bat` starts it automatically at http://127.0.0.1:8501
- manual start: `cd app` then `python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501`

Everything this page did is in the new console (System status for the Qwen link; Analyze for
prompts, files and pasted CSV / JSON; the case view for contributions, the Laya gate, the
Qwen report, what Qwen received, and latency). The API no longer serves this page at `/`.
