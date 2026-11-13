# Running SIMGuard AI

```powershell
# 1. one-time: install + train the scoring engine (~10 s)
safe\Scripts\python.exe -m pip install -r requirements.txt
safe\Scripts\python.exe -m ml.train

# 2. start the Qwen 3.5 2B service: Kaggle notebook rudraksh12/qwen3-5, GPU T4 on.
#    Its last cell should be kaggle/qwen_service_launch_cell.py (see "Qwen link" below).
copy .env.example .env   # then set QWEN_PASSWORD and QWEN_ENDPOINT or QWEN_DISCOVERY

# 3. start the API + frontend
safe\Scripts\python.exe -m uvicorn api.main:app --port 8000
# open http://127.0.0.1:8000   (demo deep link: http://127.0.0.1:8000/#sample=fraud)
```

- Without Qwen configured everything still works; reports fall back to the template.
- `MOCK_MODELS=1` runs the API without LightGBM/Laya (for UI work).
- Endpoints: `POST /analyze` (one event JSON), `POST /analyze/input` ({content, kind}),
  `POST /analyze/upload` (file), `GET /cases`, `POST /cases/{id}/report`,
  `POST /cases/{id}/review`, `GET /health`.

## Qwen link (the Kaggle link is temporary)

The backend tries, in order: the last link that worked, a link pasted in the UI (click the
"Qwen" status pill; saved in `.qwen_endpoint.json`), the link the notebook published to its
discovery topic, and `QWEN_ENDPOINT`. Dead links are skipped automatically.

- **Fixed URL (best):** free ngrok account -> Kaggle secrets `NGROK_AUTHTOKEN` + `NGROK_DOMAIN`.
  The notebook then always serves on `https://<your-domain>`; put that in `QWEN_ENDPOINT` once.
- **Automatic, no account:** Kaggle secret `SIMGUARD_TOPIC` = the topic in `QWEN_DISCOVERY`
  (`.env`, after `ntfy:`). Each new gradio.live link is published there and picked up.
- **Manual:** paste the printed ENDPOINT in the UI.

The password comes from the Kaggle secret `QWEN_PASSWORD` (the notebook is public, so it must
not be written in the code). Use "Save Version > Save & Run All" to keep the service up for
up to 12 h without the browser tab open.
