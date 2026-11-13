# Running SIMGuard AI

```powershell
# 1. one-time: install + train the scoring engine (~10 s)
python -m pip install -r requirements.txt
python -m ml.train

# 2. start the Qwen 3.5 2B service: Kaggle notebook rudraksh12/qwen3-5, GPU T4 on.
#    Its last cell should be kaggle/qwen_service_launch_cell.py (see "Qwen link" below).
copy .env.example .env   # then set QWEN_PASSWORD and QWEN_ENDPOINT or QWEN_DISCOVERY

# 3. start the API (from SIMGuard_AI)
# Set FRONTEND_URL too if Streamlit must use a port other than 8501.
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

- In a second terminal opened in `SIMGuard_AI`, switch to `app` and start the Streamlit frontend:

```powershell
Set-Location .\app
$env:API_URL = "http://127.0.0.1:8000"
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
# open http://127.0.0.1:8501
```

- If port 8501 is occupied, choose a free Streamlit port and set `FRONTEND_URL` in the API terminal to match.
- The FastAPI root now identifies the API; use `/docs` for the API reference.
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
