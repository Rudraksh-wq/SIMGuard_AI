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
  `POST /parse` ({content, kind}: what was understood and what is missing, without analysing),
  `POST /analyze/upload` (file), `GET /cases`, `POST /cases/{id}/report`,
  `POST /cases/{id}/review`, `GET /health`.

## Describing an event in your own words

**Analyze → Describe in words** takes any wording: full sentences, short notes, a list, number words
("forty five minutes"), units (minutes to years), percentages, yes/no statements in any order
("device and IMSI changed but not the ICCID") and `key: value` lines. SIMGuard reads the 11 values out
of the text. If some cannot be found it says exactly which (with an example of how to state each),
shows what it did understand, and lets you fill only the gaps. It never guesses a missing value, and
it does not use an LLM for this, so it works offline. Wording it cannot read, such as "SIM for 3
days" without saying activated, is reported as missing rather than guessed.

## Qwen link (the Kaggle link is temporary)

The backend tries, in order: the last link that worked, a link pasted in the UI (click the
Streamlit console under **System status → Qwen 3.5 2B connection**; saved in `.qwen_endpoint.json`), the link the notebook published to its
discovery topic, and `QWEN_ENDPOINT`. Dead links are skipped automatically.

- **Fixed URL (best):** free ngrok account -> Kaggle secrets `NGROK_AUTHTOKEN` + `NGROK_DOMAIN`.
  The notebook then always serves on `https://<your-domain>`; put that in `QWEN_ENDPOINT` once.
- **Automatic, no account:** Kaggle secret `SIMGUARD_TOPIC` = the topic in `QWEN_DISCOVERY`
  (`.env`, after `ntfy:`). Each new gradio.live link is published there and picked up.
- **Manual:** paste the printed ENDPOINT in the UI.

The password comes from the Kaggle secret `QWEN_PASSWORD` (the notebook is public, so it must
not be written in the code). Use "Save Version > Save & Run All" to keep the service up for
up to 12 h without the browser tab open.

## Qwen login and link in the frontend

**System status → Qwen 3.5 2B on Kaggle** has a form for the notebook's link, username
(`QWEN_USER` / `API_USER` secret, default `simguard`) and password (`QWEN_PASSWORD` secret).
They are saved in this browser's local storage (tick "Remember in this browser"), so they
survive page reloads and browser restarts; on each new visit the frontend hands them to the
API, which keeps them in memory only and never writes them to disk. Use *Forget saved login &
link* to remove them from the browser and the API. They override `QWEN_USER` / `QWEN_PASSWORD`
in `.env`. The link is optional: leave it empty to use automatic discovery.

## Starting and stopping the Qwen notebook

Start it yourself from the Kaggle editor: open the notebook → Edit → Save Version → Save & Run All
(Kaggle gives secrets only to runs started from the editor, not to runs started through its API,
so there is no start button in the app). The notebook shuts itself down after 10 idle minutes
(secret `QWEN_IDLE_MINUTES` changes this) and tells the API it is offline.
