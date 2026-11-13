# Running SIMGuard AI

## One command (Windows)

Double-click `setup.bat` (or run it in a terminal). It sets up whatever is missing, starts everything,
checks that it all works, and opens http://127.0.0.1:8501:

1. pulls the latest code from GitHub (only when you have no local edits; never merges)
2. Python environment (installs Python 3.12 with uv if the PC has none) and PyTorch (GPU build when an
   NVIDIA GPU is present, otherwise CPU)
3. pinned packages, `.env`, training of LightGBM + Isolation Forest if needed
4. downloads the fine-tuned Laya model from Hugging Face if missing or changed
5. starts the API (port 8000) and the Streamlit frontend (port 8501) **in the background**, with their
   output in `logs\`
6. **verifies**: API, LightGBM + Isolation Forest, case database, Laya loaded, free-text reading, the
   frontend, and every frontend page rendering against the live API. Qwen offline is only a warning
   (reports then use the template).

It can be run any number of times:

- servers that are already running and current are reused (a re-run takes seconds)
- servers started before a `git pull` or a code edit are **restarted automatically** with the new code
- if port 8000 or 8501 is used by another program, the next free port is used; the final message shows
  the real addresses
- if a server cannot start, the last lines of its log are printed on screen

| Command | What it does |
|---|---|
| `setup.bat` | set up, start, verify, open the browser |
| `stop.bat` or `setup.bat --stop` | stop the API and frontend |
| `setup.bat --restart` | stop, then start again |
| `setup.bat --status` | show whether they are running |
| `setup.bat --no-browser` / `--no-pause` | do not open the browser / never wait for a key (for scripts) |
| `setup.bat --no-start` / `--skip-tests` / `--no-update` / `--reinstall` | set up only / skip the tests / skip the git and Hugging Face updates / rebuild the Python environment |

Logs: `logs\api.log` and `logs\frontend.log`. Exit code is 0 when everything works, 1 otherwise.
Environment variables: `SIMGUARD_PORT`, `SIMGUARD_UI_PORT`, `SIMGUARD_TORCH=cpu|cu132`, `SIMGUARD_VENV`.

## Manual steps

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
