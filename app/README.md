# SIMGuard AI analyst UI

`app.py` is the SIMGuard analyst frontend, located inside this backend folder. It talks to the sibling FastAPI service; keep both services bound to localhost.

## Run locally

Install once from `SIMGuard_AI`:

```powershell
python -m pip install -r requirements.txt
```

Terminal 1 — from `SIMGuard_AI`, start the API:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Terminal 2 — also from `SIMGuard_AI`, start the `app.py` frontend:

```powershell
Set-Location .\app
$env:API_URL = "http://127.0.0.1:8000"
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501`. The FastAPI root at `http://127.0.0.1:8000` is API information; `/docs` shows the API documentation. The app-local `.streamlit/config.toml` disables Streamlit usage statistics and sets the gray/silver/yellow theme.
If port 8501 is occupied, choose another Streamlit port and set `FRONTEND_URL` in the API terminal to that same URL.

## API contract

The UI uses `GET /health`, `GET /cases`, `GET /cases/{id}`, `POST /analyze/input` (guided form, typed descriptions, pasted CSV / JSON), `POST /analyze/upload` (CSV, TSV, JSON, JSONL, TXT files), `POST /cases/{id}/report` (ask Qwen to write or rewrite a report), `POST /cases/{id}/review`, and `GET/POST/DELETE /settings/qwen` (the Kaggle Qwen link). Qwen reports for flagged cases arrive in the background: the case shows `report_source: "pending"` with the template report, and the page polls until the Qwen report replaces it. All inputs use the same 11 model features: `sim_swap_time_gap_minutes`, `device_change_flag`, `sim_type_change_flag`, `imsi_change_flag`, `iccid_change_flag`, `otp_and_sim_change_geo_hash_length`, `recent_sim_activation_days`, `num_sim_changes_last_30d`, `previous_sim_holder_tenure_days`, `account_age_days`, and `ip_change_flag`.

CSV uploads are parsed and analyzed by the API (up to 2,000 rows, 5 MB). Invalid rows are reported without preventing valid rows from being analyzed. If an `msisdn_hash` or `msisdn` column is present, values must use generated `SYNTH-` tokens. Use synthetic data only.

## Privacy, tracking, and DPDP readiness

- Use synthetic demo data only. The guided form generates a synthetic subscriber token; it does not accept a real subscriber ID or hash.
- The API stores the full event and result in SQLite (`DB_PATH`, default `simguard.db`). There is no automatic retention or delete flow. Do not use real-person data.
- The API has no authentication. Keep the services bound to localhost and do not expose them publicly.
- No Google Analytics, ad pixels, or iframe embeds are included. Streamlit usage statistics are disabled by the app-local config. Note: the console CSS loads its fonts from Google Fonts, so opening the page contacts fonts.googleapis.com.
- Qwen 3.5 2B runs in the operator's Kaggle notebook, not on this machine. For flagged / human-review cases and on request, the API sends it the verdict, the 11 features, model scores and contributions, evidence, non-label input context and typed prompt text (never the dataset label). The API also reads a public ntfy.sh topic to find the notebook's current link (only the link is published there).
- A linkable hash can still be personal data. The prototype has no named operator/contact, rights or grievance channel, or deletion workflow, so it is not presented as DPDP-compliant and is not ready for real personal data or public use.
- The Digital Personal Data Protection Act, 2023 is being brought into force in stages. As of 14 November 2026, section 6(9), including a business contact for questions about processing, has commenced; the Act’s main notice/consent obligations are scheduled for 13 May 2027. Review the official [Act](https://www.meity.gov.in/static/uploads/2024/02/Digital-Personal-Data-Protection-Act-2023.pdf) and [Rules/commencement notification](https://www.meity.gov.in/static/uploads/2025/11/53450e6e5dc0bfa85ebd78686cadad39.pdf) and obtain qualified legal review before deployment.

The footer’s “Privacy & prototype use” disclosure gives analysts the same prototype limitations. It is a privacy notice and use restriction, not a complete public privacy policy or binding terms contract; the repository does not identify the service operator or provide a rights-request contact.

## Applicability notes

The app is a local, single-page Streamlit dashboard with top navigation, not a public SEO website. Custom domain, page slugs, custom 404, canonical tags, sitemap, robots.txt, llms.txt, social-share image, structured/local-business schema, and breadcrumb routes are deployment/site features and are not applicable here. Streamlit provides the browser title and favicon via `st.set_page_config`; the page has one H1. There are no content images to add alt text to; decorative glyphs are marked hidden from assistive technology. There is no Vite build, custom JavaScript bundle, or production source map in this app.
