# ==========================================================================================
# SIMGuard Qwen report service: launch cell. Replaces cell 3 of rudraksh12/qwen3-5.
# (Cells 0-2, which install packages, load Qwen 3.5 2B and define generate_report, stay as they are.)
#
# One-time setup, in the notebook editor under Add-ons > Secrets (attach each one to this notebook):
#   QWEN_PASSWORD    the API password. Required. Keep it out of the code: this notebook is public.
#   NGROK_AUTHTOKEN  optional, from https://dashboard.ngrok.com (free account)
#   NGROK_DOMAIN     optional, your free static domain, e.g. something.ngrok-free.app
#   SIMGUARD_TOPIC   optional; defaults to the topic in the backend .env (QWEN_DISCOVERY)
#
# Link modes:
#   - NGROK_AUTHTOKEN + NGROK_DOMAIN set: the link is the same every session. Put it in
#     QWEN_ENDPOINT in the backend .env once and never change it again.
#   - Otherwise: a temporary gradio.live link. If SIMGUARD_TOPIC is set, the link is published
#     to ntfy.sh/<topic> on start and every 10 minutes, and the backend (QWEN_DISCOVERY=ntfy:<topic>)
#     picks it up automatically. Without ngrok or a topic, paste the link in the frontend
#     (click the "Qwen" status pill).
#
# The last loop keeps this cell running, which keeps the session (and the link) alive. Use
# "Save Version > Save & Run All" to run it in the background for up to 12 h.
# ==========================================================================================
import subprocess, sys, time, threading, requests

try:
    from kaggle_secrets import UserSecretsClient
    _secrets = UserSecretsClient()
except Exception:
    _secrets = None


def secret(name):
    try:
        return _secrets.get_secret(name) if _secrets else None
    except Exception:
        return None


API_USER = "simguard"
API_PASSWORD = secret("QWEN_PASSWORD")
if not API_PASSWORD:
    raise RuntimeError("Add a Kaggle secret named QWEN_PASSWORD (Add-ons > Secrets) and attach it to this notebook.")
NGROK_AUTHTOKEN = secret("NGROK_AUTHTOKEN")
NGROK_DOMAIN = secret("NGROK_DOMAIN")
# The topic only reveals the link (which still needs the password), so a default is fine here.
# It must match QWEN_DISCOVERY=ntfy:<topic> in the backend .env.
TOPIC = secret("SIMGUARD_TOPIC") or "simguard-2511bfdb6ca0500b"
PORT = 7860

if "demo" in globals():
    demo.close()
with gr.Blocks(title="SIMGuard Qwen Report Service", analytics_enabled=False) as demo:
    gr.Markdown("# SIMGuard reports\nQwen 3.5 2B summarizes your detector output. Use synthetic or de-identified data.")
    request = gr.Textbox(label="Laya result (JSON)", value=example, lines=8)
    response = gr.JSON(label="Report")
    submit = gr.Button("Generate report")
    submit.click(generate_report, inputs=request, outputs=response,
                 api_name="generate_report", concurrency_limit=1)
demo.queue(max_size=16, default_concurrency_limit=1)

use_ngrok = bool(NGROK_AUTHTOKEN and NGROK_DOMAIN)
demo.launch(share=not use_ngrok, auth=(API_USER, API_PASSWORD), inline=False,
            server_name="127.0.0.1", server_port=PORT, show_error=False, prevent_thread_lock=True)

if use_ngrok:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyngrok>=7,<8"], check=True)
    from pyngrok import conf, ngrok
    conf.get_default().auth_token = NGROK_AUTHTOKEN
    ngrok.kill()  # drop a tunnel left over from a previous run of this cell
    ENDPOINT = ngrok.connect(PORT, "http", domain=NGROK_DOMAIN).public_url.rstrip("/")
else:
    ENDPOINT = demo.share_url.rstrip("/")


def publish():
    if not TOPIC:
        return
    try:
        requests.post(f"https://ntfy.sh/{TOPIC}", data=ENDPOINT.encode(),
                      headers={"Title": "SIMGuard Qwen endpoint"}, timeout=10)
    except Exception as error:
        print("could not publish the link:", error)


publish()
print("ENDPOINT:", ENDPOINT, "(fixed ngrok domain)" if use_ngrok else "(temporary gradio.live link)")
print("USERNAME:", API_USER)
print("PASSWORD: from the QWEN_PASSWORD secret (not printed)")
print("DISCOVERY:", f"ntfy:{TOPIC}" if TOPIC else "off (set SIMGUARD_TOPIC to enable)")
print("API: /generate_report")

# Keep the session alive and re-publish the link every 10 minutes (ntfy keeps messages ~12 h).
while True:
    time.sleep(600)
    publish()
