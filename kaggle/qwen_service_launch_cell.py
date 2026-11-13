# ==========================================================================================
# SIMGuard Qwen report service: launch cell. Replaces cell 3 of rudraksh12/qwen3-5.
# (Cells 0-2, which install packages, load Qwen 3.5 2B and define generate_report, stay as they are.)
#
# Start it from the Kaggle editor when you need reports: Edit > Save Version > Save & Run All.
# (Kaggle only gives secrets to runs started from the editor, not to runs started through its API.)
# It shuts itself down after QWEN_IDLE_MINUTES (default 10) without a report request, so it never
# burns GPU hours idle; start it again the same way next time.
#
# Kaggle secrets (notebook editor > Add-ons > Secrets, each one attached to this notebook):
#   QWEN_USER         API username (optional; API_USER also works; default "simguard")
#   QWEN_PASSWORD     API password. Required. Keep it out of the code: this notebook is public.
#   QWEN_IDLE_MINUTES optional, minutes without a request before shutting down (default 10)
#   NGROK_AUTHTOKEN   optional, from https://dashboard.ngrok.com (free account)
#   NGROK_DOMAIN      optional, your free static domain, e.g. something.ngrok-free.app
#   SIMGUARD_TOPIC    optional; defaults to the topic in the backend .env (QWEN_DISCOVERY)
#
# Link modes:
#   - NGROK_AUTHTOKEN + NGROK_DOMAIN set: the link is the same every session.
#   - Otherwise: a temporary gradio.live link, published to ntfy.sh/<topic> on start and every
#     5 minutes; the backend (QWEN_DISCOVERY=ntfy:<topic>) picks it up automatically. On
#     shutdown "offline" is published so the backend stops trying the dead link at once.
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


API_USER = secret("QWEN_USER") or secret("API_USER") or "simguard"
API_PASSWORD = secret("QWEN_PASSWORD")
if not API_PASSWORD:
    raise RuntimeError("Add a Kaggle secret named QWEN_PASSWORD (Add-ons > Secrets) and attach it to this notebook.")
try:
    IDLE_MINUTES = float(secret("QWEN_IDLE_MINUTES") or 10)
except ValueError:
    IDLE_MINUTES = 10.0
NGROK_AUTHTOKEN = secret("NGROK_AUTHTOKEN")
NGROK_DOMAIN = secret("NGROK_DOMAIN")
# The topic only reveals the link (which still needs the password), so a default is fine here.
# It must match QWEN_DISCOVERY=ntfy:<topic> in the backend .env.
TOPIC = secret("SIMGUARD_TOPIC") or "simguard-2511bfdb6ca0500b"
PORT = 7860

# --- idle tracking: every request (start and end) counts as activity --------------------------
_activity = {"last": time.time(), "busy": 0}
_activity_lock = threading.Lock()
_generate_report = generate_report  # defined in cell 2


def generate_report_tracked(payload_json):
    with _activity_lock:
        _activity["busy"] += 1
        _activity["last"] = time.time()
    try:
        return _generate_report(payload_json)
    finally:
        with _activity_lock:
            _activity["busy"] -= 1
            _activity["last"] = time.time()


if "demo" in globals():
    demo.close()
with gr.Blocks(title="SIMGuard Qwen Report Service", analytics_enabled=False) as demo:
    gr.Markdown("# SIMGuard reports\nQwen 3.5 2B summarizes your detector output. Use synthetic or de-identified data.")
    request = gr.Textbox(label="Laya result (JSON)", value=example, lines=8)
    response = gr.JSON(label="Report")
    submit = gr.Button("Generate report")
    submit.click(generate_report_tracked, inputs=request, outputs=response,
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


def publish(message):
    if not TOPIC:
        return
    try:
        requests.post(f"https://ntfy.sh/{TOPIC}", data=message.encode(),
                      headers={"Title": "SIMGuard Qwen endpoint"}, timeout=10)
    except Exception as error:
        print("could not publish:", error)


_activity["last"] = time.time()  # the idle clock starts when the link is live
publish(ENDPOINT)
print("ENDPOINT:", ENDPOINT, "(fixed ngrok domain)" if use_ngrok else "(temporary gradio.live link)")
print("USERNAME:", API_USER, "(from a Kaggle secret)" if (secret("QWEN_USER") or secret("API_USER")) else "(default)")
print("PASSWORD: from the QWEN_PASSWORD secret (not printed)")
print("DISCOVERY:", f"ntfy:{TOPIC}" if TOPIC else "off")
print(f"IDLE SHUTDOWN: after {IDLE_MINUTES:g} minutes without a report request")

# Stay up while requests keep coming; shut down after IDLE_MINUTES of silence.
last_publish = time.time()
while True:
    time.sleep(20)
    with _activity_lock:
        idle = 0 if _activity["busy"] else time.time() - _activity["last"]
    if idle >= IDLE_MINUTES * 60:
        print(f"idle for {idle / 60:.1f} minutes - shutting down")
        publish("offline")
        demo.close()
        break
    if time.time() - last_publish >= 300:  # keep the link fresh on the discovery topic
        publish(ENDPOINT)
        last_publish = time.time()
print("Qwen report service stopped. The SIMGuard backend can start it again on demand.")
