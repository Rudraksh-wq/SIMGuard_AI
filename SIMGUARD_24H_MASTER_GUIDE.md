# SIMGuard AI: 24-Hour Master Build Guide

This guide replaces `SIMGUARD_BUILD_GUIDE.md` and `FINETUNE_GUIDE.md`. Save it in the repo as `docs/GUIDE.md`. It is written so you can give it to an AI coding assistant one step at a time.

**The plan is built to finish by hour 19, which leaves 5 hours of buffer.** Hackathons always run late. A finished demo with honest numbers wins over an unfinished one with big claims.

---

## 0. What changed from the two old guides, and why

| # | Change | Why it helps you win |
|---|---|---|
| 1 | **Fine-tune Laya only. Qwen stays prompt-only plus the validator.** A Qwen LoRA run is a stretch goal you start only if you are ahead at hour 12. | Qwen training data would come from your own templates, so the model would mostly learn your template. That is a lot of risk for little gain. "Our validator catches every invented fact" is a stronger and more honest claim to put in front of judges. |
| 2 | **Start the Laya fine-tune at hour 3, not hour 6.** It only needs the data generator and `features_text.py`. | GPU time runs in the background for most of the day. You get room for a second seed and a full evaluation. |
| 3 | **Write the Laya text with words next to the numbers from the start** (for example "location gap: 1200 km (very large)"). | The old guide only suggested this after weak results, which costs a full retrain. Text encoders read numbers poorly. |
| 4 | **Laya asks one question (`is_fraud`), not three.** Actions come from the tier. | Only `is_fraud` is fine-tuned. The untrained `verify` and `hold` answers would be base-model guesses shown next to a fine-tuned answer, which confuses the story. |
| 5 | **One checkpoint, one input format** for baseline, training and inference. | The old guides disagreed on both points (see section 2). If they differ, the before/after table compares two different things. |
| 6 | **The Laya threshold comes from `LAYA_FRAUD_THRESHOLD`**, chosen on `val.jsonl`. It is never hard-coded to 0.5. | A model trained at a 30% fraud rate gives inflated probabilities. A fixed 0.5 would flag too many normal events. |
| 7 | **Add per-feature LightGBM contributions** (`pred_contrib=True`, which is built into LightGBM). | Judges see why the score is high, with no extra library. It takes about 30 minutes. |
| 8 | **A "cost story": Qwen runs only on flagged events.** Show the latency of each stage on the dashboard. | This turns the architecture into a clear selling point: the cheap models check every event, and the LLM only runs on the few flagged ones. |
| 9 | **Three parallel tracks**, with a solo order as well. | The old plans assumed one person working in order. |

---

## 1. What we are building

A telecom SIM-swap fraud detection and investigation prototype.

1. **Scoring engine (LightGBM + Isolation Forest).** The only source of the numeric risk score.
2. **Laya (Agent 1).** A fine-tuned, non-generative decision model. It reads a short text version of the event and returns `p_fraud`. It is a fast gate and a second opinion. It never writes text.
3. **Evidence extractor.** Plain rules that produce the list of indicators. This is the only source of indicators.
4. **Qwen (Agent 2) through Ollama.** Called only when fraud is flagged. It writes a short report that explains the supplied score and evidence. Every report goes through `validate_report()`. If Ollama is down or the report fails validation twice, a template report is used instead.
5. **FastAPI + SQLite + Streamlit.** The analyst reviews cases and confirms or rejects each one. A human always makes the decision.

Rules that hold everywhere (in the code, the UI and the pitch):
- All metrics are on **synthetic data**. Label them that way every time.
- The prototype **recommends** a hold. A real hold needs telecom integration.
- Do not claim a component runs in the pipeline unless it does.
- Only hashed phone numbers. No call content.

```
Event ──> ml/score.py ───────────> risk_score + feature contributions ─┐
      ──> agents/laya_agent.py ──> p_fraud (fast gate, ~ms) ───────────┼─> pipeline: fraud_detected?
      ──> agents/evidence.py ────> indicators ─────────────────────────┘        │
                                                         yes ─> qwen_agent ─> validate ─> report (or template)
                                                         no  ─> low-risk result, no report
                                                 ─> SQLite ─> FastAPI ─> Streamlit dashboard
```

---

## 2. Settle these in the first hour

### 2a. Fixed decisions

| Topic | Decision |
|---|---|
| Score | `risk_score = 100 * (0.7 * lgbm_prob + 0.3 * anomaly)`, anomaly scaled to 0-1 |
| Tiers | low < 40, medium 40-70, high >= 70 |
| Gate | `fraud_detected = (p_fraud >= LAYA_FRAUD_THRESHOLD) OR (risk_score >= 70)` |
| Agreement | `agree` if both sides give the same answer, `disagree` if they differ, `laya_unavailable` if Laya failed |
| Laya question | `is_fraud`, type `noul`, with the same wording everywhere |
| Qwen | `qwen2.5:7b`. Switch to `qwen2.5:3b` if one report takes more than about 8 s on the demo laptop |
| Storage / UI | SQLite / Streamlit |

### 2b. Mismatches between the old guides you must resolve (30 minutes)

The two old guides disagree on two points. Settle both once, write the answer here, and use it everywhere.

1. **Which checkpoint.** The build guide uses `convaiinnovations/laya-typed-decisions` (1024 tokens). The fine-tune guide trains `--base english` (512 tokens). Run `laya-train --help` and see which bases it accepts. **Run the baseline on the same checkpoint you fine-tune from.** Otherwise "fine-tuned vs base" is not a fair comparison.
   - Chosen checkpoint: `________`
2. **Input format for `predict`.** The build guide passes `{"body": text}`. The fine-tune guide passes a plain string, which matches the `state` field in the training JSONL. Print the raw result of both on one event. **Use the format that matches training** (most likely the plain string).
   - Chosen format: `________`
   - Key that holds the yes-probability for `noul`: `________` (print one raw result to find it, then update `get_prob()`)

Also check in this half hour:
- [ ] `pip install laya` works and `laya-train --help` runs. Check every flag used in this guide against `--help`, because the CLI may change.
- [ ] The latency of one `predict` call on the laptop.
- [ ] What happens when the input is longer than the token limit (truncation or an error).
- [ ] Where training runs: Kaggle (free T4 x2), Colab T4, or a local GPU. Create the account and confirm the GPU works **now**, not at hour 3.
- [ ] `ollama pull qwen2.5:7b` has started. It downloads slowly.

### 2c. Laya fallback ladder (if Laya misbehaves at any point)

1. Rewrite the question wording (and then use that wording everywhere).
2. Switch to a two-option `choice` question with neutral keys (A/B).
3. Ship the base checkpoint instead of the fine-tune.
4. Show Laya on the dashboard as a "second opinion" only, and gate on the ML score alone. Update the pitch to match.

---

## 3. Team tracks

| Track | Owner | Covers |
|---|---|---|
| **T1: Data + models** | ML person | generator, LightGBM + IF, Laya dataset, fine-tune, all evaluation |
| **T2: Agents + backend** | backend person | schemas, evidence, text format, Qwen agent + validator, pipeline, API, tests |
| **T3: Dashboard + pitch** | frontend/pitch person | Streamlit UI (built against mock data first), slides, backup video, README |

T3 starts with **mock `AnalyzeResponse` JSON** built from `schemas.py`, so the dashboard does not wait for the backend. T2 owns `schemas.py`. Nobody else changes it without telling the team.

**Solo or pair?** Follow the hour plan in section 10 in order. Use the cut list (section 11) early, starting at item 1.

---

## 4. Repo structure

```
simguard/
├── README.md
├── requirements.txt
├── .env.example
├── docs/GUIDE.md                    # this file
├── .github/copilot-instructions.md  # section 13
├── data/
│   ├── generate.py                  # one generator for everything
│   ├── events.csv                   # gitignored
│   └── eval_cases.csv               # 50 balanced cases (smoke test)
├── ml/
│   ├── train.py  score.py
│   └── artifacts/                   # model.joblib, iforest.joblib, scaler.json, metrics.json
├── agents/
│   ├── schemas.py                   # the contract
│   ├── features_text.py             # event -> Laya text (training AND inference)
│   ├── laya_agent.py  evidence.py  qwen_agent.py
├── finetune/laya/
│   ├── build_dataset.py  train_laya.sh  eval_laya.py
├── api/
│   ├── main.py  pipeline.py  db.py
├── dashboard/app.py
├── eval/
│   ├── laya_baseline.ipynb  compare.py  eval_qwen.py
│   └── results/                     # every table shown in the pitch
├── models/                          # gitignored: laya-simguard/
├── scripts/warmup.py  seed_demo_cases.py
└── tests/test_score.py  test_evidence.py  test_qwen_validator.py  test_pipeline.py
```

`requirements.txt`: `fastapi uvicorn pydantic pandas numpy scikit-learn lightgbm joblib streamlit requests laya python-dotenv pytest`

`.env.example`:
```
OLLAMA_URL=http://localhost:11434
QWEN_MODEL=qwen2.5:7b
LAYA_MODEL=./models/laya-simguard      # fallback: the base checkpoint chosen in 2b
LAYA_BASE_MODEL=<checkpoint from 2b>
LAYA_FRAUD_THRESHOLD=0.5               # replaced by the value chosen on val.jsonl
HASH_SALT=change-me
API_URL=http://localhost:8000
```
Swapping a model must be a one-line `.env` change with no code edit.

---

## 5. Data contracts (`agents/schemas.py`): build these first

```python
from typing import Literal, Optional
from pydantic import BaseModel

class Event(BaseModel):
    event_id: str
    msisdn_hash: str
    timestamp: str                 # ISO 8601
    imei_changed: int              # 0/1
    request_hour: int              # 0-23
    location_gap_km: float
    failed_auth_count: int
    support_calls_24h: int
    days_since_last_swap: int
    otp_burst_count: int

class MLResult(BaseModel):
    risk_score: float              # 0-100
    tier: Literal["low", "medium", "high"]
    lgbm_prob: float
    anomaly: float                 # 0-1
    contributions: dict[str, float]   # feature -> LightGBM contribution (pred_contrib)
    latency_ms: float

class LayaResult(BaseModel):
    p_fraud: Optional[float]       # None if Laya failed
    model: str                     # which checkpoint actually answered
    latency_ms: float

class AnalyzeResponse(BaseModel):
    case_id: str
    event: Event
    ml: MLResult
    laya: LayaResult
    evidence: list[str]
    fraud_detected: bool
    agreement: Literal["agree", "disagree", "laya_unavailable"]
    report: Optional[str]
    report_source: Literal["qwen", "template", "none"]
    report_latency_ms: Optional[float]
    status: Literal["pending", "confirmed", "false_alarm"] = "pending"
    created_at: str
```

---

## 6. Build steps

Each step has a **Done when** check. Do not start the next step on a track until that check passes.

### Step 1: Environment (T-all, hour 0-1)
- Start `ollama pull qwen2.5:7b` first.
- `python -m venv .venv && pip install -r requirements.txt`. Create the folders, `.env` and `schemas.py`.
- Work through the section 2b checklist.
- **Done when:** `ollama run qwen2.5:7b "hi"` answers, `import laya, lightgbm, fastapi` works, and the two blanks in 2b are filled in.

### Step 2: Synthetic data (T1, hour 1-3), `data/generate.py`
- One generator with a `seed` argument and a `fraud_rate` argument. Every dataset comes from it.
- Fields as in `Event`, plus `label` and a hidden `severity` (low/medium/high) for the optional stretch question.
- Phone numbers: `sha256(salt + number)`.
- Fraud events lean toward: IMEI changed, hour 0-5, large location gap, 3+ failed auths, 2+ support calls, swap under 30 days ago, OTP burst 4+. Normal events lean the other way, **with realistic overlap**.
- Noise: about 2% label flips and about 10% of fraud events that look normal. This keeps every metric honest.
- Outputs:
  - `events.csv`: 20,000 events at about 3% fraud (for LightGBM).
  - `eval_cases.csv`: 50 cases, about 25 fraud and 25 normal, from a different seed.
- **Done when:** the row counts and fraud rates are correct.

### Step 3: Text format and evidence (T2, hour 1-3), `agents/features_text.py`, `agents/evidence.py`

`features_text.py` is used **for training and for inference**. Freeze it before the Laya dataset is built at hour 3. Changing it later means retraining. Put words next to the numbers:

```
SIM replacement request. IMEI changed: yes. Request hour: 03:12 (night).
Location gap: 1200 km (very large). Failed authentications: 4 (high).
Support calls in 24h: 2 (elevated). Days since last swap: 5 (very recent). OTP burst count: 6 (high).
```
Keep it well under 200 tokens.

`evidence.py` contains pure functions with no model:

| Rule | Indicator text |
|---|---|
| `imei_changed == 1` | "IMEI changed" |
| `request_hour <= 5` | "request at unusual hour (HH:00)" |
| `location_gap_km > 300` | "location gap N km" |
| `failed_auth_count >= 3` | "N failed authentications" |
| `support_calls_24h >= 2` | "N support calls in 24h" |
| `days_since_last_swap < 30` | "recent SIM swap (N days ago)" |
| `otp_burst_count >= 4` | "OTP burst (N)" |

- **Done when:** `tests/test_evidence.py` passes (each rule fires at its threshold and not just below it), and the text for one event has been printed and reviewed by the team.

### Step 4: Laya dataset and fine-tune launch (T1, hour 3-4), `finetune/laya/`

`build_dataset.py`: generate with different seeds, render with `event_to_text`, and **remove duplicate texts across splits**.

| File | Size | Fraud rate | Use |
|---|---|---|---|
| `train.jsonl` | 8,000 | about 30% (rebalanced) | training |
| `val.jsonl` | 2,000 | 3% (realistic) | choose the threshold and the best seed |
| `test.jsonl` | 4,000 | 3% (realistic) | final numbers, **used once** |

Row format:
```json
{"state": "<event_to_text output>",
 "questions": {"is_fraud": {"type": "noul", "instructions": "Does this SIM replacement request look like a SIM-swap fraud attempt?"}},
 "expected": {"is_fraud": true}}
```

Checks: no text in two splits, the fraud rates match the table, the row counts match.

Dry run, then train (check the flags against `--help`):
```bash
laya-train --data finetune/laya/train.jsonl --base <checkpoint from 2b> \
           --out ./models/laya-simguard --dry-run
# fix any skipped rows, then:
laya-train --data finetune/laya/train.jsonl --base <checkpoint from 2b> \
           --out ./models/laya-simguard-s0 --seed 0 --eval-data finetune/laya/val.jsonl
```
- Use the documented defaults (4 epochs, effective batch 64, `--loss rlcd`). Leave `--shuffle-options` off; it is for choice questions.
- When seed 0 is running, queue seed 1. If GPU time allows, try one `--loss soft-ce` run. Keep the best run by **validation PR-AUC**.
- **Done when:** the run is training in the background on Kaggle or Colab.

### Step 5: Scoring engine (T1, hour 4-6), `ml/train.py`, `ml/score.py`
- Use `agents.features_text.FEATURE_COLUMNS` and `LABEL_COLUMN` from the present SIM-swap dataset; do not include the target label in model input. Train with a stratified 70/15/15 split. LightGBM uses `scale_pos_weight`; Isolation Forest is fit on **normal training events only**. Scale its decision function to 0-1 by reversing and clipping against the 5th/95th percentiles of normal training scores (saved in `scaler.json`).
- `score_event(event) -> MLResult` combines the LightGBM probability and anomaly score using `100 * (0.7 * lgbm_prob + 0.3 * anomaly)`. Tiers are low < 40, medium 40-70, and high >= 70. Get `contributions` from `model.predict(X, pred_contrib=True)`: the final output column is the bias, not a feature. Return the five feature contributions with the largest absolute values.
- Evaluate the combined risk score on the held-out test split. Report precision, recall, PR-AUC and the confusion matrix at the high-tier threshold. Save them to `metrics.json` with `"data": "synthetic"`.
- Train and write artifacts with `python -m ml.train --data sim_swap_fraud_dataset_100000.csv --artifacts ml/artifacts`. `score_event` loads the persisted model, Isolation Forest and scaler from that artifact directory.
- To continue an existing LightGBM model, use `python -m ml.train --continue-existing --additional-rounds 300 --early-stopping-rounds 50`. The continuation starts from the saved model, uses the validation split for early stopping and retention, and leaves Isolation Forest unchanged. The held-out test metrics are refreshed for reporting, not for model selection.
- **Done when:** `tests/test_score.py` passes (a fraud-like event scores >= 70, a normal one < 40, and every score is within 0-100).

### Step 6: Laya agent and base baseline (T1, hour 6-8), `agents/laya_agent.py`, `eval/laya_baseline.ipynb`
- Load `LAYA_MODEL` once at import. If loading fails, load `LAYA_BASE_MODEL`. If that also fails, return `p_fraud=None`. Do one warm-up call. Record in `model` which checkpoint actually answered.
- Use the **same question dict, wording and input format** as in training.
- Run the **base** checkpoint on `eval_cases.csv` (50 cases): accuracy, precision, recall and F1 at 0.3/0.5/0.7, ROC-AUC, PR-AUC, Brier score, the majority-class baseline, and mean and p95 latency. Call this a smoke test; 50 cases give wide error bars.
- **Done when:** a fraud-like event gives a clearly higher `p_fraud` than a normal one, and the baseline table is saved in `eval/results/`.

### Step 7: Qwen agent and validator (T2, hour 3-6), `agents/qwen_agent.py`
- `POST {OLLAMA_URL}/api/chat`, `stream: false`, `options: {"temperature": 0.2, "num_predict": 350}`, timeout 20 s.
- System prompt:
```
You write short fraud investigation reports for telecom analysts.
Explain ONLY the indicators and the risk score provided. Do not add indicators,
numbers, or facts that are not in the input. Do not change the score.
Format: 1) Summary (1-2 sentences) 2) Why it was flagged (bullets, one per indicator)
3) Recommended actions (hold request, notify subscriber via a verified channel,
extra identity verification). Maximum 150 words. Say a human analyst must decide.
If "agreement" is "disagree", say the automated signals disagree and analyst review is advised.
```
- User message: JSON with `risk_score`, `tier`, `evidence`, `laya_p_fraud`, `agreement`.
- `validate_report(report, payload)` rejects a report when:
  - it contains a number that is not in the payload,
  - the printed score is not the supplied score,
  - it uses an indicator keyword that is not in the evidence (keep a small blocklist, for example "cloned", "malware", "phishing", "stolen device"),
  - it is missing a section, is over 150 words, or does not mention a human analyst.
- On failure, retry once. If it fails again, use `template_report(payload)` and set `report_source="template"`. Do the same when Ollama is down or times out.
- Cache reports by a hash of the payload.
- **Done when:** `tests/test_qwen_validator.py` passes (invented numbers and a wrong score are rejected, a faithful report passes).

### Step 8: Pipeline, API and tests (T2, hour 6-9), `api/`
`analyze(event)`:
1. `ml = score_event(event)`
2. `laya = decide(event)`. If `p_fraud is None`, gate on ML alone and set `agreement = "laya_unavailable"`.
3. `evidence = extract_evidence(event)`
4. `laya_flag = p_fraud >= LAYA_FRAUD_THRESHOLD`, `ml_flag = risk_score >= 70`, `fraud_detected = laya_flag or ml_flag`
5. `agreement = "agree" if laya_flag == ml_flag else "disagree"`
6. If `fraud_detected`, call Qwen (timed). Otherwise `report=None`, `report_source="none"`.
7. Save to SQLite, add an audit-log row, and return.

Endpoints: `POST /analyze`, `GET /cases` (sorted by risk, highest first), `GET /cases/{id}`, `POST /cases/{id}/review` with `{"status": "confirmed" | "false_alarm"}`, and `GET /health` (shows which Laya checkpoint and which Qwen model are loaded).

- **Done when:** `tests/test_pipeline.py` passes (a fraud event gets a report, a normal event gets `report=None`, stopping Ollama gives a template report, a missing Laya gives `laya_unavailable`), and a `curl` to `/analyze` shows up in `/cases`.

### Step 9: Dashboard (T3, hour 1-14, first against mock data, then wired to the API at hour 9)
- Sidebar: **Simulate SIM swap** buttons (High risk / Medium / Normal) that post ready-made events. Add a **"Replay attack"** button that posts a scripted sequence of 5 events for one subscriber, ending in a SIM swap with an OTP burst. This is the moment the demo builds to.
- Case table sorted by risk: score, tier, status, agreement.
- Detail panel:
  - risk score and tier badge,
  - top LightGBM feature contributions as a small bar chart,
  - evidence list, Laya `p_fraud` and which checkpoint produced it,
  - the Qwen report with a `qwen` / `template` label,
  - a warning banner when the models disagree,
  - **latency for each stage** (ML ms, Laya ms, Qwen s).
- **Confirm fraud** / **False alarm** buttons.
- **Metrics (synthetic data)** tab: ML metrics, the Laya before/after table, and the Qwen validator table.
- Footer: "Prototype on synthetic data. Final decisions rest with a human analyst."
- **Done when:** a click on Simulate shows a new case with a report in a few seconds.

---

## 7. Evaluation: the numbers that go in the pitch

### 7a. Laya (T1, hour 8-12), `finetune/laya/eval_laya.py`
Compare on `val.jsonl`, choose the threshold, then run **once** on `test.jsonl`:

| System | PR-AUC | ROC-AUC | Precision @ thr | Recall @ thr | Brier | ECE | p95 latency |
|---|---|---|---|---|---|---|---|
| Majority class | | | | | | | |
| Base Laya | | | | | | | |
| Fine-tuned Laya | | | | | | | |
| LightGBM + IF | | | | | | | |

Threshold: on `val.jsonl`, pick the lowest threshold that reaches your target recall (for example 0.90) with the best precision. Write it into `LAYA_FRAUD_THRESHOLD`. Do not change it after looking at `test.jsonl`.

Probabilities from a model trained at 30% fraud are **not** real-world likelihoods. Never present them as such.

Decision rule:
- Fine-tuned clearly beats base: ship it and show the table.
- About equal to LightGBM: say so. That is a strong result.
- Below LightGBM: normal for tabular data. Laya stays the fast gate and second opinion.
- Not better than base: ship the base, or make Laya display-only. Update the pitch.

**If the run stays at chance level:** first check for leaks and duplicates. Then train a copy of the data where every text starts with `label=fraud;` or `label=normal;`. If that copy also fails to learn, the problem is the training setup or budget (try more epochs or another seed). If the copy learns but the real data does not, the problem is the text format.

### 7b. Qwen (T2, hour 9-12), `eval/eval_qwen.py`
Generate 200 payloads that cover 1, 2, 3, 5 and 7 indicators, all three tiers, `disagree` cases, and a range of scores. Run the production settings:

| Metric | `qwen2.5:7b` + prompt + validator |
|---|---|
| Validator pass rate on the first try | |
| Reports with an invented number or indicator (before the validator) | |
| Invented content that reached the analyst (after validator + retry + template) | **should be 0** |
| Format compliance | |
| Template fallback rate | |
| Mean / p95 latency | |

Also read 20 reports yourself, including the `disagree` and low-tier ones.

The middle two rows are the pitch: *"The LLM invented content in X% of first drafts. Our validator stopped all of them before they reached an analyst."*

---

## 8. Stretch goals (only if you are ahead of schedule at hour 12)

In order of value for the time spent:
1. **Live stream mode**: a background loop posts one synthetic event per second, and the case table fills up live. Strong visual, about 1 hour.
2. **Qwen LoRA fine-tune** (Unsloth, 4-bit, rank 16, lr 2e-4, 1-2 epochs, trained on the response only; data from varied templates and **filtered by `validate_report()`**). Ship it only if it beats the 7b table above with no new invented content. Export to GGUF and register it in Ollama with the `TEMPLATE` and `PARAMETER` lines copied from `ollama show qwen2.5:7b --modelfile`. Hard stop: if it is not better by hour 16, drop it.
3. **`risk_tier` choice question for Laya**, trained on the generator's hidden `severity` (never on the ML score, or Laya just copies LightGBM).

---

## 9. Hardening (hour 16-19)
- `scripts/warmup.py` loads Laya and sends one prompt to Qwen. Run it before going on stage.
- `scripts/seed_demo_cases.py` inserts one high, one medium and one low case, so the dashboard is never empty.
- Test each fallback by actually breaking it: stop Ollama (template appears), point `LAYA_MODEL` to a missing folder (base loads), break both (`laya_unavailable`, ML gate).
- Every `.env` model loads from a cold start on the **demo laptop**.
- Record a **backup demo video** of the full 2-minute script.

---

## 10. Hour plan

| Hour | T1: Data + models | T2: Agents + backend | T3: Dashboard + pitch |
|---|---|---|---|
| 0-1 | Setup, section 2b checks, GPU account | Setup, `schemas.py`, start Ollama pull | Setup, mock `AnalyzeResponse` JSON |
| 1-3 | Generator (Step 2) | Text format + evidence (Step 3), freeze text format at hour 3 | Dashboard layout on mock data |
| 3-4 | **Laya dataset + dry run + start fine-tune** | Qwen agent, prompt, validator | Dashboard detail panel |
| 4-6 | LightGBM + IF + contributions (Step 5) | Validator tests, template, cache | Slides 1-2 (problem, who it is for) |
| 6-8 | Laya agent + base baseline (Step 6), queue seed 1 | Pipeline, DB, API (Step 8) | Metrics tab layout, "Replay attack" script |
| 8-9 | Laya eval on `val`, pick seed and threshold | Pipeline tests | Wire dashboard to the real API |
| 9-12 | Laya on `test` (once), save tables | Qwen eval on 200 payloads (7b) | End-to-end clicks, fix UI bugs |
| 12 | **Checkpoint: are we ahead? If yes, pick one stretch goal. If no, apply the cut list.** | | |
| 12-16 | Swap in the fine-tuned Laya if it passed, rerun tests | Integration fixes, `/health` | Metrics tab filled with real tables |
| 16 | **Model freeze. No new training runs, no new features.** | | |
| 16-19 | Fallback tests, warm-up | Seed demo cases, cold-start test | Backup video, README |
| 19-22 | Pitch rehearsal and fixes for whatever broke | | Slides final |
| 22-24 | **Buffer.** Rehearse twice. Sleep if you can. | | |

Hard cutoffs:
- Laya fine-tune not better than base on `val.jsonl` by hour 12: ship the base or make it display-only.
- Any stretch goal not working by hour 16: drop it.

---

## 11. Cut list (cut in this order)

1. All stretch goals.
2. The second Laya seed and the `soft-ce` experiment.
3. Isolation Forest (use LightGBM alone and remove it from the pitch).
4. The Qwen cache and the disagreement banner.
5. `qwen2.5:3b` instead of 7b.
6. The Laya fine-tune itself (ship the base Laya and say so).

Never cut: the validator, the template fallback, the "synthetic data" labels, the human review buttons, the backup video.

---

## 12. Pitch and demo

### Five slides
1. **Problem.** SIM-swap fraud lets an attacker take over a phone number and intercept one-time passwords. Use one real, cited statistic (for example from a national fraud-reporting agency or a telecom regulator), with the source on the slide.
2. **Who pays, operates and benefits.** The telecom operator runs it, the fraud team reviews cases, and subscribers and banks benefit.
3. **Architecture.** Three tiers of cost: the ML score (fast, every event), the fine-tuned Laya gate (fast, every event), and Qwen (slow, only flagged events). Add one line: "The score comes from ML. Qwen explains it and never produces it."
4. **Live demo** (below).
5. **Results and honest limitations.** The Laya before/after table, the validator's "0 invented facts reached an analyst" number, the latency of each stage, all labelled "synthetic data". Then the limitations.

### 2-minute demo script
1. Architecture slide, one sentence per box.
2. **Replay attack.** Events appear one by one, and the last one jumps to high risk. Show the score, the top feature contributions, the evidence and the Qwen report.
3. Point at the latency row: ML and Laya in milliseconds, and Qwen only ran because the event was flagged.
4. **Simulate normal request.** Low risk, no report, no LLM cost.
5. Show a `disagree` case and the warning banner: the system tells you when it is unsure.
6. Click **Confirm fraud**: a human makes the decision.
7. Limitations slide, said plainly.

### Wording that stays honest
- "We fine-tuned Laya on synthetic SIM-swap data and shipped it only because it beat the base model on held-out synthetic data." (Change this to match the real result.)
- "Numbers are on synthetic data and say nothing about real operator traffic."
- "Laya is a fast second opinion. The score comes from the ML engine. Qwen explains the score and does not produce it."
- "We considered fine-tuning Qwen. A prompt with a strict validator already kept invented facts away from analysts, so we put our time into the validator instead." (Only if you skipped the Qwen stretch goal.)

### Limitations to state
- Synthetic, deliberately noisy data. Real-world accuracy is not established.
- Laya's probabilities come from a model trained at a 30% fraud rate. They rank cases; they are not real-world likelihoods.
- The prototype recommends a hold. A real hold needs operator integration and approval.
- Production would need integration with SIM-change and call-detail systems, consent, access control, retention limits and audit logging.

---

## 13. Working with an AI coding assistant

Step prompt:
```
Read docs/GUIDE.md. Implement Step N only (files listed there). Follow the data contracts
in section 5 exactly. Do not modify files outside this step. Add the test named in the step.
When done, tell me how to run the test and what "Done when" result to expect.
```

Debug prompt:
```
Here is the error and the file. Fix only this file. Do not change agents/schemas.py.
Keep the function signatures from docs/GUIDE.md.
```

`.github/copilot-instructions.md`:
```
- This repo is SIMGuard: ML score (LightGBM + Isolation Forest) + fine-tuned Laya gate + Qwen report.
- Follow docs/GUIDE.md and agents/schemas.py exactly.
- The numeric risk score only comes from ml/score.py. Never let an LLM produce or change a score.
- Laya returns p_fraud only. Evidence comes only from agents/evidence.py.
- agents/features_text.py is shared by Laya training and inference. Do not change it without retraining.
- The Laya threshold comes from LAYA_FRAUD_THRESHOLD. Never hard-code it.
- Every Qwen report goes through validate_report(); fall back to template_report().
- Hashed phone numbers only. No raw numbers in logs.
- All metrics are on synthetic data; label them that way in the UI and docs.
- Keep functions small, typed, and covered by the tests in tests/.
```

---

## 14. Final checklist before going on stage

- [ ] `.env` points to the intended models, and each loads from a cold start on the demo laptop.
- [ ] `scripts/warmup.py` has been run.
- [ ] `pytest` passes with the shipped models.
- [ ] All fallbacks tested by breaking them on purpose.
- [ ] Demo cases seeded. The "Replay attack" button works three times in a row.
- [ ] Every table in `eval/results/` and on the metrics tab is labelled "synthetic data".
- [ ] The backup video is on the laptop **and** on a USB stick or in the cloud.
- [ ] The pitch has been rehearsed twice and fits in the time limit.
