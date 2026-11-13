---
license: apache-2.0
base_model: convaiinnovations/laya
language:
- en
library_name: laya
pipeline_tag: text-classification
tags:
- fraud-detection
- sim-swap
- telecom
- modernbert
- synthetic-data
---

# SIMGuard Laya: SIM-swap fraud classifier

A fine-tuned version of [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya)
(ModernBERT-large encoder, 421M parameters, single-pass yes/no head) that reads a short text
description of a SIM-swap event and returns **P(fraud)**. It is the "Agent 1" fast gate of the
SIMGuard AI prototype: it never writes text, it only scores.

> **Synthetic data only.** Every number on this page was measured on synthetic data. The model is a
> research prototype for decision support; a human analyst must make the final call.

## Results

| Test set | What it is | Base Laya | This model |
|---|---|---|---|
| **test** (6,000) | held out, same distribution as training | 25.6% | **99.17%** |
| **hard_test** (600) | synthetic knife-edge cases built to be hard | 55.0% | **83.33%** |

Errors on test fell from 4,461 to 50 (−98.9%); on hard_test from 270 to 100 (−63%).
The base model has no notion of SIM-swap fraud (it answers "not fraud" for almost everything), so the
fair statement is that fine-tuning took it from no signal (balanced accuracy 0.50) to a working classifier.

### Full metrics

| Split | Metric | Base | Fine-tuned |
|---|---|---|---|
| val (3,000) | accuracy | 0.2517 | **0.9937** |
| | balanced accuracy | 0.5000 | 0.9918 |
| | log-loss | 1.7350 | **0.0201** |
| | fraud recall / legit recall | 0.000 / 1.000 | 0.9955 / 0.9881 |
| test (6,000) | accuracy | 0.2565 | **0.9917** |
| | balanced accuracy | 0.4999 | 0.9884 |
| | log-loss | 1.7325 | 0.0251 |
| | fraud recall / legit recall | 0.0011 / 0.9987 | 0.9951 / 0.9818 |
| hard_test (600) | accuracy | 0.5500 | **0.8333** |
| | log-loss | 1.0378 | 0.4304 |
| | fraud recall / legit recall | 0.000 / 1.000 | 0.8296 / 0.8364 |

An independent re-run from the saved checkpoint reproduced these within 0.07 pp on test and 1 case in
600 on hard_test.

### Where it is weak (hard_test accuracy by category)

| Category | n | Base | Fine-tuned |
|---|---|---|---|
| far_but_flagged | 60 | 50% | 100% |
| distractor_age | 70 | 50% | 97% |
| quiet_takeover | 60 | 0% | 95% |
| behaviour_storm | 60 | 100%* | 83–85% |
| two_changes_benign | 60 | 100%* | 83% |
| flip_pair | 120 | 50% | 76–77% |
| knife_edge_double | 80 | 50% | 72–74% |
| knife_edge_single | 90 | 50% | 71% |

\*The base model scores 100% there only because it says "legit" for everything and those cases are legit.

Cases a hair's breadth from the decision line are the weak spot: a text encoder reads numbers less
precisely than a tree model. Use it next to a numeric model (SIMGuard pairs it with LightGBM + Isolation
Forest), not alone.

### Decision gate (block / allow / human review)

`gate.json` holds cut-offs fitted on a held-out calibration slice: **allow if p ≤ 0.1197, block if
p ≥ 0.8828**, otherwise send to a human.

| Split | Blocked | Allowed | To human | Fraud auto-allowed | Legit auto-blocked |
|---|---|---|---|---|---|
| test (6,000) | 4,431 | 1,487 | 82 (1.4%) | 11 (0.25% of fraud) | 13 (0.85% of legit) |
| hard_test (600) | 216 | 253 | 131 (21.8%) | 21 | 27 |

On normal traffic the gate decides 98.6% of cases automatically at about 99.6% accuracy.

## Speed (measured)

| Hardware | Single request | Batched |
|---|---|---|
| NVIDIA RTX 3050 Laptop GPU (6 GB), PyTorch | 39 ms (p50), 46 ms (p95) | 13 ms per event (batch 64) |
| Laptop CPU, PyTorch | ~510 ms per event | — |

Fine-tuning did not change inference speed versus the base model.

## Requirements

**Hardware**

| | Minimum | Recommended |
|---|---|---|
| GPU | none (CPU works, ~0.5 s per event) | NVIDIA GPU with ≥ 4 GB VRAM (tested: RTX 3050 6 GB Laptop) |
| RAM | 8 GB | 16 GB |
| Disk | 2 GB free for the model + ~6 GB for PyTorch with CUDA | |
| CPU | any 64-bit x86 CPU | 4+ cores |

The weights are 843 MB. On GPU the model uses roughly 1.5–2 GB of VRAM at batch size 1.

**Software** (tested combination)

- Windows 11, Python 3.12–3.15 (trained and tested on 3.15.0; Linux and macOS are expected to work but were not tested)
- `torch==2.15.0` (CUDA 13.2 build for NVIDIA GPUs, which needs a recent NVIDIA driver; or the CPU build)
- `laya==0.4.31`, `transformers==5.22.2`, `safetensors`, `huggingface_hub`

## How to run

### 1. Install

GPU (NVIDIA):

```bash
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
pip install laya==0.4.31 transformers==5.22.2
```

CPU only (any PC or laptop):

```bash
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install laya==0.4.31 transformers==5.22.2
```

### 2. Predict

The model only works with **the exact text format and question it was trained on**:

```python
import torch
import laya

device = "cuda" if torch.cuda.is_available() else "cpu"
agent = laya.load("Kyoiske/simguard-laya-fraud", device=device)

QUESTION = {"is_fraud": {"type": "noul",
                         "instructions": "Does this SIM swap event look like SIM-swap fraud?"}}

def event_text(e):
    yn = lambda v: "yes" if int(v) == 1 else "no"
    gap = int(e["sim_swap_time_gap_minutes"])
    return (
        "SIM swap event. "
        f"Time between OTP request and SIM change: {gap} minutes ({gap / 1440:.2f} days). "
        f"Device changed: {yn(e['device_change_flag'])}. "
        f"SIM type changed: {yn(e['sim_type_change_flag'])}. "
        f"IMSI changed: {yn(e['imsi_change_flag'])}. "
        f"ICCID changed: {yn(e['iccid_change_flag'])}. "
        "Geohash similarity between OTP location and SIM change location: "
        f"{float(e['otp_and_sim_change_geo_hash_length']):.2f} (0 = different place, 1 = same place). "
        f"Days since SIM activation: {int(e['recent_sim_activation_days'])}. "
        f"SIM changes in last 30 days: {int(e['num_sim_changes_last_30d'])}. "
        f"Previous SIM holder tenure: {int(e['previous_sim_holder_tenure_days'])} days. "
        f"Account age: {int(e['account_age_days'])} days. "
        f"IP address changed: {yn(e['ip_change_flag'])}."
    )

event = {
    "sim_swap_time_gap_minutes": 45, "device_change_flag": 1, "sim_type_change_flag": 0,
    "imsi_change_flag": 1, "iccid_change_flag": 1, "otp_and_sim_change_geo_hash_length": 0.12,
    "recent_sim_activation_days": 3, "num_sim_changes_last_30d": 2,
    "previous_sim_holder_tenure_days": 40, "account_age_days": 120, "ip_change_flag": 1,
}
p_fraud = agent.predict(event_text(event), QUESTION)["answers"]["is_fraud"]["noul"]

LO, HI = 0.1197, 0.8828  # from gate.json
decision = "block" if p_fraud >= HI else "allow" if p_fraud <= LO else "human_review"
print(f"p(fraud) = {p_fraud:.3f} -> {decision}")
```

For many events, `agent.predict_batch(texts, QUESTION, batch_size=64)` is about 3x faster per event on GPU.

### 3. Or run the whole SIMGuard AI app

The full system (LightGBM + Isolation Forest scoring, this model, rule-based evidence, Qwen 3.5 2B
reports and a Streamlit analyst UI) is in the SIMGuard AI GitHub repository. On Windows, clone it and run
`setup.bat`: it installs everything, downloads this model, and opens the app.

## Training

| | |
|---|---|
| Base model | `convaiinnovations/laya` (encoder `answerdotai/ModernBERT-large`), Apache-2.0 |
| Data | public synthetic SIM-swap dataset, 11 features, label `sim_swap_flag`; 22,000 train / 2,000 calibration / 3,000 val / 6,000 test, plus 300 + 600 synthetic hard cases |
| Leakage check | 0 exact-duplicate states between train and val / test / hard sets |
| Trainable | 198M of 421M parameters (bottom 14 encoder layers frozen) |
| Optimiser | encoder LR 1e-5, head LR 1e-4, weight decay 0.01, soft cross-entropy |
| Batch | micro-batch 4 × gradient accumulation 8 = 32 |
| Epochs | 6 (epoch 5 selected by lowest validation log-loss); a 2-epoch continuation was worse and discarded |
| Hardware / time | 1 × RTX 3050 6 GB Laptop GPU, about 6 hours |
| Calibration | temperature 1.73 for the yes/no head, fitted on the held-out calibration slice |

## Limitations

- **Synthetic data only.** Real SIM-swap traffic will differ; re-validate before any real use.
- **Borderline cases are unreliable.** About 83% accuracy on deliberately hard boundary cases.
- **Not perfectly batch-invariant.** Scoring the same case at batch size 1 vs 64 can move its probability
  by up to 0.36; 7 of 600 hard cases flipped decision, all with p between 0.41 and 0.57. Treat any
  p in roughly 0.4–0.6 as undetermined (the gate's human-review band already does).
- **Exact input format required.** Different wording, units or field order were never seen in training.
- **Decision support, not decisions.** It recommends; a human analyst decides.

## Files

| File | Purpose |
|---|---|
| `model.safetensors` | fine-tuned weights (843 MB) |
| `encoder/config.json`, `tokenizer/` | ModernBERT encoder config and tokenizer |
| `rl_agent_config.json` | Laya head configuration and calibration |
| `questions.json` | the exact question used in training |
| `gate.json` | block / allow cut-offs and how they were fitted |
| `export_report.json` | ONNX export check (decision agreement 1.0 vs PyTorch on 600 cases) |
