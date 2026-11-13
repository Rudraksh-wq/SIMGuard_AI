# Laya × SIM-swap fraud: fine-tuning kit

Run it with `.\run_all.ps1` from this folder. It runs on your CUDA GPU (sized for the RTX 3050
6 GB) and does everything below in one go.

## What the system does
Laya sits in front of the SIM-swap workflow as a **System 1 gate**. One forward pass gives a
calibrated P(fraud) and the gate turns it into one of three outcomes:

| outcome | meaning | where it goes |
|---|---|---|
| `BLOCK` (1) | clearly fraud | stopped at the gate, never reaches the system or an employee |
| `ALLOW` (0) | clearly legitimate | proceeds automatically |
| `NOUL` | not sure | **human review queue** (the only requests an employee ever sees) |

**The not-sure band is learned from calibration, not trained as a label.** The model trains on
the dataset's real 0/1 labels only. After training, `gate.py` fits two cut-offs on a held-out
calibration slice (2,000 rows, set aside before training) under two error budgets:
- at most **0.2%** of real fraud may be auto-ALLOWED
- at most **1%** of real legit may be auto-BLOCKED

**That asymmetry is a deliberate, logged bias toward catching fraud** ("missing fraud is worse").
It is the only bias in the system. It lives in the gate, not in the training data, and the
budgets are the `--max-fraud-leak` / `--max-false-block` flags. The fitted band is saved as
`final/gate.json`; `gate.route(p, gate)` applies it in SIMGuard.

## What Laya is (and why that matters here)
Laya (`laya 0.4.31`, checkpoint `convaiinnovations/laya`) is a **ModernBERT-large encoder
(about 421M parameters) with a small decision head**. It reads text and returns P(yes) for a
yes/no question in a single forward pass. It is a classifier, not a chat/generative LLM.

## Data
| file | source | used for |
|---|---|---|
| `data/train.jsonl` (24,000) | `sim_swap_fraud_dataset_100000.csv` | gradient updates (10% held out *before* training for temperature calibration) |
| `data/val.jsonl` (3,000) | same file, disjoint rows | early stopping + checkpoint selection |
| `data/test.jsonl` (6,000) | `sim_swap_fraud_dataset_50000.csv`, rows unseen in train/val | baseline and final only |
| `data/hard_dev.csv/.jsonl` (300) | generated | monitored every epoch, tie-break |
| `data/hard_test.csv/.jsonl` (600) | generated | **baseline and final only. Never used to choose anything** |

Each row is turned into neutral text: raw values only, with no risk words and no derived
scores. One fixed question is used: *"Does this SIM swap event look like SIM-swap fraud?"*

**The dataset's label rule was recovered exactly** (it reproduces 200,000/200,000 labels; see
`fraud_common.generator_score`):

    score = 3·(device + sim_type + imsi + iccid) − 2·geo − 2·gap/10080
            + [activation_days < 7] + [sim_changes_30d ≥ 2] + ip_change ;  fraud ⇔ score > 3

This rule is used **only** for two things: auditing the data and labelling the generated hard
cases, so their labels mean "what the dataset itself would say". It is never given to the
model, never put in the text, and never used as a training target.

### Hard test categories (`gen_hard_tests.py`)
- `knife_edge_single`, `knife_edge_double`: within 0.12 of the decision line, so small
  differences in geo, gap or extras decide them.
- `two_changes_benign`: two identity changes, but same place and a long gap. Not fraud.
- `quiet_takeover`: one quiet IMSI/ICCID change, tipped over the line by one small signal. Fraud.
- `behaviour_storm`: every behavioural alarm, but no identity change. Not fraud (a trap).
- `distractor_age`: account age and tenure set opposite to intuition. These features have zero
  weight in the rule.
- `far_but_flagged`: many identity changes with everything else calm (fraud), or one change
  plus two alarms that still isn't enough (not fraud).
- `flip_pair`: 60 counterfactual pairs that differ in one field (activation 6↔7 days,
  swaps 1↔2, IP, geo ±0.03, gap +100–300 min), with opposite labels.

## Training (`train_fraud.py`, driven by `run_attempts.py`)
- **Baseline first.** The untouched base model is scored on all four evaluation sets, and its
  speed is measured, before any weight changes.
- **Gentle on the base.** Encoder LR is 1e-5 (Laya's default is 2.5e-5). Embeddings and the
  bottom 8 encoder layers are frozen. Gradient clipping is 1.0 and weight decay 0.01. Loss is
  plain soft cross-entropy, so a "near-zero loss" means what it says.
- **12 epochs minimum, 16–20 maximum.** A checkpoint is saved every epoch (the best one and
  the last two are kept; every deletion is logged).
- **Health checks every epoch.** Each epoch logs:
  - val and hard_dev metrics
  - per-layer attention entropy (on a fixed probe batch, using eager attention for the probe only)
  - last-layer token cosine, which shows over-smoothing or rank collapse
  - relative weight drift from the base
  - prior-collapse check
- **Tipping point.** An epoch whose attention or representations have collapsed can never be
  selected. After the minimum number of epochs, training stops when one of these happens:
  - val stops improving (patience 2–3)
  - attention degrades
  - the model drifts from the base without improving

  The final model is the **best healthy epoch before the tipping point**. If the tipping point
  comes before epoch 12, training still runs to 12 as you required, but the earlier healthy
  checkpoint is the one promoted.
- **Bias policy.** No re-weighting and no re-sampling of the training data: the natural ~74%
  fraud prior is kept. The only bias is the gate's asymmetric error budget (above), recorded in
  `config.json`, `final_report.json`, `gate.json` and the ledger. (`--balance-legit-to` still
  exists for experiments but no rung of the ladder uses it.)
- **Attempt ladder.** `a1_no_bias` → `a2_gentler_longer` → `a3_more_data` (72k rows). It stops
  as soon as val ≥ 99.5%, val loss ≤ 0.02, hard_dev ≥ 97%, val fraud-leak ≤ 0.5% and hard_dev
  confidently-wrong ≤ 2%. The winner is chosen by val loss, then hard_dev accuracy. If a rung
  runs out of GPU memory it is retried once with a smaller footprint (and that is logged).
- **6 GB budget.** Embeddings and the bottom 14 of 28 encoder layers are frozen, micro-batch 4 ×
  accumulation 8, mixed precision and gradient checkpointing on. The attention probe uses only 6
  fixed samples so it fits beside the optimiser.
- **Calibration.** The promoted checkpoint is calibrated with Laya's own temperature and
  abstention fit on the held-out calibration slice. It loads with `laya.load(path)`.

## Everything is logged
- `runs/attempts_ledger.jsonl`: one line per attempt, including failed and skipped ones.
- `runs/<run>/`, one folder per run, containing:
  - `config.json`: every knob, data sha256 hashes, library versions and GPU
  - `baseline.json`
  - `epochs.jsonl`: one line per epoch
  - `train.log`
  - `final_report.json`: before vs after on every split and per hard category, plus speed and
    the stop reason
  - `hard_test_predictions.csv`: every case with p(fraud) and whether it was right
- `runs/winner.json` and `runs/comparison.txt`.

## Speed
Fine-tuning changes weights, not architecture, so per-decision cost is unchanged. It is
measured anyway: base vs final single-request p50/p95 and batched ms/item, in
`final_report.json → speed`. Laya quotes about 33 ms per decision on a T4.

## GGUF / Ollama
**Laya cannot be faithfully converted to GGUF for Ollama.** Ollama (llama.cpp) serves
generative LLMs and embedding models. Laya's fraud answer comes from its own head: two
transformer layers plus a scorer reading the `[MASK]` option positions. llama.cpp has no
architecture for that head, and Ollama doesn't expose the per-token hidden states it would
need. A GGUF of the encoder alone would load, but it could not return a fraud probability.

Laya's supported deployable format is **ONNX**. `export_final.py` writes
`final/laya_fraud.onnx` and checks it agrees with PyTorch on all 600 hard cases.

Your SIMGuard app already keeps the two roles apart:
- Laya scores the event: `laya.load(final_dir)`, or `ONNXAgent`, or `laya-serve`.
- Qwen in Ollama writes the report.

Point the app's Laya checkpoint at `runs/<winner>/final` and the Ollama side stays as it is.

## Honest expectations
- val and test should reach very high accuracy: the label is a fixed rule over the text.
- The hard set is built to be hard. Some knife-edge cases differ from the decision line by the
  equivalent of about 25 minutes of gap or 0.01 of geo similarity, and a text encoder reading
  numbers may still miss some of them.
- `compare.py` lists every hard-test failure, closest-to-the-line first. Nothing is tuned on
  that set to make them disappear.
- **Two ways to read the hard-test result, both reported.** *Strict*: p>0.5 vs the label.
  *Gate*: a case is a failure only if it is **confidently wrong** (real fraud ALLOWED, or real
  legit BLOCKED). A knife-edge case sent to NOUL is the gate doing its job, but it costs a human
  review, so the report also shows how many cases were sent to humans. A model that sends
  everything to NOUL would "pass" the gate view and be useless, so judge `confident_wrong`
  together with `noul_rate` and with the strict accuracy.
- Gate guarantees (0.2% / 1%) hold on the calibration slice. On new data they hold approximately,
  and the val/test/hard sets in the report show how well.
