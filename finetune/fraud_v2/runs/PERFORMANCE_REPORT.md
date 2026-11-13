# Laya SIM-swap fraud fine-tune: performance report

Model: `convaiinnovations/laya` (ModernBERT-large encoder, ~421M params, single-pass yes/no head)
Winner: run `20261113_171802_a1_6ep`, epoch 5 of 6 selected by lowest validation log-loss.
The 2-epoch continuation (`a1_cont4`) was worse and was discarded.
Task: "Does this SIM swap event look like SIM-swap fraud?" (binary, p = P(fraud)).

## 1. Headline

| Test set | What it is | Baseline (untouched Laya) | Fine-tuned | Improvement |
|---|---|---|---|---|
| **test** (6,000) | Held-out, same distribution as training | **25.6%** correct | **99.1%** correct | **+73.5 pp** (95% CI +72.3 to +74.6) |
| **hard_test** (600) | Synthetic knife-edge cases, built to be hard | **55.0%** correct | **83.2%** correct | **+28.2 pp** (95% CI +23.3 to +33.0) |

- Errors on test: 4,462 → 54 (**98.8% fewer**).
- Errors on hard_test: 270 → 101 (**62.6% fewer**).
- The model was never trained on test or hard_test. I checked for exact-duplicate leakage: 0 overlapping
  states between train and val, test, hard_dev or hard_test.

**Success rates to quote**
- Ordinary traffic: about **99% correct** (test 99.1–99.2%, val 99.4%).
- Deliberately hard boundary cases: about **83% correct**. This is the weak spot (section 4).

## 2. The baseline is not a fair "before"

Untouched Laya has no idea what a SIM-swap fraud is. It answers "not fraud" for nearly everything
(mean p(fraud) = 0.11 on a set that is 74% fraud). On test it flags 4 of 4,464 frauds. So its accuracy
is roughly the share of legit cases in each set, **below even a model that always says "fraud"**.

| Test set | Base Laya | Always-"fraud" (majority) | Fine-tuned |
|---|---|---|---|
| test | 25.6% | 74.4% | 99.1% |
| hard_test | 55.0% | 55.0% (always "legit") | 83.2% |

Measured against the stronger majority-class reference the gain is **+24.7 pp on test** and **+28.2 pp on hard_test**.
The fair statement is: fine-tuning took Laya from "no signal" (balanced accuracy 0.50) to a working classifier.

## 3. Full metrics (official pipeline run, logged in `final_report.json`)

| Split | Metric | Base | Fine-tuned | Change |
|---|---|---|---|---|
| val (3,000) | accuracy | 0.2517 | 0.9937 | +0.742 |
| | balanced accuracy | 0.5000 | 0.9918 | +0.492 |
| | log-loss | 1.7350 | 0.0201 | -1.715 |
| | fraud recall / legit recall | 0.000 / 1.000 | 0.9955 / 0.9881 | |
| test (6,000) | accuracy | 0.2565 | 0.9917 | +0.735 |
| | balanced accuracy | 0.4999 | 0.9884 | +0.489 |
| | log-loss | 1.7325 | 0.0251 | -1.707 |
| | fraud recall / legit recall | 0.0011 / 0.9987 | 0.9951 / 0.9818 | |
| hard_dev (300) | accuracy | 0.5533 | 0.8367 | +0.283 |
| | log-loss | 1.0221 | 0.4575 | -0.565 |
| hard_test (600) | accuracy | 0.5500 | 0.8333 | +0.283 |
| | balanced accuracy | 0.5000 | 0.8330 | +0.333 |
| | log-loss | 1.0378 | 0.4304 | -0.608 |
| | fraud recall / legit recall | 0.000 / 1.000 | 0.8296 / 0.8364 | |

Training curve (validation accuracy, epochs 1–6): 0.976, 0.965, 0.986, 0.985, **0.994**, 0.993.
hard_dev accuracy: 0.633, 0.740, 0.720, 0.737, **0.837**, 0.857.

## 4. Where it still fails (hard_test accuracy by category, base → fine-tuned)

| Category | n | Base | Fine-tuned |
|---|---|---|---|
| far_but_flagged | 60 | 50% | **100%** |
| distractor_age | 70 | 50% | **97%** |
| quiet_takeover | 60 | 0% | **95%** |
| behaviour_storm | 60 | 100% | 83–85% (worse) |
| two_changes_benign | 60 | 100% | 83% (worse) |
| flip_pair | 120 | 50% | 76–77% |
| knife_edge_double | 80 | 50% | 72–74% |
| knife_edge_single | 90 | 50% | 71% |

- The weakest categories are cases a hair's breadth from the decision line (the generator's margin is
  often under 0.05). These are the rows where a number-reading text encoder is least reliable.
- **Two categories got worse than baseline** (behaviour_storm, two_changes_benign). The base model scored
  100% there only because it says "legit" for everything and those cases are legit. That is not skill. The
  fine-tuned model now raises some false alarms there.

## 5. Production gate (BLOCK / ALLOW / send to human)

Cut-offs fitted on a held-out calibration slice: ALLOW if p ≤ 0.1197, BLOCK if p ≥ 0.8828.

| Split | Blocked | Allowed | To human | Fraud leaked (auto-allowed) | Legit auto-blocked |
|---|---|---|---|---|---|
| test (6,000) | 4,431 | 1,487 | 82 (1.4%) | 11 (0.25% of fraud) | 13 (0.85% of legit) |
| hard_test (600) | 216 | 253 | 131 (21.8%) | 21 | 27 |

- On normal traffic, the gate auto-decides 98.6% of cases at about **99.6% accuracy**. Humans see 1.4%.
- On hard cases it sends 22% to humans, and 48 of 600 (8%) are still confidently wrong.
- The 0.2% fraud-leak budget held on test (0.25%, a touch over). It did not hold on hard_test, which is
  an out-of-distribution stress set.

## 6. Independent re-verification (done for this report)

I re-ran base and fine-tuned models from the saved checkpoint on test and hard_test and compared
with the logged numbers (`runs/independent_verify.json`, script `tools/independent_verify.py`).

| | Logged | My re-run |
|---|---|---|
| test accuracy | 0.9917 | 0.9910 (95% CI 0.988–0.993) |
| hard_test accuracy | 0.8333 | 0.8317 (batch 64), 0.8333 (batch 1) |
| base test / hard_test | 0.2565 / 0.5500 | 0.2563 / 0.5500 |

The numbers agree within 1 case in 600 on hard_test and 0.07 pp on test. One finding to flag:

**Predictions are not perfectly batch-invariant.** Scoring the same 600 cases at batch size 1, 16 and 64
changes individual probabilities by up to 0.36, and 7 of 600 decisions flip between batch 1 and batch 64.
All flips are borderline cases (p between 0.41 and 0.57 at batch 1). Aggregate accuracy stays within
0.830–0.833, so the headline is stable, but a single borderline case can get a different answer depending
on how requests were batched. Likely cause is padding/precision effects; I did not isolate it. Treat any
single decision with p in about 0.4–0.6 as undetermined, which the gate's human-review band already does.

## 7. Speed and export
- GPU (RTX 3050): single request p50 39.1 ms (base 38.6 ms), batched 13.4 ms/item. Fine-tuning did not
  slow inference.
- ONNX export verified against PyTorch on all 600 hard cases: decision agreement 1.0, max probability
  difference 1e-4. On CPU, ONNX is 380 ms/case against 510 ms for PyTorch.

## 8. Targets: met or missed

| Target | Result | Status |
|---|---|---|
| val accuracy ≥ 99.5% | 99.37% | missed by 0.13 pp |
| val log-loss ≤ 0.02 | 0.0201 | effectively met |
| hard_dev accuracy ≥ 97% | 83.7% | **not met** |
| val fraud-leak ≤ 0.5% | 5 of 2,245 (0.22%) | met |

## 9. Honest limits
- Labels in this dataset follow a fixed rule over the fields, and hard_test is generated from the same
  rule. These numbers show how well Laya learned that rule, not how it will do against real fraudsters.
  Real-world performance is untested.
- Test-set accuracy (99.1%) is near the ceiling for the training approach. The remaining gap is in
  boundary cases, which more epochs did not fix (the continuation made hard_test worse: 79.8%).
- hard_dev was used only as a tie-break for checkpoint choice. hard_test and test were not used for any
  tuning.

## 10. What I would do next
1. Add more training rows near the decision boundary (the planned `a3_more_data` rung with 72k rows was never run; see WORKLOG step 8), which is where all the remaining errors are.
2. Keep the human-review band. It is doing real work on exactly the cases the model finds hard.
3. Do not ship fraud decisions that depend on a single borderline score without the gate.
