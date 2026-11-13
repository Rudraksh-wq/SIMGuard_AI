"""Fine-tune Laya for SIM-swap fraud: baseline first, full logging, per-epoch checkpoints,
tipping-point stop, calibrated final checkpoint, before/after comparison and speed check.

Everything a run does is written under  runs/<run_id>/ :
  config.json          every knob, the data files and their sha256, library versions, device
  baseline.json        the untouched base model on val / test / hard_dev / hard_test
  epochs.jsonl         one line per epoch: train loss, val + hard_dev metrics, attention health,
                       weight drift from the base, lr, wall time, decision taken
  checkpoints/epoch_XX one checkpoint per epoch (older non-best ones pruned unless --keep-all)
  final/               the selected checkpoint, calibrated, loads with laya.load(path)
  final_report.json    before vs after on every split, per hard-test category, speed, why it stopped
  hard_test_predictions.csv   every hard case with p(fraud), prediction, correct/incorrect
and one line per run is appended to runs/attempts_ledger.jsonl (including failed runs).

What is used for what (no peeking):
  train    -> gradient updates
  calib    -> 10% slice of train held out *before* training, only for temperature fitting
  val      -> early stopping / checkpoint selection
  hard_dev -> monitored every epoch, used as a secondary selection signal
  test, hard_test -> evaluated only for the baseline and for the final selected checkpoint
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
from dataclasses import asdict

import numpy as np
import torch

from laya import train as LT
from laya.calibrate import fit_abstention_thresholds, fit_temperature_map
from laya.common import QTYPE_NAMES, collate_items, temp_bucket

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fraud_common import QUESTION, QUESTION_ID  # noqa: E402
from gate import fit_gate, gate_metrics  # noqa: E402


class StopTraining(Exception):
    pass


# ------------------------------------------------------------------------------ small utils
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jdump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)
    os.replace(tmp, path)


def jappend(obj, path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=float) + "\n")


def log(msg, path=None):
    line = "[%s] %s" % (dt.datetime.now().strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def read_rows(path):
    rows = LT.read_jsonl(path)
    for r in rows:   # same fixed question for every split
        r["questions"] = {QUESTION_ID: dict(QUESTION)}
    return rows


# ---------------------------------------------------------------------------------- metrics
def probs_true(records, temperature=None, by_options=None):
    out = []
    for qt, logits, _t, k in records:
        z = np.asarray(logits, dtype=float)[:k]
        t = 1.0
        if temperature is not None:
            t = float(temperature[qt])
            if by_options and temp_bucket(qt, k) in by_options:
                t = float(by_options[temp_bucket(qt, k)])
        z = z / t
        p = np.exp(z - np.logaddexp.reduce(z))
        out.append(float(p[1]))          # noul options are [false, true]
    return np.array(out)


def binary_metrics(p, y, thr=0.5):
    y = np.asarray(y, dtype=int)
    pred = (p > thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    return {
        "n": int(len(y)), "accuracy": (tp + tn) / max(1, len(y)),
        "log_loss": float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()),
        "brier": float(((p - y) ** 2).mean()),
        "precision_fraud": prec, "recall_fraud": rec,
        "f1_fraud": 2 * prec * rec / max(1e-9, prec + rec),
        "recall_legit": tn / max(1, tn + fp),
        "balanced_accuracy": 0.5 * (rec + tn / max(1, tn + fp)),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def evaluate(model, tok, items, labels, dev, max_len, head_max_len, temperature=None, by_options=None,
             cats=None):
    records = LT.calibration_records(model, tok, items, dev, max_len, head_max_len)
    p = probs_true(records, temperature, by_options)
    m = binary_metrics(p, labels)
    cs = LT.collapse_stats(records)
    m["mean_logit_range"] = cs["mean_logit_range"] if cs else None
    if cats is not None:
        m["by_category"] = {}
        for c in sorted(set(cats)):
            idx = [i for i, x in enumerate(cats) if x == c]
            m["by_category"][c] = {"n": len(idx),
                                   "accuracy": float(((p[idx] > .5).astype(int) == np.asarray(labels)[idx]).mean())}
    return m, p


# --------------------------------------------------------------------- attention / drift health
def probe_batch(tok, items, max_len, head_max_len, n=6):  # eager attn maps are big: keep tiny for 6 GB
    chunk = [LT.encode_item(tok, it, max_len, head_max_len) for it in items[:n]]
    return collate_items([chunk], tok.pad_token_id)


@torch.no_grad()
def attention_health(model, batch, dev):
    """Per-layer normalised attention entropy (1 = uniform, 0 = one token) and last-layer
    token-to-token cosine (rank collapse / over-smoothing). Eager attention is switched on only
    for this probe and switched back afterwards, so training speed is unaffected."""
    enc = model.encoder
    was_training = model.training
    model.eval()
    ids = batch["input_ids"].to(dev); att = batch["attention_mask"].to(dev)
    out = {"attn_entropy_by_layer": None, "attn_entropy_mean": None, "attn_error": None}
    prev = getattr(enc.config, "_attn_implementation", None)
    try:
        if hasattr(enc, "set_attn_implementation"):
            enc.set_attn_implementation("eager")
        else:
            enc.config._attn_implementation = "eager"
        o = enc(input_ids=ids, attention_mask=att, output_attentions=True, output_hidden_states=True)
        valid = att.bool()
        ents = []
        for a in o.attentions:                    # [B, H, L, L]
            a = a.float().clamp_min(1e-12)
            n_keys = valid.sum(-1).clamp(min=2).float()                  # [B]
            e = -(a * a.log()).sum(-1) / n_keys.log()[:, None, None]     # [B, H, L]
            mask = valid[:, None, :].expand_as(e)
            ents.append(float(e[mask].mean()))
        out["attn_entropy_by_layer"] = [round(x, 5) for x in ents]
        out["attn_entropy_mean"] = float(np.mean(ents))
        hs = o.hidden_states[-1].float()
    except Exception as e:  # never let the probe kill a run
        out["attn_error"] = "%s: %s" % (type(e).__name__, e)
        hs = enc(input_ids=ids, attention_mask=att, output_hidden_states=True).hidden_states[-1].float()
    finally:
        try:
            if hasattr(enc, "set_attn_implementation") and prev:
                enc.set_attn_implementation(prev)
            elif prev:
                enc.config._attn_implementation = prev
        except Exception:
            pass
    cos = []
    for b in range(hs.size(0)):
        h = torch.nn.functional.normalize(hs[b][att[b].bool()], dim=-1)
        s = h @ h.T
        n = s.size(0)
        cos.append(float((s.sum() - n) / max(1, n * (n - 1))))
    out["token_cosine_last_layer"] = float(np.mean(cos))
    if was_training:
        model.train()
    return out


def weight_drift(model, base_snapshot):
    num = den = 0.0
    for n, p in model.encoder.named_parameters():
        b = base_snapshot[n].to(p.device, torch.float32)
        num += float((p.detach().float() - b).pow(2).sum()); den += float(b.pow(2).sum())
    return math.sqrt(num / max(den, 1e-12))


# ------------------------------------------------------------------------------------ speed
def speed(model_dir, states, device, n_single=100):
    import laya
    agent = laya.load(model_dir, device=device)
    q = {QUESTION_ID: dict(QUESTION)}
    for s in states[:5]:
        agent.predict(s, q)
    if device == "cuda":
        torch.cuda.synchronize()
    lat = []
    for s in states[:n_single]:
        t = time.perf_counter(); agent.predict(s, q)
        if device == "cuda":
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t) * 1000)
    t = time.perf_counter(); agent.predict_batch(states[:256], q, batch_size=64)
    if device == "cuda":
        torch.cuda.synchronize()
    batch_ms = (time.perf_counter() - t) * 1000 / min(256, len(states))
    del agent
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"single_p50_ms": float(np.percentile(lat, 50)), "single_p95_ms": float(np.percentile(lat, 95)),
            "batched_ms_per_item": batch_ms, "n_single": len(lat)}


# ------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="convaiinnovations/laya",
                    help="base checkpoint: Hub id (uses your HF cache) or local dir")
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--tag", default="a1_no_bias")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--min-epochs", type=int, default=12)
    ap.add_argument("--max-epochs", type=int, default=16)
    ap.add_argument("--patience", type=int, default=2, help="epochs without val improvement before stop")
    ap.add_argument("--micro-batch", type=int, default=4, help="4 fits an RTX 3050 6 GB")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--encoder-lr", type=float, default=1e-5, help="gentle on the base (laya default 2.5e-5)")
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--loss", default="soft-ce", choices=["soft-ce", "rlcd"])
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--freeze-bottom", type=int, default=14,
                    help="freeze embeddings + the bottom N encoder layers (protects the base, saves VRAM)")
    ap.add_argument("--freeze-encoder", action="store_true", help="head-only training (low-VRAM fallback)")
    ap.add_argument("--train-limit", type=int, default=0, help="use only the first N train rows (0 = all)")
    ap.add_argument("--balance-legit-to", type=float, default=0.0,
                    help="BIAS KNOB, off by default: oversample legit rows until they are this share of train")
    ap.add_argument("--attn-drop-limit", type=float, default=0.30,
                    help="tipping flag if mean attention entropy falls by more than this fraction of the base")
    ap.add_argument("--cos-rise-limit", type=float, default=0.15,
                    help="tipping flag if last-layer token cosine rises by more than this (over-smoothing)")
    ap.add_argument("--max-drift", type=float, default=0.10,
                    help="tipping flag if relative encoder weight drift from base exceeds this")
    ap.add_argument("--max-fraud-leak", type=float, default=0.002,
                    help="GATE BIAS (deliberate): max share of real fraud that may be auto-ALLOWED")
    ap.add_argument("--max-false-block", type=float, default=0.01,
                    help="GATE: max share of real legit that may be auto-BLOCKED (looser than fraud leak)")
    ap.add_argument("--keep-all", action="store_true")
    ap.add_argument("--skip-speed", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + args.tag
    run_dir = os.path.join(args.runs, run_id)
    ck_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ck_dir, exist_ok=True)
    LOG = os.path.join(run_dir, "train.log")
    EPOCHS = os.path.join(run_dir, "epochs.jsonl")
    LEDGER = os.path.join(args.runs, "attempts_ledger.jsonl")
    ledger = {"run_id": run_id, "tag": args.tag, "started": dt.datetime.now().isoformat(),
              "args": vars(args), "status": "started"}
    try:
        _run(args, run_dir, ck_dir, LOG, EPOCHS, ledger)
        ledger["status"] = "finished"
    except Exception as e:
        ledger["status"] = "failed"
        ledger["error"] = "%s: %s" % (type(e).__name__, e)
        log(traceback.format_exc(), LOG)
        raise
    finally:
        ledger["ended"] = dt.datetime.now().isoformat()
        jappend(ledger, LEDGER)


def _run(args, run_dir, ck_dir, LOG, EPOCHS, ledger):
    dev = LT.resolve_device(args.device)
    dev_name = torch.cuda.get_device_name(0) if dev.type == "cuda" else str(dev)
    vram = torch.cuda.get_device_properties(0).total_memory / 2**30 if dev.type == "cuda" else None
    log("device %s (%s, %s GB)" % (dev, dev_name, None if vram is None else round(vram, 1)), LOG)

    files = {s: os.path.join(args.data, s + ".jsonl") for s in ("train", "val", "test", "hard_dev", "hard_test")}
    for s, p in files.items():
        if not os.path.exists(p):
            raise SystemExit("missing %s - run prepare_data.py and gen_hard_tests.py first" % p)

    model_dir = LT.resolve_checkpoint_dir(args.base)
    model, tok, cfg = LT.load_checkpoint(model_dir)
    model.to(dev)
    max_len, head_max_len = cfg.get("max_len", 512), cfg.get("head_max_len", 192)
    base_T, base_TBO = cfg.get("temperature"), cfg.get("temperature_by_options")

    import laya, transformers
    config = {"run_dir": run_dir, "args": vars(args), "base_model_dir": model_dir,
              "base_rl_agent_config": {k: cfg[k] for k in cfg if k not in ("training",)},
              "data_files": {s: {"path": p, "sha256": sha256(p)} for s, p in files.items()},
              "versions": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
                           "laya": getattr(laya, "__version__", "?"), "platform": platform.platform()},
              "device": {"type": dev.type, "name": dev_name, "vram_gb": vram},
              "usage_of_splits": {"train": "gradient", "calib": "temperature fit only (held out before training)",
                                  "val": "early stopping + selection", "hard_dev": "monitoring + tie-break",
                                  "test/hard_test": "baseline and final only"}}

    # ---- data -> items
    rows = {s: read_rows(p) for s, p in files.items()}
    if args.train_limit:
        rows["train"] = rows["train"][:args.train_limit]
    items, labels, cats = {}, {}, {}
    for s, rr in rows.items():
        it, skipped = LT.items_from_rows(tok, rr, max_len, head_max_len,
                                         label_smoothing=args.label_smoothing if s == "train" else 0.0)
        if len(it) != len(rr):
            raise SystemExit("%s: %d rows -> %d items (skipped %r)" % (s, len(rr), len(it), skipped))
        items[s] = it
        labels[s] = [int(r["gold"][QUESTION_ID]["probabilities"]["true"] > .5) for r in rr]
        cats[s] = [r.get("meta", {}).get("category") for r in rr] if s.startswith("hard") else None
    train_items, calib_items = LT.split_calibration(items["train"], 2000, 0.1, 20260922)

    bias_note = "none"
    if args.balance_legit_to > 0:   # explicit, logged bias knob
        legit = [it for it in train_items if it["target"][1] < .5]
        share = len(legit) / len(train_items)
        if share < args.balance_legit_to:
            need = int((args.balance_legit_to * len(train_items) - len(legit)) / (1 - args.balance_legit_to))
            rng = np.random.default_rng(args.seed)
            train_items = train_items + [legit[i] for i in rng.integers(0, len(legit), need)]
            bias_note = "oversampled %d legit rows: legit share %.3f -> %.3f" % (need, share, args.balance_legit_to)
    config["bias"] = bias_note
    config["sizes"] = {"train_items": len(train_items), "calib_items": len(calib_items),
                       **{s: len(items[s]) for s in ("val", "test", "hard_dev", "hard_test")}}
    log("bias: %s | sizes %s" % (bias_note, config["sizes"]), LOG)

    # ---- protect the base: freeze embeddings + bottom layers
    frozen = []
    if args.freeze_bottom and not args.freeze_encoder:
        layers = getattr(model.encoder, "layers", None)
        for n, p in model.encoder.named_parameters():
            if n.startswith("embeddings."):
                p.requires_grad_(False); frozen.append(n)
        if layers is not None:
            for i in range(min(args.freeze_bottom, len(layers))):
                for n, p in layers[i].named_parameters():
                    p.requires_grad_(False); frozen.append("layers.%d.%s" % (i, n))
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) if not args.freeze_encoder else \
        sum(p.numel() for n, p in model.named_parameters() if not n.startswith("encoder."))
    config["trainable_params"] = {"total": n_total, "trainable": n_train, "frozen_tensors": len(frozen)}
    jdump(config, os.path.join(run_dir, "config.json"))
    log("params total %.1fM, trainable %.1fM" % (n_total / 1e6, n_train / 1e6), LOG)

    # ---- baseline (untouched base model), with the base's own calibration
    log("baseline evaluation ...", LOG)
    t0 = time.time()
    baseline, base_p = {}, {}
    for s in ("val", "test", "hard_dev", "hard_test"):
        m, base_p[s] = evaluate(model, tok, items[s], labels[s], dev, max_len, head_max_len, base_T, base_TBO, cats[s])
        baseline[s] = m
        log("  base %-9s acc %.4f  loss %.4f  bal_acc %.4f" % (s, m["accuracy"], m["log_loss"], m["balanced_accuracy"]), LOG)
    # same gate-fitting procedure on the untouched base model (calibration slice only)
    crec = LT.calibration_records(model, tok, calib_items, dev, max_len, head_max_len)
    base_gate = fit_gate(probs_true(crec, base_T, base_TBO), [int(it["target"][1] > .5) for it in calib_items],
                         args.max_fraud_leak, args.max_false_block)
    baseline["gate_thresholds"] = base_gate
    for s in base_p:
        baseline[s]["gate"] = gate_metrics(base_p[s], labels[s], base_gate)
        g = baseline[s]["gate"]
        log("  base %-9s gate: block %d allow %d noul %d | fraud_leak %d false_block %d" % (
            s, g["block"], g["allow"], g["noul"], g["fraud_leak"], g["false_block"]), LOG)
    probe = probe_batch(tok, items["val"], max_len, head_max_len)
    baseline["attention"] = attention_health(model, probe, dev)
    torch.cuda.empty_cache() if dev.type == "cuda" else None
    baseline["eval_seconds"] = time.time() - t0
    jdump(baseline, os.path.join(run_dir, "baseline.json"))
    base_snapshot = {n: p.detach().to("cpu", torch.float16) for n, p in model.encoder.named_parameters()}
    base_speed = None
    if not args.skip_speed:
        model.to("cpu")   # free VRAM while the speed probe loads its own copy
        torch.cuda.empty_cache() if dev.type == "cuda" else None
        base_speed = speed(model_dir, [r["state"] for r in rows["test"][:300]], dev.type)
        log("  base speed %s" % base_speed, LOG)
    model.to(dev)

    # ---- training with per-epoch evaluation and tipping-point detection
    tc = LT.TrainConfig(epochs=args.max_epochs, micro_batch=args.micro_batch, grad_accum=args.grad_accum,
                        encoder_lr=args.encoder_lr, head_lr=args.head_lr, weight_decay=args.weight_decay,
                        loss=args.loss, freeze_encoder=args.freeze_encoder, seed=args.seed,
                        label_smoothing=args.label_smoothing, log_every=200)
    state = {"best": None, "best_epoch": None, "since_best": 0, "stop_reason": None, "tip_epoch": None,
             "kept": [], "t": time.time()}
    a0 = baseline["attention"]

    def on_epoch_end(epoch, train_loss):
        e = epoch + 1
        vm, _ = evaluate(model, tok, items["val"], labels["val"], dev, max_len, head_max_len)
        hm, _ = evaluate(model, tok, items["hard_dev"], labels["hard_dev"], dev, max_len, head_max_len,
                         cats=cats["hard_dev"])
        ah = attention_health(model, probe, dev)
        drift = weight_drift(model, base_snapshot)
        collapse, drift_flag = [], []
        if ah["attn_entropy_mean"] is not None and a0["attn_entropy_mean"]:
            rel = (a0["attn_entropy_mean"] - ah["attn_entropy_mean"]) / a0["attn_entropy_mean"]
            if rel > args.attn_drop_limit:
                collapse.append("attention_entropy_drop_%.2f" % rel)
        if ah["token_cosine_last_layer"] - a0["token_cosine_last_layer"] > args.cos_rise_limit:
            collapse.append("over_smoothing")
        if vm["mean_logit_range"] is not None and vm["mean_logit_range"] < 0.05:
            collapse.append("prior_collapse")
        if drift > args.max_drift:
            drift_flag.append("weight_drift_%.3f" % drift)
        flags = collapse + drift_flag
        # selection key: val log-loss, ties broken by hard_dev accuracy.
        # An epoch whose attention has collapsed is never eligible, even if val loss improved.
        key = (round(vm["log_loss"], 5), -hm["accuracy"])
        improved = (state["best"] is None or key < state["best"]) and not collapse
        ck = os.path.join(ck_dir, "epoch_%02d" % e)
        LT.save_checkpoint(model, tok, dict(cfg, max_len=max_len, head_max_len=head_max_len), ck)
        state["kept"].append(ck)
        if improved:
            state["best"], state["best_epoch"], state["since_best"] = key, e, 0
        else:
            state["since_best"] += 1
        if (collapse or (drift_flag and not improved)) and state["tip_epoch"] is None:
            state["tip_epoch"] = e
        decision = "continue"
        if e >= args.min_epochs:
            if state["since_best"] >= args.patience:
                decision = "stop: tipping point - no val improvement for %d epochs" % args.patience
            elif collapse:
                decision = "stop: tipping point - attention/representation degradation (%s)" % ",".join(collapse)
            elif drift_flag and not improved:
                decision = "stop: tipping point - base drift %.3f > %.3f without improvement" % (drift, args.max_drift)
        if flags and decision == "continue":
            decision = "continue (health flags noted; below min_epochs or still improving)"
        rec = {"epoch": e, "train_loss": train_loss, "val": vm, "hard_dev": hm, "attention": ah,
               "weight_drift": drift, "health_flags": flags, "improved": improved,
               "best_epoch": state["best_epoch"], "lr_encoder": args.encoder_lr, "decision": decision,
               "elapsed_min": (time.time() - state["t"]) / 60}
        jappend(rec, EPOCHS)
        log("epoch %02d train %.4f | val loss %.4f acc %.4f | hard_dev acc %.4f | attnH %s cos %.3f drift %.4f %s | %s"
            % (e, train_loss, vm["log_loss"], vm["accuracy"], hm["accuracy"],
               None if ah["attn_entropy_mean"] is None else round(ah["attn_entropy_mean"], 4),
               ah["token_cosine_last_layer"], drift, flags or "", decision), LOG)
        if not args.keep_all:   # keep best + last 2 to bound disk use; log deletions
            for old in list(state["kept"]):
                keep = {os.path.join(ck_dir, "epoch_%02d" % state["best_epoch"])} if state["best_epoch"] else set()
                keep |= set(state["kept"][-2:])
                if old not in keep:
                    shutil.rmtree(old, ignore_errors=True); state["kept"].remove(old)
                    log("  pruned %s" % os.path.basename(old), LOG)
        model.train()
        if args.freeze_encoder:
            model.encoder.eval()
        if decision.startswith("stop"):
            state["stop_reason"] = decision
            raise StopTraining()

    log(LT.update_budget_message(len(train_items), tc), LOG)
    try:
        LT.train_model(model, tok, train_items, tc, dev, max_len, head_max_len, on_epoch_end=on_epoch_end)
        state["stop_reason"] = state["stop_reason"] or "reached max epochs (%d)" % args.max_epochs
    except StopTraining:
        pass
    if state["best_epoch"] is None:
        raise RuntimeError("no healthy improving epoch - nothing to promote; see epochs.jsonl")
    log("%s | selected epoch %d" % (state["stop_reason"], state["best_epoch"]), LOG)

    # ---- promote best epoch, calibrate on the held-out calib slice, save final
    best_dir = os.path.join(ck_dir, "epoch_%02d" % state["best_epoch"])
    del model
    torch.cuda.empty_cache() if dev.type == "cuda" else None
    model, tok, _ = LT.load_checkpoint(best_dir)
    model.to(dev).eval()
    recs = LT.calibration_records(model, tok, calib_items, dev, max_len, head_max_len)
    fitted = fit_temperature_map(recs)
    calibration = LT.calibration_report(recs, fitted["temperature"])
    try:
        abst = fit_abstention_thresholds(recs, temperature=fitted["temperature"],
                                         temperature_by_options=fitted["temperature_by_options"],
                                         target_error=0.10, min_bucket_n=10)
    except Exception as e:
        abst = {}; log("abstention fit failed: %s" % e, LOG)
    T, TBO = fitted["temperature"], fitted["temperature_by_options"]

    after, hard_preds = {}, None
    gate = fit_gate(probs_true(recs, T, TBO), [int(it["target"][1] > .5) for it in calib_items],
                    args.max_fraud_leak, args.max_false_block)
    log("gate fitted on %d held-out calibration rows: ALLOW if p<=%.5f, BLOCK if p>=%.5f, else NOUL (%s)"
        % (len(calib_items), gate["lo"], gate["hi"], gate["bias"]), LOG)
    for s in ("val", "test", "hard_dev", "hard_test"):
        m, p = evaluate(model, tok, items[s], labels[s], dev, max_len, head_max_len, T, TBO, cats[s])
        m["gate"] = gate_metrics(p, labels[s], gate)
        after[s] = m
        if s == "hard_test":
            hard_preds = p
    after["attention"] = attention_health(model, probe, dev)

    final_dir = os.path.join(run_dir, "final")
    out_cfg = dict(cfg, max_len=max_len, head_max_len=head_max_len, fine_tuned=True, temperature=T)
    out_cfg.pop("temperature_by_options", None)
    if TBO:
        out_cfg["temperature_by_options"] = TBO
    out_cfg["training"] = dict(out_cfg.get("training") or {},
                               laya_train=asdict(tc), laya_train_calibration=calibration,
                               fraud_run={"run_id": os.path.basename(run_dir), "selected_epoch": state["best_epoch"],
                                          "stop_reason": state["stop_reason"], "bias": bias_note,
                                          "gate": gate})
    if abst:
        out_cfg["training"]["abstention_thresholds"] = abst
    LT.save_checkpoint(model, tok, out_cfg, final_dir)
    jdump({QUESTION_ID: dict(QUESTION)}, os.path.join(final_dir, "questions.json"))
    jdump(gate, os.path.join(final_dir, "gate.json"))

    # hard-test case file
    import csv
    with open(os.path.join(run_dir, "hard_test_predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "category", "margin", "label", "p_fraud", "pred", "correct",
                    "gate_decision", "gate_outcome"])
        from gate import route
        for r, y, p in zip(rows["hard_test"], labels["hard_test"], hard_preds):
            mt = r.get("meta", {})
            d = route(p, gate)
            outcome = ("human_review" if d == "NOUL" else
                       "correct" if (d == "BLOCK") == bool(y) else
                       "FRAUD_LEAKED" if y == 1 else "FALSE_BLOCK")
            w.writerow([mt.get("case_id"), mt.get("category"), mt.get("margin"), y, round(float(p), 5),
                        int(p > .5), int(int(p > .5) == y), d, outcome])

    final_speed = None
    if not args.skip_speed:
        del model
        torch.cuda.empty_cache() if dev.type == "cuda" else None
        final_speed = speed(final_dir, [r["state"] for r in rows["test"][:300]], dev.type)

    def delta(s, k):
        return round(after[s][k] - baseline[s][k], 5)

    report = {
        "run_id": os.path.basename(run_dir), "selected_epoch": state["best_epoch"],
        "tipping_flag_first_epoch": state["tip_epoch"], "stop_reason": state["stop_reason"],
        "bias": bias_note, "temperature": T, "temperature_by_options": TBO, "abstention_thresholds": abst,
        "baseline": baseline, "after": after,
        "delta": {s: {k: delta(s, k) for k in ("accuracy", "log_loss", "balanced_accuracy", "recall_legit",
                                               "recall_fraud")} for s in ("val", "test", "hard_dev", "hard_test")},
        "gate": gate,
        "hard_test_failures": int(after["hard_test"]["confusion"]["fp"] + after["hard_test"]["confusion"]["fn"]),
        "hard_test_confident_wrong": after["hard_test"]["gate"]["confident_wrong"],
        "hard_test_sent_to_human": after["hard_test"]["gate"]["noul"],
        "speed": {"base": base_speed, "final": final_speed},
        "calibration": calibration, "final_dir": final_dir,
    }
    jdump(report, os.path.join(run_dir, "final_report.json"))
    ledger.update({"selected_epoch": state["best_epoch"], "stop_reason": state["stop_reason"], "bias": bias_note,
                   "val_loss": after["val"]["log_loss"], "val_acc": after["val"]["accuracy"],
                   "test_acc": after["test"]["accuracy"], "hard_dev_acc": after["hard_dev"]["accuracy"],
                   "hard_test_acc": after["hard_test"]["accuracy"], "final_dir": final_dir,
                   "gate": gate, "val_fraud_leak": after["val"]["gate"]["fraud_leak_rate"],
                   "hard_dev_confident_wrong_rate": after["hard_dev"]["gate"]["confident_wrong_rate"],
                   "hard_test_confident_wrong": after["hard_test"]["gate"]["confident_wrong"]})
    log("FINAL  val acc %.4f loss %.4f | test acc %.4f | hard_dev %.4f | hard_test %.4f (%d strict misses, "
        "%d confidently wrong, %d to human) | speed %s -> %s"
        % (after["val"]["accuracy"], after["val"]["log_loss"], after["test"]["accuracy"],
           after["hard_dev"]["accuracy"], after["hard_test"]["accuracy"], report["hard_test_failures"],
           report["hard_test_confident_wrong"], report["hard_test_sent_to_human"],
           base_speed and round(base_speed["single_p50_ms"], 1), final_speed and round(final_speed["single_p50_ms"], 1)),
        LOG)


if __name__ == "__main__":
    main()
