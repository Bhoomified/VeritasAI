"""Metrics used in every experiment (same protocol for all 4 approaches)."""
import csv, json, random, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve, balanced_accuracy_score)


def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def pick_device(pref="auto"):
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def best_threshold(y, p):
    """Threshold maximising Youden's J (TPR-FPR) on the VALIDATION set."""
    if len(set(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    return float(np.clip(thr[np.argmax(tpr - fpr)], 1e-4, 1 - 1e-4))


def metrics(y, p, thr=0.5):
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    out = {"acc": accuracy_score(y, pred), "bal_acc": balanced_accuracy_score(y, pred),
           "precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred, zero_division=0),
           "f1": f1_score(y, pred, zero_division=0), "auc": float("nan"), "eer": float("nan"), "n": len(y)}
    if len(set(y)) == 2:
        out["auc"] = roc_auc_score(y, p)
        fpr, tpr, _ = roc_curve(y, p); fnr = 1 - tpr
        i = np.nanargmin(np.abs(fnr - fpr)); out["eer"] = float((fpr[i] + fnr[i]) / 2)
    return {k: round(float(v), 4) if isinstance(v, float) else v for k, v in out.items()}


def log_result(path, row):
    """Append one experiment row to results.csv (the table for your paper)."""
    path = Path(path); new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row)); new and w.writeheader(); w.writerow(row)
    print(json.dumps(row, indent=1))


@torch.no_grad()
def time_it(fn, warmup=3, runs=20):
    for _ in range(warmup): fn()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(runs): fn()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return (time.perf_counter() - t) / runs * 1000  # ms


def deploy_cost(model, frames=16, threads=4):
    """Params (M), GFLOPs and CPU latency for ONE video – the deployment target is a plain CPU."""
    import copy
    from torch.utils.flop_counter import FlopCounterMode
    torch.set_num_threads(threads)
    m = copy.deepcopy(model).float().cpu().eval()
    x, f = torch.rand(1, frames, 3, 224, 224), torch.zeros(1, frames, 64)
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        m(x, f)
    ms = time_it(lambda: m(x, f), 1, 3)
    return {"params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 2),
            "gflops": round(fc.get_total_flops() / 2e9, 2), "cpu_ms_video": round(ms, 1)}
