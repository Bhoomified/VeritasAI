"""Step 3 (approaches B & C) – train a small temporal head on cached features. Takes SECONDS-MINUTES,
runs fine on a laptop CPU.

  # Approach B: frozen backbone + temporal head
  python -m ml.train_head --feats feats/ffpp_effb0 --head gru --cross feats/celebdf_effb0/test.npz
  # Approach C: tri-stream fusion (appearance + motion + frequency)
  python -m ml.train_head --feats feats/ffpp_effb0 --head fusion --cross feats/celebdf_effb0/test.npz

Heads: mean | lstm | gru | tcn | attn | fusion.   --seeds 3 gives mean±std for the paper.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .nets import build_head, count_params, Backbone, VideoModel
from .evalutils import seed_all, pick_device, metrics, best_threshold, log_result, time_it, deploy_cost


def load(path):
    d = np.load(path)
    return (torch.from_numpy(d["rgb"].astype(np.float32)), torch.from_numpy(d["freq"].astype(np.float32)),
            torch.from_numpy(d["y"].astype(np.float32)))


def augment(rgb, frq, p_drop=0.15, noise=0.05):
    """Cheap feature-space augmentation: random frame dropout (replace by neighbour) + noise."""
    B, T = rgb.shape[:2]
    keep = torch.rand(B, T, device=rgb.device) > p_drop
    keep[:, 0] = True
    idx = torch.arange(T, device=rgb.device).expand(B, T) * keep
    idx = torch.cummax(idx, 1).values                       # dropped frame -> previous kept frame
    rgb = torch.gather(rgb, 1, idx.unsqueeze(-1).expand_as(rgb))
    frq = torch.gather(frq, 1, idx.unsqueeze(-1).expand_as(frq))
    return rgb + noise * torch.randn_like(rgb) * rgb.std(), frq


@torch.no_grad()
def predict(model, rgb, frq, dev, bs=256):
    model.eval(); out = []
    for i in range(0, len(rgb), bs):
        out.append(torch.sigmoid(model(rgb[i:i + bs].to(dev), frq[i:i + bs].to(dev))[0]).cpu())
    return torch.cat(out).numpy()


def run(a, seed):
    seed_all(seed); dev = pick_device(a.device)
    feats = Path(a.feats)
    Xtr, Ftr, ytr = load(feats / "train.npz"); Xva, Fva, yva = load(feats / "val.npz")
    model = build_head(a.head, Xtr.shape[-1], freq_dim=Ftr.shape[-1], hid=a.hid, drop=a.drop, streams=a.streams).to(dev)
    pos_w = torch.tensor((len(ytr) - ytr.sum()) / max(ytr.sum(), 1), device=dev)   # handles class imbalance
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * (len(ytr) // a.bs + 1))
    best, best_state, bad = -1, None, 0
    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(len(ytr))
        for i in range(0, len(ytr), a.bs):
            b = perm[i:i + a.bs]
            x, f, y = Xtr[b].to(dev), Ftr[b].to(dev), ytr[b].to(dev)
            x, f = augment(x, f)
            logit, extra = model(x, f)
            loss = lossf(logit, y)
            if "aux" in extra:                                   # each stream must be useful on its own
                loss = loss + a.aux_w * lossf(extra["aux"], y.unsqueeze(1).expand_as(extra["aux"]))
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        pva = predict(model, Xva, Fva, dev)
        auc = metrics(yva.numpy(), pva)["auc"]
        if auc > best + 1e-4:
            best, bad = auc, 0; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"seed {seed} epoch {ep + 1:02d} loss {loss.item():.4f} val_auc {auc:.4f}")
        if bad >= a.patience:
            break
    model.load_state_dict(best_state)
    thr = best_threshold(yva.numpy(), predict(model, Xva, Fva, dev))
    res = {}
    for name, path in [("test", feats / "test.npz")] + [(f"cross{i}", p) for i, p in enumerate(a.cross or [])]:
        X, F, y = load(path); res[name] = metrics(y.numpy(), predict(model, X, F, dev), thr)
    T = Xtr.shape[1]
    x1, f1 = Xtr[:1].to(dev), Ftr[:1].to(dev); model.eval()
    ms = time_it(lambda: model(x1, f1))
    return model, thr, res, ms, T


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True); ap.add_argument("--head", default="gru")
    ap.add_argument("--cross", nargs="*", help="npz files from OTHER datasets (cross-dataset test)")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--drop", type=float, default=0.3); ap.add_argument("--aux-w", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=8); ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--streams", default="amf", help="fusion ablation: any of a(ppearance) m(otion) f(requency)")
    ap.add_argument("--device", default="auto"); ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="checkpoints"); ap.add_argument("--results", default="results.csv")
    a = ap.parse_args()
    bb = torch.load(Path(a.feats) / "backbone.pt", map_location="cpu", weights_only=True)
    all_res, best_auc = [], -1
    for s in range(a.seeds):
        model, thr, res, ms, T = run(a, 42 + s)
        all_res.append(res)
        if s == 0 or res["test"]["auc"] > best_auc:
            best_auc = res["test"]["auc"]
            Path(a.out).mkdir(parents=True, exist_ok=True)
            ck = Path(a.out) / f"{bb['name']}_{a.head}{a.tag}.pt"
            torch.save({"format": "veritas-v2", "kind": "head", "backbone": bb["name"], "backbone_state": bb["state"],
                        "head": a.head, "head_kwargs": {"hid": a.hid, "drop": a.drop, "streams": a.streams}, "head_state": model.state_dict(),
                        "num_frames": T, "img_size": 224, "threshold": thr, "metrics": res,
                        "model_version": f"veritas-{bb['name']}-{a.head}{a.tag}"}, ck)
    full = Backbone(bb["name"], pretrained=False); full.load_state_dict(bb["state"])
    row = {"approach": "C" if a.head == "fusion" else "B", "backbone": bb["name"], "head": a.head + a.tag,
           "dataset": Path(a.feats).name, "head_params": count_params(model), "head_ms": round(ms, 3),
           **deploy_cost(VideoModel(full, model.cpu()), T)}
    for split in all_res[0]:
        for k in ("auc", "acc", "f1", "eer"):
            v = np.array([r[split][k] for r in all_res])
            row[f"{split}_{k}"] = f"{v.mean():.4f}" + (f"±{v.std():.4f}" if len(v) > 1 else "")
    log_result(a.results, row)
    print("saved", ck)
