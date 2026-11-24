"""Approaches A & D – fine-tune the CNN on face strips (needs a GPU: Kaggle T4 / Colab T4 / college GPU).

  # A: frame-level EfficientNet-B0, video score = mean of frame probabilities
  python -m ml.finetune --mode frame --faces faces/ffpp --train faces/ffpp/ffpp_train.csv \
      --val faces/ffpp/ffpp_val.csv --test faces/ffpp/ffpp_test.csv --cross faces/celebdf/celebdf_test.csv
  # D: end-to-end EfficientNet-B0 (last 3 stages trainable) + temporal attention head
  python -m ml.finetune --mode video --head attn --unfreeze 3 ...same paths...

Cheap tricks used: AMP fp16, channels_last, only the last N stages trainable, 8 of the 16 cached
frames per clip during training, JPEG-compression + blur augmentation (robust to WhatsApp/YouTube re-encoding).
"""
import argparse, csv, random
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from .faces import load_strip
from .nets import Backbone, FrameModel, VideoModel, build_head, count_params
from .evalutils import seed_all, pick_device, metrics, best_threshold, log_result, time_it, deploy_cost


def aug_frame(img, rng):
    if rng.random() < 0.5:
        img = np.ascontiguousarray(img[:, ::-1])
    if rng.random() < 0.5:                                   # re-compression (social-media style)
        q = rng.randint(40, 95)
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_UNCHANGED)
    if rng.random() < 0.2:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if rng.random() < 0.3:                                   # brightness / contrast
        img = np.clip(img.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


class StripSet(Dataset):
    def __init__(self, faces_dir, index_csv, clip=8, train=False, size=224):
        self.dir, self.clip, self.train, self.size = Path(faces_dir), clip, train, size
        self.rows = list(csv.DictReader(open(index_csv)))
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]; faces = load_strip(self.dir / f"{r['uid']}.jpg", self.size)
        if self.train:
            rng = random.Random()
            T = len(faces); start = rng.randint(0, max(0, T - self.clip))
            faces = faces[start:start + self.clip]                       # random contiguous clip
            state = rng.getstate()
            out = []
            for f in faces:                                              # same aug for every frame of a clip
                rng.setstate(state); out.append(aug_frame(f, rng))
            faces = np.stack(out)
        return torch.from_numpy(faces).permute(0, 3, 1, 2), float(r["label"])


def freeze_bn(model):
    """Frozen stages keep ImageNet BN statistics (small batches would corrupt them)."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


def to_input(x, dev):
    return x.to(dev, non_blocking=True).float().div_(255)


@torch.no_grad()
def predict(model, dl, dev, amp):
    model.eval(); P, Y = [], []
    for x, y in dl:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            P.append(torch.sigmoid(model(to_input(x, dev)).float()).cpu())
        Y.append(y)
    return torch.cat(Y).numpy(), torch.cat(P).numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["frame", "video"], default="video")
    ap.add_argument("--faces", required=True, help="dir with strips of the TRAIN dataset")
    ap.add_argument("--train", required=True); ap.add_argument("--val", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--cross", nargs="*", default=[], help="index CSVs of other datasets (strips next to the CSV)")
    ap.add_argument("--backbone", default="efficientnet_b0"); ap.add_argument("--head", default="attn")
    ap.add_argument("--unfreeze", type=int, default=3, help="# last backbone stages to train (0 = frozen)")
    ap.add_argument("--clip", type=int, default=8); ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--device", default="auto"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="checkpoints"); ap.add_argument("--results", default="results.csv")
    a = ap.parse_args()
    if a.head == "fusion":
        raise SystemExit("fusion head needs cached frequency features: use ml.train_head (approach C)")
    seed_all(a.seed); dev = pick_device(a.device); amp = dev.type == "cuda"

    bb = Backbone(a.backbone, pretrained=True); bb.freeze(a.unfreeze if a.unfreeze else 0)
    if a.mode == "frame":
        model = FrameModel(bb)
    else:
        model = VideoModel(bb, build_head(a.head, bb.dim, freq_dim=64))
    model = model.to(dev).to(memory_format=torch.channels_last)

    dl_tr = DataLoader(StripSet(a.faces, a.train, a.clip if a.mode == "video" else 4, True),
                       a.bs, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=amp, persistent_workers=a.workers > 0)
    dl_va = DataLoader(StripSet(a.faces, a.val), a.bs, num_workers=a.workers)
    ytr = np.array([float(r["label"]) for r in dl_tr.dataset.rows])
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(ytr) - ytr.sum()) / max(ytr.sum(), 1), device=dev))
    bb_params = [p for p in bb.parameters() if p.requires_grad]
    other = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": bb_params, "lr": a.lr / 3}, {"params": other, "lr": a.lr}], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr / 3, a.lr], total_steps=a.epochs * len(dl_tr))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    print("trainable params:", count_params(model, trainable=True), "/", count_params(model))

    best, bad, ck = -1, 0, Path(a.out) / f"{a.backbone}_{a.mode}_{a.head if a.mode == 'video' else 'mean'}_ft.pt"
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for ep in range(a.epochs):
        model.train(); freeze_bn(model)
        for step, (x, y) in enumerate(dl_tr):
            x, y = to_input(x, dev), y.float().to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                if a.mode == "frame":                                  # every frame is a sample
                    logit = model.forward_frames(x.flatten(0, 1)); y = y.repeat_interleave(x.shape[1])
                else:
                    logit = model(x)
                loss = lossf(logit.float(), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            if step % 50 == 0:
                print(f"epoch {ep + 1} step {step}/{len(dl_tr)} loss {loss.item():.4f}")
        yv, pv = predict(model, dl_va, dev, amp); auc = metrics(yv, pv)["auc"]
        print(f"epoch {ep + 1} val_auc {auc:.4f}")
        if auc > best + 1e-4:
            best, bad = auc, 0
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, ck.with_suffix(".tmp"))
        else:
            bad += 1
            if bad >= a.patience: break

    model.load_state_dict(torch.load(ck.with_suffix(".tmp"), map_location=dev, weights_only=True))
    yv, pv = predict(model, dl_va, dev, amp); thr = best_threshold(yv, pv)
    res = {"test": metrics(*predict(model, DataLoader(StripSet(a.faces, a.test), a.bs, num_workers=a.workers), dev, amp), thr)}
    for i, c in enumerate(a.cross):
        res[f"cross{i}"] = metrics(*predict(model, DataLoader(StripSet(Path(c).parent, c), a.bs, num_workers=a.workers), dev, amp), thr)
    T = 16
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({"format": "veritas-v2", "kind": a.mode, "backbone": a.backbone, "head": a.head if a.mode == "video" else None,
                "head_kwargs": {}, "model_state": state, "num_frames": T, "img_size": 224, "threshold": thr,
                "metrics": res, "model_version": f"veritas-{a.backbone}-{a.mode}-ft"}, ck)
    ck.with_suffix(".tmp").unlink(missing_ok=True)
    row = {"approach": "A" if a.mode == "frame" else "D", "backbone": a.backbone,
           "head": "frame-mean" if a.mode == "frame" else a.head + "-ft", "dataset": Path(a.faces).name,
           "head_params": count_params(model) - count_params(bb), "head_ms": "", **deploy_cost(model, T)}
    for split, m in res.items():
        for k in ("auc", "acc", "f1", "eer"):
            row[f"{split}_{k}"] = m[k]
    log_result(a.results, row); print("saved", ck)
