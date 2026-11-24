"""Step 2 (approaches B & C) – run a FROZEN backbone once over every face strip and cache features.

  python -m ml.features --faces faces/ffpp --index faces/ffpp/ffpp_train.csv faces/ffpp/ffpp_val.csv \
        faces/ffpp/ffpp_test.csv --backbone efficientnet_b0 --out feats/ffpp_effb0

Produces <out>/<split>.npz with  rgb [V,T,D] fp16, freq [V,T,64] fp16, y [V], uid [V].
Cost: one forward pass per frame, ever. A T4 does ~2 000 EfficientNet-B0 frames/s in fp16.
"""
import argparse, csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from .faces import load_strip, freq_features
from .nets import Backbone
from .evalutils import pick_device


class Strips(Dataset):
    def __init__(self, faces_dir, index_csv, size):
        self.dir, self.size = Path(faces_dir), size
        self.rows = list(csv.DictReader(open(index_csv)))
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        faces = load_strip(self.dir / f"{r['uid']}.jpg", self.size)            # [T,S,S,3] uint8
        return torch.from_numpy(faces).permute(0, 3, 1, 2), torch.from_numpy(freq_features(faces)), int(r["label"]), r["uid"]


@torch.no_grad()
def extract(faces_dir, index_csv, backbone, device, size=224, bs=16, workers=2, tta=False):
    dl = DataLoader(Strips(faces_dir, index_csv, size), batch_size=bs, num_workers=workers, pin_memory=device.type == "cuda")
    rgb, frq, ys, uids = [], [], [], []
    amp = device.type == "cuda"
    for x, fq, y, u in dl:
        B, T = x.shape[:2]
        x = x.flatten(0, 1).to(device, non_blocking=True).float().div_(255).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            f = backbone(x)
            if tta:
                f = (f + backbone(torch.flip(x, dims=[3]))) / 2
        rgb.append(f.float().view(B, T, -1).cpu().half()); frq.append(fq.half()); ys += y.tolist(); uids += list(u)
    return torch.cat(rgb).numpy(), torch.cat(frq).numpy(), np.array(ys), np.array(uids)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", required=True); ap.add_argument("--index", nargs="+", required=True)
    ap.add_argument("--backbone", default="efficientnet_b0"); ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=224); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--tta", action="store_true")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    dev = pick_device(a.device)
    bb = Backbone(a.backbone, pretrained=True).to(dev).eval().to(memory_format=torch.channels_last)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"name": a.backbone, "state": bb.state_dict()}, out / "backbone.pt")   # frozen weights travel with the head
    for idx in a.index:
        split = Path(idx).stem.split("_")[-1]                                         # ffpp_train.csv -> train
        rgb, frq, y, uid = extract(a.faces, idx, bb, dev, a.size, a.bs, a.workers, a.tta)
        np.savez(out / f"{split}.npz", rgb=rgb, freq=frq, y=y, uid=uid)
        print(f"{split}: rgb{rgb.shape} freq{frq.shape} fake={int(y.sum())}/{len(y)}")
