"""Backbones (frame encoders) and temporal heads for VeritasAI."""
import os
import torch
from torch import nn
import torchvision.models as tvm

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

#            name                : (builder,              weights enum name)
BACKBONES = {
    "mobilenet_v3_small": (tvm.mobilenet_v3_small, "MobileNet_V3_Small_Weights"),
    "mobilenet_v3_large": (tvm.mobilenet_v3_large, "MobileNet_V3_Large_Weights"),
    "efficientnet_b0":    (tvm.efficientnet_b0,    "EfficientNet_B0_Weights"),
    "efficientnet_b2":    (tvm.efficientnet_b2,    "EfficientNet_B2_Weights"),
    "regnet_y_800mf":     (tvm.regnet_y_800mf,     "RegNet_Y_800MF_Weights"),
    "convnext_tiny":      (tvm.convnext_tiny,      "ConvNeXt_Tiny_Weights"),
    "resnext50_32x4d":    (tvm.resnext50_32x4d,    "ResNeXt50_32X4D_Weights"),  # original repo baseline
}


class Backbone(nn.Module):
    """uint8/float image batch [B,3,H,W] in 0..1 -> feature vector [B,D]. Normalisation is inside
    the module so exported ONNX models need no extra preprocessing code."""
    def __init__(self, name="efficientnet_b0", pretrained=True):
        super().__init__()
        builder, wname = BACKBONES[name]
        weights = getattr(tvm, wname).DEFAULT if pretrained and not os.getenv("VERITAS_OFFLINE") else None
        m = builder(weights=weights)
        if hasattr(m, "fc"):
            m.fc = nn.Identity()
        elif name.startswith("convnext"):
            m.classifier[2] = nn.Identity()
        else:
            m.classifier = nn.Identity()
        self.net, self.name = m, name
        self.register_buffer("mean", MEAN.clone(), persistent=False)
        self.register_buffer("std", STD.clone(), persistent=False)
        m.eval()
        with torch.no_grad():
            self.dim = self.net(torch.zeros(1, 3, 64, 64)).shape[1]
        m.train()

    def forward(self, x):
        return self.net((x - self.mean) / self.std)

    def blocks(self):
        """Ordered list of top-level stages (used for partial unfreezing)."""
        n = self.net
        if hasattr(n, "features"):
            return list(n.features)
        if hasattr(n, "trunk_output"):
            return [n.stem] + list(n.trunk_output)
        return [nn.Sequential(n.conv1, n.bn1), n.layer1, n.layer2, n.layer3, n.layer4]

    def freeze(self, trainable_last=0):
        for p in self.parameters():
            p.requires_grad = False
        if trainable_last:
            for b in self.blocks()[-trainable_last:]:
                for p in b.parameters():
                    p.requires_grad = True


# ----------------------------------------------------------------- temporal heads
class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__(); self.score = nn.Linear(d, 1)
    def forward(self, x):                               # x [B,T,d]
        w = torch.softmax(self.score(x).squeeze(-1), 1)  # [B,T]
        return (w.unsqueeze(-1) * x).sum(1), w


class SeqEncoder(nn.Module):
    """Projects per-frame features and models time. kind in mean|lstm|gru|tcn|attn."""
    def __init__(self, in_dim, kind="gru", hid=128, drop=0.3):
        super().__init__()
        self.kind = kind
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(drop), nn.Linear(in_dim, hid), nn.GELU())
        if kind in ("lstm", "gru"):
            rnn = nn.LSTM if kind == "lstm" else nn.GRU
            self.rnn = rnn(hid, hid // 2, batch_first=True, bidirectional=True)
        elif kind == "tcn":
            self.tcn = nn.Sequential(nn.Conv1d(hid, hid, 3, padding=1), nn.GELU(),
                                     nn.Conv1d(hid, hid, 3, padding=2, dilation=2), nn.GELU())
        elif kind == "attn":
            self.tf = nn.TransformerEncoderLayer(hid, 4, hid * 2, 0.1, batch_first=True, norm_first=True)
        self.pool = AttnPool(hid)
        self.out_dim = hid * 2 if kind == "mean" else hid

    def forward(self, x):
        h = self.proj(x)
        if self.kind == "mean":
            return torch.cat([h.mean(1), h.std(1, unbiased=False)], -1), None
        if self.kind in ("lstm", "gru"):
            h, _ = self.rnn(h)
        elif self.kind == "tcn":
            h = h + self.tcn(h.transpose(1, 2)).transpose(1, 2)
        elif self.kind == "attn":
            h = self.tf(h)
        return self.pool(h)


class TemporalHead(nn.Module):
    """Approach B / D head: RGB feature sequence -> logit."""
    def __init__(self, in_dim, kind="gru", hid=128, drop=0.3, **_):
        super().__init__()
        self.enc = SeqEncoder(in_dim, kind, hid, drop)
        self.cls = nn.Sequential(nn.Dropout(drop), nn.Linear(self.enc.out_dim, 1))
    def forward(self, rgb, freq=None):
        z, _ = self.enc(rgb)
        return self.cls(z).squeeze(-1), {}


class FusionHead(nn.Module):
    """Approach C: three cheap evidence streams + gated fusion + modality dropout.
       stream 1  appearance : backbone features            (what the face looks like)
       stream 2  motion     : f[t]-f[t-1] feature deltas   (temporal inconsistency, ~free optical-flow proxy)
       stream 3  frequency  : log power-spectrum profile   (up-sampling / blending artefacts)
    Each stream also has its own auxiliary classifier so no modality becomes 'lazy' (AMONet idea)."""
    def __init__(self, in_dim, freq_dim=64, kind="gru", hid=128, drop=0.3, mod_drop=0.2, streams="amf", **_):
        super().__init__()
        self.active = [c in streams for c in "amf"]        # a=appearance m=motion f=frequency (for ablations)
        self.encs = nn.ModuleList([SeqEncoder(in_dim, kind, hid, drop),
                                   SeqEncoder(in_dim, kind, hid, drop),
                                   SeqEncoder(freq_dim, "tcn", hid // 2, drop)])
        dims = [e.out_dim for e in self.encs]
        self.align = nn.ModuleList([nn.Linear(d, hid) for d in dims])
        self.gate = nn.Sequential(nn.Linear(hid * 3, hid), nn.GELU(), nn.Linear(hid, 3))
        self.aux = nn.ModuleList([nn.Linear(hid, 1) for _ in dims])
        self.cls = nn.Sequential(nn.Dropout(drop), nn.Linear(hid, 1))
        self.mod_drop = mod_drop

    def forward(self, rgb, freq):
        delta = torch.cat([torch.zeros_like(rgb[:, :1]), rgb[:, 1:] - rgb[:, :-1]], 1)
        zs = [a(e(x)[0]) for a, e, x in zip(self.align, self.encs, (rgb, delta, freq))]
        z = torch.stack(zs, 1)                                   # [B,3,hid]
        g = self.gate(z.flatten(1))
        off = torch.tensor([not a for a in self.active], device=g.device)
        g = g.masked_fill(off, -1e4)
        if self.training and self.mod_drop > 0 and sum(self.active) > 1:                  # randomly hide one stream
            drop = (torch.rand(z.shape[0], device=z.device) < self.mod_drop)
            which = torch.randint(0, 3, (z.shape[0],), device=z.device)
            mask = torch.zeros_like(g, dtype=torch.bool)
            mask[drop, which[drop]] = True
            mask &= ~off                                         # only hide streams that are active...
            mask[mask.all(1) | (mask | off).all(1)] = False      # ...and never hide every stream
            g = g.masked_fill(mask, -1e4)
        w = torch.softmax(g, 1)
        fused = (w.unsqueeze(-1) * z).sum(1)
        aux = torch.cat([h(zi) for h, zi, on in zip(self.aux, zs, self.active) if on], -1)  # [B,#active]
        return self.cls(fused).squeeze(-1), {"aux": aux, "gates": w}


def build_head(name, in_dim, **kw):
    if name == "fusion":
        return FusionHead(in_dim, **kw)
    return TemporalHead(in_dim, kind=name, **kw)


class VideoModel(nn.Module):
    """Backbone + head in one module (used for fine-tuning, inference and ONNX export)."""
    def __init__(self, backbone, head):
        super().__init__(); self.backbone, self.head = backbone, head
    def forward(self, frames, freq=None):                       # frames [B,T,3,H,W] 0..1
        B, T = frames.shape[:2]
        f = self.backbone(frames.flatten(0, 1)).view(B, T, -1)
        return self.head(f, freq)[0]


class FrameModel(nn.Module):
    """Approach A: per-frame classifier; video score = mean of frame probabilities."""
    def __init__(self, backbone, drop=0.3):
        super().__init__(); self.backbone = backbone
        self.cls = nn.Sequential(nn.Dropout(drop), nn.Linear(backbone.dim, 1))
    def forward_frames(self, x):                                 # [N,3,H,W]
        return self.cls(self.backbone(x)).squeeze(-1)
    def forward(self, frames, freq=None):                        # [B,T,3,H,W] -> video logit
        B, T = frames.shape[:2]
        p = torch.sigmoid(self.forward_frames(frames.flatten(0, 1))).view(B, T).mean(1)
        return torch.logit(p.clamp(1e-6, 1 - 1e-6))


def count_params(m, trainable=False):
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable))
