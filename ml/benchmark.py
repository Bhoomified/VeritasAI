"""Efficiency numbers for the paper + optional ONNX export.

  python -m ml.benchmark --backbones mobilenet_v3_small mobilenet_v3_large efficientnet_b0 resnext50_32x4d
  python -m ml.benchmark --checkpoint checkpoints/efficientnet_b0_fusion.pt --onnx veritas.onnx

Reports params (M), GFLOPs per video (16 frames @224) and CPU latency per video.
"""
import argparse
import torch
from torch.utils.flop_counter import FlopCounterMode
from .nets import Backbone, count_params
from .evalutils import time_it


def gflops(model, *inp):
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad(): model(*inp)
    return fc.get_total_flops() / 1e9 / 2            # MACs-style GFLOPs (common convention in papers)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="*", default=[]); ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--checkpoint"); ap.add_argument("--onnx"); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    x = torch.rand(a.frames, 3, 224, 224)
    print(f"{'backbone':22s} {'params(M)':>9s} {'GFLOPs/video':>13s} {'CPU ms/video':>13s}")
    for name in a.backbones:
        m = Backbone(name, pretrained=False).eval()
        print(f"{name:22s} {count_params(m)/1e6:9.2f} {gflops(m, x):13.2f} {time_it(lambda: m(x), 2, 5):13.1f}")
    if a.checkpoint:
        from .inference import load_checkpoint
        model, ck, _ = load_checkpoint(a.checkpoint, torch.device("cpu"))
        v, f = x.unsqueeze(0), torch.zeros(1, a.frames, 64)
        print(f"checkpoint {ck['model_version']}: params {count_params(model)/1e6:.2f}M, "
              f"{gflops(model, v, f):.2f} GFLOPs, {time_it(lambda: model(v, f), 2, 5):.1f} ms/video (CPU)")
        if a.onnx:
            kw = dict(input_names=["frames", "freq"], output_names=["logit"], opset_version=17)
            try:
                torch.onnx.export(model, (v, f), a.onnx, dynamo=False, **kw)
            except TypeError:                       # older torch has no `dynamo` argument
                torch.onnx.export(model, (v, f), a.onnx, **kw)
            import numpy as np, onnxruntime as ort
            s = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
            feed = {"frames": v.numpy(), "freq": f.numpy()}
            print(f"ONNX Runtime CPU: {time_it(lambda: s.run(None, feed), 2, 5):.1f} ms/video -> {a.onnx}")
