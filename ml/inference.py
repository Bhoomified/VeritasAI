"""Inference used by the Django app. Works with every checkpoint produced by
ml.train_head (approaches B, C) and ml.finetune (approaches A, D).

Safety rules (to avoid confident WRONG answers):
  * no checkpoint            -> InferenceError (never a fake REAL/FAKE)
  * face in < 50 % of frames -> InferenceError ("no clear face")
  * probability within ±UNCERTAIN_BAND of the tuned threshold -> reported, but flagged low-confidence
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os
import pickle
import time

import numpy as np
import torch

from .faces import FaceCropper, faces_from_video, freq_features, jpeg_roundtrip
from .nets import Backbone, FrameModel, VideoModel, build_head

MIN_FACE_RATIO = float(os.getenv("MIN_FACE_RATIO", "0.5"))
UNCERTAIN_BAND = float(os.getenv("UNCERTAIN_BAND", "0.1"))
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


class InferenceError(Exception):
    pass


@dataclass
class PredictionResult:
    prediction: str
    confidence: float
    frame_count: int
    model_version: str
    device: str
    probability_fake: float = 0.0
    low_confidence: bool = False
    face_ratio: float = 0.0
    stream_weights: dict = None
    seconds: float = 0.0


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")      # CPU is fast enough for EfficientNet-B0 x 16 frames (MPS adds overhead)


@lru_cache(maxsize=2)               # model is loaded once per process, not once per upload
def load_checkpoint(checkpoint_path, device=None):
    if not checkpoint_path or not Path(checkpoint_path).exists():
        raise InferenceError("MODEL NOT TRAINED / DEMO ONLY: no valid trained checkpoint is configured.")
    device = device or _device()
    try:
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(ck, dict) or ck.get("format") != "veritas-v2":
            raise ValueError("not a VeritasAI v2 checkpoint (train with ml.train_head or ml.finetune)")
        bb = Backbone(ck["backbone"], pretrained=False)
        if ck["kind"] == "head":
            bb.load_state_dict(ck["backbone_state"])
            head = build_head(ck["head"], bb.dim, freq_dim=64, **ck.get("head_kwargs", {}))
            head.load_state_dict(ck["head_state"])
            model = VideoModel(bb, head)
        else:
            model = FrameModel(bb) if ck["kind"] == "frame" else VideoModel(bb, build_head(ck["head"], bb.dim, freq_dim=64))
            model.load_state_dict(ck["model_state"])
    except (OSError, RuntimeError, TypeError, ValueError, KeyError, pickle.UnpicklingError) as exc:
        raise InferenceError(f"Invalid trained checkpoint: {exc}") from exc
    model.to(device).eval()
    return model, ck, device


@lru_cache(maxsize=1)
def _cropper():
    from .faces import DEFAULT_MODEL_DIR, YUNET_FILE
    return FaceCropper(model_path=DEFAULT_MODEL_DIR / YUNET_FILE)


def predict_video(video_path, checkpoint_path=None, num_frames=None, use_imagenet_weights=False):
    t0 = time.perf_counter()
    model, ck, device = load_checkpoint(checkpoint_path)
    T = ck.get("num_frames", 16)                       # always use the frame count the model was trained with
    try:
        faces, ratio, idx = faces_from_video(video_path, _cropper(), T)
    except ValueError as exc:
        raise InferenceError(str(exc)) from exc
    if ratio < MIN_FACE_RATIO:
        raise InferenceError(f"No clear face found (face visible in {ratio:.0%} of sampled frames). "
                             "VeritasAI only analyses videos with a visible face.")
    faces = np.ascontiguousarray(jpeg_roundtrip(faces))     # identical preprocessing to training
    x = torch.from_numpy(faces).permute(0, 3, 1, 2).unsqueeze(0).float().div_(255).to(device)
    f = torch.from_numpy(freq_features(faces)).unsqueeze(0).to(device)
    gates = None
    with torch.inference_mode():
        if isinstance(model, VideoModel):
            feats = model.backbone(x.flatten(0, 1)).view(1, x.shape[1], -1)
            logit, extra = model.head(feats, f)
            if "gates" in extra:
                g = extra["gates"][0].tolist(); gates = dict(zip(["appearance", "motion", "frequency"], g))
        else:
            logit = model(x)
        p = float(torch.sigmoid(logit).item())
    thr = float(ck.get("threshold", 0.5))
    fake = p >= thr
    # confidence rescaled so that the tuned threshold maps to 0.5
    conf = 0.5 + 0.5 * ((p - thr) / (1 - thr) if fake else (thr - p) / thr)
    return PredictionResult("FAKE" if fake else "REAL", round(float(conf), 4), len(idx), ck.get("model_version", "veritas"),
                            str(device), p, abs(p - thr) < UNCERTAIN_BAND, ratio, gates, time.perf_counter() - t0)
