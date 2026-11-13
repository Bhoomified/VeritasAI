from dataclasses import dataclass
from pathlib import Path
import pickle
import torch
from .device import select_device
from .model import ModelConfig, build_models
from .preprocess import PreprocessingError, extract_frames

class InferenceError(Exception): pass
@dataclass
class PredictionResult:
    prediction: str
    confidence: float
    frame_count: int
    model_version: str
    device: str

def load_checkpoint(checkpoint_path, device=None):
    if not checkpoint_path or not Path(checkpoint_path).exists():
        raise InferenceError("MODEL NOT TRAINED / DEMO ONLY: no valid trained checkpoint is configured.")
    device = device or select_device()
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint must contain a mapping")
        missing = {"config", "feature_extractor", "classifier"} - set(checkpoint)
        if missing:
            raise ValueError(f"checkpoint is missing keys: {', '.join(sorted(missing))}")
        config = ModelConfig(**checkpoint["config"])
        feature_extractor, classifier = build_models(
            config, use_imagenet_weights=checkpoint.get("use_imagenet_weights", False)
        )
        feature_extractor.load_state_dict(checkpoint["feature_extractor"])
        classifier.load_state_dict(checkpoint["classifier"])
    except (OSError, RuntimeError, TypeError, ValueError, KeyError, pickle.UnpicklingError) as exc:
        raise InferenceError(f"Invalid trained checkpoint: {exc}") from exc
    feature_extractor.to(device).eval(); classifier.to(device).eval()
    return feature_extractor, classifier, config, device

def predict_video(video_path, checkpoint_path=None, num_frames=20, use_imagenet_weights=False):
    try:
        frames = extract_frames(video_path, num_frames)
    except PreprocessingError as exc:
        raise InferenceError(str(exc)) from exc
    feature_extractor, classifier, config, device = load_checkpoint(checkpoint_path)
    with torch.no_grad():
        features = feature_extractor(frames.frames.to(device)).unsqueeze(0)
        probability = float(classifier.probability(features).item())
    return PredictionResult("FAKE" if probability >= .5 else "REAL", max(probability, 1-probability),
                            len(frames.indices), config.model_version, str(device))
